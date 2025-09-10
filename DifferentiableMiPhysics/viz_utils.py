import plotly.graph_objects as go
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def plot_model_graph_3d(
    model,
    positions=None,
    node_size=3,
    edge_width=1,
    node_opacity=0.95,
    node_color="#1f77b4",
    bg_color="rgba(0,0,0,0)",
    show_axes=False,
    title=None,
):
    """
    Performant 3D graph plot (WebGL) without NetworkX.

    Assumes PyG-style:
      - model.edge_index: [2, E] torch.LongTensor
      - model.N: number of nodes
      - model.nodes: [N, >=3] tensor with at least XYZ

    Parameters
    ----------
    positions : np.ndarray or None
        If provided, shape [N, 3]. Otherwise uses model.nodes[:,:3].
    """

    # --- Get data from the model (to numpy) ---
    edge_index = model.edge_index.t().contiguous().cpu().numpy()  # [E, 2]
    N = int(model.N)

    if positions is None:
        pos = model.nodes[:, :3].detach().cpu().numpy()
    else:
        pos = np.asarray(positions)
        if pos.shape[1] < 3:
            raise ValueError("positions must have at least 3 columns (x,y,z).")

    # --- Build a single batched line trace for all edges (fast) ---
    # Interleave (x_i, x_j, nan) for each edge i->j, same for y,z
    E = edge_index.shape[0]
    src = edge_index[:, 0]
    dst = edge_index[:, 1]

    xs = np.column_stack([pos[src, 0], pos[dst, 0], np.full(E, np.nan)]).ravel()
    ys = np.column_stack([pos[src, 1], pos[dst, 1], np.full(E, np.nan)]).ravel()
    zs = np.column_stack([pos[src, 2], pos[dst, 2], np.full(E, np.nan)]).ravel()

    edge_trace = go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(width=edge_width),
        hoverinfo="none",
        showlegend=False
    )

    # --- Node coloring and highlighting ---
    # Default color
    colors = np.full((N,), node_color)
    # Fixed nodes: red
    if hasattr(model, 'nodes') and model.nodes.shape[1] > 5:
        fixed_mask = model.nodes[:, 5].detach().cpu().numpy() > 0.5
        colors[fixed_mask] = '#d62728'  # red
    # Drivers: green
    if hasattr(model, 'nodes') and model.nodes.shape[1] > 6:
        driver_mask = model.nodes[:, 6].detach().cpu().numpy() > 0.5
        colors[driver_mask] = '#2ca02c'  # green
    # Listeners: orange
    if hasattr(model, 'nodes') and model.nodes.shape[1] > 7:
        listener_mask = model.nodes[:, 7].detach().cpu().numpy() > 0.5
        colors[listener_mask] = '#ff7f0e'  # orange

    node_trace = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode="markers",
        marker=dict(size=3*node_size, opacity=node_opacity, color=colors),
        hoverinfo="skip",
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    # --- Layout & axes ---
    ax = dict(
        showbackground=False,
        showticklabels=False,
        visible=show_axes,  # toggle all axes
        zeroline=False,
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=ax, yaxis=ax, zaxis=ax,
            aspectmode="data",  # equal aspect -> preserves geometry
            bgcolor=bg_color
        ),
        margin=dict(l=0, r=0, t=30 if title else 0, b=0)
    )
    fig.write_html("graph3d.html")

# pip install taichi imageio
import taichi as ti
import numpy as np
import imageio

ti.init(arch=ti.cpu, log_level=ti.ERROR)


def _fix_taichi_frame(window, img, target_size=None):
    """
    img: float32 [0..1] or uint8 from window.get_image_buffer_as_numpy() (H, W, 3/4)
    target_size: (W0, H0) if you want to force a fixed output size
    Returns: uint8 RGB (H0, W0, 3), top-left origin, correct orientation
    """
    # strip alpha if present
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    # Taichi’s origin is bottom-left; flip to top-left for video files
    img = np.flipud(img)

    # Some backends deliver swapped axes on macOS; compare to window size
    W_win, H_win = window.get_window_shape()   # (W, H)
    H_buf, W_buf = img.shape[:2]
    if (H_buf, W_buf) == (W_win, H_win)[::-1]:
        # swap axes if they’re transposed
        img = np.transpose(img, (1, 0, 2))

    # force fixed size if requested
    if target_size is not None:
        W0, H0 = target_size
        if (img.shape[1], img.shape[0]) != (W0, H0):
            from PIL import Image
            img = np.array(Image.fromarray((img*255).astype(np.uint8) if img.dtype!=np.uint8 else img)
                           .resize((W0, H0), resample=Image.BILINEAR))
            # ensure uint8 RGB
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            return img

    # to uint8 RGB
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img

def render_traj_taichi3d(traj, edge_index, masses=None, radii=None, k=None,
                         out_path="traj3d.mp4", fps=30, sim_rate=16000):
    if hasattr(traj, "detach"): traj = traj.detach().cpu().numpy()
    if hasattr(edge_index, "detach"): edge_index = edge_index.detach().cpu().numpy()
    steps_per_frame = int(sim_rate / fps)
    traj = traj[::steps_per_frame]
    T, N, _ = traj.shape
    i, j = edge_index
    E = i.shape[0]

    # Compute bounding box of all trajectories
    mins = traj.reshape(-1, 3).min(0)
    maxs = traj.reshape(-1, 3).max(0)
    center = (mins + maxs) / 2
    extent = (maxs - mins).max()
    dist = 0.8*extent  # padding factor

    window = ti.ui.Window("Mass–Spring 3D", (800, 600))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    canvas = window.get_canvas()
    camera.position(center[0] + dist, center[1] + dist, center[2] + dist)
    camera.lookat(center[0], center[1], center[2])
    camera.up(0, 1, 0)

    particles = ti.Vector.field(3, dtype=ti.f32, shape=N)
    writer = imageio.get_writer(out_path, fps=fps)

    for t in range(T):
        particles.from_numpy(traj[t])


        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))

        # Nodes
        scene.particles(particles, radius=1, color=(1.0, 0.3, 0.3))

        # Springs
        P = traj[t]
        verts = np.empty((2*E, 3), dtype=np.float32)
        verts[0::2] = P[i]
        verts[1::2] = P[j]

        scene.lines(verts, color=(0.3, 0.3, 1.0), width=3)

        canvas.scene(scene)

        raw = window.get_image_buffer_as_numpy()
        frame = _fix_taichi_frame(window, raw, target_size=(800, 600))

        writer.append_data(frame)
        window.show()

    writer.close()
    print(f"Saved {out_path}")
