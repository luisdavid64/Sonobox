import plotly.graph_objects as go
import numpy as np
import taichi as ti
import numpy as np
import imageio
import librosa
import matplotlib.pyplot as plt
import torch

def plot_model_graph_3d(
    model,
    positions=None,
    node_size=3,
    edge_width=1,
    node_opacity=0.95,
    node_color="#1f77b4",
    bg_color="rgba(0,0,0,0)",
    show_axes=True,
    title=None,
    path=None
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
    if path is not None:
        fig.write_html(path)
    else:
        fig.write_html("graph3d.html")


ti.init(arch=ti.cpu, log_level=ti.ERROR)

def compute_center_and_radius(particles):
    p = particles # (N, 3)
    center = p.mean(axis=0)                # entroid
    
    # bounding radius: max distance to centroid
    radius = np.linalg.norm(p - center, axis=1).max()
    return center, float(radius)

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

    center,radius = compute_center_and_radius(traj[0])

    window = ti.ui.Window("Mass–Spring 3D", (1200, 800))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Camera framing: keep the whole object in view based on fov
    fov = 45.0  # degrees
    import math
    dist = radius / math.tan(math.radians(fov) * 0.5) * 1.1  # 20% padding

    # WORLD AXES we want on screen:
    #   x -> right, y -> up, z -> back (into the screen)
    # Put the camera on the -z side, looking toward +z (the "back")
    cam_pos = (center[0], center[1], center[2] - dist)

    camera.position(*cam_pos)
    camera.lookat(*center)
    camera.up(0, 1, 0)     # y is up
    camera.fov(fov)
    canvas = window.get_canvas()


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

        writer.append_data(raw.transpose(1,0,2))
        window.show()

    writer.close()
    print(f"Saved {out_path}")


def plot_spectrogram(audio, fs):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio.squeeze(), n_fft=1024, hop_length=256, win_length=1024)), ref=np.max) 
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=fs, hop_length=256, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig('mass_spring_spectrogram.png')
    plt.close()


import math, numpy as np
import taichi as ti

def _center_radius(P):
    c = P.mean(0)
    r = np.max(np.linalg.norm(P - c, axis=1)) + 1e-6
    return c, r


def run_interactive_mi(
    model,
    fps: int = 60,
    sim_rate: int = 16000,       # your physics step rate (1/dt)
    force_gain: float = 3.0,     # N per axis when key held
    line_width: float = 3.0,
    node_radius: float = 0.8,
    bg=(0.02, 0.02, 0.03),
    win_size=(1280, 800),
):
    # ti.init(arch=ti.gpu if ti.cuda.is_available() else ti.cpu)

    # --- caches / geometry ---
    device = model.nodes.device
    N = model.N
    i_idx = model.edge_index[0].detach().cpu().numpy()
    j_idx = model.edge_index[1].detach().cpu().numpy()
    E = i_idx.shape[0]
    pos0 = model.m_pos.detach().float().cpu().numpy()
    center, radius = _center_radius(pos0)

    # taichi fields
    particles = ti.Vector.field(3, dtype=ti.f32, shape=N)
    drivers = model.get_driver_ids()
    drivers_np = drivers.detach().cpu().numpy() if drivers is not None else np.zeros(0, dtype=np.int32)
    driver_field = ti.field(dtype=ti.i32, shape=drivers_np.shape[0])
    if drivers_np.size:
        driver_field.from_numpy(drivers_np)

    # window / scene / camera
    window = ti.ui.Window("Mass–Spring 3D (interactive)", win_size)
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    cam = ti.ui.Camera()
    fov = 45.0
    cam_dist = radius / math.tan(math.radians(fov) * 0.5) * 1.3
    cam.position(center[0], center[1], center[2] - cam_dist)
    cam.lookat(center[0], center[1], center[2])
    cam.up(0, 1, 0)
    cam.fov(fov)

    # sim stepping
    steps_per_frame_exact = sim_rate / fps
    step_accum = 0.0
    paused = False
    gain = float(force_gain)

    help_txt = "[1-9] strike driver n  |  [w/d]=gain  |  [SPACE]=pause  |  [R]=reset  |  [ESC]=quit"

    model.eval()  # physics stepping doesn't need grads
    # (Optional) start from rest each run
    model.detach_state(reset_to_rest=True)

    while window.running:
        # camera user control (hold RMB to orbit)
        cam.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(cam)
        scene.ambient_light((0.85, 0.85, 0.9))
        canvas.set_background_color(bg)

        # --- input ---
        if window.is_pressed(ti.ui.ESCAPE):
            break
        if window.is_pressed(ti.ui.SPACE):
            paused = not paused
        if window.is_pressed('r') or window.is_pressed('R'):
            # reset to rest; also clears m_frc
            if hasattr(model, "detach_state"):
                with torch.no_grad():
                    model.detach_state(reset_to_rest=True)


        # gain controls
        if window.is_pressed('w'):
            gain *= 1.05
        if window.is_pressed('d'):
            gain /= 1.05
            gain = max(1e-3, gain)

        # check if pressed numeric
    

        # --- simulate substeps ---
        step_accum += steps_per_frame_exact
        steps = int(step_accum)
        step_accum -= steps
        if steps < 1:
            steps = 1  # keep sim moving even if fps fluctuates

        if not paused:
            for _ in range(steps):
                model.compute()

        # --- render ---
        # copy current positions
        P = model.m_pos.detach().float().cpu().numpy()
        particles.from_numpy(P)

        # nodes
        scene.particles(particles, radius=node_radius, color=(1.0, 0.35, 0.35))

        # highlight driver nodes (green, larger)
        if drivers_np.size:
            # make a tiny field of just drivers for drawing
            # (taichi requires a dense field; we do a quick gather)
            drv_pos = ti.Vector.field(3, dtype=ti.f32, shape=drivers_np.shape[0])
            drv_pos.from_numpy(P[drivers_np])
            scene.particles(drv_pos, radius=node_radius*1.8, color=(0.3, 1.0, 0.4))

        # springs (as lines)
        verts = np.empty((2*E, 3), dtype=np.float32)
        verts[0::2] = P[i_idx]
        verts[1::2] = P[j_idx]
        verts_field = ti.Vector.field(3, dtype=ti.f32, shape=verts.shape[0])
        verts_field.from_numpy(verts)
        scene.lines(verts_field, color=(0.35, 0.45, 1.0), width=line_width)

        canvas.scene(scene)

        # HUD
        window.GUI.begin("controls", 0.02, 0.02, 0.36, 0.22)
        window.GUI.text(help_txt)
        window.GUI.text(f"gain: {gain:6.3f}   steps/frame: {steps}")
        window.GUI.text(f"paused: {paused}")
        window.GUI.end()

        window.show()