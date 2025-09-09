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

def animate_trajectory_3d(traj, model, interval=30, node_size=30, edge_color='gray'):
    traj_np = traj.detach().cpu().numpy() if hasattr(traj, 'cpu') else traj  # [T,N,3]
    rest_pos = model.rest_pos.cpu().numpy()
    edge_list = model.edge_index.cpu().numpy().T.tolist() if hasattr(model.edge_index, 'cpu') else model.edge_index.T.tolist()
    N = traj_np.shape[1]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    abs_traj = rest_pos[None, :, :] + traj_np  # [T,N,3]
    x_min, x_max = abs_traj[:,:,0].min(), abs_traj[:,:,0].max()
    y_min, y_max = abs_traj[:,:,2].min(), abs_traj[:,:,2].max()
    z_min, z_max = abs_traj[:,:,1].min(), abs_traj[:,:,1].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    nodes = ax.scatter(abs_traj[0,:,0], abs_traj[0,:,2], abs_traj[0,:,1], s=node_size, c='b')
    lines = []
    for edge in edge_list:
        line, = ax.plot([abs_traj[0,edge[0],0], abs_traj[0,edge[1],0]],
                        [abs_traj[0,edge[0],1], abs_traj[0,edge[1],1]],
                        [abs_traj[0,edge[0],2], abs_traj[0,edge[1],2]],
                        color=edge_color, alpha=0.5)
        lines.append(line)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Mass-Spring Trajectory (3D)')

    def update(frame):
        nodes._offsets3d = (abs_traj[frame,:,0], abs_traj[frame,:,2], abs_traj[frame,:,1])
        for line, edge in zip(lines, edge_list):
            line.set_data([abs_traj[frame,edge[0],0], abs_traj[frame,edge[1],0]],
                        [abs_traj[frame,edge[0],2], abs_traj[frame,edge[1],2]])
            line.set_3d_properties([abs_traj[frame,edge[0],1], abs_traj[frame,edge[1],1]])
        ax.set_title(f"Frame {frame}")
        return [nodes] + lines

    ani = animation.FuncAnimation(fig, update, frames=abs_traj.shape[0], interval=interval, blit=False)
    plt.show()    