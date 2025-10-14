import json
import torch
from diff_mass_spring_model import MassSpringModel
import trimesh
import numpy as np

def build_geometry_from_mesh(mesh_path, config):
    mesh = trimesh.load(mesh_path, process=True)
    # assert mesh.is_watertight or mesh.is_volume, "Non-watertight mesh might behave oddly (still usable though)."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # --- Vertices → nodes
    vertices = torch.tensor(mesh.vertices, dtype=dtype, device=device)
    num_nodes = vertices.shape[0]

    masses = torch.full((num_nodes, 1), config["parameters"]["M"][0], device=device)
    radius = torch.full((num_nodes, 1), config["geometry"]["massesRadius"], device=device)
    fixed = torch.zeros((num_nodes, 1), device=device)
    drivers = torch.zeros((num_nodes, 1), device=device)
    listeners = torch.zeros((num_nodes, 1), device=device)

    # you can later assign drivers/listeners by name or region
    nodes = torch.cat([vertices, masses, radius, fixed, drivers, listeners], dim=1)

    # --- Edges → springs
    # trimesh stores unique edges for faces
    edges = torch.tensor(mesh.edges_unique, dtype=torch.long, device=device).T
    i, j = edges
    rest_len = torch.norm(vertices[j] - vertices[i], dim=1)

    # stiffness & damping based on config
    k_base = torch.tensor(config["parameters"]["K"][0], device=device)
    z_base = torch.tensor(config["parameters"]["C"][0], device=device)

    springs = torch.stack([
        k_base.repeat(len(rest_len)),
        z_base.repeat(len(rest_len)),
        rest_len,
        torch.zeros_like(rest_len)  # neighbor_type (0=first)
    ], dim=1)

    return nodes, edges, springs

def save_geometry(nodes, edge_index, springs, path):
    np.savez(path, nodes=nodes.cpu().numpy(), edge_index=edge_index.cpu().numpy(), springs=springs.cpu().numpy())

def assign_random_drivers_listeners(nodes, n_drivers=10, n_listeners=10, seed=None):
    """
    Randomly assigns `n_drivers` and `n_listeners` distinct *non-fixed* nodes
    as drivers and listeners. Modifies the nodes tensor in-place.
    """
    if seed is not None:
        torch.manual_seed(seed)

    N = nodes.shape[0]

    # Identify non-fixed nodes (column 5)
    movable_mask = nodes[:, 5] == 0
    movable_indices = torch.nonzero(movable_mask, as_tuple=False).squeeze()

    if movable_indices.numel() == 0:
        raise ValueError("No movable nodes available to assign as drivers or listeners.")

    # Shuffle movable indices
    movable_indices = movable_indices[torch.randperm(len(movable_indices))]

    # Clamp counts to available nodes
    n_drivers = min(n_drivers, len(movable_indices))
    n_listeners = min(n_listeners, len(movable_indices) - n_drivers)

    driver_ids = movable_indices[:n_drivers]
    listener_ids = movable_indices[n_drivers:n_drivers + n_listeners]

    # Clear old flags
    nodes[:, 6] = 0
    nodes[:, 7] = 0

    # Assign flags
    nodes[driver_ids, 6] = 1
    nodes[listener_ids, 7] = 1

    return driver_ids, listener_ids


def fix_outer_nodes(nodes, ratio=0.9):
    """
    Fixes nodes near the outer edge of the mesh.
    Uses distance from centroid and a threshold ratio.
    Modifies the nodes tensor in-place.
    """
    # node columns: [x, y, z, mass, radius, fixed, driver, listener]
    positions = nodes[:, :3]
    center = positions.mean(dim=0)
    distances = torch.norm(positions - center, dim=1)

    # find threshold based on max distance
    max_dist = distances.max()
    threshold = ratio * max_dist

    fixed_mask = distances >= threshold
    nodes[fixed_mask, 5] = 1  # mark as fixed (column 5)

    return nodes

def fix_inner_nodes(nodes, ratio=0.1):
    """
    Fixes nodes near the inner (central) area of the mesh.
    Uses distance from centroid and a threshold ratio.
    Modifies the nodes tensor in-place.

    Parameters:
        nodes: torch.Tensor of shape [N, 8]
            Columns: [x, y, z, mass, radius, fixed, driver, listener]
        ratio: float
            Fraction of max distance defining the inner region (default: 0.1)
    """
    positions = nodes[:, :3]
    center = positions.mean(dim=0)
    distances = torch.norm(positions - center, dim=1)

    # find threshold based on max distance
    max_dist = distances.max()
    threshold = ratio * max_dist

    fixed_mask = distances <= threshold
    nodes[fixed_mask, 5] = 1  # mark as fixed (column 5)

    return nodes


if __name__ == "__main__":
    config = "/Users/luisreyes/Sonify/SonoBox/model_configs/sonobox_data/baselines/biosonix_3D.json"
    config = json.load(open(config, 'r'))

    obj = "/Users/luisreyes/Sonify/SonoBox/plate.obj"
    path = "/Users/luisreyes/Sonify/SonoBox/model_configs/sonobox_data/baselines/plate.npz"
    conf_new = "/Users/luisreyes/Sonify/SonoBox/model_configs/sonobox_data/baselines/plate.json"
    nodes, edge_index, springs = build_geometry_from_mesh(obj, config)
    # nodes = fix_outer_nodes(nodes, ratio=0.99)
    nodes = fix_inner_nodes(nodes, ratio=0.05)
    driver_ids, listener_ids = assign_random_drivers_listeners(nodes, n_drivers=10, n_listeners=10, seed=42)
    save_geometry(nodes, edge_index, springs, path)
    config["model"] = "custom"
    config["geometry"]["source"] = path
    config["sonification_set_up"]["drivers_ids"] = driver_ids.cpu().tolist()
    config["sonification_set_up"]["listeners_ids"] = listener_ids.cpu().tolist()
    json.dump(config, open(conf_new, 'w'), indent=4)

    test = json.load(open(conf_new, 'r'))
    model = MassSpringModel.from_config(test)
    model.visualize()