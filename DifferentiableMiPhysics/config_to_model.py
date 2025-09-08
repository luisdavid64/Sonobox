import json
import torch

def parse_mass_name(name):
    # Assumes format "m_i_j_k"
    parts = name.split("_")
    if len(parts) != 4 or parts[0] != "m":
        raise ValueError(f"Invalid mass name: {name}")
    return tuple(int(x) for x in parts[1:])

def config_to_model(config_path, device=None):
    with open(config_path, "r") as f:
        config = json.load(f)

    geom = config["geometry"]
    dimX = geom["dx"]
    dimY = geom["dy"]
    dimZ = geom["dz"]
    dist = float(geom["distance"])
    mass_radius = float(geom["massesRadius"])
    numLayers = geom.get("numLayers", 1)
    numNodesPerLayer = geom.get("numNodesPerLayer", [dimY])
    mass = float(config["parameters"]["M"][0]) if "parameters" in config else 1.0
    stiffness = float(config["parameters"]["K"][0]) if "parameters" in config else 1e-3
    damping = float(config["parameters"]["C"][0]) if "parameters" in config else 0.0

    # Drivers/listeners as grid indices
    drivers = torch.stack([parse_mass_name(n) for n in config["sonification_set_up"]["drivers"]])
    listeners = torch.stack([parse_mass_name(n) for n in config["sonification_set_up"]["listeners"]])
    print(drivers)
    exit()

    nodes = build_grid_nodes(
        dimX, dimY, dimZ,
        dist,
        mass=mass,
        radius=mass_radius,
        drivers=drivers,
        listeners=listeners,
        device=device
    )
    edge_index, springs = build_edges_nearest(
        dimX, dimY, dimZ,
        dist,
        stiffness=stiffness,
        damping=damping,
        device=device
    )
    return nodes, edge_index, springs

# Example usage:
if __name__ == "__main__":
    nodes, edge_index, springs = config_to_model("../model_configs/sonobox_data/baselines/biosonix_3D.json", device=torch.device("cpu"))
    print("Nodes shape:", nodes.shape)
    print("Edge index shape:", edge_index.shape)
    print("Springs shape:", springs.shape)