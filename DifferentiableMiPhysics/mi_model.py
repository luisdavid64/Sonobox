from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from topology_utils import build_grid_nodes, build_edges_nearest
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

@dataclass
class MIModel:
    nodes: torch.Tensor         # [N, 8]
    edge_index: torch.Tensor    # [2, E]
    springs: torch.Tensor       # [E, 6]
    drivers: Optional[torch.Tensor] = None
    listeners: Optional[torch.Tensor] = None
    config: Optional[dict] = None  # Optionally keep the raw config
    dimX: Optional[int] = None
    dimY: Optional[int] = None
    dimZ: Optional[int] = None
    interactionType: Optional[str] = "FIRST"

    @classmethod
    def from_config(cls, config: dict, device: Optional[torch.device] = None):
        # Helper to parse "m_i_j_k" names
        def parse_mass_name(name):
            parts = name.split("_")
            if len(parts) != 4 or parts[0] != "m":
                raise ValueError(f"Invalid mass name: {name}")
            return tuple(int(x) for x in parts[1:])

        geom = config["geometry"]
        dimX = geom["dx"]
        dimY = geom["dy"]
        dimZ = geom["dz"]
        dist = float(geom["distance"])
        mass_radius = float(geom["massesRadius"])
        mass = float(config["parameters"]["M"][0]) if "parameters" in config else 1.0
        stiffness = float(config["parameters"]["K"][0]) if "parameters" in config else 1e-3
        damping = float(config["parameters"]["C"][0]) if "parameters" in config else 0.0

        # Drivers: (no_drivers, 3), Listeners: (no_listeners, 3)
        drivers = torch.tensor([parse_mass_name(n) for n in config["sonification_set_up"]["drivers"]])
        listeners = torch.tensor([parse_mass_name(n) for n in config["sonification_set_up"]["listeners"]])

        
        # Interactions
        interactionType = str(geom["interactionType"])

        # Import your grid/spring builder functions

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
        return cls(nodes, edge_index, springs, drivers, listeners, config, dimX=dimX, dimY=dimY, dimZ=dimZ, interactionType=interactionType)
    
    def get_listeners(self):
        return self.listeners
    
    def get_drivers(self):
        return self.drivers

    def get_nodes_and_edges_idx(self):
        # Get nodes and edges_idx without features
        nodes_idx = torch.arange(self.nodes.shape[0])
        edges_idx = self.edge_index
        return nodes_idx, edges_idx


    @classmethod
    def from_json(cls, path: str, device: Optional[torch.device] = None):
        import json
        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_config(config, device=device)

    def visualize(self, show_drivers=True, show_listeners=True, figsize=(8,6)):
        """
        Visualize the model topology in 3D using NetworkX and matplotlib.
        Y axis is up, Z is back, X is right (MI Physics convention).
        """

        positions = self.nodes[:, :3].cpu().numpy()
        node_indices = range(positions.shape[0])
        pos_dict = {i: positions[i] for i in node_indices}
        edge_list = self.edge_index.cpu().numpy().T.tolist()

        G = nx.Graph()
        G.add_nodes_from(node_indices)
        G.add_edges_from(edge_list)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Swap axes: x (right), y (up), z (back)
        ax.scatter(positions[:,0], positions[:,2], positions[:,1], c='b', s=30, label='Nodes')

        # Highlight drivers and listeners
        if show_drivers and self.drivers is not None:
            driver_idx = [self._node_index_from_tuple(tuple(d.tolist())) for d in self.drivers]
            ax.scatter(positions[driver_idx,0], positions[driver_idx,2], positions[driver_idx,1], c='r', s=60, label='Drivers')
        if show_listeners and self.listeners is not None:
            listener_idx = [self._node_index_from_tuple(tuple(l.tolist())) for l in self.listeners]
            ax.scatter(positions[listener_idx,0], positions[listener_idx,2], positions[listener_idx,1], c='g', s=60, label='Listeners')

        # Plot edges
        for edge in edge_list:
            p1 = positions[edge[0]]
            p2 = positions[edge[1]]
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]], color='gray', alpha=0.5)

        ax.set_xlabel('X (right)')
        ax.set_ylabel('Z (back)')
        ax.set_zlabel('Y (up)')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def _node_index_from_tuple(self, idx_tuple):
        i, j, k = idx_tuple
        return (i * self.dimY + j) * self.dimZ + k