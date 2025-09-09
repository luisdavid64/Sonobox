import torch
from torch import nn
from typing import Optional, List, Tuple
from topology_utils import build_grid_nodes, build_edges_nearest, build_edges_by_type
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

class MassSpringModel(nn.Module):
    def __init__(self, nodes, edge_index, springs, drivers=None, listeners=None, config=None, dimX=None, dimY=None, dimZ=None, interactionType="FIRST"):
        super().__init__()
        self.nodes = nodes         # [N, 8]
        self.edge_index = edge_index    # [2, E]
        self.springs = springs       # [E, 6]
        self.drivers = drivers
        self.listeners = listeners
        self.config = config
        self.dimX = dimX
        self.dimY = dimY
        self.dimZ = dimZ
        self.interactionType = interactionType

        self.N = nodes.shape[0]
        self.E = edge_index.shape[1]
        self.dt = 1e-3
        self.dim = 3
        # Learnable parameters (example: per-mass and per-spring)
        self.masses   = nn.Parameter(torch.full((self.N,), 1.0))          # per-mass
        self.damping  = nn.Parameter(torch.full((self.N,), 0.02))         # per-mass
        self.k        = nn.Parameter(torch.full((self.E,), 200.0))        # per-spring
        self.rest     = nn.Parameter(torch.full((self.E,), 0.5))          # per-spring
        self.radius   = nn.Parameter(torch.full((self.N,), 0.01))         # optional

        # Internal state: positions and velocities
        self.x = torch.zeros((self.N, self.dim), device=self.nodes.device)
        self.v = torch.zeros((self.N, self.dim), device=self.nodes.device)
        self.forces = torch.zeros((self.N, self.dim), device=self.nodes.device)  # external forces

    def spring_forces(self, x):
        i, j = self.edge_index[0], self.edge_index[1]
        d = x[j] - x[i]                                      # [E,dim]
        L = (d.pow(2).sum(-1) + 1e-12).sqrt()                # [E]
        dir = d / L.unsqueeze(-1)
        f = (self.k * (L - self.rest)).unsqueeze(-1) * dir   # [E,dim]

        F = torch.zeros_like(x)
        F.index_add_(0, i,  f)
        F.index_add_(0, j, -f)
        return F

    @torch.no_grad()
    def rewire(self, new_edge_index):
        self.edge_index = new_edge_index
        self.E = new_edge_index.shape[1]
        # Optionally resize k/rest to match new E
        if self.k.shape[0] != self.E:
            self.k = nn.Parameter(torch.full((self.E,), 200.0))
        if self.rest.shape[0] != self.E:
            self.rest = nn.Parameter(torch.full((self.E,), 0.5))

    def step(self, x=None, v=None, g=None):
        # Use internal state if x or v not provided
        if x is None:
            x = self.x
        if v is None:
            v = self.v
        if g is None:
            g = torch.tensor([0.0, -9.81, 0.0], device=x.device)

        F = self.spring_forces(x)
        F += g * self.masses.unsqueeze(-1)                   # gravity
        F += -self.damping.unsqueeze(-1) * v                 # viscous damping
        F += self.forces                                     # apply accumulated external forces

        a = F / self.masses.unsqueeze(-1)
        v = v + self.dt * a
        x = x + self.dt * v

        # Update internal state
        self.x = x
        self.v = v
        self.forces.zero_()  # reset after applying
        return x, v

    def apply_force(self, force):
        """
        Accumulate an external force to be applied on the next step.
        force: tensor of shape [N, dim]
        """
        if not isinstance(force, torch.Tensor):
            force = torch.tensor(force, device=self.nodes.device)
        
        if force.shape == (self.dim,):
            # Map to drivers
            if self.drivers is not None:
                for d in self.drivers:
                    idx = self._node_index_from_tuple(tuple(d.tolist()))
                    self.forces[idx] += force
        self.forces += force

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
        # For now, we can do one mass, stiffness, damping value for all
        mass = float(config["parameters"]["M"][0]) if "parameters" in config else 1.0
        stiffness = float(config["parameters"]["K"][0]) if "parameters" in config else 1e-3
        damping = float(config["parameters"]["C"][0]) if "parameters" in config else 0.0
        interactionType = str(geom.get("interactionType", "FIRST"))

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
        edge_index, springs = build_edges_by_type(
            dimX, dimY, dimZ,
            dist,
            stiffness=stiffness,
            damping=damping,
            interaction_type=interactionType,
            device=device
        )
        return cls(nodes, edge_index, springs, drivers, listeners, config, dimX=dimX, dimY=dimY, dimZ=dimZ, interactionType=interactionType)
    
    def get_listeners(self):
        return self.listeners
    
    def get_drivers(self):
        return self.drivers
    
    def get_driver_ids(self):
        if self.drivers is None:
            return None
        driver_ids = []
        for d in self.drivers:
            idx = self._node_index_from_tuple(tuple(d.tolist()))
            driver_ids.append(idx)
        return torch.tensor(driver_ids, device=self.nodes.device)

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