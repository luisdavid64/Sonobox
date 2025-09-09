from viz_utils import plot_model_graph_3d
import torch
from torch import nn
from typing import Optional
from topology_utils import build_grid_nodes, build_edges_by_type, dedupe_undirected

class MassSpringModel(nn.Module):
    def __init__(self, nodes, edge_index, springs, drivers=None, listeners=None, config=None, dimX=None, dimY=None, dimZ=None, interactionType="FIRST", bounds=[]):
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
        self.bounds = bounds
    
        self.N = nodes.shape[0]
        self.E = edge_index.shape[1]
        self.dt = 1e-3
        self.dim = 3
        # Learnable parameters (example: per-mass and per-spring)
        self.masses   = nn.Parameter(torch.full((self.N,), 1.0))          # per-mass
        self.damping  = nn.Parameter(torch.full((self.N,), 0.02))         # per-mass

        edge_index, springs = dedupe_undirected(edge_index, springs)
        self.edge_index = edge_index
        self.k    = nn.Parameter(springs[:, 0].clone())   # [E]
        self.damping = nn.Parameter(springs[:, 1].clone()) # [E] (edge damping)
        self.edge_z = self.damping #?
        self.rest = nn.Parameter(springs[:, 2].clone())   # [E]

        # Internal state: positions and velocities
        # Place nodes at their grid positions (columns 0:3 of nodes)
        self.rest_pos = self.nodes[:, 0:3].clone()
        self.x = torch.zeros((self.N, self.dim), device=self.nodes.device)  # displacement from rest
        self.v = torch.zeros((self.N, self.dim), device=self.nodes.device)
        self.forces = torch.zeros((self.N, self.dim), device=self.nodes.device)  # external forces

    def spring_forces(self, x, v):
        # positions in world space
        pos = self.rest_pos + x
        i, j = self.edge_index[0], self.edge_index[1]
        d  = pos[j] - pos[i]                              # [E,3]
        L  = (d.square().sum(-1) + 1e-12).sqrt()          # [E]
        dir = d / L.unsqueeze(-1)                         # [E,3]

        # Hooke force magnitude
        f_el = self.k * (L - self.rest)                   # [E]

        # Project relative velocity onto spring direction (edge damping)
        v_rel = (v[j] - v[i])                             # [E,3]
        v_rel_axis = (v_rel * dir).sum(-1)                # [E]
        f_damp = - self.damping * v_rel_axis               # [E]

        f_mag = (f_el + f_damp).unsqueeze(-1)             # [E,1]
        f_vec = f_mag * dir                               # [E,3]

        F = torch.zeros_like(x)
        F.index_add_(0, i,  f_vec)
        F.index_add_(0, j, -f_vec)
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

        F = self.spring_forces(x, v)
        fixed_mask = self.nodes[:, 5].view(-1, 1)            # 1.0 → fixed
        F = F + self.forces
        # zero any external/spring forces on fixed nodes
        F = F * (1.0 - fixed_mask)
        a = F / self.masses.unsqueeze(-1)
        # prevent accelerations on fixed nodes (defensive)
        a = a * (1.0 - fixed_mask)
        v = v + self.dt * a
        x = x + self.dt * v
        # enforce constraints
        v = v * (1.0 - fixed_mask)
        x = x * (1.0 - fixed_mask)

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

    def apply_force_nodes(self, node_indices: torch.Tensor, force_vec):
        f = torch.as_tensor(force_vec, device=self.nodes.device, dtype=self.x.dtype).view(1, self.dim)
        self.forces.index_add_(0, node_indices, f.expand(node_indices.numel(), -1))

    def apply_force_on_drivers(self, force_vec):
        ids = self.get_driver_ids()
        if ids is not None and ids.numel() > 0:
            self.apply_force_nodes(ids, force_vec)

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
        bounds = config.get("bounds", [])

        
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
            bounds=bounds,
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
        return cls(nodes, edge_index, springs, drivers, listeners, config, dimX=dimX, dimY=dimY, dimZ=dimZ, interactionType=interactionType, bounds=bounds)
    
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

    def get_fixed_ids(self):
        fixed_mask = self.nodes[:, 5]
        fixed_ids = torch.nonzero(fixed_mask, as_tuple=False).view(-1)
        return fixed_ids

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

    def visualize(self):
        plot_model_graph_3d(self)


    def _node_index_from_tuple(self, idx_tuple):
        i, j, k = idx_tuple
        return (i * self.dimY + j) * self.dimZ + k