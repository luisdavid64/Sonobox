from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
from topology_utils import build_grid_nodes, build_edges_nearest

@dataclass
class MassSpringModel:
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
    
    def getListeners(self):
        return self.listeners
    
    def getDrivers(self):
        return self.drivers


    @classmethod
    def from_json(cls, path: str, device: Optional[torch.device] = None):
        import json
        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_config(config, device=device)