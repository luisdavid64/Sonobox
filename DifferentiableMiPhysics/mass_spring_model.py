import torch
from torch import nn
from typing import Optional
from topology_utils import build_grid_nodes, build_edges_by_type, dedupe_undirected
from viz_utils import plot_model_graph_3d

class MassSpringModel(nn.Module):
    """
    miPhysics-style engine:
      - per-mass state: m_pos, m_posR (delayed pos), m_frc (force buffer)
      - per-mass params: inv_mass, mass_damp, radius
      - per-edge params: k (stiffness), rest (rest length), edge_z (relative damping)
      - tick order: reset_forces -> drivers -> interactions -> constraints -> integrate -> listeners
    """
    def __init__(self, nodes, edge_index, springs,
                 drivers=None, listeners=None, config=None,
                 dimX=None, dimY=None, dimZ=None,
                 interactionType="FIRST", bounds=[],
                 dt: float = 1e-3, gravity=(0.0, -9.81, 0.0)):
        super().__init__()
        # ----- topology & meta -----
        self.nodes = nodes              # [N, 8] (x,y,z,mass,radius,fixed,driver,listener)
        self.dimX, self.dimY, self.dimZ = dimX, dimY, dimZ
        self.interactionType = interactionType
        self.bounds = bounds
        self.config = config
        self.drivers = drivers
        self.listeners = listeners

        # Deduplicate edges (i<j) and merge attributes
        edge_index, springs = dedupe_undirected(edge_index, springs)
        self.register_buffer("edge_index", edge_index)     # [2, E]
        self.springs = springs                             # keep original for reference

        self.N = nodes.shape[0]
        self.E = self.edge_index.shape[1]
        self.dim = 3
        self.dt = float(dt)
        self.register_buffer("g", torch.tensor(gravity, dtype=nodes.dtype, device=nodes.device))

        # ----- per-mass parameters -----
        # mass, radius, fixed flags come from nodes
        mass_from_nodes = nodes[:, 3].clamp_min(1e-12)
        inv_mass_init = 1.0 / mass_from_nodes
        self.inv_mass  = nn.Parameter(inv_mass_init.clone())      # learnable if you want
        self.mass_damp = nn.Parameter(torch.full((self.N,), 0.02, device=nodes.device, dtype=nodes.dtype))
        self.radius    = nn.Parameter(nodes[:, 4].clone(), requires_grad=False)

        # fixed-mask (1.0 -> fixed); keep as buffer
        self.register_buffer("fixed_mask", nodes[:, 5].view(-1, 1).clone())

        # ----- per-edge parameters (from springs builder) -----
        # springs columns: [stiffness, edge_damping, rest_len, di, dj, dk]
        self.k      = nn.Parameter(springs[:, 0].clone())  # [E]
        self.edge_z = nn.Parameter(springs[:, 1].clone())  # [E]
        self.rest   = nn.Parameter(springs[:, 2].clone())  # [E]

        # ----- state -----
        # Rest configuration (world positions) from nodes[:, :3]
        self.register_buffer("rest_pos", nodes[:, 0:3].clone())
        # miPhysics-style state buffers
        self.register_buffer("m_pos", self.rest_pos.clone())   # current position
        self.register_buffer("m_posR", self.rest_pos.clone())  # delayed position (previous)
        self.register_buffer("m_frc", torch.zeros(self.N, self.dim, device=nodes.device, dtype=nodes.dtype), persistent=False)

        # Optional velocity-control (per-mass), like Mass.triggerVelocityControl in miPhysics
        self.register_buffer("ctrl_on", torch.zeros(self.N, 1, device=nodes.device, dtype=nodes.dtype))
        self.register_buffer("ctrl_vel", torch.zeros(self.N, self.dim, device=nodes.device, dtype=nodes.dtype))

        # Driver/listener index caches
        self._driver_ids = self.get_driver_ids()

        # Optional hooks
        self.driver_fn = None     # callable(self) -> writes into self.m_frc
        self.listener_fn = None   # callable(self) -> reads state

    # ---------------- miPhysics-like API ----------------
    def resetForce(self):
        self.m_frc.zero_()

    def applyForce(self, node_indices: torch.Tensor, force_vec):
        f = torch.as_tensor(force_vec, device=self.m_pos.device, dtype=self.m_pos.dtype).view(1, self.dim)
        self.m_frc.index_add_(0, node_indices, f.expand(node_indices.numel(), -1))

    def apply_force_on_drivers(self, force_vec):
        ids = self._driver_ids
        if ids is not None and ids.numel() > 0:
            self.applyForce(ids, force_vec)

    def triggerVelocityControl(self, node_indices: torch.Tensor, v_vec):
        v = torch.as_tensor(v_vec, device=self.m_pos.device, dtype=self.m_pos.dtype).view(1, self.dim)
        self.ctrl_on.index_fill_(0, node_indices, 1.0)
        self.ctrl_vel.index_copy_(0, node_indices, v.expand(node_indices.numel(), -1))

    def stopVelocityControl(self, node_indices: Optional[torch.Tensor] = None):
        if node_indices is None:
            self.ctrl_on.zero_()
            self.ctrl_vel.zero_()
        else:
            self.ctrl_on.index_fill_(0, node_indices, 0.0)
            self.ctrl_vel.index_fill_(0, node_indices, 0.0)

    # ---------------- interactions (springs) ----------------
    def spring_forces(self, pos, pos_prev):
        """
        Accumulate spring forces into a fresh tensor (does NOT modify self.m_frc).
        - Elastic: k * (L - rest)
        - Relative damping along spring: -edge_z * ((v_rel · dir)), v_rel ≈ ((pos - pos_prev) difference)
        """
        i, j = self.edge_index[0], self.edge_index[1]
        d  = pos[j] - pos[i]                       # [E,3]
        L  = (d.square().sum(-1) + 1e-12).sqrt()   # [E]
        dir = d / L.unsqueeze(-1)                  # [E,3]

        # Elastic
        f_el = self.k * (L - self.rest)            # [E]

        # Relative "velocity" along the edge (Verlet proxy)
        v_i = pos[i] - pos_prev[i]
        v_j = pos[j] - pos_prev[j]
        v_rel_axis = ((v_j - v_i) * dir).sum(-1)   # [E]
        f_damp = - self.edge_z * v_rel_axis        # [E]

        f_mag = (f_el + f_damp).unsqueeze(-1)      # [E,1]
        f_vec = f_mag * dir                        # [E,3]

        F = torch.zeros_like(pos)
        F.index_add_(0, i,  f_vec)
        F.index_add_(0, j, -f_vec)
        return F

    # ---------------- one physics tick ----------------
    def compute(self):
        """
        One miPhysics-style step:
          reset -> drivers -> interactions -> constraints -> integrate -> listeners
        Uses Verlet with per-mass viscous damping via velocity proxy.
        """
        dt = self.dt

        # 1) reset
        self.resetForce()

        # 2) drivers (external forces)
        if self.driver_fn is not None:
            self.driver_fn(self)
        # (You can also call apply_force_on_drivers(...) before compute())

        # 3) interactions (springs)
        Fspr = self.spring_forces(self.m_pos, self.m_posR)
        Fext = self.m_frc
        F = Fspr + Fext

        # 4) constraints/masks (fixed / gravity / mass damping)
        # Gravity as mass*G
        F = F + self.g.view(1, 3) * (1.0 / (self.inv_mass + 1e-12)).unsqueeze(-1)
        # Per-mass viscous damping using velocity proxy
        vel = self.m_pos - self.m_posR
        F = F - self.mass_damp.unsqueeze(-1) * vel
        # Zero forces on fixed nodes
        F = F * (1.0 - self.fixed_mask)

        # 5) integrate (Verlet with inverse mass)
        a = self.inv_mass.unsqueeze(-1) * F                    # a = F * inv_mass
        # Optional velocity control (like triggerVelocityControl in miPhysics)
        vel_ctrl = torch.where(self.ctrl_on > 0, self.ctrl_vel, torch.zeros_like(self.ctrl_vel))
        # Verlet step with control term added as extra velocity
        vel_eff = vel + dt * a + vel_ctrl * self.ctrl_on
        x_new = self.m_pos + vel_eff

        # Enforce fixed nodes: keep them at rest pose (or current posR) and zero velocity
        x_new = (1.0 - self.fixed_mask) * x_new + self.fixed_mask * self.m_pos

        # Rotate buffers
        self.m_posR, self.m_pos = self.m_pos, x_new

        # 6) listeners
        if self.listener_fn is not None:
            self.listener_fn(self)

        # clear forces for next tick (miPhysics resets each compute)
        self.resetForce()

        return self.m_pos

    def computeNsteps(self, N: int, substeps: int = 1):
        """
        Run N ticks. If substeps>1, splits dt evenly (useful for stiff systems / audio blocks).
        """
        if substeps == 1:
            for _ in range(N):
                self.compute()
        else:
            base_dt = self.dt
            self.dt = base_dt / substeps
            for _ in range(N):
                for _ in range(substeps):
                    self.compute()
            self.dt = base_dt
        return self.m_pos

    # ---------------- utilities ----------------
    @classmethod
    def from_config(cls, config: dict, device: Optional[torch.device] = None):
        def parse_mass_name(name):
            p = name.split("_")
            if len(p) != 4 or p[0] != "m":
                raise ValueError(f"Invalid mass name: {name}")
            return tuple(int(x) for x in p[1:])

        geom = config["geometry"]
        dimX, dimY, dimZ = geom["dx"], geom["dy"], geom["dz"]
        dist = float(geom["distance"])
        mass_radius = float(geom["massesRadius"])
        mass = float(config.get("parameters", {}).get("M", [1.0])[0])
        stiffness = float(config.get("parameters", {}).get("K", [1e-3])[0])
        edge_damp = float(config.get("parameters", {}).get("C", [0.0])[0])
        interactionType = str(geom.get("interactionType", "FIRST"))
        drivers = torch.tensor([parse_mass_name(n) for n in config["sonification_set_up"]["drivers"]], device=device) \
                  if "sonification_set_up" in config else None
        listeners = torch.tensor([parse_mass_name(n) for n in config["sonification_set_up"]["listeners"]], device=device) \
                  if "sonification_set_up" in config else None
        bounds = config.get("bounds", [])

        nodes = build_grid_nodes(dimX, dimY, dimZ, dist,
                                 mass=mass, radius=mass_radius,
                                 drivers=drivers, listeners=listeners,
                                 bounds=bounds, device=device)
        edge_index, springs = build_edges_by_type(dimX, dimY, dimZ, dist,
                                                  stiffness=stiffness, damping=edge_damp,
                                                  interaction_type=interactionType, device=device)
        return cls(nodes, edge_index, springs, drivers, listeners, config,
                   dimX=dimX, dimY=dimY, dimZ=dimZ,
                   interactionType=interactionType, bounds=bounds)
                
    @classmethod
    def from_json(cls, path: str, device: Optional[torch.device] = None):
        import json
        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_config(config, device=device)

    def get_driver_ids(self):
        if self.drivers is None:
            return None
        ids = [(i * self.dimY + j) * self.dimZ + k for (i, j, k) in self.drivers.tolist()]
        return torch.tensor(ids, device=self.nodes.device, dtype=torch.long)

    def get_fixed_ids(self):
        return torch.nonzero(self.fixed_mask.view(-1) > 0.5, as_tuple=False).view(-1)

    def visualize(self):
        plot_model_graph_3d(self)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Grid 3x3x3, nearest-neighbor springs
    model = MassSpringModel.from_json("../model_configs/sonobox_data/baselines/biosonix_3D.json", device=device)

    traj = []
    T = int(600)
    x = None
    v = None
    for t in range(T):
        if t == 0:
            model.apply_force_on_drivers((1,1,1))
        x = model.compute()
        traj.append(x)
    traj = torch.stack(traj)  # [T,N,dim]

    print("max |x|:", traj.abs().max().item())
    print("driver ids:", model.get_driver_ids())
    print("driver fixed flags:", model.nodes[model.get_driver_ids(), 5])
    print("fixed ids:", model.get_fixed_ids())
    from viz_utils import animate_trajectory_3d
    plot_model_graph_3d(model)
    animate_trajectory_3d(traj, model)  # XZ