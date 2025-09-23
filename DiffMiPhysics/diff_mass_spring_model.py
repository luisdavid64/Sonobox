from asyncio import events
from matplotlib.pylab import dtype
import torch
from torch import nn
from typing import Optional
from util.topology_utils import build_grid_nodes, build_edges_by_type, dedupe_undirected
from util.viz_utils import plot_model_graph_3d, render_traj_taichi3d, plot_spectrogram, run_interactive_mi
from util.audio_helpers import axis_pick_t, mix_down, normalize_rms_to_dbfs, rms_normalize
from util.util import event_dict_seconds_to_samples, load_event_from_json, save_event_to_json
from util.config_utils import load_config, save_config, model_to_config
import json
from exciters import *

class MassSpringModel(nn.Module):
    """
    miPhysics-style engine:
      - per-mass state: m_pos, m_posR (delayed pos), m_frc (force buffer)
      - per-mass params: inv_mass, mass_damp, radius
      - per-edge params: k (stiffness), rest (rest length), z (relative damping)
      - miPhysics: simulates at audio rate (dt=1/16000)
    """
    def __init__(self, nodes, edge_index, springs,
                 drivers=None, listeners=None, config=None,
                 dimX=None, dimY=None, dimZ=None, dist=None,
                 interactionType="FIRST", bounds=[],
                 dt: float = 1/44100, friction: float = 0.25, gain=10):
        super().__init__()
        # ----- topology & meta -----
        self.nodes = nodes              # [N, 8] (x,y,z,mass,radius,fixed,driver,listener)
        self.dimX, self.dimY, self.dimZ = dimX, dimY, dimZ
        self.interactionType = interactionType
        self.bounds = bounds
        self.config = config
        self.drivers = drivers
        self.listeners = listeners
        self.gain = gain

        # Deduplicate edges (i<j) and merge attributes
        edge_index, springs = dedupe_undirected(edge_index, springs)
        self.register_buffer("edge_index", edge_index)     # [2, E]
        self.springs = springs                             # keep original for reference

        self.N = nodes.shape[0]
        self.E = self.edge_index.shape[1]
        self.dim = 3
        self.dt = float(dt)
        self.dist = dist

        # ----- per-mass parameters -----
        mass_from_nodes = nodes[:, 3].clamp_min(1e-12)
        self.register_buffer("mass", mass_from_nodes.clone())
        inv_mass_init = 1.0 / mass_from_nodes[0]  # per miPhysics
        # self.inv_mass = nn.Parameter(inv_mass_init.clone(), requires_grad=False)
        self.log_inv_mass = nn.Parameter(torch.log(inv_mass_init.clone()), requires_grad=False)
        
        self.radius    = nn.Parameter(nodes[:, 4].clone(), requires_grad=False)

        # fixed nodes: 5th col of nodes is fixed flag (0/1)
        self.register_buffer("fixed_mask", nodes[:, 5].view(-1, 1).clone())

        # ----- per-edge parameters (from springs builder) -----
        # springs columns: [stiffness, edge_damping, rest_len, neighbor_type]
        # If springs has neighbor_type column, use it to parameterize k
        if springs.shape[1] == 4:
            self.neighbor_type = springs[:, 3].long()  # 0=first, 1=second
        else:
            self.neighbor_type = torch.zeros(springs.shape[0], dtype=torch.long, device=nodes.device)
        self._logk_first  = nn.Parameter(torch.log(torch.tensor(float(springs[self.neighbor_type==0, 0].mean()), device=nodes.device)))
        self._logk_second = nn.Parameter(torch.log(torch.tensor(float(springs[self.neighbor_type==1, 0].mean()) if (self.neighbor_type==1).any() else springs[:,0].mean(), device=nodes.device)))
        self._logz  = nn.Parameter(torch.log(torch.tensor(float(springs[:, 1].mean()), device=nodes.device)))
        # We do not optimize rest length directly, but compute from geometry
        self.register_buffer("rest", springs[:, 2].clone())  # [E] (will be overwritten by geometric L0 below)

        # ----- state -----
        # Rest configuration (world positions) from nodes[:, :3]
        self.register_buffer("rest_pos", nodes[:, 0:3].clone())

        # spring previous-distance state (used by dashpot term)
        self.register_buffer("m_prevDist", torch.zeros(self.E, device=nodes.device, dtype=nodes.dtype))

        # Initialize rest lengths from geometry
        with torch.no_grad():
            i, j = self.edge_index[0], self.edge_index[1]
            d0 = self.rest_pos[j] - self.rest_pos[i]
            L0 = (d0.pow(2).sum(-1) + 1e-12).sqrt()
            self.rest.data.copy_(L0)
            self.m_prevDist.copy_(L0)  # initialize dashpot distance at rest

        # miPhysics-style state buffers
        self.register_buffer("m_pos", self.rest_pos.clone())   # current position
        self.register_buffer("m_posR", self.rest_pos.clone())  # delayed position (previous)
        self.register_buffer("m_frc", torch.zeros(self.N, self.dim, device=nodes.device, dtype=nodes.dtype), persistent=False)

        # Driver/listener index caches
        self._driver_ids = self.get_driver_ids()
        self._listener_ids = self.get_listener_ids()

        # Global
        self._u_fric = nn.Parameter(torch.log(torch.tensor(friction, device=self.nodes.device, dtype=self.nodes.dtype)))  # maps to [0,2]
        # We could use gravity if desired
        self.register_buffer("gravity", torch.zeros(3, device=self.nodes.device, dtype=self.nodes.dtype))

        # listener HPF state
        self.m_coef = 0.95
        self.register_buffer("hp_x_prev", torch.zeros(0))
        self.register_buffer("hp_y_prev", torch.zeros(0))
        self.hp_primed = False  # if False, we will prime on first call

        self.compute = self.compute_explicit_euler
        self.hp_filter = self.simple_highpass

    @property
    def k_1(self): return torch.exp(self._logk_first)
    @property
    def k_2(self): return torch.exp(self._logk_second)
    @property
    def k(self):
        # Returns k for all edges, using neighbor_type
        k = torch.where(self.neighbor_type==0, self.k_1, self.k_2)
        return k
    @property
    def z(self):    return torch.exp(self._logz)
    @property
    def fric(self): return torch.exp(self._u_fric)   # keep c in [0,2]
    @property
    def inv_mass(self): return torch.exp(self.log_inv_mass)   # keep c in [0,2]

    # ---------------- miPhysics-like API ----------------

    def resetForce(self):
        self.m_frc = torch.zeros_like(self.m_frc)

    def applyForce(self, node_indices: torch.Tensor, force_vec):
        f = torch.as_tensor(force_vec, device=self.m_pos.device, dtype=self.m_pos.dtype).view(1, self.dim)
        # functional index_add to avoid in-place autograd hazards
        self.m_frc = torch.index_add(self.m_frc, 0, node_indices, f.expand(node_indices.numel(), -1))

    def apply_force_on_drivers(self, force_vec):
        ids = self._driver_ids
        if ids is not None and ids.numel() > 0:
            self.applyForce(ids, force_vec)

    def apply_force_on_driver_id(self, id, force_vec):
        self.applyForce(id, force_vec)

    # ---------------- interactions (springs) ----------------
    torch.compile()
    def spring_damper_forces(self):
        """
        miPhysics exact spring force:
        f_spring = -K*(L - rest) - Z*(L - m_prevDist)
        dir is the *current* unit direction.
        Updates self.m_prevDist <- L (kept with full graph for BPTT).
        Uses k_first for first neighbors, k_second for second neighbors.
        """
        k = self.k
        z = self.z
        i, j = self.edge_index[0], self.edge_index[1]          # [E]
        d    = self.m_pos[j] - self.m_pos[i]                   # [E,3]
        m_dist = (d.pow(2).sum(-1) + 1e-12).sqrt()             # [E]
        invL   = torch.where(m_dist > 1e-9, 1.0 / m_dist, torch.zeros_like(m_dist))
        dirv   = d * invL.unsqueeze(-1)                        # [E,3]

        # scalar link force (Hooke + dashpot on distance change)
        f_el   = - k * (m_dist - self.rest)          # [E]
        f_damp = - z * (m_dist - self.m_prevDist)    # [E]
        f_spring = f_el + f_damp                                 # [E]

        f_vec = f_spring.unsqueeze(-1) * dirv                    # [E,3]

        # scatter to nodes (functional index_add to avoid in-place)
        F = torch.zeros_like(self.m_pos)                       # [N,3]
        F = torch.index_add(F, 0, i, -f_vec)
        F = torch.index_add(F, 0, j,  f_vec)

        # update previous distance for next tick WITH graph (no detach)
        self.m_prevDist = m_dist.clone()
        return F

    def compute_semi_implicit(self):
        invM = self.inv_mass.view(-1,1)
        v = self.m_pos - self.m_posR
        a = (self.m_frc * invM) - self.gravity
        v_new = v * (1.0 - self.fric.view(-1,1)) + a
        x_new = self.m_pos + v_new
        self.m_posR = self.m_pos
        self.m_pos  = (1-self.fixed_mask)*x_new + self.fixed_mask*self.rest_pos
        Fspr = self.spring_damper_forces()
        self.m_frc = Fspr * (1.0 - self.fixed_mask)

    torch.compile()
    def compute_explicit_euler(self):
        # --- 1) integrate with previous forces ---
        invM  = self.inv_mass.view(-1, 1)                      # [N,1]

        fric = self.fric
        c     = (invM * fric.view(-1,1))                       # [N,1]
        gterm = self.gravity.view(1,3)                         # [1,3]
        F     = self.m_frc                                     # [N,3]

        x, xr = self.m_pos, self.m_posR

        acc_term = F * invM
        x_new    = x * (2.0 - c) - xr * (1.0 - c) + acc_term - gterm

        # roll states
        self.m_posR = x
        self.m_pos  = x_new

        # enforce fixed nodes → keep at rest (differentiable mask)
        fixed = self.fixed_mask
        rest  = self.rest_pos
        self.m_pos  = (1.0 - fixed) * self.m_pos  + fixed * rest
        self.m_posR = (1.0 - fixed) * self.m_posR + fixed * rest

        # --- 2) rebuild forces for next step (springs + drivers) ---
        Fspr = self.spring_damper_forces()
        # no forces on fixed nodes
        self.m_frc = Fspr * (1.0 - fixed)
        return self.m_pos

    def computeNsteps(self, N: int, substeps: int = 1):
        """
        Run N ticks. If substeps>1, splits dt evenly (useful for stiff systems / audio blocks).
        """
        for _ in range(N):
            for _ in range(substeps):
                self.compute()
        return self.m_pos

    # ---------------- io utilities ----------------
    @classmethod
    def from_config(cls, config: dict, device: Optional[torch.device] = None, dt: float = 1/16000):
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
        drivers = torch.tensor([parse_mass_name(n) for n in config.get("sonification_set_up", {}).get("drivers", [])],
                               device=device) if "sonification_set_up" in config else None
        listeners = torch.tensor([parse_mass_name(n) for n in config.get("sonification_set_up", {}).get("listeners", [])],
                                 device=device) if "sonification_set_up" in config else None
        bounds = config.get("bounds", [])
        friction = config.get("global_friction", 0.25)

        nodes = build_grid_nodes(dimX, dimY, dimZ, dist,
                                 mass=mass, radius=mass_radius,
                                 drivers=drivers, listeners=listeners,
                                 bounds=bounds, device=device)
        edge_index, springs = build_edges_by_type(dimX, dimY, dimZ, dist,
                                                  stiffness=stiffness, damping=edge_damp,
                                                  interaction_type=interactionType, device=device)
        return cls(nodes, edge_index, springs, drivers, listeners, config,
                   dimX=dimX, dimY=dimY, dimZ=dimZ, dist=dist,
                   interactionType=interactionType, bounds=bounds, dt=dt, friction=friction)

    def to_config(self):
        return model_to_config(self)

    def to_json(self, path: str):
        config = self.to_config()
        save_config(config, path)

    @classmethod
    def from_json(cls, path: str, device: Optional[torch.device] = None, dt: float = 1/16000):
        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_config(config, device=device, dt=dt)

    # ---------------- mass utilities ----------------

    def get_ids(self, tensor):
        return [(i * self.dimY + j) * self.dimZ + k for (i, j, k) in tensor.tolist()]

    def get_driver_ids(self):
        if self.drivers is None or len(self.drivers) == 0:
            return None
        ids = self.get_ids(self.drivers)
        return torch.tensor(ids, device=self.nodes.device, dtype=torch.long)

    def get_listener_ids(self):
        if self.listeners is None or len(self.listeners) == 0:
            return None
        ids = self.get_ids(self.listeners)
        return torch.tensor(ids, device=self.nodes.device, dtype=torch.long)

    def get_fixed_ids(self):
        return torch.nonzero(self.fixed_mask.view(-1) > 0.5, as_tuple=False).view(-1)

    # ---------------- visualization ----------------
    """TODO: Implement dynamic visualization and interaction"""

    def visualize(self):
        plot_model_graph_3d(self)

    #---------------- audio ----------------

    def _ensure_hp_state(self, C, device, dtype):
            if self.hp_x_prev.numel() != C:
                self.hp_x_prev = torch.zeros(C, device=device, dtype=dtype)
                self.hp_y_prev = torch.zeros(C, device=device, dtype=dtype)
                self.hp_primed = False

    torch.compile()
   # Highpass with no recursion: faster, but not loyal to miphysics sound
    def simple_highpass(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.zeros_like(x)
        y[1:] = x[1:] - x[:-1]
        return y

    torch.compile(backend="aot_eager")
    def highpass_observer3d(self, x: torch.Tensor, R: float | None = None,
                            prime_on_first_call: bool = True) -> torch.Tensor:
        """
        x: [T, C] already axis-picked (pos/force)
        y[n] = x[n] - x[n-1] + R*y[n-1], with persistent state.
        If prime_on_first_call: set x[-1]=x[0], y[-1]=0 at first call to avoid a click.
        """
        if R is None: R = self.m_coef
        T, C = x.shape
        self._ensure_hp_state(C, x.device, x.dtype)

        y = torch.empty_like(x)
        x_prev = self.hp_x_prev
        y_prev = self.hp_y_prev

        # Optional priming to avoid the initial pop
        if prime_on_first_call and not self.hp_primed and T > 0:
            x_prev = x[0]            # treat "previous input" as first sample
            y_prev = torch.zeros_like(x_prev)
            self.hp_primed = True

        R = torch.as_tensor(R, device=x.device, dtype=x.dtype)

        for n in range(T):
            y_n = x[n] - x_prev + R * y_prev
            y[n] = y_n
            x_prev = x[n]
            y_prev = y_n

        # store state (detached so no graph carry)
        self.hp_x_prev = x_prev.detach()
        self.hp_y_prev = y_prev.detach()
        return y

    def simulate_listeners(self,
                           steps: int,
                           listener_ids: torch.Tensor,
                           observable: str = "pos",   # 'pos' | 'vel' | 'acc' | 'force'
                           axis: str = "all",
                           events: dict = {},
                           exciter = None,
                           start_frame=0
                           ) -> torch.Tensor:
        """
        Run `steps` physics ticks and return raw multichannel listener signal at sim rate.
        Output shape: [steps, C] (C = #listeners). Uses current self.dt.
        """
        device = self.nodes.device
        ids = listener_ids.to(device, dtype=torch.long)
        C = ids.numel()
        out = torch.zeros(steps, C, device=device, dtype=self.m_pos.dtype)
        for t in range(steps):
            t_real = t + start_frame
            if t_real in events:
                self.apply_force_on_drivers(events[t_real])
            # if exciter is not None:
            #     f_exciter = exciter.step(t)
            #     self.apply_force_on_drivers(f_exciter)
            self.compute()

            # absolute pos
            pos_abs = self.m_pos
            if observable == "pos":
                val = pos_abs[ids]                                   # [C,3]
                out[t] = axis_pick_t(val, axis)                     # [C]
            elif observable == "force":
                # recompute spring forces at *current* state
                Fspr = self.spring_damper_forces()   # [N,3]
                val = Fspr[ids]
                out[t] = axis_pick_t(val, axis)
            else:
                raise ValueError("observable must be 'pos' | 'force'.")

        return out  # [steps, C]

    def render_audio(self,
                     seconds: float,
                     fs: int = 16000,
                     observable: str = "pos",
                     axis: str = "all",
                     listener_ids: torch.Tensor | None = None,
                     layout: str = "stereo",
                     pan_method: str = "by_position",
                     hp: bool = True,
                     gain: float | None = None,
                     events: dict = {},
                     exciter = None,
                     mix_audio: bool = True,
                     start_frame=0
                     ) -> torch.Tensor:
        """
        End-to-end: simulate at current dt for `seconds`, capture listeners,
        resample to `fs`, downmix, return audio [T_audio, K].
        """
        device = self.nodes.device

        if gain is None:
            gain = self.gain
        
        if listener_ids is None:
            ids = self.get_driver_ids()
            if ids is None or ids.numel() == 0:
                ids = torch.tensor([self.N // 2], device=device, dtype=torch.long)
            listener_ids = ids

        steps = int(round(seconds * fs))
        raw = self.simulate_listeners(steps, listener_ids, observable=observable, axis=axis, events=events, exciter=exciter, start_frame=start_frame)  # [T_sim,C]

        if hp:
            # raw = self.highpass_observer3d(raw, R=0.95, prime_on_first_call=True)
            raw = self.hp_filter(raw)


        # resample to audio
        audio = raw

        # mix to target layout
        if mix_audio:
            audio = mix_down(
                self,
                audio,
                layout=layout,             # mono is cheaper for CLAP; stereo if you need it
                method=pan_method,
                listener_ids=listener_ids,
                normalize=False,
                soft_clip=False,
                energy_comp=True,
        ) # For stereo, energy_comp helps keep loudness stable

        # final gain & clamp
        # if not self.training:
        # audio = audio * gain
        audio = audio.transpose(1,0)
        peak = torch.maximum(torch.abs(audio).amax(dim=1), torch.tensor(1e-9, device=audio.device)).detach()
        audio = audio / peak.unsqueeze(-1)
        return audio  # [T_audio, K]
    
    def postprocess_audio(self, audio, apply_gain=False):
        audio = normalize_rms_to_dbfs(audio)
        if self.gain:
            audio = audio * self.gain
        return audio

    @torch.no_grad()
    def detach_state(self, reset_to_rest=True):
        if reset_to_rest:
            # recompute L0 and ASSIGN
            i, j = self.edge_index[0], self.edge_index[1]
            d0 = self.rest_pos[j] - self.rest_pos[i]
            L0 = (d0.pow(2).sum(-1) + 1e-12).sqrt()
            self.m_prevDist = L0  # <- not .copy_()
        else:
            self.m_prevDist = self.m_prevDist.detach()
        # also reassign these (no in-place)
        self.m_pos  = self.rest_pos.clone() if reset_to_rest else self.m_pos.detach()
        self.m_posR = self.rest_pos.clone() if reset_to_rest else self.m_posR.detach()
        self.m_frc  = torch.zeros_like(self.m_frc)
        self.hp_x_prev = torch.zeros(0, device=self.nodes.device, dtype=self.nodes.dtype)
        self.hp_y_prev = torch.zeros(0, device=self.nodes.device, dtype=self.nodes.dtype)
        self.hp_primed = False

    @torch.no_grad()
    def run_interactive(self):
        run_interactive_mi(self, sim_rate=int(1/self.dt))

    # ---------------- Modal usage ----------------

    def _free_dof_index(self):
        # 1 for free dofs, 0 for fixed
        free_mask_node = (1.0 - self.fixed_mask.view(-1))  # [N]
        free_mask_dof  = free_mask_node.repeat_interleave(3) > 0.5  # [3N]
        free_idx = torch.nonzero(free_mask_dof, as_tuple=False).view(-1)
        return free_idx

    def _assemble_KZ_dense(self):
        """
        Linearization around rest: F_k ≈ -K x_k - Z (x_k - x_{k-1})
        K,Z ∈ R^{3×3N}, symmetric PSD. No h-scaling here (this is *discrete* damping).
        """
        device, dtype = self.nodes.device, self.nodes.dtype
        N = self.N
        dofN = 3 * N

        # Edge geometry
        i, j = self.edge_index[0], self.edge_index[1]    # [E]
        pi, pj = self.rest_pos[i], self.rest_pos[j]      # [E,3]
        r  = pj - pi
        L  = (r.pow(2).sum(-1) + 1e-12).sqrt()
        n  = r / L.unsqueeze(-1)                          # [E,3]
        nx, ny, nz = n[:,0], n[:,1], n[:,2]
        Pxx = nx*nx; Pxy = nx*ny; Pxz = nx*nz
        Pyx = ny*nx; Pyy = ny*ny; Pyz = ny*nz
        Pzx = nz*nx; Pzy = nz*ny; Pzz = nz*nz

        k_e = self.k   # [E]
        z_e = self.z   # [E]  (discrete “difference” dashpot gain)

        def scatter_axial(val_e):
            Bxx = val_e*Pxx; Bxy = val_e*Pxy; Bxz = val_e*Pxz
            Byx = val_e*Pyx; Byy = val_e*Pyy; Byz = val_e*Pyz
            Bzx = val_e*Pzx; Bzy = val_e*Pzy; Bzz = val_e*Pzz
            dof_i = (i*3).unsqueeze(1) + torch.tensor([0,1,2], device=device)
            dof_j = (j*3).unsqueeze(1) + torch.tensor([0,1,2], device=device)
            ii = dof_i.unsqueeze(2).expand(-1,3,3).reshape(-1)
            ij = dof_i.unsqueeze(2).expand(-1,3,3).reshape(-1)
            ji = dof_j.unsqueeze(2).expand(-1,3,3).reshape(-1)
            jj = dof_j.unsqueeze(2).expand(-1,3,3).reshape(-1)
            ci = dof_i.unsqueeze(1).expand(-1,3,3).reshape(-1)
            cj = dof_j.unsqueeze(1).expand(-1,3,3).reshape(-1)
            ci2= dof_i.unsqueeze(1).expand(-1,3,3).reshape(-1)
            cj2= dof_j.unsqueeze(1).expand(-1,3,3).reshape(-1)
            block = torch.stack([Bxx,Bxy,Bxz, Byx,Byy,Byz, Bzx,Bzy,Bzz], dim=1).reshape(-1)
            return (ii,ci, block), (ij,cj, -block), (ji,ci2, -block), (jj,cj2, block)

        K = torch.zeros((dofN, dofN), device=device, dtype=dtype)
        Z = torch.zeros((dofN, dofN), device=device, dtype=dtype)
        for A, val in ((K, k_e), (Z, z_e)):
            (ii,ci,p), (ij,cj,m1), (ji,ci2,m2), (jj,cj2,p2) = scatter_axial(val)
            A.index_put_((ii,ci), p,  accumulate=True)
            A.index_put_((ij,cj), m1, accumulate=True)
            A.index_put_((ji,ci2), m2, accumulate=True)
            A.index_put_((jj,cj2), p2, accumulate=True)

        # Reduce to free DOFs
        idx = self._free_dof_index()
        Kf = K.index_select(0, idx).index_select(1, idx)
        Zf = Z.index_select(0, idx).index_select(1, idx)
        return Kf, Zf, idx

    def _build_Bu_full(self, drivers: torch.Tensor) -> torch.Tensor:
        device, dtype = self.nodes.device, self.nodes.dtype
        Nd, dofN = int(drivers.numel()), 3*self.N
        Bu = torch.zeros((dofN, 3*Nd), device=device, dtype=dtype)
        for d, node in enumerate(drivers.tolist()):
            base = 3*node
            Bu[base+0, 3*d+0] = 1.0
            Bu[base+1, 3*d+1] = 1.0
            Bu[base+2, 3*d+2] = 1.0
        return Bu

    def _build_C_full(self, listener_ids: torch.Tensor, axis: str='avg') -> torch.Tensor:
        device, dtype = self.nodes.device, self.nodes.dtype
        Cn, dofN = int(listener_ids.numel()), 3*self.N
        C = torch.zeros((Cn, dofN), device=device, dtype=dtype)
        for c, node in enumerate(listener_ids.tolist()):
            base = 3*node
            if axis == 'x':
                C[c, base+0] = 1.0
            elif axis == 'y':
                C[c, base+1] = 1.0
            elif axis == 'z':
                C[c, base+2] = 1.0
            else:  # 'avg' keeps it linear
                C[c, base+0] = C[c, base+1] = C[c, base+2] = 1.0/3.0
        return C

    def _rasterize_events_to_u(self, T:int, Nd:int, events:dict,
                            device, dtype, hold:int=8,
                            shape:str="cos", driver_axis:str="z"):
        u = torch.zeros(T, 3*Nd, device=device, dtype=dtype)
        if not events: return u
        # envelope
        if hold <= 1:
            env = torch.ones(1, device=device, dtype=dtype)
        else:
            t = torch.arange(hold, device=device, dtype=dtype)
            if shape == "cos":
                env = 0.5 - 0.5*torch.cos(2*torch.pi*(t/(hold-1)))
            elif shape == "exp":
                env = torch.exp(-3.0 * t/(hold-1))
            else:
                env = torch.ones_like(t, dtype=dtype)
        # axis for scalar events
        ax = {"x":0,"y":1,"z":2}.get(driver_axis, 2)
        ax_mask = torch.zeros(3, device=device, dtype=dtype); ax_mask[ax] = 1.0

        for t0, f in events.items():
            t0 = int(t0)
            if t0 >= T: continue
            f = torch.as_tensor(f, device=device, dtype=dtype).view(-1)
            if f.numel()==1: f3 = f[0]*ax_mask
            elif f.numel()>=3: f3 = f[:3]
            else: continue
            fv = f3.view(1,3).expand(Nd,3).reshape(1, 3*Nd)
            t1 = min(T, t0 + env.numel())
            u[t0:t1, :] += env[:t1-t0].view(-1,1) * fv
        return u

    def _init_modal_disc(self):
        if not hasattr(self, "_modal_disc"):
            self._modal_disc = dict()

    def _compute_disc_modes(self, Kf, Zf, idx_free, drivers, listeners, axis, n_modes, diag_gamma=False):
        """
        Diagonalize Kf (M = I here since your inv_mass is scalar), keep k lowest modes.
        Then project Zf into this basis (full Γ for fidelity, or diag only).
        """
        self._init_modal_disc()
        device, dtype = Kf.device, Kf.dtype
        dof_free = Kf.shape[0]
        k = min(n_modes, dof_free)

        # Small/medium -> full eigh; large -> lobpcg
        if dof_free <= 4000:
            w2, U = torch.linalg.eigh(Kf)         # ascending
            w2 = w2[:k]; U = U[:, :k]
        else:
            # LOBPCG for k smallest
            X0 = torch.randn(dof_free, k, device=device, dtype=dtype)
            w2, U = torch.lobpcg(Kf, k=k, B=None, iK=None, niter=100, X=X0, largest=False)
            # sort just in case
            srt = torch.argsort(w2); w2 = w2[srt]; U = U[:, srt]

        # Projections
        Zk = U.T @ (Zf @ U)              # [k,k]
        if diag_gamma:
            Zk = torch.diag(torch.diag(Zk))
        

        # Drivers/listeners (reduce → project)
        Bu_full = self._build_Bu_full(drivers)
        C_full  = self._build_C_full(listeners, axis=axis)
        Bu = Bu_full.index_select(0, idx_free)         # [dof_f, 3*Nd]
        C  = C_full.index_select(1, idx_free)          # [Cl, dof_f]
        Gu = U.T @ Bu                                   # [k, 3*Nd]
        Gy = C @ U                                      # [Cl, k]
    


        # self._modal_disc.update({
        #     "U": U,                # [dof_f, k]
        #     "w2": w2,              # [k]
        #     "Zk": Zk,              # [k,k]
        #     "Gu": Gu,              # [k, 3*Nd]
        #     "Gy": Gy,              # [Cl, k]
        #     "idx_free": idx_free,
        #     "drivers": drivers.detach().clone(),
        #     "listeners": listeners.detach().clone(),
        #     "axis": axis,
        #     "diag_gamma": diag_gamma,
        # })
        det = lambda t: t.detach()
        self._modal_disc.update({
            "U":  det(U),
            "w2": det(w2),
            "Zk": det(Zk),
            "Gu": det(Gu),
            "Gy": det(Gy),
            "idx_free": idx_free,
            "drivers": drivers.detach().clone(),
            "listeners": listeners.detach().clone(),
            "axis": axis,
            "diag_gamma": diag_gamma,
        })
        

    # @torch.compile()
    def render_modal_audio(self,
                        seconds: float,
                        fs: int = 16000,
                        listener_ids: Optional[torch.Tensor] = None,
                        drivers: Optional[torch.Tensor] = None,
                        axis: str = "avg",
                        n_modes: int = 512,
                        gamma: str = "full",    # 'full' or 'diag'
                        hp: bool = True,
                        events: dict = {},
                        mix_audio: bool = True,
                        layout: str = "mono",
                        pan_method: str = "by_position",
                        start_frame: int = 0):
        """
        Modal rendering with toggleable damping:
        - gamma='full': uses full Γ = U^T Z U (k×k) → O(k^2) per step
        - gamma='diag': uses only diag(Γ)        → O(k) per step
        """
        device, dtype = self.nodes.device, self.nodes.dtype
        listener_ids = self.get_listener_ids() if listener_ids is None else listener_ids
        drivers      = self.get_driver_ids()   if drivers      is None else drivers
        diag_gamma   = (gamma == "diag")

        # 1) Assemble reduced K, Z once (you can later switch this to linear-combo bases)
        Kf, Zf, idx_free = self._assemble_KZ_dense()

        # 2) Modal cache check (same as your code)
        need_modes = (not hasattr(self, "_modal_disc") or
                    self._modal_disc.get("U") is None or
                    self._modal_disc.get("diag_gamma") != diag_gamma or
                    (self._modal_disc.get("drivers") is None) or
                    not torch.equal(self._modal_disc["drivers"], drivers) or
                    not torch.equal(self._modal_disc["listeners"], listener_ids) or
                    self._modal_disc.get("axis") != axis or
                    self._modal_disc.get("idx_free", None) is None or
                    not torch.equal(self._modal_disc["idx_free"], idx_free) or
                    self._modal_disc["U"].shape[1] < min(n_modes, Kf.shape[0]))

        if need_modes:
            # This fills a cache with U (subspace), etc. You can leave it as-is.
            self._compute_disc_modes(Kf, Zf, idx_free, drivers, listener_ids, axis, n_modes, diag_gamma=diag_gamma)

        # --- Build a *training-safe* subspace projection ---
        # If you want grads through U, avoid .detach(); if you only train scalars (k,z,fric), detaching is fine.
        U0       = self._modal_disc["U"]         # [df, k0] cached subspace (k0 >= k)
        idx_free = self._modal_disc["idx_free"]

        # Project current K, Z into U0 (small k0×k0)
        S_K = U0.T @ (Kf @ U0)                   # [k0, k0]
        S_Z = U0.T @ (Zf @ U0)                   # [k0, k0]

        # Small eigendecomp → the first k modes
        lam, Vk_full = torch.linalg.eigh(S_K)    # ascending
        k = min(n_modes, lam.numel())
        w2 = lam[:k]                             # [k]
        Vk = Vk_full[:, :k]                      # [k0, k]
        U  = U0 @ Vk                             # [df, k] final modal basis

        # Damping representation
        if diag_gamma:
            # diag(Γ) = diag( Vk^T S_Z Vk )  without forming full k×k Γ
            SZV = S_Z @ Vk                      # [k0, k]
            gamma_vec = (Vk * SZV).sum(dim=0)   # [k]
            Zk = None
        else:
            # full Γ in k×k space
            Zk = Vk.T @ S_Z @ Vk                # [k, k]
            gamma_vec = None

        # Drivers/listeners projection
        Bu_full = self._build_Bu_full(drivers)           # [3N, 3*Nd]
        C_full  = self._build_C_full(listener_ids, axis=axis)  # [C, 3N]
        Bu      = Bu_full.index_select(0, idx_free)      # [df, 3*Nd]
        C       = C_full.index_select(1, idx_free)       # [C, df]

        Gu = U.T @ Bu                                    # [k, 3*Nd]
        Gy = C  @ U                                      # [C, k]

        # 3) Two-step coefficients
        alpha = self.inv_mass                            # scalar
        c     = self.inv_mass * self.fric                # scalar

        if diag_gamma:
            # elementwise recurrence: q_{n+1} = a*q_n + b*q_{n-1} + (Uu @ u_n)
            a = (2.0 - c) - alpha * (w2 + gamma_vec)     # [k]
            b = -(1.0 - c) + alpha *  gamma_vec          # [k]
            Uu = alpha * Gu                              # [k, 3*Nd]
        else:
            # coupled recurrence with dense A,B
            I = torch.eye(k, device=device, dtype=dtype)
            Λ = torch.diag(w2)                           # [k, k]
            A = (2.0 - c) * I - alpha * (Λ + Zk)         # [k, k]
            B = -(1.0 - c) * I + alpha * Zk              # [k, k]
            Uu = alpha * Gu                              # [k, 3*Nd]

        # 4) Inputs (you can switch to on-the-fly later to save memory)
        T  = int(round(seconds * fs))
        Nd = drivers.numel()
        u  = self._rasterize_events_to_u(T=T, Nd=Nd, events=events, device=device, dtype=dtype,
                                        hold=1, shape="cos", driver_axis="z")  # [T, 3*Nd]

        # 5) State & render
        q_prev = torch.zeros(k, device=device, dtype=dtype)
        q      = torch.zeros(k, device=device, dtype=dtype)
        y      = torch.empty((T, Gy.shape[0]), device=device, dtype=dtype)  # [T, C]

        if diag_gamma:
            # O(k) per step
            for t in range(T):
                r_t   = Uu @ u[t]                    # [k]
                q_next = a * q + b * q_prev + r_t    # [k]
                y[t]   = Gy @ q_next                 # [C]
                q_prev, q = q, q_next
        else:
            # O(k^2) per step
            for t in range(T):
                r_t   = Uu @ u[t]                    # [k]
                q_next = A @ q + B @ q_prev + r_t    # [k]
                y[t]   = Gy @ q_next                 # [C]
                q_prev, q = q, q_next

        # 6) Optional HPF
        if hp:
            y = self.hp_filter(y)                    # [T, C]

        # 7) Mixdown & normalize
        audio = y
        if mix_audio:
            audio = mix_down(
                self, audio,
                layout=layout, method=pan_method,
                listener_ids=listener_ids,
                normalize=False, soft_clip=False, energy_comp=True,
            )
        audio = audio.transpose(1, 0)  # [K, T]
        peak  = torch.maximum(torch.abs(audio).amax(dim=1), torch.tensor(1e-9, device=audio.device)).detach()
        audio = audio / peak.unsqueeze(-1)
        return audio.T  # [T, K]




if __name__ == "__main__":
    fs = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MassSpringModel.from_json(
        "../model_configs/sonobox_data/baselines/biosonix_3D.json",
        device=device, dt=1/fs
    )
    model.train()  # enable grads
    # model.run_interactive()

    # visualize = False

    # if visualize:
    #     plot_model_graph_3d(model)
    #     # render_traj_taichi3d(traj.detach().cpu().numpy(), model.edge_index.cpu().numpy())

    # # Render 1s of audio and backprop a simple power loss
    seconds = 1.0
    events = load_event_from_json("events/two_hits.json")
    events = event_dict_seconds_to_samples(events, fs)
    
    
    # audio = model.render_audio(
    #     seconds=seconds,
    #     fs=fs,
    #     observable='pos',
    #     axis='all',
    #     listener_ids=model.get_listener_ids(),
    #     layout='mono',
    #     pan_method='by_position',
    #     hp=True,
    #     events=events,
    #     exciter=None,
    #     mix_audio=True
    # )  # [T_audio, 1]

    audio = model.render_modal_audio(
        seconds=1.0,
        fs=fs,
        listener_ids=model.get_listener_ids(),
        drivers=model.get_driver_ids(),
        axis='avg',          # 'x'|'y'|'z'|'avg' (linear; 'avg' ≈ your 'all')
        n_modes=1024,         # keep the most audible modes
        hp=True,
        events=events,       # same events dict you already use
        mix_audio=True,
        gamma="diag",
        layout='mono',
        pan_method='by_position',
    )

    audio = audio.squeeze()
    
    import soundfile as sf, sounddevice as sd
    sf.write("mass_spring.wav", audio.detach().cpu().numpy(), fs)
    sd.play(audio.detach().cpu().numpy(), fs); sd.wait()

    loss = torch.mean(audio**2)
    print("Audio loss:", loss.item())
    loss.backward()
