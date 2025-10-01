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

class ParamMapper(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh()
        )
        self.k_1head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.k_2head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.z_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
        self.fric_head = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())
    def forward(self, edge_features):
        h = self.shared(edge_features)
        k_1 = self.k_1head(h).squeeze(-1)
        k_2 = self.k_2head(h).squeeze(-1)
        z = self.z_head(h).squeeze(-1)
        fric = self.fric_head(h).squeeze(-1)
        return k_1,k_2, z, fric

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
                 dt: float = 1/44100, friction: float = 0.25, gain=10, use_mapping_function=True):
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
        # self.log_inv_mass = nn.Parameter(torch.log(inv_mass_init.clone()), requires_grad=False)
        
        self.radius    = nn.Parameter(nodes[:, 4].clone(), requires_grad=False)

        # fixed nodes: 5th col of nodes is fixed flag (0/1)
        self.register_buffer("fixed_mask", nodes[:, 5].view(-1, 1).clone())

        # ----- per-edge parameters (from springs builder) -----
        # springs columns: [stiffness, edge_damping, rest_len, neighbor_type]
        # If springs has neighbor_type column, use it to parameterize k
        if springs.shape[1] == 4:
            self.neighbor_type = springs[:, 3].long()  # 0=first, 1=second
            self.neighbor_type = nn.Parameter(self.neighbor_type, requires_grad=False)
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

        self.use_mapping_function = use_mapping_function
        if use_mapping_function:
            self.param_mapper = ParamMapper(in_dim=4, hidden=16).to(nodes.device)
            self._u_fric.requires_grad = False
            self._logk_first.requires_grad = False
            self._logk_second.requires_grad = False
            self._logz.requires_grad = False

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
    def spring_damper_forces(self,k, z):
        """
        miPhysics exact spring force:
        f_spring = -K*(L - rest) - Z*(L - m_prevDist)
        dir is the *current* unit direction.
        Updates self.m_prevDist <- L (kept with full graph for BPTT).
        Uses k_first for first neighbors, k_second for second neighbors.
        """
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
        k_1,k_2, z, fric = self.k_1, self.k_2, self.z, self.fric
        if self.use_mapping_function:
            edge_features = torch.stack([k_1,k_2,z,fric], dim=-1)
            k_1,k_2, z, fric = self.param_mapper(edge_features)
        k = torch.where(self.neighbor_type==0, k_1, k_2)
        
        invM  = self.inv_mass.view(-1, 1)                      # [N,1]

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
        Fspr = self.spring_damper_forces(k, z)  # [N,3]
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


if __name__ == "__main__":
    fs = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MassSpringModel.from_json(
        "../model_configs/sonobox_data/baselines/biosonix_3D.json",
        device=device, dt=1/fs
    )
    model.train()  # enable grads
    # model.to("mps")
    # model.run_interactive()

    # visualize = False

    # if visualize:
    #     plot_model_graph_3d(model)
    #     # render_traj_taichi3d(traj.detach().cpu().numpy(), model.edge_index.cpu().numpy())

    # # Render 1s of audio and backprop a simple power loss
    seconds = 1.0
    events = load_event_from_json("events/two_hits.json")
    events = event_dict_seconds_to_samples(events, fs)
    
    
    audio = model.render_audio(
        seconds=seconds,
        fs=fs,
        observable='pos',
        axis='all',
        listener_ids=model.get_listener_ids(),
        layout='mono',
        pan_method='by_position',
        hp=True,
        events=events,
        exciter=None,
        mix_audio=True
    )  # [T_audio, 1]

    audio = audio.squeeze()
    
    import soundfile as sf, sounddevice as sd
    sf.write("mass_spring.wav", audio.detach().cpu().numpy(), fs)
    sd.play(audio.detach().cpu().numpy(), fs); sd.wait()

    loss = torch.mean(audio**2)
    print("Audio loss:", loss.item())
    loss.backward()
