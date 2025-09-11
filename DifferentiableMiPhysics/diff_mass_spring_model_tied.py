import math
import os
import torch
from torch import nn
from typing import Optional
from topology_utils import build_grid_nodes, build_edges_by_type, dedupe_undirected
from viz_utils import plot_model_graph_3d, render_traj_taichi3d, plot_spectrogram
from audio_helpers import _axis_pick_t, _dc_block_t, _stereo_mixer_t, _stereo_mixer_tt
import json

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
        inv_mass_init = 1.0 / mass_from_nodes  # per miPhysics
        self.inv_mass = nn.Parameter(inv_mass_init.clone(), requires_grad=False)
        
        self.radius    = nn.Parameter(nodes[:, 4].clone(), requires_grad=False)

        # fixed nodes: 5th col of nodes is fixed flag (0/1)
        self.register_buffer("fixed_mask", nodes[:, 5].view(-1, 1).clone())

        # ----- per-edge parameters (from springs builder) -----
        # springs columns: [stiffness, edge_damping, rest_len, di, dj, dk]
        # Parameterize all k and z with a single value each
        self.k = nn.Parameter(torch.tensor(float(springs[:, 0].mean()), device=nodes.device))  # scalar
        self.z = nn.Parameter(torch.tensor(float(springs[:, 1].mean()), device=nodes.device))  # scalar
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
        self.fric = nn.Parameter(torch.tensor(friction, device=self.nodes.device, dtype=self.nodes.dtype))
        # We could use gravity if desired
        self.register_buffer("gravity", torch.zeros(3, device=self.nodes.device, dtype=self.nodes.dtype))

        # listener HPF state
        self.m_coef = 0.95
        self.register_buffer("hp_x_prev", torch.zeros(0))
        self.register_buffer("hp_y_prev", torch.zeros(0))
        self.hp_primed = False  # if False, we will prime on first call


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

    # ---------------- interactions (springs) ----------------
    def spring_damper_forces(self):
        """
        miPhysics exact link force:
        lnkFrc = -K*(L - rest) - Z*(L - m_prevDist)
        dir is the *current* unit direction.
        Updates self.m_prevDist <- L (kept with full graph for BPTT).
        """
        k = torch.exp(self.k)
        z = torch.exp(self.z)
        i, j = self.edge_index[0], self.edge_index[1]          # [E]
        d    = self.m_pos[j] - self.m_pos[i]                   # [E,3]
        m_dist = (d.pow(2).sum(-1) + 1e-12).sqrt()             # [E]
        invL   = torch.where(m_dist > 1e-9, 1.0 / m_dist, torch.zeros_like(m_dist))
        dirv   = d * invL.unsqueeze(-1)                        # [E,3]

        # scalar link force (Hooke + dashpot on distance change)
        f_el   = - k * (m_dist - self.rest)          # [E]
        f_damp = - z * (m_dist - self.m_prevDist)    # [E]
        lnkFrc = f_el + f_damp                                 # [E]

        f_vec = lnkFrc.unsqueeze(-1) * dirv                    # [E,3]

        # scatter to nodes (functional index_add to avoid in-place)
        F = torch.zeros_like(self.m_pos)                       # [N,3]
        F = torch.index_add(F, 0, i, -f_vec)
        F = torch.index_add(F, 0, j,  f_vec)

        # update previous distance for next tick WITH graph (no detach)
        self.m_prevDist = m_dist.clone()
        return F

    def compute(self):
        # --- 1) integrate with previous forces ---
        invM  = self.inv_mass.view(-1, 1)                      # [N,1]

        fric = 2.0*torch.sigmoid(self.fric) 
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

    # Convert 
    def to_json(self, path: str):
        # Convert fields to JSON-serializable types, as done in from_config
        def tensor_to_list(tensor):
            return tensor.detach().cpu().tolist() if tensor is not None else []
        def float_or_list(value):
            if isinstance(value, (list, tuple)):
                return [float(v) for v in value]
            return float(value)
        def int_or_list(value):
            if isinstance(value, (list, tuple)):
                return [int(v) for v in value]
            return int(value)   
        def parse_mass_name(name):
            p = name.split("_")
            if len(p) != 4 or p[0] != "m":
                raise ValueError(f"Invalid mass name: {name}")
            return tuple(int(x) for x in p[1:])
        def mass_name(i, j, k):
            return f"m_{i}_{j}_{k}"
        geom = {
            "dx": int_or_list(self.dimX),
            "dy": int_or_list(self.dimY),
            "dz": int_or_list(self.dimZ),
            "distance": float_or_list(self.dist),
            "massesRadius": float_or_list(self.radius[0].item()),
            "interactionType": self.interactionType,
        }
        params = {
            "M": float_or_list(torch.mean(1/self.inv_mass).item()),
            "K": float_or_list(torch.mean(self.k).item()),
            "C": float_or_list(torch.mean(self.z).item()),
        }
        sonification_set_up = {
            "drivers": [mass_name(*idx) for idx in tensor_to_list(self.drivers)],
            "listeners": [mass_name(*idx) for idx in tensor_to_list(self.listeners)],
        }
        config = {
            "geometry": geom,
            "parameters": params,
            "sonification_set_up": sonification_set_up,
            "model": "3D",
            "global_friction": float(self.fric.item()),
            "bounds": self.bounds,
            
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

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
                           events: dict = {}
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
            if t in events:
                self.apply_force_on_drivers(events[t])
            self.compute()

            # absolute pos
            pos_abs = self.m_pos
            if observable == "pos":
                val = pos_abs[ids]                                   # [C,3]
                out[t] = _axis_pick_t(val, axis)                     # [C]
            elif observable == "force":
                # recompute spring forces at *current* state
                Fspr = self.spring_damper_forces()   # [N,3]
                val = Fspr[ids]
                out[t] = _axis_pick_t(val, axis)
            else:
                raise ValueError("observable must be 'pos' | 'force'.")

        return out  # [steps, C]

    def resample_to_audio(self,
                          sig_sim: torch.Tensor,   # [T_sim, C] at dt_sim=self.dt
                          fs: int = 44100) -> torch.Tensor:
        """
        Linear resample from sim timebase (dt=self.dt) to audio (fs).
        Returns [T_audio, C]. Pure Torch (differentiable).
        """
        device = sig_sim.device
        T_sim, C = sig_sim.shape
        dt = float(self.dt)
        dur = dt * (T_sim - 1)
        T_audio = int(round(dur * fs)) + 1

        # time grids
        t_sim = torch.linspace(0.0, dur, T_sim, device=device)
        t_out = torch.linspace(0.0, dur, T_audio, device=device)

        # compute fractional indices into sim grid
        idx_float = t_out / dt
        i0 = torch.clamp(idx_float.floor().long(), 0, T_sim - 2)   # [T_audio]
        w = (idx_float - i0.float()).unsqueeze(-1)                 # [T_audio,1]

        y0 = sig_sim[i0, :]                                        # [T_audio, C]
        y1 = sig_sim[i0 + 1, :]
        y  = (1.0 - w) * y0 + w * y1                               # [T_audio, C]
        return y

    def mix_down(
        self,
        multich: torch.Tensor,            # [T, C]
        layout: str = "stereo",           # 'mono' | 'stereo'
        method: str = "by_position",
        listener_ids: torch.Tensor | None = None,
        plane: tuple[int,int] = (0,2),
        normalize: bool = False,          # keep False during training
        target_peak: float = 0.99,
        soft_clip: bool = False,          # keep False during training
        clip_drive: float = 2.0,
        energy_comp: bool = True,         # divide by sqrt(C) to stabilize loudness
    ) -> torch.Tensor:
        """
        Mix C listeners to mono/stereo (Torch). Returns [T, K].
        Designed to be training-friendly (purely linear by default).
        """
        device, dtype = multich.device, multich.dtype
        T, C = multich.shape

        if layout == "mono":
            y = multich.mean(dim=1, keepdim=True)                  # [T,1]
        elif layout == "stereo":
            if listener_ids is None:
                raise ValueError("listener_ids required for stereo mixing when method='by_position'")
            # cache G if listeners are static
            if not hasattr(self, "_G_cache") or self._G_cache is None \
            or self._G_cache.shape[0] != C:
                lp = self.rest_pos[listener_ids.to(device, dtype=torch.long)]  # [C,3]
                G = _stereo_mixer_tt(C, method=method, listener_pos=lp, plane=plane,
                                    device=device, dtype=dtype)                 # [C,2]
                if energy_comp and C > 0:
                    G = G / math.sqrt(C)                                       # stabilize loudness
                self._G_cache = G                                              # cache on module
            G = self._G_cache.to(device=device, dtype=dtype)
            y = multich @ G                                                    # [T,2]
        else:
            raise ValueError("layout must be 'mono' or 'stereo'.")

        # Optional post-faders (disable for training)
        if normalize:
            peak = torch.maximum(torch.abs(y).amax(dim=0, keepdim=True), torch.tensor(1e-9, device=device, dtype=dtype))
            y = y * (target_peak / peak)
        if soft_clip:
            y = torch.tanh(y * clip_drive) / torch.tanh(torch.as_tensor(clip_drive, device=device, dtype=dtype))

        return y

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
                     events: dict = {}
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

        # how many sim steps?
        steps = int(round(seconds / float(self.dt)))
        raw = self.simulate_listeners(steps, listener_ids, observable=observable, axis=axis, events=events)  # [T_sim,C]

        # (optional) DC-block at sim rate before resampling (helps big drifts for positions)
        if hp:
            # raw = _dc_block_t(raw)
            raw = self.highpass_observer3d(raw, R=0.95, prime_on_first_call=True)


        # resample to audio
        # audio_mc = self.resample_to_audio(raw, fs=fs)  # [T_audio, C]
        audio_mc = raw

        # mix to target layout
        # audio = self.mix_down(audio_mc, layout=layout, method=pan_method, listener_ids=listener_ids)
        audio = self.mix_down(audio_mc,
                            layout=layout,             # mono is cheaper for CLAP; stereo if you need it
                            method=pan_method,
                            listener_ids=listener_ids,
                            normalize=False,
                            soft_clip=False,
                            energy_comp=True) # For stereo, energy_comp helps keep loudness stable

        # final gain & clamp
        audio = audio * gain
        audio = audio.squeeze()
        peak = torch.maximum(torch.abs(audio).amax(), torch.tensor(1e-9, device=audio.device)).detach()
        audio = audio / peak

        return audio  # [T_audio, K]

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
        with torch.no_grad():
            self.inv_mass.data.clamp_min_(1e-8)
            # self.k.data.clamp_min_(1e-8)
            # self.z.data.clamp_min_(0.0)
            # self.fric.data.clamp_min_(0.0)


if __name__ == "__main__":
    fs = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MassSpringModel.from_json(
        "../model_configs/sonobox_data/baselines/biosonix_3D.json",
        device=device, dt=1/fs
    )
    model.train()  # enable grads

    visualize = False

    if visualize:
        plot_model_graph_3d(model)
        # render_traj_taichi3d(traj.detach().cpu().numpy(), model.edge_index.cpu().numpy())

    # Render 1s of audio and backprop a simple power loss
    seconds = 1.0

    events = {
        8000: (3, 3, 3),
        12000: (3, 3, 5),
        # more events...
    }
    audio = model.render_audio(
        seconds=seconds,
        fs=fs,
        observable='pos',
        axis='all',
        listener_ids=model.get_listener_ids(),
        layout='mono',
        pan_method='by_position',
        hp=True,
        events=events
    )  # [T_audio, 1]
    
    import soundfile as sf, sounddevice as sd
    sf.write("mass_spring.wav", audio.detach().cpu().numpy(), 16000)
    sd.play(audio.detach().cpu().numpy(), 16000); sd.wait()

    loss = torch.mean(audio**2)
    print("Audio loss:", loss.item())
    loss.backward()

    # Example: inspect gradients exist
    def mean_abs(x): return float(x.detach().abs().mean().cpu())
    # print("grad|K|   :", model.theta_k.grad)
    # print("grad|Z|:", model.theta_z.grad)
    # print("grad|fric|:", model.theta_fric.grad)
