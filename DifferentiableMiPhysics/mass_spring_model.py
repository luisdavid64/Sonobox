import torch
from torch import nn
from typing import Optional
from topology_utils import build_grid_nodes, build_edges_by_type, dedupe_undirected
from viz_utils import plot_model_graph_3d, render_traj_taichi3d
from audio_helpers import _axis_pick_t, _dc_block_t, _stereo_mixer_t  

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
                 dt: float = 1/44100):
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

        # ----- per-mass parameters -----
        # mass, radius, fixed flags come from nodes
        mass_from_nodes = nodes[:, 3].clamp_min(1e-12)
        self.mass = nn.Parameter(mass_from_nodes.clone())
        inv_mass_init = 1 / mass_from_nodes  # per MIPhysics
        self.inv_mass = nn.Parameter(inv_mass_init.clone())
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
        with torch.no_grad():
            i, j = self.edge_index[0], self.edge_index[1]
            d0 = self.rest_pos[j] - self.rest_pos[i]
            L0 = (d0.pow(2).sum(-1) + 1e-12).sqrt()
            if isinstance(self.rest, torch.nn.Parameter):
                self.rest.data.copy_(L0)
            else:
                self.rest = nn.Parameter(L0)
            self.m_prevDist = L0.clone()
        # miPhysics-style state buffers
        self.register_buffer("m_pos", self.rest_pos.clone())   # current position
        self.register_buffer("m_posR", self.rest_pos.clone())  # delayed position (previous)
        self.register_buffer("m_frc", torch.zeros(self.N, self.dim, device=nodes.device, dtype=nodes.dtype), persistent=False)

        # Optional velocity-control (per-mass), like Mass.triggerVelocityControl in miPhysics
        self.register_buffer("ctrl_on", torch.zeros(self.N, 1, device=nodes.device, dtype=nodes.dtype))
        self.register_buffer("ctrl_vel", torch.zeros(self.N, self.dim, device=nodes.device, dtype=nodes.dtype))

        # Driver/listener index caches
        self._driver_ids = self.get_driver_ids()

        # Global
        self.fric = nn.Parameter(torch.tensor(0.25))
        self.register_buffer("gravity", torch.zeros(3, device=self.nodes.device, dtype=self.nodes.dtype))
        #self.use_dt_scaling = False


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
    def spring_damper_forces(self):
        """
        miPhysics exact link force:
        lnkFrc = -K*(L - rest) - Z*(L - m_prevDist)
        dir is the *current* unit direction.
        Updates self.m_prevDist <- L (like m_prevDist = m_dist).
        """
        i, j = self.edge_index[0], self.edge_index[1]          # [E]
        d    = self.m_pos[j] - self.m_pos[i]                   # [E,3]
        m_dist    = (d.pow(2).sum(-1) + 1e-12).sqrt()               # [E]
        invL = torch.where(m_dist > 1e-9, 1.0 / m_dist, torch.zeros_like(m_dist))

        dir  = d * invL.unsqueeze(-1)                          # [E,3]

        # scalar link force (Hooke + dashpot on distance change)
        f_el   = - self.k      * (m_dist - self.rest)               # [E]
        f_damp = - self.edge_z * (m_dist - self.m_prevDist)             # [E]
        lnkFrc = f_el + f_damp                                 # [E]

        f_vec = lnkFrc.unsqueeze(-1) * dir                     # [E,3]

        # scatter to nodes
        F = torch.zeros_like(self.m_pos)                       # [N,3]
        F.index_add_(0, i, -f_vec)
        F.index_add_(0, j, f_vec)

        # update previous distance for next tick (like m_prevDist = m_dist)
        # detach: we don't want to backprop through the rolling state
        self.m_prevDist = m_dist.detach()

        return F

    def compute(self):
        # --- 1) integrate with previous forces ---
        invM  = 1 / self.mass.view(-1, 1)     # [N,1]
        fr    = getattr(self, 'fric', None)
        if fr is None:
            fr = torch.zeros_like(self.mass)
        c     = (invM * fr.view(-1,1)).clamp(0.0, 1.9)          # [N,1]
        gterm = self.gravity.view(1,3)     # [1,3]
        F     = self.m_frc                                       # [N,3]

        x, xr = self.m_pos, self.m_posR
        x_prev = x.clone()

        acc_term = F * invM
        x        = x * (2.0 - c) - xr * (1.0 - c) + acc_term - gterm

        self.m_posR = x_prev
        self.m_pos  = x

        # enforce fixed nodes → keep at rest
        fixed = self.fixed_mask; 
        rest = self.rest_pos
        self.m_pos  = (1.0 - fixed) * self.m_pos  + fixed * rest
        self.m_posR = (1.0 - fixed) * self.m_posR + fixed * rest

        # --- 2) rebuild forces for next step (springs + drivers) ---
        self.m_frc.zero_()
        Fspr = self.spring_damper_forces()
        # store Fspr as a csv
        # import pandas as pd
        # pd.DataFrame(Fspr.detach().cpu().numpy()).to_csv("Fspr_forces.csv", index=False)
        # exit()
        
        self.m_frc += Fspr * (1.0 - fixed)  # no forces on fixed nodes
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
                   interactionType=interactionType, bounds=bounds, dt=dt)
                
    @classmethod
    def from_json(cls, path: str, device: Optional[torch.device] = None, dt: float = 1/16000):
        import json
        with open(path, "r") as f:
            config = json.load(f)
        return cls.from_config(config, device=device, dt=dt)

    def get_driver_ids(self):
        if self.drivers is None:
            return None
        # Can we turn into a function
        ids = self.get_ids(self.drivers)
        return torch.tensor(ids, device=self.nodes.device, dtype=torch.long)

    def get_ids(self, tensor):
        return [(i * self.dimY + j) * self.dimZ + k for (i, j, k) in tensor.tolist()]

    def get_fixed_ids(self):
        return torch.nonzero(self.fixed_mask.view(-1) > 0.5, as_tuple=False).view(-1)

    def visualize(self):
        plot_model_graph_3d(self)

    #---------------- audio ----------------

    @torch.no_grad()
    def simulate_listeners(self,
                           steps: int,
                           listener_ids: torch.Tensor,
                           observable: str = "pos",   # 'pos' | 'vel' | 'acc' | 'force'
                           axis: str = "all") -> torch.Tensor:
        """
        Run `steps` physics ticks and return raw multichannel listener signal at sim rate.
        Output shape: [steps, C] (C = #listeners). Uses current self.dt.
        """
        device = self.nodes.device
        ids = listener_ids.to(device, dtype=torch.long)
        C = ids.numel()
        out = torch.zeros(steps, C, device=device, dtype=self.m_pos.dtype)

        # caches
        prev_abs = self.m_pos.clone()  # for vel if needed

        for t in range(steps):
            # one physics step
            self.compute()

            # absolute pos
            pos_abs = self.m_pos
            if observable == "pos":
                val = pos_abs[ids]                                   # [C,3]
                out[t] = _axis_pick_t(val, axis)                     # [C]
            elif observable == "vel" or observable == "acc":
                vel = (pos_abs - prev_abs) / self.dt                 # [N,3]
                if observable == "vel":
                    val = vel[ids]                                   # [C,3]
                    out[t] = _axis_pick_t(val, axis)
                else:
                    # one more diff for acc: a ≈ Δv / dt
                    # use previous vel from last step (store it)
                    if t == 0:
                        acc = torch.zeros_like(vel)
                    else:
                        acc = (vel - prev_vel) / self.dt
                    val = acc[ids]
                    out[t] = _axis_pick_t(val, axis)
                prev_vel = vel
            elif observable == "force":
                # recompute spring forces at *current* state
                Fspr = self.spring_damper_forces()   # [N,3]
                val = Fspr[ids]
                out[t] = _axis_pick_t(val, axis)
            else:
                raise ValueError("observable must be 'pos' | 'vel' | 'acc' | 'force'.")

            prev_abs = pos_abs

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
        # i such that t_out ≈ t_sim[i]…t_sim[i+1]
        idx_float = t_out / dt
        i0 = torch.clamp(idx_float.floor().long(), 0, T_sim - 2)   # [T_audio]
        w = (idx_float - i0.float()).unsqueeze(-1)                 # [T_audio,1]

        y0 = sig_sim[i0, :]                                        # [T_audio, C]
        y1 = sig_sim[i0 + 1, :]
        y  = (1.0 - w) * y0 + w * y1                               # [T_audio, C]
        return y

    def mix_down(self,
                 multich: torch.Tensor,     # [T, C]
                 layout: str = "stereo",    # 'mono' | 'stereo'
                 method: str = "by_position",
                 listener_ids: torch.Tensor | None = None,
                 plane: tuple[int,int] = (0,2),
                 target_peak: float = 0.99,
                 soft_clip: bool = True,
                 clip_drive: float = 2.0) -> torch.Tensor:
        """
        Mix C listeners to mono/stereo (Torch). Returns [T, K].
        """
        device = multich.device
        T, C = multich.shape
        if layout == "mono":
            y = multich.mean(dim=1, keepdim=True)                  # [T,1]
        elif layout == "stereo":
            if listener_ids is None:
                raise ValueError("listener_ids required for stereo mixing when method='by_position'")
            # get static listener positions (rest_pos are fine)
            lp = self.rest_pos[listener_ids.to(device, dtype=torch.long)]  # [C,3]
            G = _stereo_mixer_t(C, method=method, listener_pos=lp, plane=plane).to(device)  # [C,2]
            y = multich @ G                                        # [T,2]
        else:
            raise ValueError("layout must be 'mono' or 'stereo'.")

        # peak normalize
        peak = torch.max(torch.abs(y)).clamp_min(1e-9)
        y = y * (target_peak / peak)

        # optional soft clip
        if soft_clip:
            y = torch.tanh(y * clip_drive) / torch.tanh(torch.tensor(clip_drive, device=device))

        return y  # [T, K]

    @torch.no_grad()
    def render_audio_offline(self,
                             seconds: float,
                             fs: int = 16000,
                             observable: str = "pos",
                             axis: str = "all",
                             listener_ids: torch.Tensor | None = None,
                             layout: str = "stereo",
                             pan_method: str = "by_position",
                             hp: bool = True,
                             gain: float = 1.0) -> torch.Tensor:
        """
        End-to-end: simulate at current dt for `seconds`, capture listeners,
        resample to `fs`, downmix, return audio [T_audio, K].
        """
        device = self.nodes.device
        if listener_ids is None:
            ids = self.get_driver_ids()
            if ids is None or ids.numel() == 0:
                ids = torch.tensor([self.N // 2], device=device, dtype=torch.long)
            listener_ids = ids

        # how many sim steps?
        steps = int(round(seconds / float(self.dt)))
        raw = self.simulate_listeners(steps, listener_ids, observable=observable, axis=axis)  # [T_sim,C]

        # (optional) DC-block at sim rate before resampling (helps big drifts for positions)
        if hp:
            raw = _dc_block_t(raw)

        # resample to audio
        audio_mc = self.resample_to_audio(raw, fs=fs)  # [T_audio, C]

        # mix to target layout
        audio = self.mix_down(audio_mc, layout=layout, method=pan_method, listener_ids=listener_ids)

        # final gain & clamp
        audio = torch.clamp(audio * gain, -1.0, 1.0)

        return audio  # [T_audio, K]

if __name__ == "__main__":
    fs = 16000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MassSpringModel.from_json("../model_configs/sonobox_data/baselines/biosonix_3D.json", device=device, dt=1/fs)
    model.eval()


    traj = []
    T = int(16000)  # 1 second at 16kHz
    x = None
    v = None
    example_driver_id = model.get_driver_ids()[0].item()
    print("Example driver id:", example_driver_id)
    for t in range(T):
        if t == 0 or t == 4000 or t == 8000:
            model.apply_force_on_drivers((3,3,3))
        x = model.compute()
        #if t == 0 or t == 8000 or t == 4000:
        print(f"Step {t}: driver pos {x[example_driver_id].detach().cpu().numpy()}")
        traj.append(x)
    traj = torch.stack(traj)  # [T,N,dim]

    plot_model_graph_3d(model)
    render_traj_taichi3d(traj.detach().cpu().numpy(), model.edge_index.cpu().numpy())

    audio = model.render_audio_offline(
        seconds=T * model.dt,   # align with what you simulated
        fs=fs,
        observable='pos', 
        axis='all',
        listener_ids=model.get_driver_ids(),
        layout='mono',
        pan_method='by_position',
        hp=True,
        gain=0.1
    )  # [T_audio, 2]

    # Save or play (example with soundfile/sounddevice)
    # pip install soundfile sounddevice
    import soundfile as sf, sounddevice as sd
    sf.write("mass_spring.wav", audio.cpu().numpy(), 44100)
    sd.play(audio.cpu().numpy(), 44100); sd.wait()
