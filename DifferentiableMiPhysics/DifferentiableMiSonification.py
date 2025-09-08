"""
Differentiable miSonification
=============================

From-scratch, differentiable mass–spring sonification engine in PyTorch.

Input graph is defined explicitly by two feature tensors:
  • nodes:  [N, 8]
      [x, y, z, mass, radius, is_fixed, is_driver, is_listener]
  • springs: [E, 6]
      [stiffness, damping, rest_length, dx, dy, dz]
  • edge_index: [2, E] long tensor (i -> j) per spring (undirected: provide both directions or we mirror inside)

Outputs:
  • audio:  [L, T]  (L = number of listener nodes, one channel per listener)
  • (xT, vT): final displacements and velocities [N,3]

Differentiable w.r.t. node and spring parameters, strike amplitudes, and (optionally) listener weights.

Implements a stable semi-implicit (symplectic) Euler integrator with viscous damping along springs and optional global damping.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

from mi_model import MIModel
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------- utilities -----------------------

def softplus_pos(x: torch.Tensor, floor: float = 1e-8) -> torch.Tensor:
    """Strictly positive parameterization; pass-through if already positive.
    Works for Tensors that require grad.
    """
    return F.softplus(x) + floor


def as_batch(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Ensure a leading batch dim: [N,...] -> [1,N,...]. Returns (xb, B)."""
    if x.dim() == 1:
        x = x.unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(0), 1
    return x, x.shape[0]


# ----------------------- core module -----------------------

@dataclass
class StrikeSpec:
    node_ids: torch.Tensor   # [S] or [B,S] long
    times_s: torch.Tensor    # [S] or [B,S] float seconds
    amps: torch.Tensor       # [S] or [B,S] float (Newtons)
    dur_ms: float = 8.0      # Hann window duration
    direction: Optional[torch.Tensor] = None  # [3] unit vector; default: +Z

def build_strike_forces(
    nodes: torch.Tensor,
    strikes: StrikeSpec,
    fs: float,
    T: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Build external force field F_ext from StrikeSpec.
    nodes: [N,8]
    returns F_ext: [1, N, 3, T]
    Supports scalar or per-strike times/amps. If a single time/amp is given, it is
    broadcast to all strikes to avoid index errors like "index 1 is out of bounds".
    """
    if device is None:
        device = nodes.device
    N = nodes.shape[0]
    is_driver = nodes[:, 6] > 0.5

    # Normalize to batched [1,S]
    def _to_1s(t: torch.Tensor) -> torch.Tensor:
        return t.view(1, -1) if t.dim() == 1 else t

    node_ids = _to_1s(strikes.node_ids.to(torch.long).to(device))
    B, S = node_ids.shape

    # times and amplitudes: allow scalar, [S], or [B,S]
    def _prep_param(p: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32, device=device)
        p = p.to(torch.float32).to(device)
        if p.numel() == 1:
            return p.view(1, 1).expand(B, S)
        p = _to_1s(p)
        if p.shape[1] == 1 and S > 1:
            return p.expand(B, S)
        if p.shape == (B, S):
            return p
        raise ValueError(f"{name} must be scalar, length S, or shape [B,S]; got {tuple(p.shape)}")

    times_s = _prep_param(strikes.times_s, 'times_s')
    amps = _prep_param(strikes.amps, 'amps')

    L = max(1, int(round(float(strikes.dur_ms) * 1e-3 * fs)))
    win = torch.hann_window(L, device=device)

    direction = strikes.direction
    if direction is None:
        direction = torch.tensor([0.0, 0.0, 1.0], device=device)
    direction = direction.to(device=device, dtype=torch.float32)
    direction = direction / (direction.norm() + 1e-12)

    F_ext = torch.zeros(B, N, 3, T, device=device)
    for b in range(B):
        for s in range(S):
            i = int(node_ids[b, s].item())
            if i < 0 or i >= N:
                continue
            # allow non-driver strikes, but typically you want nodes[:,6]==1
            t0 = int(round(times_s[b, s].item() * fs))
            # Apply strike window and amplitude
            t1 = min(t0 + L, T)
            amp = amps[b, s].item()
            win_len = t1 - t0
            if win_len > 0:
                win_slice = win[:win_len] * amp
                F_ext[b, i, :, t0:t1] += direction.view(3, 1) * win_slice.view(1, win_len)
    return F_ext


class DifferentiableMiSonification(nn.Module):
    """Differentiable mass–spring sonification (nearest-neighbor friendly).

    Pass in graph tensors (nodes, edge_index, springs) and render audio from listener nodes.
    Supports batch=1 for simplicity; extend to B>1 by stacking graphs if needed.
    """
    def __init__(self,
                 nodes: torch.Tensor,          # [N,8]
                 edge_index: torch.Tensor,     # [2,E]
                 springs: torch.Tensor,        # [E,6]
                 drivers: torch.Tensor,
                 listeners: torch.Tensor,
                 listener_mode: str = "velocity",  # or "displacement"
                 global_damping: float = 0.0,
                 listener_signal: str = "pos",
                 listener_filter: str = "high_pass",
                 hp_coef: float = 0.95,
                 ramp_inc: float = 1e-4,
                 device: Optional[torch.device] = None):
        super().__init__()
        assert nodes.ndim == 2 and nodes.shape[1] == 8, "nodes must be [N,8]"
        assert edge_index.shape[0] == 2, "edge_index must be [2,E]"
        assert springs.ndim == 2 and springs.shape[1] == 6, "springs must be [E,6]"
        self.register_buffer("nodes", nodes.to(device) if device else nodes)
        self.register_buffer("edge_index", edge_index.to(torch.long).to(self.nodes.device))
        self.register_buffer("springs", springs.to(self.nodes.device))
        self.listener_mode = listener_mode
        self.global_damping = torch.tensor([global_damping], device=self.nodes.device)

        # Observer3D-compatible listener settings
        self.listener_signal = listener_signal.lower()  # 'pos' | 'force' | 'vel'
        self.listener_filter = listener_filter.lower()  # 'none' | 'high_pass'
        self.hp_coef = float(hp_coef)
        self.ramp_inc = float(ramp_inc)

        # Listener selection matrix W: [L,N]
        listener_mask = (self.nodes[:, 7] > 0.5)
        self.listener_ids = torch.nonzero(listener_mask, as_tuple=False).flatten()
        L = int(listener_mask.sum().item())
        if L == 0:
            # default: listen to all nodes equally
            L = 1
            W = torch.full((L, self.nodes.shape[0]), 1.0 / self.nodes.shape[0], device=self.nodes.device)
        else:
            W = torch.zeros(L, self.nodes.shape[0], device=self.nodes.device)
            for li, nid in enumerate(self.listener_ids.tolist()):
                W[li, nid] = 1.0
        self.register_buffer("W_listen", W)  # [L,N]

        # Cached masks
        self.register_buffer("fixed_mask", (self.nodes[:, 5] > 0.5).float().unsqueeze(-1))  # [N,1]

    # -------------- main simulation --------------
    def forward(self,
                T: int,
                fs: float,
                strikes: Optional[StrikeSpec] = None,
                F_ext: Optional[torch.Tensor] = None,  # [1,N,3,T]
                x0: Optional[torch.Tensor] = None,      # [N,3] initial displacement (relative to rest)
                v0: Optional[torch.Tensor] = None       # [N,3] initial velocity
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Simulate and render audio.

        Returns (audio[L,T], (xT[N,3], vT[N,3])).
        """
        device = self.nodes.device
        N = self.nodes.shape[0]
        dt = 1.0 / fs

        # Node params (keep differentiable)
        pos0 = self.nodes[:, 0:3]                  # rest positions [N,3]
        mass = softplus_pos(self.nodes[:, 3])      # [N]
        radius = self.nodes[:, 4]                  # unused in physics core but kept for features
        fixed = self.fixed_mask                    # [N,1]

        # Edge params (differentiable)
        xi = self.edge_index[0]
        xj = self.edge_index[1]
        k = softplus_pos(self.springs[:, 0])       # stiffness [E]
        c_edge = softplus_pos(self.springs[:, 1])  # viscous edge damping [E]
        l0 = softplus_pos(self.springs[:, 2])      # rest length [E]

        # External forces
        if F_ext is None:
            if strikes is None:
                F_ext = torch.zeros(1, N, 3, T, device=device)
            else:
                F_ext = build_strike_forces(self.nodes, strikes, fs, T, device)
        else:
            assert F_ext.shape == (1, N, 3, T)

        # States (displacements/velocities relative to rest)
        x = torch.zeros(N, 3, device=device) if x0 is None else x0.to(device)
        v = torch.zeros(N, 3, device=device) if v0 is None else v0.to(device)

        # Listener setup
        L = self.W_listen.shape[0]
        y = torch.zeros(L, T, device=device)

        # High-pass and ramp state (Observer3D-like)
        prev_pos = torch.zeros(N, 3, device=device)
        prev_frc = torch.zeros(N, 3, device=device)
        y_hp_pos = torch.zeros(N, 3, device=device)
        y_hp_frc = torch.zeros(N, 3, device=device)
        ramp = torch.tensor(0.0, device=device)
        coef = torch.tensor(self.hp_coef, device=device)

        # Pre-broadcast
        Minv = 1.0 / mass.clamp_min(1e-6)  # [N]
        Minv = Minv.view(N, 1)
        cglob = softplus_pos(self.global_damping)[0]

        eps = 1e-9
        for n in range(T):
            # Current absolute positions
            X = pos0 + x  # [N,3]

            # Edge relative vectors and unit directions
            Xi = X.index_select(0, xi)  # [E,3]
            Xj = X.index_select(0, xj)  # [E,3]
            d = Xj - Xi                 # [E,3]
            ell = torch.sqrt((d * d).sum(dim=-1, keepdim=True) + eps)  # [E,1]
            u = d / (ell + eps)         # [E,3]

            # Relative velocities along springs
            Vi = v.index_select(0, xi)
            Vj = v.index_select(0, xj)
            rel_v = Vj - Vi             # [E,3]
            rel_speed = (rel_v * u).sum(dim=-1, keepdim=True)  # [E,1]

            # Spring + edge damping forces (magnitudes)
            k_ = k.view(-1, 1)
            c_ = c_edge.view(-1, 1)
            l0_ = l0.view(-1, 1)
            Fspring_mag = k_ * (ell - l0_)         # [E,1]
            Fdamp_mag = c_ * rel_speed             # [E,1]
            Fedge = (Fspring_mag + Fdamp_mag) * u  # [E,3]

            # Accumulate nodal forces
            Fspr = torch.zeros(N, 3, device=device)
            Fspr.index_add_(0, xi,  Fedge)
            Fspr.index_add_(0, xj, -Fedge)

            # Global damping (viscous)
            Fglob = - cglob * v

            # Total forces (add external)
            Ftot = Fspr + Fglob + F_ext[0, :, :, n]

            # Semi-implicit Euler step
            v = v + Ftot * Minv * dt
            # Zero out fixed-node velocities (hard constraint)
            v = v * (1.0 - fixed)
            x = x + v * dt
            # Re-enforce fixed nodes displacement = 0
            x = x * (1.0 - fixed)

            # ---- Observer3D-compatible output per node/axis ----
            # Select raw observable per node-axis
            if self.listener_signal == "pos":
                obs_vec = x                   # [N,3] displacement
            elif self.listener_signal == "force":
                obs_vec = Ftot                # [N,3] total force
            elif self.listener_signal == "vel":
                obs_vec = v                   # [N,3] velocity
            else:
                raise ValueError("listener_signal must be 'pos', 'force', or 'vel'")

            # High-pass filter as in Observer3D: y = (x - x_prev + coef*y_prev) * ramp
            if self.listener_filter == "high_pass":
                if self.listener_signal == "pos":
                    y_hp_pos = (obs_vec - prev_pos) + coef * y_hp_pos
                    out_vec = y_hp_pos * ramp.clamp(max=1.0)
                    prev_pos = obs_vec
                elif self.listener_signal == "force":
                    y_hp_frc = (obs_vec - prev_frc) + coef * y_hp_frc
                    out_vec = y_hp_frc * ramp.clamp(max=1.0)
                    prev_frc = obs_vec
                else:  # vel + high_pass (optional)
                    y_hp_pos = (obs_vec - prev_pos) + coef * y_hp_pos
                    out_vec = y_hp_pos * ramp.clamp(max=1.0)
                    prev_pos = obs_vec
                ramp = (ramp + self.ramp_inc)
            else:
                out_vec = obs_vec * ramp.clamp(max=1.0)
                ramp = (ramp + self.ramp_inc)

            # Mix to channels: sum xyz per node, then W_listen @ node_scalar
            node_scalar = out_vec.sum(dim=-1)             # [N]
            y[:, n] = torch.matmul(self.W_listen, node_scalar)

        return y, (x, v)

# ----------------------- example usage -----------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Grid 3x3x3, nearest-neighbor springs
    model = MIModel.from_json("../model_configs/sonobox_data/baselines/biosonix_3D.json", device=device)
    sim = DifferentiableMiSonification(
        model.nodes, 
        model.edge_index, 
        model.springs,
        model.drivers,
        model.listeners,
        listener_mode='velocity',
        global_damping=1e-2,
        device=device
    )

    # Make a strike
    fs = 48000.0
    T = int(2 * fs)
    strikes = StrikeSpec(
        node_ids=model.get_driver_ids(),  # matches (1,1,0)
        times_s=torch.tensor([0.0]),
        amps=torch.tensor([5.0]),
        dur_ms=0.0,
        direction=torch.tensor([1.0,1.0,1.0])
    )
    print("num listeners:", int(sim.W_listen.sum().item()))
    print("listener ids:", sim.listener_ids.tolist())
    print("num drivers:", strikes.node_ids.numel())
    print("driver ids:", strikes.node_ids.flatten().tolist())
    t0 = strikes.times_s.flatten()[0].item()

    print("first strike time (s):", t0, "sample:", int(round(t0*fs)), "T:", T)


    audio, (xT, vT) = sim(T=T, fs=fs, strikes=strikes)
    peak = audio.abs().max().item()
    rms  = audio.pow(2).mean().sqrt().item()
    print("peak:", peak, "rms:", rms)
    # Save audio
    import soundfile as sf
    sf.write('test_sonification.wav', audio.cpu().numpy().T, int(fs))
    print('audio shape:', tuple(audio.shape))  # (L,T)
    print('final x shape:', tuple(xT.shape))   # (N,3)
