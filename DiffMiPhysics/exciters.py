import torch
import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

class HitExciter(nn.Module):
    """
    One Gaussian hit in *seconds*, dt-invariant energy.
    Returns a single [3] force vector for the current step.
    """
    def __init__(self,
                 t0_s: float = 0.20,     # hit time (s)
                 width_s: float = 0.003,  # std (s)
                 A_imp: float = 1.0,      # impulse area (N*s)
                 direction=(1.,0.,0.),
                 device=None, dtype=torch.float32):
        super().__init__()
        # make them learnable if you want; otherwise wrap in register_buffer
        self.t0   = nn.Parameter(torch.tensor(t0_s,   dtype=dtype, device=device))
        self.w    = nn.Parameter(torch.tensor(width_s,dtype=dtype, device=device))
        self.Aimp = nn.Parameter(torch.tensor(A_imp,  dtype=dtype, device=device))
        self.dir  = nn.Parameter(torch.tensor(direction, dtype=dtype, device=device))
        self.register_buffer("env", torch.zeros(0, dtype=dtype, device=device))
        self.T = 0

    # def prepare(self, T: int):
    #     if T == self.T and self.env.numel() == T: return
    #     t = torch.arange(T, device=self.env.device, dtype=self.env.dtype) 
    #     sigma = F.softplus(self.w) + 1e-6
    #     g = torch.exp(-0.5 * ((t - self.t0) / sigma)**2)      # [T]
    #     area = (g.sum()).clamp_min(1e-9)
    #     self.env = F.softplus(self.Aimp) * g / area           # dt-invariant; ∑F*dt = A_imp
    #     self.T = T

    def step(self, t_idx: int) -> torch.Tensor:               # -> [3]
        d = self.dir / (self.dir.norm() + 1e-9)
        sigma = F.softplus(self.w) + 1e-6
        g = torch.exp(-0.5 * ((t_idx - self.t0) / sigma)**2)      # [T]
        area = (g.sum()).clamp_min(1e-9)
        out = F.softplus(self.Aimp) * g / area
        # self.env = F.softplus(self.Aimp) * g / area           # dt-invariant; ∑F*dt = A_imp
        return out * d