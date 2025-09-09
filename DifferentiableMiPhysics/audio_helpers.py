import math
import torch

def _axis_pick_t(x3: torch.Tensor, axis: str) -> torch.Tensor:
    # x3: [..., 3] -> [...], axis in {'x','y','z','all'}
    if axis == 'x': return x3[..., 0]
    if axis == 'y': return x3[..., 1]
    if axis == 'z': return x3[..., 2]
    return x3.sum(dim=-1)

def _finite_diff_t(x: torch.Tensor, dt: float) -> torch.Tensor:
    # x: [T, C] or [T, C, 3] -> same shape
    d = torch.empty_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    d[0]    = (x[1]  - x[0])   / dt
    d[-1]   = (x[-1] - x[-2])  / dt
    return d

def _dc_block_t(sig: torch.Tensor, R: float = 0.995) -> torch.Tensor:
    # sig: [T, K]
    T, K = sig.shape
    y = torch.zeros_like(sig)
    x1 = torch.zeros(K, device=sig.device, dtype=sig.dtype)
    y1 = torch.zeros(K, device=sig.device, dtype=sig.dtype)
    for n in range(T):
        yn = sig[n] - x1 + R * y1
        y[n] = yn
        x1, y1 = sig[n], yn
    return y

def _equal_power_gains_from_angles_t(azimuths: torch.Tensor) -> torch.Tensor:
    # azimuths: [C] in radians, clamp [-pi/2, +pi/2] -> gains [C,2]
    phi = torch.clamp((azimuths + math.pi/2) / math.pi * (math.pi/2), 0.0, math.pi/2)
    gL = torch.cos(phi)
    gR = torch.sin(phi)
    G  = torch.stack([gL, gR], dim=-1)  # [C,2]
    # normalize column power (not strictly required, but stable)
    norm = torch.sqrt(torch.sum(G**2, dim=-1, keepdim=True)) + 1e-9
    return G / norm

def _stereo_mixer_t(C: int,
                    method: str = "spread",
                    listener_pos: torch.Tensor | None = None,
                    plane: tuple[int,int] = (0,2)) -> torch.Tensor:
    """
    Returns a fixed [C,2] gain matrix (listeners -> stereo).
    method: 'spread' (even over -90..+90 deg) or 'by_position' (azimuth from listener_pos)
    """
    if method == "by_position" and listener_pos is not None:
        xy = listener_pos[:, list(plane)]                # [C,2]
        az = torch.atan2(xy[:, 1], xy[:, 0])            # [-pi, pi]
        az = torch.clamp(az, -math.pi/2, math.pi/2)
    else:
        if C == 1:
            az = torch.tensor([0.0])
        else:
            az = torch.linspace(-math.pi/2, math.pi/2, C)
    G = _equal_power_gains_from_angles_t(az)            # [C,2]
    return G           