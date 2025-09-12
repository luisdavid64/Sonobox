import math
import torch

def axis_pick_t(x3: torch.Tensor, axis: str) -> torch.Tensor:
    # x3: [..., 3] -> [...], axis in {'x','y','z','all'}
    if axis == 'x': return x3[..., 0]
    if axis == 'y': return x3[..., 1]
    if axis == 'z': return x3[..., 2]
    return x3.sum(dim=-1)

def _equal_power_gains_from_angles_t(azimuths: torch.Tensor) -> torch.Tensor:
    # azimuths: [C] in radians, clamp [-pi/2, +pi/2] -> gains [C,2]
    phi = torch.clamp((azimuths + math.pi/2) / math.pi * (math.pi/2), 0.0, math.pi/2)
    gL = torch.cos(phi)
    gR = torch.sin(phi)
    G  = torch.stack([gL, gR], dim=-1)  # [C,2]
    # normalize column power (not strictly required, but stable)
    norm = torch.sqrt(torch.sum(G**2, dim=-1, keepdim=True)) + 1e-9
    return G / norm

def _equal_power_gains_from_angles_t(az: torch.Tensor) -> torch.Tensor:
    """
    az in [-pi/2, pi/2], returns [C,2] equal-power stereo gains.
    Left = cos(theta), Right = sin(theta), where theta = az/2 + pi/4.
    """
    # ensure tensor
    if not torch.is_tensor(az):
        az = torch.as_tensor(az)
    device, dtype = az.device, az.dtype
    theta = az * 0.5 + (torch.pi / 4).to(device=device, dtype=dtype)
    gL = torch.cos(theta)
    gR = torch.sin(theta)
    G = torch.stack([gL, gR], dim=-1)               # [C,2]
    # each row has unit power by construction (gL^2 + gR^2 = 1)
    return G

def stereo_mixer_t(
    C: int,
    method: str = "by_position",
    listener_pos: torch.Tensor | None = None,
    plane: tuple[int, int] = (0, 2),
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Returns a fixed [C,2] gain matrix (listeners -> stereo).
    'spread' = even over [-90°, +90°]; 'by_position' = azimuth from listener_pos.
    """
    if method == "by_position" and listener_pos is not None:
        xy = listener_pos[:, list(plane)]                        # [C,2]
        az = torch.atan2(xy[:, 1], xy[:, 0])                    # [-pi, pi]
        az = az.clamp(-torch.pi/2, torch.pi/2)                  # pan to ±90°
    else:
        if C == 1:
            az = torch.zeros(1, device=device, dtype=dtype)
        else:
            az = torch.linspace(-torch.pi/2, torch.pi/2, C, device=device, dtype=dtype)
    return _equal_power_gains_from_angles_t(az)                 # [C,2]