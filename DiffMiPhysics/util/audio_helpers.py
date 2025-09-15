import math
import torch
import torchaudio.transforms as T

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
    model,
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
        if not hasattr(model, "_G_cache") or model._G_cache is None \
        or model._G_cache.shape[0] != C:
            lp = model.rest_pos[listener_ids.to(device, dtype=torch.long)]  # [C,3]
            G = stereo_mixer_t(C, method=method, listener_pos=lp, plane=plane,
                                device=device, dtype=dtype)                 # [C,2]
            if energy_comp and C > 0:
                G = G / math.sqrt(C)                                       # stabilize loudness
            model._G_cache = G                                              # cache on module
        G = model._G_cache.to(device=device, dtype=dtype)
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

def normalize_rms_to_dbfs(x: torch.Tensor,
                          target_db: float = -20.0,
                          eps: float = 1e-6,
                          gain_max_db: float = 40.0):
    """
    x: [T, C] audio tensor
    target_db: desired RMS in dBFS (0 dBFS = full scale = 1.0)
    eps: floor on power to avoid divide-by-zero
    gain_max_db: optional safety cap on boost
    """
    # constant linear target RMS (no signal logs)
    target_rms = (10.0 ** (target_db / 20.0))  # scalar float

    # current RMS per channel (no logs; use rsqrt for stability)
    power = (x.float().pow(2).mean(dim=0, keepdim=True)).clamp_min(eps)  # [1,C]
    inv_rms = torch.rsqrt(power)                                         # 1/sqrt

    # gain to hit target RMS, with an optional cap
    gain = target_rms * inv_rms                                          # [1,C]
    gain_cap = 10.0 ** (gain_max_db / 20.0)
    gain = gain.clamp(max=gain_cap)

    y = x * gain
    # optional gentle safety limiter to catch rare overs without NANs
    # y = torch.tanh(y)
    return y

import torch
import torch.nn.functional as F
from audiotools import AudioSignal

def resample(signal, sr, resample=True, clap_sr=44100):
    audio_time_series = signal
    resample_rate = clap_sr
    sample_rate = sr

    if resample and resample_rate != sample_rate:
        resampler = T.Resample(sample_rate, resample_rate)
        resampler.to(signal.device)
        audio_time_series = resampler(audio_time_series)
        signal = audio_time_series

    return signal

def clap_preprocess(segment, fs_in, clap_sr=44100, clap_dur_s=7.0, center=True, noise_db=None):
    y = resample(segment, fs_in, resample=True, clap_sr=clap_sr)
    Tt = int(round(clap_sr * clap_dur_s))
    if y.shape[0] >= Tt:
        start = (y.shape[0]-Tt)//2
        y = y[start:start+Tt]
    else:
        pad = Tt - y.shape[0]
        pre = pad//2 if center else 0
        post = pad - pre
        y = F.pad(y, (pre, post))
        if noise_db is not None:
            # optional very low pink/white noise floor, e.g. noise_db=-60
            amp = 10**(noise_db/20)
            y = y + amp * torch.randn_like(y)

    return y.unsqueeze(0).unsqueeze(0)  # [B=1, C=1, T] for CLAP