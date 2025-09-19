import math, json
import numpy as np

def make_bow_event_samples(
    T,                      # total samples (e.g., int(2.0*16000))
    fs=16000,
    n_ramp=int(0.08*16000),        # 80 ms attack
    n_release=int(0.6*16000),     # hold until 1.70 s
    n_fall=int(0.10*16000),        # 100 ms release
    N_max=0.30,            # max normal force (Fy)
    f_flip_hz=6.0,         # stick–slip-ish flip rate (sign changes)
    mu_s=0.6, mu_k=0.3,    # “static/kinetic” friction levels
    smooth=0.02,           # smoothness of sign flip (bigger = softer)
    noise_amp=0.03,        # small bow-hair jitter
    seed=0
):
    rng = np.random.default_rng(seed)
    n = np.arange(T, dtype=np.float32)

    # --- Normal force on Y (press) ---
    N = np.zeros(T, dtype=np.float32)
    if n_ramp > 0:
        N[:n_ramp] = N_max * (n[:n_ramp] / max(1, n_ramp))
    N[n_ramp:min(n_release, T)] = N_max
    if n_release < T:
        end = min(n_release + n_fall, T)
        seg = end - n_release
        if seg > 0:
            N[n_release:end] = N_max * (1.0 - (np.arange(seg, dtype=np.float32)/max(1, seg)))
        if end < T:
            N[end:] = 0.0

    # --- Tangential force on X (stick–slip surrogate) ---
    # smooth sign flips
    phase = 2*math.pi*f_flip_hz * (n / fs)
    s = np.tanh(np.sin(phase) / (smooth + 1e-6)).astype(np.float32)

    # slow micro-variation of friction coeff (optional)
    mu = (mu_k + (mu_s - mu_k) * 0.5 * (1.0 + np.cos(2*math.pi*0.7 * (n/fs)))).astype(np.float32)

    Fx = - N * mu * s + (noise_amp * N * rng.standard_normal(T).astype(np.float32))
    Fy = N
    Fz = np.zeros(T, dtype=np.float32)

    F = np.stack([Fx, Fy, Fz], axis=1)  # [T,3]
    return F  # per-sample force vector

def event_to_json_samples(F):
    """Map sample index -> [Fx,Fy,Fz] lists (dense; large for big T)."""
    return {str(i): [float(F[i,0]), float(F[i,1]), float(F[i,2])] for i in range(F.shape[0])}

# ---- example ----
if __name__ == "__main__":
    fs = 16000
    T = int(1.0 * fs)  # 1 second
    F = make_bow_event_samples(T, fs=fs)       # F.shape == [T,3]
    # save JSON with sample indices as keys
    with open("bow_like_event_samples.json", "w") as f:
        json.dump(event_to_json_samples(F), f)
