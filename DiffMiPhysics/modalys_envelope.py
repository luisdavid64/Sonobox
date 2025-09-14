from collections import OrderedDict
from bisect import bisect_right
from util.util import save_event_to_json

class Envelope:
    """Piecewise-linear controller defined by breakpoints [(t, v), ...]."""
    def __init__(self, points):
        # points: iterable of (time_in_seconds, value)
        pts = sorted(points)
        self.t = [p[0] for p in pts]
        self.v = [p[1] for p in pts]

    def eval(self, x: float) -> float:
        if x <= self.t[0]:
            return self.v[0]
        if x >= self.t[-1]:
            return self.v[-1]
        i = bisect_right(self.t, x) - 1
        t0, t1 = self.t[i], self.t[i+1]
        v0, v1 = self.v[i], self.v[i+1]
        a = (x - t0) / (t1 - t0)
        return (1 - a) * v0 + a * v1


def sample_event_dict(
    start: float,
    duration: float,
    dt: float,
    env_x: Envelope | float = 0.0,
    env_y: Envelope | float = 0.0,
    env_z: Envelope | float = 0.0,
    time_key_fmt: str = "%.6f",
) -> dict[str, list[float]]:
    """
    Produce: { "t": [Fx, Fy, Fz], ... } for t in [start, start+duration], spaced by dt.
    Each axis can be an Envelope or a constant (float).
    """
    def get(env, t):
        return env.eval(t) if isinstance(env, Envelope) else float(env)

    t = start
    N = int(round(duration / dt))
    out = OrderedDict()
    for k in range(N + 1):
        local_t = t - start
        fx = get(env_x, local_t)
        fy = get(env_y, local_t)
        fz = get(env_z, local_t)
        out[time_key_fmt % t] = [fx, fy, fz]
        t += dt
    return out


# ---- examples ---------------------------------------------------------------

# 1) single, instantaneous “impulse” at t=0.5s (matches your JSON shape)
def impulse_event(t: float, force_vec=(3, 3, 3), time_key_fmt="%.2f"):
    return OrderedDict({time_key_fmt % t: list(map(float, force_vec))})

# 3) pluck: short triangular burst on one axis
def triangular_envelope(attack, release, peak):
    return Envelope([(0.0, 0.0), (attack, peak), (attack + release, 0.0)])

example1 = impulse_event(0.5, (3, 3, 3)) | impulse_event(0.75, (3, 3, 5))
print(example1)
# example1 == { "0.50": [3.0, 3.0, 3.0], "0.75": [3.0, 3.0, 5.0] }

# 2) “bow” style continuous excitation for 1.8 s sampled every 1/240 s
# (you can feed this straight to your simulator loop)
vBow = Envelope([(0.0, 0.10), (0.30, 0.30), (0.80, 0.10), (1.80, 0.00)])  # tangential speed → map to force if needed
fNorm = Envelope([(0.00, 0.0), (0.02, 0.5), (0.10, 0.3), (1.70, 0.3), (1.80, 0.0)])  # pressure

# simplest mapping: use envelopes directly as forces on axes (replace with your bow model if you have one)
bow_events = sample_event_dict(
    start=0.00,
    duration=1.80,
    dt=1/240.0,
    env_x=vBow,    # e.g., tangential force
    env_y=fNorm,   # e.g., normal force
    env_z=0.0
)
print(bow_events)
save_event_to_json(bow_events, "events/bow.json")

# pluck_events = sample_event_dict(
#     start=0.40,
#     duration=0.06,
#     dt=1/480.0,
#     env_x=triangular_envelope(0.01, 0.05, 5.0),  # 5 N peak
#     env_y=0.0,
#     env_z=0.0
# )
# print(pluck_events)