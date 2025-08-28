import io
import numpy as np
import soundfile as sf

def read_audio(file_bytes, dtype="float32"):
    """Return y, sr from an uploaded file (BytesIO or file-like)."""
    if hasattr(file_bytes, "read"):
        data = file_bytes.read()
    else:
        data = file_bytes
    buf = io.BytesIO(data)
    y, sr = sf.read(buf, dtype=dtype, always_2d=False)
    if y.ndim > 1:
        # mix down to mono
        y = np.mean(y, axis=1)
    return y, sr

def trim_segment(y, sr, t_start=None, t_end=None):
    if t_start is None and t_end is None:
        return y
    n = len(y)
    def to_idx(t):
        if t is None: return None
        return int(np.clip(t*sr, 0, n))
    i0 = to_idx(t_start) or 0
    i1 = to_idx(t_end) or n
    return y[i0:i1]
