from __future__ import annotations
import numpy as np
import pandas as pd
import librosa

def feature_frame_times(sr:int, hop_length:int, n_frames:int)->np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

def compute_core(y:np.ndarray, sr:int, n_fft:int=2048, hop_length:int=512, n_mels:int=128):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=hop_length)
    rms = librosa.feature.rms(S=S)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Pitch (YIN) for monophonic-ish signals; may be NaN-heavy for noisy data
    try:
        f0 = librosa.yin(y, fmin=20, fmax=sr/2*0.9, frame_length=2048, hop_length=hop_length)
    except Exception:
        f0 = np.full_like(onset_env, np.nan, dtype=float)

    frames = mfcc.shape[1]
    t = feature_frame_times(sr, hop_length, frames)

    # Assemble a tidy DataFrame
    data = {
        "time": t,
        "zcr": zcr[0, :frames],
        "rms": rms[0, :frames],
        "spec_centroid": spec_cent[0, :frames],
        "spec_bandwidth": spec_bw[0, :frames],
        "rolloff": rolloff[0, :frames],
        "onset_env": onset_env[:frames],
        "f0_hz": f0[:frames] if np.ndim(f0)>0 else np.full(frames, np.nan),
    }
    df = pd.DataFrame(data)
    # Append MFCCs and chroma
    for i in range(mfcc.shape[0]):
        df[f"mfcc_{i+1}"] = mfcc[i, :frames]
    for i in range(chroma.shape[0]):
        df[f"chroma_{i}"] = chroma[i, :frames]

    return {
        "S": S,
        "mel_db": mel_db,
        "mfcc": mfcc,
        "chroma": chroma,
        "df": df,
        "hop_length": hop_length,
        "n_fft": n_fft,
    }

def summarize_features(df:pd.DataFrame, prefix:str="")->pd.Series:
    numeric = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric.columns:
        stats[f"{prefix}{col}_mean"] = float(np.nanmean(numeric[col].values))
        stats[f"{prefix}{col}_std"] = float(np.nanstd(numeric[col].values))
        stats[f"{prefix}{col}_median"] = float(np.nanmedian(numeric[col].values))
    return pd.Series(stats)
