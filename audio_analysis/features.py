import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from feature_ranking import rank_features, plot_feature_ranking



AUDIO_DIR = Path("/Users/luisreyes/Desktop/samples/BioSonix-UserStudy-Media")
SR = 22050
SEG_DUR = 3.0
HOP = 1.5

def segment_audio(y, sr, seg_dur, hop):
    seg_len = int(seg_dur*sr); hop_len = int(hop*sr)
    return [y[i:i+seg_len] for i in range(0, max(1, len(y)-seg_len+1), hop_len)]

def features_framewise(y, sr):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-9
    mel = librosa.feature.melspectrogram(S=S**2, sr=sr)
    feats = {
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
        "rms": librosa.feature.rms(S=S)[0],
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr)[0],
        "spec_flux": np.hstack([[0], np.linalg.norm(np.diff(S, axis=1), axis=0)]),
        "mfcc": librosa.feature.mfcc(S=librosa.power_to_db(mel), sr=sr, n_mfcc=13)
    }
    # harmonics/pitch
    f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
    feats["f0"] = f0
    # chroma/tonnetz
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    feats["chroma"] = chroma
    feats["tonnetz"] = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    # onset/env
    feats["onset_env"] = librosa.onset.onset_strength(y=y, sr=sr)
    return feats

def aggregate_stats(feats):
    def agg_vec(v): 
        v = v[~np.isnan(v)]
        qs = np.percentile(v, [10,50,90]) if v.size else [np.nan]*3
        return {
            "mean": np.nanmean(v) if v.size else np.nan,
            "std": np.nanstd(v) if v.size else np.nan,
            "p10": qs[0], "p50": qs[1], "p90": qs[2]
        }
    rows = {}
    for k,v in feats.items():
        if v.ndim == 1:
            for stat, val in agg_vec(v).items():
                rows[f"{k}_{stat}"] = val
        else:
            # e.g., mfcc/chroma/tonnetz: aggregate per row then average across coeffs
            stats = [agg_vec(v[i]) for i in range(v.shape[0])]
            for stat in ["mean","std","p10","p50","p90"]:
                rows[f"{k}_{stat}_mean"] = np.nanmean([s[stat] for s in stats])
                rows[f"{k}_{stat}_std"]  = np.nanstd([s[stat] for s in stats])
    return rows

def extract_dataset(audio_dir=AUDIO_DIR):
    records = []
    for fp in audio_dir.glob("**/*.wav"):
        print(f"Processing {fp}")
        y, sr = sf.read(fp)
        if y.ndim > 1: y = np.mean(y, axis=1)  # mono policy; replace w/ loudness norm if desired
        y = librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y
        for idx, seg in enumerate(segment_audio(y, SR, SEG_DUR, HOP)):
            feats = features_framewise(seg, SR)
            row = aggregate_stats(feats)
            row.update(file=str(fp), segment=idx)
            records.append(row)
    return pd.DataFrame.from_records(records)

df = extract_dataset()
# print(df.head())
# exit()
df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")  # quick clean

# Standardize & reduce
X = StandardScaler().fit_transform(df.drop(columns=["file","segment"]))
pca = PCA(n_components=32).fit_transform(X)
embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=0).fit_transform(pca)

# Cluster
labels = hdbscan.HDBSCAN(min_cluster_size=5, metric="euclidean").fit_predict(embedding)

feat_rank_kw = rank_features(df, labels, method="kruskal", scale=True)
plot_feature_ranking(feat_rank_kw, top_k=50, title="Top 50 (Kruskal, z-scored)")