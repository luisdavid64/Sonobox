import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import librosa
from src.utils_audio import read_audio, trim_segment
from src.features import compute_core

st.set_page_config(page_title="Sonified Audio Explorer", page_icon="🎧", layout="wide")

st.title("🎧 Sonified Audio Explorer")
st.caption("Upload audio, explore features, and export what matters.")

with st.sidebar:
    st.header("Upload & Settings")
    file = st.file_uploader("Drop an audio file (wav/mp3/flac/ogg)", type=["wav","mp3","flac","ogg"])
    n_fft = st.slider("FFT size", 512, 8192, 2048, step=512)
    hop_length = st.slider("Hop length", 64, 2048, 512, step=64)
    n_mels = st.slider("Mel bands", 16, 256, 128, step=8)
    st.divider()
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        t_start = st.number_input("Segment start (s)", min_value=0.0, value=0.0, step=0.1, format="%.3f")
    with t_col2:
        t_end = st.number_input("Segment end (s, 0 = full length)", min_value=0.0, value=0.0, step=0.1, format="%.3f")
    st.info("Leave end at 0 for full length. Segments speed up visualization on long files.")

@st.cache_data(show_spinner=False)
def _compute(file_bytes, n_fft, hop_length, n_mels, t_start, t_end):
    y, sr = read_audio(file_bytes)
    y = trim_segment(y, sr, t_start if t_end>0 else None, t_end if t_end>0 else None)
    core = compute_core(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    return y, sr, core

def line_plot(x, y, name, x_title, y_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=x_title, yaxis_title=y_title,
        height=280,
    )
    return fig

def heatmap(data, x, y, title, x_title, y_title):
    fig = go.Figure(data=go.Heatmap(z=data, x=x, y=y, coloraxis="coloraxis"))
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=35, b=10),
        height=320,
        coloraxis=dict(colorbar=dict(title="dB"))
    )
    return fig

if file is not None:
    y, sr, core = _compute(file, n_fft, hop_length, n_mels, t_start, t_end)
    st.success(f"Loaded **{file.name}** at **{sr} Hz** · Duration: **{len(y)/sr:.2f}s**")
    st.audio(file)

    # Waveform
    st.subheader("Waveform")
    t = np.arange(len(y))/sr
    st.plotly_chart(line_plot(t, y, "amplitude", "Time (s)", "Amplitude"), use_container_width=True)

    # Spectrograms & MFCCs
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Mel Spectrogram")
        times = librosa.frames_to_time(np.arange(core["mel_db"].shape[1]), sr=sr, hop_length=core["hop_length"])
        mel_fig = heatmap(core["mel_db"], times, np.arange(core["mel_db"].shape[0]), "Mel Spectrogram", "Time (s)", "Mel band")
        st.plotly_chart(mel_fig, use_container_width=True)

        st.markdown("#### Chromagram")
        chroma_times = librosa.frames_to_time(np.arange(core["chroma"].shape[1]), sr=sr, hop_length=core["hop_length"])
        chroma_fig = heatmap(core["chroma"], chroma_times, [f"C{i}" for i in range(core["chroma"].shape[0])], "Chroma", "Time (s)", "Pitch class")
        st.plotly_chart(chroma_fig, use_container_width=True)

    with right:
        st.markdown("#### MFCCs")
        mfcc_times = librosa.frames_to_time(np.arange(core["mfcc"].shape[1]), sr=sr, hop_length=core["hop_length"])
        mfcc_fig = heatmap(core["mfcc"], mfcc_times, [f"MFCC {i+1}" for i in range(core['mfcc'].shape[0])], "MFCCs", "Time (s)", "Coeff")
        st.plotly_chart(mfcc_fig, use_container_width=True)

    # Scalar feature tracks
    st.subheader("Feature Tracks")
    df = core["df"]
    sel = st.multiselect("Select tracks to display", options=[c for c in df.columns if c!="time"],
                         default=["rms","spec_centroid","spec_bandwidth","f0_hz"])
    for col in sel:
        st.plotly_chart(line_plot(df["time"], df[col], col, "Time (s)", col), use_container_width=True)

    # Download features
    st.subheader("Export Features")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download per-frame features (CSV)", csv, file_name=f"{Path(file.name).stem}_features.csv", mime="text/csv")

else:
    st.info("Upload an audio file to begin.")
