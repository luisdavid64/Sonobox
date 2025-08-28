from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
from src.utils_audio import read_audio
from src.features import compute_core, summarize_features

st.title("📦 Batch Analysis")
st.caption("Drop multiple files to compute summary stats and export a table.")

files = st.file_uploader("Drop multiple audio files", type=["wav","mp3","flac","ogg"], accept_multiple_files=True)
n_fft = st.slider("FFT size", 512, 8192, 2048, step=512)
hop_length = st.slider("Hop length", 64, 2048, 512, step=64)
n_mels = st.slider("Mel bands", 16, 256, 128, step=8)

@st.cache_data(show_spinner=True)
def process_one(uploaded_file, n_fft, hop_length, n_mels):
    y, sr = read_audio(uploaded_file)
    core = compute_core(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    s = summarize_features(core["df"], prefix="")
    s["filename"] = uploaded_file.name
    return s

rows = []
if files:
    for f in files:
        try:
            rows.append(process_one(f, n_fft, hop_length, n_mels))
        except Exception as e:
            st.warning(f"Failed on {f.name}: {e}")

if rows:
    table = pd.DataFrame(rows).set_index("filename")
    st.dataframe(table, use_container_width=True)
    csv = table.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("Download summary table (CSV)", csv, file_name="batch_summary.csv", mime="text/csv")
else:
    st.info("Add some files to see the summary table.")
