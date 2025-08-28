# Sonified Audio Explorer

A responsive Streamlit app for exploring audio features of your sonified data.

## Features
- Drag & drop audio (WAV/MP3/FLAC/OGG).
- Waveform, power spectrogram, Mel-spectrogram, MFCCs, chroma, spectral features, pitch/tempo.
- Interactive zoom/pan plots (Plotly) and downloadable CSV of features.
- Batch page: analyze multiple files at once and export features.
- Caching for fast iteration; clean, responsive UI.

## Quickstart
```bash
# 1) Create & activate a venv (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py
```

## Project Structure
```
sonified-audio-explorer/
├── app.py
├── pages/
│   └── 02_Batch_Analysis.py
├── src/
│   ├── features.py
│   └── utils_audio.py
├── assets/
│   └── (place optional sample audio here)
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── README.md
```

## Notes
- For long audio, use the "Segment" controls to preview/compute features on a time slice.
- Batch analysis computes summary statistics per file (mean/std/median of selected features).
- Extend `src/features.py` to add custom descriptors or domain-specific metrics.
