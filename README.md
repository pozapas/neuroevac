# EEG Analysis Dashboard

A research-grade, interactive Streamlit application for comprehensive EEG dataset analysis. Designed for neuroscientists and researchers needing a reproducible, multi-modal analysis pipeline that integrates classical signal processing, machine learning, and large language models in a single interface.

## Overview

The dashboard ingests raw EEG recordings (OpenBCI CSV/TXT, BrainFlow CSV) and optionally paired participant survey files (XLSX). All processing is handled on-device with no data leaving the local environment by default. An optional LLM summary feature can connect to Ollama (local) or OpenRouter (cloud).

## Features

| Page | Description |
|---|---|
| **Raw EEG Viewer** | Multi-channel time-series visualization with channel selector and time-window control |
| **Preprocessing** | Bandpass/notch filtering, re-referencing, bad-channel interpolation, ICA-based artifact removal |
| **Spectral Analysis** | Power Spectral Density (Welch), topographic scalp maps, frequency band power heatmaps |
| **Topography** | Spatial cortical activity maps across standard EEG frequency bands |
| **Anomaly Detection** | EEG-specific detectors (amplitude Z-score, spectral ratio, kurtosis/entropy) and general ML detectors (Isolation Forest, One-Class SVM, LOF, Autoencoder). Interactive heatmap, epoch gallery, and feature importance |
| **AI Insights** | UMAP + HDBSCAN unsupervised epoch clustering; AI-generated analysis reports via Template, Ollama, or OpenRouter |
| **Survey Data** | Participant survey ingestion, psychometric scoring, radar and bar chart visualizations |

## Project Structure

```
eeg_dashboard/
├── Dashboard.py              # Entry point — run this with streamlit
├── requirements.txt
├── assets/
│   ├── style.css             # Global theme styles
│   └── eeg.png               # Logo
├── pages/
│   ├── 1_Raw_EEG_Viewer.py
│   ├── 2_Preprocessing.py
│   ├── 3_Spectral_Analysis.py
│   ├── 4_Topography.py
│   ├── 5_Anomaly_Detection.py
│   ├── 6_AI_Insights.py
│   └── 7_Survey_Data.py
├── analysis/
│   ├── ai_insights.py        # Clustering, LLM integration (Ollama / OpenRouter)
│   ├── anomaly.py            # Anomaly detection algorithms
│   └── report.py             # HTML report builder
└── utils/
    ├── data_loader.py        # EEG and survey file parsing
    ├── sidebar.py            # Shared sidebar CSS, logo, footer
    └── signal_processing.py  # MNE wrappers, feature extraction, band power
```

## Requirements

- Python 3.10+
- See `requirements.txt` for all Python dependencies

```
pip install -r requirements.txt
```

Optional (for AI Insights LLM features):
```
pip install openai       # for OpenRouter cloud models
pip install ollama       # for local Ollama models
```

## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/<your-username>/eeg-dashboard.git
cd eeg-dashboard

# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run Dashboard.py
```

Then open `http://localhost:8501` in your browser.

### Supported File Formats

| Type | Format | Notes |
|---|---|---|
| EEG | OpenBCI raw CSV / TXT | 1-indexed headers, µV data |
| EEG | BrainFlow CSV | Auto-detected column layout |
| Survey | XLSX | Participant ID in column A, Likert-scale responses |

## AI Insights — LLM Configuration

The **AI Insights** page supports three summary modes:

1. **Template (Offline)** — Rule-based summary, no API key required.
2. **Local (Ollama)** — Sends a structured prompt to a locally running Ollama instance. Requires `ollama serve` and a downloaded model (e.g., `ollama pull llama3`).
3. **Cloud (OpenRouter)** — Sends the prompt to any model available on [openrouter.ai](https://openrouter.ai) using your personal API key.

No EEG data is transmitted externally unless you explicitly choose a cloud provider.

## Dependencies

Core scientific stack:
- [MNE-Python](https://mne.tools) — EEG preprocessing and analysis
- [Plotly](https://plotly.com/python/) — Interactive visualizations
- [scikit-learn](https://scikit-learn.org) — Machine learning detectors
- [UMAP-learn](https://umap-learn.readthedocs.io) + [HDBSCAN](https://hdbscan.readthedocs.io) — Epoch clustering
- [SciPy](https://scipy.org) — Spectral estimation

## Citation

If you use this dashboard in published research, please cite or acknowledge:

> Rafe, A. (2025). *EEG Analysis Dashboard: An interactive platform for multi-modal EEG analysis*. Texas State University.

## License

MIT License. See `LICENSE` for details.

## Author

**Amir Rafe**  
Texas State University  
[amir.rafe@txstate.edu](mailto:amir.rafe@txstate.edu)  
[pozapas.github.io](https://pozapas.github.io)
