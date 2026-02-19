"""
AI Insights — Clustering, NLP summary, and XAI overlays.
"""

import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="AI Insights", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("AI Insights")

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

from analysis.ai_insights import cluster_epochs, generate_summary

# ── Clustering ─────────────────────────────────────────────────────────────
st.header("Unsupervised Pattern Discovery")

features = st.session_state.get("anomaly_features")
if features is None:
    st.info(
        "Run anomaly detection first (Anomaly Detection page) to generate epoch features, "
        "or click below to extract them now."
    )
    if st.button("Extract Features"):
        from utils.signal_processing import make_fixed_epochs, extract_epoch_features
        raw = st.session_state.get("raw_filtered", rec.build_mne_raw())
        with st.spinner("Extracting features..."):
            epochs = make_fixed_epochs(raw, duration=2.0)
            features = extract_epoch_features(epochs)
            st.session_state["anomaly_features"] = features
        st.success(f"Extracted {features.shape[1]} features from {features.shape[0]} epochs.")
        st.rerun()
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    n_neighbors = st.slider("UMAP neighbors", 5, 50, 15)
with col2:
    min_dist = st.slider("UMAP min_dist", 0.0, 1.0, 0.1, 0.05)
with col3:
    min_cluster = st.slider("Min cluster size", 2, 20, 5)

if st.button("Run Clustering", type="primary"):
    try:
        with st.spinner("Computing UMAP embedding and HDBSCAN clustering..."):
            x, y, labels = cluster_epochs(
                features,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                min_cluster_size=min_cluster,
            )
        st.session_state["cluster_data"] = {"x": x, "y": y, "labels": labels}
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = labels.count(-1)
        st.success(f"Found {n_clusters} clusters ({n_noise} noise points).")
    except ImportError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Clustering failed: {e}")

# Show cluster plot
cluster_data = st.session_state.get("cluster_data")
if cluster_data:
    df_cluster = pd.DataFrame({
        "UMAP 1": cluster_data["x"],
        "UMAP 2": cluster_data["y"],
        "Cluster": [str(l) for l in cluster_data["labels"]],
        "Epoch": list(range(len(cluster_data["x"]))),
    })

    fig = px.scatter(
        df_cluster, x="UMAP 1", y="UMAP 2",
        color="Cluster", hover_data=["Epoch"],
        template="plotly_dark",
        title="Epoch Clustering (UMAP + HDBSCAN)",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.update_layout(height=500, margin=dict(l=40, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)

# ── Automated Report ───────────────────────────────────────────────────────
st.header("🤖 AI Report Generation")

st.markdown(
    """
    <div style="background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
                border: 1px solid #30363d; border-radius: 12px; padding: 16px 20px;
                margin-bottom: 1.5rem;">
        Generate a professional neurophysiological report using either a standard
        template or advanced Large Language Models (LLMs).
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    provider = st.selectbox(
        "AI Provider",
        ["Template (Offline)", "Local (Ollama)", "Cloud (OpenRouter)"],
        help="Select the backend for report generation."
    )

model_name = ""
api_key = None

with col2:
    if provider == "Local (Ollama)":
        model_name = st.text_input("Model Name", value="ollama run gemini-3-flash-preview", help="e.g., llama3, mistral, gemma")
    elif provider == "Cloud (OpenRouter)":
        model_name = st.text_input("Model Name", value="google/gemini-3-flash-preview", help="e.g., openai/gpt-4o, anthropic/claude-3-opus")

with col3:
    if provider == "Cloud (OpenRouter)":
        api_key = st.text_input("OpenRouter API Key", type="password", help="Enter your OpenRouter API key here.")

# Generate Button
if st.button("Generate Analysis Report", type="primary"):
    with st.spinner("Analyzing data and streamlining insights..."):
        # 1. Gather context
        survey_responses = None
        surveys = st.session_state.get("surveys", {})
        if surveys:
            first_survey = list(surveys.values())[0]
            survey_responses = first_survey.responses

        # Band power
        from utils.signal_processing import compute_psd, compute_band_powers
        raw = st.session_state.get("raw_filtered", rec.build_mne_raw())
        try:
            _nyq = raw.info["sfreq"] / 2.0
            psds, freqs = compute_psd(raw, fmin=0.5, fmax=_nyq - 0.5)
            bp = compute_band_powers(psds, freqs)
            bp.index = rec.ch_names
        except Exception:
            bp = None

        # Anomalies
        anomaly_scores = None
        scores_dict = st.session_state.get("anomaly_scores", {})
        if "Ensemble" in scores_dict:
            anomaly_scores = scores_dict["Ensemble"]

        # 2. Call Generator
        if provider == "Template (Offline)":
            summary = generate_summary(
                rec_metadata=rec.metadata,
                ch_names=rec.ch_names,
                band_powers=bp,
                anomaly_scores=anomaly_scores,
                survey_responses=survey_responses,
            )
        else:
            from analysis.ai_insights import generate_llm_summary
            p_code = "Ollama" if "Ollama" in provider else "OpenRouter"
            summary = generate_llm_summary(
                provider=p_code,
                model_name=model_name,
                api_key=api_key,
                rec_metadata=rec.metadata,
                ch_names=rec.ch_names,
                band_powers=bp,
                anomaly_scores=anomaly_scores,
                survey_responses=survey_responses,
            )

        st.session_state["ai_summary_text"] = summary

# Display Result
summary_text = st.session_state.get("ai_summary_text")
if summary_text:
    st.subheader("Generated Report")
    st.markdown(
        f"""
        <div style="background: #0d1117; border: 1px solid #30363d; border-left: 4px solid #a371f7;
                    border-radius: 6px; padding: 1.5rem; line-height: 1.7;">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.download_button(
        "Download Report Text",
        summary_text,
        file_name="ai_analysis_report.md"
    )
