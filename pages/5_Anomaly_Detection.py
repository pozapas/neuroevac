"""
Anomaly Detection — EEG-specific and ML/DL anomaly scoring with
interactive threshold, heatmap, epoch gallery, and feature importance.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

st.set_page_config(page_title="Anomaly Detection", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Anomaly Detection")

# ── Descriptive Introduction ───────────────────────────────────────────────
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
                border: 1px solid #30363d; border-radius: 12px; padding: 20px 24px;
                margin-bottom: 1.5rem; line-height: 1.7;">
        <h3 style="color: #58a6ff; margin-top: 0; font-size: 1.1rem;">
            🔍 What does EEG anomaly detection do?
        </h3>
        <p style="color: #c9d1d9; font-size: 0.9rem; margin-bottom: 10px;">
            EEG anomaly detection automatically identifies unusual segments in brain
            recordings — artifacts from muscle movement, eye blinks, electrode pops,
            or genuinely abnormal neural patterns (e.g., epileptiform spikes). By
            flagging these epochs, researchers can clean data before analysis or
            discover clinically relevant events.
        </p>
        <p style="color: #8b949e; font-size: 0.82rem; margin-bottom: 0;">
            <strong style="color: #c9d1d9;">This page runs two categories of detectors:</strong><br/>
            <span style="color: #3fb950;">■</span> <strong>EEG-Specific</strong> —
            Statistical Z-score thresholding, spectral band-ratio outliers (θ/β, α/β),
            and kurtosis/entropy analysis for muscle artifacts and flat-line detection.<br/>
            <span style="color: #58a6ff;">■</span> <strong>General ML/DL</strong> —
            Isolation Forest, Local Outlier Factor, One-Class SVM, and an optional
            Autoencoder — all applied to a rich feature space (band powers, Hjorth
            parameters, line length) extracted from each epoch.<br/>
            Results are ensembled into a single score and visualised as time-series,
            a channel×epoch heatmap, and an epoch gallery for inspection.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

from utils.signal_processing import make_fixed_epochs, extract_epoch_features
from analysis.anomaly import (
    run_isolation_forest, run_lof, run_ocsvm,
    run_autoencoder, ensemble_scores, _check_torch,
    run_zscore_detector, run_spectral_ratio_detector,
    run_kurtosis_entropy_detector,
)

raw = st.session_state.get("raw_filtered", rec.build_mne_raw())

# ── Epoch settings ─────────────────────────────────────────────────────────
st.header("⚙️ Detection Settings")
col1, col2 = st.columns(2)
with col1:
    epoch_dur = st.slider("Epoch duration (s)", 1.0, 5.0, 2.0, 0.5)
with col2:
    contamination = st.slider("Expected anomaly rate", 0.01, 0.20, 0.05, 0.01)

# ── Run Detection ──────────────────────────────────────────────────────────
if st.button("Run Anomaly Detection", type="primary"):
    with st.spinner("Creating epochs and extracting features..."):
        epochs = make_fixed_epochs(raw, duration=epoch_dur)
        features = extract_epoch_features(epochs)
        st.session_state["anomaly_features"] = features
        st.session_state["anomaly_epoch_dur"] = epoch_dur
        st.session_state["anomaly_epochs_obj"] = epochs

    eeg_scores = {}
    ml_scores = {}

    # ── EEG-Specific Detectors ─────────────────────────────────────────
    with st.spinner("Running Z-Score Threshold Detector..."):
        eeg_scores["Z-Score Threshold"] = run_zscore_detector(epochs)

    with st.spinner("Running Spectral Ratio Detector (θ/β, α/β)..."):
        eeg_scores["Spectral Ratio"] = run_spectral_ratio_detector(epochs)

    with st.spinner("Running Kurtosis & Entropy Detector..."):
        eeg_scores["Kurtosis & Entropy"] = run_kurtosis_entropy_detector(epochs)

    # ── General ML Detectors ───────────────────────────────────────────
    with st.spinner("Running Isolation Forest..."):
        ml_scores["Isolation Forest"] = run_isolation_forest(features, contamination=contamination)

    with st.spinner("Running Local Outlier Factor..."):
        ml_scores["LOF"] = run_lof(features, contamination=contamination)

    with st.spinner("Running One-Class SVM..."):
        ml_scores["OCSVM"] = run_ocsvm(features, nu=contamination)

    # DL (optional)
    if _check_torch():
        with st.spinner("Training Autoencoder..."):
            ae_scores = run_autoencoder(features, epochs=30)
            if ae_scores is not None:
                ml_scores["Autoencoder"] = ae_scores
    else:
        st.markdown(
            '<div class="dep-warning">'
            '<strong>Note:</strong> Deep learning features require PyTorch. '
            'Install with <code>pip install torch</code>.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Ensemble ───────────────────────────────────────────────────────
    all_scores = {**eeg_scores, **ml_scores}
    all_scores["Ensemble"] = ensemble_scores(all_scores)

    st.session_state["anomaly_scores"] = all_scores
    st.session_state["anomaly_eeg_scores"] = eeg_scores
    st.session_state["anomaly_ml_scores"] = ml_scores
    st.success(
        f"Detection complete! **{len(all_scores) - 1} detectors** run on "
        f"**{len(features)} epochs** ({epoch_dur}s each)."
    )

# ── Visualization ──────────────────────────────────────────────────────────
all_scores = st.session_state.get("anomaly_scores")
if all_scores is None:
    st.info("Configure settings above and click **Run Anomaly Detection** to begin.")
    st.stop()

eeg_scores = st.session_state.get("anomaly_eeg_scores", {})
ml_scores = st.session_state.get("anomaly_ml_scores", {})
epoch_dur = st.session_state.get("anomaly_epoch_dur", 2.0)
n_epochs = len(list(all_scores.values())[0])
time_axis = np.arange(n_epochs) * epoch_dur

# Threshold slider
threshold = st.slider(
    "Anomaly threshold (percentile)",
    min_value=80, max_value=99, value=95, step=1,
)

# ── Tabbed Results ─────────────────────────────────────────────────────────
tab_eeg, tab_ml, tab_heatmap, tab_gallery, tab_importance = st.tabs([
    "🧠 EEG-Specific Detectors",
    "🤖 General ML Detectors",
    "🗺️ Channel × Epoch Heatmap",
    "📋 Anomalous Epoch Gallery",
    "📊 Feature Importance",
])


def _plot_detector_group(score_dict, colors, threshold_pct):
    """Plot a group of detector scores as subplots."""
    names = list(score_dict.keys())
    n = len(names)
    if n == 0:
        st.info("No detectors in this group.")
        return

    fig = make_subplots(
        rows=n, cols=1, shared_xaxes=True,
        subplot_titles=names,
        vertical_spacing=max(0.03, 0.15 / n),
    )
    for i, (name, sc) in enumerate(score_dict.items()):
        if sc is None or len(sc) == 0:
            continue
        thresh_val = np.percentile(sc, threshold_pct)
        is_anomaly = sc > thresh_val

        fig.add_trace(go.Scatter(
            x=time_axis, y=sc, mode="lines",
            name=name, line=dict(color=colors[i % len(colors)], width=1.2),
            showlegend=False,
        ), row=i + 1, col=1)

        # Highlight anomalies
        fig.add_trace(go.Scatter(
            x=time_axis[is_anomaly], y=sc[is_anomaly], mode="markers",
            name=f"{name} anomalies",
            marker=dict(color="#ff7b72", size=6, symbol="x"),
            showlegend=False,
        ), row=i + 1, col=1)

        fig.add_hline(
            y=thresh_val, line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            row=i + 1, col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        height=max(350, n * 180),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=n, col=1)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab: EEG-Specific ─────────────────────────────────────────────────────
with tab_eeg:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        These detectors use <strong>EEG domain knowledge</strong> — amplitude statistics,
        spectral band ratios, and signal complexity measures — to identify
        physiologically implausible segments.
        </p>
        """,
        unsafe_allow_html=True,
    )
    _plot_detector_group(eeg_scores, ["#3fb950", "#f0883e", "#a371f7"], threshold)

# ── Tab: General ML ───────────────────────────────────────────────────────
with tab_ml:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        These detectors apply <strong>unsupervised machine learning</strong> to a
        multi-dimensional feature space (band powers, Hjorth parameters, line length)
        extracted from each epoch. They learn the "normal" distribution and flag outliers.
        </p>
        """,
        unsafe_allow_html=True,
    )
    _plot_detector_group(ml_scores, ["#58a6ff", "#f0883e", "#a371f7", "#3fb950"], threshold)

# ── Tab: Heatmap ──────────────────────────────────────────────────────────
with tab_heatmap:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        This heatmap shows the <strong>Z-score of peak amplitude</strong> for each
        channel in each epoch. Hot spots reveal which channels and time segments are
        most anomalous.
        </p>
        """,
        unsafe_allow_html=True,
    )
    epochs_obj = st.session_state.get("anomaly_epochs_obj")
    if epochs_obj is not None:
        data = epochs_obj.get_data()  # (n_epochs, n_channels, n_times)
        peak_amp = np.max(np.abs(data), axis=2)  # (n_epochs, n_channels)
        mu = np.mean(peak_amp, axis=0, keepdims=True)
        sigma = np.std(peak_amp, axis=0, keepdims=True) + 1e-12
        z_matrix = np.abs((peak_amp - mu) / sigma)  # (n_epochs, n_channels)

        fig_hm = go.Figure(data=go.Heatmap(
            z=z_matrix.T,
            x=[f"{t:.1f}s" for t in time_axis],
            y=rec.ch_names[:data.shape[1]],
            colorscale="YlOrRd",
            colorbar=dict(title="|Z-score|"),
        ))
        fig_hm.update_layout(
            template="plotly_dark",
            height=max(350, data.shape[1] * 30),
            xaxis_title="Epoch start time",
            yaxis_title="Channel",
            margin=dict(l=100, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No epoch data available.")

# ── Tab: Epoch Gallery ────────────────────────────────────────────────────
with tab_gallery:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        Raw EEG waveforms for the <strong>most anomalous epochs</strong> so you can
        visually inspect what was flagged. Red epochs = highest ensemble anomaly score.
        </p>
        """,
        unsafe_allow_html=True,
    )
    epochs_obj = st.session_state.get("anomaly_epochs_obj")
    ensemble_sc = all_scores.get("Ensemble", np.array([]))

    if epochs_obj is not None and len(ensemble_sc) > 0:
        n_show = st.slider("Number of top anomalous epochs to display", 3, 10, 5)
        top_idx = np.argsort(ensemble_sc)[-n_show:][::-1]

        data = epochs_obj.get_data()
        ch_names = rec.ch_names[:data.shape[1]]
        t_epoch = np.arange(data.shape[2]) / rec.sfreq

        for rank, ep_idx in enumerate(top_idx):
            with st.expander(
                f"#{rank+1}  —  Epoch {ep_idx}  "
                f"(t = {ep_idx * epoch_dur:.1f}s,  "
                f"score = {ensemble_sc[ep_idx]:.3f})",
                expanded=(rank == 0),
            ):
                fig_ep = go.Figure()
                n_ch_show = min(8, len(ch_names))
                offset = 0
                offsets = []
                for ch_i in range(n_ch_show):
                    sig = data[ep_idx, ch_i, :] * 1e6  # V → µV
                    fig_ep.add_trace(go.Scattergl(
                        x=t_epoch, y=sig + offset,
                        mode="lines", name=ch_names[ch_i],
                        line=dict(width=0.8),
                    ))
                    offsets.append(offset)
                    offset += np.ptp(sig) * 1.3 + 50

                fig_ep.update_layout(
                    template="plotly_dark",
                    height=max(250, n_ch_show * 45),
                    xaxis_title="Time (s)",
                    yaxis=dict(
                        tickvals=offsets,
                        ticktext=ch_names[:n_ch_show],
                    ),
                    showlegend=False,
                    margin=dict(l=80, r=20, t=10, b=40),
                )
                st.plotly_chart(fig_ep, use_container_width=True, key=f"gallery_{ep_idx}")
    else:
        st.info("No ensemble scores available.")

# ── Tab: Feature Importance ───────────────────────────────────────────────
with tab_importance:
    st.markdown(
        """
        <p style="color: #8b949e; font-size: 0.88rem; margin-bottom: 1rem;">
        Feature importances from the <strong>Isolation Forest</strong> model,
        showing which extracted features contribute most to anomaly scoring.
        </p>
        """,
        unsafe_allow_html=True,
    )
    features = st.session_state.get("anomaly_features")
    if features is not None:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        X = StandardScaler().fit_transform(features.values)
        clf = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        clf.fit(X)

        # Average absolute split importances across trees
        importances = np.zeros(X.shape[1])
        for tree in clf.estimators_:
            fi = tree.feature_importances_
            importances += fi
        importances /= len(clf.estimators_)

        # Top 20
        top_n = min(20, len(importances))
        top_idx = np.argsort(importances)[-top_n:][::-1]
        top_names = [features.columns[i] for i in top_idx]
        top_values = importances[top_idx]

        fig_fi = go.Figure(go.Bar(
            x=top_values[::-1],
            y=top_names[::-1],
            orientation="h",
            marker_color="#58a6ff",
        ))
        fig_fi.update_layout(
            template="plotly_dark",
            height=max(400, top_n * 28),
            xaxis_title="Feature Importance",
            margin=dict(l=200, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("No feature data available.")

# ── Summary ────────────────────────────────────────────────────────────────
st.header("📋 Detection Summary")

ensemble_sc = all_scores.get("Ensemble", np.array([]))
if len(ensemble_sc) > 0:
    thresh_val = np.percentile(ensemble_sc, threshold)
    n_flagged = int(np.sum(ensemble_sc > thresh_val))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Epochs", n_epochs)
    col2.metric("Flagged Epochs", n_flagged)
    col3.metric("Anomaly Rate", f"{n_flagged / n_epochs * 100:.1f}%")
    col4.metric("Detectors Used", len(all_scores) - 1)

    # Per-detector comparison table
    st.subheader("Per-Detector Breakdown")
    rows = []
    for name, sc in all_scores.items():
        if sc is None or len(sc) == 0 or name == "Ensemble":
            continue
        tv = np.percentile(sc, threshold)
        nf = int(np.sum(sc > tv))
        category = "EEG-Specific" if name in eeg_scores else "General ML"
        rows.append({
            "Detector": name,
            "Category": category,
            "Flagged Epochs": nf,
            "Flagged %": f"{nf / n_epochs * 100:.1f}%",
            "Mean Score": f"{np.mean(sc):.4f}",
            "Max Score": f"{np.max(sc):.4f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
