"""
Raw EEG Viewer — Interactive multi-channel time-series visualization.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Raw EEG Viewer", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Raw EEG Viewer")

rec = st.session_state.get("active_rec")
if rec is None:
    st.warning("No recording loaded. Please upload a file on the main page.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────
col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])

with col_ctrl1:
    selected_channels = st.multiselect(
        "Channels to display",
        options=rec.ch_names,
        default=rec.ch_names[:8],
    )

with col_ctrl2:
    offset_scale = st.slider(
        "Channel offset (µV)",
        min_value=0,
        max_value=5000,
        value=500,
        step=50,
    )

with col_ctrl3:
    window_s = st.slider(
        "Time window (seconds)",
        min_value=1,
        max_value=int(rec.metadata.get("duration_s", 60)),
        value=min(10, int(rec.metadata.get("duration_s", 10))),
    )

# Time range
total_duration = rec.metadata.get("duration_s", len(rec.data) / rec.sfreq)
time_range = st.slider(
    "Time range (seconds)",
    min_value=0.0,
    max_value=max(0.1, total_duration - window_s),
    value=0.0,
    step=0.5,
)

if not selected_channels:
    st.info("Select at least one channel to display.")
    st.stop()

# ── Build plot ─────────────────────────────────────────────────────────────
start_sample = int(time_range * rec.sfreq)
end_sample = int((time_range + window_s) * rec.sfreq)
end_sample = min(end_sample, len(rec.data))

time_axis = np.arange(start_sample, end_sample) / rec.sfreq

fig = go.Figure()

for i, ch in enumerate(selected_channels):
    y = rec.data[ch].values[start_sample:end_sample]
    y_offset = y + i * offset_scale  # Stack channels vertically

    fig.add_trace(go.Scattergl(
        x=time_axis,
        y=y_offset,
        mode="lines",
        name=ch,
        line=dict(width=0.8),
        hovertemplate=f"{ch}<br>Time: %{{x:.3f}} s<br>Amplitude: %{{customdata:.1f}} µV",
        customdata=y,
    ))

fig.update_layout(
    template="plotly_dark",
    height=max(400, len(selected_channels) * 60),
    xaxis_title="Time (s)",
    yaxis_title="Amplitude (µV, offset)",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=40, b=60),
    yaxis=dict(
        tickvals=[i * offset_scale for i in range(len(selected_channels))],
        ticktext=selected_channels,
    ),
    dragmode="zoom",
)

st.plotly_chart(fig, use_container_width=True, key="raw_viewer")

# ── Annotation tools ───────────────────────────────────────────────────────
with st.expander("Annotation Tools", expanded=False):
    st.markdown("Mark segments of interest for downstream analysis.")
    ann_col1, ann_col2, ann_col3 = st.columns(3)
    ann_start = ann_col1.number_input("Start (s)", min_value=0.0, value=0.0, step=0.5)
    ann_end = ann_col2.number_input("End (s)", min_value=0.0, value=1.0, step=0.5)
    ann_label = ann_col3.text_input("Label", value="bad_segment")

    if st.button("Add Annotation"):
        if "annotations" not in st.session_state:
            st.session_state["annotations"] = []
        st.session_state["annotations"].append({
            "start": ann_start, "end": ann_end, "label": ann_label
        })
        st.success(f"Added: {ann_label} [{ann_start:.1f}s – {ann_end:.1f}s]")

    annotations = st.session_state.get("annotations", [])
    if annotations:
        st.dataframe(annotations, use_container_width=True)
