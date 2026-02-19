"""
Survey Data — Participant questionnaire viewer with radar and bar charts.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

st.set_page_config(page_title="Survey Data", layout="wide")

from utils.sidebar import render_sidebar, render_sidebar_footer
render_sidebar()
render_sidebar_footer()

st.title("Survey Data")

surveys = st.session_state.get("surveys", {})
if not surveys:
    st.warning("No survey files loaded. Upload a Participant*.xlsx file on the main page.")
    st.stop()

# ── Display each survey ───────────────────────────────────────────────────
for sname, survey in surveys.items():
    st.header(survey.participant_id)

    if survey.raw_df.empty:
        st.info(f"No question-response pairs were detected in {sname}.")
        continue

    # Response table
    st.dataframe(survey.raw_df, use_container_width=True)

    # ── Radar chart of emotional states ─────────────────────────────────
    emotional_keys = [
        "I feel calm", "I feel stressed", "I feel anxious",
        "I feel relaxed", "I feel nervous", "I feel tense",
    ]

    emotional_data = {}
    for key in emotional_keys:
        for q, v in survey.responses.items():
            if key.lower().replace("i feel ", "") in q.lower():
                emotional_data[q] = v
                break

    if emotional_data:
        st.subheader("Emotional State Profile")

        categories = list(emotional_data.keys())
        values = list(emotional_data.values())

        # Radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(88, 166, 255, 0.2)",
            line=dict(color="#58a6ff", width=2),
            name=survey.participant_id,
        ))

        fig_radar.update_layout(
            template="plotly_dark",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2, 2],
                    tickvals=[-2, -1, 0, 1, 2],
                    ticktext=["-2\n(none)", "-1", "0", "1", "+2\n(very)"],
                ),
            ),
            showlegend=True,
            height=450,
            margin=dict(l=80, r=80, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bar chart
        st.subheader("Response Values")
        df_bar = pd.DataFrame({
            "Question": categories,
            "Response": values,
        })

        colors = ["#3fb950" if v >= 0 else "#ff7b72" for v in values]

        fig_bar = go.Figure(go.Bar(
            x=df_bar["Question"],
            y=df_bar["Response"],
            marker_color=colors,
            text=[f"{v:+.0f}" for v in values],
            textposition="outside",
        ))

        fig_bar.update_layout(
            template="plotly_dark",
            height=350,
            yaxis=dict(range=[-2.5, 2.5], title="Response (Likert −2 to +2)"),
            xaxis=dict(title=""),
            margin=dict(l=60, r=20, t=20, b=80),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Cross-participant comparison ───────────────────────────────────────────
if len(surveys) > 1:
    st.header("Cross-Participant Comparison")

    all_data = []
    for sname, survey in surveys.items():
        for q, v in survey.responses.items():
            all_data.append({
                "Participant": survey.participant_id,
                "Question": q,
                "Response": v,
            })

    if all_data:
        df_all = pd.DataFrame(all_data)
        fig_comp = px.bar(
            df_all, x="Question", y="Response", color="Participant",
            barmode="group", template="plotly_dark",
            title="Response Comparison",
        )
        fig_comp.update_layout(height=400, margin=dict(l=40, r=20, t=60, b=100))
        st.plotly_chart(fig_comp, use_container_width=True)

# ── EEG Contextual Display ────────────────────────────────────────────────
rec = st.session_state.get("active_rec")
if rec:
    st.header("EEG Recording Context")
    st.markdown(
        f"**Active recording:** {rec.metadata.get('file', 'N/A')} — "
        f"{len(rec.ch_names)} channels, {rec.sfreq:.0f} Hz, "
        f"{rec.metadata.get('duration_s', 0):.1f} s"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Channels", len(rec.ch_names))
        st.metric("Sample Rate", f"{rec.sfreq:.0f} Hz")
    with col2:
        st.metric("Duration", f"{rec.metadata.get('duration_s', 0):.1f} s")
        st.metric("Format", rec.metadata.get("format", "N/A").upper())
