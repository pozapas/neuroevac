"""
ai_insights.py — Clustering, NLP summary, and XAI overlays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_epochs(
    features: pd.DataFrame,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    min_cluster_size: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """UMAP + HDBSCAN clustering on epoch features.

    Returns: (embedding_2d, embedding_2d_col2, cluster_labels)
    """
    try:
        import umap
        import hdbscan
    except ImportError:
        raise ImportError(
            "UMAP and HDBSCAN are required for clustering. "
            "Install with: pip install umap-learn hdbscan"
        )

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embedding)

    return embedding[:, 0], embedding[:, 1], labels.tolist()


# ---------------------------------------------------------------------------
# NLP Summary
# ---------------------------------------------------------------------------

def generate_summary(
    rec_metadata: dict,
    ch_names: list[str],
    band_powers: pd.DataFrame | None = None,
    anomaly_scores: np.ndarray | None = None,
    survey_responses: dict | None = None,
) -> str:
    """Generate a natural-language summary of the EEG recording analysis.

    This is template-driven (no external LLM needed).
    """
    lines = []

    # Recording overview
    fmt = rec_metadata.get("format", "unknown").upper()
    dur = rec_metadata.get("duration_s", 0)
    sfreq = rec_metadata.get("sfreq", 0)
    n_ch = len(ch_names)
    lines.append(
        f"**Recording Overview:** This {fmt}-format recording contains "
        f"{n_ch} EEG channels sampled at {sfreq:.0f} Hz, "
        f"totaling {dur:.1f} seconds of continuous data."
    )

    # Band power analysis
    if band_powers is not None and not band_powers.empty:
        mean_bp = band_powers.mean(axis=0)
        dominant_band = mean_bp.idxmax()
        lines.append(
            f"\n**Spectral Profile:** The dominant frequency band across channels "
            f"is **{dominant_band}** (mean power: {mean_bp[dominant_band]:.2e} V²/Hz)."
        )

        # Per-channel highlights
        for ch in ch_names[:4]:  # First 4 channels
            if ch in band_powers.index:
                ch_dominant = band_powers.loc[ch].idxmax()
                ratio = band_powers.loc[ch, ch_dominant] / band_powers.loc[ch].mean()
                if ratio > 2.0:
                    lines.append(
                        f"- *{ch}* shows elevated **{ch_dominant}** power "
                        f"({ratio:.1f}× mean), potentially indicating localized activity."
                    )

    # Anomaly summary
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        n_anom = np.sum(anomaly_scores > np.percentile(anomaly_scores, 95))
        pct = n_anom / len(anomaly_scores) * 100
        lines.append(
            f"\n**Anomaly Detection:** {n_anom} epochs ({pct:.1f}%) were flagged "
            f"as potentially anomalous (top 5% of anomaly scores)."
        )

        if pct > 10:
            lines.append(
                "- A high anomaly rate may indicate widespread artifacts or "
                "a non-stationary recording. Consider re-examining preprocessing parameters."
            )

    # Survey context
    if survey_responses:
        lines.append("\n**Participant Context:**")
        stress_keys = ["I feel stressed", "I feel anxious", "I feel nervous", "I feel tense"]
        calm_keys = ["I feel calm", "I feel relaxed"]

        stress_scores = [v for k, v in survey_responses.items()
                         if any(sk.lower() in k.lower() for sk in stress_keys) and not np.isnan(v)]
        calm_scores = [v for k, v in survey_responses.items()
                       if any(ck.lower() in k.lower() for ck in calm_keys) and not np.isnan(v)]

        if stress_scores:
            avg_stress = np.mean(stress_scores)
            level = "low" if avg_stress <= -1 else ("moderate" if avg_stress <= 0 else "elevated")
            lines.append(f"- Self-reported stress level: **{level}** (avg score: {avg_stress:.1f}/2)")

        if calm_scores:
            avg_calm = np.mean(calm_scores)
            level = "low" if avg_calm <= -1 else ("moderate" if avg_calm <= 0 else "high")
            lines.append(f"- Self-reported calmness: **{level}** (avg score: {avg_calm:.1f}/2)")

    return "\n".join(lines)


def generate_llm_summary(
    provider: str,
    model_name: str,
    api_key: str | None,
    rec_metadata: dict,
    ch_names: list[str],
    band_powers: pd.DataFrame | None = None,
    anomaly_scores: np.ndarray | None = None,
    survey_responses: dict | None = None,
) -> str:
    """Generate an AI summary using an LLM (Ollama or OpenRouter)."""

    # 1. Construct context string
    context = []
    fmt = rec_metadata.get("format", "unknown").upper()
    dur = rec_metadata.get("duration_s", 0)
    sfreq = rec_metadata.get("sfreq", 0)
    context.append(f"Recording: {fmt} format, {len(ch_names)} channels, {dur:.1f}s duration, {sfreq:.0f} Hz.")

    if band_powers is not None and not band_powers.empty:
        mean_bp = band_powers.mean(axis=0)
        dom = mean_bp.idxmax()
        context.append(f"Spectral Analysis: Dominant band across channels is {dom}.")
        # Top channel deviations
        for ch in ch_names[:4]:
            if ch in band_powers.index:
                ch_dom = band_powers.loc[ch].idxmax()
                val = band_powers.loc[ch, ch_dom]
                context.append(f"- Channel {ch}: Dominant {ch_dom} ({val:.2e} V²/Hz).")

    if anomaly_scores is not None and len(anomaly_scores) > 0:
        n_anom = np.sum(anomaly_scores > np.percentile(anomaly_scores, 95))
        rate = n_anom / len(anomaly_scores) * 100
        context.append(f"Anomaly Detection: {rate:.1f}% of epochs flagged as anomalous (ensemble consensus).")

    if survey_responses:
        context.append("Participant Survey Responses:")
        for k, v in survey_responses.items():
            context.append(f"- {k}: {v}")

    prompt = (
        "You are an expert neuroscientist and data analyst. "
        "Analyze the following EEG session summary data and provide a concise, "
        "professional report. Highlight the spectral composition, potential artifacts "
        "(suggested by anomalies), and any psychological context from the survey. "
        "Do not hallucinate data not present here.\n\n"
        "DATA:\n" + "\n".join(context)
    )

    # 2. Call LLM
    try:
        if provider == "Ollama":
            try:
                import ollama
            except ImportError:
                import requests
                # Fallback to raw API if lib missing
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model_name, "prompt": prompt, "stream": False},
                )
                if resp.status_code != 200:
                    raise ConnectionError(f"Ollama error: {resp.text}")
                return resp.json().get("response", "")

            # Use lib
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            return response["message"]["content"]

        elif provider == "OpenRouter":
            if not api_key:
                return "Error: OpenRouter API Key required."
            try:
                from openai import OpenAI
            except ImportError:
                return "Error: `openai` library required. Install with `pip install openai`."

            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return completion.choices[0].message.content

    except Exception as e:
        return f"Error generating summary with {provider}: {str(e)}"

    return "Error: Unknown provider."



# ---------------------------------------------------------------------------
# XAI (optional)
# ---------------------------------------------------------------------------

def compute_shap_values(
    features: pd.DataFrame,
    model,
) -> np.ndarray | None:
    """Compute SHAP values for the given model.

    Returns None if shap is not available.
    """
    try:
        import shap
    except ImportError:
        return None

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        return shap_values.values
    except Exception:
        # Fallback to KernelExplainer for unsupported models
        try:
            explainer = shap.KernelExplainer(model.decision_function, X[:100])
            return explainer.shap_values(X)
        except Exception:
            return None
