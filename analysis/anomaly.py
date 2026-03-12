"""
anomaly.py — ML/DL anomaly detection on EEG epoch features,
plus trigger-aware analysis for VR experiment paradigms.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


def _check_torch():
    """Return True if torch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Classical ML detectors
# ---------------------------------------------------------------------------

def run_isolation_forest(
    features: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    """Isolation Forest anomaly scores. Higher = more anomalous."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    clf = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    clf.fit(X)
    scores = -clf.decision_function(X)  # Negate so higher = more anomalous
    return scores


def run_lof(
    features: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> np.ndarray:
    """Local Outlier Factor anomaly scores."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    clf.fit_predict(X)
    scores = -clf.negative_outlier_factor_
    return scores


def run_ocsvm(
    features: pd.DataFrame,
    nu: float = 0.05,
    kernel: str = "rbf",
) -> np.ndarray:
    """One-Class SVM anomaly scores."""
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma="scale")
    clf.fit(X)
    scores = -clf.decision_function(X)
    return scores


# ---------------------------------------------------------------------------
# EEG-specific detectors
# ---------------------------------------------------------------------------

def run_zscore_detector(
    epochs,
    z_threshold: float = 3.0,
) -> np.ndarray:
    """Flag epochs where any channel exceeds ±z_threshold standard deviations.

    Returns a per-epoch anomaly score = max |z| across channels.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    # Peak amplitude per channel per epoch
    peak_amp = np.max(np.abs(data), axis=2)  # (n_epochs, n_channels)
    # Z-score across epochs for each channel
    mu = np.mean(peak_amp, axis=0, keepdims=True)
    sigma = np.std(peak_amp, axis=0, keepdims=True) + 1e-12
    z = np.abs((peak_amp - mu) / sigma)
    # Score = max z across channels
    return np.max(z, axis=1)


def run_spectral_ratio_detector(
    epochs,
) -> np.ndarray:
    """Detect anomalies via theta/beta and alpha/beta ratio outliers.

    Uses Mahalanobis-like distance from the median ratios.
    """
    from utils.signal_processing import compute_spectral_ratios
    ratios = compute_spectral_ratios(epochs)
    X = ratios.values  # (n_epochs, 2)
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0) + 1e-12
    # Robust z-score (Modified Z-score)
    z = np.abs((X - median) / (1.4826 * mad))
    return np.max(z, axis=1)


def run_kurtosis_entropy_detector(
    epochs,
) -> np.ndarray:
    """Detect anomalies via extreme kurtosis (muscle) or entropy (flat/noise).

    Returns a combined score (max of kurtosis z-score and entropy z-score
    across channels).
    """
    from utils.signal_processing import compute_epoch_kurtosis, compute_sample_entropy

    kurt = compute_epoch_kurtosis(epochs)  # (n_epochs, n_channels)
    # Robust z-score of kurtosis
    kurt_med = np.median(kurt, axis=0)
    kurt_mad = np.median(np.abs(kurt - kurt_med), axis=0) + 1e-12
    kurt_z = np.abs((kurt - kurt_med) / (1.4826 * kurt_mad))

    entropy = compute_sample_entropy(epochs)  # (n_epochs, n_channels)
    ent_med = np.median(entropy, axis=0)
    ent_mad = np.median(np.abs(entropy - ent_med), axis=0) + 1e-12
    ent_z = np.abs((entropy - ent_med) / (1.4826 * ent_mad))

    # Combined: max across channels of max(kurt_z, ent_z)
    combined = np.maximum(np.max(kurt_z, axis=1), np.max(ent_z, axis=1))
    return combined


# ---------------------------------------------------------------------------
# Deep Learning detectors (optional — requires torch)
# ---------------------------------------------------------------------------

def run_autoencoder(
    features: pd.DataFrame,
    epochs: int = 50,
    hidden_dim: int = 32,
    lr: float = 1e-3,
) -> np.ndarray | None:
    """Conv1D autoencoder reconstruction error as anomaly score.

    Returns None if torch is not available.
    """
    if not _check_torch():
        return None

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values).astype(np.float32)
    n_features = X.shape[1]

    # Simple autoencoder
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, n_features),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    device = torch.device("cpu")
    model = AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")

    tensor_x = torch.from_numpy(X)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = criterion(recon, batch).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        recon = model(tensor_x.to(device))
        recon_error = criterion(recon, tensor_x.to(device)).mean(dim=1).cpu().numpy()

    return recon_error


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def ensemble_scores(score_dict: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize and average anomaly scores from multiple detectors."""
    normalized = {}
    for name, scores in score_dict.items():
        if scores is not None:
            s_min, s_max = scores.min(), scores.max()
            if s_max > s_min:
                normalized[name] = (scores - s_min) / (s_max - s_min)
            else:
                normalized[name] = np.zeros_like(scores)

    if not normalized:
        return np.array([])

    stacked = np.stack(list(normalized.values()), axis=0)
    return np.mean(stacked, axis=0)


# ---------------------------------------------------------------------------
# Trigger CSV parsing
# ---------------------------------------------------------------------------

def parse_trigger_csv(file_content: bytes | str) -> pd.DataFrame:
    """Parse a Trigger Time CSV and convert trigger times to seconds.

    Handles the ``MM:SS.S`` format used in the experiment.
    Returns a DataFrame with an added ``trigger_time_s`` column (float, NaN
    if unparseable or missing).
    """
    from io import BytesIO, StringIO

    if isinstance(file_content, bytes):
        df = pd.read_csv(BytesIO(file_content))
    else:
        df = pd.read_csv(StringIO(file_content))

    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    def _parse_time(val) -> float:
        """Convert 'MM:SS.S' or 'MM:SS' to seconds. Returns NaN on failure."""
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        s = str(val).strip()
        m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", s)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        # Try plain float (seconds already)
        try:
            return float(s)
        except ValueError:
            return np.nan

    df["trigger_time_s"] = df["Trigger Time"].apply(_parse_time)
    return df


def get_trigger_info(trigger_df: pd.DataFrame, participant: str) -> dict | None:
    """Return a dict with trigger details for a given participant, or None."""
    row = trigger_df[trigger_df["Participant Number"].str.strip() == participant.strip()]
    if row.empty:
        return None
    row = row.iloc[0]
    t = row.get("trigger_time_s", np.nan)
    if pd.isna(t):
        return None
    info: dict = {
        "participant": participant,
        "group": str(row.get("User Group", "Unknown")).strip(),
        "trigger_time_s": float(t),
        "explanations": [],
    }
    for col in ["Explanation1", "Explanation2"]:
        v = row.get(col)
        if pd.notna(v) and str(v).strip():
            info["explanations"].append(str(v).strip())
    return info


# ---------------------------------------------------------------------------
# Trigger-Locked Anomaly Response (TLAR)
# ---------------------------------------------------------------------------

def compute_trigger_locked_scores(
    score_dict: dict[str, np.ndarray],
    epoch_dur: float,
    trigger_time_s: float,
    window_pre: float = 15.0,
    window_post: float = 30.0,
) -> dict:
    """Compute anomaly scores aligned to trigger onset (time 0).

    Returns dict with:
      - ``time_axis``: array of epoch-centre times relative to trigger
      - ``scores``: dict of detector name → score values within window
      - ``epoch_indices``: the original epoch indices that fall in the window
    """
    n_epochs = len(list(score_dict.values())[0])
    # Epoch centre times (absolute)
    epoch_centres = np.arange(n_epochs) * epoch_dur + epoch_dur / 2.0
    # Relative to trigger
    rel_times = epoch_centres - trigger_time_s

    mask = (rel_times >= -window_pre) & (rel_times <= window_post)
    indices = np.where(mask)[0]

    result_scores: dict[str, np.ndarray] = {}
    for name, sc in score_dict.items():
        if sc is not None and len(sc) == n_epochs:
            result_scores[name] = sc[mask]

    return {
        "time_axis": rel_times[mask],
        "scores": result_scores,
        "epoch_indices": indices,
    }


# ---------------------------------------------------------------------------
# Pre / Post Trigger Statistical Comparison
# ---------------------------------------------------------------------------

def pre_post_trigger_test(
    score_dict: dict[str, np.ndarray],
    epoch_dur: float,
    trigger_time_s: float,
    pre_window: float = 30.0,
    post_window: float = 30.0,
) -> pd.DataFrame:
    """Compare per-detector anomaly scores before vs after trigger.

    Returns a DataFrame with columns:
      Detector, Pre Mean, Post Mean, Change %, Wilcoxon W, p-value, Cohen d
    """
    n_epochs = len(list(score_dict.values())[0])
    epoch_starts = np.arange(n_epochs) * epoch_dur

    pre_mask = (epoch_starts >= trigger_time_s - pre_window) & (epoch_starts < trigger_time_s)
    post_mask = (epoch_starts >= trigger_time_s) & (epoch_starts < trigger_time_s + post_window)

    rows = []
    for name, sc in score_dict.items():
        if sc is None or len(sc) == 0 or name == "Ensemble":
            continue
        pre_vals = sc[pre_mask]
        post_vals = sc[post_mask]

        if len(pre_vals) < 3 or len(post_vals) < 3:
            continue

        pre_mean = float(np.mean(pre_vals))
        post_mean = float(np.mean(post_vals))
        change_pct = ((post_mean - pre_mean) / (pre_mean + 1e-12)) * 100

        # Wilcoxon rank-sum (independent samples)
        try:
            stat, pval = sp_stats.mannwhitneyu(
                pre_vals, post_vals, alternative="two-sided"
            )
        except ValueError:
            stat, pval = np.nan, np.nan

        # Cohen's d (unequal sample sizes)
        pooled_std = np.sqrt(
            ((len(pre_vals) - 1) * np.var(pre_vals, ddof=1)
             + (len(post_vals) - 1) * np.var(post_vals, ddof=1))
            / (len(pre_vals) + len(post_vals) - 2 + 1e-12)
        )
        cohens_d = (post_mean - pre_mean) / (pooled_std + 1e-12)

        rows.append({
            "Detector": name,
            "Pre Mean": round(pre_mean, 5),
            "Post Mean": round(post_mean, 5),
            "Change %": round(change_pct, 1),
            "Mann-Whitney U": round(float(stat), 2) if not np.isnan(stat) else "N/A",
            "p-value": round(float(pval), 4) if not np.isnan(pval) else "N/A",
            "Cohen d": round(float(cohens_d), 3),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trigger–Anomaly Temporal Coincidence (permutation test)
# ---------------------------------------------------------------------------

def trigger_coincidence_test(
    score_dict: dict[str, np.ndarray],
    epoch_dur: float,
    trigger_time_s: float,
    threshold_pct: float = 95.0,
    windows: list[float] | None = None,
    n_permutations: int = 1000,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Test whether flagged anomalies cluster near the trigger.

    For each detector and each window size, count flagged epochs within
    ±window of the trigger and compare to expected count via permutation.
    Returns a DataFrame with: Detector, Window, Observed, Expected, Fold, p-value.
    """
    if windows is None:
        windows = [5.0, 10.0, 15.0]

    rng = np.random.default_rng(rng_seed)
    n_epochs = len(list(score_dict.values())[0])
    epoch_starts = np.arange(n_epochs) * epoch_dur

    rows = []
    for name, sc in score_dict.items():
        if sc is None or len(sc) == 0 or name == "Ensemble":
            continue
        thresh_val = np.percentile(sc, threshold_pct)
        is_anomaly = sc > thresh_val
        n_flagged = int(np.sum(is_anomaly))
        if n_flagged == 0:
            continue

        for win in windows:
            near_mask = (epoch_starts >= trigger_time_s - win) & (
                epoch_starts <= trigger_time_s + win
            )
            observed = int(np.sum(is_anomaly & near_mask))
            n_near = int(np.sum(near_mask))

            # Expected under uniform distribution of anomalies
            expected = n_flagged * (n_near / n_epochs) if n_epochs > 0 else 0

            # Permutation p-value
            perm_counts = np.zeros(n_permutations)
            for p in range(n_permutations):
                perm_labels = rng.permutation(is_anomaly)
                perm_counts[p] = np.sum(perm_labels & near_mask)
            p_val = float(np.mean(perm_counts >= observed))

            fold = observed / (expected + 1e-12)

            rows.append({
                "Detector": name,
                "Window (±s)": win,
                "Observed": observed,
                "Expected": round(expected, 2),
                "Fold Enrichment": round(fold, 2),
                "p-value (perm)": round(p_val, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spectral Band Power Shift at Trigger
# ---------------------------------------------------------------------------

def compute_trigger_band_shift(
    epochs_obj,
    epoch_dur: float,
    trigger_time_s: float,
    pre_window: float = 30.0,
    post_window: float = 30.0,
) -> dict:
    """Compare band powers pre vs post trigger.

    Returns dict with:
      - ``bands``: list of band names
      - ``pre_powers``: (n_bands,) mean power before trigger
      - ``post_powers``: (n_bands,) mean power after trigger
      - ``p_values``: (n_bands,) paired t-test p-values
      - ``pre_epoch_powers``: (n_pre_epochs, n_bands) for individual data
      - ``post_epoch_powers``: (n_post_epochs, n_bands) for individual data
    """
    from scipy.signal import welch as sp_welch
    from utils.signal_processing import get_safe_bands

    data = epochs_obj.get_data()  # (n_epochs, n_channels, n_times)
    sfreq = epochs_obj.info["sfreq"]
    n_epochs_total = data.shape[0]

    epoch_starts = np.arange(n_epochs_total) * epoch_dur
    pre_mask = (epoch_starts >= trigger_time_s - pre_window) & (epoch_starts < trigger_time_s)
    post_mask = (epoch_starts >= trigger_time_s) & (epoch_starts < trigger_time_s + post_window)

    safe_bands = get_safe_bands(sfreq)
    band_names = list(safe_bands.keys())

    def _band_powers_for_epochs(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.zeros((0, len(band_names)))
        powers = np.zeros((len(idx), len(band_names)))
        for ei, ep_idx in enumerate(idx):
            # Average PSD across channels
            all_pxx = []
            for ch in range(data.shape[1]):
                freqs, pxx = sp_welch(data[ep_idx, ch], fs=sfreq,
                                      nperseg=min(data.shape[2], 256))
                all_pxx.append(pxx)
            avg_pxx = np.mean(all_pxx, axis=0)
            for bi, (_, (fmin, fmax)) in enumerate(safe_bands.items()):
                band_idx = (freqs >= fmin) & (freqs <= fmax)
                powers[ei, bi] = np.mean(avg_pxx[band_idx]) if np.any(band_idx) else 0.0
        return powers

    pre_powers = _band_powers_for_epochs(pre_mask)
    post_powers = _band_powers_for_epochs(post_mask)

    # Mean across epochs
    pre_mean = np.mean(pre_powers, axis=0) if pre_powers.shape[0] > 0 else np.zeros(len(band_names))
    post_mean = np.mean(post_powers, axis=0) if post_powers.shape[0] > 0 else np.zeros(len(band_names))

    # Paired t-test per band (use independent t-test since sample sizes may differ)
    p_values = np.ones(len(band_names))
    for bi in range(len(band_names)):
        if pre_powers.shape[0] >= 2 and post_powers.shape[0] >= 2:
            _, p_values[bi] = sp_stats.mannwhitneyu(
                pre_powers[:, bi], post_powers[:, bi], alternative="two-sided"
            )

    return {
        "bands": band_names,
        "pre_powers": pre_mean,
        "post_powers": post_mean,
        "p_values": p_values,
        "pre_epoch_powers": pre_powers,
        "post_epoch_powers": post_powers,
    }
