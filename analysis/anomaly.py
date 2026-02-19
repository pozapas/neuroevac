"""
anomaly.py — ML/DL anomaly detection on EEG epoch features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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
