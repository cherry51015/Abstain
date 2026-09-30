"""Probabilistic-forecast metrics shared by training, eval and live monitoring."""
from __future__ import annotations

import numpy as np


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(mask.sum()),
                         "mean_pred": float(p[mask].mean()), "observed": float(y[mask].mean())})
    return bins


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error: bin-weighted |predicted - observed|."""
    return float(sum(b["n"] * abs(b["mean_pred"] - b["observed"]) for b in reliability_bins(p, y, n_bins)) / len(p))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann-Whitney U), ties averaged."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):  # average ranks over ties
        tie = p == v
        ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
