"""Metric suite. Regression: rmse / gaussian NLL / 90% coverage.
Classification: accuracy / log-loss / ECE (10 equal-width bins)."""

from __future__ import annotations

import numpy as np

from gaussforge.core.errors import DataError

Z90 = 1.6448536269514722


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise DataError("rmse: shape mismatch")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def gaussian_nll(y: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    """Mean negative log-likelihood of N(mean, var) at y."""
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    var = np.asarray(var, dtype=np.float64).ravel()
    if y.shape != mean.shape or y.shape != var.shape:
        raise DataError("gaussian_nll: shape mismatch")
    if np.any(var <= 0):
        raise DataError("gaussian_nll: variance must be positive")
    return float(
        np.mean(0.5 * np.log(2.0 * np.pi * var) + (y - mean) ** 2 / (2.0 * var))
    )


def coverage90(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).ravel()
    mean = np.asarray(mean, dtype=np.float64).ravel()
    std = np.asarray(std, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y - mean) <= Z90 * std))


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    pred = (np.asarray(proba).ravel() >= 0.5).astype(int)
    return float(np.mean(pred == y_true))


def log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-9) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.clip(np.asarray(proba, dtype=np.float64).ravel(), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def ece(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error for binary probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(proba, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(p)
    err = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(p[mask]))
        frac_pos = float(np.mean(y_true[mask]))
        err += (np.sum(mask) / total) * abs(conf - frac_pos)
    return float(err)
