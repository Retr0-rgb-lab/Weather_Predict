"""Evaluation metrics and moving-block-bootstrap confidence intervals.

Primary metric is MAE in degrees C (heavy-tailed target -> robust).
skill = 1 - score_model / score_reference (e.g. zero-change).
corr is undefined for (near-)constant predictions -> NaN.

Block bootstrap (spec v1.1 section 2.2.4): moving blocks of 7 days to respect
weather-scale autocorrelation (pressure memory ~3-5 days). Differences between
two models are evaluated on the SAME resampled blocks (paired).
"""

from __future__ import annotations

import numpy as np


def mae(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.mean(np.abs(yt - yp)))


def rmse(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1 - ss_res / ss_tot


def corr_safe(yt: np.ndarray, yp: np.ndarray, eps: float = 1e-12) -> float:
    if float(np.std(yp)) <= eps:
        return float("nan")
    return float(np.corrcoef(yt, yp)[0, 1])


def skill(score_model: float, score_ref: float) -> float:
    return 1 - score_model / score_ref


def _resample_blocks(n: int, block: int, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    """Return (n_boot, n) bootstrap index arrays from moving overlapping blocks."""
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = np.arange(n)
    out = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        out[b] = np.concatenate([idx[s : s + block] for s in starts[b]])[:n]
    return out


def block_bootstrap_ci(
    yt: np.ndarray,
    yp: np.ndarray,
    metric=mae,
    n_boot: int = 1000,
    block: int = 7,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile 95% CI of `metric` under a moving-block bootstrap."""
    n = len(yt)
    rng = np.random.default_rng(seed)
    idxs = _resample_blocks(n, block, rng, n_boot)
    vals = np.array([metric(yt[s], yp[s]) for s in idxs])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def block_bootstrap_diff_ci(
    yt: np.ndarray,
    yp1: np.ndarray,
    yp2: np.ndarray,
    metric=mae,
    n_boot: int = 1000,
    block: int = 7,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired 95% CI of metric(yp1) - metric(yp2) on identical blocks."""
    n = len(yt)
    rng = np.random.default_rng(seed)
    idxs = _resample_blocks(n, block, rng, n_boot)
    d = np.array([metric(yt[s], yp1[s]) - metric(yt[s], yp2[s]) for s in idxs])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
