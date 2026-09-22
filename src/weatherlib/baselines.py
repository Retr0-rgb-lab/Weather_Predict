"""Reference baselines in delta space.

zero             : predict 0 -> this IS persistence-in-delta (the bar to beat)
yesterday_delta  : predict today's change for tomorrow (expected to lose to
                   zero; demonstrates the ~0.075 lag-1 autocorr of daily change)
monthly_delta_clim : per-calendar-month mean of the change, fit on train only
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zero(y: np.ndarray) -> np.ndarray:
    return np.zeros_like(y)


def yesterday_delta(delta_target: np.ndarray) -> np.ndarray:
    """delta_target = day-(t-1)->t change of the target variable at day t."""
    return delta_target.copy()


def monthly_delta_clim(
    y: np.ndarray, tgt_month: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    clim = pd.Series(y[train_mask]).groupby(tgt_month[train_mask]).mean()
    return pd.Series(tgt_month).map(clim).to_numpy(dtype=float)
