"""Delta-T_max dataset builder and chronological split.

Alignment (see spec v1.1 section 2.1):
    sample t (t in [1, n-2]):
      levels[t] = cleaned+imputed values at day t            (n x F)
      deltas[t] = A[t] - A[t-1]                              (n x F)
      grads[t]  = pressure gradients at day t                (n x G, or empty)
      y[t]      = T_max(t+1) - T_max(t) at target station    (scalar)
      target date = dates[t+1]

Masks train/val/test are defined on the TARGET date:
    train:  < 2008-01-01
    val:    2008-01-01 .. 2008-12-31
    test:   >= 2009-01-01          (2009-01-01 .. 2010-01-01, 366 days)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import pressure_gradients


@dataclass
class DeltaMaxData:
    X: dict[str, np.ndarray]          # feature blocks: levels, deltas, grads
    y: np.ndarray                     # (n,) target day-over-day change
    dates: np.ndarray                 # (n,) target dates (datetime64)
    tgt_month: np.ndarray             # (n,) month of target date
    tr: np.ndarray                    # bool masks
    va: np.ndarray
    te: np.ndarray
    feature_cols: list[str]           # order of the 163 columns in levels/deltas
    target_col: str
    up_cols: list[int]                # indices into feature_cols: upstream temp_max
    grad_cols: list[str]              # names of gradient features (may be empty)


def make_delta_max_dataset(
    imputed: pd.DataFrame,
    dates: pd.DatetimeIndex,
    feature_cols: list[str],
    stations: list[str],
    upstream: list[str],
    target_station: str = "BASEL",
    target_var: str = "temp_max",
) -> DeltaMaxData:
    A = imputed[feature_cols].to_numpy(dtype=float)
    j = feature_cols.index(f"{target_station}_{target_var}")
    n = len(A)
    t = np.arange(1, n - 1)

    levels = A[t]
    deltas = A[t] - A[t - 1]
    y = A[t + 1, j] - A[t, j]

    tgt_dates = dates.to_numpy()[t + 1]
    tgt_month = pd.DatetimeIndex(tgt_dates).month.to_numpy()

    grads_df, grad_cols = pressure_gradients(
        imputed=pd.DataFrame(A, columns=feature_cols),
        feature_cols=feature_cols,
        stations=stations,
        center=target_station,
    )
    grads = grads_df.to_numpy(dtype=float)[t]

    tr = tgt_dates < np.datetime64("2008-01-01")
    va = (tgt_dates >= np.datetime64("2008-01-01")) & (tgt_dates < np.datetime64("2009-01-01"))
    te = tgt_dates >= np.datetime64("2009-01-01")

    up_cols = [feature_cols.index(f"{s}_{target_var}") for s in upstream]

    return DeltaMaxData(
        X={"levels": levels, "deltas": deltas, "grads": grads},
        y=y,
        dates=tgt_dates,
        tgt_month=tgt_month,
        tr=tr, va=va, te=te,
        feature_cols=feature_cols,
        target_col=f"{target_station}_{target_var}",
        up_cols=up_cols,
        grad_cols=grad_cols,
    )
