"""Feature engineering beyond raw levels.

- pressure_gradients: P_upstream(t) - P_BASEL(t), a geostrophic-wind / flow proxy.
- season_features: day-of-year sin/cos. CONTROLLED ablation only, OFF by default
  (spec v1.1 section 2.3.4) -- in delta space seasonality is weak and adding it
  without an ablation control would muddle the story.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pressure_gradients(
    imputed: pd.DataFrame,
    feature_cols: list[str],
    stations: list[str],
    center: str = "BASEL",
) -> tuple[pd.DataFrame, list[str]]:
    """Return (grad_df, grad_col_names): P_s(t) - P_center(t) for each station
    pair that reports pressure at both ends. Skipped otherwise (pressure is
    missing at DRESDEN/MALMO/SONNBLICK)."""
    cols: dict[str, np.ndarray] = {}
    names: list[str] = []
    for s in stations:
        a, b = f"{center}_pressure", f"{s}_pressure"
        if a in feature_cols and b in feature_cols and s != center:
            names.append(f"{s}-{center}_p")
            cols[names[-1]] = imputed[b].to_numpy() - imputed[a].to_numpy()
    return pd.DataFrame(cols, index=imputed.index), names


def season_features(dates: pd.DatetimeIndex) -> np.ndarray:
    """day-of-year sin/cos encoding, shape (n, 2). Use only as a controlled
    ablation; do not add to the default feature set."""
    doy = np.array([d.timetuple().tm_yday for d in dates])
    ang = 2 * np.pi * doy / 365.25
    return np.stack([np.sin(ang), np.cos(ang)], axis=1)
