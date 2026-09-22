"""Data loading, physical-range cleaning, train-only imputation.

Design notes
- Cleaning runs BEFORE differencing: sentinel cells (e.g. STOCKHOLM_cloud_cover
  = -99 on 20090625, inside the test period) would otherwise create +/-100 C
  fake jumps in delta features.
- Every statistic used for imputation / scaling / climatology must be estimated
  on training years only (dates < 2008-01-01) and then applied to val/test.
- Missing-indicator features are produced but OFF by default (optional ablation,
  spec v1.1 section 2.4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"

STATIONS = [
    "BASEL", "BUDAPEST", "DE_BILT", "DRESDEN", "DUSSELDORF", "HEATHROW", "KASSEL",
    "LJUBLJANA", "MAASTRICHT", "MALMO", "MONTELIMAR", "MUENCHEN", "OSLO",
    "PERPIGNAN", "ROMA", "SONNBLICK", "STOCKHOLM", "TOURS",
]
UPSTREAM = ["MAASTRICHT", "TOURS", "DUSSELDORF", "MUENCHEN", "KASSEL", "DE_BILT", "HEATHROW"]
TARGET_STATION = "BASEL"
TARGET_VAR = "temp_max"

# Physical plausibility windows (see data_analysis.md section 3.2).
RANGE_CHECKS = {
    "cloud_cover": (0.0, 8.0),
    "sunshine": (0.0, np.inf),
    "global_radiation": (0.0, np.inf),
    "precipitation": (0.0, np.inf),
    "humidity": (0.0, 1.0),
    "pressure": (0.5, 1.1),
}


def split_column(col: str) -> tuple[str, str]:
    """Map a raw column name to (station, variable) using longest-prefix match."""
    for s in STATIONS:
        if col.startswith(s + "_"):
            return s, col[len(s) + 1 :]
    raise ValueError(f"unmapped column: {col}")


def load_raw(path: Path | None = None) -> tuple[pd.DataFrame, pd.DatetimeIndex, list[str]]:
    df = pd.read_csv(path or DATA / "weather_prediction_dataset.csv")
    dates = pd.DatetimeIndex(pd.to_datetime(df["DATE"], format="%Y%m%d"))
    feature_cols = [c for c in df.columns if c not in ("DATE", "MONTH")]
    return df, dates, feature_cols


def clean(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Mask physically impossible cells to NaN, BEFORE any differencing.

    - single-variable range checks (RANGE_CHECKS)
    - temp_min > temp_max: both cells masked (41 cells, HEATHROW/ROMA)
    """
    out = df.copy()
    out[feature_cols] = out[feature_cols].astype(float)
    masked: dict = {}
    for var, (lo, hi) in RANGE_CHECKS.items():
        cols_v = [c for c in feature_cols if c.endswith("_" + var)]
        if not cols_v:
            continue
        bad = (out[cols_v] < lo) | (out[cols_v] > hi)
        masked[f"range_{var}"] = int(bad.sum().sum())
        out.loc[:, cols_v] = out[cols_v].where(~bad)
    n_pair = 0
    for s in STATIONS:
        mn, mx = f"{s}_temp_min", f"{s}_temp_max"
        if mn in feature_cols and mx in feature_cols:
            bad = out[mn] > out[mx]
            n_pair += int(bad.sum())
            out.loc[bad, [mn, mx]] = np.nan
    masked["temp_min_gt_temp_max_pairs"] = n_pair
    return out, masked


def impute_station_month(
    values: pd.DataFrame,
    month: np.ndarray,
    train_mask: np.ndarray,
    feature_cols: list[str],
    stat: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute per-column (station x variable) monthly statistics from train only.

    Returns (imputed, missing_indicator) where indicator is 1 where the original
    cell was NaN. The indicator is OFF by default in experiments (spec v1.1).
    """
    grouped = values[train_mask].groupby(month[train_mask])
    stats = grouped.median() if stat == "median" else grouped.mean()
    ind = values[feature_cols].isna().astype(float)
    imp = values[feature_cols].copy()
    for mm in range(1, 13):
        sel = month == mm
        imp.loc[sel, :] = values.loc[sel, feature_cols].fillna(stats.loc[mm])
    return imp, ind
