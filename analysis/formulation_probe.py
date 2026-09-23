"""Probe the space-time information structure, to choose the problem formulation.

Question: for a target station and a lead time h, how much information about
y(t+h) sits in (a) the target station's own past (temporal memory) versus
(b) the other 17 stations' observations (spatial memory)?

Anomalies are computed against a climatology fitted on 2000-2007 only, so no
test-period information leaks into the features or the targets.

Run:  "/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/formulation_probe.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

STATIONS = [
    "BASEL", "BUDAPEST", "DE_BILT", "DRESDEN", "DUSSELDORF", "HEATHROW", "KASSEL",
    "LJUBLJANA", "MAASTRICHT", "MALMO", "MONTELIMAR", "MUENCHEN", "OSLO",
    "PERPIGNAN", "ROMA", "SONNBLICK", "STOCKHOLM", "TOURS",
]
TARGET_STATION = "BASEL"
TARGET_VAR = "temp_mean"
TARGET_COL = f"{TARGET_STATION}_{TARGET_VAR}"
HORIZONS = [1, 2, 3, 5, 7, 10]

df = pd.read_csv(DATA / "weather_prediction_dataset.csv")
dates = pd.to_datetime(df["DATE"], format="%Y%m%d")
feature_cols = [c for c in df.columns if c not in ("DATE", "MONTH")]
values = df[feature_cols].astype(float)
month = dates.dt.month.to_numpy()

# --- deseasonalise on training years only -------------------------------------
train_years = (dates < np.datetime64("2008-01-01")).to_numpy()
clim = values[train_years].groupby(month[train_years]).mean()
anom = values.copy()
for mm in range(1, 13):
    sel = month == mm
    anom.loc[sel, :] = values.loc[sel, :].to_numpy() - clim.loc[mm].to_numpy()
A = anom.to_numpy()
y_idx = feature_cols.index(TARGET_COL)

# feature groups for the ablation
own_cols = [i for i, c in enumerate(feature_cols) if c.startswith(TARGET_STATION + "_")]
other_own = [i for i, c in enumerate(feature_cols)
             if any(c.startswith(s + "_") for s in STATIONS if s != TARGET_STATION)]
others_temp_mean = [i for i, c in enumerate(feature_cols)
                    if c.endswith("_temp_mean") and i != y_idx]

target_dates = dates.to_numpy()

print(f"target = {TARGET_COL}(t+h),  anomalies vs train-year climatology\n")
header = (f"{'h':>3} {'n_test':>7} {'clim':>7} {'persist':>8} {'own':>7} {'others':>7} "
          f"{'oth+d':>7} {'all':>7} {'no-own':>7}   {'vs persist':>11} {'vs clim':>9}")
print(header)
print("-" * len(header))

rows = []
for h in HORIZONS:
    n = len(A) - h
    src = A[:n]                      # features at day t
    prev = np.vstack([A[:1], A[:n - 1]]) if n > 1 else A[:n]   # features at day t-1
    delta = src - prev               # day-to-day change of every feature
    tgt = A[h:, y_idx]               # target at day t+h
    tgt_day = target_dates[h:]       # date of the target row

    te = tgt_day >= np.datetime64("2009-01-01")
    tr = tgt_day < np.datetime64("2008-01-01")
    Xtr, Xte = src[tr], src[te]
    ytr, yte = tgt[tr], tgt[te]

    def rmse(pred, truth=yte):
        return float(np.sqrt(np.mean((truth - pred) ** 2)))

    def fit_rmse(cols, source=src, extra=None):
        """RidgeCV on `cols` of `source` (optionally stacked with `extra`), scored on the test targets."""
        def take(mask):
            blocks = [source[mask][:, cols]]
            if extra is not None:
                blocks.append(extra[mask][:, cols])
            return np.hstack(blocks)
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
        model.fit(take(tr), ytr)
        return rmse(model.predict(take(te)))

    results = {"clim": rmse(np.zeros_like(yte)),
               "persist": rmse(Xte[:, y_idx])}

    # Own station: its own 9 variables today.
    results["own"] = fit_rmse(own_cols)
    # Spatial level only: the other 17 stations' temp_mean today.
    results["others"] = fit_rmse(others_temp_mean)
    # Spatial level + change (advection proxy): same 17, today plus yesterday.
    results["oth+d"] = fit_rmse(others_temp_mean, extra=delta)
    # Everything, and everything except the target station's own variables.
    results["all"] = fit_rmse(list(range(len(feature_cols))))
    results["no-own"] = fit_rmse([i for i in range(len(feature_cols)) if i not in own_cols])

    skill = 1 - results["others"] / results["persist"]
    skill_clim = 1 - results["others"] / results["clim"]
    rows.append({"h": h, "n_test": int(te.sum()),
                 "skill_spatial_vs_persist": round(skill, 4),
                 "skill_spatial_vs_climatology": round(skill_clim, 4),
                 **{k: round(v, 3) for k, v in results.items()}})
    print(f"{h:>3} {int(te.sum()):>7} {results['clim']:>7.3f} {results['persist']:>8.3f} "
          f"{results['own']:>7.3f} {results['others']:>7.3f} {results['oth+d']:>7.3f} "
          f"{results['all']:>7.3f} {results['no-own']:>7.3f}   "
          f"{skill:>+10.1%} {skill_clim:>+8.1%}")

print("\nRMSE in degrees C, anomaly space, test targets from 2009-01-01 onwards (2008 held out).")
print("clim      = predict the seasonal mean (0 anomaly)")
print("persist   = predict today's anomaly as tomorrow's  (the bar to beat)")
print("own       = ridge on the target station's own 9 variables today")
print("others    = ridge on the other 17 stations' temp_mean today")
print("oth+d     = same 17 stations, today AND yesterday  (level + advection proxy)")
print("all       = ridge on all 163 features today")
print("no-own    = all 163 minus the target station's own variables")
print("skill     = 1 - RMSE(others)/RMSE(persist)")

# --- which stations carry the information, and at what lead? -------------------
print("\nPer-station single-feature correlation with BASEL_temp_mean(t+h) on the training years:")
print(f"{'station':<15}" + "".join(f"h={h:<7}" for h in HORIZONS))
station_table: dict[str, dict[str, float]] = {}
for s in STATIONS:
    col = feature_cols.index(f"{s}_temp_mean")
    cells = []
    for h in HORIZONS:
        n = len(A) - h
        src = A[:n, col]
        tgt = A[h:, y_idx]
        tgt_day = target_dates[h:]
        tr = (tgt_day < np.datetime64("2008-01-01")) & (tgt_day >= np.datetime64("2000-02-01"))
        cells.append(float(np.corrcoef(src[tr], tgt[tr])[0, 1]))
    station_table[s] = dict(zip([f"h={h}" for h in HORIZONS], [round(c, 3) for c in cells]))
    mark = "  <-- target" if s == TARGET_STATION else ""
    print(f"{s:<15}" + "".join(f"{c:>+8.3f} " for c in cells) + mark)

pd.DataFrame(rows).to_csv(ROOT / "docs" / "formulation_probe_horizons.csv", index=False)
pd.DataFrame(station_table).T.to_csv(ROOT / "docs" / "formulation_probe_stations.csv")
print("\nwrote docs/formulation_probe_horizons.csv, docs/formulation_probe_stations.csv")
