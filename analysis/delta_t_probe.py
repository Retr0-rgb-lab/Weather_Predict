"""Feasibility probe for the ΔT_max formulation.

Question: can day-t multi-station observations predict the day-to-day change of
BASEL T_max,  ΔT_max(t+1) = T_max(t+1) - T_max(t),  better than the
zero-change (persistence-in-Δ) baseline?

Protocol:
- Single-variable physical-range violations (cloud_cover/sunshine/global_radiation/
  precipitation/humidity/pressure) are masked to NaN BEFORE differencing (sentinel
  cells such as STOCKHOLM_cloud_cover=-99 would otherwise create ±100 °C fake
  jumps in Δ features). The temp_min>temp_max check (41 cells, EDA §3.2) is NOT
  applied here — the target column BASEL_temp_max is unaffected; it belongs to
  the Wave-2 pipeline (spec §2.4).
- Chronological split only: train 2000-2007, val 2008 (unused here), test
  2009-01-01 .. 2010-01-01. No shuffle anywhere.
- Imputation statistics and the standardizer are fit on train years only.
- This is a feasibility probe: imputation is plain per-column monthly means
  and missing-indicator features are off by default (optional ablation, spec §2.4).
  The strict Wave-2 pipeline (station x month statistics) is separate work.

Run: uv run --no-project --with pandas --with numpy --with scikit-learn python analysis/delta_t_probe.py
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
TARGET_COL = f"{TARGET_STATION}_temp_max"
# westerly upstream stations, ordered by h=1 lead on BASEL anomaly
# (docs/formulation_probe_stations.csv)
UPSTREAM = ["MAASTRICHT", "TOURS", "DUSSELDORF", "MUENCHEN", "KASSEL", "DE_BILT", "HEATHROW"]

df = pd.read_csv(DATA / "weather_prediction_dataset.csv")
dates = pd.to_datetime(df["DATE"], format="%Y%m%d")
feature_cols = [c for c in df.columns if c not in ("DATE", "MONTH")]
values = df[feature_cols].astype(float)
month = dates.dt.month.to_numpy()
n_days = len(values)

# ---------------------------------------------------------------- 1. clean
RANGE_CHECKS = {
    "cloud_cover": (0.0, 8.0),
    "sunshine": (0.0, np.inf),
    "global_radiation": (0.0, np.inf),
    "precipitation": (0.0, np.inf),
    "humidity": (0.0, 1.0),
    "pressure": (0.5, 1.1),
}
n_masked = 0
for var, (lo, hi) in RANGE_CHECKS.items():
    cols_v = [c for c in feature_cols if c.endswith("_" + var)]
    if not cols_v:
        continue
    bad = (values[cols_v] < lo) | (values[cols_v] > hi)
    n_masked += int(bad.sum().sum())
    values.loc[:, cols_v] = values[cols_v].where(~bad)
print(f"[1] masked {n_masked} out-of-range cells to NaN (before differencing)")

# ------------------------------------------------- 2. impute (train-only)
train_full = (dates < np.datetime64("2008-01-01")).to_numpy()
clim = values[train_full].groupby(month[train_full]).mean()
vals = values.copy()
for mm in range(1, 13):
    sel = month == mm
    vals.loc[sel, :] = values.loc[sel, :].fillna(clim.loc[mm])
assert not vals.isna().any().any()
A = vals.to_numpy(dtype=float)                    # 3654 x 163 levels

# ---------------------------------------------------------------- 3. align
# sample t uses levels at day t and deltas over t-1 -> t; target is the
# day-to-day change of BASEL temp_max over t -> t+1. t in [1, n_days-2].
t_idx = np.arange(1, n_days - 1)
tgt_dates = dates.to_numpy()[t_idx + 1]
y = A[t_idx + 1, feature_cols.index(TARGET_COL)] - A[t_idx, feature_cols.index(TARGET_COL)]
levels = A[t_idx]                                  # day-t levels
deltas = A[t_idx] - A[t_idx - 1]                   # day-(t-1)->t changes

tr = tgt_dates < np.datetime64("2008-01-01")
va = (tgt_dates >= np.datetime64("2008-01-01")) & (tgt_dates < np.datetime64("2009-01-01"))
te = tgt_dates >= np.datetime64("2009-01-01")
print(f"[2] samples: train={tr.sum()} val={va.sum()} test={te.sum()}")

tgt_month = pd.DatetimeIndex(tgt_dates).month.to_numpy()


def metrics(yt: np.ndarray, yp: np.ndarray) -> dict[str, float]:
    err = yt - yp
    ss_res = float((err**2).sum())
    ss_tot = float(((yt - yt.mean())**2).sum())
    corr = float(np.corrcoef(yt, yp)[0, 1]) if float(np.std(yp)) > 1e-12 else float("nan")
    return {
        "MAE": round(float(np.abs(err).mean()), 3),
        "RMSE": round(float(np.sqrt((err**2).mean())), 3),
        "R2": round(1 - ss_res / ss_tot, 4),
        "bias": round(float(err.mean()), 3),
        "corr": round(corr, 4) if corr == corr else corr,
    }


rows: list[dict] = []

# ------------------------------------------------------------- 4. baselines
zero_pred = np.zeros_like(y)
yday_pred = deltas[:, feature_cols.index(TARGET_COL)]      # yesterday's ΔT_max
clim_d = pd.Series(y[tr]).groupby(tgt_month[tr]).mean()
clim_pred = pd.Series(tgt_month).map(clim_d).to_numpy(dtype=float)  # per-month train mean of Δ

for name, pred in [("zero_change", zero_pred), ("yesterday_delta", yday_pred), ("monthly_delta_clim", clim_pred)]:
    m = metrics(y[te], pred[te])
    rows.append({"model": name, **m})
    print(f"[baseline] {name:<20} test MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:.3f}  corr {m['corr']:+.3f}")

mae_zero = metrics(y[te], zero_pred[te])["MAE"]

# ---------------------------------------------------------------- 5. ridges
up_tmax = [feature_cols.index(f"{s}_temp_max") for s in UPSTREAM]
own_tmax = feature_cols.index(TARGET_COL)

feature_sets = {
    "levels_all163": levels,
    "lvl+d_all163": np.hstack([levels, deltas]),
    "up7_dtmax": deltas[:, up_tmax],
    "up7_dtmax+own": deltas[:, np.array(up_tmax + [own_tmax])],
    "up7_lvl+dtmax": np.hstack([levels[:, up_tmax], deltas[:, up_tmax]]),
}

for name, X in feature_sets.items():
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    model.fit(X[tr], y[tr])
    m = metrics(y[te], model.predict(X[te]))
    skill = 1 - m["MAE"] / mae_zero
    rows.append({"model": f"ridge_{name}", **m, "skill_vs_zero": round(skill, 4)})
    alpha = model.named_steps["ridgecv"].alpha_
    print(f"[ridge]    {name:<20} test MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  "
          f"R2 {m['R2']:.3f}  corr {m['corr']:+.3f}  skill {skill:+.1%}  alpha={alpha:g}")

# ------------------------------------------- 6. advection table (train corr)
lead_rows = []
for s in STATIONS:
    for fname, arr in [("level", levels[:, feature_cols.index(f'{s}_temp_max')]),
                       ("delta", deltas[:, feature_cols.index(f'{s}_temp_max')])]:
        r = float(np.corrcoef(arr[tr], y[tr])[0, 1])
        lead_rows.append({"station": s, "feature": fname, "corr_with_dT_target_train": round(r, 3)})
lead = pd.DataFrame(lead_rows).sort_values("corr_with_dT_target_train", ascending=False)
print("\n[advection] corr of each station's temp_max level/Δ(t) with ΔT_max(t+1) (train):")
print(lead.head(12).to_string(index=False))

# ---------------------------------------------------------------- 7. dump
out = pd.DataFrame(rows)
out.to_csv(ROOT / "docs" / "delta_t_probe.csv", index=False)
lead.to_csv(ROOT / "docs" / "delta_t_probe_lead.csv", index=False)

print("\nΔT_max target stats (train): mean %.3f  std %.3f  MAE_zero(test) %.3f" % (
    y[tr].mean(), y[tr].std(), mae_zero))
print("wrote docs/delta_t_probe.csv, docs/delta_t_probe_lead.csv")
