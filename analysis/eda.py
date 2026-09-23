"""Exploratory data analysis for the weather prediction dataset.

Generates every number and figure quoted in `docs/data_analysis.md`.
Run:  "/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

STATIONS = [
    "BASEL", "BUDAPEST", "DE_BILT", "DRESDEN", "DUSSELDORF", "HEATHROW", "KASSEL",
    "LJUBLJANA", "MAASTRICHT", "MALMO", "MONTELIMAR", "MUENCHEN", "OSLO",
    "PERPIGNAN", "ROMA", "SONNBLICK", "STOCKHOLM", "TOURS",
]

# Variable taxonomy by physical process (see docs/data_analysis.md section 2).
VARIABLE_GROUPS = {
    "Thermal": ["temp_mean", "temp_min", "temp_max"],
    "Moisture": ["humidity", "precipitation"],
    "Radiation / cloud": ["global_radiation", "sunshine", "cloud_cover"],
    "Dynamic / pressure": ["pressure", "wind_speed", "wind_gust"],
}
VAR_ORDER = [v for vs in VARIABLE_GROUPS.values() for v in vs]

# Physical units, as documented in data/metadata.txt. `scale` is the multiplier
# needed to convert a raw column value into the stated native unit.
UNITS = {
    "temp_mean": ("°C", 1.0, "mean daily temperature"),
    "temp_min": ("°C", 1.0, "minimum daily temperature"),
    "temp_max": ("°C", 1.0, "maximum daily temperature"),
    "humidity": ("fraction", 1.0, "relative humidity as fraction of 1"),
    "precipitation": ("10 mm", 1.0, "daily precipitation"),
    "global_radiation": ("100 W/m2", 1.0, "global radiation"),
    "sunshine": ("0.1 h", 1.0, "sunshine duration"),
    "cloud_cover": ("oktas", 1.0, "cloud cover, 0-8 eighths"),
    "pressure": ("1000 hPa", 1.0, "sea level pressure"),
    "wind_speed": ("m/s", 1.0, "wind speed"),
    "wind_gust": ("m/s", 1.0, "wind gust"),
}

out: dict = {}


def split_column(col: str) -> tuple[str, str]:
    """Map a raw column name to (station, variable) using longest-prefix match."""
    for s in STATIONS:
        if col.startswith(s + "_"):
            return s, col[len(s) + 1 :]
    raise ValueError(f"unmapped column: {col}")


def savefig(fig, name: str) -> None:
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- #
# 1. Load and reshape                                                            #
# --------------------------------------------------------------------------- #
print("[1] load")
main = pd.read_csv(DATA / "weather_prediction_dataset.csv")
picnic = pd.read_csv(DATA / "weather_prediction_picnic_labels.csv")

dates = pd.to_datetime(main["DATE"], format="%Y%m%d")
feature_cols = [c for c in main.columns if c not in ("DATE", "MONTH")]
pairs = [split_column(c) for c in feature_cols]
meta = pd.DataFrame(pairs, columns=["station", "variable"], index=feature_cols)

out["shape"] = {"rows": len(main), "cols": len(main.columns), "features": len(feature_cols)}
out["date"] = {
    "first": str(dates.min().date()),
    "last": str(dates.max().date()),
    "n_days": len(dates),
    "span_days": (dates.max() - dates.min()).days + 1,
    "continuous": bool(
        pd.date_range(dates.min(), dates.max(), freq="D").equals(pd.DatetimeIndex(dates))
    ),
    "duplicate_dates": int(dates.duplicated().sum()),
    "month_col_consistent": bool((main["MONTH"] == dates.dt.month).all()),
    "leap_years": sorted({int(y) for y in dates.dt.year.unique() if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0}),
}

# --------------------------------------------------------------------------- #
# 2. Feature taxonomy                                                            #
# --------------------------------------------------------------------------- #
print("[2] taxonomy")
cov = pd.crosstab(meta["variable"], meta["station"]).reindex(index=VAR_ORDER, columns=STATIONS)
out["variable_counts"] = cov.sum(axis=1).astype(int).to_dict()
out["station_counts"] = cov.sum(axis=0).astype(int).to_dict()
out["group_counts"] = {
    g: int(sum(out["variable_counts"].get(v, 0) for v in vs)) for g, vs in VARIABLE_GROUPS.items()
}
out["station_variable_table"] = {
    s: sorted(meta.loc[meta["station"] == s, "variable"].tolist()) for s in STATIONS
}
out["absent_pairs"] = [
    {"station": s, "variable": v}
    for v in VAR_ORDER
    for s in STATIONS
    if cov.loc[v, s] == 0
]

# Figure 1: station x variable coverage heatmap.
fig, ax = plt.subplots(figsize=(11, 4.2))
ax.imshow(cov.to_numpy(), cmap=ListedColormap(["#f2f2f2", "#2b6cb0"]), vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(STATIONS)), STATIONS, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(VAR_ORDER)), VAR_ORDER, fontsize=8)
for i in range(len(VAR_ORDER)):
    for j in range(len(STATIONS)):
        if cov.iloc[i, j]:
            ax.text(j, i, "✓", ha="center", va="center", color="white", fontsize=9)
ax.set_title("Variable availability per station (163 features, 18 stations)", fontsize=10)
ax.set_xlabel("station")
ax.set_ylabel("variable")
savefig(fig, "fig01_coverage_heatmap.png")

# Figure 2: counts per variable and per station.
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
cnt_v = cov.sum(axis=1).reindex(VAR_ORDER)
axes[0].barh(range(len(cnt_v)), cnt_v.to_numpy(), color="#2b6cb0")
axes[0].set_yticks(range(len(cnt_v)), cnt_v.index, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("number of stations reporting the variable")
axes[0].set_xlim(0, 18)
for i, v in enumerate(cnt_v.to_numpy()):
    axes[0].text(v + 0.3, i, str(v), va="center", fontsize=8)
cnt_s = cov.sum(axis=0).reindex(STATIONS)
axes[1].bar(range(len(cnt_s)), cnt_s.to_numpy(), color="#c05621")
axes[1].set_xticks(range(len(cnt_s)), cnt_s.index, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("number of variables")
axes[1].set_ylim(0, 12)
for i, v in enumerate(cnt_s.to_numpy()):
    axes[1].text(i, v + 0.15, str(v), ha="center", fontsize=8)
axes[1].set_title("features per station", fontsize=10)
axes[0].set_title("stations per variable", fontsize=10)
fig.tight_layout()
savefig(fig, "fig02_feature_counts.png")

# --------------------------------------------------------------------------- #
# 3. Data quality                                                                #
# --------------------------------------------------------------------------- #
print("[3] quality")
values = main[feature_cols]
# exact duplicate columns: different names holding identical values
dup_groups: dict[bytes, list[str]] = {}
for c in feature_cols:
    dup_groups.setdefault(values[c].to_numpy().tobytes(), []).append(c)

out["quality"] = {
    "cells": int(values.size),
    "nan_cells": int(values.isna().sum().sum()),
    "sentinel_-9999_cells": int((values == -9999).sum().sum()),
    "zero_variance_cols": [c for c in feature_cols if values[c].nunique() == 1],
    "identical_column_groups": [g for g in dup_groups.values() if len(g) > 1],
}


def count_if(cond: pd.DataFrame) -> int:
    """Count True cells in a boolean frame (safe for zero-column selections)."""
    return int(cond.sum().sum()) if cond.shape[1] else 0


def cols(suffix) -> list[str]:
    suffixes = (suffix,) if isinstance(suffix, str) else suffix
    return [c for c in feature_cols if c.endswith(suffixes)]


# physical plausibility checks
minmax = [(f"{s}_temp_min", f"{s}_temp_max") for s in STATIONS
          if f"{s}_temp_min" in feature_cols and f"{s}_temp_max" in feature_cols]
out["sanity"] = {
    "temp_min_gt_temp_max_cells": int(sum((values[a] > values[b]).sum() for a, b in minmax)),
    "humidity_out_of_0_1": count_if((values[cols("humidity")] < 0) | (values[cols("humidity")] > 1)),
    "cloud_cover_out_of_0_8": count_if((values[cols("cloud_cover")] < 0) | (values[cols("cloud_cover")] > 8)),
    "negative_precipitation": count_if(values[cols("precipitation")] < 0),
    "negative_radiation_or_sunshine": count_if(values[cols(("global_radiation", "sunshine"))] < 0),
}

# describe table
desc = values.describe().T
desc["skew"] = values.skew()
desc["kurtosis"] = values.kurtosis()
desc["range"] = desc["max"] - desc["min"]
desc["cv"] = desc["std"] / desc["mean"].replace(0, np.nan)
out["describe_summary"] = desc[["mean", "std", "min", "max", "range", "skew", "kurtosis"]].round(3).to_dict("index")

# scale heterogeneity: range of per-variable raw scale
per_var = desc.join(meta)
scale_tbl = per_var.groupby("variable").agg(
    n=("count", "size"), min=("min", "min"), max=("max", "max"),
    mean=("mean", "mean"), std=("std", "mean"),
).reindex(VAR_ORDER)
scale_tbl["unit"] = [UNITS[v][0] for v in scale_tbl.index]
out["scale_table"] = scale_tbl.round(4).to_dict("index")

# Physical range violations: cells that cannot be a real observation.
RANGE_CHECKS = {
    "cloud_cover": (0.0, 8.0),        # oktas
    "sunshine": (0.0, np.inf),        # hours, cannot be negative
    "global_radiation": (0.0, np.inf),
    "precipitation": (0.0, np.inf),
    "humidity": (0.0, 1.0),           # fraction
    "pressure": (0.5, 1.1),           # 1000 hPa; no surface record is near zero
}
violations = []
for var, (lo, hi) in RANGE_CHECKS.items():
    sel = values[cols(var)]
    if not sel.shape[1]:
        continue
    bad = sel.where((sel < lo) | (sel > hi))
    for c in bad.columns:
        idx = bad[c].dropna()
        if len(idx):
            violations.append({
                "column": c, "n": int(len(idx)), "values": sorted({round(float(x), 4) for x in idx}),
                "dates": main.loc[idx.index, "DATE"].astype(str).tolist(),
            })
for a, b in minmax:
    bad = values[a] > values[b]
    if bad.any():
        violations.append({"column": f"{a} > {b}", "n": int(bad.sum()),
                           "values": sorted({round(float(x), 2) for x in (values[b] - values[a])[bad]}),
                           "dates": main.loc[bad[bad].index, "DATE"].astype(str).tolist()})
out["range_violations"] = violations
out["range_violation_cells"] = int(sum(v["n"] for v in violations))
out["range_violation_frac"] = round(sum(v["n"] for v in violations) / values.size, 6)

# Figure 3: (a) raw scale heterogeneity, (b) the offending cells in context.
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5), gridspec_kw={"width_ratios": [1.15, 1]})
data = [values[[c for c in feature_cols if c.endswith("_" + v)]].to_numpy().ravel() for v in VAR_ORDER]
bp = axes[0].boxplot(data, tick_labels=VAR_ORDER, showfliers=True, patch_artist=True,
                     flierprops=dict(marker=".", markersize=2, alpha=0.4))
for patch in bp["boxes"]:
    patch.set_facecolor("#bee3f8")
axes[0].set_yscale("symlog", linthresh=1)
axes[0].set_ylabel("raw column value (symlog scale)")
axes[0].set_title("(a) Raw ranges per variable — units are NOT standardised", fontsize=10)
axes[0].tick_params(axis="x", rotation=30, labelsize=8)
for lbl in axes[0].get_xticklabels():
    lbl.set_ha("right")
axes[0].grid(axis="y", alpha=0.3)

sp = pd.Series(main["STOCKHOLM_pressure"].to_numpy(), index=dates).sort_index()
win = (sp.index >= "2007-01-01") & (sp.index <= "2007-12-31")
axes[1].plot(sp.index[win], sp[win], lw=0.9, color="#2b6cb0", label="STOCKHOLM_pressure")
badmask = win & (sp < 0.5)
axes[1].scatter(sp.index[badmask], sp[badmask], color="#c53030", zorder=5, s=28,
                label="physically impossible (−0.099 in 1000 hPa)")
axes[1].set_yscale("symlog", linthresh=0.5)
axes[1].set_ylabel("value (1000 hPa)")
axes[1].set_title("(b) Leftover sentinel-like cells survive in the release", fontsize=10)
axes[1].legend(fontsize=7, loc="lower left")
axes[1].grid(alpha=0.3)
axes[1].tick_params(axis="x", rotation=20, labelsize=8)
fig.tight_layout()
savefig(fig, "fig03_value_ranges.png")

# --------------------------------------------------------------------------- #
# 4. Temporal structure                                                          #
# --------------------------------------------------------------------------- #
print("[4] temporal")
ts = pd.concat([main[feature_cols], pd.Series(dates.to_numpy(), index=main.index, name="date")], axis=1).set_index("date")
out["acf"] = {}
for col in ["BASEL_temp_mean", "BASEL_global_radiation", "BASEL_precipitation", "BASEL_pressure"]:
    x = ts[col].to_numpy(dtype=float)
    x = x - x.mean()
    denom = (x ** 2).sum()
    acf = [1.0] + [(x[:-k] * x[k:]).sum() / denom for k in range(1, 41)]
    out["acf"][col] = np.round(acf, 3).tolist()

# Figure 4: ACF
fig, ax = plt.subplots(figsize=(9, 3.6))
for col in ["BASEL_temp_mean", "BASEL_global_radiation", "BASEL_precipitation", "BASEL_pressure"]:
    ax.plot(range(41), out["acf"][col], marker="o", markersize=3, label=col)
ax.axvline(1, color="grey", ls="--", lw=1)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("lag (days)")
ax.set_ylabel("autocorrelation")
ax.set_title("Autocorrelation of selected BASEL series — the forecastability floor", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
savefig(fig, "fig04_autocorrelation.png")

# monthly climatology
monthly = ts.groupby(ts.index.month).mean()
monthly_std = ts.groupby(ts.index.month).std()
out["monthly_climatology"] = monthly[
    ["BASEL_temp_mean", "BASEL_global_radiation", "BASEL_precipitation", "BASEL_humidity"]
].round(3).to_dict("index")

# Figure 5: seasonality, 2 panels
sel = ["BASEL_temp_mean", "OSLO_temp_mean", "ROMA_temp_mean", "BASEL_global_radiation",
       "BASEL_sunshine", "BASEL_cloud_cover"]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, cols, title, unit in [
    (axes[0], ["BASEL_temp_mean", "OSLO_temp_mean", "ROMA_temp_mean"], "mean temperature", "°C"),
    (axes[1], ["BASEL_global_radiation"], "global radiation (BASEL)", "100 W/m2"),
    (axes[2], ["BASEL_sunshine", "BASEL_cloud_cover"], "sunshine & cloud cover (BASEL)", "h / oktas"),
]:
    for c in cols:
        ax.plot(monthly.index, monthly[c], marker="o", markersize=3, label=c)
        ax.fill_between(monthly.index, monthly[c] - monthly_std[c], monthly[c] + monthly_std[c], alpha=0.15)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month")
    ax.set_ylabel(unit)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "fig05_seasonality.png")

# inter-annual variability of annual mean temperature
annual = ts["BASEL_temp_mean"].resample("YE").mean()
out["annual_mean_basel_temp"] = {str(k.year): round(float(v), 3) for k, v in annual.items()}

# --------------------------------------------------------------------------- #
# 5. Cross-feature correlation                                                  #
# --------------------------------------------------------------------------- #
print("[5] correlation")
corr = values.corr()
corr_arr = corr.to_numpy().copy()
iu = np.triu_indices_from(corr_arr, k=1)
absr = np.abs(corr_arr[iu])
out["pair_correlation"] = {
    "n_pairs": int(len(absr)),
    "mean_abs_r": round(float(absr.mean()), 3),
    "median_abs_r": round(float(np.median(absr)), 3),
    "frac_abs_r_gt_0.8": round(float((absr > 0.8).mean()), 3),
    "frac_abs_r_gt_0.9": round(float((absr > 0.9).mean()), 3),
    "frac_abs_r_gt_0.95": round(float((absr > 0.95).mean()), 3),
    "max_abs_r": round(float(absr.max()), 3),
}
# count of near-duplicate pairs
mask = np.abs(corr_arr) > 0.95
np.fill_diagonal(mask, False)
out["pair_correlation"]["cols_in_gt0.95_pairs"] = int(mask.any(axis=1).sum())

# Figure 6: correlation among temp_mean across stations
tm = [f"{s}_temp_mean" for s in STATIONS if f"{s}_temp_mean" in feature_cols]
ctm = values[tm].corr()
short = [c[: -len("_temp_mean")].replace("_", " ") for c in tm]
fig, ax = plt.subplots(figsize=(7.6, 6.4))
im = ax.imshow(ctm.to_numpy(), cmap="RdYlBu_r", vmin=0.4, vmax=1)
ax.set_xticks(range(len(tm)), short, rotation=90, fontsize=7)
ax.set_yticks(range(len(tm)), short, fontsize=7)
ax.set_title("Correlation of temp_mean across stations (spatial coherence)", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
savefig(fig, "fig06_tempmean_corr.png")

# --------------------------------------------------------------------------- #
# 6. Target construction and baselines                                          #
# --------------------------------------------------------------------------- #
print("[6] baselines")
TARGET, SOURCE = "BASEL_temp_mean", "BASEL_temp_mean"
y_all = values[TARGET].to_numpy(dtype=float)[1:]          # t+1
X_persist = values[SOURCE].to_numpy(dtype=float)[:-1]    # t
days = dates.to_numpy()[1:]

train = days < np.datetime64("2008-01-01")
val = (days >= np.datetime64("2008-01-01")) & (days < np.datetime64("2009-01-01"))
test = days >= np.datetime64("2009-01-01")
out["split"] = {k: int(v.sum()) for k, v in [("train", train), ("val", val), ("test", test)]}


def metrics(yt, yp):
    err = yt - yp
    ss_res = float((err ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return {
        "MAE": round(float(np.abs(err).mean()), 3),
        "RMSE": round(float(np.sqrt((err ** 2).mean())), 3),
        "R2": round(1 - ss_res / ss_tot, 4),
        "bias": round(float(err.mean()), 3),
    }


# baseline 1: persistence
# baseline 2: train climatology (per calendar day-of-year, smoothed by month)
month_of_day = pd.DatetimeIndex(days).month
clim = pd.Series(X_persist[train]).groupby(month_of_day[train]).mean()
y_clim = clim.reindex(month_of_day).to_numpy()

out["baselines"] = {}
for name, i in [("train", train), ("val", val), ("test", test)]:
    out["baselines"][name] = {
        "persistence_y(t)->y(t+1)": metrics(y_all[i], X_persist[i]),
        "train_monthly_climatology": metrics(y_all[i], y_clim[i]),
    }

# how strong is the trivial signal?
r_persist_test = float(np.corrcoef(y_all[test], X_persist[test])[0, 1])
out["persistence_test_corr"] = round(r_persist_test, 4)

# --- raw correlation with the target (inflated by the shared seasonal cycle) ---
target_corr = values.iloc[:-1].corrwith(pd.Series(y_all, index=values.index[:-1]), axis=0).sort_values(ascending=False)
out["top_target_corr"] = target_corr.head(15).round(3).to_dict()
out["bottom_target_corr"] = target_corr.tail(5).round(3).to_dict()

# --- deseasonalised analysis: what is left once the annual cycle is removed? ---
# Climatology is estimated on training years only, so the val/test periods stay clean.
month_full = dates.dt.month.to_numpy()
train_full = (dates < np.datetime64("2008-01-01")).to_numpy()
clim_feat = values[train_full].groupby(month_full[train_full]).mean()
anom = values.astype(float).copy()
for mm in range(1, 13):
    sel = month_full == mm
    anom.loc[sel, :] = values.loc[sel, :].to_numpy() - clim_feat.loc[mm].to_numpy()

season_var_share = 1 - float(anom[TARGET].var() / values[TARGET].var())
out["seasonality"] = {
    "variance_share_explained_by_monthly_cycle": round(season_var_share, 4),
    "anomaly_std_C": round(float(np.sqrt(anom[TARGET].var())), 3),
    "anomaly_lag1_autocorr": round(float(anom[TARGET].iloc[1:].corr(anom[TARGET].shift(1).iloc[1:])), 3),
    "daily_change_lag1_autocorr": round(float(values[TARGET].diff().iloc[2:].corr(values[TARGET].diff().shift(1).iloc[2:])), 3),
}


def lead_corr(frame: pd.DataFrame, target_series: pd.Series, mask: np.ndarray) -> pd.Series:
    """Correlate feature values at t against the target at t+1, on `mask` rows."""
    x = frame.iloc[:-1]
    y = target_series.iloc[:-1]
    return x[mask[:-1]].corrwith(y[mask[:-1]]).dropna().sort_values(ascending=False)


anom_corr = lead_corr(anom, anom[TARGET].shift(-1), train_full)
out["top_anomaly_corr"] = anom_corr.head(15).round(3).to_dict()
out["bottom_anomaly_corr"] = anom_corr.tail(5).round(3).to_dict()

change_corr = lead_corr(values.diff(), values[TARGET].diff().shift(-1), train_full)
out["top_change_corr"] = change_corr.head(8).round(3).to_dict()
out["bottom_change_corr"] = change_corr.tail(5).round(3).to_dict()

# Figure 7: what correlates with the target, raw vs deseasonalised.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
top20 = target_corr.abs().sort_values(ascending=False).head(20)
axes[0].barh(range(len(top20)), target_corr[top20.index].to_numpy(), color="#2b6cb0")
axes[0].set_yticks(range(len(top20)), top20.index, fontsize=8)
axes[0].invert_yaxis()
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("Pearson r with BASEL_temp_mean(t+1)")
axes[0].set_title("(a) Raw correlation — dominated by the shared seasonal cycle", fontsize=10)
axes[0].grid(axis="x", alpha=0.3)

an20 = anom_corr.abs().sort_values(ascending=False).head(20)
cols_an = anom_corr[an20.index].to_numpy()
axes[1].barh(range(len(an20)), cols_an, color=["#2b6cb0" if v > 0 else "#c05621" for v in cols_an])
axes[1].set_yticks(range(len(an20)), an20.index, fontsize=8)
axes[1].invert_yaxis()
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("Pearson r with deseasonalised BASEL_temp_mean(t+1)")
axes[1].set_title("(b) After removing the monthly cycle (train years only)", fontsize=10)
axes[1].grid(axis="x", alpha=0.3)
fig.tight_layout()
savefig(fig, "fig07_target_correlation.png")

# Figure 8: persistence scatter on test set + residual distribution
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(X_persist[test], y_all[test], s=6, alpha=0.4, color="#2b6cb0")
lims = [min(X_persist[test].min(), y_all[test].min()), max(X_persist[test].max(), y_all[test].max())]
axes[0].plot(lims, lims, "k--", lw=1, label="y = x")
axes[0].set_xlabel("BASEL_temp_mean(t)  [°C]")
axes[0].set_ylabel("BASEL_temp_mean(t+1)  [°C]")
axes[0].set_title("Persistence baseline, test period 2009-2010", fontsize=10)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
res = y_all[test] - X_persist[test]
axes[1].hist(res, bins=50, color="#c05621", alpha=0.85)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].set_xlabel("residual  y(t+1) - y(t)  [°C]")
axes[1].set_ylabel("days")
axes[1].set_title(f"Persistence residuals (std = {res.std():.2f} °C)", fontsize=10)
axes[1].grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "fig08_persistence.png")

# --------------------------------------------------------------------------- #
# 7. Classification labels                                                       #
# --------------------------------------------------------------------------- #
print("[7] labels")
pc = picnic.drop(columns=["DATE"])
out["picnic"] = {
    "shape": list(picnic.shape),
    "n_stations": int(pc.shape[1]),
    "missing_stations": [s for s in STATIONS if not any(c.startswith(s) for c in pc.columns)],
    "positive_rate_overall": round(float(pc.to_numpy().mean()), 4),
    "positive_rate_by_station": pc.mean().round(4).to_dict(),
    "n_all_false_days": int((~pc.any(axis=1)).sum()),
    "n_all_true_days": int((pc.all(axis=1)).sum()),
}
pm2 = pc.copy()
pm2.insert(0, "month", pd.to_datetime(picnic["DATE"], format="%Y%m%d").dt.month)
out["picnic_rate_by_month"] = pm2.groupby("month").mean().mean(axis=1).round(4).to_dict()

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
rate = pc.mean().sort_values()
axes[0].bar(range(len(rate)), rate.to_numpy(), color="#2f855a")
axes[0].set_xticks(range(len(rate)), [c.replace("_picnic_weather", "") for c in rate.index], rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("positive rate")
axes[0].set_title("picnic-suitable days per station (class imbalance)", fontsize=10)
axes[0].grid(axis="y", alpha=0.3)
axes[1].plot(list(out["picnic_rate_by_month"].keys()), list(out["picnic_rate_by_month"].values()), marker="o", color="#2f855a")
axes[1].set_xticks(range(1, 13))
axes[1].set_xlabel("month")
axes[1].set_ylabel("positive rate")
axes[1].set_title("picnic suitability by month (mean over stations)", fontsize=10)
axes[1].grid(alpha=0.3)
fig.tight_layout()
savefig(fig, "fig09_picnic_labels.png")

# --------------------------------------------------------------------------- #
# 8. Dump                                                                        #
# --------------------------------------------------------------------------- #
payload = json.loads(json.dumps(out, default=str))
(ROOT / "docs" / "eda_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print("\n[done] docs/eda_results.json")
print(json.dumps(payload, indent=2, ensure_ascii=False)[:6000])
