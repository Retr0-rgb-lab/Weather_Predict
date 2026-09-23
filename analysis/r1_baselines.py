"""R1 experiment: formal baseline table for the delta-T_max task.

Pipeline (shared library src/weatherlib, spec v1.1):
  load -> clean (range + temp_min>temp_max, BEFORE differencing)
  -> impute (station x month MEDIAN, train-only) -> build dataset
  -> baselines x3 + linear(torch, alpha on val 2008) x6 feature sets
  -> metrics + block-bootstrap CI

Models are 100% PyTorch (framework decision 2026-09-22). Output:
docs/r1_baseline.csv (machine-readable evidence, committed).

Run (Windows Python 3.12 from WSL):
  "/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r1_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import weatherlib as wl  # noqa: E402

BOOT = dict(n_boot=1000, block=7, seed=42)


def concat(data: wl.DeltaMaxData, spec: list[tuple[str, list[int] | None]]) -> np.ndarray:
    """Spec: list of (block_key, col_indices or None-for-all)."""
    blocks = [data.X[k][:, cols] if cols is not None else data.X[k] for k, cols in spec]
    return np.hstack(blocks)


def main() -> None:
    df, dates, feature_cols = wl.load_raw()
    month = dates.month.to_numpy()
    cleaned, masked = wl.clean(df, feature_cols)
    train_full = dates < np.datetime64("2008-01-01")
    imputed, _ind = wl.impute_station_month(cleaned, month, train_full, feature_cols)
    d = wl.make_delta_max_dataset(imputed, dates, feature_cols, wl.UPSTREAM, wl.UPSTREAM)
    te, tr, va = d.te, d.tr, d.va
    y = d.y

    print(f"[pipeline] masked={masked}  samples train/val/test = "
          f"{int(tr.sum())}/{int(d.va.sum())}/{int(te.sum())}  "
          f"grads={len(d.grad_cols)}  target_train_std={d.y[tr].std():.3f} C")

    rows = []
    zero_pred = wl.zero(y)
    mae_zero = wl.mae(y[te], zero_pred[te])

    def record(name: str, pred: np.ndarray) -> None:
        yt, yp = y[te], pred[te]
        lo, hi = wl.block_bootstrap_ci(yt, yp, seed=BOOT["seed"], **{k: v for k, v in BOOT.items() if k != "seed"})
        rows.append({
            "model": name,
            "MAE": round(wl.mae(yt, yp), 3), "MAE_lo": round(lo, 3), "MAE_hi": round(hi, 3),
            "RMSE": round(wl.rmse(yt, yp), 3), "R2": round(wl.r2(yt, yp), 4),
            "corr": round(wl.corr_safe(yt, yp), 4),
            "skill_vs_zero": round(wl.skill(wl.mae(yt, yp), mae_zero), 4),
        })
        print(f"  {name:<20} MAE {rows[-1]['MAE']:.3f} [{lo:.3f},{hi:.3f}]  RMSE {rows[-1]['RMSE']:.3f}  "
              f"R2 {rows[-1]['R2']:.3f}  corr {rows[-1]['corr']:+.3f}  skill {rows[-1]['skill_vs_zero']:+.1%}")

    # ---- baselines ----
    j = feature_cols.index(d.target_col)
    print("[baselines]")
    record("zero_change", zero_pred)
    record("yesterday_delta", wl.yesterday_delta(d.X["deltas"][:, j]))
    record("monthly_delta_clim", wl.monthly_delta_clim(y, d.tgt_month, tr))

    # ---- linear (torch) feature sets ----
    up = d.up_cols
    sets = {
        "linear_levels_all163": [("levels", None)],
        "linear_lvl+d_all163": [("levels", None), ("deltas", None)],
        "linear_lvl+d+grad": [("levels", None), ("deltas", None), ("grads", None)],
        "linear_up7_lvl+dtmax": [("levels", up), ("deltas", up)],
        "linear_up7_dtmax": [("deltas", up)],
        "linear_own_lvl+dtmax": [("levels", [j]), ("deltas", [j])],
    }
    print("[linear L2 (torch), alpha on val 2008]")
    for name, spec in sets.items():
        X = concat(d, spec)
        fitted = wl.fit_linear(X[tr], y[tr], X[va], y[va], penalty="l2")
        print(f"    {name:<22} alpha={fitted.info['alpha']:g}  val_mae={fitted.info['val_mae']:.3f}")
        record(name, fitted.predict(X))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "docs" / "r1_baseline.csv", index=False)
    print(f"\nwrote docs/r1_baseline.csv ({len(rows)} rows)")
    print(f"consistency check vs sklearn ridge: linear_lvl+d_all163 MAE "
          f"{out.loc[out['model']=='linear_lvl+d_all163','MAE'].values[0]:.3f} (sklearn was 1.851)")


if __name__ == "__main__":
    main()
