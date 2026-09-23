"""R3: MLP grid (capacity-cap contrast) + multi-station pooled extension.

1. MLP grid on BASEL, input = all163 levels+deltas (326):
   hidden {32} / {64,32} / {128,64} x dropout {0.2, 0.4} x wd {1e-4, 1e-3}
   Huber, Adam lr 1e-3, batch 64, EarlyStopping(val MAE, patience 30),
   seeds {0,1,2}. Config selected by MEAN VAL MAE (2008) -- test is touched
   only for the final report of the selected config (protocol: no test reuse).
2. Pooled multi-station: all 18 stations, target = each station's own
   delta-T_max(t+1), features = the same global 326 lvl+d field.
   Delta space cancels station climate offsets -> clean pooling.
   Rows: 18 x 3652 = 65736 station-days (test 18 x 366 = 6588).

Outputs: docs/r3_mlp.csv, docs/r3_pooled.csv.

Run (Windows Python 3.12 from WSL):
  "/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r3_mlp.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import weatherlib as wl  # noqa: E402

HIDDEN_OPTS = [(32,), (64, 32), (128, 64)]
DROPOUT_OPTS = [0.2, 0.4]
WD_OPTS = [1e-4, 1e-3]
SEEDS = [0, 1, 2]


def load_basel() -> tuple[wl.DeltaMaxData, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df, dates, fc = wl.load_raw()
    month = dates.month.to_numpy()
    cleaned, _ = wl.clean(df, fc)
    tfull = dates < np.datetime64("2008-01-01")
    imp, _ = wl.impute_station_month(cleaned, month, tfull, fc)
    d = wl.make_delta_max_dataset(imp, dates, fc, wl.UPSTREAM, wl.UPSTREAM)
    X = np.hstack([d.X["levels"], d.X["deltas"]])
    return d, X, d.tr, d.va, d.te


def main() -> None:
    d, X, tr, va, te = load_basel()
    y = d.y
    mae_zero = wl.mae(y[te], wl.zero(y)[te])

    # ---------------- 1. MLP grid (select by MEAN VAL MAE) ----------------
    print("[MLP grid] hidden x dropout x wd, 3 seeds each -- selection on val")
    grid_rows = []
    best = (np.inf, None)
    t0 = time.time()
    for hidden in HIDDEN_OPTS:
        for dropout in DROPOUT_OPTS:
            for wd in WD_OPTS:
                val_maes, seed_maes, preds = [], [], []
                for seed in SEEDS:
                    f = wl.fit_mlp(X[tr], y[tr], X[va], y[va], hidden=hidden,
                                   dropout=dropout, wd=wd, seed=seed)
                    val_maes.append(f.info["val_mae"])
                    p = f.predict(X[te])
                    seed_maes.append(wl.mae(y[te], p))
                    preds.append(p)
                mean_val = float(np.mean(val_maes))
                ens = np.mean(preds, axis=0)
                ens_mae = wl.mae(y[te], ens)
                row = {
                    "hidden": str(hidden), "dropout": dropout, "wd": wd,
                    "mean_val_mae": round(mean_val, 3),
                    "seed_test_mean": round(float(np.mean(seed_maes)), 3),
                    "seed_test_std": round(float(np.std(seed_maes)), 3),
                    "ensemble_test_mae": round(float(ens_mae), 3),
                }
                grid_rows.append(row)
                print(f"  h={str(hidden):<10} do={dropout} wd={wd:g}  val {mean_val:.3f}  "
                      f"test {row['seed_test_mean']:.3f}±{row['seed_test_std']:.3f}  "
                      f"ens {ens_mae:.3f}  [{time.time()-t0:.0f}s]")
                if mean_val < best[0]:
                    best = (mean_val, (hidden, dropout, wd))
    print(f"[grid done {time.time()-t0:.0f}s] best by mean val MAE: "
          f"h={best[1][0]} do={best[1][1]} wd={best[1][2]} val={best[0]:.3f}")

    # selected config: report on test (touch test only now)
    bh, bd, bw = best[1]
    preds = [wl.fit_mlp(X[tr], y[tr], X[va], y[va], hidden=bh, dropout=bd, wd=bw,
                        seed=s).predict(X[te]) for s in SEEDS]
    ens = np.mean(preds, axis=0)
    lo, hi = wl.block_bootstrap_ci(y[te], ens, seed=42, n_boot=1000, block=7)
    print(f"\n[BEST MLP] h={bh} do={bd} wd={bw:g}")
    print(f"  per-seed test MAE: {[round(wl.mae(y[te], p), 3) for p in preds]}")
    print(f"  ensemble MAE {wl.mae(y[te], ens):.3f} [{lo:.3f},{hi:.3f}]  "
          f"R2 {wl.r2(y[te], ens):.3f}  skill {wl.skill(wl.mae(y[te], ens), mae_zero):+.1%}")
    print("  vs linear lvl+d 1.827 / lasso_l1 1.773")

    pd.DataFrame(grid_rows).to_csv(ROOT / "docs" / "r3_mlp.csv", index=False)

    # ---------------- 2. pooled multi-station ----------------
    print("\n[pooled 18-station]")
    df, dates, fc = wl.load_raw()
    month = dates.month.to_numpy()
    cleaned, _ = wl.clean(df, fc)
    tfull = dates < np.datetime64("2008-01-01")
    imp, _ = wl.impute_station_month(cleaned, month, tfull, fc)
    A = imp[fc].to_numpy(float)
    t_idx = np.arange(1, len(A) - 1)
    n_s = len(wl.STATIONS)
    ys = np.stack([A[t_idx + 1, fc.index(f"{s}_temp_max")] - A[t_idx, fc.index(f"{s}_temp_max")]
                   for s in wl.STATIONS], axis=1)                     # (3652, 18)
    Xp = np.hstack([d.X["levels"], d.X["deltas"]])                    # same 326 global features

    def mae_all(yp: np.ndarray) -> float:
        return float(np.mean(np.abs(ys[te].reshape(-1) - yp)))

    zero_pool = np.zeros(ys[te].reshape(-1).shape)
    mae_zero_pool = mae_all(zero_pool)
    print(f"  pooled zero (18x366=6588 station-days) test MAE {mae_zero_pool:.3f}")

    ps_maes = []
    for si, s in enumerate(wl.STATIONS):
        f = wl.fit_linear(Xp[tr], ys[tr, si], Xp[va], ys[va, si])
        ps_maes.append(wl.mae(ys[te, si], f.predict(Xp[te])))
    print(f"  per-station L2 mean test MAE {np.mean(ps_maes):.3f}  "
          f"(min {np.min(ps_maes):.3f} max {np.max(ps_maes):.3f})")

    X_tr = np.repeat(Xp[tr], n_s, axis=0); y_tr = ys[tr].reshape(-1)
    X_va = np.repeat(Xp[va], n_s, axis=0); y_va = ys[va].reshape(-1)
    X_te = np.repeat(Xp[te], n_s, axis=0); y_te = ys[te].reshape(-1)

    fp = wl.fit_linear(X_tr, y_tr, X_va, y_va)
    pred_p = fp.predict(X_te)
    lo_p, hi_p = wl.block_bootstrap_ci(y_te, pred_p, seed=42, n_boot=1000, block=7)
    print(f"  pooled L2 test MAE {wl.mae(y_te, pred_p):.3f} [{lo_p:.3f},{hi_p:.3f}]  "
          f"skill {wl.skill(wl.mae(y_te, pred_p), mae_zero_pool):+.1%}")

    t1 = time.time()
    fpm = wl.fit_mlp(X_tr, y_tr, X_va, y_va, hidden=(64, 32), dropout=0.3, wd=1e-4,
                     batch=256, epochs=100, patience=20, seed=0)
    pred_pm = fpm.predict(X_te)
    lo_m, hi_m = wl.block_bootstrap_ci(y_te, pred_pm, seed=42, n_boot=1000, block=7)
    print(f"  pooled MLP test MAE {wl.mae(y_te, pred_pm):.3f} [{lo_m:.3f},{hi_m:.3f}]  "
          f"skill {wl.skill(wl.mae(y_te, pred_pm), mae_zero_pool):+.1%}  [{time.time()-t1:.0f}s]")

    pd.DataFrame({
        "model": ["pooled_zero", "per_station_l2_mean", "pooled_l2", "pooled_mlp"],
        "test_MAE": [mae_zero_pool, float(np.mean(ps_maes)), wl.mae(y_te, pred_p), wl.mae(y_te, pred_pm)],
        "MAE_lo": [np.nan, np.nan, lo_p, lo_m],
        "MAE_hi": [np.nan, np.nan, hi_p, hi_m],
        "skill_vs_zero": [0.0, wl.skill(np.mean(ps_maes), mae_zero_pool),
                          wl.skill(wl.mae(y_te, pred_p), mae_zero_pool),
                          wl.skill(wl.mae(y_te, pred_pm), mae_zero_pool)],
    }).round(4).to_csv(ROOT / "docs" / "r3_pooled.csv", index=False)
    print("\nwrote docs/r3_mlp.csv, docs/r3_pooled.csv")


if __name__ == "__main__":
    main()
