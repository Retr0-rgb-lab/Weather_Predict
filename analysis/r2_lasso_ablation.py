"""R2: Lasso variable selection + feature-group ablation for delta-T_max.

Two products:
1. docs/r2_ablation.csv -- systematic feature-group ablation (all L2 linear,
   alpha on val 2008, block-bootstrap CI). Season and missing-indicator
   features are CONTROLLED ablations, off by default (spec v1.2 section 2.3.4).
2. docs/r2_lasso_coefs.csv -- Lasso (torch, L1 gradient) on levels+deltas at
   the val-selected alpha: top coefficients (raw units) = physical channels.

Run (Windows Python 3.12 from WSL):
  "/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r2_lasso_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import weatherlib as wl  # noqa: E402

ALPHAS_L1 = np.geomspace(1e-3, 1.0, 8)
L1_EPOCHS = 200
N_TOP = 25
BOOT = dict(n_boot=1000, block=7, seed=42)


def concat(data: wl.DeltaMaxData, spec: list[tuple[str, list[int] | None]]) -> np.ndarray:
    blocks = [data.X[k][:, cols] if cols is not None else data.X[k] for k, cols in spec]
    return np.hstack(blocks)


def main() -> None:
    df, dates, feature_cols = wl.load_raw()
    month = dates.month.to_numpy()
    cleaned, masked = wl.clean(df, feature_cols)
    train_full = dates < np.datetime64("2008-01-01")
    imputed, ind = wl.impute_station_month(cleaned, month, train_full, feature_cols)
    d = wl.make_delta_max_dataset(imputed, dates, feature_cols, wl.UPSTREAM, wl.UPSTREAM,
                                  missing_ind=ind)
    te, tr, va = d.te, d.tr, d.va
    y = d.y
    j = feature_cols.index(d.target_col)
    up = d.up_cols

    rows = []
    mae_zero = wl.mae(y[te], wl.zero(y)[te])

    def record(name: str, pred: np.ndarray) -> None:
        yt, yp = y[te], pred[te]
        lo, hi = wl.block_bootstrap_ci(yt, yp, seed=42, n_boot=1000, block=7)
        rows.append({
            "model": name,
            "MAE": round(wl.mae(yt, yp), 3), "MAE_lo": round(lo, 3), "MAE_hi": round(hi, 3),
            "RMSE": round(wl.rmse(yt, yp), 3), "R2": round(wl.r2(yt, yp), 4),
            "corr": round(wl.corr_safe(yt, yp), 4),
            "skill_vs_zero": round(wl.skill(wl.mae(yt, yp), mae_zero), 4),
        })
        print(f"  {name:<22} MAE {rows[-1]['MAE']:.3f} [{lo:.3f},{hi:.3f}]  "
              f"R2 {rows[-1]['R2']:.3f}  skill {rows[-1]['skill_vs_zero']:+.1%}")

    # ---------------- 1. feature-group ablation (L2, alpha on val) -------------
    record("zero_change", wl.zero(y))
    record("yesterday_delta", wl.yesterday_delta(d.X["deltas"][:, j]))

    sets = {
        "levels_all163": [("levels", None)],
        "deltas_all163": [("deltas", None)],
        "lvl+d_all163": [("levels", None), ("deltas", None)],
        "lvl+d+grad": [("levels", None), ("deltas", None), ("grads", None)],
        "lvl+d+grad+season": [("levels", None), ("deltas", None), ("grads", None), ("season", None)],
        "lvl+d+grad+missing": [("levels", None), ("deltas", None), ("grads", None), ("missings", None)],
        "up7_lvl+d": [("levels", up), ("deltas", up)],
        "up7_d": [("deltas", up)],
        "up7_d+grad": [("deltas", up), ("grads", None)],
        "own_lvl+d": [("levels", [j]), ("deltas", [j])],
    }
    print("[ablation: L2 linear, alpha on val 2008]")
    for name, spec in sets.items():
        X = concat(d, spec)
        fitted = wl.fit_linear(X[tr], y[tr], X[va], y[va], penalty="l2")
        print(f"    alpha={fitted.info['alpha']:g} val_mae={fitted.info['val_mae']:.3f}")
        record(name, fitted.predict(X))

    # ---------------- 2. Lasso variable selection on levels+deltas -------------
    print("\n[lasso: L1 gradient on levels+deltas (326 feats), alpha on val]")
    Xfull = concat(d, [("levels", None), ("deltas", None)])
    l1 = wl.fit_linear(Xfull[tr], y[tr], Xfull[va], y[va], penalty="l1",
                       alphas=ALPHAS_L1, epochs=L1_EPOCHS)
    w_std = l1.net.weight.detach().numpy().squeeze()
    w_raw = w_std / l1.std                       # dy/dx_raw per feature
    n_nonzero = int((np.abs(w_std) > 1e-4).sum())
    print(f"alpha={l1.info['alpha']:g}  val_mae={l1.info['val_mae']:.3f}  "
          f"nonzero={n_nonzero}/326  test MAE={wl.mae(y[te], l1.predict(Xfull)[te]):.3f}")

    cols = list(feature_cols) + [f"d_{c}" for c in feature_cols]
    tags = ["level" for _ in feature_cols] + ["delta" for _ in feature_cols]
    coef_df = pd.DataFrame({
        "feature": cols, "type": tags,
        "coef_raw": np.round(w_raw, 4), "coef_std": np.round(w_std, 4),
        "abs_coef_raw": np.abs(w_raw),
    }).sort_values("abs_coef_raw", ascending=False).head(N_TOP)
    coef_df = coef_df.drop(columns="abs_coef_raw").reset_index(drop=True)
    print(coef_df.to_string(index=False))
    coef_df.to_csv(ROOT / "docs" / "r2_lasso_coefs.csv", index=False)

    record("lasso_l1_full", l1.predict(Xfull))

    # selected (nonzero) columns -> compressed L2 model on the SAME feature set
    sel = np.where(np.abs(w_std) > 1e-4)[0]
    sel_lvl = sel[sel < len(feature_cols)]
    sel_dl = sel[sel >= len(feature_cols)] - len(feature_cols)
    Xtop = concat(d, [("levels", sel_lvl.tolist()), ("deltas", sel_dl.tolist())])
    ftop = wl.fit_linear(Xtop[tr], y[tr], Xtop[va], y[va], penalty="l2")
    record(f"lasso_top{n_nonzero}_l2", ftop.predict(Xtop))

    # top-25 by |coef_std| -> how much skill does an interpretable subset keep?
    order = np.argsort(np.abs(w_std))[::-1][:N_TOP]
    s_lvl = order[order < len(feature_cols)].tolist()
    s_dl = (order[order >= len(feature_cols)] - len(feature_cols)).tolist()
    Xt25 = concat(d, [("levels", s_lvl), ("deltas", s_dl)])
    ft25 = wl.fit_linear(Xt25[tr], y[tr], Xt25[va], y[va], penalty="l2")
    record(f"lasso_top{N_TOP}_l2", ft25.predict(Xt25))

    # ---------------- 3. dump ----------------------------------------------
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "docs" / "r2_ablation.csv", index=False)
    print(f"\nwrote docs/r2_ablation.csv ({len(rows)} rows), docs/r2_lasso_coefs.csv "
          f"(top {N_TOP} of {2 * len(feature_cols)})")


if __name__ == "__main__":
    main()
