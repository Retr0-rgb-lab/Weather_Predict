"""R1: formal pipeline smoke test -- reproduce the R0 probe's headline numbers
through the shared library (src/weatherlib), which upgrades the probe:

- imputation: station x month MEDIAN (probe used monthly mean)
- cleaning:   adds temp_min>temp_max masking (41 cells)
- features:   adds pressure gradients P_upstream - P_BASEL
- metrics:    adds moving-block bootstrap CI (block=7d) on test MAE

Run:  uv run --no-project --with pandas --with numpy --with scikit-learn python src/weatherlib/_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import weatherlib as wl  # noqa: E402


def main() -> None:
    df, dates, feature_cols = wl.load_raw()
    month = dates.month.to_numpy()

    cleaned, masked = wl.clean(df, feature_cols)
    print("masked:", masked)

    train_full = dates < np.datetime64("2008-01-01")
    imputed, _ind = wl.impute_station_month(cleaned, month, train_full, feature_cols)

    d = wl.make_delta_max_dataset(imputed, dates, feature_cols, wl.STATIONS, wl.UPSTREAM)
    print("samples train/val/test:", int(d.tr.sum()), int(d.va.sum()), int(d.te.sum()))
    print("gradients:", d.grad_cols)
    print("target train std: %.3f C" % float(d.y[d.tr].std()))

    # baselines
    zero_pred = wl.zero(d.y)
    yday_pred = wl.yesterday_delta(d.X["deltas"][:, d.feature_cols.index(d.target_col)])
    clim_pred = wl.monthly_delta_clim(d.y, d.tgt_month, d.tr)
    for name, pred in [("zero", zero_pred), ("yesterday_delta", yday_pred), ("monthly_clim", clim_pred)]:
        mae_test = wl.mae(d.y[d.te], pred[d.te])
        print(f"[baseline] {name:<16} test MAE {mae_test:.3f}")

    # ridge on full levels+deltas (probe headline was ~1.859)
    X = np.hstack([d.X["levels"], d.X["deltas"]])
    model = wl.fit_ridge(X[d.tr], d.y[d.tr])
    yp = model.predict(X[d.te])
    print("ridge lvl+d all163 test MAE %.3f  RMSE %.3f  R2 %.3f" % (
        wl.mae(d.y[d.te], yp), wl.rmse(d.y[d.te], yp), wl.r2(d.y[d.te], yp)))


if __name__ == "__main__":
    main()
