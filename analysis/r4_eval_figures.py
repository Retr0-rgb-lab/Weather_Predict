"""R4: canonical final table + paired significance + report figures.

Consolidates R1-R3 into one self-contained evaluation:
  docs/r4_final_table.csv -- model ladder: MAE [block-CI], RMSE, R2, corr, skill
  docs/r4_pairwise.csv    -- paired block-bootstrap MAE differences + significance
  docs/figures/fig10..14  -- report-quality figures (English labels, gitignored)

All models recomputed here under the same protocol (train-only stats, alpha on
val 2008, test evaluated once). Block bootstrap: moving blocks of 7 days,
n=1000, seed 42; pairwise differences on the SAME blocks.

Run (Windows Python 3.12 from WSL):
  "/mnt/d/Program Files/Pythons/python3.12/python.exe" analysis/r4_eval_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import weatherlib as wl  # noqa: E402

FIGS = ROOT / "docs" / "figures"
BOOT = dict(n_boot=1000, block=7, seed=42)
ALPHAS_L1 = np.geomspace(1e-3, 1.0, 8)
BEST_MLP = dict(hidden=(64, 32), dropout=0.4, wd=1e-4)
SEEDS = [0, 1, 2]


def main() -> None:
    df, dates, fc = wl.load_raw()
    month = dates.month.to_numpy()
    cleaned, _ = wl.clean(df, fc)
    tfull = dates < np.datetime64("2008-01-01")
    imp, _ = wl.impute_station_month(cleaned, month, tfull, fc)
    d = wl.make_delta_max_dataset(imp, dates, fc, wl.UPSTREAM, wl.UPSTREAM)
    X = np.hstack([d.X["levels"], d.X["deltas"]])
    tr, va, te = d.tr, d.va, d.te
    y = d.y
    j = fc.index(d.target_col)
    up = d.up_cols
    yt, mte = y[te], wl.mae(y[te], np.zeros_like(y[te]))

    # ---------------- build predictions on test ----------------
    def lin(spec: list[tuple[str, list[int] | None]]) -> np.ndarray:
        Xb = wl.linear_concat(d, spec) if hasattr(wl, "linear_concat") else _concat(d, spec)
        f = wl.fit_linear(Xb[tr], y[tr], Xb[va], y[va], penalty="l2")
        return f.predict(Xb[te])

    Xfull = _concat(d, [("levels", None), ("deltas", None)])
    l1 = wl.fit_linear(Xfull[tr], y[tr], Xfull[va], y[va], penalty="l1",
                       alphas=ALPHAS_L1, epochs=200)
    mlp_preds = [wl.fit_mlp(X[tr], y[tr], X[va], y[va], **BEST_MLP, seed=s).predict(X[te])
                 for s in SEEDS]
    ens_mlp = np.mean(mlp_preds, axis=0)

    models = {
        "zero": np.zeros_like(yt),
        "yesterday_delta": wl.yesterday_delta(d.X["deltas"][:, j])[te],
        "monthly_delta_clim": wl.monthly_delta_clim(y, d.tgt_month, tr)[te],
        "lin_levels_all163": lin([("levels", None)]),
        "lin_lvld_all163": lin([("levels", None), ("deltas", None)]),
        "lin_lvld_grad": lin([("levels", None), ("deltas", None), ("grads", None)]),
        "lin_up7_lvld": lin([("levels", up), ("deltas", up)]),
        "lin_own_lvld": lin([("levels", [j]), ("deltas", [j])]),
        "lasso_l1_all163": l1.predict(Xfull[te]),
        "mlp_ens_best": ens_mlp,
    }

    # ---------------- canonical table ----------------
    rows = []
    for name, p in models.items():
        lo, hi = wl.block_bootstrap_ci(yt, p, **BOOT)
        rows.append({
            "model": name, "MAE": round(wl.mae(yt, p), 3),
            "MAE_lo": round(lo, 3), "MAE_hi": round(hi, 3),
            "RMSE": round(wl.rmse(yt, p), 3), "R2": round(wl.r2(yt, p), 4),
            "corr": round(wl.corr_safe(yt, p), 4),
            "skill_vs_zero": round(wl.skill(wl.mae(yt, p), mte), 4),
        })
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "docs" / "r4_final_table.csv", index=False)
    print(table.to_string(index=False))

    # ---------------- pairwise paired-block significance ----------------
    def sig(yt_, p1, p2):
        lo, hi = wl.block_bootstrap_diff_ci(yt_, p1, p2, **BOOT)
        return round(lo, 3), round(hi, 3), bool((lo > 0) or (hi < 0))

    pairs = [("mlp_ens_best", "zero"), ("lasso_l1_all163", "zero"),
             ("lin_lvld_all163", "zero"), ("mlp_ens_best", "lin_lvld_all163"),
             ("mlp_ens_best", "lasso_l1_all163"), ("lasso_l1_all163", "lin_lvld_all163"),
             ("lin_lvld_all163", "lin_up7_lvld")]
    prow = []
    for a, b in pairs:
        lo, hi, s = sig(yt, models[a], models[b])
        prow.append({"A": a, "B": b, "diff_MAE_A_minus_B": round(wl.mae(yt, models[a]) - wl.mae(yt, models[b]), 3),
                     "CI_lo": lo, "CI_hi": hi, "sig_95": s})
    pw = pd.DataFrame(prow)
    pw.to_csv(ROOT / "docs" / "r4_pairwise.csv", index=False)
    print("\n" + pw.to_string(index=False))

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(9, 4.2))
    keys = ["zero", "yesterday_delta", "monthly_delta_clim", "lin_levels_all163",
            "lin_lvld_all163", "lin_lvld_grad", "lasso_l1_all163", "mlp_ens_best"]
    mae = [table.loc[table["model"] == k, "MAE"].values[0] for k in keys]
    lo = [table.loc[table["model"] == k, "MAE_lo"].values[0] for k in keys]
    hi = [table.loc[table["model"] == k, "MAE_hi"].values[0] for k in keys]
    ax.bar(range(len(keys)), mae, yerr=[np.array(mae) - np.array(lo), np.array(hi) - np.array(mae)],
           capsize=3, color=["#718096", "#a0aec0", "#a0aec0", "#2b6cb0", "#2b6cb0", "#2b6cb0", "#c05621", "#c53030"])
    ax.axhline(mte, ls="--", color="black", lw=1, label=f"zero-change bar ({mte:.2f} C)")
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("test MAE (C, 95% block-bootstrap CI)")
    ax.set_title("Delta-T_max(1) at BASEL: model ladder, test 2009-2010 (366 days)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig10_model_ladder.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(yt, ens_mlp, s=6, alpha=0.4, color="#c53030")
    lims = [min(yt.min(), ens_mlp.min()), max(yt.max(), ens_mlp.max())]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("observed Delta-T_max (C)"); ax.set_ylabel("MLP ensemble (C)")
    ax.set_title(f"MLP ensemble, test: MAE {wl.mae(yt, ens_mlp):.2f}, r {np.corrcoef(yt, ens_mlp)[0,1]:.2f}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig11_test_scatter_mlp.png", dpi=150); plt.close(fig)

    tgt_month_te = d.tgt_month[te]
    zm = [wl.mae(yt[tgt_month_te == m], models["zero"][tgt_month_te == m]) for m in range(1, 13)]
    mm = [wl.mae(yt[tgt_month_te == m], ens_mlp[tgt_month_te == m]) for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(range(1, 13), zm, "o-", label="zero-change", color="#718096")
    ax.plot(range(1, 13), mm, "s-", label="MLP ensemble", color="#c53030")
    ax.set_xticks(range(1, 13)); ax.set_xlabel("target month")
    ax.set_ylabel("MAE (C)"); ax.set_title("Seasonal error profile, test 2009-2010")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig12_seasonal_mae.png", dpi=150); plt.close(fig)

    lc = pd.read_csv(ROOT / "docs" / "r2_lasso_coefs.csv").head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["#c05621" if c < 0 else "#2b6cb0" for c in lc["coef_std"]]
    ax.barh(range(len(lc)), lc["coef_std"], color=colors)
    ax.set_yticks(range(len(lc)), lc["feature"], fontsize=7)
    ax.set_xlabel("standardized Lasso coefficient")
    ax.set_title("Top physical channels for Delta-T_max (standardized scale)")
    ax.axvline(0, color="black", lw=0.8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig13_lasso_coefs.png", dpi=150); plt.close(fig)

    # per-station: zero vs independent linear
    A = imp[fc].to_numpy(float)
    t_idx = np.arange(1, len(A) - 1)
    zero_s, lin_s = [], []
    for s in wl.STATIONS:
        yss = A[t_idx + 1, fc.index(f"{s}_temp_max")] - A[t_idx, fc.index(f"{s}_temp_max")]
        f = wl.fit_linear(X[tr], yss[tr], X[va], yss[va])
        zero_s.append(wl.mae(yss[te], np.zeros_like(yss[te])))
        lin_s.append(wl.mae(yss[te], f.predict(X[te])))
    order = np.argsort(zero_s)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xx = np.arange(len(wl.STATIONS))
    ax.bar(xx - 0.2, np.array(zero_s)[order], width=0.4, label="zero-change", color="#718096")
    ax.bar(xx + 0.2, np.array(lin_s)[order], width=0.4, label="per-station linear", color="#2b6cb0")
    ax.set_xticks(xx, [wl.STATIONS[i] for i in order], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("test MAE (C)")
    ax.set_title("Per-station skill, 18 stations (independent L2 models), test 2009-2010")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig14_station_skill.png", dpi=150); plt.close(fig)

    print(f"\nwrote docs/r4_final_table.csv, docs/r4_pairwise.csv + "
          f"{len(list(FIGS.glob('fig1*.png')))} figures")


def _concat(data: wl.DeltaMaxData, spec: list[tuple[str, list[int] | None]]) -> np.ndarray:
    blocks = [data.X[k][:, cols] if cols is not None else data.X[k] for k, cols in spec]
    return np.hstack(blocks)


if __name__ == "__main__":
    main()
