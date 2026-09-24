# Weather_Predict — ΔT_max Report

Candidate screening task (PolyU, Ms. PAN Yubing's group). Main deliverable:
`report/report.pdf` — "Predicting Day-to-Day Temperature Change Rather Than
Temperature: An Evaluation-First Reformulation for a Multi-Station European
Weather Dataset".

**Task**: predict the day-to-day change in Basel's daily maximum temperature,
ΔT_max(t+1) = T_max(t+1) − T_max(t), from day-t observations at 18 European
stations (163 features). In this target space the persistence forecast is
identically zero, so skill against the zero-change baseline (test MAE 2.393 °C)
measures meteorological learning rather than calendar knowledge.

## Headline results (test 2009, 366 days, block-bootstrap 95% CIs)

| Model | MAE (°C) | Skill vs zero |
|---|---|---|
| zero-change (bar) | 2.393 [2.104, 2.680] | — |
| ridge levels+Δ (326) | 1.827 [1.623, 2.014] | +23.7% |
| Lasso (326) | 1.773 [1.574, 1.955] | +25.9% |
| MLP ensemble | 1.751 [1.546, 1.946] | +26.9% |

All learned-model gains over zero are significant; differences among learned
models are within noise. See `docs/r4_final_table.csv`, `docs/r4_pairwise.csv`.

## Repository layout

```
analysis/   one script per experiment (eda, probe, r1 baselines, r2 lasso+ablation,
            r3 mlp+pooled, r4 final tables+figures)
src/        shared library (weatherlib: pipeline, features, models, metrics)
report/     LaTeX source + built PDF (report.pdf) + figures/
docs/       task brief, data-analysis report, machine-readable results (CSV/JSON),
            per-experiment progress records (docs/progress/)
data/       upstream dataset (NOT in git; download, see below)
```

## Requirements

- Python 3.12
- numpy, pandas, matplotlib, PyTorch (CPU sufficient; CUDA optional)
- (exploratory scripts only) scikit-learn

No `requirements.txt` is vendored by the upstream task; pinned versions used for
the report: numpy 2.4, pandas 3.0, matplotlib 3.10, torch 2.13+cu126.

## Data

Download the dataset from https://github.com/florian-huber/weather_prediction_dataset
(MIT; release commit `83d70ee`, Zenodo doi:10.5281/zenodo.7525955) and place
`weather_prediction_dataset.csv` under `data/`.

## One-click reproduction

```bash
bash reproduce_all.sh          # runs every experiment in report order
```

or, step by step (`PY` = your Python 3.12 interpreter):

```bash
$PY analysis/eda.py                    # dataset audit + exploratory figures
$PY analysis/delta_t_probe.py          # formulation feasibility probe
$PY analysis/r1_baselines.py           # pipeline regression test + baseline table
$PY analysis/r2_lasso_ablation.py      # Lasso + feature ablations (report Table 4)
$PY analysis/r3_mlp.py                 # MLP grid + pooled extension (report Sec. 5.6)
$PY analysis/r4_eval_figures.py        # final tables + report Figures 1–5
$PY report/build.sh                    # rebuild report/report.pdf (requires pdflatex)
```

Every number in the report traces to a CSV under `docs/` produced by the scripts
above; per-experiment design records (including abandoned directions) live in
`docs/progress/`.

## Evaluation protocol (summary)

Strict chronological split — train 2000–2007 (2920 aligned samples) /
validation 2008 (366) / test 2009 (366); all preprocessing statistics estimated
on the training years only; test set evaluated once per final configuration;
moving-block bootstrap CIs (7-day blocks, 1000 resamples, seed 42); paired
bootstraps for model differences.
