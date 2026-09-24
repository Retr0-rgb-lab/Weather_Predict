#!/usr/bin/env bash
# One-click reproduction of every number in report/report.pdf.
# Usage: bash reproduce_all.sh [python-interpreter]
#        (default: Windows Python 3.12 via WSL path, as used for the report)
set -euo pipefail

PY="${1:-/mnt/d/Program Files/Pythons/python3.12/python.exe}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$ROOT/data/weather_prediction_dataset.csv" ]; then
    echo "ERROR: data/weather_prediction_dataset.csv not found."
    echo "Download from https://github.com/florian-huber/weather_prediction_dataset"
    echo "(commit 83d70ee) and place it under data/."
    exit 1
fi

for step in \
    "analysis/eda.py:exploratory analysis" \
    "analysis/delta_t_probe.py:formulation probe" \
    "analysis/r1_baselines.py:R1 pipeline + baselines" \
    "analysis/r2_lasso_ablation.py:R2 Lasso + ablation" \
    "analysis/r3_mlp.py:R3 MLP grid + pooled" \
    "analysis/r4_eval_figures.py:R4 final tables + figures"; do
    script="${step%%:*}"; desc="${step#*:}"
    echo "=== [$script] $desc ==="
    "$PY" "$ROOT/$script"
done

echo "All experiment scripts finished. Result tables: docs/*.csv, figures: docs/figures/."
echo "Rebuild the PDF with: (cd report && bash build.sh)"
