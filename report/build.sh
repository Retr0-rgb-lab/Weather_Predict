#!/usr/bin/env bash
# Rebuild report/report.pdf from report.tex (requires pdflatex, TeX Live).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p figures
for f in fig10_model_ladder fig11_test_scatter_mlp fig12_seasonal_mae \
         fig13_lasso_coefs fig14_station_skill; do
    if [ -f "../docs/figures/$f.png" ]; then
        cp "../docs/figures/$f.png" "figures/$f.png"
    elif [ ! -f "figures/$f.png" ]; then
        echo "WARNING: $f.png missing (run analysis/r4_eval_figures.py first)" >&2
    fi
done

pdflatex -interaction=nonstopmode report.tex > build1.log 2>&1 || true
pdflatex -interaction=nonstopmode report.tex > build2.log 2>&1
grep -E "Output written" build2.log || { echo "BUILD FAILED"; tail -30 build2.log; exit 1; }
grep -cE "LaTeX Warning: (Reference|Citation).*undefined" build2.log && {
    echo "WARNING: undefined references"; exit 1; } || true
echo "OK: report/report.pdf"
