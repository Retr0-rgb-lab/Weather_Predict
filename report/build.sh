#!/usr/bin/env bash
# Rebuild the report PDFs:
#   report.pdf     -- English submission version (LaTeX, needs pdflatex / TeX Live)
#   report_zh.pdf  -- Chinese reference version (HTML, needs google-chrome)
# Pass "en" or "zh" to build only one of them (default: both).
set -euo pipefail
cd "$(dirname "$0")"
WHAT="${1:-all}"

mkdir -p figures
for f in fig10_model_ladder fig11_test_scatter_mlp fig12_seasonal_mae \
         fig13_lasso_coefs fig14_station_skill; do
    if [ -f "../docs/figures/$f.png" ]; then
        cp "../docs/figures/$f.png" "figures/$f.png"
    elif [ ! -f "figures/$f.png" ]; then
        echo "WARNING: $f.png missing (run analysis/r4_eval_figures.py first)" >&2
    fi
done

if [ "$WHAT" = "all" ] || [ "$WHAT" = "en" ]; then
    pdflatex -interaction=nonstopmode report.tex > build1.log 2>&1 || true
    pdflatex -interaction=nonstopmode report.tex > build2.log 2>&1
    grep -E "Output written" build2.log || { echo "EN BUILD FAILED"; tail -30 build2.log; exit 1; }
    grep -cE "LaTeX Warning: (Reference|Citation).*undefined" build2.log && {
        echo "WARNING: undefined references"; exit 1; } || true
    echo "OK: report/report.pdf (English)"
fi

if [ "$WHAT" = "all" ] || [ "$WHAT" = "zh" ]; then
    # Chinese version: HTML -> PDF via headless Chrome. CJK fonts must be
    # available to fontconfig (SimSun/SimHei or any CJK font).
    fc-list :lang=zh > /dev/null 2>&1 || { echo "ZH BUILD FAILED: no CJK font"; exit 1; }
    google-chrome --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
        --print-to-pdf=report_zh.pdf --virtual-time-budget=8000 \
        "file://$PWD/report_zh.html" > /dev/null 2>&1
    [ -s report_zh.pdf ] || { echo "ZH BUILD FAILED: no report_zh.pdf"; exit 1; }
    echo "OK: report/report_zh.pdf (Chinese)"
fi

