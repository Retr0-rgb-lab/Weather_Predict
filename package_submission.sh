#!/usr/bin/env bash
# Build the submission ZIP: Task Report_Huang Haoran.zip
#
# Contents : CV.pdf, report.pdf (English), code (analysis/ + src/), report source,
#           README.md, requirements.txt, reproduce_all.sh
# Excluded : the Chinese report PDF, docs/ (EDA report + progress log), data/,
#            build artefacts, __pycache__
set -euo pipefail
cd "$(dirname "$0")"

NAME="Task Report_Huang Haoran"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

DEST="$NAME"
mkdir -p "$STAGE/$DEST"

# ---- deliverables -------------------------------------------------------
cp CV.pdf                      "$STAGE/$DEST/CV.pdf"
cp report/report.pdf           "$STAGE/$DEST/report.pdf"
cp README.md                   "$STAGE/$DEST/README.md"
cp requirements.txt            "$STAGE/$DEST/requirements.txt"
cp reproduce_all.sh            "$STAGE/$DEST/reproduce_all.sh"
chmod +x                      "$STAGE/$DEST/reproduce_all.sh"

# ---- code ---------------------------------------------------------------
cp -r analysis                "$STAGE/$DEST/analysis"
mkdir -p "$STAGE/$DEST/src"
cp -r src/weatherlib           "$STAGE/$DEST/src/weatherlib"

# ---- report source (so report.pdf is verifiable, not just readable) -----
mkdir -p "$STAGE/$DEST/report/figures"
cp report/report.tex           "$STAGE/$DEST/report/report.tex"
cp report/build.sh             "$STAGE/$DEST/report/build.sh"
chmod +x                      "$STAGE/$DEST/report/build.sh"
cp report/figures/*.png        "$STAGE/$DEST/report/figures/"

# ---- clean up -----------------------------------------------------------
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" \( -name '*.pyc' -o -name '.DS_Store' -o -name '*.aux' -o -name '*.log' -o -name '*.out' \) -delete 2>/dev/null || true
find "$STAGE" -name '.ipynb_checkpoints' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# ---- sanity checks ------------------------------------------------------
for must in CV.pdf report.pdf README.md requirements.txt reproduce_all.sh \
            analysis/r4_eval_figures.py src/weatherlib/models.py report/report.tex; do
    [ -e "$STAGE/$DEST/$must" ] || { echo "MISSING from package: $must"; exit 1; }
done
for banned in report_zh.pdf report_zh.html docs data; do
    if [ -e "$STAGE/$DEST/$banned" ]; then echo "UNEXPECTED in package: $banned"; exit 1; fi
done

# ---- zip ----------------------------------------------------------------
ZIP="$PWD/${NAME}.zip"
rm -f "$ZIP"
( cd "$STAGE" && zip -qr "$ZIP" "$DEST" )

echo "built: ${NAME}.zip"
sync   # the repo lives on a Windows mount; give the writes a moment to land
sleep 1
unzip -l "$ZIP" | tail -1
echo
echo "contents:"
unzip -Z1 "$ZIP" | sed '/\/$/d' | sed 's/^/  /'
