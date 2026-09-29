#!/usr/bin/env bash
# Rebuild the CV PDF from cv/cv.html (requires google-chrome with CJK fonts).
# The photo is already embedded as a base64 data URI, so the HTML is
# self-contained and the build needs no external image file.
set -euo pipefail
cd "$(dirname "$0")"

google-chrome --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --print-to-pdf=CV.pdf --virtual-time-budget=6000 "file://$PWD/cv.html" > /dev/null 2>&1

[ -s CV.pdf ] || { echo "BUILD FAILED: no CV.pdf produced"; exit 1; }

pages=$(python3 -c "d=open('CV.pdf','rb').read(); print(d.count(b'/Type /Page') - d.count(b'/Type /Pages'))")
if [ "$pages" = "1" ]; then
    echo "OK: cv/CV.pdf (1 page)"
else
    echo "WARNING: cv/CV.pdf is ${pages} pages, expected 1"
fi
