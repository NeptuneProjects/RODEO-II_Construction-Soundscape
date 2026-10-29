#!/bin/bash
# Build the report sections as Word documents with pandoc.
#
# Usage: scripts/publish.sh [draft.docx]
#
# Reads reports/hdr_report/<section>.md and writes <section>.docx beside it.
# Figure paths in the Markdown are relative to the repository root, so the
# results (results/<site>/report, results/compare) and the detector figure
# (results/southfork/events/A02/detections_VLA1.png) must be present locally.
#
# The sections take their styles from reports/hdr_report/styles.docx. Given a
# draft (e.g. the shared report downloaded as .docx), styles.docx is first
# rebuilt from it: a copy with the body, images, and comments removed, so that
# only the styles and page setup reach the sections. The draft itself is not
# modified. Without a draft or styles.docx, pandoc's default styles are used.
# Runs from any directory.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$ROOT/reports/hdr_report"
STYLES="$REPORT/styles.docx"
SECTIONS=("metrics" "pulse_detector" "interpulse" "background" "correlation")

if [[ $# -ge 1 ]]; then
    DRAFT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
    uv run --project "$ROOT" python -c '
import sys
from pathlib import Path
from soundscape.document import styles_only
styles_only(Path(sys.argv[1]), Path(sys.argv[2]))
' "$DRAFT" "$STYLES"
    echo "$STYLES (styles from $(basename "$DRAFT"))"
fi

ARGS=(--citeproc --bibliography="$REPORT/references.bib" --csl="$REPORT/aip.csl"
      --resource-path="$ROOT")
if [[ -f "$STYLES" ]]; then
    ARGS+=(--reference-doc="$STYLES")
fi

cd "$ROOT"
for name in "${SECTIONS[@]}"; do
    # native_numbering: "Figure n" and "Table n" captions as Word fields, as
    # in the documents soundscape report writes.
    pandoc "$REPORT/$name.md" -t docx+native_numbering -o "$REPORT/$name.docx" "${ARGS[@]}"
    echo "$REPORT/$name.docx"
done
