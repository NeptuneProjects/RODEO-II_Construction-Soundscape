#!/bin/bash
# Build the report sections as Word documents with pandoc.
#
# Usage: scripts/publish.sh [reference.docx]
#
# Reads reports/hdr_report/<section>.md and writes <section>.docx beside it.
# Figure paths in the Markdown are relative to the repository root, so the
# results (results/<site>/report, results/compare) and the detector figure
# (results/southfork/events/A02/detections_VLA1.png) must be present locally.
# A reference .docx (e.g. the shared report) gives the output its styles.
# Runs from any directory.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORT="$ROOT/reports/hdr_report"
SECTIONS=("metrics" "pulse_detector" "interpulse" "background")

ARGS=(--citeproc --bibliography="$REPORT/references.bib" --csl="$REPORT/aip.csl"
      --resource-path="$ROOT")
if [[ $# -ge 1 ]]; then
    ARGS+=(--reference-doc="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")")
fi

cd "$ROOT"
for name in "${SECTIONS[@]}"; do
    # native_numbering: "Figure n" and "Table n" captions as Word fields, as
    # in the documents soundscape report writes.
    pandoc "$REPORT/$name.md" -t docx+native_numbering -o "$REPORT/$name.docx" "${ARGS[@]}"
    echo "$REPORT/$name.docx"
done
