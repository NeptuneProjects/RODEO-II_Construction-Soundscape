#!/bin/bash
# Collect the report documents into one zip file for sharing.
#
# Usage: scripts/distribute.sh [output.zip]
#
# The zip holds the report sections built by scripts/publish.sh, each
# site's report and data supplement, and the cross-site comparison:
#
#   report/            metrics, pulse_detector, interpulse, background (.docx)
#   south_fork_wind/   report.docx, supplement.pdf
#   vineyard_wind_1/   report.docx, supplement.pdf
#   comparison/        compare.docx
#   README.txt         contents, version, commit, and date
#
# By default it is written to dist/construction_soundscape_<version>_<date>.zip.
# Stops if a document is missing, and warns if a report section's Markdown is
# newer than its .docx (run scripts/publish.sh first). Runs from any directory.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml)"
DATE="$(date +%Y-%m-%d)"
OUT="${1:-dist/construction_soundscape_${VERSION}_${DATE}.zip}"
case "$OUT" in
    /*) ;;
    *) OUT="$ROOT/$OUT" ;;
esac

# Destination in the zip, then source in the repository.
FILES=(
    "report/metrics.docx"          "reports/hdr_report/metrics.docx"
    "report/pulse_detector.docx"   "reports/hdr_report/pulse_detector.docx"
    "report/interpulse.docx"       "reports/hdr_report/interpulse.docx"
    "report/background.docx"       "reports/hdr_report/background.docx"
    "south_fork_wind/report.docx"     "results/southfork/report/report.docx"
    "south_fork_wind/supplement.pdf"  "results/southfork/report/supplement.pdf"
    "vineyard_wind_1/report.docx"     "results/vineyard/report/report.docx"
    "vineyard_wind_1/supplement.pdf"  "results/vineyard/report/supplement.pdf"
    "comparison/compare.docx"      "results/compare/compare.docx"
)

missing=()
for ((i = 1; i < ${#FILES[@]}; i += 2)); do
    [[ -f "${FILES[i]}" ]] || missing+=("${FILES[i]}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "missing (run the workflow, scripts/compare.sh, or scripts/publish.sh):" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
fi

for md in reports/hdr_report/{metrics,pulse_detector,interpulse,background}.md; do
    if [[ "$md" -nt "${md%.md}.docx" ]]; then
        echo "warning: $md is newer than its .docx; run scripts/publish.sh" >&2
    fi
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for ((i = 0; i < ${#FILES[@]}; i += 2)); do
    mkdir -p "$STAGE/$(dirname "${FILES[i]}")"
    cp "${FILES[i + 1]}" "$STAGE/${FILES[i]}"
done

{
    echo "RODEO II construction soundscape: report materials"
    echo
    echo "Version: $VERSION"
    echo "Commit:  $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- || echo ' (with uncommitted changes)')"
    echo "Date:    $DATE"
    echo
    echo "Contents:"
    for ((i = 0; i < ${#FILES[@]}; i += 2)); do
        echo "  ${FILES[i]}"
    done
} > "$STAGE/README.txt"

mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
(cd "$STAGE" && zip -q -r -X "$OUT" README.txt report south_fork_wind vineyard_wind_1 comparison)
echo "$OUT ($(du -h "$OUT" | cut -f1))"
