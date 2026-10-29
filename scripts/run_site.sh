#!/bin/bash
# Run the soundscape workflow for one site, from detection to the supplement.
#
# Usage: scripts/run_site.sh <site> [step ...]
#
# <site> is a config in config/sites (southfork, vineyard). Detection is
# re-applied to the saved candidates (--reselect), without reading the raw
# data, then each step runs in turn; by default every step from metrics on.
# Name steps to run only those, e.g. scripts/run_site.sh vineyard report.
# Stops at the first step that fails. Runs from any directory.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <site> [step ...]" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SITE="$ROOT/config/sites/$1.toml"
shift
if [[ ! -f "$SITE" ]]; then
    echo "no site config $SITE" >&2
    exit 2
fi

cd "$ROOT"
if [[ $# -eq 0 ]]; then
    uv run soundscape detect --reselect --site "$SITE"
    set -- metrics psd background kurtosis interpulse correlation report supplement
fi
for step in "$@"; do
    uv run soundscape "$step" --site "$SITE"
done
