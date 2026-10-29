#!/bin/bash
# Compare the sites: cross-site figures, tables, and compare.docx in
# results/compare. Run after the site workflows (scripts/run_site.sh).
#
# Usage: scripts/compare.sh
#
# Runs from any directory.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
uv run soundscape compare \
    --site config/sites/vineyard.toml \
    --site config/sites/southfork.toml
