#!/bin/bash

DIR="./config/events"
SCRIPT="./scripts/pulse_metrics.py"
AGG="./scripts/agg_metrics.py"

find "$DIR" -type f -name "*.toml" | \
parallel \
    --bar \
    --line-buffer \
    --halt soon,fail=1 \
    python "$SCRIPT" --event={}

python "$AGG"