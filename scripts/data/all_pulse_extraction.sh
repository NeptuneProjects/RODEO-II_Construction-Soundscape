#!/bin/bash

DIR="./config/events"
SCRIPT="./scripts/pulse_extraction.py"

find "$DIR" -type f -name "*.toml" | \
parallel \
    --bar \
    --line-buffer \
    --halt soon,fail=1 \
    python "$SCRIPT" --event={}