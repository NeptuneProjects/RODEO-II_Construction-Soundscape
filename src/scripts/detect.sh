#!/bin/zsh

python ./src/detect.py \
    --catalogue catalogues/SHRU921_Tetra_FileInfo.csv \
    --time_start "2023-07-09T10:00:00" \
    --time_end "2023-07-09T11:30:00" \
    --channels "0,1,2,3" \
    --threshold 2.25 \
    --stwin 0.025 \
    --ltwin 5.0 \
    --database "detections/SHRU921_Tetra.h5" \
    --metadata "A9"
