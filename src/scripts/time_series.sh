#!/bin/zsh

# This script generates time series for all the audio files in the given directory

# 1 min intervals
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/time_series/1min --time_interval 1 --title SHRU905/VLA1
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/time_series/1min --time_interval 1 --title SHRU920/VLA2
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/time_series/1min --time_interval 1 --title SHRU921/Tetra

# 5 min intervals
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/time_series/5min --time_interval 5 --title SHRU905/VLA1
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/time_series/5min --time_interval 5 --title SHRU920/VLA2
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/time_series/5min --time_interval 5 --title SHRU921/Tetra

# 90 min intervals
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/time_series/90min --time_interval 90 --title SHRU905/VLA1
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/time_series/90min --time_interval 90 --title SHRU920/VLA2
python src/visualizations/time_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/time_series/90min --time_interval 90 --title SHRU921/Tetra