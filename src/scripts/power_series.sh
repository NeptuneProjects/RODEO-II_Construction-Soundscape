#!/bin/zsh

# This script generates spectrograms for all the audio files in the given directory

# 1 min intervals
# Instantaneous power
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/inst --time_interval 1 --title SHRU905/VLA1
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/inst --time_interval 1 --title SHRU920/VLA2
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/inst --time_interval 1 --title SHRU921/Tetra
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/inst --time_interval 1 --title SHRU905/VLA1 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/inst --time_interval 1 --title SHRU920/VLA2 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/inst --time_interval 1 --title SHRU921/Tetra --db

# 300 Hz Window
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/300Hz --time_interval 1 --title SHRU905/VLA1 --cutoff 300
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/300Hz --time_interval 1 --title SHRU920/VLA2 --cutoff 300
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/300Hz --time_interval 1 --title SHRU921/Tetra --cutoff 300
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/300Hz --time_interval 1 --title SHRU905/VLA1 --cutoff 300 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/300Hz --time_interval 1 --title SHRU920/VLA2 --cutoff 300 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/300Hz --time_interval 1 --title SHRU921/Tetra --cutoff 300 --db

# 100 Hz Window
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/100Hz --time_interval 1 --title SHRU905/VLA1 --cutoff 100
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/100Hz --time_interval 1 --title SHRU920/VLA2 --cutoff 100
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/100Hz --time_interval 1 --title SHRU921/Tetra --cutoff 100
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/100Hz --time_interval 1 --title SHRU905/VLA1 --cutoff 100 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/100Hz --time_interval 1 --title SHRU920/VLA2 --cutoff 100 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/100Hz --time_interval 1 --title SHRU921/Tetra --cutoff 100 --db

# 50 Hz Window
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/50Hz --time_interval 1 --title SHRU905/VLA1 --cutoff 50
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/50Hz --time_interval 1 --title SHRU920/VLA2 --cutoff 50
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/50Hz --time_interval 1 --title SHRU921/Tetra --cutoff 50
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/1min/50Hz --time_interval 1 --title SHRU905/VLA1 --cutoff 50 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/1min/50Hz --time_interval 1 --title SHRU920/VLA2 --cutoff 50 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/1min/50Hz --time_interval 1 --title SHRU921/Tetra --cutoff 50 --db


# 5 min intervals
# Instantaneous power
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/inst --time_interval 5 --title SHRU905/VLA1
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/inst --time_interval 5 --title SHRU920/VLA2
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/inst --time_interval 5 --title SHRU921/Tetra
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/inst --time_interval 5 --title SHRU905/VLA1 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/inst --time_interval 5 --title SHRU920/VLA2 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/inst --time_interval 5 --title SHRU921/Tetra --db

# 300 Hz Window
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/300Hz --time_interval 5 --title SHRU905/VLA1 --cutoff 300
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/300Hz --time_interval 5 --title SHRU920/VLA2 --cutoff 300
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/300Hz --time_interval 5 --title SHRU921/Tetra --cutoff 300
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/300Hz --time_interval 5 --title SHRU905/VLA1 --cutoff 300 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/300Hz --time_interval 5 --title SHRU920/VLA2 --cutoff 300 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/300Hz --time_interval 5 --title SHRU921/Tetra --cutoff 300 --db

# 100 Hz Window
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/100Hz --time_interval 5 --title SHRU905/VLA1 --cutoff 100
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/100Hz --time_interval 5 --title SHRU920/VLA2 --cutoff 100
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/100Hz --time_interval 5 --title SHRU921/Tetra --cutoff 100
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/100Hz --time_interval 5 --title SHRU905/VLA1 --cutoff 100 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/100Hz --time_interval 5 --title SHRU920/VLA2 --cutoff 100 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/100Hz --time_interval 5 --title SHRU921/Tetra --cutoff 100 --db

# 50 Hz Window
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/50Hz --time_interval 5 --title SHRU905/VLA1 --cutoff 50
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/50Hz --time_interval 5 --title SHRU920/VLA2 --cutoff 50
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/50Hz --time_interval 5 --title SHRU921/Tetra --cutoff 50
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_series/5min/50Hz --time_interval 5 --title SHRU905/VLA1 --cutoff 50 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_series/5min/50Hz --time_interval 5 --title SHRU920/VLA2 --cutoff 50 --db
python src/visualizations/power_series.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_series/5min/50Hz --time_interval 5 --title SHRU921/Tetra --cutoff 50 --db