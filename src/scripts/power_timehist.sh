#!/bin/zsh

# This script generates spectrograms for all the audio files in the given directory

python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_timehist/1min/inst --title SHRU905/VLA1
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_timehist/1min/inst --title SHRU920/VLA2
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_timehist/1min/inst --title SHRU921/Tetra

python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_timehist/1min/50Hz --title SHRU905/VLA1 --cutoff 50
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_timehist/1min/50Hz --title SHRU920/VLA2 --cutoff 50
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_timehist/1min/50Hz --title SHRU921/Tetra --cutoff 50

python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_timehist/1min/100Hz --title SHRU905/VLA1 --cutoff 100
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_timehist/1min/100Hz --title SHRU920/VLA2 --cutoff 100
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_timehist/1min/100Hz --title SHRU921/Tetra --cutoff 100

python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --destination reports/power_timehist/1min/300Hz --title SHRU905/VLA1 --cutoff 300
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --destination reports/power_timehist/1min/300Hz --title SHRU920/VLA2 --cutoff 300
python src/visualizations/power_timehist.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --destination reports/power_timehist/1min/300Hz --title SHRU921/Tetra --cutoff 300