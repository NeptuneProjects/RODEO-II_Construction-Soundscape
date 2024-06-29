#!/bin/zsh

# This script generates spectrograms for all the audio files in the given directory

# 5 minutes
# python src/visualizations/spectrograms.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --title SHRU905/VLA1
# python src/visualizations/spectrograms.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --title SHRU920/VLA2
# python src/visualizations/spectrograms.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --title SHRU921/Tetra

# 1 minute
python src/visualizations/spectrograms.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --title SHRU905/VLA1 --time_interval 30 --time_unit s --destination reports/spectrograms/30s --tperseg 0.2
python src/visualizations/spectrograms.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --title SHRU920/VLA2 --time_interval 30 --time_unit s --destination reports/spectrograms/30s --tperseg 0.2
python src/visualizations/spectrograms.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --title SHRU921/Tetra --time_interval 30 --time_unit s --destination reports/spectrograms/30s --tperseg 0.2