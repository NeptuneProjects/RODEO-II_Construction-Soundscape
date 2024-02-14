#!/bin/zsh

# This script generates spectrograms for all the audio files in the given directory

python src/spectrograms.py --catalogue data/catalogues/SHRU905_VLA1_FileInfo.csv --title SHRU905/VLA1
python src/spectrograms.py --catalogue data/catalogues/SHRU920_VLA2_FileInfo.csv --title SHRU920/VLA2
python src/spectrograms.py --catalogue data/catalogues/SHRU921_Tetra_FileInfo.csv --title SHRU921/Tetra