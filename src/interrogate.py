#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Optional

from datautils.data import DataStream, read
from datautils.query import CatalogueQuery

catalogues = [
    "data/catalogues/SHRU905_VLA1_FileInfo.csv",
    "data/catalogues/SHRU920_VLA2_FileInfo.csv",
    "data/catalogues/SHRU921_Tetra_FileInfo.csv",
]

def main():
    for catalogue in catalogues:
        cq = CatalogueQuery(catalogue)
        for event in cq.events:
            print(event)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_start", help="Start time", type=str, default="2023-07-09T10:30:00"
    )
    parser.add_argument(
        "--time_end", help="End time", type=str, default="2023-07-09T10:30:20"
    )
    args = parser.parse_args()
    main(args)
