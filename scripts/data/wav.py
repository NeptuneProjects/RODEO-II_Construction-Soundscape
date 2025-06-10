#!/usr/bin/env python3


import argparse
from pathlib import Path

from datautils.data import read
from datautils.query import CatalogueQuery
from datautils.time import TIME_PRECISION
import numpy as np

WAVPATH = Path("data/wav")

catalogues = [
    Path("data/catalogues/SHRU905_VLA1_FileInfo.csv"),
    Path("data/catalogues/SHRU920_VLA2_FileInfo.csv"),
    Path("data/catalogues/SHRU921_Tetra_FileInfo.csv"),
]


def main(args: argparse.Namespace) -> None:
    time_start = np.datetime64(args.time_start, TIME_PRECISION)
    time_end = np.datetime64(args.time_end, TIME_PRECISION)

    for catalogue in catalogues:
        cq = CatalogueQuery(
            catalogue=catalogue,
            destination="data/interrogation",
            time_start=time_start,
            time_end=time_end,
            channels=[int(c) for c in args.channels.split(",")],
        )
        ds = read(cq)
        filename = (
            "_".join(catalogue.name.split("_")[0:2])
            + "_"
            + np.datetime_as_string(time_start, unit="s")
            .replace("-", "")
            .replace("T", "_")
            .replace(":", "")
            + ".wav"
        )
        ds.write_wav(WAVPATH / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_start", help="Start time", type=str, default="2023-07-09T11:44:00"
    )
    parser.add_argument(
        "--time_end", help="End time", type=str, default="2023-07-09T12:04:00"
    )
    parser.add_argument(
        "--channels", help="Channels to be read", type=str, default="0,1,2,3"
    )
    args = parser.parse_args()
    main(args)
