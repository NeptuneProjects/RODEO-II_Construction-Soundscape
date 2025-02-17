#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tritonoa.data.time import TIME_CONVERSION_FACTOR

from rodeo import plotting, utils

dotenv.load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))

STATIONS = ["SHRU921/Tetra", "SHRU905/VLA1", "SHRU920/VLA2"]


def load_data(file: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    stations = []
    data = []
    dt = []
    with h5py.File(file, "r") as f:
        for grp in f.values():
            stations.append(grp.name[1:].replace("-", "/"))
            data.append(grp["corr"][:])
            dt.append(grp["time_diff"][:] / TIME_CONVERSION_FACTOR / 1e3)

    # Sort by order of STATIONS
    data, dt, stations = map(
        list,
        zip(
            *sorted(
                zip(data, dt, stations),
                key=lambda x: STATIONS.index(x[2]),
            )
        ),
    )

    return stations, np.array(data), np.array(dt)


def main(args: argparse.Namespace) -> None:
    files = list(args.data.rglob("*.h*5"))
    for file in tqdm(files):
        event = file.stem.split("_")[1]
        tqdm.write(f"Processing {event}")
        title = f"Event {event}"
        stations, data, dt = load_data(file)

        fig = plotting.plot_max_corr_mat(stations, data, dt, title=title)
        fig.savefig(
            args.outdir / event / "max_corr.png",
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the maximum correlation of the data."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.getenv("MAXCORRPATH")),
        help="Path to the data file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(os.getenv("EVENTRESULTS")),
        help="Path to save.",
    )
    args = parser.parse_args()
    main(args)
