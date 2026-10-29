#!/usr/bin/env python3


import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tritonoa.data.time import TIME_CONVERSION_FACTOR

from rodeo import plotting

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
            dt.append(grp["time_diff"][:] / TIME_CONVERSION_FACTOR)

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
        fig = plotting.plot_max_corr_stack(
            stations, data, dt, title=title, window=args.window
        )
        fig.savefig(
            args.outdir / event / "max_corr_stack.png",
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
        default=Path("data/max_corr"),
        help="Path to the data file.",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=60.0,
        help="Single-sided window to plot (s).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Path to save.",
    )
    args = parser.parse_args()
    main(args)
