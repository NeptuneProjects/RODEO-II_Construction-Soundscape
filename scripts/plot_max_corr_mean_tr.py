#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm
from tritonoa.data.time import TIME_CONVERSION_FACTOR

from rodeo import plotting

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


def load_range(df: pl.DataFrame, event: str, station: str) -> float:
    return (
        df.filter(pl.col("Event") == event)
        .select(f"R to {station.replace("/", "_")} (m)")
        .item()
    )


def main(args: argparse.Namespace) -> None:
    channel = 0
    df = pl.read_csv(args.events)
    files = list(args.data.rglob("*.h*5"))
    corr_means = []
    ranges = []
    for file in tqdm(files):
        event = file.stem.split("_")[1]
        tqdm.write(f"Processing {event}")
        stations, data, dt = load_data(file)
        for i, station in enumerate(stations):
            ranges.append(load_range(df, event, station))
            data_sta = data[i][channel]
            dt_sta = dt[i][channel]

            corr = plotting.drop_nans(data_sta)
            print(corr)
            dt_cl = plotting.drop_nans(dt_sta)
            print(dt_cl)
            t = np.cumsum(np.insert(np.diff(dt_cl[0]), 0, 0.0)) / 1e3

            windows = plotting.stack_data(corr, dt_cl, args.window)
            time_grid, pulse_grid, corr_grid = plotting.regularize_grid(
                windows, t, window=args.window, nbins=100
            )
            corr_means.append(np.mean(corr_grid, axis=0))

    # Sort `corr_means` and `ranges` by `ranges`
    corr_means, ranges = map(
        list,
        zip(
            *sorted(
                zip(corr_means, ranges),
                key=lambda x: x[1],
            )
        ),
    )

    print(len(corr_means), len(ranges))
    print(corr_means[0].shape)

    # fig = plotting.plot_max_corr_stack(
    #     stations, data, dt, title=title, window=args.window
    # )
    # fig.savefig(
    #     args.outdir / event / "max_corr_stack.png",
    #     dpi=600,
    #     bbox_inches="tight",
    #     facecolor="white",
    # )
    # plt.close()


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
        "--events",
        type=Path,
        default=Path(os.getenv("EVENTCONFIG")),
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
        default=Path(os.getenv("EVENTRESULTS")),
        help="Path to save.",
    )
    args = parser.parse_args()
    main(args)
