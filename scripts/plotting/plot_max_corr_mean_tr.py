#!/usr/bin/env python3


import argparse
import os
from pathlib import Path

import dotenv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import interp1d
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
            data.append(grp["corr"][:, :, 500:])
            dt.append(grp["time_diff"][:, :, 500:] / TIME_CONVERSION_FACTOR)

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
    # print(len(data[0][0]), len(dt[0][0]), len(stations))
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
    files = list(args.data.glob("*.h*5"))
    ranges = []
    corr_means = []
    for file in tqdm(files):
        event = file.stem.split("_")[1]
        tqdm.write(f"Processing {event}")
        stations, data, dt = load_data(file)
        for i, station in enumerate(stations):
            ranges.append(load_range(df, event, station))
            data_sta = data[i][channel]
            dt_sta = dt[i][channel]

            corr = plotting.drop_nans(data_sta)
            dt_cl = plotting.drop_nans(dt_sta)
            t = np.cumsum(np.insert(np.diff(dt_cl[0]), 0, 0.0))

            windows = plotting.stack_data(corr, dt_cl, args.window)
            time_grid, _, corr_grid = plotting.regularize_grid(
                windows, t, window=args.window, nbins=100, drop_final_data_row=False
            )

            corr_grid[np.where(np.isnan(corr_grid))] = 0.0
            corr_mean = np.mean(corr_grid, axis=0)
            max_idx = np.argmax(corr_mean)
            right_corr_mean = corr_mean[max_idx:]
            right_time_grid = time_grid[0, max_idx:]

            # print(right_time_grid.shape, right_corr_mean.shape)
            # print(right_time_grid.min(), right_time_grid.max())
        
            
            new_time_grid = np.linspace(0.0, args.window, 100)
            f = interp1d(right_time_grid, right_corr_mean)
            new_corr_mean = f(new_time_grid)

            corr_means.append(new_corr_mean)

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

    corr_all = np.array(corr_means)
    ranges_all = np.array(ranges)
    
    fig = plt.figure()
    plt.pcolormesh(new_time_grid, ranges_all / 1e3, corr_all, cmap="magma", vmin=0.0, vmax=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Range (km)")
    plt.colorbar()
    fig.savefig(
        args.outdir / "max_corr.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )
    plt.show()



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
        default=300.0,
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
