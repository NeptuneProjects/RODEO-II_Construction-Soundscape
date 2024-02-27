#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import warnings

from datautils.data import read
from datautils.query import CatalogueQuery
import matplotlib.pyplot as plt
import numpy as np

import plotting


def main(args: argparse.Namespace):
    time_start = np.datetime64(args.time_start, "us")
    time_end = np.datetime64(args.time_end, "us")
    if args.time_interval % 1 != 0:
        warnings.warn("Time interval is not an integer. Rounding to the nearest integer.")
        args.time_interval = round(args.time_interval)
    time_interval = np.timedelta64(args.time_interval, args.time_units)
    if time_end - time_start < time_interval:
        time_interval = time_end - time_start
    num_intervals = int((time_end - time_start) / time_interval)
    channels = [int(c) for c in args.channels.split(",")]

    for i in range(num_intervals):
        time_start_i = time_start + i * time_interval
        time_end_i = time_start + (i + 1) * time_interval
        title = (
            f"{args.title}\n"
            f"{np.datetime_as_string(time_start_i, unit='s')}"
            f" - {np.datetime_as_string(time_end_i, unit='s')}"
        )

        cat_query = CatalogueQuery(
            catalogue=args.catalogue,
            destination=args.destination,
            time_start=time_start_i,
            time_end=time_end_i,
            channels=channels,
        )
        ds = read(cat_query)
        title += f"\nAmplitude [{ds.stats.units}]"

        fig = plotting.plot_time_series(ds, title=title, ymin=args.ymin, ymax=args.ymax)
        filename = (
            np.datetime_as_string(time_start_i, unit="s")
            .replace("-", "")
            .replace("T", "_")
            .replace(":", "")
            + "_"
            + args.title.replace("/", "_")
        )
        args.destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(args.destination / filename), bbox_inches="tight", facecolor="w"
        )
        print(f"{filename}.png saved to {args.destination}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectrogram plotting routine")
    parser.add_argument(
        "--catalogue",
        help="Path to the catalogue file",
        type=Path,
        default="data/catalogues/SHRU905_VLA1_FileInfo.csv",
    )
    parser.add_argument(
        "--destination",
        help="Path to the destination directory",
        type=Path,
        default="reports/time_series",
    )
    parser.add_argument("--time_interval", help="Time interval", type=int, default=5)
    parser.add_argument("--time_units", help="Time units", type=str, default="m")
    parser.add_argument(
        "--time_start", help="Start time", type=str, default="2023-07-09T09:45:00"
    )
    parser.add_argument(
        "--time_end", help="End time", type=str, default="2023-07-09T11:45:00"
    )
    parser.add_argument(
        "--channels", help="Channels to be read", type=str, default="0,1,2,3"
    )
    parser.add_argument(
        "--title", help="Title of the plot", type=str, default="SHRU905/VLA1"
    )
    parser.add_argument(
        "--ymin", help="Minimum value for the color scale", type=float
    )
    parser.add_argument(
        "--ymax", help="Maximum value for the color scale", type=float
    )

    args = parser.parse_args()
    main(args)
