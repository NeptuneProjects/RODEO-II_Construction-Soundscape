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


def moving_average(data, window_size):
    if data.ndim == 1:
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    return np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="valid"),
        axis=0,
        arr=data,
    )


def safe_log10(x, eps=1e-10):
    result = np.where(x > eps, x, -10)
    np.log10(result, out=result, where=result > 0)
    return result


def main(args: argparse.Namespace):
    time_start = np.datetime64(args.time_start, "us")
    time_end = np.datetime64(args.time_end, "us")
    if args.time_interval % 1 != 0:
        warnings.warn(
            "Time interval is not an integer. Rounding to the nearest integer."
        )
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

        power = ds.waveform**2

        if args.cutoff is not None:
            win_size = int(1 / args.cutoff * ds.stats.sampling_rate)
            power = moving_average(power, win_size)
            title += f"\nAverage Power (window: $1 / {args.cutoff}$ Hz) [{ds.stats.units}$^2$]"
        else:
            title += f"\nInstantaneous Power [{ds.stats.units}$^2$]"

        if args.db:
            power = 10 * safe_log10(power)

        ds.waveform = power

        if not args.ymax:
            ymax = ds.waveform.max()
        else:
            ymax = args.ymax
        if args.db and args.ymin is None:
            ymin = ds.waveform.max() - 30
        else:
            ymin = args.ymin

        fig = plotting.plot_time_series(ds, title=title, ymin=ymin, ymax=ymax)
        filename = (
            np.datetime_as_string(time_start_i, unit="s")
            .replace("-", "")
            .replace("T", "_")
            .replace(":", "")
            + "_"
            + args.title.replace("/", "_")
        )
        if args.db:
            filename += "_db"
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
        default="reports/power/inst",
    )
    parser.add_argument("--time_interval", help="Time interval", type=int, default=5)
    parser.add_argument("--time_units", help="Time units", type=str, default="m")
    parser.add_argument(
        "--time_start", help="Start time", type=str, default="2023-07-09T10:00:00"
    )
    parser.add_argument(
        "--time_end", help="End time", type=str, default="2023-07-09T11:30:00"
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
    parser.add_argument(
        "--cutoff", help="Cutoff frequency for moving average", type=float
    )
    parser.add_argument(
        "--db", help="Use decibel scale", default=False, action="store_true"
    )

    args = parser.parse_args()
    main(args)
