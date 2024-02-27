#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import warnings

from datautils.data import read
from datautils.query import CatalogueQuery
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

        fig = plotting.plot_spectrograms(
            ds,
            title=title,
            tperseg=args.tperseg,
            overlap=args.overlap,
            nfft=args.nfft,
            fmin=args.fmin,
            fmax=args.fmax,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        filename = (
            np.datetime_as_string(time_start_i, unit="s")
            .replace("-", "")
            .replace("T", "_")
            .replace(":", "")
            + "_"
            + args.title.replace("/", "_")
        )
        fig.savefig(
            Path(args.destination / filename), bbox_inches="tight", facecolor="w", dpi=300
        )
        fig.clf()
        


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
        default="reports/spectrograms",
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
    parser.add_argument("--tperseg", help="Window size in seconds", type=int, default=1)
    parser.add_argument(
        "--overlap", help="Overlap size in percentage", type=float, default=0.75
    )
    parser.add_argument(
        "--nfft", help="Number of data points used in each block", type=int
    )
    parser.add_argument(
        "--fmin", help="Minimum frequency in Hz", type=float, default=20.0
    )
    parser.add_argument(
        "--fmax", help="Maximum frequency in Hz", type=float, default=300.0
    )
    parser.add_argument(
        "--vmin", help="Minimum value for the color scale", type=float, default=60
    )
    parser.add_argument(
        "--vmax", help="Maximum value for the color scale", type=float, default=120
    )

    args = parser.parse_args()
    main(args)
