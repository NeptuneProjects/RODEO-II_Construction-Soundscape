#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

dotenv.load_dotenv()

STATIONS = ["SHRU905_VLA1", "SHRU920_VLA2"]
CHANNELS = [0, 1, 2, 3]


def main(args: argparse.Namespace) -> None:

    df = pl.read_csv(args.input).filter(
        [
            pl.col("event_name") == args.event,
            pl.col("station").is_in(STATIONS),
        ]
    )
    print(df)

    # TODO: Plot histogram of SPL_pk with subplots (2) for each station,
    # and curves (4) for each channel.

    for station in STATIONS:
        fig = plt.figure()
        for channel in CHANNELS:
            series = df.filter(
                [
                    pl.col("station") == station,
                    pl.col("channel") == channel,
                ]
            )["SPL_pk"]
            density, bins = np.histogram(series.to_numpy(), bins=50, density=True)

            plt.plot(bins[:-1], density, label=f"Channel {channel}")
            plt.xlim(140, 165)
            plt.xlabel("SPL_pk (dB)")
            plt.ylabel("Probability density")
            plt.legend()
            plt.title(f"{station} - SPL_pk")
        
        plt.draw()




    # TODO: Plot the average PSD across all detections with subplots (2)
    # for each station, and curves (4) for each channel.
    plt.show()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot depth-dependent data.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("METRICSFILE")),
        help="Path to data directory.",
    )
    parser.add_argument("--event", type=str, default="A10-B", help="Event name.")
    args = parser.parse_args()
    main(args)
