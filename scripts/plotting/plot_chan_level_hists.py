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
plt.style.use(Path("config/jasa_fig_style.mplstyle"))

STATIONS = ["SHRU905_VLA1", "SHRU920_VLA2"]
CHANNELS = [0, 1, 2, 3]


def plot_channel_level_hists(
    df: pl.DataFrame, stations: list[str], channels: list[int]
) -> plt.Figure:
    fig, axs = plt.subplots(figsize=(6.5, 3), nrows=2, gridspec_kw={"hspace": 0.1})

    for i, station in enumerate(stations):
        ax = axs[i]

        for channel in channels:
            series = df.filter(
                [
                    pl.col("station") == station,
                    pl.col("channel") == channel,
                ]
            )["SPL_pk"]
            density, bins = np.histogram(series.to_numpy(), bins=50, density=True)

            ax.plot(bins[:-1], density, label=f"Channel {channel}")

        ax.set_xlim(140, 165)
        ax.set_ylim(0.0, 0.3)
        ax.grid(which="both")
        ax.set_xlabel("SPL_pk (dB)")
        ax.set_ylabel("Probability density")
        if i == 0:
            ax.set_xticklabels([])
        if i == 1:
            ax.legend(loc="upper right")
        ax.set_title(f"{station.replace("_", "/")}", y=0.8)

    return fig


def main(args: argparse.Namespace) -> None:

    df = pl.read_csv(args.input).filter(
        [
            pl.col("event_name") == args.event,
            pl.col("station").is_in(STATIONS),
        ]
    )
    fig = plot_channel_level_hists(df, STATIONS, CHANNELS)
    fig.savefig(
        args.output / f"{args.event}_channel_level_hists.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("RESULTS")),
        help="Path to output directory.",
    )
    args = parser.parse_args()
    main(args)
