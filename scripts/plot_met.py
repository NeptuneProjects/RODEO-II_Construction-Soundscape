#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

dotenv.load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))


def plot_met_data(data: pl.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(
        2, 1, figsize=(6, 3), sharex=True, gridspec_kw={"hspace": 0.1}
    )

    ax = axs[0]
    ax.plot(data["datetime"], data["WVHT"], label="Wave Height (m)")
    ax.grid(True)
    ax.set_ylabel("Wave Height (m)")

    ax = axs[1]
    ax.plot(data["datetime"], data["WSPD"], label="Wind Speed (m/s)")
    ax.grid(True)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_xlabel("Date")

    return fig


def main(args: argparse.Namespace) -> None:
    data = pl.read_csv(args.file, try_parse_dates=True)
    fig = plot_met_data(data)
    fig.savefig(
        args.savepath / "met_data.png", bbox_inches="tight", facecolor="white", dpi=600
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(os.getenv("METPATH")),
        help="Path to the file containing the data",
    )
    parser.add_argument(
        "--savepath",
        type=Path,
        default=Path(os.getenv("FIGPATH")),
        help="Path to save the plot",
    )
    args = parser.parse_args()
    main(args)
