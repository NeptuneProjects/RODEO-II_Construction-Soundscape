#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
import polars as pl

from rodeo import plotting

dotenv.load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))


def main(args: argparse.Namespace) -> None:
    data = pl.read_csv(args.file, try_parse_dates=True)
    fig = plotting.plot_met_data(data)
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
        default=Path(os.getenv("RESULTS")),
        help="Path to save the plot",
    )
    args = parser.parse_args()
    main(args)
