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
    df = pl.read_csv(args.input, try_parse_dates=True)
    plotting.plot_event_metrics(df, args.outdir)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot site metrics from CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("METRICSFILE")),
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(os.getenv("EVENTRESULTS")),
        help="Path to save.",
    )
    args = parser.parse_args()
    main(args)
