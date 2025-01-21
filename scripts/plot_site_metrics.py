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

events = [1]


ROW_NAMES = {
    "SPL_rms": "$L_{p,\\text{rms}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
    "SPL_pk": "$L_{p,\\text{pk}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
    "SPL_pkpk": "$L_{p,\\text{pk-pk}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
    "SEL": "$L_{E,p}$\n(dB re 1 $\\mathrm{\\mu Pa^2 s}$)",
    "SEL_cum": "cSEL\n(dB re 1 $\\mathrm{\\mu Pa^2 s}$)",
    "t_rise": "$t_\\mathrm{r}$ (s)",
    "t_decay": "$t_\\mathrm{d}$ (s)",
    "kurtosis": "$\\beta_2$",
}


STATIONS = [
    "Tetra",
    "VLA1",
    "VLA2",
]
YLIM_SCALE = 1.1


def plot_event_metrics():

    for event in events:

        fig, axs = plt.subplots(
            nrows=len(ROW_NAMES),
            ncols=len(STATIONS),
            figsize=(8, 10),
            gridspec_kw={"hspace": 0.1, "wspace": 0.05},
        )

        for i, (row_key, row_name) in enumerate(ROW_NAMES.items()):

            min_val = np.inf
            max_val = 0.0

            for j, station in enumerate(STATIONS):

                index = np.arange(1, 10)
                data = np.arange(1, 10) * (i + 1)

                xlim_margin = 0.1 * (index.max() - index.min())
                xlim = (index.min() - xlim_margin, index.max() + xlim_margin)

                if max_val < max(data):
                    max_val = max(data)
                if min_val > min(data):
                    min_val = min(data)

                ax = axs[i, j]

                ax.plot(index, data)

                ax.grid()
                ax.set_xlim(xlim)

                # Control column titles
                if i == 0:
                    ax.set_title(station)

                # Control row titles & ytick labels
                if j == 0:
                    ax.set_ylabel(row_name)
                else:
                    ax.set_yticklabels([])

                # Control x-axis ticks and labels
                if i == len(ROW_NAMES) - 1:
                    ax.set_xlabel("Strike Index")
                else:
                    ax.set_xticklabels([])

            # Control y-axis limits
            for j in range(len(STATIONS)):
                ylim_margin = 0.1 * (max_val - min_val)
                axs[i, j].set_ylim(min_val - ylim_margin, max_val + ylim_margin)
        

        plt.show()


def main(args: argparse.Namespace) -> None:
    df = pl.read_csv(args.input)
    print(df)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot site metrics from CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("METRICSFILE")),
        help="Path to the input CSV file.",
    )
    args = parser.parse_args()
    main(args)
