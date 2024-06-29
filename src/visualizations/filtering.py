#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from datautils.data import read
from datautils.query import CatalogueQuery
from datautils.time import TIME_PRECISION
import matplotlib.pyplot as plt
import numpy as np

import plotting

catalogues = [
    # Path("data/catalogues/SHRU905_VLA1_FileInfo.csv"),
    # Path("data/catalogues/SHRU920_VLA2_FileInfo.csv"),
    Path("data/catalogues/SHRU921_Tetra_FileInfo.csv"),
]

# filter_args = [
#     {"filt_type": "lowpass", "freq": 50},
#     {"filt_type": "bandpass", "freqmin": 50, "freqmax": 350},
#     {"filt_type": "highpass", "freq": 350},
# ]


def main(args: argparse.Namespace) -> None:

    time_start = np.datetime64(args.time_start, TIME_PRECISION)
    time_end = np.datetime64(args.time_end, TIME_PRECISION)

    for i, catalogue in enumerate(catalogues):
        title = (
            f"{'_'.join(catalogue.name.split('_')[0:2])}\n"
            f"{np.datetime_as_string(time_start, unit='s')}"
            f" - {np.datetime_as_string(time_end, unit='s')}"
        )
        cq = CatalogueQuery(
            catalogue=catalogue,
            destination="data/interrogation",
            time_start=time_start,
            time_end=time_end,
            channels=[int(c) for c in args.channels.split(",")],
        )
        ds = read(cq)

        ds_low = ds.copy()
        ds_low.filter("lowpass", freq=5)
        
        ds_hi = ds.copy()
        ds_hi.filter("highpass", freq=350)

        plot_chan = 0
        samp_start = 0
        samp_end = -1
        
                
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(80, 10))
        
        ax = axs[0]
        ax.plot(ds.waveform[samp_start:samp_end, plot_chan])
        ax.set_title("Original")

        ax = axs[1]
        ax.plot(ds_low.waveform[samp_start:samp_end, plot_chan])
        ax.set_title("Lowpass Filtered")
        
        ax = axs[2]
        ax.plot(ds_hi.waveform[samp_start:samp_end, plot_chan])
        ax.set_title("Highpass Filtered")

        fig.tight_layout()

        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--time_start", help="Start time", type=str, default="2023-07-09T10:30:00"
    )
    parser.add_argument(
        "--time_end", help="End time", type=str, default="2023-07-09T10:30:05"
    )
    parser.add_argument(
        "--channels", help="Channels to be read", type=str, default="0,1,2,3"
    )
    args = parser.parse_args()
    main(args)
