#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from datautils.data import DataStream, read
from datautils.query import CatalogueQuery
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def plot_spectrogram(f, t, Sxx, ax=None, vmin=None, vmax=None):
    if ax is None:
        ax = plt.gca()
    return ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap="inferno", vmin=vmin, vmax=vmax)


def plot_spectrograms(
    ds: DataStream,
    title: Optional[str] = None,
    window: int = 1,
    overlap: float = 0.75,
    nfft: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    vmin: Optional[float] = 60,
    vmax: Optional[float] = 120,
    fig_size: tuple[float] = (8, 6),
):
    fs = ds.stats.sampling_rate

    window = 1
    window_size = int(window * fs)
    overlap = 0.75
    overlap_size = int(overlap * fs)
    if nfft is None:
        nfft = window_size

    fig, axs = plt.subplots(4, 1, figsize=fig_size, gridspec_kw={"hspace": 0.25})
    fig.suptitle(title)

    for i, channel in enumerate(ds.stats.channels):
        f, t, Sxx = signal.spectrogram(
            ds.waveform[:, i],
            fs=fs,
            nperseg=window_size,
            noverlap=overlap_size,
            nfft=nfft,
        )

        if fmin and fmax:
            Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
            f = f[(f >= fmin) & (f <= fmax)]
        elif fmin:
            Sxx = Sxx[f >= fmin, :]
            f = f[f >= fmin]
        elif fmax:
            Sxx = Sxx[f <= fmax, :]
            f = f[f <= fmax]

        ax = axs[i]
        im = plot_spectrogram(f, t, Sxx, ax=ax, vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Channel {channel}", fontsize=10, ha="left", x=0)
        if channel == ds.stats.channels[-1]:
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Frequency [Hz]")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"PSD [$\\mathrm{{{ds.stats.units}}} / \\mathrm{{Hz}}^2$]")

    return fig


def main(args):
    time_start = np.datetime64(args.time_start, "us")
    time_end = np.datetime64(args.time_end, "us")
    time_interval = np.timedelta64(args.time_interval, args.time_units)
    num_intervals = int((time_end - time_start) / time_interval)
    channels = [int(c) for c in args.channels.split(",")]

    for i in range(num_intervals):
        time_start_i = time_start + i * time_interval
        time_end_i = time_start + (i + 1) * time_interval

        cat_query = CatalogueQuery(
            catalogue=args.catalogue,
            destination=args.destination,
            time_start=time_start_i,
            time_end=time_end_i,
            channels=channels,
        )
        ds = read(cat_query)

        fig = plot_spectrograms(
            ds,
            title=f"{args.title}\n{np.datetime_as_string(time_start_i, unit="s")} - {np.datetime_as_string(time_end_i, unit="s")}",
            window=args.window,
            overlap=args.overlap,
            nfft=args.nfft,
            fmin=args.fmin,
            fmax=args.fmax,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        filename = (
            np.datetime_as_string(time_start_i, unit='s')
            .replace("-", "")
            .replace("T", "_")
            .replace(":", "") + \
            "_" + args.title.replace("/", "_")
        )
        fig.savefig(Path(args.destination / filename), bbox_inches="tight", facecolor="w")


if __name__ == "__main__":
    parser = ArgumentParser(description="Spectrogram plotting routine")
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
    parser.add_argument("--time_interval", help="Time interval", type=float, default=5)
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
    parser.add_argument("--title", help="Title of the plot", type=str, default="SHRU905/VLA1")
    parser.add_argument("--window", help="Window size in seconds", type=int, default=1)
    parser.add_argument(
        "--overlap", help="Overlap size in percentage", type=float, default=0.75
    )
    parser.add_argument("--nfft", help="Number of data points used in each block", type=int)
    parser.add_argument("--fmin", help="Minimum frequency in Hz", type=float, default=20.0)
    parser.add_argument("--fmax", help="Maximum frequency in Hz", type=float, default=300.0)
    parser.add_argument(
        "--vmin", help="Minimum value for the color scale", type=float, default=60
    )
    parser.add_argument(
        "--vmax", help="Maximum value for the color scale", type=float, default=120
    )

    args = parser.parse_args()
    main(args)
