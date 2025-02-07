#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Sequence

import dotenv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann
from tritonoa.data.reader import read_numpy
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_CONVERSION_FACTOR

dotenv.load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))

STATION_ORDER = ["SHRU921_Tetra", "SHRU905_VLA1", "SHRU920_VLA2"]


def parse_stations(files: list[Path]) -> list[str]:
    return ["/".join(file.stem.split("_")[1:]) for file in files]


def spectrogram_args(
    tperseg: float, fs: float, pcoverlap: float = 0.0
) -> tuple[int, np.ndarray, int]:
    nperseg = int(tperseg * fs)
    win = hann(nperseg)
    hop = max(int(nperseg * (1 - pcoverlap)), 1)
    return nperseg, win, hop


def plot_dispersion(
    streams: list[DataStream],
    stations: list[str],
    channel: int = 0,
    tperseg: int = 0.05,
    pcoverlap: float = 0.90,
    nfft_factor: float = 2,
    vmin: float = 80.0,
    vmax: float = 130.0,
    plot_freqs_between: Sequence[float] | None = (0.0, 500.0),
    filt: str = "highpass",
    filt_freq: float = 10.0,
) -> plt.Figure:
    fig = plt.figure(figsize=(4.5, 6))
    outer_gs = gridspec.GridSpec(3, 1, hspace=0.1)

    for i, ds in enumerate(streams):
        if filt is not None:
            ds.filter(filt, freq=filt_freq)
        data = ds.data[channel]
        if i == 0:
            fs = ds.stats.sampling_rate
            t_series = (ds.time_vector - ds.time_vector[0]).astype(
                float
            ) / TIME_CONVERSION_FACTOR
            xlim = (t_series.min(), t_series.max())

            nperseg, win, hop = spectrogram_args(tperseg, fs, pcoverlap)
            nfft = int(nperseg * nfft_factor)

        STFT = ShortTimeFFT(win, hop, fs, mfft=nfft, scale_to="psd")
        S = STFT.stft(data)
        SdB = 20 * np.log10(np.abs(S))
        f = STFT.f
        t_spec = STFT.t(len(data))

        # Plot only specified frequencies:
        if plot_freqs_between is not None:
            mask = (f >= plot_freqs_between[0]) & (f <= plot_freqs_between[1])
            f = f[mask]
            SdB = SdB[mask, :]

        inner_gs = outer_gs[i].subgridspec(2, 1, height_ratios=[1, 2], hspace=0.0)

        # Time series
        ax = fig.add_subplot(inner_gs[0])
        ax.plot(t_series, data / 1e6, zorder=20)
        ax.text(
            0.005, 0.96, stations[i], transform=ax.transAxes, va="top",
            bbox=dict(facecolor="white", edgecolor="black", pad=1)
        )
        ax.set_xlim(xlim)
        ax.grid(True)
        ax.set_xticklabels([])

        if i == 2:
            ax.set_ylabel("Pressure\n(Pa)")

        # Spectrogram
        ax = fig.add_subplot(inner_gs[1])
        ax.pcolormesh(t_spec, f, SdB, shading="auto", vmin=vmin, vmax=vmax, cmap="jet")
        ax.set_xlim(xlim)

        if i == 2:
            ax.set_xlabel("Reduced time (s)")
            ax.set_ylabel("Frequency\n(Hz)")
        else:
            ax.set_xticklabels([])

    # Add a colorbar to the bottom of the figure:
    cax = fig.add_axes([0.92, 0.25, 0.03, 0.5])
    cbar = fig.colorbar(ax.collections[0], cax=cax, orientation="vertical")
    cbar.set_label("PSD (dB re 1 $\\mu$Pa$^2$/Hz)")

    return fig


def main(args: argparse.Namespace) -> None:
    files = list(args.data.glob("*.npz"))
    files.sort(key=lambda x: STATION_ORDER.index("_".join(x.stem.split("_")[1:])))
    stations = parse_stations(files)
    streams = [read_numpy(file) for file in files]

    fig = plot_dispersion(
        streams,
        stations,
        channel=args.channel,
        tperseg=args.tperseg,
        pcoverlap=args.overlap,
        nfft_factor=args.nfft,
        vmin=args.vmin,
        vmax=args.vmax,
        filt=args.filter,
        filt_freq=args.filt_freq,
    )
    fig.savefig(
        args.output / "dispersion.png", bbox_inches="tight", facecolor="w", dpi=600
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot signal dispersion.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.getenv("DISPDATA")),
        help="Path to the data files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(os.getenv("RESULTS")), help="Output file."
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel to plot.")
    parser.add_argument("--tperseg", type=float, default=0.05, help="Time per segment.")
    parser.add_argument(
        "--overlap", type=float, default=0.90, help="Fractional overlap."
    )
    parser.add_argument("--nfft", type=float, default=2, help="NFFT factor.")
    parser.add_argument("--vmin", type=float, default=80.0, help="Minimum value.")
    parser.add_argument("--vmax", type=float, default=130.0, help="Maximum value.")
    parser.add_argument("--filter", type=str, default="highpass", help="Filter type.")
    parser.add_argument(
        "--filt_freq", type=float, default=10.0, help="Filter frequency."
    )
    args = parser.parse_args()
    main(args)
