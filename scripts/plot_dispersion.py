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
) -> plt.Figure:
    fig = plt.figure(figsize=(6.5, 8))
    outer_gs = gridspec.GridSpec(3, 1, hspace=0.1)

    for i, ds in enumerate(streams):
        data = ds.data[channel]
        if i == 0:
            fs = ds.stats.sampling_rate
            t_series = (ds.time_vector - ds.time_vector[0]).astype(float) / TIME_CONVERSION_FACTOR
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
        ax.text(0.01, 0.9, stations[i], transform=ax.transAxes, va="top")
        ax.set_xlim(xlim)
        ax.grid(True)

        # Spectrogram
        ax = fig.add_subplot(inner_gs[1])
        ax.pcolormesh(t_spec, f, SdB, shading="auto", vmin=vmin, vmax=vmax, cmap="jet")
        ax.set_xlim(xlim)

        if i == 2:
            ax.set_xlabel("Reduced time (s)")
            ax.set_ylabel("Frequency (Hz)")
        else:
            ax.set_xticklabels([])


    return fig


def main(args: argparse.Namespace) -> None:
    files = list(args.data.glob("*.npz"))
    files.sort(key=lambda x: STATION_ORDER.index("_".join(x.stem.split("_")[1:])))
    stations = parse_stations(files)
    streams = [read_numpy(file) for file in files]

    fig = plot_dispersion(streams, stations, args.channel)
    fig.savefig(args.output / "dispersion.png", bbox_inches="tight", facecolor="w", dpi=600)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot signal dispersion.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.getenv("DISPDATA")),
        help="Path to the data files.",
    )
    parser.add_argument("--output", type=Path, default=Path(os.getenv("RESULTS")), help="Output file.")
    parser.add_argument("--channel", type=int, default=0, help="Channel to plot.")
    args = parser.parse_args()
    main(args)
