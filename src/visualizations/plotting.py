# -*- coding: utf-8 -*-

from typing import Optional

from datautils.data import DataStream
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def plot_spectrogram(f, t, Sxx, ax=None, vmin=None, vmax=None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    return ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap="inferno", vmin=vmin, vmax=vmax)


def plot_spectrograms(
    ds: DataStream,
    title: Optional[str] = None,
    tperseg: int = 1,
    overlap: float = 0.75,
    nfft: Optional[int] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    vmin: float = 60,
    vmax: float = 120,
    fig_size: tuple[float] = (8, 6),
) -> plt.Figure:
    fs = ds.stats.sampling_rate

    nperseg = int(tperseg * fs)
    overlap_size = int(overlap * nperseg)
    if nfft is None:
        nfft = nperseg

    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=fig_size, gridspec_kw={"hspace": 0.3}
    )
    fig.suptitle(title)

    for i, channel in enumerate(ds.stats.channels):
        f, t, Sxx = signal.spectrogram(
            ds.waveform[:, i],
            fs=fs,
            nperseg=nperseg,
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


def plot_time_series(
    ds: DataStream,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    fig_size: tuple[float] = (12, 6),
    title: Optional[str] = None,
):
    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=fig_size, gridspec_kw={"hspace": 0.25}
    )
    fig.suptitle(title)

    for i, channel in enumerate(ds.stats.channels):
        ax = axs[i]
        ax.plot(ds.time_vector, ds.waveform[:, i])
        ax.set_xlim(ds.time_vector[0], ds.time_vector[-1])

        ax.set_ylabel(f"Channel {channel}")
        if channel == ds.stats.channels[-1]:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    if ymin is None:
        ymin = ds.waveform.min()
    if ymax is None:
        ymax = ds.waveform.max()
    [ax.set_ylim(ymin, ymax) for ax in axs]

    return fig
