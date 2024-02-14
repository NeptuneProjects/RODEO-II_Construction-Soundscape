# -*- coding: utf-8 -*-

from typing import Optional

from datautils.data import DataStream
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
