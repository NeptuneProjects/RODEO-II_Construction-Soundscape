# -*- coding: utf-8 -*-

from pathlib import Path
import re
from typing import Optional, Union

from datautils.data import DataStream
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import detect


def plot_avg_power_timehist(
    mean_hist: np.ndarray,
    bins: np.ndarray,
    events: list[dict],
    title: Optional[str] = None,
    power_range: tuple[float] = (100, 160),
    channels=None,
    figsize: tuple[float] = (8, 10),
) -> plt.Figure:

    LINE_STYLES = {
        "Ambient": "-",
        "Soft Start": "--",
        "Full Power": ":",
    }

    fig, axs = plt.subplots(nrows=len(channels), ncols=1, figsize=figsize)

    for i in range(len(channels)):
        for j, event in enumerate(events):
            ax = axs[i]
            base_name = re.sub(r" \d+$", "", event["name"])
            ls = LINE_STYLES.get(base_name, "-")
            ax.plot(bins[0:-1], mean_hist[j, i, :], linestyle=ls, label=event["name"])
            ax.set_xlim(power_range)
            ax.text(
                0.0,
                1.02,
                f"Channel {channels[i]}",
                fontsize=10,
                ha="left",
                va="bottom",
                transform=ax.transAxes,
            )
            if i < len(channels) - 1:
                ax.set_xticklabels([])

            if i == 0:
                ax.set_title(title)
                ax.legend()

    ax.set_xlabel("Power [dB re 1 uPa$^2$]")
    ax.set_ylabel("PDF")

    return fig


def plot_pulse_coherence(
    time_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    n_peaks: int,
    title: str,
    channel: int,
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    # n_traces = corr_matrix.shape[0]
    n_traces = 20
    [ax.plot(time_matrix[k], corr_matrix[k], "k-", alpha=0.5) for k in range(n_traces)]
    ax.set_xlim(0, 33)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Time elapsed from pulse [s]")
    ax.set_ylabel("Normalized Correlation")
    ax.grid(True)
    ax.set_title(f"{n_traces} pulses from {title} Channel {channel}")
    return fig


def plot_pulse_stack(
    sig_matrix: np.ndarray, corr_matrix: np.ndarray, fs: float, title: str, channel: int
):
    fig, axs = plt.subplots(
        nrows=20, ncols=1, figsize=(10, 10), gridspec_kw={"hspace": 0.0}
    )

    tvec = np.arange(0, sig_matrix.shape[1]) / fs
    for i, ax in enumerate(axs):
        sig = sig_matrix[i]
        ax.plot(tvec, sig / max(abs(sig)), "k")

        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", which="both", bottom=False, top=False)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.set_yticklabels([])
        ax.grid(True)
        ax.set_xlim(0, max(tvec))
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"{corr_matrix[0, i]:.2f}", rotation=0, ha="right", va="center")
        if i == len(axs) - 1:
            ax.set_xlabel("Time [s]")
        else:
            ax.set_xticklabels([])
        if i == 0:
            ax.set_title(f"{title} Channel {channel}")

    return fig


def plot_summed_pulses(sig_matrix: np.ndarray, fs: float):
    fig = plt.figure(figsize=(15, 5))
    ax = plt.gca()

    summed = sig_matrix.sum(axis=0)
    tvec = np.arange(0, sig_matrix.shape[1]) / fs
    [
        ax.plot(tvec, sig_matrix[i] / max(abs(sig_matrix[i])), "k", alpha=0.5)
        for i in range(10)
    ]
    ax.plot(tvec, summed / max(abs(summed)), "r", linewidth=2, zorder=50)
    ax.set_xlim(0, max(tvec))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True)
    return fig, summed


def plot_inst_power_timehist(
    hist_data: np.ndarray,
    title: Optional[str] = None,
    power_range: tuple[float] = (100, 160),
    channels=None,
    fig_size: tuple[float] = (8, 10),
) -> plt.Figure:

    num_channels = hist_data.shape[0]
    num_intervals = hist_data.shape[1]
    vmax = hist_data.max()

    fig, axs = plt.subplots(
        hist_data.shape[0], 1, figsize=fig_size, gridspec_kw={"hspace": 0.3}
    )
    fig.suptitle(title)

    enumerator = enumerate(channels) if channels else enumerate(range(num_channels))

    for i, channel in enumerator:
        ax = axs[i]
        im = ax.imshow(
            hist_data[i].T,
            aspect="auto",
            origin="lower",
            cmap="jet",
            extent=[0, num_intervals, power_range[0], power_range[1]],
            interpolation="none",
            vmin=0,
            vmax=vmax,
        )

        ax.set_title(f"Channel {channel}", fontsize=10, ha="left", x=0)
        if channel == channels[-1]:
            ax.set_xlabel("Time Interval")
            ax.set_ylabel("Power [dB re 1 uPa$^2$]")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.set_yticklabels([])
            ax.set_ylabel("")

    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Frequency")

    return fig


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
    vmin: float = None,
    vmax: float = None,
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

        if vmin is None:
            vmin = Sxx.min()
        if vmax is None:
            vmax = Sxx.max()

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
            ax.set_xlabel("Time")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    if ymin is None:
        ymin = ds.waveform.min()
    if ymax is None:
        ymax = ds.waveform.max()
    [ax.set_ylim(ymin, ymax) for ax in axs]

    return fig


def plot_detection_diagnostic(
    ds: DataStream,
    stwin: Union[float, list[float]] = 0.025,
    ltwin: Union[float, list[float]] = 5.0,
    threshold: Union[float, list[float]] = 2.25,
    distance: Union[float, list[float]] = 0.75,
    taper: float = 5.0,
    figsize: tuple[float] = (24, 6),
    sensor: str = "UnkSta",
) -> list[plt.Figure]:
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    if not isinstance(stwin, list):
        stwin = [stwin] * ds.num_channels
    if not isinstance(ltwin, list):
        ltwin = [ltwin] * ds.num_channels
    if not isinstance(threshold, list):
        threshold = [threshold] * ds.num_channels
    if not isinstance(distance, list):
        distance = [distance] * ds.num_channels

    fig_list = []
    for i, channel in enumerate(ds.stats.channels):
        print(f"Processing channel {channel}...")

        title = f"{sensor} | Channel {channel} | STWIN: {stwin[i]} | LTWIN: {ltwin[i]} | THRESHOLD: {threshold[i]} | DISTANCE: {distance[i]}"

        data = detect.apply_linear_taper(ds.waveform[:, i], int(fs * taper))
        chf = detect.characteristic_function(data, stwin=int(fs * stwin[i]), ltwin=int(fs * ltwin[i]))
        peaks, _ = signal.find_peaks(chf, height=threshold[i], distance=distance[i] * fs)

        fig, axs = plt.subplots(
            nrows=3, ncols=1, figsize=figsize, gridspec_kw={"hspace": 0.25}
        )
        fig.suptitle(title)

        ax = axs[0]
        ax.plot(time, data)
        ax.set_xticklabels([])
        ax.set_ylabel(f"Pressure [{ds.stats.units}]")

        ax = axs[1]
        ax.plot(time, detect.rms(data, int(stwin[i] * fs)))
        ax.set_xticklabels([])
        ax.set_ylabel("RMS")

        ax = axs[2]
        ax.plot(time, chf)
        ax.plot(time[peaks], chf[peaks], "rx")
        ax.axhline(threshold[i], color="r", linestyle="--")
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic\nFunction")

        fig_list.append(fig)

        print(f"Channel {channel} complete.")

    return fig_list
