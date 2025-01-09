# -*- coding: utf-8 -*-

import re

import bathyreq
import cmasher as cmr
import cmocean as cmo
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.basemap import Basemap
import numpy as np
import pandas as pd
from scipy import signal
from tritonoa.data.stream import DataStream

from rodeo import detect


def plot_bathy(
    data: np.ndarray,
    lonvec: np.ndarray,
    latvec: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:

    if ax is None:
        ax = plt.gca()

    cmap = cmr.get_sub_cmap("cmo.deep", 0.0, 1.0)

    im = ax.imshow(
        np.abs(data),
        extent=(lonvec[0], lonvec[-1], latvec[0], latvec[-1]),
        cmap=cmap,
    )
    CS = ax.contour(
        lonvec, latvec, np.abs(np.flipud(data)), colors="k", levels=10, linewidths=0.5
    )
    ax.clabel(CS, inline=1, fmt="%1.0f", fontsize=plt.rcParams["font.size"] - 2)
    return im, ax


def plot_ctd_profiles(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 3),
    gridspec_kw: dict = {"wspace": 0.1},
    dpi: int = 150,
    **plot_kwargs,
) -> plt.Figure:
    fig, axs = plt.subplots(figsize=figsize, ncols=3, gridspec_kw=gridspec_kw, dpi=dpi)
    ylim = (0, 35)

    ax0 = axs[0]
    ax0.plot(df["Temperature"], df["Depth"], **plot_kwargs["Temperature"])
    ax0.set_ylim(ylim)
    ax0.invert_yaxis()
    ax0.grid()
    ax0.set_xlabel("Temp. (C)")
    ax0.set_ylabel("Depth (m)")

    ax1 = axs[1]
    ax1.plot(df["Salinity"], df["Depth"], **plot_kwargs["Salinity"])
    ax1.set_ylim(ylim)
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_yticklabels([])
    ax1.set_xlabel("Sal. (PSU)")

    ax2 = axs[2]
    ax2.plot(df["Speed of sound"], df["Depth"], **plot_kwargs["Speed of sound"])
    ax2.set_ylim(ylim)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_yticklabels([])
    ax2.set_xlabel("Sound speed (m/s)")
    return fig


def plot_avg_power_timehist(
    mean_hist: np.ndarray,
    bins: np.ndarray,
    events: list[dict],
    title: str | None = None,
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
    n_traces: int | None = None,
    title: str | None = None,
    channel: int | None = None,
) -> plt.Figure:
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    n_traces = corr_matrix.shape[0] if n_traces is None else n_traces
    [ax.plot(time_matrix[k], corr_matrix[k], "k-", alpha=0.5) for k in range(n_traces)]
    ax.set_xlim(0, 33)
    ax.set_ylim(0.6, 1.01)
    ax.set_xlabel("Time elapsed from pulse [s]")
    ax.set_ylabel("Normalized Correlation")
    ax.grid(True)
    ax.set_title(f"{n_traces} pulses from {title} Channel {channel}")
    return fig


def plot_pulse_stack(
    sig_matrix: np.ndarray,
    corr_matrix: np.ndarray,
    fs: float,
    title: str | None = None,
    channel: int | None = None,
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

    summed = sig_matrix.mean(axis=0)
    tvec = np.arange(0, sig_matrix.shape[1]) / fs
    [ax.plot(tvec, sig_matrix[i], "k", alpha=0.5) for i in range(10)]
    ax.plot(tvec, summed, "r", linewidth=2, zorder=50)
    ax.set_xlim(0, max(tvec))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True)
    return fig, summed


def plot_inst_power_timehist(
    hist_data: np.ndarray,
    title: str | None = None,
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
    # return ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap="inferno", vmin=vmin, vmax=vmax)
    extent = (t[0], t[-1], f[0], f[-1])
    return ax.imshow(
        10 * np.log10(Sxx),
        extent=extent,
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )


def plot_spectrograms(
    ds: DataStream,
    title: str | None = None,
    tperseg: int = 1,
    overlap: float = 0.75,
    nfft: int | None = None,
    fmin: float | None = None,
    fmax: float | None = None,
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
            ds.data[i],
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
    ymin: float | None = None,
    ymax: float | None = None,
    fig_size: tuple[float] = (12, 6),
    title: str | None = None,
):
    fig, axs = plt.subplots(
        ds.num_channels, 1, figsize=fig_size, gridspec_kw={"hspace": 0.25}
    )
    fig.suptitle(title)

    for i, channel in enumerate(ds.stats.channels):
        ax = axs[i]
        ax.plot(ds.time_vector, ds.data[i])
        ax.set_xlim(ds.time_vector[0], ds.time_vector[-1])

        ax.set_ylabel(f"Channel {channel}")
        if channel == ds.stats.channels[-1]:
            ax.set_xlabel("Time")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    if ymin is None:
        ymin = ds.data.min()
    if ymax is None:
        ymax = ds.data.max()
    [ax.set_ylim(ymin, ymax) for ax in axs]

    return fig


def plot_detection_diagnostic(
    ds: DataStream,
    stwin: float | list[float] = 0.025,
    ltwin: float | list[float] = 5.0,
    threshold: float | list[float] = 2.25,
    distance: float | list[float] = 0.75,
    taper: float = 5.0,
    sensor: str = "UnkSta",
    channel: int | None = None,
    figsize: tuple[float] = (24, 6),
    title: str | None = None,
) -> list[plt.Figure]:
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    if channel is not None:
        channels = [channel]
        num_channels = 1
    else:
        channels = ds.stats.channels
        num_channels = ds.num_channels

    if not isinstance(stwin, list):
        stwin = [stwin] * num_channels
    if not isinstance(ltwin, list):
        ltwin = [ltwin] * num_channels
    if not isinstance(threshold, list):
        threshold = [threshold] * num_channels
    if not isinstance(distance, list):
        distance = [distance] * num_channels

    fig_list = []
    for i, chan in enumerate(channels):
        print(f"Processing channel {chan}...")

        data = detect.apply_linear_taper(ds.data[chan], int(fs * taper))
        chf = detect.characteristic_function(
            data, stwin=int(fs * stwin[i]), ltwin=int(fs * ltwin[i])
        )
        peaks, _ = signal.find_peaks(
            chf, height=threshold[i], distance=distance[i] * fs
        )

        if title is None:
            title = (
                f"{sensor} | Channel {chan} | STWIN: {stwin[i]} | LTWIN: {ltwin[i]} | "
                f"THRESHOLD: {threshold[i]} | DISTANCE: {distance[i]} | N_PEAKS: {len(peaks)}"
            )

        fig, axs = plt.subplots(
            nrows=2, ncols=1, figsize=figsize, gridspec_kw={"hspace": 0.25}, sharex=True
        )
        fig.suptitle(title)

        ax = axs[0]
        ax.plot(time, data)
        # ax.set_xticklabels([])
        ax.set_ylabel(f"Pressure [{ds.stats.units}]")

        # ax = axs[1]
        # ax.plot(time, detect.rms(data, int(stwin[i] * fs)))
        # ax.set_xticklabels([])
        # ax.set_ylabel("RMS")

        ax = axs[1]
        ax.plot(time, chf)
        ax.plot(time[peaks], chf[peaks], "rx")
        ax.axhline(threshold[i], color="r", linestyle="--")
        # ax.xaxis.set_tick_params(labelbottom=True, )
        ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Characteristic\nFunction")

        fig_list.append(fig)

        print(f"Channel {chan} complete.")

    return fig_list


def plot_detector_output(
    ds: DataStream,
    stwin: float | list[float] = 0.025,
    ltwin: float | list[float] = 5.0,
    threshold: float | list[float] = 2.25,
    distance: float | list[float] = 0.75,
    taper: float = 5.0,
    sensor: str = "UnkSta",
    channel: int | None = None,
    zoom_start: int = 0,
    zoom_end: int = 1000,
    figsize: tuple[float] = (8, 3),
    title: str | None = None,
) -> list[plt.Figure]:
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    data = detect.apply_linear_taper(ds.data[channel], int(fs * taper))
    chf = detect.characteristic_function(
        data, stwin=int(fs * stwin), ltwin=int(fs * ltwin)
    )
    peaks, _ = signal.find_peaks(chf, height=threshold, distance=distance * fs)
    xlim = (time.min(), time.max())


    # Compute zoomed section:
    data_zoom = data[zoom_start:zoom_end]
    time_zoom = time[zoom_start:zoom_end]
    chf_zoom = chf[zoom_start:zoom_end]
    peaks_zoom = peaks[(peaks > zoom_start) & (peaks < zoom_end)] - zoom_start

    if title is None:
        title = (
            f"{sensor} | Channel {channel} | STWIN: {stwin} | LTWIN: {ltwin} | "
            f"THRESHOLD: {threshold} | DISTANCE: {distance} | N_PEAKS: {len(peaks)}"
        )

    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=figsize,
        gridspec_kw={"hspace": 0.08, "wspace": 0.12, "width_ratios": [3, 2]},
    )
    fig.suptitle(title)

    # Time series, full
    ax = axs[0, 0]
    ax.plot(time, data, c="k")
    ax.axvspan(time[zoom_start], time[zoom_end], facecolor="gray", alpha=0.5, edgecolor="none")
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xticklabels([])
    ax.set_ylabel(f"Pressure [{ds.stats.units}]")

    # Time series, zoomed
    ax = axs[0, 1]
    ax.plot(time_zoom, data_zoom, c="k")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.set_xticklabels([])

    # Characteristic function, full
    ax = axs[1, 0]
    ax.plot(time, chf, c="k")
    ax.plot(time[peaks], chf[peaks], "rx")
    ax.axhline(threshold, color="r", linestyle="--")
    ax.axvspan(time[zoom_start], time[zoom_end], facecolor="gray", alpha=0.5, edgecolor="none")
    ax.set_xlim(xlim)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(0, chf.max() * 1.1)
    ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
    ax.set_xlabel("Time")
    ax.set_ylabel("Characteristic\nFunction")

    # Characteristic function, zoomed
    ax = axs[1, 1]
    ax.plot(time_zoom, chf_zoom, c="k")
    ax.plot(time_zoom[peaks_zoom], chf_zoom[peaks_zoom], "rx")
    ax.axhline(threshold, color="r", linestyle="--")
    locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.set_ylim(0, chf_zoom.max() * 1.1)

    return fig


def plot_study_area(
    bounds: list[list[float]],
    turbines_df: pd.DataFrame,
    geo_equipment_df: pd.DataFrame,
    vla_equipment_df: pd.DataFrame,
) -> plt.Figure:
    req = bathyreq.BathyRequest()
    data, lonvec, latvec = req.get_area(
        longitude=bounds[0], latitude=bounds[1], size=[400, 400]
    )

    fig = plt.figure(figsize=(3.37, 2), dpi=300)
    ax = plt.gca()

    m = Basemap(
        projection="tmerc",
        llcrnrlat=bounds[1][0],
        urcrnrlat=bounds[1][1],
        llcrnrlon=bounds[0][0],
        urcrnrlon=bounds[0][1],
        resolution="i",
        lon_0=np.mean(lonvec),
        lat_0=np.mean(latvec),
    )
    m.drawmeridians(
        np.arange(bounds[0][0], bounds[0][1], 0.1), labels=[0, 0, 1, 0], ax=ax
    )
    m.drawparallels(
        np.arange(bounds[1][0], bounds[1][1], 0.05), labels=[1, 0, 0, 0], ax=ax
    )
    xlim = m(np.array(bounds[0]), np.ones_like(bounds[0]) * np.mean(bounds[1]))[0]
    ylim = m(np.ones_like(bounds[0]) * np.mean(bounds[0]), np.array(bounds[1]))[1]

    im, ax = plot_bathy(
        data,
        lonvec=m(lonvec, np.ones_like(lonvec) * np.mean(latvec))[0],
        latvec=m(np.ones_like(latvec) * np.mean(lonvec), latvec)[1],
        ax=ax,
    )
    [
        ax.scatter(
            *m(turbines_df.iloc[i]["lon"], turbines_df.iloc[i]["lat"]),
            marker="h",
            c="yellow",
            edgecolors="k",
            # linewidth=1,
            # s=150,
            zorder=50,
            label=turbines_df.iloc[i]["Turbine"],
        )
        for i in range(len(turbines_df))
    ]
    texts = [
        ax.text(
            *m(turbines_df.iloc[i]["lon"], turbines_df.iloc[i]["lat"] + 0.004),
            turbines_df.iloc[i]["Turbine"],
            # fontsize=plt.rcParams["legend.fontsize"],
            ha="right",
            va="bottom",
            zorder=50,
            bbox=dict(facecolor="w", alpha=0.85, edgecolor="none", pad=0.5),
        )
        for i in range(len(turbines_df))
    ]
    ax.scatter(
        *m(geo_equipment_df["Longitude [dd]"], geo_equipment_df["Latitude [dd]"]),
        marker="^",
        c="tab:green",
        edgecolors="k",
        # linewidth=1,
        # s=150,
        zorder=50,
        label="Geosled",
    )
    ax.scatter(
        *m(
            vla_equipment_df.iloc[0]["Longitude [dd]"],
            vla_equipment_df.iloc[0]["Latitude [dd]"],
        ),
        marker="v",
        c="tab:orange",
        edgecolors="k",
        # linewidth=1,
        # s=150,
        zorder=50,
        label="VLA 1",
    )
    ax.scatter(
        *m(
            vla_equipment_df.iloc[1]["Longitude [dd]"],
            vla_equipment_df.iloc[1]["Latitude [dd]"],
        ),
        marker="v",
        c="tab:red",
        edgecolors="k",
        # linewidth=1,
        # s=150,
        zorder=50,
        label="VLA 2",
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # leg = ax.legend(facecolor="white", edgecolor="black", loc="upper right")
    # Make the legend so it only shows one symbol for all the turbines
    handles, labels = ax.get_legend_handles_labels()
    labels = ["Turbines", "Geosled", "VLA1", "VLA2"]
    # handles = handles[:3]
    leg = ax.legend(
        handles[-4:], labels, facecolor="white", edgecolor="black", loc="upper right"
    )
    leg.get_frame().set_alpha(None)

    scalebar = AnchoredSizeBar(
        ax.transData,
        5000,
        "5 km",
        "lower right",
        pad=0.1,
        color="k",
        frameon=True,
        size_vertical=200,
    )
    ax.add_artist(scalebar)

    return fig
