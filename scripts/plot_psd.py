#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path

import dotenv
import h5py
import matplotlib.pyplot as plt
import numpy as np

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
plt.style.use(Path("config/jasa_fig_style.mplstyle"))


def plot_psd_pdf(
    freq: np.ndarray,
    data: np.ndarray,
    event_name: str = None,
    distance: list[float] = None,
) -> plt.Figure:

    # (3 channels, 4 stations, # detections, 4096 freq bins)

    channels = [0, 1, 2, 3]
    stations = ["SHRU921/Tetra", "SHRU905/VLA1", "SHRU920/VLA2"]

    fig, axs = plt.subplots(
        nrows=4, ncols=3, figsize=(8, 9), gridspec_kw={"hspace": 0.05, "wspace": 0.05}
    )

    for i, channel in enumerate(channels):
        for j, station in enumerate(stations):
            ax = axs[i][j]

            view = data[j, i]

            if np.isnan(view).all():
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                # Convert to dB
                view = 10 * np.log10(view)
                mean = np.mean(view, axis=0)
                std = np.std(view, axis=0)
                ucb = mean + 2 * std
                lcb = mean - 2 * std

                ax.plot(freq, mean, c="tab:red", lw=2, label="Mean")
                ax.fill_between(
                    freq, ucb, lcb, color="tab:red", alpha=0.25, label="95% CI"
                )

            ax.set_xscale("log")
            ax.set_xticks(
                [10, 100, 1000, 10000], ["$10^1$", "$10^2$", "$10^3$", "$10^4$"]
            )
            for ticklabel in ax.get_xticklabels():
                if ticklabel.get_text() == "$10^1$":
                    ticklabel.set_horizontalalignment("left")
                if ticklabel.get_text() == "$10^4$":
                    ticklabel.set_horizontalalignment("right")
            ax.set_xlim(10, 10000)
            ax.set_ylim(50, 150)
            ax.grid(True, which="both", ls="--", alpha=0.5)

            # Control column titles
            if i == 0:  # TODO: Handle no-data stations (see plot_site_metrics.py, L103)
                if distance[j] is None:
                    title = f"{station}"
                else:
                    title = f"{station} (R = {distance[j] / 1e3:.1f} km)"
                ax.set_title(title)

            # Control row titles
            if j == 0:
                ax.set_ylabel(
                    f"Channel {channel}\n PSD (dB re 1 $\\mathrm{{\\mu Pa^2/Hz}}$)"
                )
            else:
                ax.set_yticklabels([])

            # Control x-axis ticks and labels
            if i == len(channels) - 1:
                ax.set_xlabel("Frequency (Hz)")
            else:
                ax.set_xticklabels([])

    fig.suptitle(
        f"Event {event_name} | Total strikes: {view.shape[0]:,}",
        y=0.925,
    )
    return fig


def data_dimensions(grp: h5py.Group) -> tuple[int, int, int, int]:
    grp_sta = list(grp.values())[0]
    grp_chan = list(grp_sta.values())[0]
    grp_det = list(grp_chan.values())[0]

    num_stations = len(grp)
    num_channels = len(grp_sta)
    num_detections = len(grp_chan)
    num_freqbins = grp_det["Pxx"].shape[0]

    return num_stations, num_channels, num_detections, num_freqbins


def retrieve_data(
    grp: h5py.Group,
) -> tuple[str, list[str], list[float], list[str], np.ndarray]:
    event_name = grp.attrs["event_name"]

    dims = data_dimensions(grp)
    data = np.zeros(dims)
    distances = []
    for i, grp_sta in enumerate(grp.values()):
        distances.append(grp_sta.attrs["range"])
        for j, grp_chan in enumerate(grp_sta.values()):
            for k, grp_det in enumerate(grp_chan.values()):
                data[i, j, k, :] = grp_det["Pxx"][:]

    return event_name, distances, data


def main(args: argparse.Namespace) -> None:
    # Load ambient SPL:
    pass
    # Load PSD:
    with h5py.File(args.psddb, "r") as f:
        freq = f["f"][:]
        for grp in f.values():
            logging.info(f"Processing {grp.name}")
            if grp.name == "/f":
                continue
            event, distances, data = retrieve_data(grp)
            if data.shape[0] < 3:
                nan_array = np.full(
                    (1, data.shape[1], data.shape[2], data.shape[3]), np.nan
                )
                data = np.concatenate((nan_array, data), axis=0)
                distances = [None] + distances

            fig = plot_psd_pdf(freq, data, event, distances)
            savepath = Path(os.getenv("FIGPATH")) / "events" / f"{event}" / "psd.png"
            fig.savefig(savepath, dpi=600, bbox_inches="tight", facecolor="white")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot empirical PDFs of PSD at each station."
    )
    parser.add_argument(
        "--psddb", type=Path, default=os.getenv("PSDFILE"), help="Path to the database."
    )
    args = parser.parse_args()
    main(args)
