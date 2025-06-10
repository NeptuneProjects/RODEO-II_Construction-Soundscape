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

from rodeo import plotting

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
plt.style.use(Path("config/jasa_fig_style.mplstyle"))


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

    return event_name, np.array(distances), data


def retrieve_ambient_noise(grp: h5py.Group) -> np.ndarray:
    data = []
    for i, grp_sta in enumerate(grp.values()):
        if i == 0:
            freq = grp_sta["freqs"][:]
        data.append(grp_sta["psd"][:])

    return freq, np.array(data)


def main(args: argparse.Namespace) -> None:
    # Load PSD:
    with h5py.File(args.psddb, "r") as fpsd, h5py.File(args.ambdb, "r") as famb:
        freq = fpsd["f"][:]
        for grp in fpsd.values():
            logging.info(f"Processing {grp.name}")
            if grp.name == "/f":
                continue
            event, distances, data = retrieve_data(grp)

            # Extract ambient noise data
            freq_amb, data_amb = retrieve_ambient_noise(famb[event])

            if data.shape[0] < 3:
                nan_array = np.full(
                    (1, data.shape[1], data.shape[2], data.shape[3]), np.nan
                )
                data = np.concatenate((nan_array, data), axis=0)
                distances = np.insert(distances, 0, 0.0)
            if data_amb.shape[0] < 3:
                nan_array = np.full((1, data_amb.shape[1], data_amb.shape[2]), np.nan)
                data_amb = np.concatenate((nan_array, data_amb), axis=0)

            fig = plotting.plot_psd_pdf(
                freq, data, freq_amb, data_amb, event, distances
            )
            savepath = Path(os.getenv("EVENTRESULTS")) / f"{event}" / "psd.png"
            fig.savefig(savepath, dpi=600, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot empirical PDFs of PSD at each station."
    )
    parser.add_argument(
        "--psddb", type=Path, default=os.getenv("PSDFILE"), help="Path to the database."
    )
    parser.add_argument(
        "--ambdb", type=Path, default=os.getenv("AMBFILE"), help="Path to the database."
    )
    args = parser.parse_args()
    main(args)
