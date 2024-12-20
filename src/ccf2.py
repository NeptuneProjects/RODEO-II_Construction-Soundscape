#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from itertools import product
from pathlib import Path
import tomllib

from datautils.data import read
from datautils.query import CatalogueQuery
from datautils.time import TIME_PRECISION
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm

import master

EVENT_LOG_PATH = Path("data/logs/events.csv")
SENSORS_PATH = Path("src/config/data.toml")
SAVEPATH = Path("data/results/")

SENSORS = ["SHRU921_Tetra", "SHRU905_VLA1", "SHRU920_VLA2"]
CHANNELS = [0, 1, 2, 3]


def cross_correlation(
    sig1: np.ndarray,
    sig2: np.ndarray,
    step_size: int = 1,
    window_size: int | None = None,
):
    # Length of the signals
    signal_length = len(sig1)
    lags = correlation_lags(signal_length, signal_length, mode="full")

    if window_size is None:
        window_size = signal_length

    # Number of windows
    num_windows = (signal_length - window_size) // step_size + 1

    # Pre-allocate the cross-correlation results as a numpy matrix
    cross_corr_length = 2 * window_size - 1
    ccf = np.zeros((num_windows, cross_corr_length))

    for i, start in enumerate(range(0, signal_length - window_size + 1, step_size)):
        segment1 = sig1[start : start + window_size]
        segment2 = sig2[start : start + window_size]
        ccf[i] = correlate(segment1, segment2, mode="full")
    return ccf, lags


def parse_event_data(data: pd.Series) -> Path:
    event_folder = f"{data['Site']}_{data['Start'][0:10]}"
    metrics_path = SAVEPATH / event_folder / f"metrics_{SENSORS[0]}.csv"
    return metrics_path


def process_detections(
    inventory: list[dict], stations: list[str], channels: list[int], savepath: Path
) -> np.ndarray:
    sta_chan = list(product(stations, channels))
    sensor_pairs = list(product(sta_chan, sta_chan))

    # for pair in sensor_pairs:
    for pair in tqdm(
        sensor_pairs, desc="Processing pairs", unit="pair", total=len(sensor_pairs)
    ):
        (sta1, chan1), (sta2, chan2) = pair
        key = f"{sta1}_ch{chan1}__{sta2}_ch{chan2}"

        num_samples = inventory[0]["stream1"].waveform.shape[1]
        ccfs = np.zeros((len(inventory), 2 * num_samples - 1))
        fs = inventory[0]["stream1"].stats.sampling_rate

        for i, detection in enumerate(inventory):

            data1 = detection[f"stream{stations.index(sta1)+1}"].waveform[chan1]
            data2 = detection[f"stream{stations.index(sta2)+1}"].waveform[chan2]

            ccf, lags = cross_correlation(data1, data2)
            ccfs[i] = ccf

        savepath.mkdir(parents=True, exist_ok=True)
        np.savez(
            savepath / f"{key}_ccf.npz",
            key=key,
            ccfdata=ccfs,
            lags=lags,
            fs=fs,
        )

    return lags


def read_data(
    sensor_path: Path,
    channels: list,
    time_start: list[np.datetime64],
    time_end: list[np.datetime64],
    filter_type: str | None = None,
    freq1: float | None = None,
    freq2: float | None = None,
    dec_factor: int | None = None,
) -> list[dict]:
    with open(sensor_path, "rb") as f:
        sensor_info = tomllib.load(f)["data"]["sensors"]
    catalogue_paths = [master.get_catalogue(sensor_info, sn) for sn in SENSORS]

    inv = []
    for i, (t0, t1) in tqdm(
        enumerate(zip(time_start, time_end)),
        desc="Reading data",
        unit="detection",
        total=len(time_start),
    ):

        detection = {
            "index": i,
            "time_start": t0,
            "time_end": t1,
        }

        ds_list = []
        for j, path in enumerate(catalogue_paths):
            cq = CatalogueQuery(
                catalogue=path,
                destination=None,
                time_start=np.datetime64(t0, TIME_PRECISION),
                time_end=np.datetime64(t1, TIME_PRECISION),
                channels=channels,
            )
            ds = read(cq)
            if filter_type == "bandpass":
                ds = ds.filter(filt_type=filter_type, freqmin=freq1, freqmax=freq2)
            if filter_type in ["lowpass", "highpass"]:
                ds = ds.filter(filt_type=filter_type, freq=freq1)
            if dec_factor:
                ds = ds.decimate(dec_factor)

            # Transpose data to (channel, time) format
            ds.waveform = ds.waveform.T

            ds_list.append(ds)

        # Ensure all streams have the same number of samples
        num_samples = min([len(ds) for ds in ds_list])
        for j, ds in enumerate(ds_list):
            detection[f"stream{j+1}"] = ds[:num_samples]

        inv.append(detection)

    return inv


def read_detection_times(
    path: Path, duration: int
) -> tuple[list[np.datetime64], list[np.datetime64]]:
    df_times = pd.read_csv(path)
    df_times = df_times[df_times["channel"] == 0]["time_init"]
    time_start = [np.datetime64(t, TIME_PRECISION) for t in df_times]
    time_end = [
        time_start + np.timedelta64(duration, "ms") for time_start in time_start
    ]
    if len(time_start) != len(time_end):
        raise ValueError("Mismatched number of detection times.")
    return time_start, time_end


def save_sensor_pairs(time_start: list, time_end: list, savepath: Path) -> None:
    inv = read_data(
        SENSORS_PATH,
        CHANNELS,
        time_start,
        time_end,
        args.filter,
        args.filt_f1,
        args.filt_f2,
        args.decimate,
    )
    process_detections(inv, SENSORS, CHANNELS, savepath)


def main(args: argparse.Namespace) -> None:

    saved = args.saved

    event_data = pd.read_csv(EVENT_LOG_PATH).set_index("Event").loc[args.event]
    metrics_path = parse_event_data(event_data)
    time_start, time_end = read_detection_times(metrics_path, args.duration)
    # time_start = time_start[0:100]
    # time_end = time_end[0:100]
    savepath = metrics_path.parents[0] / "ccf"
    if not saved:
        save_sensor_pairs(time_start, time_end, savepath)
    else:
        pair = args.pair
        loaded_data = np.load(
            savepath / f"{pair}_ccf.npz",
            allow_pickle=True,
        )
        ccfdata = loaded_data["ccfdata"]
        lags = loaded_data["lags"]
        fs = loaded_data["fs"]

        lags = lags / fs

        plt.figure()
        for i in range(1500, 1800):
            plotccf = ccfdata[i]
            plotccf /= np.max(np.abs(plotccf))
            plt.plot(lags, 0.75 * plotccf + i, "k")
        egf = np.sum(ccfdata, axis=0) / np.max(np.abs(np.sum(ccfdata, axis=0)))
        plt.plot(lags, 0.75 * egf + i + 5, "r")
        plt.gca().invert_yaxis()
        plt.xlabel("Lag (s)")
        plt.ylabel("Detection")
        plt.title(f"CCF for {pair}")
        plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cross-correlation functions.")
    parser.add_argument(
        "--event", type=str, default="A08", help="The event ID to process."
    )
    parser.add_argument(
        "--duration", type=int, default=1200, help="Duration of each detection (ms)."
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["bandpass", "lowpass", "highpass"],
        default="bandpass",
        help="Filter the event data.",
    )
    parser.add_argument(
        "--filt_f1",
        type=float,
        default=10,
        help="Cutoff for low/hi-pass/bandpass (low).",
    )
    parser.add_argument(
        "--filt_f2", type=float, default=200, help="Cutoff for bandpass (high)."
    )
    parser.add_argument(
        "--decimate",
        type=int,
        default=20,
        help="Decimation factor for the data.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="SHRU921_Tetra_ch0__SHRU905_VLA1_ch1",
    )
    parser.add_argument(
        "--saved",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    main(args)
