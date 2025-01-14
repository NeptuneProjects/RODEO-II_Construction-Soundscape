#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from itertools import product
import logging
import os
from pathlib import Path
import tomllib

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION

from rodeo.metrics import REF_PRESSURE, rl_rms

load_dotenv()

TIME_BEFORE_MINUTES = 11
DURATION_MINUTES = 10
STEP_MINUTES = 1


def start_end_time(event: pd.Series) -> tuple[np.datetime64, np.datetime64]:
    start = np.datetime64(event["Start"], TIME_PRECISION) - np.timedelta64(
        TIME_BEFORE_MINUTES, "m"
    )
    end = start + np.timedelta64(DURATION_MINUTES, "m")
    return start, end


def compute_rlrms_segment(
    ds: DataStream, duration: int = 1, unit: str = "s"
) -> np.ndarray:
    fs = ds.stats.sampling_rate
    duration = np.timedelta64(duration, unit)

    rlrms = []

    # fig = plt.figure()
    # plt.plot(ds.time_vector[-1000:], ds.data[0, -1000:])
    # plt.grid(which="both", linestyle="--")
    # plt.savefig(f"waveform_full.png")
    # plt.close()

    for i, time_start in enumerate(np.arange(
        ds.stats.time_init, ds.stats.time_end - duration, duration
    )):
        time_end = time_start + duration
        ds_interval = ds.copy()
        ds_interval.trim(time_start, time_end)
        tmp = rl_rms(ds_interval.data)
        print("")
        print("Requested:", time_start, time_end)
        print("Obtained:", ds_interval.stats.time_init, ds_interval.stats.time_end)
        # rlrms.append(rl_rms(ds_interval.data))
        # print(ds_interval.data.shape)
        # print(tmp)
        # print(tmp.shape)

        # fig = plt.figure()
        # plt.plot(ds_interval.time_vector, ds_interval.data[0])
        # plt.savefig(f"waveform_{i}.png")
        # plt.close()

    return rlrms


def compute_psd(
    ds: DataStream,
    segment_dur: float = 10.0,
    filter: str | None = None,
    filt_freq: float | tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # Filter data
    if filter is not None:
        ds.filter(filter, freq=filt_freq)
    fs = ds.stats.sampling_rate

    # Compute nfft length based on segment_dur and fs:
    nperseg = int(segment_dur * fs)

    # Compute PSD using Welch's method
    freqs, psd = welch(
        ds.data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        nfft=nperseg,
        scaling="density",
    )
    return freqs, psd


def loop_over(df: pd.DataFrame, inv_config: dict) -> None:
    skip_events = {"Z01-A", "Z01-B", "Z01-C", "Z01-D", "A04"}
    skip_sensors = {
        "A14": {"SHRU921_Tetra"},
        "A11": {"SHRU921_Tetra"},
        "A02": {"SHRU921_Tetra"},
        "A15": {"SHRU921_Tetra"},
        "A13": {"SHRU921_Tetra"},
        "A01": {"SHRU921_Tetra"},
        "A07": {"SHRU921_Tetra", "SHRU905_VLA1"},
    }

    # Create a single loop using itertools.product
    for (i, event), (sensor_name, sensor_config) in product(
        df.iterrows(), inv_config.items()
    ):
        if event["Event"] in skip_events:
            continue
        if sensor_name in skip_sensors.get(event["Event"], set()):
            continue

        start, end = start_end_time(event)
        print(start, end)
        print("Reading inventory")
        ds = read_inventory(
            file_path=Path(sensor_config["data"]["destination"]),
            time_start=start,
            time_end=end,
        )
        print("Data loaded")
        print(ds.stats.time_init, ds.stats.time_end)
        print(ds.time_vector[-10:])

        RL_rms = compute_rlrms_segment(ds, duration=1, unit="m")

        # freqs, psd = compute_psd(
        #     ds,
        #     segment_dur=5.0,
        #     filter="highpass",
        #     filt_freq=10.0,
        # )
        # psd_db = 10 * np.log10(psd / (REF_PRESSURE ** 2))
        # fig = plt.figure()
        # [plt.plot(freqs, psd_db[j]) for j in range(ds.num_channels)]
        # plt.xscale("log")
        # plt.xlim(10, 10000)
        # plt.ylim(60, 110)
        # plt.grid(which="both", linestyle="--")
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Power Spectral Density (dB re 1 $\\mu Pa^2$/Hz)")
        # fig.savefig("spectrum.png")

        return


def main(args: argparse.Namespace) -> None:
    df_events = pd.read_csv(os.getenv("EVENTCONFIG"))
    with open(args.inv, "rb") as f:
        inv_config = tomllib.load(f)
    loop_over(df_events, inv_config)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inv",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    parser.add_argument("--event", type=Path, default=Path("config/events/A08.toml"))
    parser.add_argument("--filter", type=str, default="highpass")
    parser.add_argument("--filt_freq", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
