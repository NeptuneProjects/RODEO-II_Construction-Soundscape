#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from itertools import product
import logging
import os
from pathlib import Path
import tomllib
import time

from dotenv import load_dotenv
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION

from rodeo.metrics import spl_rms
from rodeo.utils import get_welch

load_dotenv()
logging.basicConfig(level=logging.INFO)
welch, cuda_enabled = get_welch()

TIME_BEFORE_MINUTES = 11
DURATION_MINUTES = 10
STEP_MINUTES = 1


def start_end_time(event: pd.Series) -> tuple[np.datetime64, np.datetime64]:
    start = np.datetime64(event["Start"], TIME_PRECISION) - np.timedelta64(
        TIME_BEFORE_MINUTES, "m"
    )
    end = start + np.timedelta64(DURATION_MINUTES, "m")
    return start, end


def compute_splrms_segment(
    ds: DataStream, duration: int = 1, unit: str = "s"
) -> np.ndarray:
    """Compute the RMS SPL of a data stream in segments of duration `duration`
    and unit `unit`.
    """
    duration = np.timedelta64(duration, unit)

    spl = []

    for time_start in np.arange(
        ds.stats.time_init, ds.stats.time_end - duration, duration
    ):
        time_end = time_start + duration
        ds_interval = ds.copy()
        ds_interval.trim(time_start, time_end)
        spl.append(spl_rms(ds_interval.data))

    return np.array(spl).T


def compute_psd(
    ds: DataStream,
    segment_dur: float = 10.0,
    pcoverlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Power Spectral Density (PSD) of a data stream using Welch's
    method.

    Args:
        ds: DataStream object.
        segment_dur: Duration of the segment in seconds. Defaults to 10.0.

    Returns:
        freqs: Array of frequencies.
        psd: Array of PSD values.
    """
    fs = ds.stats.sampling_rate

    # Compute nfft length based on segment_dur and fs:
    nperseg = int(segment_dur * fs)
    noverlap = int(nperseg * pcoverlap)

    # Compute PSD using Welch's method
    freqs, psd = welch(
        ds.data,
        fs=ds.stats.sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        scaling="density",
    )
    if cuda_enabled:
        freqs = freqs.get()
        psd = psd.get()
    return freqs, psd


def loop_over(
    df: pd.DataFrame,
    inv_config: dict,
    output: Path,
    filt: str | None = None,
    filt_freq: float | None = 10.0,
    segment_dur: float = 5.0,
    pcoverlap: float = 0.9,
) -> None:
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

    for (_, event), (sensor_name, sensor_config) in tqdm(
        product(df.iterrows(), inv_config.items()), total=len(df) * len(inv_config)
    ):
        event_name = event["Event"]
        if event_name in skip_events:
            continue
        if sensor_name in skip_sensors.get(event_name, set()):
            continue

        start, end = start_end_time(event)
        logging.info(f"Reading inventory for {event_name}/{sensor_name}")
        logging.info(f"{start} - {end}")
        ds = read_inventory(
            file_path=Path(sensor_config["data"]["destination"]),
            time_start=start,
            time_end=end,
        )
        fs = ds.stats.sampling_rate
        logging.info("Data loaded.")

        # Filter data
        if filt is not None:
            logging.info("Filtering data.")
            ds.filter(filt, freq=filt_freq)
            logging.info("Filtering complete.")

        # Compute RMS SPL over segments
        SPL_rms = compute_splrms_segment(ds, duration=1, unit="m")

        # Compute PSD
        logging.info("Computing PSD.")
        t0 = time.time()
        freqs, psd = compute_psd(
            ds,
            segment_dur=segment_dur,
            pcoverlap=pcoverlap,
        )
        logging.info(f"PSD computed: {time.time() - t0} s elapsed.")
        save_hdf5(
            event_name,
            sensor_name,
            SPL_rms,
            freqs,
            psd,
            output,
            filter=filt,
            filter_freq=filt_freq,
            nperseg=int(segment_dur * fs),
            noverlap=int(segment_dur * fs * pcoverlap),
        )


def save_hdf5(
    event: str,
    sensor_name: str,
    SPL_rms: np.ndarray,
    freqs: np.ndarray,
    psd: np.ndarray,
    output: Path,
    **kwargs,
) -> None:
    with h5py.File(output, "a") as f:
        if "filter" not in f.attrs:
            f.attrs.create("filter", kwargs.get("filter", None))
        if "filter_freq" not in f.attrs:
            f.attrs.create("filter_freq", kwargs.get("filter_freq", None))
        if "nperseg" not in f.attrs:
            f.attrs.create("nperseg", kwargs.get("nperseg", None))
        if "noverlap" not in f.attrs:
            f.attrs.create("noverlap", kwargs.get("noverlap", None))
        if event not in f:
            f.create_group(event)
        if sensor_name not in f[event]:
            f[event].create_group(sensor_name)
        f[event][sensor_name].create_dataset("SPL_rms", data=SPL_rms)
        f[event][sensor_name].create_dataset("freqs", data=freqs)
        f[event][sensor_name].create_dataset("psd", data=psd)


def main(args: argparse.Namespace) -> None:
    df_events = pd.read_csv(os.getenv("EVENTCONFIG"))
    with open(args.inv, "rb") as f:
        inv_config = tomllib.load(f)
    loop_over(
        df_events,
        inv_config,
        args.output,
        filt=args.filt,
        filt_freq=args.filt_freq,
        segment_dur=args.segment_dur,
        pcoverlap=args.pcoverlap,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inv",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    parser.add_argument("--filt", type=str, default="highpass")
    parser.add_argument("--filt_freq", type=float, default=10.0)
    parser.add_argument("--segment_dur", type=float, default=5.0)
    parser.add_argument("--pcoverlap", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=Path(os.getenv("AMBFILE")))
    args = parser.parse_args()
    main(args)
