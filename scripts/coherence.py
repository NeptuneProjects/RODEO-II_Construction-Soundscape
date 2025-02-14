#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ThreadPoolExecutor as Executor
from dataclasses import dataclass
from itertools import product
import logging
import os
from pathlib import Path

import dotenv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from tqdm import tqdm
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import TIME_PRECISION

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)


@dataclass
class Record:
    event: str
    station: str
    channel: int
    time_diff: np.ndarray
    corr_func: np.ndarray
    corr_lags: np.ndarray
    coherence: np.ndarray
    coh_freq: np.ndarray


def parse_metadata(grp: h5py.Group) -> DataStreamStats:
    attrs = grp.attrs
    return DataStreamStats(
        channels=attrs.get("channel"),
        time_init=np.datetime64(attrs.get("time_init")[:-1], TIME_PRECISION),
        time_end=np.datetime64(attrs.get("time_end")[:-1], TIME_PRECISION),
        sampling_rate=attrs.get("sampling_rate"),
        units=attrs.get("unit"),
    )


def condition_signal(stream: DataStream, target_fs: float) -> DataStream:
    dec_factor = int(stream.stats.sampling_rate // target_fs)
    ds = stream.decimate(dec_factor)
    ds.filter("highpass", freq=10.0)
    return ds


def process_pair(pair):
    (i, det_group1), (j, det_group2) = pair
    logging.info(f"Processing pair {i}, {j}")

    ds1 = condition_signal(
        DataStream(
            stats=parse_metadata(det_group1),
            data=det_group1["waveform"][:],
        ),
        target_fs=args.target_fs,
    )
    ds2 = condition_signal(
        DataStream(
            stats=parse_metadata(det_group2),
            data=det_group2["waveform"][:],
        ),
        target_fs=args.target_fs,
    )

    sig1 = ds1.data.squeeze()
    sig2 = ds2.data.squeeze()

    # Compute time difference
    dt = (
        ds2.stats.time_init - ds1.stats.time_init
    ) / np.timedelta64(1, "s")

    # Compute cross-correlation:
    cf = signal.correlate(sig1, sig2, mode="full")
    lags = signal.correlation_lags(
        len(sig1), len(sig2), mode="full"
    )

    # Compute coherence:
    freq, coh = signal.coherence(sig1, sig2)

    return i, j, dt, cf, lags, coh, freq


def main(args: argparse.Namespace) -> None:
    event_dirs = args.input.iterdir()

    for event_dir in event_dirs:
        event_name = event_dir.name.split("_")[0]
        logging.info(f"Processing event {event_name}.")

        dbfiles = event_dir.glob("*.hdf5")

        for dbfile in dbfiles:
            logging.info(f"Loaded {dbfile}")
            with h5py.File(dbfile, "r") as file:
                for chan_group in file.values():
                    detection_pairs = list(
                        product(enumerate(chan_group.values()), repeat=2)
                    )
                    num_detections = len(chan_group)
                    time_diff, corr_func, corr_lags, coherence, coh_freq = (
                        None,
                        None,
                        None,
                        None,
                        None,
                    )

                    with Executor(max_workers=args.max_workers) as executor:
                        results = list(tqdm(executor.map(process_pair, detection_pairs), total=len(detection_pairs)))

                    for i, j, dt, cf, lags, coh, freq in results:
                        if time_diff is None:
                            time_diff = np.zeros((num_detections, num_detections))
                        if corr_func is None:
                            corr_func = np.zeros((num_detections, num_detections, len(cf)))
                        if corr_lags is None:
                            corr_lags = np.zeros((num_detections, num_detections, len(lags)))
                        if coherence is None:
                            coherence = np.zeros((num_detections, num_detections, len(coh)))
                        if coh_freq is None:
                            coh_freq = freq

                        time_diff[i, j] = dt
                        corr_func[i, j] = cf
                        corr_lags[i, j] = lags
                        coherence[i, j] = coh

                        # if j == 100:
                        #     plt.figure(figsize=(10, 6))
                        #     plt.subplot(311)
                        #     plt.plot(ds1.time_vector, ds1.data.squeeze())
                        #     plt.show()
                        #     return
                    # record = Record(
                    #     event=event_name,
                    #     station=ds1.stats.channel,
                    #     channel=ds1.stats.channel,
                    #     time_diff=time_diff,
                    #     corr_func=corr_func,
                    #     corr_lags=corr_lags,
                    #     coherence=coherence,
                    #     coh_freq=coh_freq,
                    # )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the coherence between two signals."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input HDF5 file containing the signals.",
        default=Path(os.getenv("DBPATH")),
    )
    parser.add_argument(
        "--target_fs",
        type=float,
        help="Target sampling rate for the signals.",
        default=1000.0,
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Number of parallel workers.",
        default=8,
    )
    args = parser.parse_args()
    main(args)
