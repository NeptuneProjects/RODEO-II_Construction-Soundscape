#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from concurrent.futures import ProcessPoolExecutor
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


def form_pairs(grp: h5py.Group) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    return list(product(enumerate([v.name for v in grp.values()]), repeat=2))


def data_shape(
    dbfile: Path,
) -> tuple[list[tuple[tuple[int, str], tuple[int, str]]], list[int], list[int]]:
    detection_pairs = []
    channels = []
    num_detections = []
    with h5py.File(dbfile, "r") as file:
        logging.info(f"Loaded {dbfile}")
        for chan_group in file.values():
            detection_pairs.append(form_pairs(chan_group))
            channels.append(int(chan_group.name[-1]))
            num_detections.append(len(chan_group))

    return detection_pairs, channels, num_detections


def process_pair(pair: tuple, target_fs: float, dbpath: Path):
    (i, key1), (j, key2) = pair


    with h5py.File(dbpath, "r") as file:
        det_group1 = file[key1]
        det_group2 = file[key2]

        ds1 = condition_signal(
            DataStream(
                stats=parse_metadata(det_group1),
                data=det_group1["waveform"][:],
            ),
            target_fs=target_fs,
        )
        ds2 = condition_signal(
            DataStream(
                stats=parse_metadata(det_group2),
                data=det_group2["waveform"][:],
            ),
            target_fs=target_fs,
        )

    sig1 = ds1.data.squeeze()
    sig2 = ds2.data.squeeze()

    # Compute time difference
    dt = (ds2.stats.time_init - ds1.stats.time_init) / np.timedelta64(1, "s")

    # Compute cross-correlation:
    cf = signal.correlate(sig1, sig2, mode="full")
    lags = signal.correlation_lags(len(sig1), len(sig2), mode="full")

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
            
            detection_pairs, channels, num_detections = data_shape(dbfile)

            for det_pairs, channel, num_det in zip(
                detection_pairs, channels, num_detections
            ):
                logging.info(f"Processing channel {channel}.")

                time_diff, corr_func, corr_lags, coherence, coh_freq = (
                    None,
                    None,
                    None,
                    None,
                    None,
                )

                with ProcessPoolExecutor(max_workers=4) as executor:
                    logging.info(f"Submitting {len(det_pairs)} pairs.")
                    futures = [
                        executor.submit(process_pair, pair, args.target_fs, dbfile)
                        for pair in det_pairs
                    ]
                    logging.info(f"{len(det_pairs)} pairs submitted.")

                    for future in tqdm(futures, leave=True, total=len(futures)):
                        i, j, dt, cf, lags, coh, freq = future.result()

                        if time_diff is None:
                            time_diff = np.zeros((num_det, num_det))
                        if corr_func is None:
                            corr_func = np.zeros((num_det, num_det, len(cf)))
                        if corr_lags is None:
                            corr_lags = np.zeros((num_det, num_det, len(lags)))
                        if coherence is None:
                            coherence = np.zeros((num_det, num_det, len(coh)))
                        if coh_freq is None:
                            coh_freq = freq

                        time_diff[i, j] = dt
                        corr_func[i, j] = cf
                        corr_lags[i, j] = lags
                        coherence[i, j] = coh

            return




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
    args = parser.parse_args()
    main(args)
