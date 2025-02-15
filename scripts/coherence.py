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
    sampling_rate: float
    time_diff: np.ndarray
    corr: np.ndarray
    lags: np.ndarray

    def save_h5(self, path: Path) -> None:
        with h5py.File(path, "a") as file:
            # Set "event" attr if not already set:
            if "event" not in file.attrs:
                file.attrs["event"] = self.event

            # Create a new group for the station if it doesn't exist:
            if self.station not in file:
                file.create_group(self.station)

            # Save the data:
            grp = file[f"{self.station}"]
            grp.attrs["event"] = self.event
            grp.attrs["station"] = self.station
            grp.attrs["sampling_rate"] = self.sampling_rate
            grp.create_dataset("time_diff", data=self.time_diff)
            grp.create_dataset("corr", data=self.corr)
            grp.create_dataset("lags", data=self.lags)


def compute_array_size(arrays: list[np.ndarray]) -> int:
    return sum([a.nbytes for a in arrays])


def condition_signal(stream: DataStream, target_fs: float) -> DataStream:
    dec_factor = int(stream.stats.sampling_rate // target_fs)
    ds = stream.decimate(dec_factor)
    ds.filter("highpass", freq=10.0)
    return ds


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

    print(num_detections)
    return detection_pairs, channels, num_detections


def form_pairs(grp: h5py.Group) -> list[tuple[tuple[int, str], tuple[int, str]]]:
    return list(product(enumerate([v.name for v in grp.values()]), repeat=2))


def parse_metadata(grp: h5py.Group) -> DataStreamStats:
    attrs = grp.attrs
    return DataStreamStats(
        channels=attrs.get("channel"),
        time_init=np.datetime64(attrs.get("time_init")[:-1], TIME_PRECISION),
        time_end=np.datetime64(attrs.get("time_end")[:-1], TIME_PRECISION),
        sampling_rate=attrs.get("sampling_rate"),
        units=attrs.get("unit"),
    )


def load_signals(
    dbpath: Path, keys: list[str], target_fs: float
) -> tuple[DataStream, DataStream]:
    streams = []
    with h5py.File(dbpath, "r") as file:
        for key in keys:
            grp = file[key]
            streams.append(
                condition_signal(
                    DataStream(
                        stats=parse_metadata(grp),
                        data=grp["waveform"][:],
                    ),
                    target_fs=target_fs,
                )
            )
    return tuple(streams)


def process_pair(
    pair: tuple, target_fs: float, dbpath: Path
) -> tuple[int, int, float, float, float, float]:
    (j, key1), (k, key2) = pair

    ds1, ds2 = load_signals(dbpath, [key1, key2], target_fs)
    sig1 = ds1.data.squeeze()
    sig2 = ds2.data.squeeze()

    x1 = (sig1 - np.mean(sig1)) / np.std(sig1)
    x2 = (sig2 - np.mean(sig2)) / np.std(sig2)
    norm_factor = np.sqrt(np.sum(x1**2) * np.sum(x2**2))

    # Compute time difference
    dt = (ds2.stats.time_init - ds1.stats.time_init) / np.timedelta64(1, "s")

    # Compute max cross-correlation & lag:
    ccf = signal.correlate(x1, x2, mode="full") / norm_factor
    lags = signal.correlation_lags(len(x1), len(x2), mode="full")
    max_idx = np.argmax(ccf)
    corr = ccf[max_idx]
    lag = lags[max_idx]

    return j, k, dt, corr, lag, ds1.stats.sampling_rate


def main(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    event_dirs = args.input.iterdir()

    for event_dir in event_dirs:
        event_name = event_dir.name.split("_")[0]
        logging.info(f"Processing event {event_name}.")

        dbfiles = event_dir.glob("*.hdf5")

        for dbfile in dbfiles:
            station_name = "-".join(dbfile.stem.split("_")[1:])
            logging.info(f"Processing station {station_name}.")

            detection_pairs, channels, num_detections = data_shape(dbfile)

            num_ch = len(channels)
            num_det = max(num_detections)
            time_diff = np.full((num_ch, num_det, num_det), np.nan)
            corr = np.full((num_ch, num_det, num_det), np.nan)
            lags = np.full((num_ch, num_det, num_det), np.nan)

            for i, (det_pairs, channel) in enumerate(zip(detection_pairs, channels)):
                logging.info(f"Processing channel {channel}.")

                with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
                    logging.info(f"Processing {len(det_pairs):,} pairs.")
                    results = list(
                        tqdm(
                            executor.map(
                                process_pair,
                                det_pairs,
                                [args.target_fs] * len(det_pairs),
                                [dbfile] * len(det_pairs),
                            ),
                            leave=True,
                            total=len(det_pairs),
                        )
                    )

                    for result in results:
                        j, k, dt, cf, lag, target_fs = result
                        time_diff[i, j, k] = dt
                        corr[i, j, k] = cf
                        lags[i, j, k] = lag

                nbytes = compute_array_size([time_diff, corr, lags])
                logging.info(f"Total size of arrays: {nbytes / (1024 ** 3):,} GB.")

            record = Record(
                event=event_name,
                station=station_name,
                sampling_rate=target_fs,
                time_diff=time_diff,
                corr=corr,
                lags=lags,
            )

            record.save_h5(args.output / f"maxcorr_{event_name}.h5")
            logging.info(
                f"Saved results to {args.output / f'maxcorr_{event_name}.h5'}."
            )


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
        default=500.0,
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        help="Number of workers to use for parallel processing.",
        default=4,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output HDF5 file to save the results.",
        default=Path(os.getenv("MAXCORRPATH")),
    )
    args = parser.parse_args()
    main(args)
