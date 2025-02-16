#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import importlib.resources as pkg_resources
import logging
import os
from pathlib import Path

import dotenv
import h5py
import numpy as np
from tqdm import tqdm
from tritonoa.data.stream import DataStream, DataStreamStats
from tritonoa.data.time import TIME_PRECISION

import rodeo

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


@dataclass
class Record:
    event: str
    station: str
    sampling_rate: float
    time_diff: np.ndarray
    corr: np.ndarray

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


def condition_signal(stream: DataStream, target_fs: float) -> DataStream:
    dec_factor = decimation_factor(stream.stats.sampling_rate, target_fs)
    ds = stream.decimate(dec_factor)
    ds.filter("highpass", freq=10.0)
    return ds


def data_shape(
    dbfile: Path,
) -> tuple[int, int]:
    channels = []
    num_detections = []
    num_samples = None
    with h5py.File(dbfile, "r") as file:
        logging.info(f"Loaded {dbfile}")
        for chan_group in file.values():
            channels.append(int(chan_group.name[-1]))
            num_detections.append(len(chan_group))
            if num_samples is None:
                num_samples = next(iter(chan_group.values()))["waveform"].shape[1]

    return len(channels), max(num_detections)


def decimation_factor(fs: float, target_fs: float) -> int:
    return int(fs // target_fs)


def parse_metadata(grp: h5py.Group) -> DataStreamStats:
    attrs = grp.attrs
    return DataStreamStats(
        channels=attrs.get("channel"),
        time_init=np.datetime64(attrs.get("time_init")[:-1], TIME_PRECISION),
        time_end=np.datetime64(attrs.get("time_end")[:-1], TIME_PRECISION),
        sampling_rate=attrs.get("sampling_rate"),
        units=attrs.get("unit"),
    )


def retrieve_data(dbpath: Path, target_fs: float) -> np.ndarray:
    num_channels, num_detections = data_shape(dbpath)

    data = None
    t0 = np.full((num_channels, num_detections), np.nan)

    with h5py.File(dbpath, "r") as file:
        for i, chan_group in enumerate(file.values()):
            logging.info(f"Loading channel {i}.")
            for j, det_group in tqdm(
                enumerate(chan_group.values()), total=num_detections
            ):
                ds = condition_signal(
                    DataStream(
                        stats=parse_metadata(det_group),
                        data=det_group["waveform"][:],
                    ),
                    target_fs=target_fs,
                )
                if data is None:
                    # Initialize within loop since num_samples depends on target_fs.
                    data = np.full(
                        (num_channels, num_detections, ds.num_samples), np.nan
                    )

                data[i, j] = ds.data.squeeze()
                t0[i, j] = ds.stats.time_init

    return data, t0, ds.stats.sampling_rate


def initialize_julia() -> object:
    from juliacall import Main as jl

    julia_path = pkg_resources.files(rodeo).joinpath("CrossCorr.jl")
    jl.include(str(julia_path))
    return jl


def main(args: argparse.Namespace) -> None:
    # Initialize Julia
    jl = initialize_julia()

    args.output.mkdir(parents=True, exist_ok=True)
    event_dirs = args.input.iterdir()

    for event_dir in event_dirs:
        event_name = event_dir.name.split("_")[0]
        logging.info(f"Processing event {event_name}.")

        dbfiles = event_dir.glob("*.hdf5")

        for dbfile in dbfiles:
            station_name = "-".join(dbfile.stem.split("_")[1:])

            data, t0, target_fs = retrieve_data(dbfile, args.target_fs)

            num_channels = data.shape[0]
            num_detections = data.shape[1]
            max_corr = np.full((num_channels, num_detections, num_detections), np.nan)
            time_diff = np.full((num_channels, num_detections, num_detections), np.nan)

            threads = jl.seval("Threads.nthreads()")
            logging.info(f"Number of Julia threads: {threads}.")

            for i in range(num_channels):
                jl_data = jl.seval("x -> Matrix{Float64}(x)")(data[i])
                jl_time = jl.seval("x -> Vector{Float64}(x)")(t0[i])
                
                logging.info(f"Computing corr for channel {i}.")
                max_corr[i] = np.array(jl.CrossCorr.corr_matrix(jl_data))
                logging.info(f"Computed corr for channel {i}.")
                
                logging.info(f"Computing dt for channel {i}.")
                time_diff[i] = np.array(jl.CrossCorr.dt_matrix(jl_time))
                logging.info(f"Computed dt for channel {i}.")

            record = Record(
                event=event_name,
                station=station_name,
                sampling_rate=target_fs,
                time_diff=time_diff,
                corr=max_corr,
            )

            record.save_h5(args.output / f"maxcorr_{event_name}.h5")
            logging.info(
                f"Saved results to {args.output / f'maxcorr_{event_name}.h5'}."
            )


if __name__ == "__main__":
    os.environ["JULIA_NUM_THREADS"] = "12"
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
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
