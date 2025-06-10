#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import logging
import os
from pathlib import Path

import dotenv
import h5py
import numpy as np
from tqdm import tqdm
from tritonoa.data.stream import DataStream

from rodeo.detect import parse_detection
from rodeo.utils import (
    compute_array_size,
    decimation_factor,
    initialize_julia,
    logging_kwargs,
)

dotenv.load_dotenv()
logging.basicConfig(**logging_kwargs)


@dataclass
class Record:
    event: str
    station: str
    sampling_rate: float
    time_diff: np.ndarray
    corr: np.ndarray
    etime: np.ndarray

    def save_h5(self, path: Path) -> None:
        """Save the record to an HDF5 file.

        The record is saved to the output file in the following format:
        ```
        /station
            /time_diff
            /corr
            /etime
        ```
        `time_diff` and `corr` are 3D arrays with the shape
        `(num_channels, num_detections, num_detections)`.

        Args:
            path: Path to the output HDF5 file.
        """
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
            grp.create_dataset("etime", data=self.etime)


def condition_signal(stream: DataStream, target_fs: float) -> tuple[DataStream, int]:
    """Condition the signal by decimating and filtering it.

    Args:
        stream: Data stream.
        target_fs: Target sampling rate for the signal.

    Returns:
        Decimated and filtered data stream.
        Decimation factor.
    """
    dec_factor = decimation_factor(stream.stats.sampling_rate, target_fs)
    ds = stream.decimate(dec_factor)
    ds.filter("highpass", freq=10.0)
    return ds, dec_factor


def data_shape(
    dbfile: Path,
) -> tuple[int, int]:
    """Return shape of data within the HDF file.

    Args:
        dbfile: Path to the HDF5 file.

    Returns:
        Number of channels and number of detections.
    """
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


def retrieve_data(dbpath: Path, target_fs: float) -> np.ndarray:
    """Retrieve data from the HDF5 file.

    Recursively loads data from the HDF5 file into a NumPy array.

    Args:
        dbpath: Path to the HDF5 file.
        target_fs: Target sampling rate for the signals.

    Returns:
        Array containing the signals.
        Array containing the initial times of the signals.
        Sampling rate of the signals.
    """
    num_channels, num_detections = data_shape(dbpath)

    data = None
    t0 = np.full((num_channels, num_detections), np.nan)

    with h5py.File(dbpath, "r") as file:
        for i, chan_group in enumerate(file.values()):
            logging.info(f"Loading channel {i}.")
            for j, det_group in tqdm(
                enumerate(chan_group.values()), total=num_detections
            ):
                ds, _ = condition_signal(
                    DataStream(
                        stats=parse_detection(det_group),
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


def main(args: argparse.Namespace) -> None:
    jl = initialize_julia("CrossCorr")

    args.output.mkdir(parents=True, exist_ok=True)
    event_dirs = args.input.iterdir()

    for event_dir in event_dirs:
        event_name = event_dir.name.split("_")[0]

        if args.event is not None and event_name != args.event:
            continue
        if args.omit is not None and event_name in args.omit:
            continue

        logging.info(f"Processing event {event_name}.")

        dbfiles = event_dir.glob("*.hdf5")

        for dbfile in dbfiles:
            station_name = "-".join(dbfile.stem.split("_")[1:])
            logging.info(f"Processing station {station_name}.")

            data, t0, target_fs = retrieve_data(dbfile, args.target_fs)
            num_channels = data.shape[0]
            num_detections = data.shape[1]
            time_diff = np.full((num_channels, num_detections, num_detections), np.nan)
            max_corr = np.full((num_channels, num_detections, num_detections), np.nan)
            etime = np.full((num_channels, num_detections, num_detections), np.nan)
            logging.info(
                f"Data shape: {num_channels} channels; {num_detections} detections."
            )
            logging.info(
                f"Size of arrays: {compute_array_size([max_corr, etime, time_diff]) / (1024 ** 3):.2f} GB."
            )

            threads = jl.seval("Threads.nthreads()")
            logging.info(f"Number of Julia threads: {threads}.")

            for i in range(num_channels):
                jl_data = jl.seval("x -> Matrix{Float64}(x)")(data[i])
                jl_time = jl.seval("x -> Vector{Float64}(x)")(t0[i])

                # logging.info(f"Computing dt for channel {i}.")
                # time_diff[i] = np.array(jl.CrossCorr.dt_matrix(jl_time))
                # logging.info(f"Computed dt for channel {i}.")

                logging.info(f"Computing corr & etime for channel {i}.")
                max_corr_mat, etime_mat = jl.CrossCorr.corr_matrix(jl_data, target_fs)
                max_corr[i] = np.array(max_corr_mat)
                etime[i] = np.array(etime_mat)
                logging.info(f"Computed corr & etime for channel {i}.")

            record = Record(
                event=event_name,
                station=station_name,
                sampling_rate=target_fs,
                time_diff=time_diff,
                corr=max_corr,
                etime=etime,
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
        "--event",
        type=list[str],
        nargs="+",
        help="Name of the event to process.",
        default=None,
    )
    parser.add_argument(
        "--omit",
        type=str,
        nargs="+",
        help="Name of the event(s) to omit.",
        default=None,
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
    os.environ["JULIA_NUM_THREADS"] = str(args.max_workers)
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    main(args)
