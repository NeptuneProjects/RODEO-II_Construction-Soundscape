#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from itertools import permutations
import os
from pathlib import Path

from dotenv import load_dotenv
import h5py
import numpy as np
from scipy.signal import correlate
from tqdm import tqdm

load_dotenv()


def process_detection(
    detection_index: int, data_files: list[Path], window_size: int
) -> tuple[str, dict]:
    """Process a single detection index to compute cross-correlation.

    Parameters:
        detection_index (int): The index of the detection to process.
        data_files (dict): Dictionary where keys are station names and values are
            paths to the HDF5 files containing the detection data for each station.
        window_size (int): The size of the sliding window for cross-correlation
            computation.

    Returns:
        A tuple containing the detection key and the corresponding results.
    """
    hfs = {
        station: h5py.File(data_files[station], "r") for station in data_files.keys()
    }
    stations = list(data_files.keys())
    detection_results = {}

    # Generate permutations of stations and channels
    station_permutations = list(permutations(stations, 2)) + [(station, station) for station in stations]

    for station_pair in station_permutations:
        station_1, station_2 = station_pair

        # Access channel data for each station
        for channel_1 in range(4):
            for channel_2 in range(4):
                data_1 = hfs[station_1][f"chan={channel_1}"][
                    f"det={detection_index:04d}"
                ]["waveform"][:]
                data_2 = hfs[station_2][f"chan={channel_2}"][
                    f"det={detection_index:04d}"
                ]["waveform"][:]

                # Compute cross-correlation with sliding window
                ccf = []
                for start in range(0, len(data_1) - window_size + 1, window_size):
                    segment_1 = data_1[start : start + window_size]
                    segment_2 = data_2[start : start + window_size]

                    # Compute cross-correlation for the current window
                    ccf_segment = correlate(segment_1, segment_2, mode="full")
                    ccf.append(ccf_segment)

                # Store the CCF results in a nested dictionary
                key = f"{station_1}_ch{channel_1}_vs_{station_2}_ch{channel_2}"
                detection_results[key] = np.array(ccf)

    # Close all HDF5 files
    for hf in hfs.values():
        hf.close()

    return f"detection_{detection_index:04d}", detection_results


def compute_cross_correlation(
    data_files: list[Path],
    # output_file: Path,
    savedir: str | Path = ".tmp",
    window_size: int = 256,
) -> None:
    """Compute the cross-correlation function (CCF) for each permutation of
    stations and channels. Saves results intermittently to avoid memory issues.

    Parameters:
        data_files: Dictionary where keys are station names and values are
            paths to the HDF5 files containing the detection data for each
            station.
        output_file: Path to the HDF5 file where results will be stored.
        window_size: The size of the sliding window for cross-correlation
            computation (default: 256).
        temp_dir: Directory to store intermediate results (default: 'temp_results').
    """
    # Create a temporary directory for intermediate results
    temp_dir_path = Path(savedir)
    temp_dir_path.mkdir(parents=True, exist_ok=True)

    stations = list(data_files.keys())
    num_detections = len(h5py.File(data_files[stations[0]], "r")["chan=0"])

    # Process all detections in parallel and save results intermittently
    for detection_index in tqdm(
        range(num_detections), total=num_detections, desc="Processing detections"
    ):
        detection, detection_results = process_detection(
            detection_index, data_files, window_size
        )
        temp_file_path = temp_dir_path / f"{detection}.h5"
        with h5py.File(temp_file_path, "w") as temp_hf:
            for key, ccf_data in detection_results.items():
                temp_hf.create_dataset(key, data=ccf_data)
            
        return


def main(args: argparse.Namespace) -> None:
    base_path = Path(os.getenv("RESPATH")) / args.event
    FILE_NAMES = [
        "detections_SHRU905_VLA1.hdf5",
        "detections_SHRU920_VLA2.hdf5",
        "detections_SHRU921_Tetra.hdf5",
    ]
    results_paths = [base_path / fname for fname in FILE_NAMES]

    if not all([path.exists() for path in results_paths]):
        raise FileNotFoundError("One or more detection files do not exist.")

    data_files = {
        station: path
        for station, path in zip(
            ["SHRU921_Tetra", "SHRU905_VLA1", "SHRU920_VLA2"], results_paths
        )
    }
    savdir = Path(os.getenv("CCFPATH")) / args.event
    savdir.mkdir(parents=True, exist_ok=True)

    compute_cross_correlation(
        data_files,
        # savdir / "ccf.h5",
        savedir=savdir,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cross-correlation function")
    parser.add_argument(
        "--event",
        type=str,
        help="Name of the event",
        default="A03_2023-07-12",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Size of the sliding window for cross-correlation computation",
        default=128,
    )
    args = parser.parse_args()
    main(args)
