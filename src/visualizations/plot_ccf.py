#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np


def read_single_detection(file_path, station_1, channel_1, station_2, channel_2):
    """
    Reads a single detection HDF5 file and accesses the cross-correlation data between two specified stations and channels.

    Parameters:
        file_path (str): Path to the HDF5 file containing the detection data.
        station_1 (str): The first station name to compare.
        channel_1 (int): The first channel number to compare.
        station_2 (str): The second station name to compare.
        channel_2 (int): The second channel number to compare.

    Returns:
        tuple: A tuple containing the key and the corresponding data if found, otherwise None.
    """
    # Open the HDF5 file
    with h5py.File(file_path, "r") as hf:
        # Iterate over datasets in the file to find the matching stations and channels
        for key in hf.keys():
            if f"{station_1}_ch{channel_1}_vs_{station_2}_ch{channel_2}" in key:
                data = hf[key][:]
                normalized_data = np.array([row / np.max(np.abs(row)) for row in data])
                return key, normalized_data

        print(
            f"No cross-correlation data found between station {station_1} channel {channel_1} and station {station_2} channel {channel_2}."
        )
        return None, None


def plot_cross_correlation(key, data, station_1, channel_1, station_2, channel_2):
    """
    Plots the cross-correlation function between two specified stations and channels.

    Parameters:
        key (str): The key representing the data.
        data (ndarray): The cross-correlation data to be plotted.
        station_1 (str): The first station name.
        channel_1 (int): The first channel number.
        station_2 (str): The second station name.
        channel_2 (int): The second channel number.
    """
    if data is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(data.mean(axis=0), label=f"Sum of all segments")
        # for i, ccf_segment in enumerate(data):
        #     plt.plot(ccf_segment, label=f"Segment {i + 1}")
        plt.xlabel("Lag")
        plt.ylabel("Cross-Correlation Amplitude")
        plt.title(
            f"Cross-Correlation between {station_1} Channel {channel_1} and {station_2} Channel {channel_2}"
        )
        plt.legend()
        plt.grid(True)
        plt.show()


def main(args):
    # Read the single detection HDF5 file
    key, data = read_single_detection(
        args.file_path, args.station_1, args.channel_1, args.station_2, args.channel_2
    )

    # Plot the cross-correlation data if found
    if data is not None:
        plot_cross_correlation(
            key, data, args.station_1, args.channel_1, args.station_2, args.channel_2
        )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Read cross-correlation data from a single detection HDF5 file."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the HDF5 file containing the detection data.",
    )
    parser.add_argument(
        "station_1", type=str, help="The first station name to compare."
    )
    parser.add_argument(
        "channel_1", type=int, help="The first channel number to compare."
    )
    parser.add_argument(
        "station_2", type=str, help="The second station name to compare."
    )
    parser.add_argument(
        "channel_2", type=int, help="The second channel number to compare."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call main with parsed arguments
    main(args)
