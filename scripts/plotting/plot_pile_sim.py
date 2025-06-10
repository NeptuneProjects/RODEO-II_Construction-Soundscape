# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift


def plot_data(time, depth, pressure, freq, pf, pif, tl) -> list[plt.Figure]:
    """Plot the data and results."""

    figs = []

    # Plot pressure as a function of time and depth
    fig = plt.figure()
    plt.pcolor(time, depth, pressure.T, cmap="RdBu")
    plt.xlabel("Time [s]")
    plt.ylabel("Depth [m]")
    plt.title("Pressure Field [Pa]")
    plt.clim(-(10**6), 10**6)
    plt.colorbar()
    figs.append(fig)

    # Plot pressure over time
    fig = plt.figure()
    plt.plot(time, pressure[:, :])
    plt.ylim(-(10**6), 10**6)
    plt.xlabel("Time [s]")
    plt.ylabel("Pressure [Pa]")
    plt.title("Pressure Field Over Time")
    figs.append(fig)

    # Plot FFT magnitude
    fig = plt.figure()
    plt.pcolor(freq, depth, np.abs(pf.T), cmap="turbo")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Depth [m]")
    plt.title("FFT Magnitude of Pressure Field")
    plt.clim(0, 2.5 * 10**6)
    plt.colorbar()
    figs.append(fig)

    # Plot FFT results over depth
    fig = plt.figure()
    plt.plot(pf.T[:, :], depth)
    plt.xlabel("FFT Magnitude")
    plt.ylabel("Depth [m]")
    plt.title("FFT Results Over Depth")
    figs.append(fig)

    # Plot transmission loss
    fig = plt.figure()
    plt.pcolor(time, depth, tl, cmap="turbo")
    plt.xlabel("Time [s]")
    plt.ylabel("Depth [m]")
    plt.title("Transmission Loss [dB]")
    plt.clim(-80, 0)
    plt.colorbar()
    figs.append(fig)

    # Plot transmission loss over time
    fig = plt.figure()
    plt.plot(time, tl.T[:, :])
    plt.xlabel("Time [s]")
    plt.ylabel("Transmission Loss [dB]")
    plt.title("Transmission Loss Over Time")
    figs.append(fig)

    return figs


def main(args: argparse.Namespace) -> None:
    # Load dataset
    dataset = nc.Dataset(args.nc_file, "r")

    # Extract variables
    depth = dataset.variables["depth"][:]
    time = dataset.variables["time"][:]
    pressure = dataset.variables["pressure"][:, :]

    # Replace NaN values with 0 for FFT computation
    pressure[np.isnan(pressure)] = 0

    # Compute FFT of the pressure field
    pf = fft(pressure, axis=0)
    pif = ifft(pf, axis=0)

    timestep = 4 * 10**-7 * 10000
    n = len(time)
    freq = fftshift(fftfreq(n, d=timestep))

    # Compute transmission loss
    pr = np.nanmax(np.abs(pif.T))
    tl = 20 * np.log10(np.abs(pif.T) / pr)

    # Plot the data
    figs = plot_data(time, depth, pressure, freq, pf, pif, tl)
    [
        fig.savefig(
            args.nc_file.parent
            / f"{re.sub(r"[ \[\]]", "", fig.axes[0].get_title())}.png",
            dpi=300,
            bbox_inches="tight",
        )
        for fig in figs
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot pile simulation data.")
    parser.add_argument(
        "--nc_file",
        type=Path,
        default="data/simulation/impact_ex1.nc",
        help="Directory containing the simulation data.",
    )
    args = parser.parse_args()
    main(args)
