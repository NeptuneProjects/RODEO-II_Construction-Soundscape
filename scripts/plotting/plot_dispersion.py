#!/usr/bin/env python3


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rodeo import plotting
from scipy.signal.windows import hann
from tritonoa.data.reader import read_numpy

plt.style.use(Path("config/jasa_fig_style.mplstyle"))


def parse_stations(files: list[Path]) -> list[str]:
    return ["/".join(file.stem.split("_")[1:]) for file in files]


def main(args: argparse.Namespace) -> None:
    files = list(args.data.glob("*.npz"))
    streams = [read_numpy(file) for file in files]
    # Nearest sensor first: the strike arrives there first.
    order = np.argsort([ds.stats.time_init for ds in streams])
    files = [files[k] for k in order]
    streams = [streams[k] for k in order]
    stations = parse_stations(files)

    fig = plotting.plot_pulse_dispersion(
        streams,
        stations,
        hann,
        channel=args.channel,
        tperseg=args.tperseg,
        pcoverlap=args.overlap,
        nfft_factor=args.nfft,
        vmin=args.vmin,
        vmax=args.vmax,
        filt=args.filter,
        filt_freq=args.filt_freq,
    )
    fig.savefig(
        args.output / "dispersion.png", bbox_inches="tight", facecolor="w", dpi=600
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot signal dispersion.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/dispersion"),
        help="Path to the data files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"), help="Output file."
    )
    parser.add_argument("--channel", type=int, default=0, help="Channel to plot.")
    parser.add_argument("--tperseg", type=float, default=0.05, help="Time per segment.")
    parser.add_argument(
        "--overlap", type=float, default=0.90, help="Fractional overlap."
    )
    parser.add_argument("--nfft", type=float, default=2, help="NFFT factor.")
    parser.add_argument("--vmin", type=float, default=80.0, help="Minimum value.")
    parser.add_argument("--vmax", type=float, default=130.0, help="Maximum value.")
    parser.add_argument("--filter", type=str, default="highpass", help="Filter type.")
    parser.add_argument(
        "--filt_freq", type=float, default=10.0, help="Filter frequency."
    )
    args = parser.parse_args()
    main(args)
