#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from tritonoa.data.reader import read_numpy

from rodeo import plotting

dotenv.load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))

STATION_ORDER = ["SHRU921_Tetra", "SHRU905_VLA1", "SHRU920_VLA2"]


def parse_stations(files: list[Path]) -> list[str]:
    return ["/".join(file.stem.split("_")[1:]) for file in files]


def main(args: argparse.Namespace) -> None:
    files = list(args.data.glob("*.npz"))
    files.sort(key=lambda x: STATION_ORDER.index("_".join(x.stem.split("_")[1:])))
    stations = parse_stations(files)
    streams = [read_numpy(file) for file in files]

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
        default=Path(os.getenv("DISPDATA")),
        help="Path to the data files.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(os.getenv("RESULTS")), help="Output file."
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
