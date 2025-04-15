#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tritonoa.modeling.at.kraken import run_kraken
import tritonoa.plotting as plotting
from tritonoa.signal.util import normalize_pressure

from rodeo import model as mdl, utils


def plot_transmission_loss(
    environment: dict,
    p: npt.NDArray[np.float_],
    rec_z: npt.NDArray[np.float_],
    rec_r: npt.NDArray[np.float_],
    title: str | None = None,
) -> plt.Figure:
    layerdata = environment["layerdata"]
    z_b = np.max(layerdata[0]["z"])

    fig, axs = plt.subplots(
        ncols=2,
        figsize=(12, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 7], "wspace": 0.0},
    )

    ax = axs[0]
    ax = plotting.plot_SSP(layerdata[0]["z"], layerdata[0]["c_p"], ax=ax)
    ax.axhline(z_b, color="k", linestyle="--")
    ax.set_xlabel("Speed of Sound (m/s)")
    ax.set_ylabel("Depth (m)")

    ax = axs[1]
    ax = plotting.plot_TL_2d(
        p,
        rec_z,
        rec_r,
        title=title,
        xlabel="Range (km)",
        clabel="Transmission Loss (dB)",
        cmap="inferno",
        ax=ax,
    )
    ax.axhline(z_b, color="w", linestyle="--")

    return fig


def main():
    ctd_df = utils.load_ssp(args.ssp)
    env = mdl.build_base_environment(ctd_df)
    rec_r = np.arange(1, 10e3, 1)
    rec_z = np.arange(0, 50, 0.1)

    for freq in args.freq:
        model = run_kraken(
            env | {"freq": freq, "rec_r": rec_r / 1e3, "rec_z": rec_z}
        )
        p = normalize_pressure(model.modes.p, log=True)
        fig = plot_transmission_loss(
            env, p, rec_z, rec_r, title=f"Source: {env["src_z"]} m, {freq} Hz"
        )
        fig.savefig(
            args.output / f"tl_{freq}.png", dpi=300, bbox_inches="tight", facecolor="w"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot transmission loss.")
    parser.add_argument(
        "--ssp",
        type=Path,
        default=Path(
            "data/ctd/sfw_leg2_2023-07-23/CTD_cast_south_fork_07_23_2023_data.txt"
        ),
    )
    parser.add_argument(
        "--freq",
        type=list[float],
        default=[50.0, 100.0, 200.0, 400.0],
        help="Frequency (Hz)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/model"))
    args = parser.parse_args()
    main()
