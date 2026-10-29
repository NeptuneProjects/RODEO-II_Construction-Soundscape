#!/usr/bin/env python


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tritonoa.modeling.at.kraken import run_kraken

from rodeo import model, plotting

plt.style.use(Path("config/jasa_fig_style.mplstyle"))

SAVEPATH = Path.cwd() / "results"


def main(args: argparse.Namespace) -> None:
    freqs = args.freqs
    zgrid = np.arange(0, 80, 1)
    environment = model.construct_environment() | {"rec_z": zgrid}

    phis = []
    for freq in freqs:
        res = run_kraken(environment | {"freq": freq, "rec_z": zgrid})
        phis.append(res.modes.phi_rec)

    fig = plotting.plot_mode_function(environment, phis, freqs, zgrid)
    fig.savefig(
        args.savepath / "mode_function.png", dpi=600, bbox_inches="tight", facecolor="w"
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mode functions")
    parser.add_argument(
        "--savepath",
        type=Path,
        default=Path("results"),
        help="Path to save the figure",
    )
    parser.add_argument(
        "--freqs",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 200.0],
        help="Frequencies to plot",
    )
    args = parser.parse_args()
    main(args)
