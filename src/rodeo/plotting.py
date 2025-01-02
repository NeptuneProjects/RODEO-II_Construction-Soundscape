# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd


def plot_ctd_profiles(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 3),
    gridspec_kw: dict = {"wspace": 0.1},
    dpi: int = 300,
    **plot_kwargs,
) -> plt.Figure:
    fig, axs = plt.subplots(figsize=figsize, ncols=3, gridspec_kw=gridspec_kw, dpi=dpi)
    ylim = (0, 35)

    ax0 = axs[0]
    ax0.plot(df["Temperature"], df["Depth"], **plot_kwargs["Temperature"])
    ax0.set_ylim(ylim)
    ax0.invert_yaxis()
    ax0.grid()
    ax0.set_xlabel("Temp. (C)")
    ax0.set_ylabel("Depth (m)")

    ax1 = axs[1]
    ax1.plot(df["Salinity"], df["Depth"], **plot_kwargs["Salinity"])
    ax1.set_ylim(ylim)
    ax1.invert_yaxis()
    ax1.grid()
    ax1.set_yticklabels([])
    ax1.set_xlabel("Sal. (PSU)")

    ax2 = axs[2]
    ax2.plot(df["Speed of sound"], df["Depth"], **plot_kwargs["Speed of sound"])
    ax2.set_ylim(ylim)
    ax2.invert_yaxis()
    ax2.grid()
    ax2.set_yticklabels([])
    ax2.set_xlabel("Sound speed (m/s)")
    return fig
