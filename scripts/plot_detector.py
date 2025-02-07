#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import tomllib

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tritonoa.data.reader import read_inventory
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION

from rodeo import plotting

load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))


def load_data(
    inv_config: dict,
    event_data: dict,
    sensor_name: str,
    dec_factor: int,
    filter_type: str,
    filter_freq: float,
) -> DataStream:
    ds = read_inventory(
        file_path=Path(inv_config[sensor_name]["data"]["destination"]),
        time_start=np.datetime64(event_data["Start"], TIME_PRECISION),
        time_end=np.datetime64(event_data["Stop"], TIME_PRECISION),
    )
    ds.decimate(dec_factor)
    ds.filter(filt_type=filter_type, freq=filter_freq)
    return ds


def plot_detector(
    ds: DataStream,
    sensor_name: str,
    channel: int,
    stwin: float,
    ltwin: float,
    threshold: float,
    distance: float,
    taper: float,
    zoom_start: int,
    zoom_end: int,
    figsize: tuple,
    title: str = None,
):
    fig = plotting.plot_detector_output(
        ds,
        stwin=stwin,
        ltwin=ltwin,
        threshold=threshold,
        distance=distance,
        taper=taper,
        sensor=sensor_name,
        channel=channel,
        figsize=figsize,
        zoom_start=zoom_start,
        zoom_end=zoom_end,
        title=title,
    )
    return fig


def main(args: argparse.Namespace) -> None:
    with open(args.inv, "rb") as f:
        data = tomllib.load(f)

    event_data = (
        pd.read_csv(Path(os.getenv("EVENTCONFIG"))).set_index("Event").loc[args.event]
    )

    ds = load_data(
        data,
        event_data,
        args.sensor,
        args.dec,
        args.filter,
        args.filt_freq,
    )

    fig = plot_detector(
        ds,
        args.sensor,
        args.channel,
        args.stwin,
        args.ltwin,
        args.threshold,
        args.distance,
        args.taper,
        args.zoom_start,
        args.zoom_end,
        args.figsize,
        title=(
            f"Event {args.event} | "
            f"{args.sensor.split("_")[1]} Channel {args.channel + 1} | "
            f"{args.filter.capitalize()} {args.filt_freq} Hz | "
            f"$M_1$: {args.stwin} s | "
            f"$M_2$: {args.ltwin} s | "
            f"Thresh.: {args.threshold}"
        ),
    )
    fig.savefig(
        args.savedir / f"detector_example_{args.sensor}_chan{args.channel}.png",
        dpi=600,
        bbox_inches="tight",
        facecolor="w",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inv",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    parser.add_argument("--event", type=str, default="A02")
    parser.add_argument("--sensor", type=str, default="SHRU905_VLA1")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--dec", type=int, default=39)
    parser.add_argument("--filter", type=str, default="highpass")
    parser.add_argument("--filt_freq", type=float, default=50.0)
    parser.add_argument("--taper", type=float, default=5.0)
    parser.add_argument("--stwin", type=float, default=0.025)
    parser.add_argument("--ltwin", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--distance", type=float, default=0.95)
    parser.add_argument("--savedir", type=Path, default=Path.cwd() / "results")
    parser.add_argument("--zoom_start", type=int, default=770000)
    parser.add_argument("--zoom_end", type=int, default=820000)
    parser.add_argument("--figsize", type=tuple, default=(8, 3))
    args = parser.parse_args()
    main(args)
