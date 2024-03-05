#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import tomllib
import warnings

from datautils.data import read
from datautils.query import CatalogueQuery
import matplotlib.pyplot as plt
import numpy as np

import helpers
import plotting


def main(args: argparse.Namespace):
    if args.time_interval % 1 != 0:
        warnings.warn(
            "Time interval is not an integer. Rounding to the nearest integer."
        )
        args.time_interval = round(args.time_interval)
    time_interval = np.timedelta64(args.time_interval, args.time_units)
    
    with open(args.events, "rb") as f:
        data = tomllib.load(f)
    events = data["events"]

    title = (
        f"{args.title}\n"
        f"{events[0]["start"]}"
        f" - {events[-1]["end"]}"
    )
    num_bins = 100
    channels = [int(c) for c in args.channels.split(",")]
    power_range = (args.ymin, args.ymax)
    mean_hist = np.zeros((len(events), len(channels), num_bins))

    for k, event in enumerate(events):
        time_start = np.datetime64(event["start"], "us")
        time_end = np.datetime64(event["end"], "us")
        if time_end - time_start < time_interval:
            time_interval = time_end - time_start

        num_intervals = int((time_end - time_start) / time_interval)
        
        hist_data = np.zeros((len(channels), num_intervals, num_bins))
        
        for j in range(num_intervals):
            time_start_i = time_start + j * time_interval
            time_end_i = time_start + (j + 1) * time_interval

            cat_query = CatalogueQuery(
                catalogue=args.catalogue,
                destination=args.destination,
                time_start=time_start_i,
                time_end=time_end_i,
                channels=channels,
            )
            ds = read(cat_query)

            for i in range(ds.num_channels):
                power = ds.waveform[:, i] ** 2
                if args.cutoff is not None:
                    win_size = int(1 / args.cutoff * ds.stats.sampling_rate)
                    power = helpers.moving_average(power, win_size)
                
                power = 10 * helpers.safe_log10(power)
                hist, bins = np.histogram(power, bins=num_bins, range=power_range, density=True)
                hist_data[i, j, :] = hist

        mean_hist[k] = np.mean(hist_data, axis=1)

    fig = plotting.plot_avg_power_timehist(
        mean_hist, bins, events, title, power_range=power_range, channels=channels
    )
    filename = (
        events[0]["start"]
        .replace("-", "")
        .replace("T", "_")
        .replace(":", "")
        + "_"
        + args.title.replace("/", "_")
    )
    args.destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        Path(args.destination / filename), bbox_inches="tight", facecolor="w", dpi=300
    )
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectrogram plotting routine")
    parser.add_argument(
        "--catalogue",
        help="Path to the catalogue file",
        type=Path,
        default="data/catalogues/SHRU905_VLA1_FileInfo.csv",
    )
    parser.add_argument(
        "--destination",
        help="Path to the destination directory",
        type=Path,
        default="reports/power_event_hist/inst",
    )
    parser.add_argument("--time_interval", help="Time interval", type=int, default=1)
    parser.add_argument("--time_units", help="Time units", type=str, default="m")
    # parser.add_argument(
    #     "--time_start", help="Start time", type=str, default="2023-07-09T09:45:00"
    # )
    # parser.add_argument(
    #     "--time_end", help="End time", type=str, default="2023-07-09T10:02:10"
    # )
    parser.add_argument("--events", help="Log with pile driving events", type=Path, default="data/logs/events.toml")
    parser.add_argument(
        "--channels", help="Channels to be read", type=str, default="0,1,2,3"
    )
    parser.add_argument(
        "--title", help="Title of the plot", type=str, default="SHRU905/VLA1"
    )
    parser.add_argument("--ymin", help="Minimum value for the color scale", type=float, default=120)
    parser.add_argument("--ymax", help="Maximum value for the color scale", type=float, default=160)
    parser.add_argument(
        "--cutoff", help="Cutoff frequency for moving average", type=float, default=None
    )

    args = parser.parse_args()
    main(args)
