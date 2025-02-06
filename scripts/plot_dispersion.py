#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import tomllib

import dotenv
import numpy as np
import polars as pl
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION

dotenv.load_dotenv()

FS_DEC = 500


def load_inventory(inventory_file: Path) -> dict:
    with open(inventory_file, "rb") as f:
        return tomllib.load(f)


def load_metrics(metrics_file: Path, **kwargs) -> pl.DataFrame:
    df = pl.read_csv(metrics_file, try_parse_dates=True)
    filters = [pl.col(key) == value for key, value in kwargs.items()]
    return df.filter(filters)


def parse_data(df: pl.DataFrame) -> tuple[list, list, list]:
    # Sort columns of interest by range
    stations = sorted(
        df["station"].unique().to_list(),
        key=lambda x: df.filter(pl.col("station") == x)["range"].min(),
    )
    time_inits = sorted(
        df["time_init"].unique().to_list(),
        key=lambda x: df.filter(pl.col("time_init") == x)["range"].min(),
    )
    time_ends = sorted(
        df["time_end"].unique().to_list(),
        key=lambda x: df.filter(pl.col("time_end") == x)["range"].min(),
    )
    # Convert to numpy datetime64
    time_inits = [np.datetime64(x, TIME_PRECISION) for x in time_inits]
    time_ends = [np.datetime64(x, TIME_PRECISION) for x in time_ends]

    return stations, time_inits, time_ends


def main(args: argparse.Namespace):
    df = load_metrics(
        args.metrics, event_name=args.event, channel=args.channel, index=args.strike
    )
    stations, time_inits, time_ends = parse_data(df)
    inv_config = load_inventory(args.inventory)

    streams = []
    for station, time_init, time_end in zip(stations, time_inits, time_ends):
        ds = read_inventory(
            file_path=Path(inv_config[station]["data"]["destination"]),
            time_start=time_init,
            time_end=time_end,
            channels=args.channel,
        )
        ds.decimate(int(ds.stats.sampling_rate // FS_DEC))
        ds.filter(filt_type="highpass", freq=10.0)
        streams.append(ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dispersion.")
    parser.add_argument("--event", type=str, default="A10-B", help="Event name.")
    parser.add_argument("--channel", type=int, default=0, help="Channel index.")
    parser.add_argument("--strike", type=int, default=1974, help="Strike index.")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(os.getenv("METRICSFILE")),
        help="Metrics file.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(os.getenv("INVCONFIG")),
        help="Inventory file.",
    )
    args = parser.parse_args()
    main(args)
