#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path
import tomllib

import dotenv
import numpy as np
import polars as pl
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)

FS_DEC = 500


def load_inventory(inventory_file: Path) -> dict:
    with open(inventory_file, "rb") as f:
        return tomllib.load(f)


def load_metrics(metrics_file: Path, **kwargs) -> pl.DataFrame:
    df = pl.read_csv(metrics_file, try_parse_dates=True)
    filters = [pl.col(key) == value for key, value in kwargs.items()]
    return df.filter(filters)


def parse_data(df: pl.DataFrame, buffer: int, units: str) -> tuple[list, list, list]:
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
    time_inits = [
        np.datetime64(x.replace(tzinfo=None), TIME_PRECISION)
        - np.timedelta64(args.tbuffer, args.tbuffer_units)
        for x in time_inits
    ]
    time_ends = [
        np.datetime64(x.replace(tzinfo=None), TIME_PRECISION)
        + np.timedelta64(args.tbuffer, args.tbuffer_units)
        for x in time_ends
    ]

    return stations, time_inits, time_ends


def main(args: argparse.Namespace):
    logging.info(f"Extracting data from event {args.event} for strike {args.strike}.")
    df = load_metrics(
        args.metrics, event_name=args.event, channel=0, index=args.strike
    )
    stations, time_inits, time_ends = parse_data(df, args.tbuffer, args.tbuffer_units)
    inv_config = load_inventory(args.inventory)

    for station, time_init, time_end in zip(stations, time_inits, time_ends):
        logging.info(f"Extracting data for station {station}.")
        ds = read_inventory(
            file_path=Path(inv_config[station]["data"]["destination"]),
            time_start=time_init,
            time_end=time_end,
            channels=[0, 1, 2, 3],
        )
        ds.decimate(int(ds.stats.sampling_rate // FS_DEC))
        ds.filter(filt_type="highpass", freq=10.0)
    
        fname = args.output / f"disp_{station}.npz"
        ds.write(fname)
        logging.info(f"Extracted data saved to {fname}")
    
    logging.info("Data extraction complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data for dispersion example.")
    parser.add_argument("--event", type=str, default="A10-B", help="Event name.")
    parser.add_argument("--strike", type=int, default=1974, help="Strike index.")
    parser.add_argument(
        "--tbuffer",
        type=int,
        default=400,
        help="Time buffer (unitless; specify with `tbuffer_units`).",
    )
    parser.add_argument(
        "--tbuffer_units", type=str, default="ms", help="Time buffer units."
    )
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("RESULTS")),
        help="Output directory.",
    )
    args = parser.parse_args()
    main(args)
