#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import dotenv
import polars as pl

from rodeo.utils import add_range_columns

dotenv.load_dotenv()


def main(args: argparse.Namespace) -> None:
    df = pl.read_csv(args.events)
    sensor_data = pl.read_csv(args.sensors)
    df = add_range_columns(df, sensor_data)
    df.write_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute ranges between piles and sensors."
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path(os.getenv("EVENTCONFIG")),
        help="Path to the piles file.",
    )
    parser.add_argument(
        "--sensors",
        type=Path,
        default=Path(os.getenv("EQUIPCONFIG")),
        help="Path to the sensors file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("EVENTCONFIG")),
        help="Path to save the output.",
    )
    args = parser.parse_args()
    main(args)
