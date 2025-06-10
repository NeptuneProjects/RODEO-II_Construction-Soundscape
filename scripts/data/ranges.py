#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import dotenv
import polars as pl

from rodeo.utils import compute_range

dotenv.load_dotenv()


def add_range_columns(df: pl.DataFrame, sensor_data: pl.DataFrame) -> pl.DataFrame:
    STATIONS = ["SHRU921_Tetra", "SHRU905_VLA1", "SHRU920_VLA2"]

    for station in STATIONS:
        column_name = f"R to {station} (m)"
        if column_name in df.columns:
            continue

        lon1 = sensor_data.filter(sensor_data["Name"] == station)[
            "Longitude [dd]"
        ].to_numpy()[0]
        lat1 = sensor_data.filter(sensor_data["Name"] == station)[
            "Latitude [dd]"
        ].to_numpy()[0]

        df = df.with_columns(
            pl.struct(["Lon", "Lat"])
            .map_elements(
                lambda row: compute_range(row["Lon"], row["Lat"], lon1, lat1),
                return_dtype=pl.Float64,
            )
            .alias(column_name)
        )

    return df


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
