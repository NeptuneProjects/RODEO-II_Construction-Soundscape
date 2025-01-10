#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from itertools import product
import logging
import os
from pathlib import Path
import tomllib

from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tritonoa.data.reader import read_inventory
from tritonoa.data.time import TIME_PRECISION

load_dotenv()

TIME_BEFORE_MINUTES = 11
DURATION_MINUTES = 10


def start_end_time(event: pd.Series) -> tuple[np.datetime64, np.datetime64]:
    start = np.datetime64(event["Start"], TIME_PRECISION) - np.timedelta64(
        TIME_BEFORE_MINUTES, "m"
    )
    end = start + np.timedelta64(DURATION_MINUTES, "m")
    return start, end


def loop_over(df: pd.DataFrame, inv_config: dict) -> None:
    skip_events = {"Z01-A", "Z01-B", "Z01-C", "Z01-D", "A04"}
    skip_sensors = {
        "A14": {"SHRU921_Tetra"},
        "A11": {"SHRU921_Tetra"},
        "A02": {"SHRU921_Tetra"},
        "A15": {"SHRU921_Tetra"},
        "A13": {"SHRU921_Tetra"},
        "A01": {"SHRU921_Tetra"},
        "A07": {"SHRU921_Tetra", "SHRU905_VLA1"},
    }

    # Create a single loop using itertools.product
    for (i, event), (sensor_name, sensor_config) in product(df.iterrows(), inv_config.items()):
        if event["Event"] in skip_events:
            continue

        if sensor_name in skip_sensors.get(event["Event"], set()):
            continue

        start, end = start_end_time(event)

        ds = read_inventory(
            file_path=Path(sensor_config["data"]["destination"]),
            time_start=start,
            time_end=end,
        )
        print(ds)
        return


def main(args: argparse.Namespace) -> None:
    df_events = pd.read_csv(os.getenv("EVENTCONFIG"))
    with open(args.inv, "rb") as f:
        inv_config = tomllib.load(f)
    loop_over(df_events, inv_config)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inv",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    parser.add_argument("--event", type=Path, default=Path("config/events/A08.toml"))
    parser.add_argument("--filter", type=str, default="highpass")
    parser.add_argument("--filt_freq", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
