#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
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
    for i, event in df.iterrows():
        start, end = start_end_time(event)

        for sensor_name in inv_config.keys():
            sensor_channels = inv_config[sensor_name]["channels"]
            ds = read_inventory(
                file_path=Path(inv_config[sensor_name]["data"]["destination"]),
                time_start=start,
                time_end=end,
                channels=[i["channel"] for i in sensor_channels],
            )
            print(ds)





        # ds = read_inventory(
        #     file_path=Path(inv_config[sensor_name]["data"]["destination"]),
        #     time_start=np.datetime64(event["start"], TIME_PRECISION),
        #     time_end=np.datetime64(event["end"], TIME_PRECISION),
        #     channels=[i["channel"] for i in sensor_channels],
        # )




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
