#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script compiles all detector configuration files into a single table
formatted for markdown.
"""

import argparse
import os
from pathlib import Path
import subprocess
import tomllib

import dotenv
import pandas as pd

dotenv.load_dotenv()


def collate_data_from_toml(directory: Path) -> pd.DataFrame:
    dfs = []
    for file in Path(directory).glob("*.toml"):
        with open(file, "rb") as f:
            dfs.append(parse_data(tomllib.load(f)))

    return pd.concat(dfs, ignore_index=True)


def parse_data(data: dict) -> pd.DataFrame:
    # Initialize an empty list for the flat table
    flat_table = []

    # Extract event data
    for event in data.get("event", []):
        event_name = event.get("name")
        event_start = event.get("start")[:-3]
        event_end = event.get("end")[:-3]

        # Extract sensor data
        for sensor in event.get("sensors", []):
            sensor_name = sensor.get("name")
            sensor_filter = sensor.get("filter")
            sensor_freq = sensor.get("freq")

            # Extract channel data
            for channel in sensor.get("channels", []):
                channel_row = {
                    "Event": event_name,
                    "Start": event_start,
                    "End": event_end,
                    "Array/Chan.": f"{sensor_name.split('_')[1]}/{channel.get('channel')}",
                    "Filter": sensor_filter.capitalize(),
                    "Cutoff Freq. (Hz)": sensor_freq,
                    "$M_1$ (s)": channel.get("stwin"),
                    "$M_2$ (s)": channel.get("ltwin"),
                    "Threshold": channel.get("threshold"),
                }
                flat_table.append(channel_row)
    return pd.DataFrame(flat_table)


def format_markdown(df: pd.DataFrame) -> str:

    PREPEND = "---\n" "font-size: 10pt\n" "geometry: margin=1in\n" "---\n" "\\tiny\n"

    table = df.to_markdown(tablefmt="github", index=False)
    return f"{PREPEND}\n{table}"


def main(args: argparse.Namespace) -> None:
    directory = Path(os.getenv("DETECTORPATH"))
    df = collate_data_from_toml(directory)
    contents = format_markdown(df)

    with open(args.savepath, "w") as f:
        f.write(contents)

    subprocess.run(["pandoc", args.savepath, "-o", args.savepath.with_suffix(".docx")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--savepath",
        type=Path,
        default=Path("products/pulse_detector_table.md"),
        help="Path to directory containing detector configuration files",
    )
    args = parser.parse_args()
    main(args)
