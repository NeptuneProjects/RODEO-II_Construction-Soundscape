#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path
import tomllib

from dotenv import load_dotenv
import pandas as pd

from rodeo import detect


load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_configs(invpath: Path, eventpath: Path) -> tuple[dict, dict]:
    with open(invpath, "rb") as f:
        inv_config = tomllib.load(f)
    with open(eventpath, "rb") as f:
        event_config = tomllib.load(f)["event"][0]
    return inv_config, event_config


def main(args: argparse.Namespace) -> None:
    data, event = load_configs(args.inv, args.event)

    df_equip = pd.read_csv(Path(os.getenv("EQUIPCONFIG")))
    df_turb = pd.read_csv(Path(os.getenv("TURBCONFIG")))

    detect.event_detections(
        data,
        event,
        df_equip,
        df_turb,
        pulse_filter=args.filter,
        pulse_filt_freq=args.filt_freq,
    )


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
