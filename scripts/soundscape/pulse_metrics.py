#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path
import tomllib

import dotenv

from rodeo.soundscape import compute_event_metrics

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main(args: argparse.Namespace) -> None:
    with open(args.event, "rb") as f:
        event_config = tomllib.load(f)["event"][0]

    compute_event_metrics(event_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, default=Path("config/events/A08.toml"))
    args = parser.parse_args()
    main(args)
