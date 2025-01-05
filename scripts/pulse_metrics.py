#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import tomllib

from rodeo import metrics


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def compute_metrics(
    event: dict,
) -> None:
    logging.info(f"Computing metrics for event {event_name}.")

    event_name = event["name"]
    event_start = event["start"][0:10]

    path = Path.cwd() / "results" / "events" / f"{event_name}_{event_start}"
    path.mkdir(parents=True, exist_ok=True)

    for sensor in event["sensors"]:
        sensor_name = sensor["name"]
        logging.info(f"    Computing metrics for {sensor_name}.")

        dbpath = path / f"detections_{sensor_name}.hdf5"

        all_metrics = metrics.compute_metrics(dbpath)
        metrics.save_metrics_to_csv(all_metrics, path / f"metrics_{sensor_name}.csv")

        logging.info(f"    Completed {sensor_name}.")
    logging.info(f"Completed {event_name}.")


def main(args: argparse.Namespace) -> None:
    with open(args.event, "rb") as f:
        event_config = tomllib.load(f)["event"][0]

    compute_metrics(event_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path, default=Path("config/events/A08.toml"))
    args = parser.parse_args()
    main(args)
