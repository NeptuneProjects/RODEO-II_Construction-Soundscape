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
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION

from rodeo import detect, metrics, utils

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_configs(invpath: Path, eventpath: Path) -> tuple[dict, dict]:
    with open(invpath, "rb") as f:
        inv_config = tomllib.load(f)
    with open(eventpath, "rb") as f:
        event_config = tomllib.load(f)["event"][0]
    return inv_config, event_config


def load_data(inv_config: dict, event: dict, sensor_name: str, sensor_channels) -> DataStream:
    invpath = Path(inv_config[sensor_name]["data"]["destination"])
    return read_inventory(
        file_path=invpath,
        time_start=np.datetime64(event["start"], TIME_PRECISION),
        time_end=np.datetime64(event["end"], TIME_PRECISION),
        channels=[i["channel"] for i in sensor_channels],
    )


def run_workflow(
    inv_config: dict, event: dict, df_equip: pd.DataFrame, df_turb: pd.DataFrame
) -> None:
    base_path = Path.cwd() / "results" / "events"

    event_name = event["name"]
    event_site = event["site"]
    event_start = event["start"]

    logging.info(f"Analyzing event {event_name} at site {event_site} at {event_start}.")

    path = base_path / f"{event_name}_{event_start[0:10]}"
    path.mkdir(parents=True, exist_ok=True)

    for sensor in event["sensors"]:
        sensor_name = sensor["name"]
        sensor_filter = sensor.get("filter", None)
        sensor_freq = sensor.get("freq", None)
        sensor_channels = sensor["channels"]

        logging.info(f"    Analayzing sensor {sensor_name}.")

        ds = load_data(inv_config, event, sensor_name, sensor_channels)
        return
        if sensor_filter is not None:
            ds = ds.filter(filt_type=sensor_filter, freq=sensor_freq)

        # Compute the range between the sensor and the event site
        site_lon = df_turb.loc[df_turb["Turbine"] == event_site, "lon"].values[0]
        site_lat = df_turb.loc[df_turb["Turbine"] == event_site, "lat"].values[0]
        sensor_lon = df_equip.loc[
            df_equip["Name"] == sensor_name, "Longitude [dd]"
        ].values[0]
        sensor_lat = df_equip.loc[
            df_equip["Name"] == sensor_name, "Latitude [dd]"
        ].values[0]

        r = utils.compute_range(sensor_lon, sensor_lat, site_lon, site_lat)

        # Run analyses for individual pulses
        dbpath = path / f"detections_{sensor_name}.hdf5"
        detect.process_stream(
            ds,
            pulse_start=[d["pulse_start"] for d in sensor["channels"]],
            pulse_end=[d["pulse_end"] for d in sensor["channels"]],
            stwin=[d["stwin"] for d in sensor["channels"]],
            ltwin=[d["ltwin"] for d in sensor["channels"]],
            threshold=[d["threshold"] for d in sensor["channels"]],
            distance=[d["distance"] for d in sensor["channels"]],
            taper=sensor["taper"],
            station=sensor_name,
            dbpath=dbpath,
            metadata={
                "event_name": event_name,
                "event_site": event_site,
                "range": r,
            },
        )

        destination = path / f"metrics_{sensor_name}.csv"
        all_metrics = metrics.compute_metrics(dbpath)
        metrics.save_metrics_to_csv(all_metrics, destination)

        logging.info(f"    Completed sensor {sensor_name}.")
    logging.info(f"Completed {event_name} at {event_site} on {event_start[0:10]}.")

    return


def main(args: argparse.Namespace) -> None:
    data, event = load_configs(args.inv, args.event)

    df_equip = pd.read_csv(Path(os.getenv("EQUIPCONFIG")))
    df_turb = pd.read_csv(Path(os.getenv("TURBCONFIG")))

    run_workflow(data, event, df_equip, df_turb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inv",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    parser.add_argument("--event", type=Path, default=Path("config/events/A08.toml"))
    args = parser.parse_args()
    main(args)
