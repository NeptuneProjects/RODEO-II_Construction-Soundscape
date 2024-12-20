#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path
import tomllib

from datautils.data import DataStream, read
from datautils.query import CatalogueQuery
from datautils.time import TIME_PRECISION
import numpy as np
import pandas as pd
import pyproj

import detect, metrics


BASEPATH = Path("data/results")
SENSOR_DATA = Path("data/arrays/equipment.csv")
TURBINE_DATA = Path("data/logs/turbines.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def load_configs(data: Path, events: Path) -> tuple[dict, dict]:
    with open(data, "rb") as f:
        data = tomllib.load(f)
    with open(events, "rb") as f:
        events = tomllib.load(f)
    return data, events


def load_data(data: dict, event: dict, sensor: dict) -> DataStream:
    catalogue = get_catalogue(data["data"]["sensors"], sensor["name"])
    cq = CatalogueQuery(
        catalogue=catalogue,
        destination="data/interrogation",
        time_start=np.datetime64(event["start"], TIME_PRECISION),
        time_end=np.datetime64(event["end"], TIME_PRECISION),
        channels=[i["channel"] for i in sensor["channels"]],
    )
    return read(cq)


def get_catalogue(sensors: list[dict], sensor: str) -> Path:
    return Path(
        next(
            (item["catalogue"] for item in sensors if item["name"] == sensor),
            None,
        )
    )


def get_range(
    point1_lon: float, point1_lat: float, point2_lon: float, point2_lat: float
) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist = geod.inv(point1_lon, point1_lat, point2_lon, point2_lat)
    return dist


def run_workflow(data: dict, events: dict) -> None:
    for event in events["events"]:
        event_name = event["name"]
        event_site = event["site"]
        event_start = event["start"]

        logging.info(
            f"Analyzing event {event_name} at site {event_site} at {event_start}."
        )

        path = BASEPATH / f"{event_name}_{event_start[0:10]}"
        path.mkdir(parents=True, exist_ok=True)
        for sensor in event["sensors"]:

            sensor_name = sensor["name"]
            sensor_filter = sensor.get("filter", None)
            sensor_freq = sensor.get("freq", None)

            logging.info(f"    Analayzing sensor {sensor_name}.")

            ds = load_data(data, event, sensor)
            if sensor_filter is not None:
                ds = ds.filter(filt_type=sensor_filter, freq=sensor_freq)

            # Compute the range between the sensor and the event site
            df_t = pd.read_csv(TURBINE_DATA)
            site_lon = df_t.loc[df_t["Turbine"] == event_site, "lon"].values[0]
            site_lat = df_t.loc[df_t["Turbine"] == event_site, "lat"].values[0]
            df_s = pd.read_csv(SENSOR_DATA)
            sensor_lon = df_s.loc[df_s["Name"] == sensor_name, "Longitude [dd]"].values[
                0
            ]
            sensor_lat = df_s.loc[df_s["Name"] == sensor_name, "Latitude [dd]"].values[
                0
            ]

            r = get_range(sensor_lon, sensor_lat, site_lon, site_lat)

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
    data, events = load_configs(args.data, args.events)
    run_workflow(data, events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("src/config/data.toml"))
    parser.add_argument(
        "--events", type=Path, default=Path("src/config/events/A08.toml")
    )
    args = parser.parse_args()
    main(args)
