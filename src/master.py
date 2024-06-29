#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import tomllib

from datautils.data import DataStream, read
from datautils.query import CatalogueQuery
from datautils.time import TIME_PRECISION
import matplotlib.pyplot as plt
import numpy as np

import detect, plotting


BASEPATH = Path("data/results")


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


def run_workflow(data: dict, events: dict) -> None:
    for event in events["events"]:
        event_name = event["name"]
        event_site = event["site"]
        event_start = event["start"]

        path = BASEPATH / f"{event_name}_{event_start[0:10]}"
        path.mkdir(parents=True, exist_ok=True)
        for sensor in event["sensors"]:

            sensor_name = sensor["name"]
            sensor_filter = sensor.get("filter", None)
            sensor_freq = sensor.get("freq", None)

            ds = load_data(data, event, sensor)
            if sensor_filter is not None:
                ds = ds.filter(filt_type=sensor_filter, freq=sensor_freq)

            # # Time series
            # fig = plotting.plot_time_series(
            #     ds,
            #     title=f"Site {event_site} | {event_start[0:10]} | {sensor_name.replace('_', '/')}\nPressure [$\\mu$Pa]",
            # )
            # fig.savefig(path / f"time-series_{sensor_name}.png", dpi=600, bbox_inches="tight", facecolor="w")
            # plt.close(fig)

            # # Spectrogram
            # fig = plotting.plot_spectrograms(
            #     ds,
            #     fmin=10,
            #     fmax=1000,
            #     vmin=60,
            #     vmax=120,
            # )
            # fig.savefig(path / f"spectrogram_{sensor_name}.png", dpi=600, bbox_inches="tight", facecolor="w")
            # plt.close(fig)

            # Run analyses for individual pulses
            detect.process_stream(
                ds,
                pulse_start=[d["pulse_start"] for d in sensor["channels"]],
                pulse_end=[d["pulse_end"] for d in sensor["channels"]],
                stwin=[d["stwin"] for d in sensor["channels"]],
                ltwin=[d["ltwin"] for d in sensor["channels"]],
                threshold=[d["threshold"] for d in sensor["channels"]],
                distance=[d["distance"] for d in sensor["channels"]],
                taper=sensor["taper"],
                station=event_site,
                dbpath=path / f"detections_{sensor_name}.hdf5",
            )

            # figs = plotting.plot_detection_diagnostic(
            #     ds,
            #     stwin=[d["stwin"] for d in sensor["channels"]],
            #     ltwin=[d["ltwin"] for d in sensor["channels"]],
            #     threshold=[d["threshold"] for d in sensor["channels"]],
            #     distance=[d["distance"] for d in sensor["channels"]],
            #     taper=sensor["taper"],
            #     sensor=sensor_name,
            # )
            # for i, fig in enumerate(figs):
            #     fig.savefig(
            #         path / f"detection-diagnostic_{sensor_name}_chan{i}.png",
            #         dpi=600,
            #         bbox_inches="tight",
            #         facecolor="w",
            #     )
            #     plt.close(fig)

    return


def main(args: argparse.Namespace) -> None:
    data, events = load_configs(args.data, args.events)
    run_workflow(data, events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("src/config/data.toml"))
    parser.add_argument("--events", type=Path, default=Path("src/config/events.toml"))
    args = parser.parse_args()
    main(args)
