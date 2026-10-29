#!/usr/bin/env python3
"""Extract one strike on every sensor of a site, for the dispersion figure.

The strike is given by its time at the sensor nearest the pile. On each other
sensor, the same strike is the detection nearest the time it would arrive
there: that time plus the extra range from the pile divided by the sound
speed. Detections and inventories are the soundscape workflow's (run
`detect` and `inventory` first), and data are calibrated to uPa with the
site file's calibration.

Each sensor's window, `pre` before its detection to `post` after, is written
as a TritonOA DataStream to <output>/disp_<sensor>.npz, which
scripts/plotting/plot_dispersion.py reads.

The defaults reproduce the 2025 example: strike 320 of event A10-B at South
Fork Wind (2023-07-09 10:30:00.55 UTC on the Tetra), with 1 s on either side.
Run on the server, where the raw data and inventories are.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl
from tritonoa.data.stream import DataStream, DataStreamStats

from soundscape.config import load_site
from soundscape.events import load_events, source_range
from soundscape.sources import read_segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Largest difference between the predicted arrival and the detection taken as
# the same strike (s); strikes at full energy are 1.4-1.7 s apart.
TOLERANCE_S = 0.5


def main(args: argparse.Namespace) -> None:
    site = load_site(args.site)
    event = next(e for e in load_events(site, [args.event]) if e.name == args.event)
    t_ref = np.datetime64(args.time, "us")
    sensors = [
        s for s in site.sensors if site.detections_path(event.name, s.name).exists()
    ]
    ranges = {s.name: source_range(event, s) for s in sensors}
    nearest = min(ranges.values())
    pre = np.timedelta64(round(args.pre * 1e6), "us")
    post = np.timedelta64(round(args.post * 1e6), "us")
    args.output.mkdir(parents=True, exist_ok=True)

    for sensor in sorted(sensors, key=lambda s: ranges[s.name]):
        delay_s = (ranges[sensor.name] - nearest) / args.sound_speed
        expected = t_ref + np.timedelta64(round(delay_s * 1e6), "us")
        times = (
            pl.read_csv(
                site.detections_path(event.name, sensor.name),
                schema_overrides={"time": pl.Datetime("us")},
            )["time"]
            .to_numpy()
            .astype("datetime64[us]")
        )
        k = np.argmin(np.abs(times - expected))
        offset_s = (times[k] - expected) / np.timedelta64(1, "s")
        if abs(offset_s) > TOLERANCE_S:
            logger.warning(
                f"{sensor.name}: nearest detection is {offset_s:+.2f} s from the "
                f"expected arrival {expected}; skipped"
            )
            continue
        seg = read_segment(
            sensor, site.inventory_path(sensor.name), times[k] - pre, times[k] + post
        )
        ds = DataStream(
            stats=DataStreamStats(
                channels=seg.channels,
                time_init=seg.time_init,
                time_end=seg.time_end,
                sampling_rate=seg.sampling_rate,
                units="uPa",
            ),
            data=seg.data,
        )
        path = args.output / f"disp_{sensor.name}.npz"
        ds.write(path)
        logger.info(
            f"{sensor.name} ({ranges[sensor.name] / 1e3:.1f} km): detection at "
            f"{times[k]} ({offset_s:+.3f} s from expected) -> {path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=Path("config/sites/southfork.toml"),
        help="Site file.",
    )
    parser.add_argument("--event", default="A10-B", help="Event name.")
    parser.add_argument(
        "--time",
        default="2023-07-09T10:30:00.55",
        help="Time of the strike at the sensor nearest the pile (UTC).",
    )
    parser.add_argument(
        "--pre", type=float, default=1.0, help="Window before the detection (s)."
    )
    parser.add_argument(
        "--post", type=float, default=1.0, help="Window after the detection (s)."
    )
    parser.add_argument(
        "--sound-speed", type=float, default=1500.0, help="Sound speed (m/s)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dispersion"),
        help="Output directory.",
    )
    main(parser.parse_args())
