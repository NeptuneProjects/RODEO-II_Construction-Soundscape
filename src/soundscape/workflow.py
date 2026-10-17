#! /usr/bin/env python3
"""Soundscape analysis of pile-driving events at a wind farm site.

Stages, in order:
    inventory  index raw data for each sensor
    detect     detect strikes on each sensor's detection channel
    extract    window all channels around each strike
    metrics    per-pulse metrics; also writes the site-wide summary
    psd        power spectral density of each pulse
    ambient    ambient noise preceding each event
    report     figures and LaTeX tables
    run        all of the above
"""

import logging
import shlex
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import polars as pl
from tritonoa.data.reader import NoDataError

from soundscape import detect as det
from soundscape.config import SensorConfig, SiteConfig, load_site
from soundscape.events import Event, load_events
from soundscape.extract import extract
from soundscape.metrics import compute_metrics, read_table
from soundscape.plotting import plot_detections
from soundscape.report import make_report
from soundscape.sources import coverage, get_source
from soundscape.spectra import ambient_noise, pulse_psd, write_ambient_psd

logger = logging.getLogger(__name__)


def with_hydrophones(df: pl.DataFrame, sensor: SensorConfig) -> pl.DataFrame:
    """Add hydrophone depth and serial number columns after `channel`."""
    channels = sorted(df["channel"].unique().to_list())
    info = pl.DataFrame(
        {
            "channel": channels,
            "depth_m": [sensor.hydrophone(c).depth_m for c in channels],
            "serial_number": [sensor.hydrophone(c).serial_number for c in channels],
        },
        schema={
            "channel": df.schema["channel"],
            "depth_m": pl.Float64,
            "serial_number": pl.String,
        },
    )
    cols = df.columns
    at = cols.index("channel") + 1
    order = cols[:at] + ["depth_m", "serial_number"] + cols[at:]
    return df.join(info, on="channel", how="left").select(order)


class Workflow:
    def __init__(
        self,
        site: SiteConfig,
        events: list[str] | None = None,
        sensors: list[str] | None = None,
        data_dir: Path | None = None,
    ):
        self.site = site
        self.data_dir = data_dir
        self.events = load_events(site, events)
        # Events named explicitly run even if the site file excludes them.
        self.named = set(events or [])
        self.sensors = (
            site.sensors if sensors is None else [site.sensor(s) for s in sensors]
        )
        self._coverage_cache: dict[tuple[str, str], float | None] = {}

    def _each(self, fn: Callable, label: str) -> None:
        """Call fn(event, sensor) for each selected pair that is not excluded
        (unless its event was named) and has recorded data during the event."""
        for event in self.events:
            for sensor in self.sensors:
                pair = f"{event.name}/{sensor.name}"
                if event.name not in self.named and self.site.is_excluded(
                    event.name, sensor.name
                ):
                    logger.info(f"[{label}] {pair}: excluded")
                    continue
                frac = self._coverage(event, sensor)
                if frac == 0.0:
                    logger.warning(f"[{label}] {pair}: no data during event, skipped")
                    continue
                if frac is not None and frac < 0.999:
                    logger.warning(f"[{label}] {pair}: data cover {frac:.0%} of event")
                logger.info(f"[{label}] {pair}")
                fn(event, sensor)

    def _coverage(self, event: Event, sensor: SensorConfig) -> float | None:
        """Fraction of the event the sensor recorded; None without an inventory."""
        key = (event.name, sensor.name)
        if key not in self._coverage_cache:
            path = self.site.inventory_path(sensor.name)
            self._coverage_cache[key] = (
                coverage(path, event.start, event.end) if path.exists() else None
            )
        return self._coverage_cache[key]

    def inventory(self) -> None:
        for sensor in self.sensors:
            dest = self.site.inventory_path(sensor.name)
            get_source(sensor, self.data_dir).prepare(dest)
            logger.info(f"[inventory] {sensor.name} -> {dest}")

    def detect(self, reselect: bool = False, plot: bool = True) -> None:
        def _detect(event: Event, sensor) -> None:
            config = self.site.detector_for(sensor)
            cand_path = self.site.candidates_path(event.name, sensor.name)
            if reselect:
                candidates = pl.read_parquet(cand_path)
            else:
                candidates = det.scan(
                    sensor, self.site.inventory_path(sensor.name), event, config
                )
                cand_path.parent.mkdir(parents=True, exist_ok=True)
                candidates.write_parquet(cand_path)

            scored = det.select(candidates, config)
            strikes = det.strikes_table(scored, event, sensor)
            strikes.write_csv(self.site.detections_path(event.name, sensor.name))
            logger.info(
                f"[detect] {event.name}/{sensor.name}: {len(strikes)} strikes "
                f"from {len(candidates)} candidates"
            )
            if plot and not scored.is_empty():
                plot_detections(
                    scored,
                    config,
                    f"{event.name} / {sensor.name}",
                    self.site.event_dir(event.name) / f"detections_{sensor.name}.png",
                )

        self._each(_detect, "detect")

    def extract(self) -> None:
        def _extract(event: Event, sensor) -> None:
            path = self.site.detections_path(event.name, sensor.name)
            if not path.exists():
                logger.warning(f"[extract] missing {path}")
                return
            detections = pl.read_csv(path, schema_overrides={"time": pl.Datetime("us")})
            if detections.is_empty():
                logger.warning(f"[extract] {event.name}/{sensor.name}: no strikes")
                return
            extract(
                sensor,
                self.site.inventory_path(sensor.name),
                event,
                detections,
                self.site.extract,
                self.site.detector.chunk_s,
                self.site.pulses_path(event.name, sensor.name),
            )

        self._each(_extract, "extract")

    def metrics(self) -> None:
        def _metrics(event: Event, sensor) -> None:
            path = self.site.pulses_path(event.name, sensor.name)
            if not path.exists():
                logger.warning(f"[metrics] missing {path}")
                return
            with_hydrophones(compute_metrics(path), sensor).write_csv(
                self.site.metrics_path(event.name, sensor.name)
            )

        self._each(_metrics, "metrics")
        self.summarize()

    def summarize(self) -> None:
        """Concatenate per-event metrics into one site-wide table.

        Covers every event and sensor in the site file (not only those selected
        on the command line); results for sensors or events no longer in the
        site file, or now excluded, are left out.
        """
        files = [
            path
            for event in load_events(self.site)
            for sensor in self.site.sensors
            if not self.site.is_excluded(event.name, sensor.name)
            and (path := self.site.metrics_path(event.name, sensor.name)).exists()
        ]
        if not files:
            return
        df = pl.concat([read_table(f) for f in files], how="diagonal_relaxed").sort(
            "time", "sensor", "channel"
        )
        df.write_csv(self.site.metrics_summary_path)
        logger.info(f"[metrics] {len(df):,} rows -> {self.site.metrics_summary_path}")

    def psd(self) -> None:
        def _psd(event: Event, sensor) -> None:
            path = self.site.pulses_path(event.name, sensor.name)
            if not path.exists():
                logger.warning(f"[psd] missing {path}")
                return
            pulse_psd(
                path,
                self.site.pulse_psd,
                self.site.pulse_psd_path(event.name, sensor.name),
            )

        self._each(_psd, "psd")

    def ambient(self) -> None:
        tables = []

        def _ambient(event: Event, sensor) -> None:
            try:
                table, spectra = ambient_noise(
                    sensor,
                    self.site.inventory_path(sensor.name),
                    event,
                    self.site.ambient,
                )
            except NoDataError:
                logger.warning(f"[ambient] {event.name}/{sensor.name}: no data")
                return
            write_ambient_psd(
                self.site.ambient_path, event, sensor, spectra, self.site.ambient
            )
            tables.append(with_hydrophones(table, sensor))

        self._each(_ambient, "ambient")
        if tables:
            path = self.site.ambient_spl_path
            new = pl.concat(tables)
            if path.exists():
                # Keep results for events/sensors not rerun this time.
                old = read_table(path).filter(
                    pl.col("sensor").is_in([s.name for s in self.site.sensors])
                )
                keys = new.select("event", "sensor").unique()
                old = old.join(keys, on=["event", "sensor"], how="anti")
                new = pl.concat([old, new], how="diagonal_relaxed").select(new.columns)
            new.sort("event", "sensor", "channel", "interval_start").write_csv(path)

    def report(self) -> None:
        make_report(self.site, self.events)

    def run(self) -> None:
        self.inventory()
        self.detect()
        self.extract()
        self.metrics()
        self.psd()
        self.ambient()
        self.report()


def log_to_file(log_dir: Path, command: str) -> Path:
    """Copy log messages to `<log_dir>/<command>_<YYYYmmdd-HHMMSS>.log`, in
    the same format as the terminal."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{command}_{datetime.now():%Y%m%d-%H%M%S}.log"
    handler = logging.FileHandler(path)
    root = logging.getLogger()
    if root.handlers:
        handler.setFormatter(root.handlers[0].formatter)
    root.addHandler(handler)
    return path


COMMANDS = [
    "inventory",
    "detect",
    "extract",
    "metrics",
    "psd",
    "ambient",
    "report",
    "run",
]


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--site",
        type=Path,
        required=True,
        help="Site TOML, e.g. config/sites/southfork.toml",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory holding the site's raw data; data source directories in "
        "the site file are relative to it (inventory, run)",
    )
    parser.add_argument(
        "--event",
        action="append",
        help="Only this event (repeatable). Naming an event overrides its exclusion.",
    )
    parser.add_argument(
        "--sensor", action="append", help="Only this sensor (repeatable)"
    )
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="detect: re-apply thresholds to saved candidates without reading data",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="detect: skip diagnostic figures"
    )
    args = parser.parse_args()

    site = load_site(args.site)
    path = log_to_file(site.log_dir, args.command)
    logger.info(f"Command: {shlex.join(sys.argv)}")
    logger.info(f"Log: {path}")
    wf = Workflow(
        site,
        events=args.event,
        sensors=args.sensor,
        data_dir=args.data_dir,
    )
    if args.command == "detect":
        wf.detect(reselect=args.reselect, plot=not args.no_plot)
    else:
        getattr(wf, args.command)()


if __name__ == "__main__":
    main()
