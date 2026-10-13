#! /usr/bin/env python3
"""Soundscape analysis of pile-driving events at a wind farm site.

Stages, in order:
    inventory  index raw data for each sensor
    detect     detect strikes on each sensor's detection channel
    extract    window all channels around each strike
    metrics    per-pulse metrics; also writes the site-wide summary
    psd        power spectral density of each pulse
    ambient    ambient noise preceding each event
    run        all of the above
"""

import logging
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from collections.abc import Callable
from pathlib import Path

import polars as pl

from soundscape import detect as det
from soundscape.config import SiteConfig, load_site
from soundscape.events import Event, load_events
from soundscape.extract import extract
from soundscape.metrics import compute_metrics
from soundscape.plotting import plot_detections
from soundscape.sources import get_source
from soundscape.spectra import ambient_noise, pulse_psd, write_ambient_psd

logger = logging.getLogger(__name__)


class Workflow:
    def __init__(
        self,
        site: SiteConfig,
        events: list[str] | None = None,
        sensors: list[str] | None = None,
    ):
        self.site = site
        self.events = load_events(site, events)
        self.sensors = (
            site.sensors if sensors is None else [site.sensor(s) for s in sensors]
        )

    def _each(self, fn: Callable, label: str) -> None:
        """Call fn(event, sensor) for each selected, non-excluded pair."""
        for event in self.events:
            for sensor in self.sensors:
                if self.site.is_excluded(event.name, sensor.name):
                    logger.info(f"[{label}] {event.name}/{sensor.name}: excluded")
                    continue
                logger.info(f"[{label}] {event.name}/{sensor.name}")
                fn(event, sensor)

    def inventory(self) -> None:
        for sensor in self.sensors:
            dest = self.site.inventory_path(sensor.name)
            get_source(sensor).prepare(dest)
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
            detections = pl.read_csv(
                self.site.detections_path(event.name, sensor.name),
                schema_overrides={"time": pl.Datetime("us")},
            )
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
            compute_metrics(path).write_csv(
                self.site.metrics_path(event.name, sensor.name)
            )

        self._each(_metrics, "metrics")
        self.summarize()

    def summarize(self) -> None:
        """Concatenate all per-event metrics into one site-wide table."""
        files = sorted(self.site.output_dir.glob("events/*/metrics_*.csv"))
        if not files:
            return
        df = pl.concat(
            [pl.read_csv(f, try_parse_dates=True) for f in files], how="vertical"
        ).sort("time", "sensor", "channel")
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
            table, spectra = ambient_noise(
                sensor,
                self.site.inventory_path(sensor.name),
                event,
                self.site.ambient,
            )
            write_ambient_psd(
                self.site.ambient_path, event, sensor, spectra, self.site.ambient
            )
            tables.append(table)

        self._each(_ambient, "ambient")
        if tables:
            path = self.site.ambient_spl_path
            new = pl.concat(tables)
            if path.exists():
                # Keep results for events/sensors not rerun this time.
                old = pl.read_csv(path, try_parse_dates=True)
                keys = new.select("event", "sensor").unique()
                old = old.join(keys, on=["event", "sensor"], how="anti")
                new = pl.concat([old, new], how="vertical_relaxed")
            new.sort("event", "sensor", "channel", "interval_start").write_csv(path)

    def run(self) -> None:
        self.inventory()
        self.detect()
        self.extract()
        self.metrics()
        self.psd()
        self.ambient()


COMMANDS = ["inventory", "detect", "extract", "metrics", "psd", "ambient", "run"]


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument(
        "--site", type=Path, required=True, help="Site TOML, e.g. config/sites/southfork.toml"
    )
    parser.add_argument(
        "--event", action="append", help="Only this event (repeatable). "
        "Naming an event overrides its exclusion."
    )
    parser.add_argument("--sensor", action="append", help="Only this sensor (repeatable)")
    parser.add_argument(
        "--reselect",
        action="store_true",
        help="detect: re-apply thresholds to saved candidates without reading data",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="detect: skip diagnostic figures"
    )
    args = parser.parse_args()

    wf = Workflow(load_site(args.site), events=args.event, sensors=args.sensor)
    if args.command == "detect":
        wf.detect(reselect=args.reselect, plot=not args.no_plot)
    else:
        getattr(wf, args.command)()


if __name__ == "__main__":
    main()
