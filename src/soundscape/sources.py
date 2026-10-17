"""Data discovery and reading.

Every sensor has a data source that knows how to produce a TritonOA inventory
(a CSV index of records in the raw files). Everything downstream reads data
through that inventory, so it does not matter whether the raw files sit on a
local disk, a mounted share, or a remote server where the inventory was built.

To support a new kind of source, add a config model in `config.py` and a class
here implementing `DataSource`, then register it in `SOURCES`.

Inventories are always stored *unconditioned* (raw counts or volts); physical
units are applied by `soundscape.calibration` so that calibration lives in one
place, under version control, alongside the site configuration.
"""

import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
from tqdm import tqdm
from tritonoa.data.formats import factory
from tritonoa.data.formats.base import DataRecord
from tritonoa.data.inventory import Inventory
from tritonoa.data.reader import read_inventory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import ClockParameters

from soundscape.calibration import calibrate
from soundscape.config import (
    GlobSourceConfig,
    InventorySourceConfig,
    SensorConfig,
)

logger = logging.getLogger(__name__)

# What corrupt file contents raise while headers are parsed: undecodable or
# non-numeric text fields (ValueError, including UnicodeDecodeError) and
# records cut short (struct.error). Anything else, such as an OSError from the
# file system or a bug, is not a property of the file and is raised.
_CORRUPT_FILE_ERRORS = (ValueError, struct.error)

_NEUTRAL_CONDITIONING = {"adc_vref": "1.0", "gain": "0.0", "sensitivity": "0.0"}


class DataSource(Protocol):
    def prepare(self, destination: Path) -> Path:
        """Write a readable, unconditioned inventory to `destination`."""
        ...


class GlobSource:
    """Build an inventory by reading headers of files matching a glob."""

    def __init__(self, config: GlobSourceConfig, data_dir: Path | None = None):
        self.config = config
        self.data_dir = data_dir

    def directory(self) -> Path:
        """The source directory, resolved against the data directory."""
        directory = Path(self.config.directory)
        if directory.is_absolute():
            return directory
        if self.data_dir is None:
            raise ValueError(
                f"The data directory is needed to find '{directory}': pass --data-dir."
            )
        return Path(self.data_dir) / directory

    def prepare(self, destination: Path) -> Path:
        cfg = self.config
        directory = self.directory()
        clock = ClockParameters()
        if cfg.clock is not None:
            clock = ClockParameters(
                time_check_0=np.datetime64(cfg.clock.time_check_0, "us"),
                time_check_1=np.datetime64(cfg.clock.time_check_1, "us"),
                offset_0=cfg.clock.offset_0,
                offset_1=cfg.clock.offset_1,
            )
        logger.info(f"Building inventory from {directory}/{cfg.pattern}")
        files = Inventory._get_files(directory, cfg.pattern)
        # TritonOA's default SignalParams is shared between calls and resized
        # in place, so always pass a fresh one.
        conditioner = SignalParams()

        def read(file: Path) -> list[DataRecord] | None:
            try:
                return read_file_records(file, cfg.file_format, clock, conditioner)
            except _CORRUPT_FILE_ERRORS as exc:
                logger.warning(
                    f"Unreadable, left out: {file} ({type(exc).__name__}: {exc})"
                )
                return None

        with ThreadPoolExecutor(max_workers=min(len(files), 64)) as executor:
            results = list(
                tqdm(executor.map(read, files), total=len(files), desc="Reading files")
            )
        bad = [f.name for f, r in zip(files, results) if r is None]
        if len(bad) == len(files):
            raise RuntimeError(f"No readable files in {directory}/{cfg.pattern}.")
        if bad:
            logger.warning(
                f"{len(bad)} of {len(files)} files unreadable and left out of the "
                f"inventory: {', '.join(bad)}"
            )
        inv = Inventory([rec for r in results if r is not None for rec in r])
        destination.parent.mkdir(parents=True, exist_ok=True)
        inv.save(destination)
        return destination


def read_file_records(
    file: Path,
    file_format: str | None,
    clock: ClockParameters,
    conditioner: SignalParams,
) -> list[DataRecord]:
    """Inventory records for one file, as TritonOA's `Inventory.build` makes
    them; raises if the file cannot be read."""
    key = file.suffix if file_format is None else file_format
    reader, formatter = factory.get_reader(key), factory.get_formatter(key)
    records = [
        formatter.format_record(
            filename=file,
            record_number=i,
            header=header,
            clock=clock,
            conditioner=conditioner,
        )
        for i, header in enumerate(reader.read_headers(file))
    ]
    return formatter.callback(records)


class InventorySource:
    """Reuse an inventory built elsewhere, remapping file paths as needed.

    Any signal conditioning recorded in the inventory is removed so that the
    site calibration is the single source of truth.
    """

    def __init__(self, config: InventorySourceConfig, data_dir: Path | None = None):
        self.config = config

    def prepare(self, destination: Path) -> Path:
        df = pl.read_csv(self.config.inventory, infer_schema_length=0)
        df = remap_filenames(df, self.config.path_map)
        df = neutralize_conditioning(df)
        destination.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(destination)
        return destination


SOURCES: dict[str, type] = {"glob": GlobSource, "inventory": InventorySource}


def get_source(sensor: SensorConfig, data_dir: Path | None = None) -> DataSource:
    return SOURCES[sensor.source.type](sensor.source, data_dir)


def remap_filenames(df: pl.DataFrame, path_map: dict[str, str]) -> pl.DataFrame:
    """Replace leading path prefixes in the `filename` column."""
    expr = pl.col("filename")
    for old, new in path_map.items():
        expr = (
            pl.when(expr.str.starts_with(old))
            .then(pl.lit(new) + expr.str.slice(len(old)))
            .otherwise(expr)
        )
    return df.with_columns(expr.alias("filename"))


def neutralize_conditioning(df: pl.DataFrame) -> pl.DataFrame:
    """Set ADC reference, gain and sensitivity so TritonOA returns raw values.

    Older inventories (e.g. South Fork, 2025) lack `adc_vref` and store the
    hydrophone sensitivity, which TritonOA would otherwise apply on read.
    """
    cols = []
    for col, value in _NEUTRAL_CONDITIONING.items():
        if col in df.columns:
            values = [
                float(v) for row in df[col].drop_nulls() for v in row.split(",") if v
            ]
            if any(v != float(value) for v in values):
                logger.warning(
                    f"Inventory column '{col}' is not neutral; it is reset and "
                    "the site calibration is applied instead."
                )
        cols.append(
            pl.col("nch")
            .cast(pl.Int64)
            .map_elements(lambda n, v=value: ",".join([v] * n), return_dtype=pl.String)
            .alias(col)
        )
    return df.with_columns(cols)


def record_spans(inventory: Path) -> tuple[np.ndarray, np.ndarray]:
    """Start and end times (datetime64[us]) of the records in an inventory,
    sorted by start time."""
    df = (
        pl.read_csv(inventory, columns=["timestamp", "npts", "sampling_rate"])
        .with_columns(pl.col("timestamp").str.to_datetime(time_unit="us"))
        .sort("timestamp")
    )
    t0 = df["timestamp"].to_numpy()
    seconds = df["npts"].to_numpy() / df["sampling_rate"].to_numpy()
    return t0, t0 + (seconds * 1e6).round().astype("timedelta64[us]")


def coverage(inventory: Path, start: np.datetime64, end: np.datetime64) -> float:
    """Fraction of [start, end) spanned by records listed in an inventory."""
    start, end = np.datetime64(start, "us"), np.datetime64(end, "us")
    t0, t1 = record_spans(inventory)
    covered = (np.minimum(t1, end) - np.maximum(t0, start)).clip(min=np.timedelta64(0))
    return min(1.0, covered.sum() / (end - start))


def recorded_intervals(
    inventory: Path, max_gap_s: float = 60.0
) -> list[tuple[np.datetime64, np.datetime64]]:
    """Continuous recording intervals, merging records less than `max_gap_s`
    apart."""
    t0, t1 = record_spans(inventory)
    gap = np.timedelta64(round(max_gap_s * 1e6), "us")
    intervals: list[tuple[np.datetime64, np.datetime64]] = []
    for a, b in zip(t0, t1):
        if intervals and a - intervals[-1][1] <= gap:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], b))
        else:
            intervals.append((a, b))
    return intervals


@dataclass
class Segment:
    """A calibrated multichannel time series (uPa)."""

    data: np.ndarray  # (n_channels, n_samples)
    sampling_rate: float
    time_init: np.datetime64  # time of the first sample
    channels: list[int]

    @property
    def time_end(self) -> np.datetime64:
        return self.time_init + self.duration(self.data.shape[1])

    def duration(self, n: int) -> np.timedelta64:
        return np.timedelta64(round(n / self.sampling_rate * 1e6), "us")

    def times(self) -> np.ndarray:
        offsets = np.arange(self.data.shape[1]) / self.sampling_rate * 1e6
        return self.time_init + offsets.round().astype("timedelta64[us]")

    def index(self, t: np.datetime64) -> int:
        seconds = (t - self.time_init) / np.timedelta64(1, "s")
        return round(seconds * self.sampling_rate)


def read_segment(
    sensor: SensorConfig,
    inventory: Path,
    start: np.datetime64,
    end: np.datetime64,
    channels: list[int] | None = None,
) -> Segment:
    """Read and calibrate data for a sensor between `start` and `end`.

    Gaps in the recording are zero-filled by TritonOA.
    """
    channels = sensor.channels if channels is None else channels
    ds = read_inventory(
        file_path=inventory,
        time_start=np.datetime64(start, "us"),
        time_end=np.datetime64(end, "us"),
        channels=channels,
    )
    data = calibrate(sensor.calibration, ds.data, ds.stats.sampling_rate, channels)
    return Segment(
        data=data,
        sampling_rate=float(ds.stats.sampling_rate),
        time_init=np.datetime64(ds.stats.time_init, "us"),
        channels=list(channels),
    )
