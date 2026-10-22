"""Pile-driving events and source-receiver geometry."""

from dataclasses import dataclass

import numpy as np
import polars as pl
from pyproj import Geod

from soundscape.config import SensorConfig, SiteConfig

_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class Event:
    name: str
    pile: str
    start: np.datetime64
    end: np.datetime64
    latitude: float
    longitude: float


def load_events(site: SiteConfig, names: list[str] | None = None) -> list[Event]:
    """Load events joined with pile positions, dropping excluded events.

    `events` CSV columns: event, pile, start, end (ISO 8601, UTC).
    `piles` CSV columns: pile, latitude, longitude.
    """
    events = pl.read_csv(site.events)
    piles = pl.read_csv(site.piles)
    df = events.join(piles, on="pile", how="left")
    missing = df.filter(pl.col("latitude").is_null())["pile"].to_list()
    if missing:
        raise ValueError(f"Piles missing from {site.piles}: {sorted(set(missing))}")

    out = []
    for row in df.iter_rows(named=True):
        if names is not None and row["event"] not in names:
            continue
        if names is None and site.is_excluded(row["event"]):
            continue
        out.append(
            Event(
                name=row["event"],
                pile=row["pile"],
                start=np.datetime64(row["start"], "us"),
                end=np.datetime64(row["end"], "us"),
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
        )
    if names is not None:
        unknown = set(names) - {e.name for e in out}
        if unknown:
            raise KeyError(f"Unknown events: {sorted(unknown)}")
    return out


def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Geodesic distance (m) between two positions."""
    return float(_GEOD.inv(lon1, lat1, lon2, lat2)[2])


def source_range(event: Event, sensor: SensorConfig) -> float:
    """Geodesic distance (m) between the pile and the sensor."""
    _, _, dist = _GEOD.inv(
        event.longitude, event.latitude, sensor.longitude, sensor.latitude
    )
    return float(dist)
