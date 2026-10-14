"""Synthetic pile-driving recordings for tests that need no real data."""

from pathlib import Path

import numpy as np
import pytest

from soundscape.config import (
    DetectorConfig,
    GlobSourceConfig,
    HydrophoneConfig,
    SensorConfig,
    SiteConfig,
)
from soundscape.sources import Segment

FS = 2000.0
T0 = np.datetime64("2023-07-06T15:00:00", "us")


def make_pulse(fs: float, f0: float = 200.0, duration: float = 0.15) -> np.ndarray:
    """A decaying tone burst, roughly like a distant pile strike."""
    t = np.arange(int(duration * fs)) / fs
    return np.sin(2 * np.pi * f0 * t) * np.exp(-t / 0.03)


class SyntheticRecording:
    """Noise with pulses on every channel at known times and amplitudes."""

    def __init__(
        self,
        duration_s: float = 900.0,
        interval_s: float = 1.5,
        n_channels: int = 2,
        noise: float = 1.0,
        gap: tuple[float, float] | None = None,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        n = int(duration_s * FS)
        self.data = noise * rng.standard_normal((n_channels, n))
        pulse = make_pulse(FS)
        # Soft start: amplitude ramps up over the first two minutes, with a
        # pause in the middle of the record.
        onsets = np.arange(10.0, duration_s - 10.0, interval_s)
        pause = (onsets > duration_s / 2) & (onsets < duration_s / 2 + 60)
        onsets = onsets[~pause]
        amps = 20 + 180 * np.clip((onsets - 10.0) / 120.0, 0, 1)
        for t, a in zip(onsets, amps):
            i = int(t * FS)
            self.data[:, i : i + len(pulse)] += a * pulse
        if gap is not None:
            self.data[:, int(gap[0] * FS) : int(gap[1] * FS)] = 0.0
            ends = onsets + len(pulse) / FS
            keep = (ends <= gap[0]) | (onsets >= gap[1])
            onsets, amps = onsets[keep], amps[keep]
        self.onsets = onsets
        self.amps = amps
        self.peak_times = T0 + (onsets * 1e6).astype("timedelta64[us]")

    def read(self, sensor, inventory, start, end, channels=None) -> Segment:
        channels = sensor.channels if channels is None else channels
        i0 = max(0, int(np.ceil((start - T0) / np.timedelta64(1, "us") * 1e-6 * FS)))
        i1 = min(
            self.data.shape[1], int((end - T0) / np.timedelta64(1, "us") * 1e-6 * FS)
        )
        rows = [sensor.channels.index(c) for c in channels]
        return Segment(
            data=self.data[rows, i0:i1].copy(),
            sampling_rate=FS,
            time_init=T0 + np.timedelta64(round(i0 / FS * 1e6), "us"),
            channels=list(channels),
        )


@pytest.fixture
def recording() -> SyntheticRecording:
    return SyntheticRecording()


@pytest.fixture
def sensor() -> SensorConfig:
    return SensorConfig(
        name="VLA1",
        latitude=41.08506,
        longitude=-71.07804,
        channels=[0, 1],
        detect_channel=0,
        source=GlobSourceConfig(directory=Path("."), pattern="*.D23"),
        hydrophones=[
            HydrophoneConfig(channel=0, depth_m=16.2, serial_number="722013"),
            HydrophoneConfig(channel=1, depth_m=17.7, serial_number="722014"),
        ],
    )


@pytest.fixture
def detector() -> DetectorConfig:
    return DetectorConfig(band=(100.0, 300.0), chunk_s=120.0)


@pytest.fixture
def site(tmp_path, sensor) -> SiteConfig:
    events = tmp_path / "events.csv"
    events.write_text(
        "event,pile,start,end\nE1,P1,2023-07-06T15:00:00,2023-07-06T15:15:00\n"
    )
    piles = tmp_path / "piles.csv"
    piles.write_text("pile,latitude,longitude\nP1,41.09197,-71.16789\n")
    return SiteConfig(
        name="test",
        output_dir=tmp_path / "results",
        events=events,
        piles=piles,
        sensors=[sensor],
        detector=DetectorConfig(chunk_s=120.0),
    )
