"""Site configuration.

A site is described by a single TOML file (see `config/sites/`). Paths may
reference environment variables using `${VAR}` or `${VAR:-default}`, which
replaces the per-machine `paths.toml`/`.env` scheme: set e.g.
`SOUTHFORK_DATA=/Volumes/T7_Shield/...` on a laptop and leave it unset on the
server to fall back to the default.

Relative paths are resolved against the current working directory, so the
workflow is expected to be run from the repository root.
"""

import os
import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: str) -> str:
    """Expand `${VAR}` and `${VAR:-default}` in a string.

    Raises:
        KeyError: If a variable without a default is not set.
    """

    def _sub(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise KeyError(f"Environment variable '{name}' is not set and has no default.")

    return _ENV_PATTERN.sub(_sub, value)


def _expand_paths(data):
    """Recursively expand environment variables in all strings of a TOML dict."""
    if isinstance(data, dict):
        return {k: _expand_paths(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_paths(v) for v in data]
    if isinstance(data, str):
        return expand_env(data)
    return data


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------
# Data sources (data discovery)
# --------------------------------------------------------------------------


class ClockConfig(StrictModel):
    """Linear clock-drift correction between two time checks."""

    time_check_0: datetime
    time_check_1: datetime
    offset_0: float
    offset_1: float


class GlobSourceConfig(StrictModel):
    """Raw files discovered by globbing a directory (local disk or mounted share).

    An inventory is built from the file headers by `soundscape inventory`.
    """

    type: Literal["glob"] = "glob"
    directory: Path
    pattern: str
    file_format: str | None = None
    clock: ClockConfig | None = None


class InventorySourceConfig(StrictModel):
    """A prebuilt TritonOA inventory CSV, e.g. one built on the remote server.

    `path_map` rewrites filename prefixes so an inventory built on one machine
    can be read on another, e.g. `{"/project/RODEO_II_South_Fork_Data" =
    "/Volumes/T7_Shield/southfork"}`.
    """

    type: Literal["inventory"] = "inventory"
    inventory: Path
    path_map: dict[str, str] = Field(default_factory=dict)


DataSourceConfig = Annotated[
    GlobSourceConfig | InventorySourceConfig, Field(discriminator="type")
]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


class NoCalibration(StrictModel):
    """Data are already in uPa."""

    type: Literal["none"] = "none"


class FlatCalibration(StrictModel):
    """Frequency-independent calibration of ADC counts (e.g. SHRU).

    p [uPa] = counts * (adc_halfscale_v / adc_max_count) / 10^((S + G) / 20)
    where S is hydrophone sensitivity [dB re 1 V/uPa] and G is fixed gain [dB].
    Lists are per channel, indexed by channel number.
    """

    type: Literal["flat"] = "flat"
    sensitivity: list[float]
    fixed_gain: list[float]
    adc_halfscale_v: float = 2.5
    adc_max_count: int = 2**23 - 1

    @field_validator("sensitivity")
    @classmethod
    def _negative_sensitivity(cls, v: list[float]) -> list[float]:
        if any(s > 0 for s in v):
            raise ValueError(
                "Sensitivity must be in dB re 1 V/uPa (negative, e.g. -204.0)."
            )
        return v


class FrequencyResponseCalibration(StrictModel):
    """Frequency-dependent calibration of a voltage time series.

    `file` is a CSV with columns (frequency [Hz], sensitivity [dB re 1 V/uPa]).
    `correction_db` is added to the tabulated sensitivity.
    """

    type: Literal["frequency_response"] = "frequency_response"
    file: Path
    correction_db: float = 0.0


CalibrationConfig = Annotated[
    NoCalibration | FlatCalibration | FrequencyResponseCalibration,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# Processing stages
# --------------------------------------------------------------------------

FilterType = Literal["bandpass", "highpass", "lowpass"]


class FilterConfig(StrictModel):
    type: FilterType
    freq: float | tuple[float, float]
    order: int = 4

    @model_validator(mode="after")
    def _check_freq(self) -> "FilterConfig":
        if self.type == "bandpass" and not isinstance(self.freq, tuple):
            raise ValueError("A bandpass filter needs freq = [low, high].")
        if self.type != "bandpass" and isinstance(self.freq, tuple):
            raise ValueError(f"A {self.type} filter needs a single corner frequency.")
        return self


class DetectorConfig(StrictModel):
    """Band-limited energy detector for pile-driving strikes.

    The detection channel is band-passed, squared, smoothed over `smooth_s`
    (0 disables smoothing, as in Vineyard Wind), and local maxima at least
    `min_interval_s` apart become candidates. A candidate is a strike when its
    energy is at least `threshold` times the largest candidate energy within
    `normalization_window_s` (or within the whole event if 0, which is the
    Vineyard Wind behavior) and at least `min_snr_db` above the noise floor.

    Pile strikes come in regular trains, so isolated transients (e.g. mooring
    noise) can be rejected by requiring at least `min_neighbors` other strikes
    within ±`neighbor_window_s` (0 disables the check).
    """

    band: tuple[float, float] = (100.0, 300.0)
    threshold: float = Field(default=0.05, gt=0.0, le=1.0)
    min_interval_s: float = Field(default=1.0, gt=0.0)
    smooth_s: float = Field(default=0.02, ge=0.0)
    normalization_window_s: float = Field(default=60.0, ge=0.0)
    min_snr_db: float = 10.0
    min_neighbors: int = Field(default=2, ge=0)
    neighbor_window_s: float = Field(default=4.0, gt=0.0)
    chunk_s: float = Field(default=300.0, gt=0.0)
    filter_order: int = 4


class DetectorOverride(StrictModel):
    """Per-sensor overrides of the site-level detector settings."""

    band: tuple[float, float] | None = None
    threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    min_interval_s: float | None = Field(default=None, gt=0.0)
    smooth_s: float | None = Field(default=None, ge=0.0)
    normalization_window_s: float | None = Field(default=None, ge=0.0)
    min_snr_db: float | None = None
    min_neighbors: int | None = Field(default=None, ge=0)
    neighbor_window_s: float | None = Field(default=None, gt=0.0)


class ExtractConfig(StrictModel):
    """Windowing of each strike for metric computation."""

    pre_s: float = 0.6
    post_s: float = 0.6
    filter: FilterConfig | None = FilterConfig(type="highpass", freq=10.0)


class AmbientConfig(StrictModel):
    """Ambient noise in a window preceding each event."""

    offset_s: float = 660.0
    duration_s: float = 600.0
    interval_s: float = 60.0
    psd_segment_s: float = 5.0
    psd_overlap: float = Field(default=0.9, ge=0.0, lt=1.0)
    filter: FilterConfig | None = FilterConfig(type="highpass", freq=10.0)


class PulsePSDConfig(StrictModel):
    nperseg: int = 2**14
    overlap: float = Field(default=0.95, ge=0.0, lt=1.0)


class ReportConfig(StrictModel):
    """Figures and LaTeX tables.

    `met` is an optional CSV of weather observations with columns `datetime`,
    `WSPD` (wind speed, m/s) and `WVHT` (significant wave height, m), averaged
    over each event in the event tables.
    """

    style: Path | None = None
    dpi: int = 300
    met: Path | None = None
    metrics: list[str] = [
        "SPL_rms",
        "SPL_pk",
        "SPL_pkpk",
        "SEL",
        "SEL_cum",
        "t_rise",
        "t_decay",
        "kurtosis",
    ]
    histogram_metric: str = "SPL_pk"
    range_metrics: list[str] = ["SPL_pk", "SPL_pkpk", "SEL"]


# --------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------


class SensorConfig(StrictModel):
    name: str
    latitude: float
    longitude: float
    water_depth_m: float | None = None
    channels: list[int]
    detect_channel: int
    source: DataSourceConfig
    calibration: CalibrationConfig = NoCalibration()
    detector: DetectorOverride = DetectorOverride()

    @model_validator(mode="after")
    def _check_channels(self) -> "SensorConfig":
        if self.detect_channel not in self.channels:
            raise ValueError(
                f"{self.name}: detect_channel {self.detect_channel} is not in "
                f"channels {self.channels}."
            )
        if isinstance(self.calibration, FlatCalibration):
            n = max(self.channels) + 1
            cal = self.calibration
            if len(cal.sensitivity) < n or len(cal.fixed_gain) < n:
                raise ValueError(
                    f"{self.name}: flat calibration needs a sensitivity and gain "
                    f"for every channel up to {n - 1}."
                )
        return self


class Exclusion(StrictModel):
    """Skip an event entirely, or only certain sensors for that event."""

    event: str
    sensors: list[str] | None = None
    reason: str = ""


class SiteConfig(StrictModel):
    name: str
    output_dir: Path
    events: Path
    piles: Path
    sensors: list[SensorConfig]
    exclude: list[Exclusion] = Field(default_factory=list)
    detector: DetectorConfig = DetectorConfig()
    extract: ExtractConfig = ExtractConfig()
    ambient: AmbientConfig = AmbientConfig()
    pulse_psd: PulsePSDConfig = PulsePSDConfig()
    report: ReportConfig = ReportConfig()

    @model_validator(mode="after")
    def _unique_sensors(self) -> "SiteConfig":
        names = [s.name for s in self.sensors]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sensor names: {names}")
        return self

    def sensor(self, name: str) -> SensorConfig:
        for s in self.sensors:
            if s.name == name:
                return s
        raise KeyError(f"No sensor named '{name}' at site '{self.name}'.")

    def detector_for(self, sensor: SensorConfig) -> DetectorConfig:
        """Site-level detector settings with the sensor's overrides applied."""
        overrides = sensor.detector.model_dump(exclude_unset=True)
        return self.detector.model_copy(update=overrides)

    def is_excluded(self, event: str, sensor: str | None = None) -> bool:
        for ex in self.exclude:
            if ex.event != event:
                continue
            if ex.sensors is None or (sensor is not None and sensor in ex.sensors):
                return True
        return False

    # Output layout -------------------------------------------------------

    def inventory_path(self, sensor: str) -> Path:
        return self.output_dir / "inventory" / f"{sensor}.csv"

    def event_dir(self, event: str) -> Path:
        return self.output_dir / "events" / event

    def detections_path(self, event: str, sensor: str) -> Path:
        return self.event_dir(event) / f"detections_{sensor}.csv"

    def candidates_path(self, event: str, sensor: str) -> Path:
        return self.event_dir(event) / f"candidates_{sensor}.parquet"

    def pulses_path(self, event: str, sensor: str) -> Path:
        return self.event_dir(event) / f"pulses_{sensor}.h5"

    def metrics_path(self, event: str, sensor: str) -> Path:
        return self.event_dir(event) / f"metrics_{sensor}.csv"

    def pulse_psd_path(self, event: str, sensor: str) -> Path:
        return self.event_dir(event) / f"psd_{sensor}.h5"

    @property
    def report_dir(self) -> Path:
        return self.output_dir / "report"

    @property
    def metrics_summary_path(self) -> Path:
        return self.output_dir / "metrics.csv"

    @property
    def ambient_path(self) -> Path:
        return self.output_dir / "ambient.h5"

    @property
    def ambient_spl_path(self) -> Path:
        return self.output_dir / "ambient_spl.csv"


def load_site(path: Path) -> SiteConfig:
    """Load and validate a site configuration file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return SiteConfig(**_expand_paths(raw))


def to_datetime64(t: datetime | str) -> np.datetime64:
    return np.datetime64(t, "us")
