"""Site configuration.

A site is described by a single TOML file (see `config/sites/`). The file
does not say where the raw data are on a given machine: data source
directories are relative to a data directory given at run time
(`soundscape inventory --data-dir ...`).

Other relative paths are resolved against the current working directory, so
the workflow is expected to be run from the repository root.
"""

import tomllib
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    A relative `directory` is taken relative to the data directory given at
    run time (`--data-dir`); an absolute one is used as is.
    """

    type: Literal["glob"] = "glob"
    directory: Path
    pattern: str
    file_format: str | None = None
    clock: ClockConfig | None = None


class InventorySourceConfig(StrictModel):
    """A prebuilt TritonOA inventory CSV, e.g. one built on the remote server.

    `path_map` rewrites filename prefixes so an inventory built on one machine
    can be read on another, e.g. `{"/data/server/southfork" =
    "/Volumes/drive/southfork"}`.
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


class KurtosisConfig(StrictModel):
    """Kurtosis over consecutive windows of each event (Martin et al. 2020)."""

    window_s: float = Field(default=60.0, gt=0.0)
    chunk_windows: int = Field(default=5, ge=1)
    filter: FilterConfig | None = FilterConfig(type="highpass", freq=10.0)


class BandsConfig(StrictModel):
    """Decidecade bands for band levels: those with lower edge at least
    `fmin` and upper edge at most `fmax`. By default `fmax` is
    `nyquist_fraction` of the Nyquist frequency, which keeps bands clear of the
    recorders' anti-aliasing filter roll-off just below Nyquist."""

    fmin: float = Field(default=10.0, gt=0.0)
    fmax: float | None = Field(default=None, gt=0.0)
    nyquist_fraction: float = Field(default=0.8, gt=0.0, le=1.0)

    def upper_limit(self, fs: float) -> float:
        return self.fmax or self.nyquist_fraction * fs / 2


class PulsePSDConfig(StrictModel):
    nperseg: int = 2**14
    overlap: float = Field(default=0.95, ge=0.0, lt=1.0)


class MetStationConfig(StrictModel):
    """A weather or wave station whose observations are in `[report] met`.

    `variables` names the `met` columns it supplies (e.g. ["WSPD"]). The
    position is used to give the distance from the station to the sensors.
    """

    name: str
    description: str | None = None
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    variables: list[str]
    source: str | None = None  # e.g. the NDBC station page


class ReportConfig(StrictModel):
    """Figures and LaTeX tables.

    `met` is an optional CSV of weather observations with columns `datetime`,
    `WSPD` (wind speed, m/s) and `WVHT` (significant wave height, m), averaged
    over each event in the event tables. `met_stations` records where they
    were measured.

    `titles = false` leaves titles off the figures, for documents whose
    captions say the same. `docx_template` is a .docx whose styles the Word
    document uses (e.g. the shared report, downloaded as .docx).
    """

    style: Path | None = None
    titles: bool = True
    docx_template: Path | None = None
    dpi: int = 300
    met: Path | None = None
    met_stations: list[MetStationConfig] = Field(default_factory=list)
    # Per-pulse metrics in the event figures and tables (SEL_cum is given as
    # an event total in the tables).
    metrics: list[str] = [
        "SPL_pk",
        "SPL_pkpk",
        "SPL_rms90",
        "SEL_ss",
        "SEL_cum",
        "T90",
        "t_rise",
        "t_decay",
    ]
    histogram_metric: str = "SPL_pk"
    range_metrics: list[str] = ["SPL_pk", "SPL_pkpk", "SEL_ss"]


# --------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------


class HydrophoneConfig(StrictModel):
    """Description of the hydrophone on one recorder channel.

    The serial number identifies the physical hydrophone across deployments;
    sensor names such as "VLA1" are labels for one site only.

    `depth_m` is the hydrophone's depth below the sea surface. Moorings are
    adjusted at each deployment to keep sensor depths consistent, whereas the
    water depth (`SensorConfig.water_depth_m`) changes with the site.

    Descriptive only: calibration is set in `[sensors.calibration]`. Channels
    that are not analyzed may also be listed, to document the instrument.
    """

    channel: int
    depth_m: float | None = None
    serial_number: str | None = None
    description: str | None = None


class SensorConfig(StrictModel):
    name: str
    description: str | None = None
    latitude: float
    longitude: float
    water_depth_m: float | None = None
    channels: list[int]
    detect_channel: int
    source: DataSourceConfig
    calibration: CalibrationConfig = NoCalibration()
    detector: DetectorOverride = DetectorOverride()
    hydrophones: list[HydrophoneConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_channels(self) -> "SensorConfig":
        if self.detect_channel not in self.channels:
            raise ValueError(
                f"{self.name}: detect_channel {self.detect_channel} is not in "
                f"channels {self.channels}."
            )
        listed = [h.channel for h in self.hydrophones]
        if len(listed) != len(set(listed)):
            raise ValueError(f"{self.name}: hydrophone channels repeat: {listed}")
        if isinstance(self.calibration, FlatCalibration):
            n = max(self.channels) + 1
            cal = self.calibration
            if len(cal.sensitivity) < n or len(cal.fixed_gain) < n:
                raise ValueError(
                    f"{self.name}: flat calibration needs a sensitivity and gain "
                    f"for every channel up to {n - 1}."
                )
        return self

    def hydrophone(self, channel: int) -> HydrophoneConfig:
        """The hydrophone on a channel; empty if it is not described."""
        for h in self.hydrophones:
            if h.channel == channel:
                return h
        return HydrophoneConfig(channel=channel)


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
    kurtosis: KurtosisConfig = KurtosisConfig()
    bands: BandsConfig = BandsConfig()
    report: ReportConfig = ReportConfig()

    @model_validator(mode="after")
    def _unique_sensors(self) -> "SiteConfig":
        names = [s.name for s in self.sensors]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate sensor names: {names}")
        # A serial number identifies one physical hydrophone.
        seen: dict[str, str] = {}
        for sensor in self.sensors:
            for h in sensor.hydrophones:
                if h.serial_number is None:
                    continue
                where = f"{sensor.name} channel {h.channel}"
                if h.serial_number in seen:
                    raise ValueError(
                        f"Hydrophone {h.serial_number} is listed on both "
                        f"{seen[h.serial_number]} and {where}."
                    )
                seen[h.serial_number] = where
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

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

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

    @property
    def kurtosis_path(self) -> Path:
        return self.output_dir / "kurtosis.csv"


def load_site(path: Path) -> SiteConfig:
    """Load and validate a site configuration file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return SiteConfig(**raw)


def to_datetime64(t: datetime | str) -> np.datetime64:
    return np.datetime64(t, "us")
