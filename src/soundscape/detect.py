"""Automatic detection of pile-driving strikes by band-limited mean-square
sound pressure.

Generalizes the Vineyard Wind detector. Detection runs in two steps:

1. `scan`: read the detection channel in chunks, band-pass it, average its
   square over a moving window (the mean-square sound pressure e), and keep
   every local maximum at least `min_interval_s` apart as a candidate,
   together with the chunk's median e (`e_bg`). This is the only step
   that reads acoustic data.
2. `select`: classify candidates as strikes by their e relative to the
   strongest candidate nearby (`e_rel` against `threshold`) and to `e_bg`
   (`SNR_det` against `min_snr_db`). Because it only needs the candidates, thresholds can be
   re-tuned without re-reading data (`soundscape detect --reselect`).

All channels of a sensor are later windowed at the same strike times, so
detections are synchronized across channels by construction.
"""

import bisect
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from tritonoa.data.reader import NoDataError

from soundscape.config import DetectorConfig, SensorConfig
from soundscape.events import Event
from soundscape.sources import read_segment

logger = logging.getLogger(__name__)

# Extra data read on each side of a chunk so filter transients and peaks that
# straddle chunk boundaries are handled; added to `min_interval_s`.
_CHUNK_PAD_S = 2.0

CANDIDATE_SCHEMA = {
    "time": pl.Datetime("us"),
    "e": pl.Float64,
    "e_bg": pl.Float64,
}


def band_mean_square(
    x: np.ndarray, fs: float, band: tuple[float, float], order: int, smooth_s: float
) -> np.ndarray:
    """Zero-phase band-pass, square, and average over a centered moving window:
    the band-limited mean-square sound pressure e(t)."""
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    e = sosfiltfilt(sos, x) ** 2
    n = round(smooth_s * fs)
    if n > 1:
        e = uniform_filter1d(e, size=n, mode="nearest")
    return e


def noise_floor(e: np.ndarray, recorded: np.ndarray) -> float:
    """Median of e over recorded (non-zero-filled) samples."""
    values = e[recorded]
    return float(np.median(values)) if values.size else np.nan


def find_candidates(e: np.ndarray, fs: float, min_interval_s: float) -> np.ndarray:
    distance = max(1, int(min_interval_s * fs))
    return find_peaks(e, distance=distance)[0]


def scan(
    sensor: SensorConfig,
    inventory: Path,
    event: Event,
    config: DetectorConfig,
) -> pl.DataFrame:
    """Scan an event for strike candidates on the sensor's detection channel."""
    chunk = np.timedelta64(int(config.chunk_s * 1e6), "us")
    pad = np.timedelta64(int((config.min_interval_s + _CHUNK_PAD_S) * 1e6), "us")

    frames = []
    core_start = event.start
    while core_start < event.end:
        core_end = min(core_start + chunk, event.end)
        try:
            seg = read_segment(
                sensor,
                inventory,
                max(core_start - pad, event.start),
                min(core_end + pad, event.end),
                channels=[sensor.detect_channel],
            )
        except NoDataError:
            logger.warning(f"{sensor.name}: no data {core_start} - {core_end}")
            core_start = core_end
            continue

        x = seg.data[0]
        recorded = x != 0.0
        e = band_mean_square(
            x, seg.sampling_rate, config.band, config.filter_order, config.smooth_s
        )
        peaks = find_candidates(e, seg.sampling_rate, config.min_interval_s)
        times = seg.times()

        in_core = (times >= core_start) & (times < core_end)
        peaks = peaks[in_core[peaks] & recorded[peaks]]
        frames.append(
            pl.DataFrame(
                {
                    "time": times[peaks],
                    "e": e[peaks],
                    "e_bg": noise_floor(e[in_core], recorded[in_core]),
                },
                schema=CANDIDATE_SCHEMA,
            )
        )
        core_start = core_end

    if not frames:
        return pl.DataFrame(schema=CANDIDATE_SCHEMA)
    return pl.concat(frames).sort("time")


def suppress_close(times_us: np.ndarray, e: np.ndarray, min_us: int) -> np.ndarray:
    """Greedy non-maximum suppression: indices of peaks kept, strongest first."""
    kept_times: list[int] = []
    kept = []
    for i in np.argsort(-e, kind="stable"):
        t = int(times_us[i])
        j = bisect.bisect_left(kept_times, t)
        if j > 0 and t - kept_times[j - 1] < min_us:
            continue
        if j < len(kept_times) and kept_times[j] - t < min_us:
            continue
        kept_times.insert(j, t)
        kept.append(i)
    return np.sort(np.array(kept, dtype=int))


def reference_level(times_us: np.ndarray, e: np.ndarray, window_s: float) -> np.ndarray:
    """Largest candidate e within ±window/2 of each candidate (or overall)."""
    if window_s == 0:
        return np.full_like(e, e.max())
    half = int(window_s * 1e6 / 2)
    left = np.searchsorted(times_us, times_us - half, side="left")
    right = np.searchsorted(times_us, times_us + half, side="right")
    return np.array([e[lo:hi].max() for lo, hi in zip(left, right)])


def count_neighbors(times_us: np.ndarray, window_s: float) -> np.ndarray:
    """Number of other events within ±window of each event (times sorted)."""
    half = int(window_s * 1e6)
    left = np.searchsorted(times_us, times_us - half, side="left")
    right = np.searchsorted(times_us, times_us + half, side="right")
    return right - left - 1


def select(candidates: pl.DataFrame, config: DetectorConfig) -> pl.DataFrame:
    """Score candidates and flag strikes.

    Adds columns `e_rel` (e relative to the reference), `SNR_det`
    (10 log10(e / e_bg), dB), `neighbors` (other strikes passing both criteria within
    ±`neighbor_window_s`), and `is_strike`.
    """
    if candidates.is_empty():
        return candidates.with_columns(
            pl.lit(None, pl.Float64).alias("e_rel"),
            pl.lit(None, pl.Float64).alias("SNR_det"),
            pl.lit(None, pl.Int64).alias("neighbors"),
            pl.lit(None, pl.Boolean).alias("is_strike"),
        )
    candidates = candidates.sort("time")
    times_us = candidates["time"].dt.epoch("us").to_numpy()
    e = candidates["e"].to_numpy()

    keep = suppress_close(times_us, e, int(config.min_interval_s * 1e6))
    candidates = candidates[keep]
    times_us, e = times_us[keep], e[keep]

    cf = e / reference_level(times_us, e, config.normalization_window_s)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db = 10 * np.log10(e / candidates["e_bg"].to_numpy())
    is_strike = (cf >= config.threshold) & (np.nan_to_num(snr_db) >= config.min_snr_db)

    neighbors = np.zeros(len(is_strike), dtype=np.int64)
    neighbors[is_strike] = count_neighbors(
        times_us[is_strike], config.neighbor_window_s
    )
    is_strike &= neighbors >= config.min_neighbors
    return candidates.with_columns(
        pl.Series("e_rel", cf),
        pl.Series("SNR_det", snr_db),
        pl.Series("neighbors", neighbors),
        pl.Series("is_strike", is_strike),
    )


def strikes_table(
    scored: pl.DataFrame, event: Event, sensor: SensorConfig
) -> pl.DataFrame:
    """Detected strikes in the tidy format consumed by `extract`."""
    return (
        scored.filter(pl.col("is_strike"))
        .with_row_index("strike")
        .select(
            pl.lit(event.name).alias("event"),
            pl.lit(sensor.name).alias("sensor"),
            pl.col("strike").cast(pl.Int64),
            "time",
            "e",
            "e_bg",
            "SNR_det",
            "e_rel",
        )
    )
