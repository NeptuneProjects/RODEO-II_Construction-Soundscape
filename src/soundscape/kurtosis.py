"""Kurtosis of the continuous recording in fixed windows over each event.

Following Martin et al. (2020), kurtosis is the Pearson kurtosis

    beta_2 = mean((p - mean p)^4) / mean((p - mean p)^2)^2

of the pressure over consecutive, non-overlapping windows (60 s by default),
rather than over a single pulse. Gaussian noise has beta_2 = 3; impulsive sound
such as pile driving has beta_2 well above 3. The same statistic over the
background noise intervals before each event is in the background noise table.

Martin, S. B., Lucke, K., & Barclay, D. R. (2020). Techniques for
distinguishing between impulsive and non-impulsive sound in the context of
regulating sound exposure for marine mammals. JASA 147(4), 2159-2176.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from tritonoa.data.reader import NoDataError

from soundscape.config import KurtosisConfig, SensorConfig
from soundscape.events import Event, source_range
from soundscape.extract import apply_filter
from soundscape.sources import read_segment

logger = logging.getLogger(__name__)

# Data read on each side of a chunk so filter transients fall outside it.
_PAD_S = 1.0

# A window with more than this fraction of zero-filled samples is a gap.
_GAP_FRACTION = 0.01


def beta2(blocks: np.ndarray) -> np.ndarray:
    """Pearson kurtosis along the last axis; NaN where the variance is zero
    (e.g. a zero-filled gap)."""
    x = blocks - blocks.mean(axis=-1, keepdims=True)
    x2 = x * x
    m2 = x2.mean(axis=-1)
    m4 = (x2 * x2).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m2 > 0, m4 / (m2 * m2), np.nan)


def windowed_kurtosis(
    sensor: SensorConfig,
    inventory: Path,
    event: Event,
    config: KurtosisConfig,
    strikes: np.ndarray | None = None,
) -> pl.DataFrame:
    """Kurtosis in consecutive windows from the start to the end of an event,
    on every channel. `strikes` (detection times) are counted in each window.

    Returns a tidy table: event, sensor, range_m, channel, window_start,
    valid (no gap), n_strikes, kurtosis.
    """
    window = np.timedelta64(round(config.window_s * 1e6), "us")
    n_windows = int((event.end - event.start) // window)
    pad = np.timedelta64(round(_PAD_S * 1e6), "us")
    frames = []
    for k0 in range(0, n_windows, config.chunk_windows):
        k1 = min(k0 + config.chunk_windows, n_windows)
        t0, t1 = event.start + k0 * window, event.start + k1 * window
        try:
            seg = read_segment(sensor, inventory, t0 - pad, t1 + pad)
        except NoDataError:
            logger.warning(f"[kurtosis] {sensor.name}: no data {t0} - {t1}")
            continue
        fs = seg.sampling_rate
        n_win = round(config.window_s * fs)
        recorded = seg.data != 0.0
        data = apply_filter(seg.data, fs, config.filter)
        i0 = round((t0 - seg.time_init) / np.timedelta64(1, "us") * 1e-6 * fs)
        n = k1 - k0
        if i0 < 0 or i0 + n * n_win > data.shape[1]:
            # The chunk starts or ends in a gap the reader did not fill.
            logger.warning(f"[kurtosis] {sensor.name}: incomplete data {t0} - {t1}")
            continue
        blocks = data[:, i0 : i0 + n * n_win].reshape(data.shape[0], n, n_win)
        rec = recorded[:, i0 : i0 + n * n_win].reshape(data.shape[0], n, n_win)
        valid = 1.0 - rec.mean(axis=-1) <= _GAP_FRACTION
        starts = t0 + np.arange(n) * window
        if strikes is not None:
            counts = np.searchsorted(strikes, starts + window) - np.searchsorted(
                strikes, starts
            )
        else:
            counts = np.zeros(n, dtype=int)
        n_ch = len(seg.channels)
        frames.append(
            pl.DataFrame(
                {
                    "channel": np.repeat(seg.channels, n),
                    "window_start": np.tile(starts, n_ch),
                    "valid": valid.reshape(-1),
                    "n_strikes": np.tile(counts, n_ch),
                    # Windows in gaps get no value.
                    "beta": np.where(valid, beta2(blocks), np.nan).reshape(-1),
                }
            )
        )
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames)
        .with_columns(
            pl.lit(event.name).alias("event"),
            pl.lit(sensor.name).alias("sensor"),
            pl.lit(source_range(event, sensor)).alias("range_m"),
        )
        .select(
            "event",
            "sensor",
            "range_m",
            "channel",
            "window_start",
            "valid",
            "n_strikes",
            "beta",
        )
        .sort("channel", "window_start")
    )
