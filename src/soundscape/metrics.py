"""Per-pulse soundscape metrics.

Definitions follow ISO 18405:2017 with a reference pressure of 1 uPa. For a
pulse window p(t) sampled at fs, with T90 the interval holding 5-95% of the
cumulative energy:

    SPL_pk     20 log10(max|p|)                         dB re 1 uPa
    SPL_pkpk   20 log10(max p - min p)                  dB re 1 uPa
    SPL_rms    10 log10(mean p^2 over the whole window) dB re 1 uPa
    SPL_rms90  10 log10(mean p^2 over T90)              dB re 1 uPa
    SEL        10 log10(sum p^2 / fs)                   dB re 1 uPa^2 s
    SEL_cum    10 log10(sum_i 10^(SEL_i/10)), running over strikes in time order
    T90        duration of the 90% energy interval      s
    t_rise     time from the start of T90 to the peak   s
    t_decay    time from the peak to the end of T90     s
    kurtosis   Pearson kurtosis (beta_2; 3 for Gaussian noise)

The window is broadband apart from the extraction filter and includes ambient
noise, which biases SPL_rms and SEL upward at low signal-to-noise ratio.
"""

import numpy as np
import polars as pl
from scipy.stats import kurtosis as _kurtosis

from soundscape.extract import iter_waveforms, read_pulses

REF_PRESSURE_UPA = 1.0

METRIC_COLUMNS = [
    "SPL_pk",
    "SPL_pkpk",
    "SPL_rms",
    "SPL_rms90",
    "SEL",
    "SEL_cum",
    "T90",
    "t_rise",
    "t_decay",
    "kurtosis",
]


def _db_power(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10 * np.log10(x / REF_PRESSURE_UPA**2)


def _db_amplitude(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 20 * np.log10(x / REF_PRESSURE_UPA)


def energy_interval(energy: np.ndarray, lower: float = 0.05, upper: float = 0.95):
    """Indices bounding the [lower, upper] fraction of cumulative energy.

    Returns `(i0, i1)` such that `energy[..., i0:i1]` is the interval, and the
    zero-padded cumulative sum for computing partial sums.
    """
    cum = np.concatenate(
        [np.zeros((*energy.shape[:-1], 1)), np.cumsum(energy, axis=-1)], axis=-1
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = cum[..., 1:] / cum[..., -1:]
    i0 = np.argmax(frac >= lower, axis=-1)
    i1 = np.argmax(frac >= upper, axis=-1)
    return i0, i1, cum


def pulse_metrics(x: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Metrics for pulses `x` with time on the last axis (all but SEL_cum)."""
    x = np.asarray(x, dtype=np.float64)
    energy = x**2
    i0, i1, cum = energy_interval(energy)
    n90 = np.maximum(i1 - i0, 1)
    e90 = (
        np.take_along_axis(cum, i1[..., None], -1)[..., 0]
        - np.take_along_axis(cum, i0[..., None], -1)[..., 0]
    )
    i_pk = np.argmax(energy, axis=-1)

    return {
        "SPL_pk": _db_amplitude(np.abs(x).max(axis=-1)),
        "SPL_pkpk": _db_amplitude(x.max(axis=-1) - x.min(axis=-1)),
        "SPL_rms": _db_power(energy.mean(axis=-1)),
        "SPL_rms90": _db_power(e90 / n90),
        "SEL": _db_power(cum[..., -1] / fs),
        "T90": (i1 - i0) / fs,
        "t_rise": (i_pk - i0) / fs,
        "t_decay": (i1 - i_pk) / fs,
        "kurtosis": _kurtosis(x, axis=-1, fisher=False),
    }


def cumulative_sel(sel: np.ndarray, include: np.ndarray) -> np.ndarray:
    """Running cumulative SEL along axis 0, skipping excluded rows."""
    contributions = np.where(include[:, None], 10 ** (sel / 10), 0.0)
    return _db_power(np.cumsum(contributions, axis=0))


def compute_metrics(pulses_path) -> pl.DataFrame:
    """Metrics for every pulse and channel in a pulse file, in tidy form."""
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    fs = attrs["sampling_rate"]
    batches = [pulse_metrics(w, fs) for w in iter_waveforms(pulses_path)]
    m = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}

    # Strikes are numbered in time order, so this is a running sum in time.
    if np.any(np.diff(arrays["time"]) < np.timedelta64(0, "us")):
        raise ValueError(f"Pulses in {pulses_path} are not in time order.")
    m["SEL_cum"] = cumulative_sel(m["SEL"], arrays["valid"])

    n_strikes, n_channels = m["SEL"].shape
    channels = np.array(attrs["channels"])
    df = pl.DataFrame(
        {
            "event": attrs["event"],
            "sensor": attrs["sensor"],
            "pile": attrs["pile"],
            "range_m": attrs["range_m"],
            "channel": np.tile(channels, n_strikes),
            "strike": np.repeat(arrays["strike"], n_channels),
            "time": np.repeat(arrays["time"], n_channels),
            "valid": np.repeat(arrays["valid"], n_channels),
            **{k: m[k].reshape(-1) for k in METRIC_COLUMNS},
        }
    )
    return df.sort("channel", "time")
