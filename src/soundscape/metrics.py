"""Per-pulse metrics of pile-driving strikes.

Metrics follow ISO 18406:2017, with quantities and levels as defined in
ISO 18405:2017 and a reference pressure p0 = 1 uPa. For a pulse window p(t)
sampled at fs, the pulse duration T90 is the 90 % energy signal duration, the
interval from 5 % to 95 % of the cumulative energy:

    SPL_pk     10 log10(max p^2 / p0^2)                  dB re 1 uPa
    SPL_pkpk   10 log10((max p - min p)^2 / p0^2)        dB re 1 uPa
    SPL_rms90  10 log10(mean p^2 over T90 / p0^2)        dB re 1 uPa
    SEL_ss     10 log10(sum p^2 / fs / E0), E0 = 1 uPa^2 s, over the whole
               window, so 100 % of the pulse energy      dB re 1 uPa^2 s
    SEL_cum    10 log10(sum_i 10^(SEL_ss,i / 10)), running over valid strikes
               in time order                             dB re 1 uPa^2 s
    T90        pulse duration                            s
    t_rise     time from the start of T90 to the peak    s
    t_decay    time from the peak to the end of T90      s

The window (see ExtractConfig) is long enough to hold the whole pulse, so
SEL_ss includes the ambient noise in it, which biases it upward at low
signal-to-noise ratio. Kurtosis is computed over 60 s windows of the
continuous recording instead of per pulse (see soundscape.kurtosis).
"""

import numpy as np
import polars as pl

from soundscape.extract import iter_waveforms, read_pulses

REF_PRESSURE_UPA = 1.0

METRIC_COLUMNS = [
    "SPL_pk",
    "SPL_pkpk",
    "SPL_rms90",
    "SEL_ss",
    "SEL_cum",
    "T90",
    "t_rise",
    "t_decay",
]


def read_table(path) -> pl.DataFrame:
    """Read a metrics or ambient table written by the workflow."""
    return pl.read_csv(
        path, try_parse_dates=True, schema_overrides={"serial_number": pl.String}
    )


def _db_power(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 10 * np.log10(x / REF_PRESSURE_UPA**2)


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
        "SPL_pk": _db_power(energy.max(axis=-1)),
        "SPL_pkpk": _db_power((x.max(axis=-1) - x.min(axis=-1)) ** 2),
        "SPL_rms90": _db_power(e90 / n90),
        "SEL_ss": _db_power(cum[..., -1] / fs),
        "T90": (i1 - i0) / fs,
        "t_rise": (i_pk - i0) / fs,
        "t_decay": (i1 - i_pk) / fs,
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
    m["SEL_cum"] = cumulative_sel(m["SEL_ss"], arrays["valid"])

    n_strikes, n_channels = m["SEL_ss"].shape
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
