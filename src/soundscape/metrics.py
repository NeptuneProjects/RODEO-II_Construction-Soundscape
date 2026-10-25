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
    t_rise     time from the start of T90 to the largest p^2 within T90   s
    t_decay    time from that peak to the end of T90                       s

The peak for t_rise and t_decay is taken within T90 so that both are
non-negative; SPL_pk is the peak over the whole window.

A pulse is invalid on a hydrophone whose window holds a single-sample spike
(see `spike_count`), a recorder or cable fault rather than sound in the
water. Such pulses are dropped, not de-spiked, and do not count towards
SEL_cum on that hydrophone.

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
    n = np.arange(energy.shape[-1])
    in90 = (n >= i0[..., None]) & (n <= i1[..., None])
    i_pk = np.argmax(np.where(in90, energy, -1.0), axis=-1)

    return {
        "SPL_pk": _db_power(energy.max(axis=-1)),
        "SPL_pkpk": _db_power((x.max(axis=-1) - x.min(axis=-1)) ** 2),
        "SPL_rms90": _db_power(e90 / n90),
        "SEL_ss": _db_power(cum[..., -1] / fs),
        "T90": (i1 - i0) / fs,
        "t_rise": (i_pk - i0) / fs,
        "t_decay": (i1 - i_pk) / fs,
        # Start and end of T90 from the start of the window.
        "t5": i0 / fs,
        "t95": i1 / fs,
    }


def spike_count(x: np.ndarray, threshold: float) -> np.ndarray:
    """Number of single-sample spikes in each window of `x` (time last).

    A sample is a spike where its departure from the mean of its two
    neighbours exceeds `threshold` times the median of that departure over
    the window, and is at least half the largest |x| within two samples.
    Sound in the water is band-limited well below the sampling rate, so it
    changes little from one sample to the next compared with its amplitude
    (a tone of frequency f and amplitude A by at most A (1 - cos 2 pi f/fs));
    a jump as large as the signal itself comes from the recording chain.
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.abs(x[..., 1:-1] - 0.5 * (x[..., :-2] + x[..., 2:]))
    scale = np.median(r, axis=-1, keepdims=True)
    # A spike also displaces its neighbours' departures (by half), so count
    # only local maxima.
    pad = np.pad(r, [(0, 0)] * (r.ndim - 1) + [(1, 1)])
    local_max = (r >= pad[..., :-2]) & (r > pad[..., 2:])
    # |x| within two samples of each r (r[i] is at sample i + 1).
    a = np.pad(np.abs(x), [(0, 0)] * (x.ndim - 1) + [(1, 1)])
    near = np.max([a[..., k : k + r.shape[-1]] for k in range(5)], axis=0)
    jump = (r > threshold * scale) & (r >= 0.5 * near)
    count = (jump & local_max).sum(axis=-1)
    return np.where(scale[..., 0] > 0, count, 0)


def cumulative_sel(sel: np.ndarray, include: np.ndarray) -> np.ndarray:
    """Running cumulative SEL along axis 0, skipping excluded rows. `include`
    is per row, or per row and column."""
    include = include if include.ndim == sel.ndim else include[:, None]
    contributions = np.where(include, 10 ** (sel / 10), 0.0)
    return _db_power(np.cumsum(contributions, axis=0))


def channel_valid(valid: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    """Per strike and channel validity: the strike's window is clear of data
    gaps and the channel's window has no spikes."""
    return valid[:, None] & (spikes == 0)


def compute_metrics(pulses_path, spike_threshold: float | None = None) -> pl.DataFrame:
    """Metrics for every pulse and channel in a pulse file, in tidy form.

    With `spike_threshold`, pulses with spikes on a channel are invalid on
    that channel; the `spikes` column holds the count either way (0 without
    a threshold).
    """
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    fs = attrs["sampling_rate"]
    batches = []
    for w in iter_waveforms(pulses_path):
        b = pulse_metrics(w, fs)
        b["spikes"] = (
            spike_count(w, spike_threshold)
            if spike_threshold is not None
            else np.zeros(w.shape[:2], dtype=int)
        )
        batches.append(b)
    m = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    valid = channel_valid(arrays["valid"], m["spikes"])

    # Strikes are numbered in time order, so this is a running sum in time.
    if np.any(np.diff(arrays["time"]) < np.timedelta64(0, "us")):
        raise ValueError(f"Pulses in {pulses_path} are not in time order.")
    m["SEL_cum"] = cumulative_sel(m["SEL_ss"], valid)

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
            "valid": valid.reshape(-1),
            "spikes": m["spikes"].reshape(-1),
            # Start and end of T90 relative to the detection time, for the
            # inter-pulse intervals.
            "t5_s": (m["t5"] - attrs["pre_s"]).reshape(-1),
            "t95_s": (m["t95"] - attrs["pre_s"]).reshape(-1),
            **{k: m[k].reshape(-1) for k in METRIC_COLUMNS},
        }
    )
    return df.sort("channel", "time")
