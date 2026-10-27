"""Per-pulse metrics of pile-driving strikes.

Metrics follow ISO 18406:2017, with quantities, levels, and symbols as
defined in ISO 18405:2017; p0 = 1 uPa and E_p,0 = 1 uPa^2 s. Column names
transliterate the symbols:

    column   symbol        quantity                                     unit
    Lp_pk    L_p,pk        peak sound pressure level, 20 log10(p_pk/p0) dB re 1 uPa
    Lp_pkpk  L_p,pk-pk     peak-to-peak sound pressure level            dB re 1 uPa
    Lp_90    L_p,90        mean-square sound pressure level over tau90  dB re 1 uPa
    LEp_ss   L_E,p,ss      single-strike sound exposure level (SEL_ss)  dB re 1 uPa^2 s
    LEp_cum  L_E,p,cum     cumulative sound exposure level (SEL_cum),
                           running over valid strikes in time order     dB re 1 uPa^2 s
    tau90    tau_90%       pulse duration: 90 % energy signal duration  s
    t_r      t_r           start of tau90 to the largest p^2 within it  s
    t_d      t_d           that peak to the end of tau90                s
    t5, t95  t_5, t_95     start and end of tau90 from the detection
                           time (5 % and 95 % of the sound exposure)    s
    TW       T_W           integration time of the pulse                s
    clicks                 clicks from the recording chain in the window

The peak for t_r and t_d is taken within tau90 so that both are
non-negative; Lp_pk is the peak over the whole integration time.

Every metric is taken over the pulse's integration time T_W: the whole
window, or, where strikes are closer together than the window, the part of
it closer to the strike's own detection time than to its neighbors' (ISO
18406:2017, 6.4.2.1.2; see `extract.integration_bounds`). Lp_pk is thus the
peak over the pulse, from t0 to t100.

A pulse is invalid on a hydrophone whose window holds a click (see
`click_count`), a recorder or cable fault rather than sound in the water.
Such pulses are dropped, not repaired, and do not count toward LEp_cum on
that hydrophone.

The window (see ExtractConfig) is long enough to hold the whole pulse, so
LEp_ss includes the background noise in it, which biases it upward at low
signal-to-noise ratio. Kurtosis is computed over 60 s windows of the
continuous recording instead of per pulse (see soundscape.kurtosis).
"""

import numpy as np
import polars as pl

from soundscape.extract import integration_bounds, iter_waveforms, read_pulses, trim

REF_PRESSURE_UPA = 1.0

METRIC_COLUMNS = [
    "Lp_pk",
    "Lp_pkpk",
    "Lp_90",
    "LEp_ss",
    "LEp_cum",
    "tau90",
    "t_r",
    "t_d",
]


def read_table(path) -> pl.DataFrame:
    """Read a metrics or background noise table written by the workflow."""
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
    """Metrics for pulses `x` with time on the last axis (all but LEp_cum)."""
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
        "Lp_pk": _db_power(energy.max(axis=-1)),
        "Lp_pkpk": _db_power((x.max(axis=-1) - x.min(axis=-1)) ** 2),
        "Lp_90": _db_power(e90 / n90),
        "LEp_ss": _db_power(cum[..., -1] / fs),
        "tau90": (i1 - i0) / fs,
        "t_r": (i_pk - i0) / fs,
        "t_d": (i1 - i_pk) / fs,
        # Start and end of tau90 from the start of the window.
        "t5": i0 / fs,
        "t95": i1 / fs,
    }


def click_count(x: np.ndarray, threshold: float) -> np.ndarray:
    """Number of clicks (jumps between consecutive samples) in each window of
    `x` (time last).

    A sample is a click where its departure from the mean of its two
    neighbors exceeds `threshold` times the median of that departure over
    the window, and is at least half the largest |x| within two samples.
    Sound in the water is band-limited well below the sampling rate, so it
    changes little from one sample to the next compared with its amplitude
    (a tone of frequency f and amplitude A by at most A (1 - cos 2 pi f/fs));
    a jump as large as the signal itself comes from the recording chain.
    """
    x = np.asarray(x, dtype=np.float64)
    r = np.abs(x[..., 1:-1] - 0.5 * (x[..., :-2] + x[..., 2:]))
    scale = np.median(r, axis=-1, keepdims=True)
    # A click also displaces its neighbors' departures (by half), so count
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


def channel_valid(valid: np.ndarray, clicks: np.ndarray) -> np.ndarray:
    """Per strike and channel validity: the strike's window is clear of data
    gaps and the channel's window has no clicks."""
    return valid[:, None] & (clicks == 0)


def compute_metrics(pulses_path, click_threshold: float | None = None) -> pl.DataFrame:
    """Metrics for every pulse and channel in a pulse file, in tidy form.

    With `click_threshold`, pulses with clicks on a channel are invalid on
    that channel; the `clicks` column holds the count either way (0 without
    a threshold).
    """
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    fs = attrs["sampling_rate"]
    lo, hi = integration_bounds(pulses_path)
    batches, i = [], 0
    for w in iter_waveforms(pulses_path):
        k = slice(i, i + len(w))
        i += len(w)
        b = pulse_metrics(trim(w, lo[k], hi[k]), fs)
        # Clicks are sought in the whole window: the cut edges would look
        # like jumps.
        b["clicks"] = (
            click_count(w, click_threshold)
            if click_threshold is not None
            else np.zeros(w.shape[:2], dtype=int)
        )
        batches.append(b)
    m = {k: np.concatenate([b[k] for b in batches]) for k in batches[0]}
    m["TW"] = np.repeat(((hi - lo) / fs)[:, None], m["LEp_ss"].shape[1], axis=1)
    valid = channel_valid(arrays["valid"], m["clicks"])

    # Strikes are numbered in time order, so this is a running sum in time.
    if np.any(np.diff(arrays["time"]) < np.timedelta64(0, "us")):
        raise ValueError(f"Pulses in {pulses_path} are not in time order.")
    m["LEp_cum"] = cumulative_sel(m["LEp_ss"], valid)

    n_strikes, n_channels = m["LEp_ss"].shape
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
            "clicks": m["clicks"].reshape(-1),
            # Start and end of tau90 relative to the detection time, for the
            # inter-pulse intervals.
            "t5": (m["t5"] - attrs["pre_s"]).reshape(-1),
            "t95": (m["t95"] - attrs["pre_s"]).reshape(-1),
            "TW": m["TW"].reshape(-1),
            **{k: m[k].reshape(-1) for k in METRIC_COLUMNS},
        }
    )
    return df.sort("channel", "time")
