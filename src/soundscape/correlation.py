"""Correlation between pile-driving pulses.

For pulses p_i(t) and p_j(t) on one hydrophone, each in reduced time t about
its strike, the cross-correlation function and its normalized form are

    R_ij(tau) = integral p_i(t) p_j(t + tau) dt
    rho_ij(tau) = R_ij(tau) / sqrt(R_ii(0) R_jj(0))

and the maximum of rho_ij over tau is the maximum normalized
cross-correlation coefficient rho_ij of the pair (Jenkins & Lin, Sci. Rep.,
Eqs. 5-6). It is 1 for identical waveforms at any delay within the window and
falls as the waveforms differ. Computed for every pair of valid pulses with
rodeo's CrossCorr.jl, after each window is detrended, tapered, band-passed and
downsampled (see CorrelationConfig).

Correlations for one event and sensor are stored in HDF5:

    /corr     float32 (n_channels, n_strikes, n_strikes)  rho_ij; NaN where
              either pulse is invalid on the hydrophone
    /lag      float32 (n_channels, n_strikes, n_strikes)  lag of the maximum
              (s); positive when pulse i arrives later in its window than
              pulse j
    /strike   int64   (n_strikes,), strike number from the detections table
    /time     int64   (n_strikes,), detection time [us since epoch]
    attrs: event, sensor, pile, range_m, channels, depth_m, serial_number,
           band, taper_pc, sampling_rate (after downsampling)

The time between strikes i and j is time[j] - time[i].
"""

import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.signal.windows import tukey

from soundscape.config import CorrelationConfig
from soundscape.events import too_few_strikes
from soundscape.extract import iter_waveforms, read_pulses

logger = logging.getLogger(__name__)

# Width of the bins of rho in the histograms behind the probability densities.
RHO_BIN = 0.02

_JULIA = None


def condition(
    w: np.ndarray, fs: float, config: CorrelationConfig
) -> tuple[np.ndarray, float]:
    """Detrend, taper, band-pass and downsample pulse windows (last axis).

    The band-pass filter is the anti-aliasing filter: the band's upper edge
    lies below half of `target_fs` (checked by CorrelationConfig).
    """
    q = max(1, int(fs // config.target_fs))
    y = detrend(np.asarray(w, dtype=np.float64), axis=-1)
    y *= tukey(y.shape[-1], 2 * config.taper_pc)
    sos = butter(4, config.band, btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, y, axis=-1)
    return y[..., ::q], fs / q


def channel_valid(
    metrics: pl.DataFrame, strikes: np.ndarray, channels: list[int]
) -> np.ndarray:
    """(n_channels, n_strikes) validity of each pulse on each hydrophone, from
    the metrics table (strike, channel, valid); strikes missing from it are
    invalid."""
    wide = (
        pl.DataFrame({"strike": strikes})
        .join(
            metrics.pivot(on="channel", index="strike", values="valid"),
            on="strike",
            how="left",
            maintain_order="left",
        )
        .fill_null(False)
    )
    return np.stack(
        [
            wide[str(c)].to_numpy()
            if str(c) in wide.columns
            else np.zeros(len(strikes), bool)
            for c in channels
        ]
    )


def _julia(threads: int):
    """The CrossCorr Julia module, starting Julia on `threads` threads if it
    is not yet running (the thread count is fixed once Julia starts)."""
    global _JULIA
    if _JULIA is None:
        if "juliacall" not in sys.modules:
            os.environ["PYTHON_JULIACALL_THREADS"] = str(threads)
            # Needed for Julia to run multithreaded inside Python.
            os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
        from rodeo.utils import initialize_julia

        jl = initialize_julia("CrossCorr")
        n = jl.seval("Threads.nthreads()")
        if n != threads:
            logger.warning(f"[correlation] Julia is running on {n} threads")
        _JULIA = jl
    return _JULIA


def corr_matrix(x: np.ndarray, threads: int) -> tuple[np.ndarray, np.ndarray]:
    """Maximum normalized cross-correlation coefficients (n, n) of the rows
    of `x`, and the lags of the maxima in samples (positive when row i is
    later than row j)."""
    jl = _julia(threads)
    corr, shifts = jl.CrossCorr.corr_matrix(jl.seval("x -> Matrix{Float64}(x)")(x))
    return np.asarray(corr), np.asarray(shifts)


def correlate(
    pulses_path: Path, metrics: pl.DataFrame, config: CorrelationConfig
) -> tuple[dict[str, np.ndarray], dict]:
    """Correlations between every pair of valid pulses on each hydrophone.

    Returns the datasets and attributes of the HDF5 layout in the module
    docstring.
    """
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    channels = attrs["channels"]
    valid = channel_valid(metrics, arrays["strike"], channels)
    fs = attrs["sampling_rate"]
    batches = []
    for w in iter_waveforms(pulses_path):
        y, fs_out = condition(w, fs, config)
        batches.append(y.astype(np.float32))
    x = np.concatenate(batches)  # (n_strikes, n_channels, n_samples)

    n = x.shape[0]
    corr = np.full((len(channels), n, n), np.nan, np.float32)
    lag = np.full((len(channels), n, n), np.nan, np.float32)
    for c in range(len(channels)):
        k = np.flatnonzero(valid[c])
        if len(k) < 2:
            continue
        r, s = corr_matrix(x[k, c].astype(np.float64), config.threads)
        corr[c][np.ix_(k, k)] = r
        lag[c][np.ix_(k, k)] = s / fs_out
        logger.info(f"[correlation] channel {channels[c]}: {len(k)} pulses")

    out_attrs = {
        key: attrs[key]
        for key in ("event", "sensor", "pile", "range_m", "depth_m", "serial_number")
    }
    out_attrs |= {
        "channels": channels,
        "band": list(config.band),
        "taper_pc": config.taper_pc,
        "sampling_rate": fs_out,
    }
    data = {
        "corr": corr,
        "lag": lag,
        "strike": arrays["strike"],
        "time": arrays["time"].astype(np.int64),
    }
    return data, out_attrs


def write_correlation(path: Path, data: dict[str, np.ndarray], attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)
        f.attrs.update(attrs)


def read_correlation(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with h5py.File(path, "r") as f:
        data = {k: f[k][:] for k in ("corr", "lag", "strike", "time")}
        attrs = dict(f.attrs)
    data["time"] = data["time"].astype("datetime64[us]")
    attrs["channels"] = [int(c) for c in attrs["channels"]]
    return data, attrs


# Summaries for the report.


def pairs(corr: np.ndarray, time: np.ndarray, max_dt: float):
    """rho and time between strikes (s) of every pair (i, j > i) at most
    `max_dt` apart with a finite rho, from one hydrophone's matrix."""
    t = (time - time[0]) / np.timedelta64(1, "s")
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    keep = dt <= max_dt
    i, j, dt = i[keep], j[keep], dt[keep]
    rho = corr[i, j]
    ok = np.isfinite(rho)
    return rho[ok], dt[ok]


def _dt_bin(dt: np.ndarray, config: CorrelationConfig) -> np.ndarray:
    """Index of each time between strikes in bins of `bin_s` from 0 to
    `max_dt_s`; a pair exactly `max_dt_s` apart falls in the last bin."""
    n_bins = int(np.ceil(config.max_dt_s / config.bin_s))
    return np.minimum(np.floor(dt / config.bin_s), n_bins - 1).astype(np.int64)


def binned(rho: np.ndarray, dt: np.ndarray, config: CorrelationConfig) -> pl.DataFrame:
    """Number of pairs and the 5th, 50th and 95th percentiles of rho in bins
    of time between strikes: dt_s (bin center), n, rho_p5, rho_median,
    rho_p95."""
    b = config.bin_s
    return (
        pl.DataFrame({"bin": _dt_bin(dt, config), "rho": rho})
        .group_by("bin")
        .agg(
            pl.len().alias("n"),
            pl.col("rho").quantile(0.05).alias("rho_p5"),
            pl.col("rho").median().alias("rho_median"),
            pl.col("rho").quantile(0.95).alias("rho_p95"),
        )
        .with_columns(dt_s=(pl.col("bin") + 0.5) * b)
        .sort("bin")
        .select("dt_s", "n", "rho_p5", "rho_median", "rho_p95")
    )


def histogram(
    rho: np.ndarray, dt: np.ndarray, config: CorrelationConfig
) -> pl.DataFrame:
    """Counts of pairs in bins of time between strikes (dt_s, bin center) and
    of rho (rho, bin lower edge, RHO_BIN wide; rho below 0 counts in the
    lowest bin), nonzero bins only."""
    n_rho = round(1 / RHO_BIN)
    r = np.clip(np.floor(rho / RHO_BIN), 0, n_rho - 1).astype(np.int64)
    return (
        pl.DataFrame({"bin": _dt_bin(dt, config), "r": r})
        .group_by("bin", "r")
        .len("count")
        .with_columns(
            dt_s=(pl.col("bin") + 0.5) * config.bin_s,
            rho=pl.col("r") * RHO_BIN,
        )
        .sort("bin", "r")
        .select("dt_s", "rho", "count")
    )


def decorrelation_time(dt: np.ndarray, rho: np.ndarray, level: float) -> float:
    """Time between strikes at which `rho` (the binned median) first falls
    below `level`, by linear interpolation from rho = 1 at dt = 0 (a pulse is
    identical to itself); inf if it never does."""
    d = np.concatenate([[0.0], dt])
    r = np.concatenate([[1.0], rho])
    below = np.flatnonzero(r < level)
    if len(below) == 0:
        return np.inf
    j = below[0]
    return float(d[j - 1] + (r[j - 1] - level) / (r[j - 1] - r[j]) * (d[j] - d[j - 1]))


def level_label(level: float) -> str:
    """Column name of the decorrelation time at `level`, e.g. `T_0.9`."""
    return f"T_{level:g}"


def stacked(
    corr: np.ndarray, time: np.ndarray, max_dt: float, n_grid: int = 600
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rho of each strike with the strikes after it, interpolated onto a
    regular grid of time between strikes.

    Returns the grid (s), each strike's time since the first strike (s), and
    the image (n_strikes, n_grid), NaN beyond each strike's last valid pair
    or where it has fewer than two.
    """
    t = (time - time[0]) / np.timedelta64(1, "s")
    grid = np.linspace(0.0, max_dt, n_grid)
    image = np.full((len(t), n_grid), np.nan, np.float32)
    last = np.searchsorted(t, t + max_dt, side="right")
    for i in range(len(t)):
        if not np.isfinite(corr[i, i]):
            continue
        j = np.arange(i, last[i])
        ok = np.isfinite(corr[i, j])
        if ok.sum() < 2:
            continue
        dt, rho = t[j[ok]] - t[i], corr[i, j[ok]]
        image[i] = np.interp(grid, dt, rho, right=np.nan)
    return grid, t, image


KEYS = ["event", "sensor", "range_m", "channel", "depth_m", "serial_number"]


def summarize(path: Path, config: CorrelationConfig) -> dict[str, pl.DataFrame]:
    """Summaries of one correlation file, per hydrophone:

    bins     `binned` statistics of rho against time between strikes
    hist     `histogram` counts
    summary  number of valid pulses, median rho of the first bin, and
             the decorrelation time at each of `config.levels`
    """
    data, attrs = read_correlation(path)
    out: dict[str, list[pl.DataFrame]] = {"bins": [], "hist": [], "summary": []}
    for c, channel in enumerate(attrs["channels"]):
        label = {
            "event": attrs["event"],
            "sensor": attrs["sensor"],
            "range_m": float(attrs["range_m"]),
            "channel": channel,
            "depth_m": float(attrs["depth_m"][c]),
            "serial_number": str(attrs["serial_number"][c]),
        }
        rho, dt = pairs(data["corr"][c], data["time"], config.max_dt_s)
        if len(rho) == 0:
            continue
        b = binned(rho, dt, config)
        med = b["rho_median"].to_numpy()
        row = {
            **label,
            "n_pulses": int(np.isfinite(np.diagonal(data["corr"][c])).sum()),
            "rho_first": float(med[0]),
            **{
                level_label(v): decorrelation_time(b["dt_s"].to_numpy(), med, v)
                for v in config.levels
            },
        }
        lit = [pl.lit(v).alias(k) for k, v in label.items()]
        out["bins"].append(b.select(*lit, pl.all()))
        out["hist"].append(histogram(rho, dt, config).select(*lit, pl.all()))
        out["summary"].append(pl.DataFrame([row]))
    return {
        k: pl.concat(v, how="diagonal_relaxed") if v else pl.DataFrame()
        for k, v in out.items()
    }


def summarize_site(site, events: list) -> dict[str, pl.DataFrame]:
    """`summarize` over the events' correlation files, concatenated."""
    out: dict[str, list[pl.DataFrame]] = {"bins": [], "hist": [], "summary": []}
    for e in events:
        for s in site.sensors:
            path = site.correlation_path(e.name, s.name)
            if (
                site.is_excluded(e.name, s.name)
                or too_few_strikes(site, e.name, s.name)
                or not path.exists()
            ):
                continue
            for k, v in summarize(path, site.correlation).items():
                if not v.is_empty():
                    out[k].append(v)
    return {
        k: pl.concat(v, how="diagonal_relaxed").sort(
            "event", "range_m", "channel", maintain_order=True
        )
        if v
        else pl.DataFrame()
        for k, v in out.items()
    }
