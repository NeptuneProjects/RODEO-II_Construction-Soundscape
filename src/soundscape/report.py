"""Figures, LaTeX tables, and a Word document from the workflow outputs.

Figures (data coverage, event metrics, pulse PSD distributions, decidecade
band levels, level histograms, metrics versus range) and tables (event
summaries) for any site. The Word document collects them with captions. The
pulse-by-pulse table is a separate PDF (see soundscape.supplement).

Output layout:

    report/
    ├── report.docx
    ├── supplement.pdf                      (soundscape.supplement)
    ├── figures/<event>/{metrics,psd,bands,levels}.png
    ├── figures/<metric>_vs_range.png
    ├── figures/coverage.png
    └── tables/{detections,<event>,<event>_event}.tex, <event>_bands.csv
"""

import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import NullFormatter, ScalarFormatter

from soundscape.bands import band_sum, band_weights
from soundscape.config import SiteConfig
from soundscape.document import Document
from soundscape.events import distance_m, load_events, source_range
from soundscape.interpulse import summary_band
from soundscape.metrics import read_table
from soundscape.sources import coverage, recorded_intervals

logger = logging.getLogger(__name__)

DB = "dB re 1 $\\mathrm{\\mu Pa}$"
DB_E = "dB re 1 $\\mathrm{\\mu Pa^2 s}$"

# Axis labels (matplotlib mathtext) and table labels (LaTeX) for each metric,
# named as in ISO 18406:2017 and ISO 18405:2017.
METRIC_LABELS = {
    "SPL_pk": ("$L_{p,\\mathrm{pk}}$", DB),
    "SPL_pkpk": ("$L_{p,\\text{pk-pk}}$", DB),
    "SPL_rms90": ("$L_{p,90}$", DB),
    "SPL_rms": ("$L_{p}$", DB),  # ambient: mean-square SPL over an interval
    "SEL_ss": ("$\\mathrm{SEL}_\\mathrm{ss}$", DB_E),
    "SEL_cum": ("$\\mathrm{SEL}_\\mathrm{cum}$", DB_E),
    "T90": ("$T_{90}$", "ms"),
    "t_rise": ("$t_\\mathrm{r}$", "ms"),
    "t_decay": ("$t_\\mathrm{d}$", "ms"),
    "kurtosis": ("$\\beta_2$", ""),
    "WSPD": ("Wind speed", "m/s"),
    "WVHT": ("$H_{1/3}$", "m"),
    # Inter-pulse: mean-square SPL of 100 ms segments in the summary band, and
    # its median excess over ambient.
    "Lp_ip": ("$L_{p,\\mathrm{ip}}$", DB),
    "dL_ip": ("$\\Delta L_\\mathrm{ip}$", "dB"),
}

# The same labels for Word (see soundscape.document for the markup).
WORD_LABELS = {
    "SPL_pk": ("L_{p,pk}", "dB re 1 µPa"),
    "SPL_pkpk": ("L_{p,pk-pk}", "dB re 1 µPa"),
    "SPL_rms90": ("L_{p,90}", "dB re 1 µPa"),
    "SPL_rms": ("L_{p}", "dB re 1 µPa"),
    "SEL_ss": ("SEL_{ss}", "dB re 1 µPa^{2} s"),
    "SEL_cum": ("SEL_{cum}", "dB re 1 µPa^{2} s"),
    "T90": ("T_{90}", "ms"),
    "t_rise": ("t_{r}", "ms"),
    "t_decay": ("t_{d}", "ms"),
    "kurtosis": ("β_{2}", ""),
    "WSPD": ("Wind speed", "m/s"),
    "WVHT": ("H_{1/3}", "m"),
    "Lp_ip": ("L_{p,ip}", "dB re 1 µPa"),
    "dL_ip": ("ΔL_{ip}", "dB"),
}

# Factors from the units in the metrics tables to the units shown, and the
# decimals shown.
SCALE = {"T90": 1e3, "t_rise": 1e3, "t_decay": 1e3}
DIGITS = {"T90": 0, "t_rise": 0, "t_decay": 0}


def scaled(metric: str) -> pl.Expr:
    return pl.col(metric) * SCALE.get(metric, 1.0)


def _word_label(metric: str) -> str:
    name, unit = WORD_LABELS.get(metric, (metric, ""))
    return f"{name} ({unit})" if unit else name


def _axis_label(metric: str) -> str:
    name, unit = METRIC_LABELS.get(metric, (metric, ""))
    return f"{name}\n({unit})" if unit else name


def _tex_label(metric: str, stacked: bool = False) -> str:
    """LaTeX label; `stacked` puts the unit on a second line (needs makecell)."""
    name, unit = METRIC_LABELS.get(metric, (metric, ""))
    name = name.replace("\\text{", "\\textrm{")
    if not unit:
        return name
    if stacked:
        return f"\\makecell{{{name}\\\\({unit})}}"
    return f"{name} ({unit})"


def tex_escape(text: str) -> str:
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(char, f"\\{char}")
    return text


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def load_metrics(site: SiteConfig) -> pl.DataFrame:
    """Site-wide metrics table, valid pulses only."""
    return read_table(site.metrics_summary_path).filter(pl.col("valid"))


def total_sel() -> pl.Expr:
    """Cumulative SEL over all rows in a group, as column `SEL_total`
    (dB re 1 uPa^2 s); the last value of `SEL_cum` for valid pulses."""
    return ((10 ** (pl.col("SEL_ss") / 10)).sum().log10() * 10).alias("SEL_total")


def sensors_by_range(df: pl.DataFrame) -> list[tuple[str, float]]:
    """(sensor, range in m) for one event, nearest first."""
    rows = df.group_by("sensor").agg(pl.col("range_m").first()).sort("range_m")
    return list(rows.iter_rows())


def _db_percentiles(psd: np.ndarray) -> dict[str, np.ndarray]:
    """Median and 5th and 95th percentiles in dB along axis 0."""
    level = 10 * np.log10(psd.astype(np.float64))
    p5, med, p95 = np.percentile(level, [5, 50, 95], axis=0)
    return {"median": med, "p5": p5, "p95": p95}


def _channel_valid(f: h5py.File) -> np.ndarray:
    """Validity per strike and channel from a pulse PSD file; files written
    before /channel_valid only have it per strike."""
    if "channel_valid" in f:
        return f["channel_valid"][:]
    return np.repeat(f["valid"][:, None], len(f.attrs["channels"]), axis=1)


def _pulse_count(n: np.ndarray) -> str:
    """Valid pulses per hydrophone, as one number or a range."""
    lo, hi = int(np.min(n)), int(np.max(n))
    return f"{hi:,}" if lo == hi else f"{lo:,}\u2013{hi:,}"


def pulse_psd_stats(path: Path) -> dict:
    """Median and 5th and 95th percentiles of the valid pulse PSDs in dB, per
    channel (arrays of shape (n_channels, n_freqs)); `n` is the number of
    valid pulses per channel."""
    with h5py.File(path, "r") as f:
        freqs = f["freqs"][:]
        valid = _channel_valid(f)
        channels = [int(c) for c in f.attrs["channels"]]
        dset = f["psd"]
        per_ch = [
            _db_percentiles(dset[:, c, :][valid[:, c]]) for c in range(dset.shape[1])
        ]
    return {
        "freqs": freqs,
        "channels": channels,
        **{k: np.array([p[k] for p in per_ch]) for k in ("median", "p5", "p95")},
        "n": valid.sum(axis=0),
    }


def ambient_psd(path: Path, event: str, sensor: str) -> dict | None:
    """Ambient PSD in dB per channel: `level` over the whole window (NaN where
    it is zero), and, if the ambient file has interval PSDs, their median and
    5th and 95th percentiles over valid intervals. None if the window holds
    no recorded data."""
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        name = f"{event}/{sensor}"
        if name not in f:
            return None
        g = f[name]
        psd = g["psd"][:]
        # A spectrum of zeros means the window was a recording gap.
        if not (psd > 0).any():
            return None
        with np.errstate(divide="ignore"):
            level = np.where(psd > 0, 10 * np.log10(psd), np.nan)
        out = {
            "freqs": g["freqs"][:],
            "channels": [int(c) for c in g.attrs["channels"]],
            "level": level,
        }
        if "psd_intervals" in g:
            psd, valid = g["psd_intervals"][:], g["interval_valid"][:]
            per_ch = [
                _db_percentiles(psd[c][valid[c]])
                for c in range(psd.shape[0])
                if valid[c].any()
            ]
            if len(per_ch) == psd.shape[0]:
                for k in ("median", "p5", "p95"):
                    out[k] = np.array([p[k] for p in per_ch])
        return out


def pulse_band_stats(path: Path) -> dict | None:
    """Median and 5th and 95th percentiles over valid pulses of the
    single-strike sound exposure level in each decidecade band, in dB re
    1 uPa^2 s, per channel (arrays of shape (n_channels, n_bands)). None if the
    PSD file predates band exposure (rerun the psd stage)."""
    with h5py.File(path, "r") as f:
        if "band_exposure" not in f:
            return None
        valid = _channel_valid(f)
        exposure = f["band_exposure"][:]
        out = {k: f[f"band_{k}"][:] for k in ("center", "lower", "upper", "nominal")}
        out["channels"] = [int(c) for c in f.attrs["channels"]]
    per_ch = [
        _db_percentiles(exposure[valid[:, c], c, :]) for c in range(exposure.shape[1])
    ]
    for k in ("median", "p5", "p95"):
        out[k] = np.array([p[k] for p in per_ch])
    out["n"] = valid.sum(axis=0)
    return out


def ambient_band_stats(
    path: Path,
    event: str,
    sensor: str,
    lower: np.ndarray,
    upper: np.ndarray,
    window_s: float,
) -> dict | None:
    """Ambient noise in decidecade bands from the interval PSDs before an
    event: median and 5th and 95th percentiles over valid intervals of the
    mean-square sound pressure level in each band (`Lp_*`, dB re 1 uPa), and
    of the sound exposure that noise would give over `window_s` (`E_*`, dB re
    1 uPa^2 s), to compare with the band exposure of a pulse window of that
    length. Channels without valid intervals are NaN. None without interval
    PSDs."""
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        name = f"{event}/{sensor}"
        if name not in f or "psd_intervals" not in f[name]:
            return None
        g = f[name]
        freqs = g["freqs"][:]
        psd, valid = g["psd_intervals"][:], g["interval_valid"][:]
        channels = [int(c) for c in g.attrs["channels"]]
    weights = band_weights(freqs, lower, upper)
    mean_square = band_sum(psd.astype(np.float64) * (freqs[1] - freqs[0]), weights)
    n_ch, n_bands = len(channels), len(lower)
    out = {"channels": channels}
    for k in ("median", "p5", "p95"):
        out[f"Lp_{k}"] = np.full((n_ch, n_bands), np.nan)
    for c in range(n_ch):
        if valid[c].any():
            stats = _db_percentiles(mean_square[c][valid[c]])
            for k in ("median", "p5", "p95"):
                out[f"Lp_{k}"][c] = stats[k]
    for k in ("median", "p5", "p95"):
        out[f"E_{k}"] = out[f"Lp_{k}"] + 10 * np.log10(window_s)
    return out


def band_table(
    df: pl.DataFrame, pulses: dict[str, dict], ambient: dict[str, dict | None]
) -> pl.DataFrame:
    """Tidy band statistics for one event: per sensor, channel, and band, the
    pulse SEL_ss and ambient L_p and window exposure (median, p5, p95)."""
    info = {
        (r["sensor"], r["channel"]): (r.get("serial_number"), r.get("depth_m"))
        for r in hydrophones(df).iter_rows(named=True)
    }
    rows = []
    for sensor, st in pulses.items():
        amb = ambient.get(sensor)
        for i, channel in enumerate(st["channels"]):
            j = (
                amb["channels"].index(channel)
                if amb and channel in amb["channels"]
                else None
            )
            for b, nominal in enumerate(st["nominal"]):
                row = {
                    "sensor": sensor,
                    "channel": channel,
                    "serial_number": info.get((sensor, channel), (None, None))[0],
                    "depth_m": info.get((sensor, channel), (None, None))[1],
                    "band_hz": float(nominal),
                    "band_center_hz": float(st["center"][b]),
                }
                for k in ("median", "p5", "p95"):
                    row[f"SEL_ss_{k}"] = float(st[k][i, b])
                for k in ("median", "p5", "p95"):
                    row[f"ambient_Lp_{k}"] = (
                        float(amb[f"Lp_{k}"][j, b]) if j is not None else None
                    )
                for k in ("median", "p5", "p95"):
                    row[f"ambient_E_window_{k}"] = (
                        float(amb[f"E_{k}"][j, b]) if j is not None else None
                    )
                rows.append(row)
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def channel_colors(df: pl.DataFrame) -> dict[int, str]:
    """One color per channel number, consistent across sensors and figures."""
    channels = sorted(df["channel"].unique().to_list())
    return {c: f"C{k}" for k, c in enumerate(channels)}


def hydrophone_label(
    sensor: str, channel: int, serial: str | None, depth: float | None
) -> str:
    """e.g. "VLA1 Serial 722013 (16.2 m)", to cross-reference hydrophone
    sensitivities, which are listed by serial number for each deployment."""
    name = f"Serial {serial}" if serial else f"Ch. {channel}"
    where = f" ({depth:.1f} m)" if depth is not None else ""
    return f"{sensor} {name}{where}"


def hydrophone_labels(df: pl.DataFrame) -> dict[tuple[str, int], str]:
    """(sensor, channel) -> hydrophone label (see `hydrophone_label`)."""
    cols = [c for c in ("serial_number", "depth_m") if c in df.columns]
    rows = df.group_by("sensor", "channel").agg(*[pl.col(c).first() for c in cols])
    return {
        (r["sensor"], r["channel"]): hydrophone_label(
            r["sensor"], r["channel"], r.get("serial_number"), r.get("depth_m")
        )
        for r in rows.iter_rows(named=True)
    }


def _hydrophone_legend(
    ax: plt.Axes, sensor: str, channels: list[int], colors, labels, **kwargs
) -> None:
    handles = [
        plt.Line2D(
            [], [], ls="", marker="o", ms=3, color=colors[c], label=labels[(sensor, c)]
        )
        for c in channels
    ]
    ax.legend(handles=handles, fontsize="x-small", handletextpad=0.2, **kwargs)


def plot_event_metrics(
    df: pl.DataFrame,
    metrics: list[str],
    title: str,
    kurt: pl.DataFrame | None = None,
    window_s: float = 60.0,
) -> plt.Figure:
    """Metric time series for one event: rows are metrics (and, given `kurt`,
    kurtosis in windows of `window_s`), columns sensors (nearest first),
    points colored by hydrophone."""
    sensors = sensors_by_range(df)
    colors = channel_colors(df)
    labels = hydrophone_labels(df)
    rows = metrics + (["kurtosis"] if kurt is not None and not kurt.is_empty() else [])
    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=len(sensors),
        figsize=(2.4 * len(sensors) + 1.2, 1.35 * len(rows) + 0.8),
        sharex=True,
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    half = np.timedelta64(round(window_s * 5e5), "us")
    for j, (sensor, range_m) in enumerate(sensors):
        df_s = df.filter(pl.col("sensor") == sensor)
        axs[0, j].set_title(f"{sensor}, R = {range_m / 1e3:.1f} km")
        channels = sorted(df_s["channel"].unique().to_list())
        for channel in channels:
            df_c = df_s.filter(pl.col("channel") == channel)
            for i, metric in enumerate(metrics):
                axs[i, j].scatter(
                    df_c["time"],
                    df_c.select(scaled(metric)).to_series(),
                    s=0.5,
                    alpha=0.5,
                    color=colors[channel],
                    rasterized=True,
                )
            if rows[-1] == "kurtosis":
                k = kurt.filter(
                    (pl.col("sensor") == sensor)
                    & (pl.col("channel") == channel)
                    & pl.col("valid")
                ).sort("window_start")
                axs[-1, j].plot(
                    k["window_start"].to_numpy() + half,
                    k["kurtosis"],
                    marker=".",
                    ms=3,
                    lw=0.6,
                    color=colors[channel],
                )
        # Below the column, one hydrophone per line, so long labels fit.
        _hydrophone_legend(
            axs[-1, j],
            sensor,
            channels,
            colors,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=1,
        )
    for i, metric in enumerate(rows):
        label = _axis_label(metric)
        if metric == "kurtosis":
            label = f"{label} ({window_s:g} s)"
            for ax in axs[i]:
                # Gaussian noise has beta_2 = 3.
                ax.axhline(3.0, color="0.5", lw=0.6, ls="--")
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
        axs[i, 0].set_ylabel(label)
        for ax in axs[i]:
            ax.grid(True, lw=0.3)
    for ax in axs[-1]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time (UTC)")
    fig.suptitle(title)
    return fig


def plot_pulse_psd(
    event: str,
    sensors: list[tuple[str, float]],
    stats: dict[str, dict],
    ambient: dict[str, dict | None],
    labels: dict[tuple[str, int], str] | None = None,
    fmin: float = 10.0,
) -> plt.Figure:
    """Pulse PSD (median and 5th to 95th percentiles in dB) against ambient
    noise (the same over intervals before the event, if available): rows are
    hydrophones, columns sensors (nearest first)."""
    labels = labels or {}
    n_rows = max(len(stats[s]["channels"]) for s, _ in sensors)
    fmax = min(s["freqs"][-1] for s in stats.values())
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=len(sensors),
        figsize=(2.4 * len(sensors) + 0.8, 1.8 * n_rows + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    lo, hi = np.inf, -np.inf
    for j, (sensor, range_m) in enumerate(sensors):
        st, amb = stats[sensor], ambient.get(sensor)
        axs[0, j].set_title(
            f"{sensor}, R = {range_m / 1e3:.1f} km\n{_pulse_count(st['n'])} pulses"
        )
        f = st["freqs"]
        band = (f >= fmin) & (f <= fmax)
        n_ch = len(st["channels"])
        for i in range(n_rows):
            ax = axs[i, j]
            if i >= n_ch:
                ax.set_axis_off()
                continue
            channel = st["channels"][i]
            lo = min(lo, st["p5"][i][band].min())
            hi = max(hi, st["p95"][i][band].max())
            ax.fill_between(
                f,
                st["p5"][i],
                st["p95"][i],
                color="C3",
                alpha=0.25,
                lw=0,
                label="Pulses, 5th–95th percentile",
            )
            ax.plot(f, st["median"][i], c="C3", lw=1, label="Pulses, median")
            if amb is not None and channel in amb["channels"]:
                k = amb["channels"].index(channel)
                fa = amb["freqs"]
                if "median" in amb:
                    ax.fill_between(
                        fa,
                        amb["p5"][k],
                        amb["p95"][k],
                        color="C0",
                        alpha=0.25,
                        lw=0,
                        label="Ambient, 5th–95th percentile",
                    )
                    ax.plot(
                        fa, amb["median"][k], c="C0", lw=0.8, label="Ambient, median"
                    )
                    lo = min(lo, amb["p5"][k][(fa >= fmin) & (fa <= fmax)].min())
                else:
                    ax.plot(fa, amb["level"][k], c="C0", lw=0.8, label="Ambient")
            ax.text(
                0.03,
                0.05,
                labels.get((sensor, channel), f"Ch. {channel}"),
                transform=ax.transAxes,
            )
            ax.grid(True, which="both", lw=0.3)
            if i == n_ch - 1:
                # The bottom panel of a short column carries its own x labels.
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xlabel("Frequency (Hz)")
    for row in axs:
        # Tick labels on the first visible panel in each row (a short column
        # may be empty).
        next(ax for ax in row if ax.axison).yaxis.set_tick_params(labelleft=True)
    fig.supylabel("PSD (dB re 1 $\\mathrm{\\mu Pa^2/Hz}$)")
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_xlim(fmin, fmax)
    pad = 0.05 * (hi - lo)
    axs[0, 0].set_ylim(lo - pad, hi + pad)
    handles, names = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2)
    fig.suptitle(f"Event {event}")
    return fig


BAND_TICKS = [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]


def plot_band_levels(
    event: str,
    sensors: list[tuple[str, float]],
    pulses: dict[str, dict],
    ambient: dict[str, dict | None],
    labels: dict[tuple[str, int], str] | None = None,
    window_s: float = 1.2,
) -> plt.Figure:
    """Decidecade band exposure of the pulses (SEL_ss per band) and of the
    ambient noise over a window of the same length, each as the median and
    5th to 95th percentiles: rows are hydrophones, columns sensors."""
    labels = labels or {}
    n_rows = max(len(pulses[s]["channels"]) for s, _ in sensors)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=len(sensors),
        figsize=(2.4 * len(sensors) + 0.8, 1.8 * n_rows + 1.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    lo, hi = np.inf, -np.inf
    for j, (sensor, range_m) in enumerate(sensors):
        st, amb = pulses[sensor], ambient.get(sensor)
        f = st["center"]
        axs[0, j].set_title(
            f"{sensor}, R = {range_m / 1e3:.1f} km\n{_pulse_count(st['n'])} pulses"
        )
        n_ch = len(st["channels"])
        for i in range(n_rows):
            ax = axs[i, j]
            if i >= n_ch:
                ax.set_axis_off()
                continue
            channel = st["channels"][i]
            ax.fill_between(
                f,
                st["p5"][i],
                st["p95"][i],
                color="C3",
                alpha=0.25,
                lw=0,
                step="mid",
                label="Pulses, 5th–95th percentile",
            )
            ax.plot(
                f,
                st["median"][i],
                c="C3",
                lw=1,
                marker="o",
                ms=2.5,
                label="Pulses, median",
            )
            lo, hi = min(lo, np.nanmin(st["p5"][i])), max(hi, np.nanmax(st["p95"][i]))
            if amb is not None and channel in amb["channels"]:
                k = amb["channels"].index(channel)
                if np.isfinite(amb["E_median"][k]).any():
                    ax.fill_between(
                        f,
                        amb["E_p5"][k],
                        amb["E_p95"][k],
                        color="C0",
                        alpha=0.25,
                        lw=0,
                        step="mid",
                        label="Ambient, 5th–95th percentile",
                    )
                    ax.plot(
                        f,
                        amb["E_median"][k],
                        c="C0",
                        lw=1,
                        marker="s",
                        ms=2.5,
                        label="Ambient, median",
                    )
                    lo = min(lo, np.nanmin(amb["E_p5"][k]))
            ax.text(
                0.03,
                0.05,
                labels.get((sensor, channel), f"Ch. {channel}"),
                transform=ax.transAxes,
            )
            ax.grid(True, which="both", lw=0.3)
            if i == n_ch - 1:
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xlabel("Band center frequency (Hz)")
    for row in axs:
        next(ax for ax in row if ax.axison).yaxis.set_tick_params(labelleft=True)
    fig.supylabel(
        f"Band exposure over {window_s:g} s (dB re 1 $\\mathrm{{\\mu Pa^2\\,s}}$)"
    )
    ax0 = axs[0, 0]
    ax0.set_xscale("log")
    centers = next(iter(pulses.values()))["center"]
    ax0.set_xlim(centers[0] / 1.2, centers[-1] * 1.2)
    ticks = [
        t for t in BAND_TICKS if centers[0] <= t * 1.06 and t <= centers[-1] * 1.06
    ]
    ax0.set_xticks(ticks, [f"{t:g}" if t < 1000 else f"{t / 1000:g}k" for t in ticks])
    ax0.xaxis.set_minor_formatter(NullFormatter())
    pad = 0.05 * (hi - lo)
    ax0.set_ylim(lo - pad, hi + pad)
    handles, names = ax0.get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2)
    fig.suptitle(f"Event {event}: decidecade band levels")
    return fig


def plot_level_histograms(df: pl.DataFrame, metric: str, title: str) -> plt.Figure:
    """Distribution of a metric per hydrophone, one panel per sensor."""
    sensors = sensors_by_range(df)
    colors = channel_colors(df)
    labels = hydrophone_labels(df)
    fig, axs = plt.subplots(
        nrows=len(sensors),
        figsize=(5.0, 1.3 * len(sensors) + 0.8),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    values = df.select(scaled(metric)).to_series().drop_nans().to_numpy()
    bins = np.histogram_bin_edges(values, bins=60)
    for ax, (sensor, range_m) in zip(axs[:, 0], sensors):
        df_s = df.filter(pl.col("sensor") == sensor)
        channels = sorted(df_s["channel"].unique().to_list())
        for channel in channels:
            df_c = df_s.filter(pl.col("channel") == channel)
            ax.hist(
                df_c.select(scaled(metric)).to_series().to_numpy(),
                bins=bins,
                density=True,
                histtype="step",
                color=colors[channel],
            )
        ax.text(
            0.02, 0.85, f"{sensor}, R = {range_m / 1e3:.1f} km", transform=ax.transAxes
        )
        _hydrophone_legend(ax, sensor, channels, colors, labels, loc="upper right")
        ax.set_ylabel("Density")
        ax.grid(True, lw=0.3)
    axs[-1, 0].set_xlabel(_axis_label(metric).replace("\n", " "))
    fig.suptitle(title)
    return fig


def plot_metric_vs_range(df: pl.DataFrame, metric: str, site_name: str) -> plt.Figure:
    """A metric against range for all events, on each sensor's detection
    channel, colored by strike number, with the median per event and sensor."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    sc = ax.scatter(
        df["range_m"] / 1e3,
        df[metric],
        c=df["strike"],
        s=1,
        alpha=0.4,
        cmap="viridis",
        rasterized=True,
    )
    med = df.group_by("event", "sensor").agg(
        pl.col("range_m").first(),
        pl.col(metric).median().alias("median"),
        pl.col(metric).quantile(0.25).alias("q25"),
        pl.col(metric).quantile(0.75).alias("q75"),
    )
    ax.errorbar(
        med["range_m"] / 1e3,
        med["median"],
        yerr=[med["median"] - med["q25"], med["q75"] - med["median"]],
        fmt="o",
        ms=3,
        c="k",
        lw=0.8,
        label="Median and IQR per event and sensor",
    )
    fig.colorbar(sc, ax=ax, label="Strike number")
    ax.set_xlabel("Range (km)")
    ax.set_ylabel(_axis_label(metric))
    ax.grid(True, lw=0.3)
    ax.legend(loc="upper right", fontsize="small")
    ax.set_title(f"{site_name}: all events, detection channels")
    return fig


def plot_interpulse_decay(
    decay: pl.DataFrame, thresholds: list[float], site_name: str
) -> plt.Figure:
    """Median summary-band level against time since the strike, relative to
    the ambient median, on each sensor's detection channel (`decay` already
    filtered to them): one panel per sensor, one line per event, colored by
    range. Times where fewer than half the event's intervals reach are
    dropped (the strike spacing varies)."""
    sensors = (
        decay.group_by("sensor").agg(pl.col("range_m").min()).sort("range_m")["sensor"]
    )
    fig, axs = plt.subplots(
        1,
        len(sensors),
        figsize=(2.6 * len(sensors) + 1.0, 3.0),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    norm = plt.Normalize(decay["range_m"].min() / 1e3, decay["range_m"].max() / 1e3)
    cmap = plt.get_cmap("viridis")
    for ax, sensor in zip(axs[0], sensors):
        d = decay.filter(pl.col("sensor") == sensor)
        for (_,), c in d.group_by(["event"], maintain_order=True):
            c = c.filter(pl.col("n") >= 0.5 * pl.col("n").max()).sort("t_s")
            ax.plot(
                c["t_s"] + 0.05,
                c["dL"],
                lw=0.8,
                color=cmap(norm(c["range_m"][0] / 1e3)),
            )
        ax.axhline(0, color="k", lw=0.6)
        for t in thresholds:
            ax.axhline(t, color="gray", lw=0.6, ls=":")
        ax.set_title(sensor)
        ax.set_xlabel("Time since strike (s)")
        ax.grid(True, lw=0.3)
    axs[0, 0].set_ylabel("Level above ambient (dB)")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0, -1], label="Range (km)"
    )
    fig.suptitle(f"{site_name}: inter-pulse decay, detection channels")
    return fig


def plot_interpulse_cdf(hist: pl.DataFrame, site_name: str) -> plt.Figure:
    """Cumulative distributions of the 100 ms summary-band levels between
    pulses and in the ambient windows, pooled over events, on each sensor's
    detection channel (`hist` already filtered to them)."""
    sensors = (
        hist.group_by("sensor").agg(pl.col("range_m").min()).sort("range_m")["sensor"]
    )
    fig, axs = plt.subplots(
        1,
        len(sensors),
        figsize=(2.6 * len(sensors), 3.0),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    styles = {"interpulse": ("C0", "Inter-pulse"), "ambient": ("0.5", "Ambient")}
    for ax, sensor in zip(axs[0], sensors):
        for kind, (color, label) in styles.items():
            h = (
                hist.filter(pl.col("sensor") == sensor, pl.col("kind") == kind)
                .group_by("level")
                .agg(pl.col("count").sum())
                .sort("level")
            )
            if h.is_empty():
                continue
            cdf = np.cumsum(h["count"].to_numpy()) / h["count"].sum() * 100
            ax.step(h["level"] + 0.5, cdf, where="post", color=color, label=label)
        ax.axhline(50, color="k", lw=0.4, ls=":")
        ax.set_title(sensor)
        ax.set_xlabel(f"{METRIC_LABELS['Lp_ip'][0]} ({DB})")
        ax.grid(True, lw=0.3)
    axs[0, 0].set_ylabel("Cumulative percentage")
    axs[0, 0].legend(loc="lower right", fontsize="small")
    fig.suptitle(f"{site_name}: 100 ms levels, detection channels")
    return fig


def plot_data_coverage(
    events: list,
    recorded: dict[str, list[tuple[np.datetime64, np.datetime64]]],
    covered: dict[tuple[str, str], float],
    site_name: str,
    pad_days: float = 3.0,
) -> plt.Figure:
    """Recording coverage of each sensor (top) and pile-driving events
    (bottom, one row per event) on a shared time axis.

    Each event is a bar over its time window with a diamond at its center,
    which stays visible when the window is too short to see at the scale of
    the deployment. A column at the right gives each window and the sensors that recorded it.
    Events that no sensor recorded are drawn in gray.

    `recorded` maps each sensor to its continuous recording intervals and
    `covered` maps (event, sensor) to the fraction of the event recorded.
    """
    events = sorted(events, key=lambda e: e.start)
    sensors = list(recorded)
    pad = np.timedelta64(round(pad_days * 86400), "s")
    t_lo = min(e.start for e in events) - pad
    t_hi = max(e.end for e in events) + pad

    fig, (ax_s, ax_e) = plt.subplots(
        nrows=2,
        figsize=(7.0, 0.3 * len(sensors) + 0.2 * len(events) + 1.2),
        sharex=True,
        height_ratios=[len(sensors) + 0.5, len(events) + 0.5],
        layout="constrained",
    )
    ink, muted = "0.15", "0.6"

    for i, sensor in enumerate(sensors):
        spans = [
            (mdates.date2num(max(a, t_lo)), mdates.date2num(min(b, t_hi)))
            for a, b in recorded[sensor]
            if b > t_lo and a < t_hi
        ]
        ax_s.broken_barh(
            [(a, b - a) for a, b in spans], (i - 0.3, 0.6), color=f"C{i}", lw=0
        )
    ax_s.set_yticks(range(len(sensors)), sensors)
    ax_s.set_ylim(len(sensors) - 0.5, -0.5)
    ax_s.set_title(f"{site_name}: Data coverage and pile-driving events")

    # Right-hand columns, offset in points from the right edge of the axes.
    col_when, col_by = 6, 84
    to_row = ax_e.get_yaxis_transform()

    def column(x: float, y: float, text: str, color: str) -> None:
        ax_e.annotate(
            text,
            (1, y),
            xycoords=to_row,
            xytext=(x, 0),
            textcoords="offset points",
            va="center",
            color=color,
        )

    column(col_when, -1, "Window (UTC)", muted)
    column(col_by, -1, "Recorded by", muted)

    for i, event in enumerate(events):
        x0, x1 = mdates.date2num(event.start), mdates.date2num(event.end)
        by = [s for s in sensors if covered.get((event.name, s), 0.0) > 0.0]
        color = ink if by else muted
        ax_e.hlines(i, x0, x1, color=color, lw=3)
        ax_e.plot((x0 + x1) / 2, i, marker="D", ms=4, color=color, ls="")
        when = f"{str(event.start)[5:16].replace('T', ' ')}–{str(event.end)[11:16]}"
        names = [
            s
            if covered[(event.name, s)] >= 0.999
            else f"{s} ({covered[(event.name, s)]:.0%})"
            for s in by
        ]
        column(col_when, i, when, color)
        column(col_by, i, ", ".join(names) or "no data", color)
    ax_e.set_yticks(range(len(events)), [f"{e.name}" for e in events])
    for tick, event in zip(ax_e.get_yticklabels(), events):
        if not any(covered.get((event.name, s), 0.0) > 0.0 for s in sensors):
            tick.set_color(muted)
    ax_e.set_ylim(len(events) - 0.5, -0.5)
    ax_e.set_ylabel("Event")
    ax_e.set_xlim(mdates.date2num(t_lo), mdates.date2num(t_hi))
    ax_e.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax_e.xaxis.set_minor_locator(mdates.DayLocator())
    ax_e.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax_e.set_xlabel("Date (UTC)")
    for ax in (ax_s, ax_e):
        ax.grid(True, lw=0.3)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    return fig


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _num(v: float | None, digits: int) -> str:
    if v is None or not np.isfinite(v):
        return "--"
    # Adding 0.0 turns a rounded -0.0 into 0.0.
    return f"{round(v, digits) + 0.0:.{digits}f}"


def _stat(
    median: float | None, p5: float | None, p95: float | None, digits: int
) -> str:
    """Median with the 5th and 95th percentiles in brackets."""
    if median is None:
        return "--"
    return f"{_num(median, digits)} [{_num(p5, digits)}, {_num(p95, digits)}]"


def _percentiles(expr: pl.Expr, name: str) -> list[pl.Expr]:
    return [
        expr.median().alias(f"{name}_median"),
        expr.quantile(0.05, interpolation="linear").alias(f"{name}_p5"),
        expr.quantile(0.95, interpolation="linear").alias(f"{name}_p95"),
    ]


def _tabular(
    columns: list[str], rows: list[list[str]], align: str, note: str | None = None
) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\midrule")
    lines += [" & ".join(r) + " \\\\" for r in rows]
    if note:
        lines += ["\\midrule", f"\\multicolumn{{{len(columns)}}}{{l}}{{{note}}} \\\\"]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def hydrophones(df: pl.DataFrame) -> pl.DataFrame:
    """One row per sensor and channel: range_m, depth_m, serial_number,
    ordered by range and then depth (or channel)."""
    cols = [c for c in ("depth_m", "serial_number") if c in df.columns]
    return _by_hydrophone(
        df.group_by("sensor", "channel").agg(
            pl.col("range_m").first(), *[pl.col(c).first() for c in cols]
        )
    )


def _by_hydrophone(table: pl.DataFrame) -> pl.DataFrame:
    """Sort by range, then depth (if known), then channel. Joins do not keep
    row order, so this follows every join."""
    order = ["range_m", "sensor", "depth_m", "channel"]
    return table.sort([c for c in order if c in table.columns])


HYDROPHONE_COLUMNS = ["Sensor", "Range (km)", "Serial", "Depth (m)"]
HYDROPHONE_ALIGN = "lrlr"


def _hydrophone_cells(r: dict, text: Callable[[str], str]) -> list[str]:
    return [
        text(r["sensor"]),
        f"{r['range_m'] / 1e3:.1f}",
        text(r.get("serial_number") or "--"),
        _num(r.get("depth_m"), 1),
    ]


def pulse_stats_rows(
    df: pl.DataFrame,
    metrics: list[str],
    label: Callable[[str], str] = _tex_label,
    text: Callable[[str], str] = tex_escape,
) -> tuple[list[str], list[list[str]], str]:
    """Columns, rows, and alignment of the per-pulse statistics for one
    event: one row per hydrophone, and for each metric its median with the
    5th and 95th percentiles over valid pulses in brackets. SEL_cum is left
    out (see `event_rows`)."""
    metrics = [m for m in metrics if m != "SEL_cum"]
    stats = df.group_by("sensor", "channel").agg(
        *[e for m in metrics for e in _percentiles(scaled(m), m)]
    )
    table = _by_hydrophone(hydrophones(df).join(stats, on=["sensor", "channel"]))
    columns = HYDROPHONE_COLUMNS + [label(m) for m in metrics]
    rows = [
        _hydrophone_cells(r, text)
        + [
            _stat(r[f"{m}_median"], r[f"{m}_p5"], r[f"{m}_p95"], DIGITS.get(m, 1))
            for m in metrics
        ]
        for r in table.iter_rows(named=True)
    ]
    return columns, rows, HYDROPHONE_ALIGN + "r" * len(metrics)


def event_rows(
    event: str,
    df: pl.DataFrame,
    ambient_spl: pl.DataFrame | None = None,
    kurt: pl.DataFrame | None = None,
    label: Callable[[str], str] = _tex_label,
    text: Callable[[str], str] = tex_escape,
    inter: pl.DataFrame | None = None,
) -> tuple[list[str], list[list[str]], str]:
    """Columns, rows, and alignment of the event-level quantities for one
    event, one row per hydrophone: serial number, number of valid pulses,
    cumulative SEL over them, kurtosis in windows containing strikes, and the
    ambient kurtosis before the event (median with the 5th and 95th
    percentiles in brackets). Ambient levels are given in decidecade bands
    (see `ambient_band_stats`), not here. With `inter` (the event's rows of
    `interpulse.excess_summary`), also the inter-pulse level, its median
    excess over ambient, and the percentage of inter-pulse time more than
    each threshold above ambient."""
    table = hydrophones(df).join(
        df.group_by("sensor", "channel").agg(
            pl.col("strike").n_unique().alias("pulses"), total_sel()
        ),
        on=["sensor", "channel"],
    )
    columns = HYDROPHONE_COLUMNS + ["Pulses", label("SEL_cum")]
    extra: list[tuple[str, str]] = []  # (prefix, header)
    if kurt is not None and not kurt.is_empty():
        k = kurt.filter(
            (pl.col("event") == event) & pl.col("valid") & (pl.col("n_strikes") > 0)
        )
        table = table.join(
            k.group_by("sensor", "channel").agg(
                *_percentiles(pl.col("kurtosis"), "kurt")
            ),
            on=["sensor", "channel"],
            how="left",
        )
        extra.append(("kurt", f"{label('kurtosis')}, piling"))
    if ambient_spl is not None and "kurtosis" in ambient_spl.columns:
        a = ambient_spl.filter((pl.col("event") == event) & pl.col("valid"))
        table = table.join(
            a.group_by("sensor", "channel").agg(
                *_percentiles(pl.col("kurtosis"), "ambkurt")
            ),
            on=["sensor", "channel"],
            how="left",
        )
        extra.append(("ambkurt", f"Ambient {label('kurtosis')}"))
    columns += [h for _, h in extra]
    above = []
    if inter is not None and not inter.is_empty():
        above = [c for c in inter.columns if c.startswith("above_")]
        table = table.join(
            inter.select(
                "sensor", "channel", "Lp_median", "Lp_p5", "Lp_p95", "dL", *above
            ),
            on=["sensor", "channel"],
            how="left",
        )
        columns += [label("Lp_ip"), label("dL_ip")] + [
            text(f"> {c.removeprefix('above_')} dB (%)") for c in above
        ]
    table = _by_hydrophone(table)
    rows = []
    for r in table.iter_rows(named=True):
        cells = (
            _hydrophone_cells(r, text)
            + [
                f"{r['pulses']:,}",
                _num(r["SEL_total"], 1),
            ]
            + [
                _stat(
                    r[f"{x}_median"],
                    r[f"{x}_p5"],
                    r[f"{x}_p95"],
                    1,
                )
                for x, _ in extra
            ]
        )
        if above:
            cells += [
                _stat(r["Lp_median"], r["Lp_p5"], r["Lp_p95"], 1),
                _num(r["dL"], 1),
            ] + [_num(None if r[c] is None else 100 * r[c], 0) for c in above]
        rows.append(cells)
    n_ip = 2 + len(above) if above else 0
    return columns, rows, HYDROPHONE_ALIGN + "rr" + "r" * (len(extra) + n_ip)


def weather_note(
    weather: dict[str, float | None] | None, label: Callable[[str], str]
) -> str | None:
    """Mean wind speed and significant wave height over the event."""
    if not weather:
        return None
    parts = [
        f"{label(k)}: {'--' if v is None else f'{v:.1f}'}" for k, v in weather.items()
    ]
    return "Means over the event: " + "; ".join(parts)


MET_NAMES = {"WSPD": "wind speed", "WVHT": "significant wave height"}


def station_note(site: SiteConfig) -> str | None:
    """Which station each weather variable is from, and its distance from
    the site's sensors (km, as a range over sensors)."""
    parts = []
    for st in site.report.met_stations:
        km = [
            distance_m(st.latitude, st.longitude, s.latitude, s.longitude) / 1e3
            for s in site.sensors
        ]
        lo, hi = round(min(km)), round(max(km))
        dist = f"{lo} km" if lo == hi else f"{lo}–{hi} km"
        what = _and([MET_NAMES.get(v, v) for v in st.variables])
        parts.append(f"{what} from station {st.name} ({dist} from the sensors)")
    if not parts:
        return None
    return "Data: " + "; ".join(parts) + "."


def detections_table(site: SiteConfig, events: list) -> str:
    """Detections per event and sensor (LaTeX tabular)."""
    return _tabular(*detection_rows(site, events, tex_escape), DETECTIONS_ALIGN)


DETECTIONS_ALIGN = "lllrllrr"


def detection_rows(
    site: SiteConfig, events: list, text: Callable[[str], str] = tex_escape
) -> tuple[list[str], list[list[str]]]:
    """Columns and rows of the detections per event and sensor."""
    columns = [
        "Event",
        "Pile",
        "Sensor",
        "Range (km)",
        "Start (UTC)",
        "End (UTC)",
        "Strikes",
        "Median SNR (dB)",
    ]
    rows = []
    for event in events:
        for sensor in site.sensors:
            path = site.detections_path(event.name, sensor.name)
            if site.is_excluded(event.name, sensor.name) or not path.exists():
                continue
            det = pl.read_csv(path, try_parse_dates=True)
            snr = det["snr_db"].median() if len(det) else None
            rows.append(
                [
                    text(event.name),
                    text(event.pile),
                    text(sensor.name),
                    f"{source_range(event, sensor) / 1e3:.1f}",
                    str(event.start)[:16].replace("T", " "),
                    str(event.end)[:16].replace("T", " "),
                    f"{len(det):,}",
                    "--" if snr is None else f"{snr:.1f}",
                ]
            )
    return columns, rows


def event_weather(met: pl.DataFrame, start, end) -> dict[str, float | None]:
    sub = met.filter((pl.col("datetime") >= start) & (pl.col("datetime") <= end))
    return {key: sub[key].mean() if len(sub) else None for key in ("WSPD", "WVHT")}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path, dpi: int, title: bool = True) -> None:
    """Save and close a figure. Without `title`, drop the figure title (or,
    if it has none, the title of its first axes)."""
    if not title:
        if fig.get_suptitle():
            fig.suptitle("")
        elif fig.axes:
            fig.axes[0].set_title("")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"[report] {path}")


def write_coverage_figure(site: SiteConfig, path: Path) -> None:
    """Data coverage for every event in the site file, from the inventories."""
    inventories = {
        s.name: site.inventory_path(s.name)
        for s in site.sensors
        if site.inventory_path(s.name).exists()
    }
    if not inventories:
        logger.warning("[report] no inventories; run inventory first")
        return
    for s in site.sensors:
        if s.name not in inventories:
            logger.warning(f"[report] coverage: no inventory for {s.name}")
    events = load_events(site)
    recorded = {s: recorded_intervals(p) for s, p in inventories.items()}
    covered = {
        (e.name, s): coverage(p, e.start, e.end)
        for e in events
        for s, p in inventories.items()
    }
    _save(
        plot_data_coverage(events, recorded, covered, site.name),
        path,
        site.report.dpi,
        site.report.titles,
    )


def _window(event) -> str:
    return f"{str(event.start)[:16].replace('T', ' ')}–{str(event.end)[11:16]}"


def _and(items: list[str]) -> str:
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def interpulse_summaries(site: SiteConfig) -> dict[str, pl.DataFrame]:
    """Site-wide inter-pulse summaries written by the interpulse stage
    (excess restricted to the t95-to-t5 interval); empty if not run."""
    paths = {k: site.interpulse_summary_path(k) for k in ("excess", "decay", "hist")}
    if not all(p.exists() for p in paths.values()):
        return {}
    out = {k: pl.read_csv(p) for k, p in paths.items()}
    out["excess"] = out["excess"].filter(pl.col("definition") == "interval")
    return out


def _event_excess(inter: dict | None, event) -> pl.DataFrame | None:
    if not inter:
        return None
    return inter["excess"].filter(pl.col("event") == event.name)


def interpulse_band(site: SiteConfig) -> tuple[float, float]:
    """Edges of the inter-pulse summary band at the site's sampling rate."""
    fs = site_sampling_rate(site)
    return summary_band(site.interpulse, site.bands, fs)


def site_sampling_rate(site: SiteConfig) -> float:
    """Sampling rate from the first pulse file found (19531.25 Hz at both
    sites)."""
    for path in sorted(site.output_dir.glob("events/*/pulses_*.h5")):
        with h5py.File(path, "r") as f:
            return float(f.attrs["sampling_rate"])
    raise FileNotFoundError(f"no pulse files under {site.output_dir}")


def interpulse_note(site: SiteConfig) -> str:
    """Caption text for the inter-pulse columns of the event tables (Word)."""
    lo, hi = interpulse_band(site)
    c = site.interpulse
    return (
        f"L_{{p,ip}} is the mean-square SPL from {lo:g} Hz to {hi / 1e3:.1f} kHz in "
        f"{c.segment_s * 1e3:g}-ms segments between strikes, over the inter-pulse "
        f"interval from t_{{95}} of one pulse to t_{{5}} of the next (Guan et al. "
        f"2015); ΔL_{{ip}} is its median less the median of the same quantity in "
        f"the ambient window before the event, and the last columns give the "
        f"percentage of inter-pulse time more than "
        + _and([f"{t:g} dB" for t in c.thresholds_db])
        + " above that ambient median."
    )


def make_report(site: SiteConfig, events: list) -> None:
    """Write all figures and tables for the given events, and the Word
    document that collects them."""
    cfg = site.report
    if cfg.style is not None:
        plt.style.use(cfg.style)
    figs = site.report_dir / "figures"
    tables = site.report_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    write_coverage_figure(site, figs / "coverage.png")

    if not site.metrics_summary_path.exists():
        logger.warning(
            f"[report] missing {site.metrics_summary_path}; run metrics first"
        )
        return
    names = [e.name for e in events]
    df = load_metrics(site).filter(pl.col("event").is_in(names))
    missing = [m for m in cfg.metrics if m not in df.columns]
    if missing:
        raise ValueError(
            f"{site.metrics_summary_path} has no {missing}; rerun the metrics stage"
        )
    ambient_spl = (
        read_table(site.ambient_spl_path) if site.ambient_spl_path.exists() else None
    )
    kurt = read_table(site.kurtosis_path) if site.kurtosis_path.exists() else None
    inter = interpulse_summaries(site)
    if kurt is None:
        logger.warning(f"[report] missing {site.kurtosis_path}; run kurtosis first")
    met = pl.read_csv(cfg.met, try_parse_dates=True) if cfg.met is not None else None

    for event in events:
        df_e = df.filter(pl.col("event") == event.name)
        if df_e.is_empty():
            logger.warning(f"[report] {event.name}: no metrics")
            continue
        kurt_e = (
            kurt.filter(pl.col("event") == event.name) if kurt is not None else None
        )
        title = f"Event {event.name} (pile {event.pile}), {_window(event)} UTC"
        _save(
            plot_event_metrics(
                df_e, cfg.metrics, title, kurt_e, site.kurtosis.window_s
            ),
            figs / event.name / "metrics.png",
            cfg.dpi,
            cfg.titles,
        )
        _save(
            plot_level_histograms(df_e, cfg.histogram_metric, title),
            figs / event.name / "levels.png",
            cfg.dpi,
            cfg.titles,
        )

        sensors = sensors_by_range(df_e)
        stats = {
            s: pulse_psd_stats(site.pulse_psd_path(event.name, s))
            for s, _ in sensors
            if site.pulse_psd_path(event.name, s).exists()
        }
        if stats:
            sensors = [(s, r) for s, r in sensors if s in stats]
            ambient = {
                s: ambient_psd(site.ambient_path, event.name, s) for s, _ in sensors
            }
            _save(
                plot_pulse_psd(
                    event.name, sensors, stats, ambient, hydrophone_labels(df_e)
                ),
                figs / event.name / "psd.png",
                cfg.dpi,
                cfg.titles,
            )
            write_band_outputs(site, event, df_e, sensors, figs, tables)

        weather = (
            event_weather(met, event.start, event.end) if met is not None else None
        )
        (tables / f"{event.name}.tex").write_text(
            _tabular(*pulse_stats_rows(df_e, cfg.metrics))
        )
        columns, rows, align = event_rows(
            event.name, df_e, ambient_spl, kurt_e, inter=_event_excess(inter, event)
        )
        (tables / f"{event.name}_event.tex").write_text(
            _tabular(columns, rows, align, weather_note(weather, _tex_label))
        )

    if inter:
        _save(
            plot_interpulse_decay(
                detection_channels(site, inter["decay"]),
                site.interpulse.thresholds_db,
                site.name,
            ),
            figs / "interpulse_decay.png",
            cfg.dpi,
            cfg.titles,
        )
        _save(
            plot_interpulse_cdf(detection_channels(site, inter["hist"]), site.name),
            figs / "interpulse_cdf.png",
            cfg.dpi,
            cfg.titles,
        )

    for metric in cfg.range_metrics:
        _save(
            plot_metric_vs_range(detection_channels(site, df), metric, site.name),
            figs / f"{metric}_vs_range.png",
            cfg.dpi,
            cfg.titles,
        )

    (tables / "detections.tex").write_text(detections_table(site, events))
    logger.info(f"[report] tables -> {tables}")

    write_document(site, events, df, ambient_spl, met, kurt, inter)


def write_band_outputs(
    site: SiteConfig,
    event,
    df_e: pl.DataFrame,
    sensors: list[tuple[str, float]],
    figs: Path,
    tables: Path,
) -> None:
    """Decidecade band figure and table for one event."""
    cfg = site.report
    window_s = site.extract.pre_s + site.extract.post_s
    pulses = {}
    for s, _ in sensors:
        st = pulse_band_stats(site.pulse_psd_path(event.name, s))
        if st is None:
            logger.warning(
                f"[report] {event.name}/{s}: no band exposure; rerun the psd stage"
            )
            continue
        pulses[s] = st
    if not pulses:
        return
    sensors = [(s, r) for s, r in sensors if s in pulses]
    ambient = {
        s: ambient_band_stats(
            site.ambient_path,
            event.name,
            s,
            pulses[s]["lower"],
            pulses[s]["upper"],
            window_s,
        )
        for s, _ in sensors
    }
    _save(
        plot_band_levels(
            event.name, sensors, pulses, ambient, hydrophone_labels(df_e), window_s
        ),
        figs / event.name / "bands.png",
        cfg.dpi,
        cfg.titles,
    )
    band_table(df_e, pulses, ambient).write_csv(tables / f"{event.name}_bands.csv")


def detection_channels(site: SiteConfig, df: pl.DataFrame) -> pl.DataFrame:
    """Rows of `df` on each sensor's detection channel."""
    detect_ch = {s.name: s.detect_channel for s in site.sensors}
    return df.filter(
        pl.col("channel") == pl.col("sensor").replace_strict(detect_ch, default=None)
    )


def _word_name(metric: str) -> str:
    return WORD_LABELS.get(metric, (metric,))[0]


def write_document(
    site: SiteConfig,
    events: list,
    df: pl.DataFrame,
    ambient_spl: pl.DataFrame | None = None,
    met: pl.DataFrame | None = None,
    kurt: pl.DataFrame | None = None,
    inter: dict[str, pl.DataFrame] | None = None,
) -> Path:
    """Word document of the figures (as already written to the report
    directory) and tables, with captions, for pasting into a shared report.

    Event tables are on landscape pages. The pulse-by-pulse table is the
    separate data supplement (`supplement.pdf`).
    """
    cfg = site.report
    figs = site.report_dir / "figures"
    doc = Document(cfg.docx_template)
    a = site.ambient
    k_win = site.kurtosis.window_s

    def figure(path: Path, caption: str) -> None:
        if path.exists():
            doc.figure(path, caption)
        else:
            logger.warning(f"[report] docx: missing {path}")

    def at_range(sensors: list[tuple[str, float]]) -> str:
        return _and([f"{s} ({r / 1e3:.1f} km)" for s, r in sensors])

    n_events = df["event"].n_unique()
    doc.heading(site.name, 1)
    doc.paragraph(
        f"Figures and tables generated by soundscape report on "
        f"{date.today():%Y-%m-%d} from {len(df):,} valid pulse records "
        f"(all hydrophones) in {n_events} event{'s' * (n_events != 1)}. Metrics "
        f"follow ISO 18406:2017, with quantities and levels defined in ISO "
        f"18405:2017. Statistics are medians with the 5th and 95th percentiles "
        f"in brackets. Hydrophones are identified by sensor, serial number, and "
        f"depth below the sea surface, e.g. VLA1 Serial 722013 (16.2 m), for "
        f"cross-reference with the table of hydrophone sensitivities."
    )

    doc.heading("Data coverage", 2)
    figure(
        figs / "coverage.png",
        f"Recording periods of each sensor at {site.name} (top) and the "
        f"pile-driving events (bottom), with each event's time window and the "
        f"sensors that recorded it. Events that no sensor recorded are gray; a "
        f"percentage is the fraction of an event that a sensor recorded.",
    )

    doc.heading("Detections", 2)
    columns, rows = detection_rows(site, events, str)
    doc.table(
        columns,
        rows,
        f"Pile strikes detected at {site.name} for each event and sensor. Range "
        f"is the horizontal distance from the pile to the sensor; SNR is the "
        f"strike energy relative to the noise floor in the detection band on the "
        f"sensor's detection channel.",
        DETECTIONS_ALIGN,
    )

    doc.heading("Received levels versus range", 2)
    for metric in cfg.range_metrics:
        figure(
            figs / f"{metric}_vs_range.png",
            f"{_word_name(metric)} versus range for all events at {site.name}, on "
            f"each sensor's detection channel. Points are individual pulses "
            f"colored by strike number; black markers are the median and "
            f"interquartile range for each event and sensor.",
        )

    if inter:
        lo, hi = interpulse_band(site)
        doc.heading("Inter-pulse sound field", 2)
        figure(
            figs / "interpulse_decay.png",
            f"Decay of the sound field between strikes at {site.name}: median "
            f"mean-square SPL L_{{p}} in {lo:g} Hz to {hi / 1e3:.1f} kHz over "
            f"consecutive {site.interpulse.segment_s * 1e3:g}-ms segments after "
            f"each strike, relative to the median of the same quantity over the "
            f"ambient window before the event, on each sensor's detection "
            f"channel. Lines are events, colored by range; dotted lines mark "
            + _and([f"{t:g} dB" for t in site.interpulse.thresholds_db])
            + " above ambient.",
        )
        figure(
            figs / "interpulse_cdf.png",
            f"Cumulative distributions of the {site.interpulse.segment_s * 1e3:g}-ms "
            f"levels L_{{p,ip}} ({lo:g} Hz to {hi / 1e3:.1f} kHz) in the inter-pulse "
            f"intervals (from t_{{95}} of one pulse to t_{{5}} of the next) and in "
            f"the ambient windows before each event, pooled over the events at "
            f"{site.name}, on each sensor's detection channel.",
        )

    per_pulse = [m for m in cfg.metrics if m != "SEL_cum"]
    for event in events:
        df_e = df.filter(pl.col("event") == event.name)
        if df_e.is_empty():
            continue
        kurt_e = (
            kurt.filter(pl.col("event") == event.name) if kurt is not None else None
        )
        sensors = sensors_by_range(df_e)
        weather = (
            event_weather(met, event.start, event.end) if met is not None else None
        )
        when = f"event {event.name} (pile {event.pile}), {_window(event)} UTC"

        doc.orientation(landscape=True)
        doc.heading(f"Event {event.name}", 2)
        columns, rows, align = pulse_stats_rows(df_e, cfg.metrics, _word_label, str)
        doc.table(
            columns,
            rows,
            f"Per-pulse metrics for {when}, for each hydrophone: median, with the "
            f"5th and 95th percentiles over valid pulses in brackets. Range is "
            f"from the pile to the sensor. "
            + _and([f"{_word_name(m)}" for m in per_pulse])
            + " are as defined in ISO 18406:2017; SEL_{ss} is over the whole "
            f"{site.extract.pre_s + site.extract.post_s:g}-s pulse window, which "
            f"holds 100 % of the pulse energy, and L_{{p,90}} is the mean-square "
            f"SPL over the pulse duration T_{{90}}.",
            align,
            font_pt=7.5,
        )
        caption = (
            f"Event-level quantities for {when}, for each hydrophone: number of "
            f"valid pulses, and their cumulative sound exposure level SEL_{{cum}}."
        )
        if kurt_e is not None and not kurt_e.is_empty():
            caption += (
                f" Kurtosis β_{{2}} is computed in {k_win:g}-s windows (Martin et "
                f"al. 2020) and summarized over the windows containing strikes."
            )
        if ambient_spl is not None:
            caption += (
                f" Ambient kurtosis is computed in {a.interval_s:g}-s intervals "
                f"over the {a.duration_s / 60:g} min starting "
                f"{a.offset_s / 60:g} min before the event."
            )
        if weather is not None:
            caption += (
                " "
                + weather_note(weather, _word_label)
                + (" (H_{1/3}: significant wave height).")
            )
            if note := station_note(site):
                caption += " " + note
        caption += " Values are medians with the 5th and 95th percentiles in brackets."
        inter_e = _event_excess(inter, event)
        if inter_e is not None and not inter_e.is_empty():
            caption += " " + interpulse_note(site)
        columns, rows, align = event_rows(
            event.name, df_e, ambient_spl, kurt_e, _word_label, str, inter_e
        )
        doc.table(columns, rows, caption, align, font_pt=7.5)

        doc.orientation(landscape=False)
        event_figs = figs / event.name
        rows_text = ", ".join(_word_name(m) for m in cfg.metrics)
        if kurt_e is not None and not kurt_e.is_empty():
            rows_text += f", and kurtosis β_{{2}} in {k_win:g}-s windows"
        figure(
            event_figs / "metrics.png",
            f"Pulse metrics for {when} versus time at {at_range(sensors)}, colored "
            f"by hydrophone (serial number and depth). From top: {rows_text}.",
        )
        figure(
            event_figs / "psd.png",
            f"Power spectral density of the pulses for {when} (red) and of "
            f"ambient noise in {a.interval_s:g}-s intervals before the event (blue): "
            f"median (line) and 5th to 95th percentiles (shaded), at "
            f"{at_range(sensors)} (columns), by hydrophone (rows).",
        )
        window_s = site.extract.pre_s + site.extract.post_s
        figure(
            event_figs / "bands.png",
            f"Decidecade band levels for {when} at {at_range(sensors)} (columns), "
            f"by hydrophone (rows): single-strike sound exposure level "
            f"SEL_{{ss}} in each band (red), and the sound exposure of the ambient "
            f"noise over a window of the same length ({window_s:g} s), from "
            f"{a.interval_s:g}-s intervals before the event (blue). Lines are "
            f"medians and shading the 5th to 95th percentiles; the gap between "
            f"them is the signal-to-noise ratio in each band. Bands are base-10 "
            f"one-third octaves (decidecades, IEC 61260-1).",
        )
        figure(
            event_figs / "levels.png",
            f"Distribution of {_word_label(cfg.histogram_metric)} for {when} at "
            f"{at_range(sensors)}, by hydrophone.",
        )

    path = site.report_dir / "report.docx"
    doc.save(path)
    logger.info(f"[report] {path}")
    return path
