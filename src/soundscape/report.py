"""Figures, LaTeX tables, and a Word document from the workflow outputs.

Figures (data coverage, event metrics, pulse PSD distributions, decidecade
band levels, level histograms, correlation between pulses, metrics versus
range) and tables (event summaries) for any site. The Word document collects them with captions. The
pulse-by-pulse table is a separate PDF (see soundscape.supplement).

Output layout:

    report/
    ├── report.docx
    ├── supplement.pdf                      (soundscape.supplement)
    ├── figures/<event>/{metrics_levels,metrics_other,psd,bands,levels,correlation,
    │                    correlation_density}.png
    ├── figures/<metric>_vs_range.png
    ├── figures/{coverage,interpulse_cdf}.png
    └── tables/{detections,<event>,<event>_event,<event>_correlation}.tex, <event>_bands.csv
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

from soundscape.config import SiteConfig
from soundscape.correlation import RHO_BIN, level_label, read_correlation, stacked
from soundscape.document import Document
from soundscape.events import distance_m, load_events, source_range, too_few_strikes
from soundscape.interpulse import summary_band
from soundscape.metrics import read_table
from soundscape.sources import coverage, recorded_intervals

logger = logging.getLogger(__name__)

DB = "dB re 1 $\\mathrm{\\mu Pa}$"
DB_E = "dB re 1 $\\mathrm{\\mu Pa^2 s}$"

# Axis labels (matplotlib mathtext) and table labels (LaTeX) for each metric,
# named as in ISO 18406:2017 and ISO 18405:2017.
METRIC_LABELS = {
    "Lp_pk": ("$L_{p,\\mathrm{pk}}$", DB),
    "Lp_pkpk": ("$L_{p,\\text{pk-pk}}$", DB),
    "Lp_90": ("$L_{p,90}$", DB),
    "Lp": (
        "$L_{p}$",
        DB,
    ),  # background: mean-square sound pressure level over an interval
    "LEp_ss": ("$L_{E,p,\\mathrm{ss}}$", DB_E),
    "LEp_cum": ("$L_{E,p,\\mathrm{cum}}$", DB_E),
    "tau90": ("$\\tau_{90\\%}$", "ms"),
    "t_r": ("$t_\\mathrm{r}$", "ms"),
    "t_d": ("$t_\\mathrm{d}$", "ms"),
    "beta": ("$\\beta$", ""),
    "SNR": ("SNR", "dB"),
    "WSPD": ("Wind speed", "m/s"),
    "WVHT": ("$H_{1/3}$", "m"),
    # Inter-pulse: mean-square sound pressure level of 100 ms segments in the summary band, and
    # its median excess over the background noise.
    "Lp_ip": ("$L_{p,\\mathrm{ip}}$", DB),
    "dL_ip": ("$\\Delta L_\\mathrm{ip}$", "dB"),
    # Maximum normalized cross-correlation coefficient between two pulses.
    "rho": ("$\\rho_{ij}$", ""),
}

# The same labels for Word (see soundscape.document for the markup).
WORD_LABELS = {
    "Lp_pk": ("L_{p,pk}", "dB re 1 µPa"),
    "Lp_pkpk": ("L_{p,pk-pk}", "dB re 1 µPa"),
    "Lp_90": ("L_{p,90}", "dB re 1 µPa"),
    "Lp": ("L_{p}", "dB re 1 µPa"),
    "LEp_ss": ("L_{E,p,ss}", "dB re 1 µPa^{2} s"),
    "LEp_cum": ("L_{E,p,cum}", "dB re 1 µPa^{2} s"),
    "tau90": ("τ_{90%}", "ms"),
    "t_r": ("t_{r}", "ms"),
    "t_d": ("t_{d}", "ms"),
    "beta": ("β", ""),
    "SNR": ("SNR", "dB"),
    "WSPD": ("Wind speed", "m/s"),
    "WVHT": ("H_{1/3}", "m"),
    "Lp_ip": ("L_{p,ip}", "dB re 1 µPa"),
    "dL_ip": ("ΔL_{ip}", "dB"),
    "rho": ("ρ_{ij}", ""),
}

# Factors from the units in the metrics tables to the units shown, and the
# decimals shown.
SCALE = {"tau90": 1e3, "t_r": 1e3, "t_d": 1e3}
DIGITS = {"tau90": 0, "t_r": 0, "t_d": 0}


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


# ISO 18406:2017, 4.2.4: the least signal-to-noise ratio for measurements.
MIN_SNR_DB = 10.0


def load_metrics(site: SiteConfig) -> pl.DataFrame:
    """Site-wide metrics table, valid pulses only, with the signal-to-noise
    ratio (see `with_snr`)."""
    return with_snr(
        read_table(site.metrics_summary_path).filter(pl.col("valid")),
        background_levels(site.background_path),
    )


def with_snr(df: pl.DataFrame, background: pl.DataFrame) -> pl.DataFrame:
    """Add `SNR` (dB): the broadband signal-to-noise ratio of each pulse
    (ISO 18406:2017, 3.17 and 8.4.1), LEp_ss less the exposure the
    background noise (the median of `background_levels`) gives over the pulse's
    integration time `TW`. Null where the background is unknown."""
    if "TW" not in df.columns or background.is_empty():
        return df.with_columns(pl.lit(None, pl.Float64).alias("SNR"))
    return (
        df.join(
            background.select("event", "sensor", "channel", "Lp_bg"),
            on=["event", "sensor", "channel"],
            how="left",
        )
        .with_columns(
            (pl.col("LEp_ss") - pl.col("Lp_bg") - 10 * pl.col("TW").log10()).alias(
                "SNR"
            )
        )
        .drop("Lp_bg")
    )


def total_sel() -> pl.Expr:
    """Cumulative SEL over all rows in a group, as column `LEp_cum_event`
    (dB re 1 uPa^2 s); the last value of `LEp_cum` for valid pulses."""
    return ((10 ** (pl.col("LEp_ss") / 10)).sum().log10() * 10).alias("LEp_cum_event")


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


def background_psd(path: Path, event: str, sensor: str) -> dict | None:
    """Background noise PSD in dB per channel: `level` over the whole window (NaN where
    it is zero) at `freqs`, and, if the background noise file has them, the median and
    5th and 95th percentiles over valid segments at `segment_freqs` (those
    of the pulse PSDs). None if the window holds no recorded data."""
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
        if "segment_psd_median" in g and g["segment_valid"][:].any(axis=1).all():
            out["segment_freqs"] = g["segment_freqs"][:]
            for k in ("median", "p5", "p95"):
                out[k] = g[f"segment_psd_{k}"][:]
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


def background_band_stats(
    path: Path, event: str, sensor: str, nominal: np.ndarray
) -> dict | None:
    """Background noise in decidecade bands (labeled `nominal`, as for the
    pulses) before an event: median and 5th and 95th percentiles over valid
    segments of the mean-square sound pressure level in each band (`Lp_*`,
    dB re 1 uPa), and of the sound exposure over the segment (`E_*`, dB re
    1 uPa^2 s), whose duration is that of the pulse window. Channels without
    valid segments, and bands the background noise file lacks, are NaN. None without
    segment band levels (rerun the background stage)."""
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        name = f"{event}/{sensor}"
        if name not in f or "segment_band_ms" not in f[name]:
            return None
        g = f[name]
        mean_square = g["segment_band_ms"][:].astype(np.float64)
        valid = g["segment_valid"][:]
        have = list(g["band_nominal"][:])
        segment_s = float(g.attrs["segment_s"])
        channels = [int(c) for c in g.attrs["channels"]]
    n_ch, n_bands = len(channels), len(nominal)
    out = {"channels": channels, "segment_s": segment_s}
    for k in ("median", "p5", "p95"):
        out[f"Lp_{k}"] = np.full((n_ch, n_bands), np.nan)
    cols = [(b, have.index(f)) for b, f in enumerate(nominal) if f in have]
    for c in range(n_ch):
        if valid[c].any() and cols:
            stats = _db_percentiles(mean_square[c][valid[c]][:, [j for _, j in cols]])
            for k in ("median", "p5", "p95"):
                out[f"Lp_{k}"][c, [b for b, _ in cols]] = stats[k]
    for k in ("median", "p5", "p95"):
        out[f"E_{k}"] = out[f"Lp_{k}"] + 10 * np.log10(segment_s)
    return out


def background_levels(path: Path) -> pl.DataFrame:
    """Background noise level per event, sensor, and channel: the median over
    valid background noise segments of the mean-square sound pressure level (`Lp_bg`, dB re 1 uPa),
    with the number of valid segments. Empty without segment levels."""
    rows = []
    if path.exists():
        with h5py.File(path, "r") as f:
            for event, ge in f.items():
                for sensor, g in ge.items():
                    if "segment_Lp" not in g:
                        continue
                    lp, valid = g["segment_Lp"][:], g["segment_valid"][:]
                    for c, channel in enumerate(g.attrs["channels"]):
                        ok = valid[c] & np.isfinite(lp[c])
                        rows.append(
                            {
                                "event": event,
                                "sensor": sensor,
                                "channel": int(channel),
                                "Lp_bg": float(np.median(lp[c][ok]))
                                if ok.any()
                                else None,
                                "n_bg": int(ok.sum()),
                            }
                        )
    schema = {
        "event": pl.String,
        "sensor": pl.String,
        "channel": pl.Int64,
        "Lp_bg": pl.Float64,
        "n_bg": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def band_table(
    df: pl.DataFrame, pulses: dict[str, dict], background: dict[str, dict | None]
) -> pl.DataFrame:
    """Tidy band statistics for one event: per sensor, channel, and band, the
    pulse LEp_ss and background noise L_p and window exposure (median, p5, p95)."""
    info = {
        (r["sensor"], r["channel"]): (r.get("serial_number"), r.get("depth_m"))
        for r in hydrophones(df).iter_rows(named=True)
    }
    rows = []
    for sensor, st in pulses.items():
        bg = background.get(sensor)
        for i, channel in enumerate(st["channels"]):
            j = (
                bg["channels"].index(channel)
                if bg and channel in bg["channels"]
                else None
            )
            for b, nominal in enumerate(st["nominal"]):
                row = {
                    "sensor": sensor,
                    "channel": channel,
                    "serial_number": info.get((sensor, channel), (None, None))[0],
                    "depth_m": info.get((sensor, channel), (None, None))[1],
                    "nominal_Hz": float(nominal),
                    "f_n": float(st["center"][b]),
                }
                for k in ("median", "p5", "p95"):
                    row[f"LEp_ss_{k}"] = float(st[k][i, b])
                for k in ("median", "p5", "p95"):
                    row[f"Lp_bg_{k}"] = (
                        float(bg[f"Lp_{k}"][j, b]) if j is not None else None
                    )
                for k in ("median", "p5", "p95"):
                    row[f"LEp_bg_{k}"] = (
                        float(bg[f"E_{k}"][j, b]) if j is not None else None
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


def metric_groups(metrics: list[str]) -> dict[str, list[str]]:
    """The event metrics split into two figures of at most about five rows,
    so each stays legible at page width: the levels ("levels"), and the
    durations, signal-to-noise ratio, and kurtosis ("other")."""
    levels = [m for m in metrics if m.startswith("L")]
    return {"levels": levels, "other": [m for m in metrics if m not in levels]}


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
    rows = metrics + (["beta"] if kurt is not None and not kurt.is_empty() else [])
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
            if rows[-1] == "beta":
                k = kurt.filter(
                    (pl.col("sensor") == sensor)
                    & (pl.col("channel") == channel)
                    & pl.col("valid")
                ).sort("window_start")
                axs[-1, j].plot(
                    k["window_start"].to_numpy() + half,
                    k["beta"],
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
        if metric == "beta":
            label = f"{label} ({window_s:g} s)"
            for ax in axs[i]:
                # Gaussian noise has beta_2 = 3.
                ax.axhline(3.0, color="0.5", lw=0.6, ls="--")
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
        if metric == "SNR":
            for ax in axs[i]:
                ax.axhline(MIN_SNR_DB, color="0.5", lw=0.6, ls="--")
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
    background: dict[str, dict | None],
    labels: dict[tuple[str, int], str] | None = None,
    fmin: float = 10.0,
) -> plt.Figure:
    """Pulse PSD (median and 5th to 95th percentiles in dB) against background
    noise (the same over pulse-length segments before the event, if
    available): rows are
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
        st, bg = stats[sensor], background.get(sensor)
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
            if bg is not None and channel in bg["channels"]:
                k = bg["channels"].index(channel)
                fa = bg["freqs"]
                if "median" in bg:
                    fa = bg["segment_freqs"]
                    ax.fill_between(
                        fa,
                        bg["p5"][k],
                        bg["p95"][k],
                        color="C0",
                        alpha=0.25,
                        lw=0,
                        label="Background noise, 5th–95th percentile",
                    )
                    ax.plot(
                        fa,
                        bg["median"][k],
                        c="C0",
                        lw=0.8,
                        label="Background noise, median",
                    )
                    lo = min(lo, bg["p5"][k][(fa >= fmin) & (fa <= fmax)].min())
                else:
                    ax.plot(
                        fa, bg["level"][k], c="C0", lw=0.8, label="Background noise"
                    )
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
    fig.supylabel("$L_{p,f}$ (dB re 1 $\\mathrm{\\mu Pa^2/Hz}$)")
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
    background: dict[str, dict | None],
    labels: dict[tuple[str, int], str] | None = None,
    window_s: float = 1.2,
) -> plt.Figure:
    """Decidecade band exposure of the pulses (LEp_ss per band) and of the
    background noise over a window of the same length, each as the median and
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
        st, bg = pulses[sensor], background.get(sensor)
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
            if bg is not None and channel in bg["channels"]:
                k = bg["channels"].index(channel)
                if np.isfinite(bg["E_median"][k]).any():
                    ax.fill_between(
                        f,
                        bg["E_p5"][k],
                        bg["E_p95"][k],
                        color="C0",
                        alpha=0.25,
                        lw=0,
                        step="mid",
                        label="Background noise, 5th–95th percentile",
                    )
                    ax.plot(
                        f,
                        bg["E_median"][k],
                        c="C0",
                        lw=1,
                        marker="s",
                        ms=2.5,
                        label="Background noise, median",
                    )
                    lo = min(lo, np.nanmin(bg["E_p5"][k]))
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
        f"$L_{{E,p}}$ in band over {window_s:g} s (dB re 1 $\\mathrm{{\\mu Pa^2\\,s}}$)"
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


def plot_interpulse_cdf(hist: pl.DataFrame, site_name: str) -> plt.Figure:
    """Cumulative distributions of the 100 ms summary-band levels between
    pulses and in the background noise windows, pooled over events, on each sensor's
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
    styles = {
        "interpulse": ("C0", "Inter-pulse"),
        "background": ("0.5", "Background noise"),
    }
    for ax, sensor in zip(axs[0], sensors):
        for kind, (color, label) in styles.items():
            h = (
                hist.filter(pl.col("sensor") == sensor, pl.col("kind") == kind)
                .group_by("Lp_band")
                .agg(pl.col("count").sum())
                .sort("Lp_band")
            )
            if h.is_empty():
                continue
            cdf = np.cumsum(h["count"].to_numpy()) / h["count"].sum() * 100
            ax.step(h["Lp_band"] + 0.5, cdf, where="post", color=color, label=label)
        ax.axhline(50, color="k", lw=0.4, ls=":")
        ax.set_title(sensor)
        ax.set_xlabel(f"{METRIC_LABELS['Lp_ip'][0]} ({DB})")
        ax.grid(True, lw=0.3)
    axs[0, 0].set_ylabel("Cumulative percentage")
    axs[0, 0].legend(loc="lower right", fontsize="small")
    fig.suptitle(f"{site_name}: 100 ms levels, detection channels")
    return fig


# Lower end of the color scale of rho (as in Jenkins & Lin, Sci. Rep., Fig. 3),
# and upper end of the color scale of the percentage of pairs in each
# RHO_BIN-wide bin of rho.
RHO_MIN = 0.2
RHO_PERCENT_MAX = 20.0


def _hydrophone_grid(n_rows: int, n_cols: int) -> tuple[plt.Figure, np.ndarray]:
    """Panels for each hydrophone (rows) at each sensor (columns)."""
    return plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.3 * n_cols + 0.9, 1.6 * n_rows + 0.7),
        layout="constrained",
        sharex=True,
        sharey=True,
        squeeze=False,
    )


def _panel_label(
    ax: plt.Axes, text: str, top: bool = True, right: bool = False
) -> None:
    ax.text(
        0.97 if right else 0.03,
        0.95 if top else 0.05,
        text,
        transform=ax.transAxes,
        fontsize="x-small",
        va="top" if top else "bottom",
        ha="right" if right else "left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )


# Largest distance (s) from a strike to the edge of its row in the image of
# rho_ij against time, so that pauses in piling stay blank.
ROW_HALF_S = 15.0


def strike_rows(t: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row edges for an image with one row per strike at times `t` (s): each
    row reaches halfway to the neighboring strikes but no more than
    ROW_HALF_S from its own, and a blank (NaN) row fills each pause between
    rows. The first and last strikes, with a neighbor on one side only, reach
    as far on the other. Returns the edges (n_rows + 1) and the image with the
    blank rows."""
    gap = np.minimum(np.diff(t) / 2, ROW_HALF_S)
    if len(gap) == 0:
        gap = np.array([ROW_HALF_S])
    hi = t + np.r_[gap, gap[-1:]]
    lo = t - np.r_[gap[:1], gap]
    blank = np.full(image.shape[1], np.nan, image.dtype)
    edges, rows = [lo[0]], []
    for i in range(len(t)):
        rows.append(image[i])
        edges.append(hi[i])
        if i + 1 < len(t) and lo[i + 1] > hi[i]:
            rows.append(blank)
            edges.append(lo[i + 1])
    return np.array(edges), np.array(rows)


def plot_event_correlation(
    paths: dict[str, Path], sensors: list[tuple[str, float]], max_dt: float, title: str
) -> plt.Figure:
    """rho_ij of each pulse i with every later pulse j within `max_dt`,
    against the time between the strikes, for each hydrophone (rows) at each
    sensor (columns, by range); `paths` are the sensors' correlation files."""
    files = {s: read_correlation(paths[s]) for s, _ in sensors}
    n_rows = max(len(a["channels"]) for _, a in files.values())
    fig, axs = _hydrophone_grid(n_rows, len(sensors))
    mesh = None
    for col, (s, r) in enumerate(sensors):
        data, attrs = files[s]
        axs[0, col].set_title(f"{s} ({r / 1e3:.1f} km)", fontsize="small")
        for row in range(n_rows):
            ax = axs[row, col]
            if row >= len(attrs["channels"]):
                ax.set_visible(False)
                continue
            grid, t, image = stacked(data["corr"][row], data["time"], max_dt)
            y, image = strike_rows(t, image)
            step = grid[1] - grid[0]
            mesh = ax.pcolormesh(
                np.r_[grid - step / 2, grid[-1] + step / 2],
                y / 60,
                image,
                cmap="viridis",
                vmin=RHO_MIN,
                vmax=1.0,
                rasterized=True,
            )
            _panel_label(
                ax,
                hydrophone_label(
                    s,
                    attrs["channels"][row],
                    str(attrs["serial_number"][row]) or None,
                    float(attrs["depth_m"][row]),
                ),
                right=True,
            )
    for ax in axs[-1]:
        ax.set_xlabel("$t_j - t_i$ (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("Time of strike $i$ (min)")
    axs[0, 0].set_xlim(0, max_dt)
    if mesh is not None:
        fig.colorbar(mesh, ax=axs, label=METRIC_LABELS["rho"][0], shrink=0.6)
    fig.suptitle(title)
    return fig


def correlation_density(
    h: pl.DataFrame, bin_s: float, max_dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From histogram counts of one hydrophone (dt_s, rho, count): the edges
    of the time bins, the percentage of the pairs in each time bin that fall
    in each RHO_BIN-wide bin of rho (n_rho, n_dt), and the median of rho in
    each time bin, interpolated linearly within the bin where the cumulative
    fraction reaches one half. Empty time bins are NaN."""
    n_rho = round(1 / RHO_BIN)
    n_dt = int(np.ceil(max_dt / bin_s))
    wide = (
        h.with_columns(
            r=(pl.col("rho") / RHO_BIN).round().cast(pl.Int64),
            b=(pl.col("dt_s") / bin_s - 0.5).round().cast(pl.Int64),
        )
        .group_by("b", "r")
        .agg(pl.col("count").sum())
    )
    counts = np.zeros((n_rho, n_dt))
    counts[wide["r"].to_numpy(), wide["b"].to_numpy()] = wide["count"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        total = counts.sum(axis=0)
        percent = 100 * counts / total
        cdf = np.cumsum(counts, axis=0) / total
        k = np.argmax(cdf >= 0.5, axis=0)
        cols = np.arange(n_dt)
        below = np.where(k > 0, cdf[k - 1, cols], 0.0)
        frac = (0.5 - below) / (cdf[k, cols] - below)
    median = np.where(total > 0, (k + frac) * RHO_BIN, np.nan)
    return np.arange(n_dt + 1) * bin_s, percent, median


def plot_correlation_density(
    hist: pl.DataFrame, title: str, bin_s: float, max_dt: float
) -> plt.Figure:
    """Distribution of rho_ij against the time between strikes, over the
    events in `hist` (one event in the report), for each hydrophone (rows) at
    each sensor (columns, by range), with the median."""
    sensors = (
        hist.group_by("sensor").agg(pl.col("range_m").min()).sort("range_m")["sensor"]
    )
    channels = {
        s: sorted(hist.filter(pl.col("sensor") == s)["channel"].unique().to_list())
        for s in sensors
    }
    n_rows = max(len(c) for c in channels.values())
    fig, axs = _hydrophone_grid(n_rows, len(sensors))
    im = None
    for col, s in enumerate(sensors):
        # The sensor's range from the pile, or its span over the pooled events.
        r = hist.filter(pl.col("sensor") == s)["range_m"] / 1e3
        lo, hi = f"{r.min():.1f}", f"{r.max():.1f}"
        span = lo if lo == hi else f"{lo}–{hi}"
        axs[0, col].set_title(f"{s} ({span} km)", fontsize="small")
        for row in range(n_rows):
            ax = axs[row, col]
            if row >= len(channels[s]):
                ax.set_visible(False)
                continue
            h = hist.filter(
                pl.col("sensor") == s, pl.col("channel") == channels[s][row]
            )
            edges, percent, median = correlation_density(h, bin_s, max_dt)
            im = ax.pcolormesh(
                edges,
                np.arange(round(1 / RHO_BIN) + 1) * RHO_BIN,
                percent,
                cmap="Greys",
                vmin=0,
                vmax=RHO_PERCENT_MAX,
                rasterized=True,
            )
            ax.plot((edges[:-1] + edges[1:]) / 2, median, color="C3", lw=1.0)
            first = h.row(0, named=True)
            _panel_label(
                ax,
                hydrophone_label(
                    s, first["channel"], first["serial_number"], first["depth_m"]
                ),
                top=False,
            )
    for ax in axs[-1]:
        ax.set_xlabel("$t_j - t_i$ (s)")
    for ax in axs[:, 0]:
        ax.set_ylabel(METRIC_LABELS["rho"][0])
    axs[0, 0].set_xlim(0, max_dt)
    axs[0, 0].set_ylim(RHO_MIN, 1.0)
    if im is not None:
        fig.colorbar(im, ax=axs, label="Percentage of pairs", shrink=0.6)
    fig.suptitle(title)
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
    5th and 95th percentiles over valid pulses in brackets. LEp_cum is left
    out (see `event_rows`)."""
    metrics = [m for m in metrics if m != "LEp_cum"]
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
    background_spl: pl.DataFrame | None = None,
    kurt: pl.DataFrame | None = None,
    label: Callable[[str], str] = _tex_label,
    text: Callable[[str], str] = tex_escape,
    inter: pl.DataFrame | None = None,
) -> tuple[list[str], list[list[str]], str]:
    """Columns, rows, and alignment of the event-level quantities for one
    event, one row per hydrophone: serial number, number of valid pulses,
    cumulative SEL over them, the percentage of them whose signal-to-noise
    ratio is below the ISO 18406 minimum of 10 dB (if known), kurtosis in
    windows containing strikes, and the
    background noise kurtosis before the event (median with the 5th and 95th
    percentiles in brackets). Background noise levels are given in decidecade bands
    (see `background_band_stats`), not here. With `inter` (the event's rows of
    `interpulse.excess_summary`), also the inter-pulse level, its median
    excess over the background noise, and the percentage of inter-pulse time more than
    each threshold above the background noise."""
    has_snr = "SNR" in df.columns and df["SNR"].is_not_null().any()
    table = hydrophones(df).join(
        df.group_by("sensor", "channel").agg(
            pl.col("strike").n_unique().alias("pulses"),
            total_sel(),
            *(
                [(pl.col("SNR") < MIN_SNR_DB).mean().alias("low_snr")]
                if has_snr
                else []
            ),
        ),
        on=["sensor", "channel"],
    )
    columns = HYDROPHONE_COLUMNS + ["Pulses", label("LEp_cum")]
    if has_snr:
        columns.append(text(f"SNR < {MIN_SNR_DB:g} dB (%)"))
    extra: list[tuple[str, str]] = []  # (prefix, header)
    if kurt is not None and not kurt.is_empty():
        k = kurt.filter(
            (pl.col("event") == event) & pl.col("valid") & (pl.col("n_strikes") > 0)
        )
        table = table.join(
            k.group_by("sensor", "channel").agg(*_percentiles(pl.col("beta"), "kurt")),
            on=["sensor", "channel"],
            how="left",
        )
        extra.append(("kurt", f"{label('beta')}, piling"))
    if (
        background_spl is not None
        and "beta" in background_spl.columns
        and not background_spl.is_empty()
    ):
        a = background_spl.filter((pl.col("event") == event) & pl.col("valid"))
        table = table.join(
            a.group_by("sensor", "channel").agg(
                *_percentiles(pl.col("beta"), "bgkurt")
            ),
            on=["sensor", "channel"],
            how="left",
        )
        extra.append(("bgkurt", f"Background noise {label('beta')}"))
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
                _num(r["LEp_cum_event"], 1),
            ]
            + (
                [_num(None if r["low_snr"] is None else 100 * r["low_snr"], 0)]
                if has_snr
                else []
            )
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
    return (
        columns,
        rows,
        HYDROPHONE_ALIGN + "rr" + "r" * (has_snr + len(extra) + n_ip),
    )


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
        "Median SNR_{det} (dB)",
    ]
    rows = []
    for event in events:
        for sensor in site.sensors:
            path = site.detections_path(event.name, sensor.name)
            if (
                site.is_excluded(event.name, sensor.name)
                or not path.exists()
                or too_few_strikes(site, event.name, sensor.name)
            ):
                continue
            det = pl.read_csv(path, try_parse_dates=True)
            snr = det["SNR_det"].median() if len(det) else None
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
    """Site-wide inter-pulse summaries written by the interpulse stage; empty
    if not run."""
    paths = {k: site.interpulse_summary_path(k) for k in ("excess", "hist")}
    if not all(p.exists() for p in paths.values()):
        return {}
    out = {k: pl.read_csv(p) for k, p in paths.items()}
    pairs = out["excess"].select("event", "sensor").unique()
    few = [(e, s) for e, s in pairs.iter_rows() if too_few_strikes(site, e, s)]
    if few:
        drop = pl.DataFrame(few, schema=["event", "sensor"], orient="row")
        out = {
            k: v.join(drop, on=["event", "sensor"], how="anti") for k, v in out.items()
        }
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
        f"L_{{p,ip}} is the mean-square sound pressure level from {lo:g} Hz to {hi / 1e3:.1f} kHz in "
        f"{c.segment_s * 1e3:g}-ms segments between strikes (after Guan et al. "
        f"2015), over the inter-pulse interval from t_{{95}} of one pulse to "
        f"t_{{5}} of the next; ΔL_{{ip}} is its median less the median of the same quantity in "
        f"the background noise window before the event, and the last columns give the "
        f"percentage of inter-pulse time more than "
        + _and([f"{t:g} dB" for t in c.thresholds_db])
        + " above that background noise median."
    )


def correlation_summaries(site: SiteConfig) -> dict[str, pl.DataFrame]:
    """Site-wide correlation summaries written by the correlation stage
    ("hist" and "summary"); empty if not run."""
    paths = {k: site.correlation_summary_path(k) for k in ("hist", "summary")}
    if not all(p.exists() for p in paths.values()):
        return {}
    return {
        k: pl.read_csv(p, schema_overrides={"serial_number": pl.String})
        for k, p in paths.items()
    }


def correlation_rows(
    site: SiteConfig, summary: pl.DataFrame, tex: bool = True
) -> tuple[list[str], list[list[str]], str]:
    """Columns, rows, and alignment of one event's correlation summary, one
    row per hydrophone: number of valid pulses, the median rho_ij of pairs
    less than one bin apart (consecutive strikes), and the decorrelation time
    at each level, "> max_dt" where the median stays above the level."""
    c = site.correlation
    text = tex_escape if tex else str
    if tex:
        rho = METRIC_LABELS["rho"][0]
        first = f"Median {rho}, $t_j - t_i < {c.bin_s:g}$ s"
        times = [f"$T_{{{v:g}}}$ (s)" for v in c.levels]
    else:
        first = f"Median ρ_{{ij}}, t_{{j}} − t_{{i}} < {c.bin_s:g} s"
        times = [f"T_{{{v:g}}} (s)" for v in c.levels]
    columns = HYDROPHONE_COLUMNS + ["Pulses", first, *times]
    rows = []
    for r in _by_hydrophone(summary).iter_rows(named=True):
        t = [r[level_label(v)] for v in c.levels]
        rows.append(
            _hydrophone_cells(r, text)
            + [f"{r['n_pulses']:,}", _num(r["rho_first"], 2)]
            + [text(f"> {c.max_dt_s:g}") if np.isinf(v) else _num(v, 0) for v in t]
        )
    return columns, rows, HYDROPHONE_ALIGN + "rr" + "r" * len(c.levels)


def correlation_note(site: SiteConfig) -> str:
    """Caption text (Word) describing how rho_ij is computed."""
    lo, hi = site.correlation.band
    return (
        f"ρ_{{ij}} is the maximum over lag of the normalized cross-correlation "
        f"of pulses i and j on one hydrophone, after both pulses are band-passed "
        f"from {lo:g} Hz to {hi:g} Hz; ρ_{{ij}} = 1 for identical waveforms."
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
    background_spl = (
        read_table(site.background_spl_path)
        if site.background_spl_path.exists()
        else None
    )
    kurt = read_table(site.kurtosis_path) if site.kurtosis_path.exists() else None
    inter = interpulse_summaries(site)
    corr = correlation_summaries(site)
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
        for group, metrics in metric_groups(cfg.metrics).items():
            if not metrics:
                continue
            _save(
                plot_event_metrics(
                    df_e,
                    metrics,
                    title,
                    kurt_e if group == "other" else None,
                    site.kurtosis.window_s,
                ),
                figs / event.name / f"metrics_{group}.png",
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
            background = {
                s: background_psd(site.background_path, event.name, s)
                for s, _ in sensors
            }
            _save(
                plot_pulse_psd(
                    event.name, sensors, stats, background, hydrophone_labels(df_e)
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
            event.name, df_e, background_spl, kurt_e, inter=_event_excess(inter, event)
        )
        (tables / f"{event.name}_event.tex").write_text(
            _tabular(columns, rows, align, weather_note(weather, _tex_label))
        )
        write_event_correlation(site, event, df_e, corr, title, figs, tables)

    if inter:
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

    write_document(site, events, df, background_spl, met, kurt, inter, corr)


def _event_correlation(corr: dict, event) -> pl.DataFrame | None:
    if not corr:
        return None
    return corr["summary"].filter(pl.col("event") == event.name)


def write_event_correlation(
    site: SiteConfig,
    event,
    df_e: pl.DataFrame,
    corr: dict,
    title: str,
    figs: Path,
    tables: Path,
) -> None:
    """Correlation figures (rho_ij of each strike, and its distribution) and
    table for one event."""
    c = site.correlation
    sensors = sensors_by_range(df_e)
    paths = {s: site.correlation_path(event.name, s) for s, _ in sensors}
    sensors = [(s, r) for s, r in sensors if paths[s].exists()]
    if sensors:
        _save(
            plot_event_correlation(paths, sensors, c.max_dt_s, title),
            figs / event.name / "correlation.png",
            site.report.dpi,
            site.report.titles,
        )
    hist = corr["hist"].filter(pl.col("event") == event.name) if corr else None
    if hist is not None and not hist.is_empty():
        _save(
            plot_correlation_density(hist, title, c.bin_s, c.max_dt_s),
            figs / event.name / "correlation_density.png",
            site.report.dpi,
            site.report.titles,
        )
    summary = _event_correlation(corr, event)
    if summary is not None and not summary.is_empty():
        (tables / f"{event.name}_correlation.tex").write_text(
            _tabular(*correlation_rows(site, summary))
        )


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
    background = {
        s: background_band_stats(
            site.background_path, event.name, s, pulses[s]["nominal"]
        )
        for s, _ in sensors
    }
    _save(
        plot_band_levels(
            event.name, sensors, pulses, background, hydrophone_labels(df_e), window_s
        ),
        figs / event.name / "bands.png",
        cfg.dpi,
        cfg.titles,
    )
    band_table(df_e, pulses, background).write_csv(tables / f"{event.name}_bands.csv")


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
    background_spl: pl.DataFrame | None = None,
    met: pl.DataFrame | None = None,
    kurt: pl.DataFrame | None = None,
    inter: dict[str, pl.DataFrame] | None = None,
    corr: dict[str, pl.DataFrame] | None = None,
) -> Path:
    """Word document of the figures (as already written to the report
    directory) and tables, with captions, for pasting into a shared report.

    Event tables are on landscape pages. The pulse-by-pulse table is the
    separate data supplement (`supplement.pdf`).
    """
    cfg = site.report
    figs = site.report_dir / "figures"
    doc = Document(cfg.docx_template)
    a = site.background
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
        f"is the horizontal distance from the pile to the sensor; SNR_{{det}} is the "
        f"detector's signal-to-noise ratio, the band-limited mean-square sound "
        f"pressure of the strike relative to the background noise level in the "
        f"detection band on the sensor's detection channel (see Pile Driving "
        f"Pulse Detection); it differs from the SNR of the pulse metrics.",
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
            figs / "interpulse_cdf.png",
            f"Cumulative distributions of the {site.interpulse.segment_s * 1e3:g}-ms "
            f"levels L_{{p,ip}} ({lo:g} Hz to {hi / 1e3:.1f} kHz) in the inter-pulse "
            f"intervals (from t_{{95}} of one pulse to t_{{5}} of the next) and in "
            f"the background noise windows before each event, pooled over the events at "
            f"{site.name}, on each sensor's detection channel.",
        )

    cc = site.correlation

    per_pulse = [m for m in cfg.metrics if m != "LEp_cum"]
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
            + " are as defined in ISO 18405:2017 and ISO 18406:2017, with the "
            "symbols of ISO 18405; L_{E,p,ss} is the single-strike sound exposure "
            "level (SEL_{ss}), over the pulse's integration time (the "
            f"{site.extract.pre_s + site.extract.post_s:g}-s pulse window, or the "
            f"period between strikes where they are closer), which holds 100% of "
            f"the pulse energy, and L_{{p,90}} is the mean-square sound pressure "
            f"level over the 90% energy signal duration τ_{{90%}}.",
            align,
            font_pt=7.5,
        )
        caption = (
            f"Event-level quantities for {when}, for each hydrophone: number of "
            f"valid pulses, and their cumulative sound exposure level L_{{E,p,cum}} (SEL_{{cum}})."
        )
        if kurt_e is not None and not kurt_e.is_empty():
            caption += (
                f" Kurtosis β is computed in {k_win:g}-s windows (Martin et "
                f"al. 2020) and summarized over the windows containing strikes."
            )
        if background_spl is not None:
            caption += (
                f" Background noise kurtosis is computed in {a.interval_s:g}-s intervals "
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
            event.name, df_e, background_spl, kurt_e, _word_label, str, inter_e
        )
        doc.table(columns, rows, caption, align, font_pt=7.5)

        doc.orientation(landscape=False)
        event_figs = figs / event.name
        groups = metric_groups(cfg.metrics)
        for group, metrics in groups.items():
            if not metrics:
                continue
            rows_text = [_word_name(m) for m in metrics]
            if group == "other" and kurt_e is not None and not kurt_e.is_empty():
                rows_text.append(f"kurtosis β in {k_win:g}-s windows")
            kind = "Levels" if group == "levels" else "Durations and other metrics"
            figure(
                event_figs / f"metrics_{group}.png",
                f"{kind} of the pulses for {when} versus time at "
                f"{at_range(sensors)}, colored by hydrophone (serial number and "
                f"depth). From top: {_and(rows_text)}.",
            )
        figure(
            event_figs / "psd.png",
            f"Mean-square sound pressure spectral density level L_{{p,f}} of the pulses for {when} (red) and of "
            f"background noise in {site.background_segment_s:g}-s segments before the "
            f"event (blue): "
            f"median (line) and 5th to 95th percentiles (shaded), at "
            f"{at_range(sensors)} (columns), by hydrophone (rows).",
        )
        window_s = site.extract.pre_s + site.extract.post_s
        figure(
            event_figs / "bands.png",
            f"Decidecade band levels for {when} at {at_range(sensors)} (columns), "
            f"by hydrophone (rows): single-strike sound exposure level "
            f"L_{{E,p,ss}} in each band (red), and the sound exposure level of the "
            f"background noise in segments of the same length ({window_s:g} s) "
            f"before the event (blue). Lines are "
            f"medians and shading the 5th to 95th percentiles; the gap between "
            f"them is the signal-to-noise ratio in each band. Bands are base-10 "
            f"one-third octaves (decidecades, IEC 61260-1).",
        )
        figure(
            event_figs / "levels.png",
            f"Distribution of {_word_label(cfg.histogram_metric)} for {when} at "
            f"{at_range(sensors)}, by hydrophone.",
        )
        corr_e = _event_correlation(corr, event)
        if corr_e is not None and not corr_e.is_empty():
            figure(
                event_figs / "correlation.png",
                f"Maximum normalized cross-correlation coefficient ρ_{{ij}} "
                f"between each pulse i of {when} and every later pulse j no more "
                f"than {cc.max_dt_s:g} s after it, against the time between the "
                f"strikes t_{{j}} − t_{{i}} (horizontal) and the time of strike i "
                f"(vertical), at {at_range(sensors)} (columns), by hydrophone "
                f"(rows). Blank rows are pauses in piling and pulses that are "
                f"not valid on the hydrophone. " + correlation_note(site),
            )
            figure(
                event_figs / "correlation_density.png",
                f"Distribution of ρ_{{ij}} against the time between strikes "
                f"t_{{j}} − t_{{i}}, over every pair of pulses of {when} no more "
                f"than {cc.max_dt_s:g} s apart, at {at_range(sensors)} "
                f"(columns), by hydrophone (rows). The shading is the percentage "
                f"of the pairs in each {cc.bin_s:g}-s bin of t_{{j}} − t_{{i}} "
                f"whose ρ_{{ij}} falls in each {RHO_BIN:g}-wide bin of "
                f"ρ_{{ij}}; the red line is the median.",
            )
            columns, rows, align = correlation_rows(site, corr_e, tex=False)
            doc.table(
                columns,
                rows,
                f"Correlation between pulses for {when}, for each hydrophone: the "
                f"number of valid pulses; the median of ρ_{{ij}} over pairs of "
                f"strikes less than {cc.bin_s:g} s apart, which are consecutive "
                f"strikes during full-energy piling; and T_{{x}}, the time between "
                f"strikes at which the median of ρ_{{ij}}, in {cc.bin_s:g}-s bins "
                f"of t_{{j}} − t_{{i}}, first falls below x. A value of "
                f"“> {cc.max_dt_s:g}” means that the median stays at or "
                f"above x out to {cc.max_dt_s:g} s. " + correlation_note(site),
                align,
            )

    path = site.report_dir / "report.docx"
    doc.save(path)
    logger.info(f"[report] {path}")
    return path
