"""Figures, LaTeX tables, and a Word document from the workflow outputs.

Figures (data coverage, event metrics, pulse PSD distributions, channel level
histograms, metrics versus range) and tables (event summaries, pulse-by-pulse
appendix) for any site. The Word document collects the figures and tables
(except the pulse-by-pulse table) with captions.

Output layout:

    report/
    ├── report.docx
    ├── figures/<event>/{metrics,psd,levels}.png
    ├── figures/<metric>_vs_range.png
    ├── figures/coverage.png
    └── tables/{detections,pulses,<event>}.tex
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

from soundscape.config import SiteConfig
from soundscape.document import Document
from soundscape.events import load_events, source_range
from soundscape.metrics import read_table
from soundscape.sources import coverage, recorded_intervals

logger = logging.getLogger(__name__)

# Axis labels (matplotlib mathtext) and table labels (LaTeX) for each metric.
METRIC_LABELS = {
    "SPL_rms": ("$L_{p,\\mathrm{rms}}$", "dB re 1 $\\mathrm{\\mu Pa}$"),
    "SPL_rms90": ("$L_{p,\\mathrm{rms},90}$", "dB re 1 $\\mathrm{\\mu Pa}$"),
    "SPL_pk": ("$L_{p,\\mathrm{pk}}$", "dB re 1 $\\mathrm{\\mu Pa}$"),
    "SPL_pkpk": ("$L_{p,\\text{pk-pk}}$", "dB re 1 $\\mathrm{\\mu Pa}$"),
    "SEL": ("$L_{E,p}$", "dB re 1 $\\mathrm{\\mu Pa^2 s}$"),
    "SEL_cum": ("cSEL", "dB re 1 $\\mathrm{\\mu Pa^2 s}$"),
    "T90": ("$T_{90}$", "s"),
    "t_rise": ("$t_\\mathrm{r}$", "s"),
    "t_decay": ("$t_\\mathrm{d}$", "s"),
    "kurtosis": ("$\\beta_2$", ""),
    "WSPD": ("Wind speed", "m/s"),
    "WVHT": ("$H_{1/3}$", "m"),
}

# The same labels for Word (see soundscape.document for the markup).
WORD_LABELS = {
    "SPL_rms": ("L_{p,rms}", "dB re 1 µPa"),
    "SPL_rms90": ("L_{p,rms,90}", "dB re 1 µPa"),
    "SPL_pk": ("L_{p,pk}", "dB re 1 µPa"),
    "SPL_pkpk": ("L_{p,pk-pk}", "dB re 1 µPa"),
    "SEL": ("L_{E,p}", "dB re 1 µPa^{2} s"),
    "SEL_cum": ("cSEL", "dB re 1 µPa^{2} s"),
    "T90": ("T_{90}", "s"),
    "t_rise": ("t_{r}", "s"),
    "t_decay": ("t_{d}", "s"),
    "kurtosis": ("β_{2}", ""),
    "WSPD": ("Wind speed", "m/s"),
    "WVHT": ("H_{1/3}", "m"),
}


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
    return ((10 ** (pl.col("SEL") / 10)).sum().log10() * 10).alias("SEL_total")


def sensors_by_range(df: pl.DataFrame) -> list[tuple[str, float]]:
    """(sensor, range in m) for one event, nearest first."""
    rows = df.group_by("sensor").agg(pl.col("range_m").first()).sort("range_m")
    return list(rows.iter_rows())


def pulse_psd_stats(path: Path) -> dict:
    """Mean and standard deviation of pulse PSDs in dB, per channel."""
    with h5py.File(path, "r") as f:
        freqs = f["freqs"][:]
        valid = f["valid"][:]
        channels = [int(c) for c in f.attrs["channels"]]
        dset = f["psd"]
        mean, std = [], []
        for c in range(dset.shape[1]):
            level = 10 * np.log10(dset[:, c, :][valid].astype(np.float64))
            mean.append(level.mean(axis=0))
            std.append(level.std(axis=0))
    return {
        "freqs": freqs,
        "channels": channels,
        "mean": np.array(mean),
        "std": np.array(std),
        "n": int(valid.sum()),
    }


def ambient_psd(path: Path, event: str, sensor: str) -> dict | None:
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        name = f"{event}/{sensor}"
        if name not in f:
            return None
        g = f[name]
        return {
            "freqs": g["freqs"][:],
            "channels": [int(c) for c in g.attrs["channels"]],
            "level": 10 * np.log10(g["psd"][:]),
        }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def channel_colors(df: pl.DataFrame) -> dict[int, str]:
    """One color per channel number, consistent across sensors and figures."""
    channels = sorted(df["channel"].unique().to_list())
    return {c: f"C{k}" for k, c in enumerate(channels)}


def _channel_legend(fig: plt.Figure, colors: dict[int, str], **kwargs) -> None:
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=4, color=color, label=f"Ch. {c}")
        for c, color in colors.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), **kwargs)


def plot_event_metrics(df: pl.DataFrame, metrics: list[str], title: str) -> plt.Figure:
    """Metric time series for one event: rows are metrics, columns sensors
    (nearest first), points colored by channel."""
    sensors = sensors_by_range(df)
    colors = channel_colors(df)
    fig, axs = plt.subplots(
        nrows=len(metrics),
        ncols=len(sensors),
        figsize=(2.4 * len(sensors) + 1.2, 1.35 * len(metrics) + 1.0),
        sharex=True,
        sharey="row",
        squeeze=False,
        layout="constrained",
    )
    for j, (sensor, range_m) in enumerate(sensors):
        df_s = df.filter(pl.col("sensor") == sensor)
        axs[0, j].set_title(f"{sensor}, R = {range_m / 1e3:.1f} km")
        for (channel,), df_c in df_s.sort("channel").group_by(
            "channel", maintain_order=True
        ):
            for i, metric in enumerate(metrics):
                axs[i, j].scatter(
                    df_c["time"],
                    df_c[metric],
                    s=0.5,
                    alpha=0.5,
                    color=colors[channel],
                    rasterized=True,
                )
    for i, metric in enumerate(metrics):
        axs[i, 0].set_ylabel(_axis_label(metric))
        for ax in axs[i]:
            ax.grid(True, lw=0.3)
    for ax in axs[-1]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=2, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time (UTC)")
    _channel_legend(fig, colors)
    fig.suptitle(title)
    return fig


def plot_pulse_psd(
    event: str,
    sensors: list[tuple[str, float]],
    stats: dict[str, dict],
    ambient: dict[str, dict | None],
    fmin: float = 10.0,
) -> plt.Figure:
    """Pulse PSD (mean ± 2σ in dB) against ambient noise: rows are channels,
    columns sensors (nearest first)."""
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
        axs[0, j].set_title(f"{sensor}, R = {range_m / 1e3:.1f} km\n{st['n']:,} pulses")
        f = st["freqs"]
        band = (f >= fmin) & (f <= fmax)
        n_ch = len(st["channels"])
        for i in range(n_rows):
            ax = axs[i, j]
            if i >= n_ch:
                ax.set_axis_off()
                continue
            channel = st["channels"][i]
            lower = st["mean"][i] - 2 * st["std"][i]
            upper = st["mean"][i] + 2 * st["std"][i]
            lo, hi = min(lo, lower[band].min()), max(hi, upper[band].max())
            ax.fill_between(
                f, lower, upper, color="C3", alpha=0.25, lw=0, label="Pulses, mean ± 2σ"
            )
            ax.plot(f, st["mean"][i], c="C3", lw=1, label="Pulses, mean")
            if amb is not None and channel in amb["channels"]:
                k = amb["channels"].index(channel)
                ax.plot(amb["freqs"], amb["level"][k], c="C0", lw=0.8, label="Ambient")
            ax.text(0.03, 0.05, f"Ch. {channel}", transform=ax.transAxes)
            ax.grid(True, which="both", lw=0.3)
            if i == n_ch - 1:
                # The bottom panel of a short column carries its own x labels.
                ax.xaxis.set_tick_params(labelbottom=True)
                ax.set_xlabel("Frequency (Hz)")
    for row in axs:
        # Label the first visible panel in each row (a short column may be empty).
        first = next(ax for ax in row if ax.axison)
        first.yaxis.set_tick_params(labelleft=True)
        first.set_ylabel("PSD (dB re 1 $\\mathrm{\\mu Pa^2/Hz}$)")
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_xlim(fmin, fmax)
    pad = 0.05 * (hi - lo)
    axs[0, 0].set_ylim(lo - pad, hi + pad)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(f"Event {event}")
    return fig


def plot_level_histograms(df: pl.DataFrame, metric: str, title: str) -> plt.Figure:
    """Distribution of a metric per channel, one panel per sensor."""
    sensors = sensors_by_range(df)
    colors = channel_colors(df)
    fig, axs = plt.subplots(
        nrows=len(sensors),
        figsize=(5.0, 1.3 * len(sensors) + 1.0),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    bins = np.histogram_bin_edges(df[metric].drop_nans().to_numpy(), bins=60)
    for ax, (sensor, range_m) in zip(axs[:, 0], sensors):
        df_s = df.filter(pl.col("sensor") == sensor)
        for (channel,), df_c in df_s.sort("channel").group_by(
            "channel", maintain_order=True
        ):
            ax.hist(
                df_c[metric].to_numpy(),
                bins=bins,
                density=True,
                histtype="step",
                color=colors[channel],
            )
        ax.text(
            0.02, 0.85, f"{sensor}, R = {range_m / 1e3:.1f} km", transform=ax.transAxes
        )
        ax.set_ylabel("Density")
        ax.grid(True, lw=0.3)
    axs[-1, 0].set_xlabel(_axis_label(metric).replace("\n", " "))
    _channel_legend(fig, colors)
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


def _fmt(values, digits: int = 1) -> str:
    """One value per channel, separated by slashes."""
    return "/".join(_num(v, digits) for v in values)


def _tabular(columns: list[str], rows: list[list[str]], align: str) -> str:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\midrule")
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def event_table(
    event: str,
    df: pl.DataFrame,
    metrics: list[str],
    ambient_spl: pl.DataFrame | None = None,
    weather: dict[str, float] | None = None,
) -> str:
    """Summary statistics for one event (LaTeX tabular); see `event_rows`."""
    columns, rows = event_rows(
        event, df, metrics, ambient_spl, weather, _tex_label, tex_escape
    )
    return _tabular(columns, rows, EVENT_ALIGN)


EVENT_ALIGN = "llrrrrr"


def event_rows(
    event: str,
    df: pl.DataFrame,
    metrics: list[str],
    ambient_spl: pl.DataFrame | None = None,
    weather: dict[str, float | None] | None = None,
    label: Callable[[str], str] = _tex_label,
    text: Callable[[str], str] = tex_escape,
) -> tuple[list[str], list[list[str]]]:
    """Columns and rows of the summary statistics for one event.

    Each cell lists one value per channel, separated by slashes, in channel
    order. Ambient SPL is the mean over valid intervals before the event.
    `label` names a metric and `text` escapes other text for the output format.
    """
    sensors = sensors_by_range(df)
    columns = ["Metric", "Sensor", "Mean", "Median", "Std", "Min", "Max"]
    rows = []
    if "depth_m" in df.columns:
        for name, col, fmt in (
            ("Hydrophone depth (m)", "depth_m", lambda v: _num(v, 1)),
            ("Hydrophone S/N", "serial_number", lambda v: v or "--"),
        ):
            for k, (sensor, _) in enumerate(sensors):
                per_ch = (
                    df.filter(pl.col("sensor") == sensor)
                    .group_by("channel")
                    .agg(pl.col(col).first())
                    .sort("channel")
                )
                rows.append(
                    [name if k == 0 else "", text(sensor)]
                    + ["/".join(fmt(v) for v in per_ch[col].to_list())]
                    + [""] * 4
                )
    for metric in metrics:
        digits = 1 if metric.startswith(("SPL", "SEL")) else 2
        if metric == "SEL_cum":
            # Statistics of a running sum mean little; give the event total.
            for k, (sensor, _) in enumerate(sensors):
                total = (
                    df.filter(pl.col("sensor") == sensor)
                    .group_by("channel")
                    .agg(total_sel())
                    .sort("channel")
                )
                rows.append(
                    [f"Event total {label(metric)}" if k == 0 else "", text(sensor)]
                    + [_fmt(total["SEL_total"].to_list(), digits)]
                    + [""] * 4
                )
            continue
        for k, (sensor, _) in enumerate(sensors):
            by_ch = (
                df.filter(pl.col("sensor") == sensor)
                .group_by("channel")
                .agg(
                    pl.col(metric).mean().alias("mean"),
                    pl.col(metric).median().alias("median"),
                    pl.col(metric).std().alias("std"),
                    pl.col(metric).min().alias("min"),
                    pl.col(metric).max().alias("max"),
                )
                .sort("channel")
            )
            rows.append(
                [label(metric) if k == 0 else "", text(sensor)]
                + [
                    _fmt(by_ch[c].to_list(), digits)
                    for c in ("mean", "median", "std", "min", "max")
                ]
            )
    if ambient_spl is not None:
        name = f"Ambient {label('SPL_rms')}"
        for sensor, _ in sensors:
            amb = (
                ambient_spl.filter(
                    (pl.col("event") == event)
                    & (pl.col("sensor") == sensor)
                    & pl.col("valid")
                )
                .group_by("channel")
                .agg(pl.col("SPL_rms").mean())
                .sort("channel")
            )
            if not amb.is_empty():
                rows.append(
                    [
                        name,
                        text(sensor),
                        _fmt(amb["SPL_rms"].to_list()),
                    ]
                    + [""] * 4
                )
                name = ""
    for key, value in (weather or {}).items():
        rows.append(
            [label(key), "", "--" if value is None else f"{value:.1f}"] + [""] * 4
        )
    return columns, rows


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


def pulses_longtable(df: pl.DataFrame, metrics: list[str]) -> str:
    """Every pulse and channel (LaTeX longtable, needs the longtable,
    booktabs, and makecell packages)."""
    columns = ["Event", "Sensor", "Ch.", "Depth (m)", "R (km)", "Strike"]
    columns += ["Time (UTC)"] + [_tex_label(m, stacked=True) for m in metrics]
    header = " & ".join(columns) + " \\\\"
    lines = [
        f"\\begin{{longtable}}{{lllrrrl{'r' * len(metrics)}}}",
        "\\toprule",
        header,
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header,
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{len(columns)}}}{{r}}{{Continued on next page}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    df = df.sort("event", "sensor", "channel", "strike").with_columns(
        pl.col("time").dt.strftime("%Y-%m-%d %H:%M:%S%.3f").alias("time_str")
    )
    for r in df.iter_rows(named=True):
        cells = [
            tex_escape(r["event"]),
            tex_escape(r["sensor"]),
            str(r["channel"]),
            _num(r.get("depth_m"), 1),
            f"{r['range_m'] / 1e3:.2f}",
            str(r["strike"]),
            r["time_str"],
        ] + [_num(r[m], 2) for m in metrics]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


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
    ambient_spl = (
        read_table(site.ambient_spl_path) if site.ambient_spl_path.exists() else None
    )
    met = pl.read_csv(cfg.met, try_parse_dates=True) if cfg.met is not None else None

    for event in events:
        df_e = df.filter(pl.col("event") == event.name)
        if df_e.is_empty():
            logger.warning(f"[report] {event.name}: no metrics")
            continue
        title = f"Event {event.name} (pile {event.pile}), {_window(event)} UTC"
        _save(
            plot_event_metrics(df_e, cfg.metrics, title),
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
                plot_pulse_psd(event.name, sensors, stats, ambient),
                figs / event.name / "psd.png",
                cfg.dpi,
                cfg.titles,
            )

        weather = (
            event_weather(met, event.start, event.end) if met is not None else None
        )
        (tables / f"{event.name}.tex").write_text(
            event_table(event.name, df_e, cfg.metrics, ambient_spl, weather)
        )

    for metric in cfg.range_metrics:
        _save(
            plot_metric_vs_range(detection_channels(site, df), metric, site.name),
            figs / f"{metric}_vs_range.png",
            cfg.dpi,
            cfg.titles,
        )

    (tables / "detections.tex").write_text(detections_table(site, events))
    (tables / "pulses.tex").write_text(pulses_longtable(df, cfg.metrics))
    logger.info(f"[report] tables -> {tables}")

    write_document(site, events, df, ambient_spl, met)


def detection_channels(site: SiteConfig, df: pl.DataFrame) -> pl.DataFrame:
    """Rows of `df` on each sensor's detection channel."""
    detect_ch = {s.name: s.detect_channel for s in site.sensors}
    return df.filter(
        pl.col("channel") == pl.col("sensor").replace_strict(detect_ch, default=None)
    )


def write_document(
    site: SiteConfig,
    events: list,
    df: pl.DataFrame,
    ambient_spl: pl.DataFrame | None = None,
    met: pl.DataFrame | None = None,
) -> Path:
    """Word document of the figures (as already written to the report
    directory) and tables, with captions, for pasting into a shared report.

    Event tables are on landscape pages. The pulse-by-pulse table is left out;
    `metrics.csv` serves as the data supplement.
    """
    cfg = site.report
    figs = site.report_dir / "figures"
    doc = Document(cfg.docx_template)

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
        f"(all channels) in {n_events} event{'s' * (n_events != 1)}. "
        f"Where a sensor has several channels, a table cell lists one value "
        f"per channel, in channel order, separated by slashes."
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
            f"{WORD_LABELS.get(metric, (metric,))[0]} versus range for all "
            f"events at {site.name}, on each sensor's detection channel. Points "
            f"are individual pulses colored by strike number; black markers are "
            f"the median and interquartile range for each event and sensor.",
        )

    for event in events:
        df_e = df.filter(pl.col("event") == event.name)
        if df_e.is_empty():
            continue
        sensors = sensors_by_range(df_e)
        n = {
            s: df_e.filter(pl.col("sensor") == s)["strike"].n_unique()
            for s, _ in sensors
        }
        weather = (
            event_weather(met, event.start, event.end) if met is not None else None
        )

        doc.orientation(landscape=True)
        doc.heading(f"Event {event.name}", 2)
        caption = (
            f"Statistics of pulse metrics for event {event.name} (pile "
            f"{event.pile}), {_window(event)} UTC, from "
            + _and([f"{n[s]:,} pulses at {s} ({r / 1e3:.1f} km)" for s, r in sensors])
            + "."
        )
        if ambient_spl is not None:
            a = site.ambient
            caption += (
                f" Ambient L_{{p,rms}} is the mean over {a.interval_s:g}-s "
                f"intervals in the {a.duration_s / 60:g} min starting "
                f"{a.offset_s / 60:g} min before the event."
            )
        if weather is not None:
            caption += (
                " Wind speed and significant wave height H_{1/3} are means over "
                "the event."
            )
        columns, rows = event_rows(
            event.name, df_e, cfg.metrics, ambient_spl, weather, _word_label, str
        )
        doc.table(columns, rows, caption, EVENT_ALIGN, font_pt=7.5)

        doc.orientation(landscape=False)
        event_figs = figs / event.name
        figure(
            event_figs / "metrics.png",
            f"Pulse metrics for event {event.name} versus time at "
            f"{at_range(sensors)}, colored by channel. From top: "
            + ", ".join(WORD_LABELS.get(m, (m,))[0] for m in cfg.metrics)
            + ".",
        )
        figure(
            event_figs / "psd.png",
            f"Power spectral density of the pulses for event {event.name} "
            f"(red: mean ± 2σ in dB) and of ambient noise before the event "
            f"(blue) at {at_range(sensors)} (columns), by channel (rows).",
        )
        figure(
            event_figs / "levels.png",
            f"Distribution of {_word_label(cfg.histogram_metric)} for event "
            f"{event.name} at {at_range(sensors)}, by channel.",
        )

    path = site.report_dir / "report.docx"
    doc.save(path)
    logger.info(f"[report] {path}")
    return path
