"""Figures and tables comparing sites.

Each site contributes its valid pulses from events it does not exclude.
Levels are compared on each sensor's detection channel, except the depth
comparison, which uses every channel of sensors with hydrophones at more than
one depth.

Output layout:

    <output>/
    ├── compare.docx
    ├── figures/{levels_vs_range,depth,psd}.png
    └── tables/summary.{csv,tex}
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter, ScalarFormatter

from soundscape.config import SiteConfig
from soundscape.document import Document
from soundscape.events import Event, load_events
from soundscape.metrics import read_table
from soundscape.report import (
    METRIC_LABELS,
    WORD_LABELS,
    _and,
    _axis_label,
    _num,
    _save,
    _tabular,
    _tex_label,
    _word_label,
    ambient_psd,
    detection_channels,
    load_metrics,
    pulse_psd_stats,
    tex_escape,
    total_sel,
)

logger = logging.getLogger(__name__)

MARKERS = ["o", "s", "^", "D", "v"]
RANGE_TICKS = [0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100]


@dataclass
class SiteData:
    site: SiteConfig
    events: list[Event]
    df: pl.DataFrame  # valid pulses, all channels, with a `site` column
    ambient_spl: pl.DataFrame | None


def load_site_data(site: SiteConfig) -> SiteData | None:
    if not site.metrics_summary_path.exists():
        logger.warning(f"[compare] missing {site.metrics_summary_path}; skipped")
        return None
    events = sorted(load_events(site), key=lambda e: e.start)
    df = (
        load_metrics(site)
        .filter(pl.col("event").is_in([e.name for e in events]))
        .with_columns(pl.lit(site.name).alias("site"))
    )
    ambient = (
        read_table(site.ambient_spl_path) if site.ambient_spl_path.exists() else None
    )
    return SiteData(site, events, df, ambient)


def _range_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    lo, hi = ax.get_xlim()
    ax.set_xticks([t for t in RANGE_TICKS if lo <= t <= hi])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summary(data: list[SiteData], metrics: list[str]) -> pl.DataFrame:
    """One row per site, event, and sensor on the detection channel: range,
    hydrophone depth, pulses, median of each metric, and event total
    SEL_cum."""
    frames = []
    for d in data:
        order = {e.name: k for k, e in enumerate(d.events)}
        det = detection_channels(d.site, d.df)
        rows = det.group_by("site", "event", "pile", "sensor", "channel").agg(
            (pl.col("range_m").first() / 1e3).alias("range_km"),
            pl.col("serial_number").first(),
            pl.col("depth_m").first(),
            pl.col("strike").n_unique().alias("pulses"),
            *[pl.col(m).median().alias(m) for m in metrics],
            total_sel(),
        )
        frames.append(
            rows.with_columns(
                pl.col("event").replace_strict(order).alias("_order")
            ).sort("_order", "range_km")
        )
    return pl.concat(frames, how="diagonal_relaxed").drop("_order")


def summary_columns(metrics: list[str], label) -> list[tuple[str, str, str]]:
    """(column, header, alignment) of the summary table."""
    return (
        [
            ("site", "Site", "l"),
            ("event", "Event", "l"),
            ("sensor", "Sensor", "l"),
            ("range_km", "Range (km)", "r"),
            ("serial_number", "Serial", "l"),
            ("depth_m", "Depth (m)", "r"),
            ("pulses", "Pulses", "r"),
        ]
        + [(m, f"Median {label(m)}", "r") for m in metrics]
        + [
            ("SEL_total", f"Event total {label('SEL_cum')}", "r"),
        ]
    )


def summary_rows(table: pl.DataFrame, columns, text) -> list[list[str]]:
    digits = {"range_km": 1, "depth_m": 1}
    rows = []
    for r in table.iter_rows(named=True):
        cells = []
        for col, _, _ in columns:
            v = r[col]
            if col == "pulses":
                cells.append(f"{v:,}")
            elif isinstance(v, str):
                cells.append(text(v))
            else:
                cells.append(_num(v, digits.get(col, 1)))
        rows.append(cells)
    return rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def range_fit(range_km: np.ndarray, level: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of level = a - b log10(R / 1 km); returns (a, b)."""
    slope, a = np.polyfit(np.log10(range_km), level, 1)
    return a, -slope


def plot_levels_vs_range(
    data: list[SiteData], metrics: list[str]
) -> tuple[plt.Figure, dict]:
    """Median and interquartile range of each metric per event and sensor
    against range, one panel per metric plus the event total cSEL, with a fit
    of a - b log10(R) per site for each per-pulse metric. Returns the figure
    and the fits {(site, metric): (a, b)}."""
    panels = metrics + ["SEL_total"]
    ncols = 2
    nrows = -(-len(panels) // ncols)
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(6.5, 2.4 * nrows + 0.6),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    fits = {}
    for k, d in enumerate(data):
        det = detection_channels(d.site, d.df)
        stats = det.group_by("event", "sensor").agg(
            (pl.col("range_m").first() / 1e3).alias("range_km"),
            *[pl.col(m).median().alias(m) for m in metrics],
            *[pl.col(m).quantile(0.25).alias(f"{m}_q25") for m in metrics],
            *[pl.col(m).quantile(0.75).alias(f"{m}_q75") for m in metrics],
            total_sel(),
        )
        r = stats["range_km"].to_numpy()
        style = {
            "fmt": MARKERS[k % len(MARKERS)],
            "color": f"C{k}",
            "ms": 4,
            "lw": 0.8,
            "capsize": 0,
        }
        for ax, m in zip(axs.flat, panels):
            y = stats[m].to_numpy()
            if m == "SEL_total":
                ax.errorbar(r, y, label=d.site.name, **style)
                continue
            yerr = np.vstack(
                [y - stats[f"{m}_q25"].to_numpy(), stats[f"{m}_q75"].to_numpy() - y]
            )
            ax.errorbar(r, y, yerr=yerr, label=d.site.name, **style)
            if len(np.unique(r)) > 1:
                a, b = range_fit(r, y)
                fits[(d.site.name, m)] = (a, b)
                rr = np.geomspace(r.min(), r.max(), 50)
                ax.plot(rr, a - b * np.log10(rr), color=f"C{k}", lw=1, ls="--")
    for ax, m in zip(axs.flat, panels):
        name = "SEL_cum" if m == "SEL_total" else m
        label = _axis_label(name)
        if m == "SEL_total":
            label = "Event total " + label
        ax.set_ylabel(label)
        ax.grid(True, which="both", lw=0.3)
    for ax in axs.flat[len(panels) :]:
        ax.set_axis_off()
    for ax in axs[-1]:
        ax.set_xlabel("Range (km)")
    for ax in axs.flat[: len(panels)]:
        _range_axis(ax)
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(data))
    fig.suptitle("Received levels on detection channels, by site")
    return fig, fits


def depth_profiles(data: list[SiteData], metric: str) -> pl.DataFrame:
    """Median of `metric` on each channel relative to the mean over the
    sensor's channels, for each site, event, and sensor whose hydrophones are
    at more than one depth."""
    frames = []
    for d in data:
        med = d.df.group_by("site", "event", "sensor", "channel").agg(
            pl.col("depth_m").first(), pl.col(metric).median().alias("level")
        )
        frames.append(
            med.with_columns(
                (
                    pl.col("level") - pl.col("level").mean().over("event", "sensor")
                ).alias("relative"),
                pl.col("depth_m").n_unique().over("event", "sensor").alias("n_depths"),
            ).filter(pl.col("n_depths") > 1)
        )
    return pl.concat(frames).sort("site", "event", "sensor", "depth_m")


def plot_depth(data: list[SiteData], metrics: list[str]) -> plt.Figure | None:
    """Level relative to the sensor mean against hydrophone depth: thin lines
    for each event and sensor, thick for the site mean at each depth (to the
    nearest meter)."""
    profiles = {m: depth_profiles(data, m) for m in metrics}
    if all(p.is_empty() for p in profiles.values()):
        return None
    fig, axs = plt.subplots(
        1,
        len(metrics),
        figsize=(2.2 * len(metrics) + 0.6, 3.2),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for ax, m in zip(axs[0], metrics):
        p = profiles[m]
        for k, d in enumerate(data):
            ps = p.filter(pl.col("site") == d.site.name)
            if ps.is_empty():
                continue
            for _, g in ps.group_by("event", "sensor"):
                ax.plot(g["relative"], g["depth_m"], color=f"C{k}", lw=0.5, alpha=0.4)
            mean = (
                ps.group_by(pl.col("depth_m").round(0).alias("depth"))
                .agg(pl.col("relative").mean())
                .sort("depth")
            )
            ax.plot(
                mean["relative"],
                mean["depth"],
                color=f"C{k}",
                lw=2,
                marker=MARKERS[k % len(MARKERS)],
                ms=4,
                label=d.site.name,
            )
        ax.axvline(0, color="0.5", lw=0.5)
        ax.set_xlabel(f"Relative {METRIC_LABELS.get(m, (m,))[0]} (dB)")
        ax.grid(True, lw=0.3)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_ylabel("Hydrophone depth (m)")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(data))
    fig.suptitle("Depth dependence of received level")
    return fig


def plot_psd(data: list[SiteData], fmin: float = 10.0) -> plt.Figure | None:
    """Median pulse PSD on the detection channel for each event and sensor,
    colored by range, one panel per site, with the ambient PSD before each
    event in gray."""
    curves = []  # (site index, range km, freqs, mean dB)
    ambient = []  # (site index, freqs, level dB)
    for k, d in enumerate(data):
        det_ch = {s.name: s.detect_channel for s in d.site.sensors}
        pairs = d.df.group_by("event", "sensor").agg(pl.col("range_m").first())
        for event, sensor, range_m in pairs.iter_rows():
            path = d.site.pulse_psd_path(event, sensor)
            if not path.exists():
                continue
            st = pulse_psd_stats(path)
            if det_ch[sensor] not in st["channels"]:
                continue
            i = st["channels"].index(det_ch[sensor])
            curves.append((k, range_m / 1e3, st["freqs"], st["median"][i]))
            amb = ambient_psd(d.site.ambient_path, event, sensor)
            if amb is not None and det_ch[sensor] in amb["channels"]:
                j = amb["channels"].index(det_ch[sensor])
                level = amb["median"][j] if "median" in amb else amb["level"][j]
                ambient.append((k, amb["freqs"], level))
    if not curves:
        return None
    norm = Normalize(min(c[1] for c in curves), max(c[1] for c in curves))
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(
        1,
        len(data),
        figsize=(3.2 * len(data) + 0.8, 3.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    fmax = min(c[2][-1] for c in curves)
    lo, hi = np.inf, -np.inf
    for k, f, level in ambient:
        axs[0, k].plot(f, level, color="0.6", lw=0.6)
    for k, r, f, level in sorted(curves, key=lambda c: -c[1]):
        axs[0, k].plot(f, level, color=cmap(norm(r)), lw=0.8)
    for f, level in [(c[1], c[2]) for c in ambient] + [(c[2], c[3]) for c in curves]:
        band = (f >= fmin) & (f <= fmax) & np.isfinite(level)
        if band.any():
            lo, hi = min(lo, level[band].min()), max(hi, level[band].max())
    for k, d in enumerate(data):
        axs[0, k].set_title(d.site.name)
        axs[0, k].set_xlabel("Frequency (Hz)")
        axs[0, k].grid(True, which="both", lw=0.3)
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_xlim(fmin, fmax)
    pad = 0.05 * (hi - lo)
    axs[0, 0].set_ylim(lo - pad, hi + pad)
    axs[0, 0].set_ylabel("PSD (dB re 1 $\\mathrm{\\mu Pa^2/Hz}$)")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0, -1], label="Range (km)"
    )
    fig.suptitle("Median pulse spectra on detection channels")
    return fig


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def compare_sites(sites: list[SiteConfig], output: Path) -> None:
    """Write the cross-site figures, tables, and Word document to `output`.
    Figure style, metrics, and Word template come from the first site."""
    cfg = sites[0].report
    if cfg.style is not None:
        plt.style.use(cfg.style)
    data = [d for s in sites if (d := load_site_data(s)) is not None]
    if not data:
        logger.warning("[compare] no site has metrics")
        return
    figs = output / "figures"
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    metrics = cfg.range_metrics
    names = _and([d.site.name for d in data])

    table = summary(data, metrics)
    table.write_csv(tables / "summary.csv")
    tex_cols = summary_columns(metrics, lambda m: _tex_label(m, stacked=True))
    (tables / "summary.tex").write_text(
        _tabular(
            [h for _, h, _ in tex_cols],
            summary_rows(table, tex_cols, tex_escape),
            "".join(a for _, _, a in tex_cols),
        )
    )

    fig, fits = plot_levels_vs_range(data, metrics)
    _save(fig, figs / "levels_vs_range.png", cfg.dpi, cfg.titles)
    fits_table = pl.DataFrame(
        [
            {"site": s, "metric": m, "a_dB": a, "b_dB_per_decade": b}
            for (s, m), (a, b) in fits.items()
        ]
    )
    if not fits_table.is_empty():
        fits_table.write_csv(tables / "range_fits.csv")
    depth = plot_depth(data, metrics)
    if depth is not None:
        _save(depth, figs / "depth.png", cfg.dpi, cfg.titles)
    psd = plot_psd(data)
    if psd is not None:
        _save(psd, figs / "psd.png", cfg.dpi, cfg.titles)
    else:
        logger.warning("[compare] no pulse PSD files; psd figure skipped")

    # Word document ---------------------------------------------------------
    doc = Document(cfg.docx_template)
    doc.heading(f"Comparison of {names}", 1)
    doc.paragraph(
        f"Figures and tables generated by soundscape compare on "
        f"{date.today():%Y-%m-%d} from the valid pulses of "
        + _and([f"{d.df['event'].n_unique()} events at {d.site.name}" for d in data])
        + ". Levels are on each sensor's detection channel unless stated."
    )

    word_cols = summary_columns(metrics, _word_label)
    doc.orientation(landscape=True)
    doc.heading("Summary by event and sensor", 2)
    doc.table(
        [h for _, h, _ in word_cols],
        summary_rows(table, word_cols, str),
        "Received levels for each event and sensor on the sensor's detection "
        "channel: range from the pile, hydrophone depth, number of pulses, median "
        "per-pulse levels, and the cumulative sound exposure level SEL_{cum} over "
        "all valid pulses of the event.",
        "".join(a for _, _, a in word_cols),
        font_pt=7.5,
    )
    doc.orientation(landscape=False)

    doc.heading("Received levels versus range", 2)
    fit_text = "; ".join(
        f"{s}, {WORD_LABELS.get(m, (m,))[0]}: {a:.1f} − {b:.1f} log_{{10}} R"
        for (s, m), (a, b) in fits.items()
    )
    doc.figure(
        figs / "levels_vs_range.png",
        "Median and interquartile range of per-pulse levels for each event and "
        "sensor against range, and cumulative SEL over each event, by site. "
        + (
            "Dashed lines are least-squares fits of a − b log_{10} R (R in km) to "
            f"the medians of each site ({fit_text})."
            if fits
            else ""
        ),
    )
    if depth is not None:
        doc.heading("Depth dependence", 2)
        doc.figure(
            figs / "depth.png",
            "Median level on each hydrophone relative to the mean over the "
            "sensor's hydrophones, against hydrophone depth, for sensors with "
            "hydrophones at more than one depth. Thin lines are single events "
            "and sensors; thick lines are site means at each depth (to the "
            "nearest meter).",
        )
    if psd is not None:
        doc.heading("Pulse spectra", 2)
        doc.figure(
            figs / "psd.png",
            "Median power spectral density of the pulses on the detection channel "
            "for each event and sensor, colored by range, with the ambient noise "
            "spectrum before each event in gray.",
        )
    path = output / "compare.docx"
    doc.save(path)
    logger.info(f"[compare] {path}")
