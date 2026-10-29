"""Figures and tables comparing sites.

Each site contributes its valid pulses from events it does not exclude.
Levels are compared on each sensor's detection channel, except the depth
comparison, which uses every channel of sensors with hydrophones at more than
one depth.

Output layout:

    <output>/
    ├── compare.docx
    ├── figures/{levels_vs_range,depth,psd,background_vs_range,interpulse_vs_range,
    │            interpulse_bands,masking_bands,correlation_vs_range}.png
    └── tables/{summary.{csv,tex},range_fits.csv,background_fits.csv,masking.csv,
                masking_bands.csv,correlation_vs_range.csv}
"""

import logging
from dataclasses import dataclass, field
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
from soundscape.interpulse import masking_factors
from soundscape.metrics import read_table
from soundscape.report import (
    METRIC_LABELS,
    RHO_MIN,
    WORD_LABELS,
    _and,
    _axis_label,
    _num,
    _save,
    _stat,
    _tabular,
    _tex_label,
    _word_label,
    background_psd,
    detection_channels,
    interpulse_band,
    interpulse_summaries,
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
    background_spl: pl.DataFrame | None
    interpulse: dict[str, pl.DataFrame]  # see report.interpulse_summaries
    # "summary" and "bins" of the correlation stage (see correlation.summarize)
    correlation: dict[str, pl.DataFrame] = field(default_factory=dict)


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
    background = (
        read_table(site.background_spl_path)
        if site.background_spl_path.exists()
        else None
    )
    inter = interpulse_summaries(site)
    if inter and site.interpulse_summary_path("bands").exists():
        inter["bands"] = pl.read_csv(site.interpulse_summary_path("bands"))
    names = [e.name for e in events]
    inter = {k: v.filter(pl.col("event").is_in(names)) for k, v in inter.items()}
    corr = {
        k: pl.read_csv(p, schema_overrides={"serial_number": pl.String}).filter(
            pl.col("event").is_in(names)
        )
        for k in ("summary", "bins")
        if (p := site.correlation_summary_path(k)).exists()
    }
    return SiteData(site, events, df, background, inter, corr)


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
    LEp_cum."""
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
            ("LEp_cum_event", f"Event total {label('LEp_cum')}", "r"),
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
    against range, one panel per metric plus the event total LEp_cum, with a fit
    of a - b log10(R) per site for each per-pulse metric. Returns the figure
    and the fits {(site, metric): (a, b)}."""
    panels = metrics + ["LEp_cum_event"]
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
            if m == "LEp_cum_event":
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
        name = "LEp_cum" if m == "LEp_cum_event" else m
        label = _axis_label(name)
        if m == "LEp_cum_event":
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
    colored by range, one panel per site, with the background noise PSD before each
    event in gray."""
    curves = []  # (site index, range km, freqs, mean dB)
    background = []  # (site index, freqs, level dB)
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
            bg = background_psd(d.site.background_path, event, sensor)
            if bg is not None and det_ch[sensor] in bg["channels"]:
                j = bg["channels"].index(det_ch[sensor])
                if "median" in bg:
                    background.append((k, bg["segment_freqs"], bg["median"][j]))
                else:
                    background.append((k, bg["freqs"], bg["level"][j]))
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
    for k, f, level in background:
        axs[0, k].plot(f, level, color="0.6", lw=0.6)
    for k, r, f, level in sorted(curves, key=lambda c: -c[1]):
        axs[0, k].plot(f, level, color=cmap(norm(r)), lw=0.8)
    for f, level in [(c[1], c[2]) for c in background] + [(c[2], c[3]) for c in curves]:
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
    axs[0, 0].set_ylabel("$L_{p,f}$ (dB re 1 $\\mathrm{\\mu Pa^2/Hz}$)")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[0, -1], label="Range (km)"
    )
    fig.suptitle("Median pulse spectra on detection channels")
    return fig


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Inter-pulse sound field
# --------------------------------------------------------------------------


def _inter(data: list[SiteData], kind: str) -> pl.DataFrame:
    """One inter-pulse summary on the detection channels, over sites."""
    frames = [
        detection_channels(d.site, d.interpulse[kind]).with_columns(
            site=pl.lit(d.site.name)
        )
        for d in data
        if kind in d.interpulse
    ]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def background_vs_range(data: list[SiteData]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Pre-piling background noise level against range from the pile.

    The level is the median over the background noise window before each event of the
    100 ms mean-square sound pressure level in the inter-pulse summary band, on each sensor's
    detection channel (from the interpulse stage). Returns the points (site,
    event, sensor, range_km, level) and least-squares fits of
    level = a - b log10(R / 1 km) per site (sensor "all") and per site and
    sensor, with the correlation coefficient r of level with log10 R.
    """
    pts = _inter(data, "excess")
    if pts.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    pts = pts.select(
        "site",
        "event",
        "sensor",
        (pl.col("range_m") / 1e3).alias("range_km"),
        pl.col("Lp_bg").alias("level"),
    ).sort("site", "range_km")
    fits = []
    groups = [
        ((s, "all"), g) for (s,), g in pts.group_by(["site"], maintain_order=True)
    ]
    groups += [
        ((s, sensor), g)
        for (s, sensor), g in pts.group_by(["site", "sensor"], maintain_order=True)
    ]
    for (site, sensor), g in groups:
        r_km, level = g["range_km"].to_numpy(), g["level"].to_numpy()
        if len(np.unique(r_km)) < 3:
            continue
        a, b = range_fit(r_km, level)
        fits.append(
            {
                "site": site,
                "sensor": sensor,
                "n": len(g),
                "range_min_km": r_km.min(),
                "range_max_km": r_km.max(),
                "a_dB": a,
                "b_dB_per_decade": b,
                "r": float(np.corrcoef(np.log10(r_km), level)[0, 1]),
            }
        )
    return pts, pl.DataFrame(fits)


def plot_background_vs_range(pts: pl.DataFrame, fits: pl.DataFrame) -> plt.Figure:
    """Pre-piling background noise level per event and sensor against range, colored
    by site with a marker per sensor, and the fit for each site."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
    sensors = pts["sensor"].unique(maintain_order=True).to_list()
    for k, (site,) in enumerate(
        pts.select("site").unique(maintain_order=True).iter_rows()
    ):
        d = pts.filter(pl.col("site") == site)
        for sensor, g in d.group_by("sensor", maintain_order=True):
            ax.plot(
                g["range_km"],
                g["level"],
                MARKERS[sensors.index(sensor[0]) % len(MARKERS)],
                ms=4,
                color=f"C{k}",
                mfc="none",
                label=f"{site}, {sensor[0]}",
            )
        f = (
            fits.filter(pl.col("site") == site, pl.col("sensor") == "all")
            if not fits.is_empty()
            else fits
        )
        if not f.is_empty():
            fit = f.row(0, named=True)
            rr = np.geomspace(d["range_km"].min(), d["range_km"].max(), 50)
            ax.plot(
                rr,
                fit["a_dB"] - fit["b_dB_per_decade"] * np.log10(rr),
                color=f"C{k}",
                lw=1,
                ls="--",
                label=f"{site} fit: {-fit['b_dB_per_decade']:+.0f} dB/decade, "
                f"r = {fit['r']:.2f}",
            )
    ax.set_xlabel("Range from pile (km)")
    ax.set_ylabel(f"Pre-piling background noise {METRIC_LABELS['Lp'][0]} (dB)")
    _range_axis(ax)
    ax.grid(True, which="both", lw=0.3)
    ax.legend(fontsize="x-small", ncol=1, loc="upper right")
    return fig


def plot_interpulse_vs_range(
    data: list[SiteData], thresholds: list[float]
) -> plt.Figure | None:
    """Median excess of the inter-pulse level over the background noise, and the
    percentage of inter-pulse time above each threshold, against range, for
    each event and sensor."""
    ex = _inter(data, "excess")
    if ex.is_empty():
        return None
    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.2), layout="constrained")
    for k, (site, d) in enumerate(ex.group_by(["site"], maintain_order=True)):
        r = d["range_m"] / 1e3
        axs[0].plot(r, d["dL"], MARKERS[k], ms=4, color=f"C{k}", label=site[0])
        for j, t in enumerate(thresholds):
            axs[1].plot(
                r,
                100 * d[f"above_{t:g}"],
                MARKERS[k],
                ms=4,
                color=f"C{k}",
                mfc="none" if j else None,
                label=f"{site[0]}, > {t:g} dB",
            )
    axs[0].axhline(0, color="k", lw=0.6)
    axs[0].set_ylabel(f"{METRIC_LABELS['dL_ip'][0]} (dB)")
    axs[1].set_ylabel("Inter-pulse time above background noise (%)")
    axs[1].set_ylim(0, 102)
    for ax in axs:
        ax.set_xlabel("Range (km)")
        _range_axis(ax)
        ax.grid(True, lw=0.3)
        ax.legend(fontsize="small")
    return fig


# Times between strikes (s) at which the correlation is compared against range,
# with their labels, after the consecutive strikes.
CORRELATION_DT = {60.0: "1 min", 300.0: "5 min"}


def correlation_vs_range(data: list[SiteData]) -> pl.DataFrame:
    """Median rho_ij against range for each event and sensor on the detection
    channel: of consecutive strikes (`rho_first`), and in the bin of time
    between strikes nearest each of CORRELATION_DT (`rho_<label>`, e.g.
    `rho_1min`), with the decorrelation times `T_<x>`."""
    frames = []
    for d in data:
        if not {"summary", "bins"} <= d.correlation.keys():
            continue
        s = detection_channels(d.site, d.correlation["summary"])
        b = detection_channels(d.site, d.correlation["bins"])
        centers = b["dt_s"].unique().sort().to_numpy()
        for t, label in CORRELATION_DT.items():
            c = centers[np.argmin(np.abs(centers - t))]
            at = b.filter(pl.col("dt_s") == c).select(
                "event", "sensor", pl.col("rho_median").alias(_rho_column(label))
            )
            s = s.join(at, on=["event", "sensor"], how="left")
        frames.append(
            s.with_columns(
                site=pl.lit(d.site.name), range_km=pl.col("range_m") / 1e3
            ).drop("range_m")
        )
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed")
    first = ["site", "event", "sensor", "channel", "depth_m", "serial_number"]
    first += ["range_km", "n_pulses", "rho_first"]
    first += [_rho_column(v) for v in CORRELATION_DT.values()]
    return out.select(first + [c for c in out.columns if c not in first]).sort(
        "site", "range_km"
    )


def _rho_column(label: str) -> str:
    return "rho_" + label.replace(" ", "")


def plot_correlation_vs_range(pts: pl.DataFrame) -> plt.Figure:
    """Median rho_ij of consecutive strikes and at each of CORRELATION_DT
    against range, one panel each, colored by site."""
    panels = [("rho_first", "Consecutive strikes")] + [
        (_rho_column(v), f"{v} between strikes") for v in CORRELATION_DT.values()
    ]
    fig, axs = plt.subplots(
        1, len(panels), figsize=(8.5, 3.0), layout="constrained", sharey=True
    )
    sites = pts["site"].unique(maintain_order=True).to_list()
    for ax, (col, title) in zip(axs, panels):
        for k, site in enumerate(sites):
            d = pts.filter(pl.col("site") == site, pl.col(col).is_not_null())
            ax.plot(d["range_km"], d[col], MARKERS[k], ms=4, color=f"C{k}", label=site)
        ax.set_title(title, fontsize="small")
        ax.set_xlabel("Range (km)")
        _range_axis(ax)
        ax.grid(True, which="both", lw=0.3)
        ax.legend(fontsize="x-small", loc="lower left")
    axs[0].set_ylabel(f"Median {METRIC_LABELS['rho'][0]}")
    axs[0].set_ylim(RHO_MIN, 1.0)
    return fig


def plot_interpulse_bands(data: list[SiteData]) -> plt.Figure | None:
    """Median excess over the background noise in each decidecade band between pulses,
    per sensor (median over events), one panel per site."""
    b = _inter(data, "bands")
    if b.is_empty():
        return None
    sites = b["site"].unique(maintain_order=True).to_list()
    fig, axs = plt.subplots(
        1,
        len(sites),
        figsize=(3.6 * len(sites), 3.0),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    for ax, site in zip(axs[0], sites):
        d = b.filter(pl.col("site") == site)
        order = d.group_by("sensor").agg(pl.col("range_m").median()).sort("range_m")
        for k, (sensor, rng) in enumerate(order.iter_rows()):
            m = (
                d.filter(pl.col("sensor") == sensor)
                .group_by("nominal_Hz")
                .agg(pl.col("dL").median())
                .sort("nominal_Hz")
            )
            ax.plot(
                m["nominal_Hz"],
                m["dL"],
                marker=".",
                color=f"C{k}",
                label=f"{sensor} ({rng / 1e3:.0f} km)",
            )
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Decidecade band (Hz)")
        ax.set_title(site)
        ax.grid(True, lw=0.3)
        ax.legend(fontsize="small")
    axs[0, 0].set_ylabel("Inter-pulse level above background noise (dB)")
    return fig


def masking_table(data: list[SiteData]) -> pl.DataFrame:
    """Communication space remaining between strikes, per site and sensor: the
    median over events of the inter-pulse excess on the detection channel,
    and the detection range and communication space remaining at the site's spreading
    coefficient (see interpulse.masking_factors)."""
    rows = []
    for d in data:
        if "excess" not in d.interpulse:
            continue
        ex = detection_channels(d.site, d.interpulse["excess"])
        for (sensor,), g in ex.group_by(["sensor"], maintain_order=True):
            dl = float(g["dL"].median())
            row = {
                "site": d.site.name,
                "sensor": sensor,
                "range_min_km": g["range_m"].min() / 1e3,
                "range_max_km": g["range_m"].max() / 1e3,
                "events": g["event"].n_unique(),
                "dL_median": dl,
                "dL_min": float(g["dL"].min()),
                "dL_max": float(g["dL"].max()),
            }
            r, a = masking_factors(dl, d.site.interpulse.spreading)
            row["range_left"], row["area_left"] = float(r), float(a)
            rows.append(row)
    return pl.DataFrame(rows).sort("site", "range_min_km") if rows else pl.DataFrame()


def masking_bands_table(data: list[SiteData]) -> pl.DataFrame:
    """Communication space remaining between strikes in each decidecade band, per
    site and sensor: the median over events of the band's inter-pulse excess
    on the detection channel (as in plot_interpulse_bands), and the detection
    range and area remaining at the site's spreading coefficient."""
    frames = []
    for d in data:
        if "bands" not in d.interpulse or d.interpulse["bands"].is_empty():
            continue
        b = detection_channels(d.site, d.interpulse["bands"])
        t = (
            b.with_columns(
                (pl.col("range_m") / 1e3).alias("range_km"),
            )
            .with_columns(
                pl.col("range_km").min().over("sensor").alias("range_min_km"),
                pl.col("range_km").max().over("sensor").alias("range_max_km"),
            )
            .group_by("sensor", "range_min_km", "range_max_km", "nominal_Hz")
            .agg(
                pl.col("event").n_unique().alias("events"),
                pl.col("dL").median().alias("dL_median"),
                pl.col("dL").min().alias("dL_min"),
                pl.col("dL").max().alias("dL_max"),
            )
        )
        r, a = masking_factors(t["dL_median"].to_numpy(), d.site.interpulse.spreading)
        t = t.with_columns(pl.Series("range_left", r), pl.Series("area_left", a))
        frames.append(t.with_columns(site=pl.lit(d.site.name)))
    if not frames:
        return pl.DataFrame()
    columns = ["site", "sensor", "range_min_km", "range_max_km", "nominal_Hz"]
    t = pl.concat(frames, how="diagonal_relaxed")
    return t.select(columns + [c for c in t.columns if c not in columns]).sort(
        "site", "range_min_km", "nominal_Hz"
    )


def plot_masking_bands(table: pl.DataFrame) -> plt.Figure:
    """Detection range remaining between strikes against decidecade band, per
    sensor, one panel per site."""
    sites = table["site"].unique(maintain_order=True).to_list()
    fig, axs = plt.subplots(
        1,
        len(sites),
        figsize=(3.6 * len(sites), 3.0),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )
    for ax, site in zip(axs[0], sites):
        d = table.filter(pl.col("site") == site)
        order = (
            d.group_by("sensor")
            .agg(pl.col("range_min_km").first())
            .sort("range_min_km")
        )
        for k, (sensor, _) in enumerate(order.iter_rows()):
            g = d.filter(pl.col("sensor") == sensor).sort("nominal_Hz")
            r0, r1 = g["range_min_km"][0], g["range_max_km"][0]
            ax.plot(
                g["nominal_Hz"],
                100 * g["range_left"],
                marker=".",
                color=f"C{k}",
                label=f"{sensor} ({r0:.0f}–{r1:.0f} km)",
            )
        ax.set_xscale("log")
        ax.set_ylim(0, 102)
        ax.set_xlabel("Decidecade band (Hz)")
        ax.set_title(site)
        ax.grid(True, lw=0.3)
        ax.legend(fontsize="small", loc="lower right")
    axs[0, 0].set_ylabel("Detection range remaining (%)")
    return fig


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
    bg_pts, bg_fits = background_vs_range(data)
    if not bg_pts.is_empty():
        _save(
            plot_background_vs_range(bg_pts, bg_fits),
            figs / "background_vs_range.png",
            cfg.dpi,
            cfg.titles,
        )
        bg_fits.write_csv(tables / "background_fits.csv")
    ip = sites[0].interpulse
    ip_range = plot_interpulse_vs_range(data, ip.thresholds_db)
    if ip_range is not None:
        _save(ip_range, figs / "interpulse_vs_range.png", cfg.dpi, cfg.titles)
    ip_bands = plot_interpulse_bands(data)
    if ip_bands is not None:
        _save(ip_bands, figs / "interpulse_bands.png", cfg.dpi, cfg.titles)
    masking = masking_table(data)
    if not masking.is_empty():
        masking.write_csv(tables / "masking.csv")
    corr = correlation_vs_range(data)
    if not corr.is_empty():
        corr.write_csv(tables / "correlation_vs_range.csv")
        _save(
            plot_correlation_vs_range(corr),
            figs / "correlation_vs_range.png",
            cfg.dpi,
            cfg.titles,
        )
    masking_bands = masking_bands_table(data)
    if not masking_bands.is_empty():
        masking_bands.write_csv(tables / "masking_bands.csv")
        _save(
            plot_masking_bands(masking_bands),
            figs / "masking_bands.png",
            cfg.dpi,
            cfg.titles,
        )

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
        "per-pulse levels, and the cumulative sound exposure level L_{E,p,cum} over "
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
            "Median mean-square sound pressure spectral density level L_{p,f} of the pulses on the detection channel "
            "for each event and sensor, colored by range, with the background noise "
            "spectrum before each event in gray.",
        )
    if not bg_pts.is_empty():
        _background_section(doc, data, figs, bg_fits)
    if ip_range is not None:
        _interpulse_section(
            doc, data, figs, masking, ip_bands is not None, not masking_bands.is_empty()
        )
    if not corr.is_empty():
        _correlation_section(doc, data, figs)
    path = output / "compare.docx"
    doc.save(path)
    logger.info(f"[compare] {path}")


def _interpulse_section(
    doc: Document,
    data: list[SiteData],
    figs: Path,
    masking: pl.DataFrame,
    bands: bool,
    masking_bands: bool,
) -> None:
    """Inter-pulse figures, the communication space table, and the detection range
    by decidecade band in compare.docx."""
    site = data[0].site
    ip = site.interpulse
    lo, hi = interpulse_band(site)
    band = f"{lo:g} Hz to {hi / 1e3:.1f} kHz"
    doc.heading("Inter-pulse sound field", 2)
    doc.figure(
        figs / "interpulse_vs_range.png",
        f"Sound between strikes against range, for each event and sensor on the "
        f"detection channel. Left: ΔL_{{ip}}, the median mean-square sound pressure level ({band}) "
        f"of the {ip.segment_s * 1e3:g}-ms segments in the inter-pulse interval "
        f"(from t_{{95}} of one pulse to t_{{5}} of the next) less the median of "
        f"the same quantity in the background noise window before the event. Right: the "
        f"percentage of inter-pulse time more than "
        + _and([f"{t:g} dB" for t in ip.thresholds_db])
        + " (filled and open markers) above that background noise median.",
    )
    if bands:
        doc.figure(
            figs / "interpulse_bands.png",
            "Median excess of the inter-pulse level over the background noise in each "
            "decidecade band, per sensor (median over events), on the detection "
            "channel. Band levels are from the DFT of each "
            f"{ip.segment_s * 1e3:g}-ms segment (10 Hz resolution), so bands "
            "below about 50 Hz are approximate.",
        )
    if masking.is_empty():
        return
    columns = [
        "Site",
        "Sensor",
        "Range (km)",
        "Events",
        "ΔL_{ip} (dB)",
        "Range left (%)",
        "Area left (%)",
    ]
    rows = [
        [
            r["site"],
            r["sensor"],
            f"{r['range_min_km']:.1f}–{r['range_max_km']:.1f}",
            str(r["events"]),
            _stat(r["dL_median"], r["dL_min"], r["dL_max"], 1),
            _num(100 * r["range_left"], 0),
            _num(100 * r["area_left"], 0),
        ]
        for r in masking.iter_rows(named=True)
    ]
    doc.table(
        columns,
        rows,
        "Communication space remaining between strikes for low-frequency cetaceans "
        f"({band}), per site and sensor on the detection channel. ΔL_{{ip}} is "
        "the median over events of the inter-pulse excess over the background noise, with the "
        "smallest and largest event medians in brackets. With cylindrical "
        f"spreading, transmission loss {ip.spreading:g} log_{{10}} r, a rise ΔL "
        "in the noise scales the range at which a call can be detected by "
        f"10^{{−ΔL/{ip.spreading:g}}} and the communication space of Clark et "
        "al. (2009), the area within which the call can be detected, by the "
        "square of that. "
        "This is a first-order estimate: it treats the median "
        "excess as steady and ignores critical ratios and call structure. The "
        "summary band combines all frequencies; the reduction in each "
        "decidecade band is in the next figure.",
        "llrrrrr",
    )
    if masking_bands:
        doc.figure(
            figs / "masking_bands.png",
            "Range at which a call can be detected between strikes, as a "
            "percentage of its value in the background noise, in each decidecade "
            "band, per sensor on the detection channel: "
            f"10^{{−ΔL/{ip.spreading:g}}} for cylindrical spreading, with ΔL the "
            "median over events of the band's inter-pulse excess (the previous "
            "figure). The area remaining is the square of the range remaining. "
            "Bands below about 50 Hz are approximate (see above).",
        )


def _correlation_section(doc: Document, data: list[SiteData], figs: Path) -> None:
    """Correlation between pulses against range in compare.docx."""
    c = data[0].site.correlation
    lo, hi = c.band
    times = _and([f"about {v}" for v in CORRELATION_DT.values()])
    doc.heading("Correlation between pulses versus range", 2)
    doc.figure(
        figs / "correlation_vs_range.png",
        f"Median maximum normalized cross-correlation coefficient ρ_{{ij}} "
        f"between pulses ({lo:g} Hz to {hi:g} Hz) against range, for each event "
        f"and sensor on the detection channel: for consecutive strikes (pairs "
        f"less than {c.bin_s:g} s apart), and for strikes {times} apart (the "
        f"{c.bin_s:g}-s bin of time between strikes nearest each).",
    )


def _background_section(
    doc: Document, data: list[SiteData], figs: Path, fits: pl.DataFrame
) -> None:
    """Pre-piling background noise level against range in compare.docx."""
    site = data[0].site
    a = site.background
    lo, hi = interpulse_band(site)
    doc.heading("Pre-piling background noise versus range", 2)
    doc.figure(
        figs / "background_vs_range.png",
        f"Background noise before each event against range from the pile: the median "
        f"of the {site.interpulse.segment_s * 1e3:g}-ms mean-square sound pressure level L_{{p}} "
        f"({lo:g} Hz to {hi / 1e3:.1f} kHz) over the {a.duration_s / 60:g} min "
        f"starting {a.offset_s / 60:g} min before the event, on each sensor's "
        f"detection channel. Dashed lines are least-squares fits of a − b "
        f"log_{{10}} R (R in km) for each site, with the correlation coefficient "
        f"r of the level with log_{{10}} R.",
    )
    if fits.is_empty():
        return
    rows = [
        [
            r["site"],
            "All" if r["sensor"] == "all" else r["sensor"],
            str(r["n"]),
            f"{r['range_min_km']:.1f}–{r['range_max_km']:.1f}",
            _num(r["a_dB"], 1),
            _num(-r["b_dB_per_decade"], 1),
            _num(r["r"], 2),
        ]
        for r in fits.iter_rows(named=True)
    ]
    doc.table(
        ["Site", "Sensor", "Events", "Range (km)", "a (dB)", "Slope (dB/decade)", "r"],
        rows,
        "Fits of the pre-piling background noise level to a − b log_{10} R for each site, "
        "over all sensors and for each sensor: the level at 1 km (a), the change "
        "per decade of range (−b), and the correlation coefficient r of the level "
        "with log_{10} R.",
        "llrrrrr",
    )
