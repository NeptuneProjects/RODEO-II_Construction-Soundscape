"""Sound between strikes: the inter-pulse sound field (Optional Task 2.4.4).

Following Guan et al. (2015, 2017), the level is computed by the incremental
computation method: the mean-square sound pressure level in consecutive 100 ms segments,

    L_p = 10 log10(mean p^2 over the segment / p0^2)    dB re 1 uPa

starting at each strike's detection time and running to the next strike, on
every hydrophone. Each segment is tagged with its time since the strike and
with whether it lies in the inter-pulse interval: from t95 of the pulse to t5
of the next on that hydrophone (the energy-based pulse duration, see
soundscape.metrics), or, for comparison, Guan et al.'s interval from 100 ms
after one strike to 500 ms before the next. Levels are broadband (above the
10 Hz high-pass), in a summary band (by default 20 Hz to the top of the
decidecade bands, which covers the low-frequency cetacean hearing group
within the recorded bandwidth), and in each decidecade band, all from the DFT
of the segment (10 Hz resolution for 100 ms, so bands below about 50 Hz,
narrower than a bin, are approximate).

Each interval is summarized by the median over its segments, not by its
mean square: the interval holds the last 5 % of one pulse's energy and the
first 5 % of the next, which would otherwise dominate it. The same segments
over the background noise window before each event give the background noise distribution,
computed the same way so that the two compare like for like.

Guan, S., Vignola, J., Judge, J., & Turo, D. (2015). Airgun inter-pulse noise
field during a seismic survey in an Arctic ultra shallow marine environment.
JASA 138(6), 3447-3457.

Guan, S., Southall, B. L., Barlow, J., Vignola, J. F., Judge, J. A., & Turo, D.
(2017). Inter-ping sound field from a simulated mid-frequency active sonar,
and its implication to marine mammal tonal masking. POMA 27, 070023.
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl
from tritonoa.data.reader import NoDataError

from soundscape.bands import band_sum, band_weights, decidecade_bands, exposure_spectrum
from soundscape.config import (
    BackgroundConfig,
    BandsConfig,
    InterpulseConfig,
    SensorConfig,
)
from soundscape.events import Event, source_range, too_few_strikes
from soundscape.extract import apply_filter
from soundscape.metrics import _db_power
from soundscape.sources import read_segment
from soundscape.spectra import background_window

logger = logging.getLogger(__name__)

# Data read on each side of a chunk so filter transients fall outside it.
_PAD_S = 1.0

# A segment with more than this fraction of zero-filled samples is a gap.
_GAP_FRACTION = 0.01

# Segments per DFT batch.
_BATCH = 2048


def band_label(nominal: float) -> str:
    """Column name of a decidecade band mean-square sound pressure level
    L_p,n, e.g. `Lp_12.5Hz` or `Lp_1000Hz`."""
    return f"Lp_{nominal:g}Hz"


def is_band_label(column: str) -> bool:
    return column.startswith("Lp_") and column.endswith("Hz")


def summary_band(
    config: InterpulseConfig, bands: BandsConfig, fs: float
) -> tuple[float, float]:
    """(lower, upper) edges of the summary band in Hz."""
    lower, upper = config.band
    if upper is None:
        upper = decidecade_bands(bands.fmin, bands.upper_limit(fs))["upper"][-1]
    return lower, upper


def masking_factors(delta_db, spreading: float):
    """Listening range and area remaining when the noise rises by `delta_db`
    and transmission loss is `spreading` log10 r (Clark et al. 2009): the
    range scales by 10^(-dL/N) and the area by its square. No reduction for
    dL <= 0."""
    d = np.clip(np.asarray(delta_db, dtype=float), 0.0, None)
    r = 10 ** (-d / spreading)
    return r, r**2


def strike_intervals(metrics: pl.DataFrame, config: InterpulseConfig) -> pl.DataFrame:
    """Consecutive strike pairs of one event and sensor, without pauses.

    `metrics` holds the event's rows of the metrics table (strike, time,
    channel, valid, t5, t95). Returns one row per interval: strike, time,
    next_time, spacing_s. A pair is kept when the strikes are consecutive in
    the detections and their spacing is at most `max_spacing_factor` times
    the median spacing.
    """
    s = metrics.select("strike", "time").unique().sort("time")
    s = s.with_columns(
        next_strike=pl.col("strike").shift(-1),
        next_time=pl.col("time").shift(-1),
    ).with_columns(
        spacing_s=(pl.col("next_time") - pl.col("time")).dt.total_microseconds() / 1e6
    )
    s = s.filter(pl.col("next_strike") == pl.col("strike") + 1)
    cols = ["strike", "time", "next_time", "spacing_s"]
    if s.is_empty():
        return s.select(cols)
    limit = config.max_spacing_factor * s["spacing_s"].median()
    return s.filter(pl.col("spacing_s") <= limit).select(cols)


def _per_channel(metrics: pl.DataFrame, strikes: np.ndarray, channels: list[int]):
    """(n_channels, n_strikes) arrays of valid, t5 and t95, in the order
    of `strikes` and `channels`; strikes missing on a channel are invalid."""
    out = {}
    for k, default in (("valid", False), ("t5", np.nan), ("t95", np.nan)):
        wide = (
            pl.DataFrame({"strike": strikes})
            .join(
                metrics.pivot(on="channel", index="strike", values=k),
                on="strike",
                how="left",
                maintain_order="left",
            )
            .fill_null(default)
        )
        out[k] = np.stack([wide[str(c)].to_numpy() for c in channels])
    return out


def _chunks(times_us: np.ndarray, next_us: np.ndarray, span_us: int) -> list[slice]:
    """Split intervals into runs that each span at most `span_us` (at least
    one interval each)."""
    chunks, start = [], 0
    for i in range(1, len(times_us) + 1):
        if i == len(times_us) or next_us[i] - times_us[start] > span_us:
            chunks.append(slice(start, i))
            start = i
    return chunks


class _Levels:
    """Mean-square levels of segments: broadband, summary band, and
    decidecade bands, from the DFT of each segment."""

    def __init__(self, n: int, fs: float, config: InterpulseConfig, bands: BandsConfig):
        self.n, self.fs = n, fs
        self.edges = decidecade_bands(bands.fmin, bands.upper_limit(fs))
        freqs = np.fft.rfftfreq(n, 1 / fs)
        lo, hi = summary_band(config, bands, fs)
        self.w_bands = band_weights(freqs, self.edges["lower"], self.edges["upper"])
        self.w_summary = band_weights(freqs, np.array([lo]), np.array([hi]))

    def __call__(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """`x` (..., n): levels in dB, with the band levels on a last axis."""
        out = {"Lp": [], "Lp_band": [], "bands": []}
        flat = x.reshape(-1, self.n)
        for b0 in range(0, len(flat), _BATCH):
            b = flat[b0 : b0 + _BATCH]
            _, e = exposure_spectrum(b, self.fs)
            dur = self.n / self.fs
            out["Lp"].append(_db_power(np.mean(b**2, axis=-1)))
            out["Lp_band"].append(_db_power(band_sum(e, self.w_summary)[:, 0] / dur))
            out["bands"].append(_db_power(band_sum(e, self.w_bands) / dur))
        shape = x.shape[:-1]
        return {
            "Lp": np.concatenate(out["Lp"]).reshape(shape),
            "Lp_band": np.concatenate(out["Lp_band"]).reshape(shape),
            "bands": np.concatenate(out["bands"]).reshape(*shape, -1),
        }


def interpulse(
    sensor: SensorConfig,
    inventory: Path,
    event: Event,
    metrics: pl.DataFrame,
    config: InterpulseConfig,
    bands: BandsConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """100 ms segment levels between strikes, and their medians per interval.

    Returns:
        segments: one row per segment and hydrophone: channel, strike (of the
            interval's first strike), t_s (segment start after that strike),
            spacing_s, Lp, Lp_band, in_interval (inside t95 to the next t5),
            in_guan (inside Guan et al.'s interval), valid (both strikes'
            pulses valid on the hydrophone and the segment not in a gap).
        intervals: one row per interval and hydrophone: channel, strike, time,
            spacing_s, start_s and duration_s of the inter-pulse interval
            (t95 to the next t5), n_segments (valid segments inside it), and
            the median over them of Lp, Lp_band and each decidecade band level
            `L_<nominal>`, and Lp_band_guan, the median of Lp_band over Guan
            et al.'s interval. Levels are NaN where no valid segment lies in
            the interval.
    """
    iv = strike_intervals(metrics, config)
    if iv.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    channels = sorted(metrics["channel"].unique().to_list())
    strikes = metrics.select("strike", "time").unique().sort("time")["strike"]
    per = _per_channel(metrics, strikes.to_numpy(), channels)
    pos = {s: k for k, s in enumerate(strikes.to_list())}
    first = np.array([pos[s] for s in iv["strike"]])
    t0_us = iv["time"].to_numpy().astype("datetime64[us]")
    t1_us = iv["next_time"].to_numpy().astype("datetime64[us]")
    spacing = iv["spacing_s"].to_numpy()
    strike_ids = iv["strike"].to_numpy()
    pad = np.timedelta64(round(_PAD_S * 1e6), "us")
    span = round(config.chunk_s * 1e6)
    n_ch = len(channels)
    seg_frames, int_frames = [], []
    levels = None

    for ck in _chunks(t0_us.astype(np.int64), t1_us.astype(np.int64), span):
        a, b = t0_us[ck][0], t1_us[ck][-1]
        try:
            seg = read_segment(sensor, inventory, a - pad, b + pad, channels=channels)
        except NoDataError:
            logger.warning(f"[interpulse] {sensor.name}: no data {a} - {b}")
            continue
        fs = seg.sampling_rate
        n_seg = round(config.segment_s * fs)
        if levels is None:
            levels = _Levels(n_seg, fs, config, bands)
        recorded = seg.data != 0.0
        data = apply_filter(seg.data, fs, config.filter)

        # 100 ms segments from each strike to the next, within the data read.
        k = np.arange(ck.start, ck.stop)
        counts = np.floor(spacing[k] / config.segment_s + 1e-9).astype(int)
        which = np.repeat(np.arange(len(k)), counts)
        j = np.concatenate([np.arange(c) for c in counts])
        offset_us = (t0_us[k] - seg.time_init) / np.timedelta64(1, "us")
        i0 = np.round(offset_us * 1e-6 * fs).astype(int)[which] + j * n_seg
        inside = (i0 >= 0) & (i0 + n_seg <= data.shape[1])
        which, j, i0 = which[inside], j[inside], i0[inside]
        t_s = j * config.segment_s
        idx = i0[:, None] + np.arange(n_seg)
        lv = levels(data[:, idx])
        gap = 1.0 - recorded[:, idx].mean(axis=-1) > _GAP_FRACTION

        f0, f1 = first[k][which], first[k][which] + 1
        sp = spacing[k][which]
        valid = per["valid"][:, f0] & per["valid"][:, f1] & ~gap
        in_interval = (t_s >= per["t95"][:, f0]) & (
            t_s + config.segment_s <= sp + per["t5"][:, f1]
        )
        in_guan = np.broadcast_to(
            (t_s >= config.guan_after_s - 1e-9)
            & (t_s + config.segment_s <= sp - config.guan_before_s + 1e-9),
            valid.shape,
        )
        strike = strike_ids[k][which]
        seg_frames.append(
            pl.DataFrame(
                {
                    "channel": np.repeat(channels, len(j)),
                    "strike": np.tile(strike, n_ch),
                    "t_s": np.tile(t_s, n_ch).astype(np.float32),
                    "spacing_s": np.tile(sp, n_ch).astype(np.float32),
                    "Lp": lv["Lp"].reshape(-1).astype(np.float32),
                    "Lp_band": lv["Lp_band"].reshape(-1).astype(np.float32),
                    "in_interval": in_interval.reshape(-1),
                    "in_guan": in_guan.reshape(-1),
                    "valid": valid.reshape(-1),
                }
            )
        )

        # Medians over each interval's valid segments, per hydrophone.
        names = [band_label(n) for n in levels.edges["nominal"]]
        use = valid & in_interval
        rows = pl.DataFrame(
            {
                "channel": np.repeat(channels, len(j)),
                "strike": np.tile(strike, n_ch),
                "use": use.reshape(-1),
                "guan": (valid & in_guan).reshape(-1),
                "Lp": lv["Lp"].reshape(-1),
                "Lp_band": lv["Lp_band"].reshape(-1),
                **{
                    name: lv["bands"][..., n].reshape(-1)
                    for n, name in enumerate(names)
                },
            }
        )
        med = rows.group_by("channel", "strike").agg(
            pl.col("use").sum().alias("n_segments"),
            *[pl.col(c).filter("use").median() for c in ["Lp", "Lp_band", *names]],
            pl.col("Lp_band").filter("guan").median().alias("Lp_band_guan"),
        )
        start95 = per["t95"][:, first[k]]
        end5 = spacing[k] + per["t5"][:, first[k] + 1]
        base = pl.DataFrame(
            {
                "channel": np.repeat(channels, len(k)),
                "strike": np.tile(strike_ids[k], n_ch),
                "time": np.tile(iv["time"].to_numpy()[k], n_ch),
                "spacing_s": np.tile(spacing[k], n_ch),
                "start_s": start95.reshape(-1),
                "duration_s": np.clip(end5 - start95, 0.0, None).reshape(-1),
            }
        )
        int_frames.append(
            base.join(med, on=["channel", "strike"], how="left").with_columns(
                pl.col("n_segments").fill_null(0)
            )
        )

    if not seg_frames:
        return pl.DataFrame(), pl.DataFrame()
    return (
        _label(pl.concat(seg_frames), event, sensor).sort("channel", "strike", "t_s"),
        _label(pl.concat(int_frames), event, sensor).sort("channel", "strike"),
    )


def background_segments(
    sensor: SensorConfig,
    inventory: Path,
    event: Event,
    config: InterpulseConfig,
    bands: BandsConfig,
    background: BackgroundConfig,
) -> pl.DataFrame:
    """The same 100 ms segment levels over the background noise window before the
    event: channel, start (time), valid (not in a gap), Lp, Lp_band, and the
    decidecade band levels `L_<nominal>`."""
    start, end = background_window(event, background)
    pad = np.timedelta64(round(_PAD_S * 1e6), "us")
    try:
        seg = read_segment(sensor, inventory, start - pad, end + pad)
    except NoDataError:
        return pl.DataFrame()
    fs = seg.sampling_rate
    n_seg = round(config.segment_s * fs)
    i0 = round((start - seg.time_init) / np.timedelta64(1, "us") * 1e-6 * fs)
    want = round((end - start) / np.timedelta64(1, "us") * 1e-6 * fs)
    n = min(want, seg.data.shape[1] - max(i0, 0)) // n_seg
    if i0 < 0 or n <= 0:
        return pl.DataFrame()
    levels = _Levels(n_seg, fs, config, bands)
    data = apply_filter(seg.data, fs, config.filter)

    def blocks(x):
        return x[:, i0 : i0 + n * n_seg].reshape(x.shape[0], n, n_seg)

    lv = levels(blocks(data))
    starts = start + (np.arange(n) * config.segment_s * 1e6).astype("timedelta64[us]")
    n_ch = len(seg.channels)
    frame = {
        "channel": np.repeat(seg.channels, n),
        "start": np.tile(starts, n_ch),
        "valid": (np.mean(blocks(seg.data) == 0.0, -1) <= _GAP_FRACTION).reshape(-1),
        "Lp": lv["Lp"].reshape(-1).astype(np.float32),
        "Lp_band": lv["Lp_band"].reshape(-1).astype(np.float32),
    }
    for k, nom in enumerate(levels.edges["nominal"]):
        frame[band_label(nom)] = lv["bands"][..., k].reshape(-1).astype(np.float32)
    return _label(pl.DataFrame(frame), event, sensor)


def _label(df: pl.DataFrame, event: Event, sensor: SensorConfig) -> pl.DataFrame:
    return df.select(
        pl.lit(event.name).alias("event"),
        pl.lit(sensor.name).alias("sensor"),
        pl.lit(source_range(event, sensor)).alias("range_m"),
        pl.all(),
    )


# Summaries for the report and the site comparison.

KEYS = ["event", "sensor", "range_m", "channel"]


def read_interpulse(site, events: list, kind: str) -> pl.DataFrame:
    """Concatenate one kind of inter-pulse output ("segments", "intervals",
    or "background") over events and sensors; empty if none exist."""
    path = {
        "segments": site.interpulse_segments_path,
        "intervals": site.interpulse_intervals_path,
        "background": site.interpulse_background_path,
    }[kind]
    files = [
        p
        for e in events
        for s in site.sensors
        if (p := path(e.name if hasattr(e, "name") else e, s.name)).exists()
    ]
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def background_medians(background: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Median background noise level per event and hydrophone, for each of `columns`,
    over valid 100 ms segments (named `<column>_bg`)."""
    return (
        background.filter("valid")
        .group_by(KEYS)
        .agg(pl.col(c).median().alias(f"{c}_bg") for c in columns)
    )


def excess_summary(
    segments: pl.DataFrame,
    background: pl.DataFrame,
    thresholds_db: list[float],
    column: str = "in_interval",
) -> pl.DataFrame:
    """Per event and hydrophone, over valid inter-pulse segments (`column`
    tags which): the summary-band level (median, 5th and 95th percentiles),
    the background noise median, the median excess dL = median - background noise median, and
    the fraction of inter-pulse time more than each threshold above the
    background noise median (`above_<t>`); also the share of the strike spacing taken
    up by the interval (`interval_share`)."""
    bg = background_medians(background, ["Lp_band"])
    seg = segments.join(bg, on=KEYS, how="inner")
    inside = seg.filter("valid", pl.col(column))
    share = (
        seg.filter("valid")
        .group_by(KEYS)
        .agg(pl.col(column).mean().alias("interval_share"))
    )
    excess = pl.col("Lp_band") - pl.col("Lp_band_bg")
    out = inside.group_by(KEYS).agg(
        pl.len().alias("n_segments"),
        pl.col("Lp_band").median().alias("Lp_median"),
        pl.col("Lp_band").quantile(0.05).alias("Lp_p5"),
        pl.col("Lp_band").quantile(0.95).alias("Lp_p95"),
        pl.col("Lp_band_bg").first().alias("Lp_bg"),
        *[(excess > t).mean().alias(f"above_{t:g}") for t in thresholds_db],
    )
    return (
        out.with_columns(dL=pl.col("Lp_median") - pl.col("Lp_bg"))
        .join(share, on=KEYS, how="left")
        .sort("event", "range_m", "channel")
    )


def decay_curves(segments: pl.DataFrame, background: pl.DataFrame) -> pl.DataFrame:
    """Median summary-band level against time since the strike, relative to
    the background noise median, per event and hydrophone: KEYS, t_s, dL, n."""
    bg = background_medians(background, ["Lp_band"])
    return (
        segments.filter("valid")
        .join(bg, on=KEYS, how="inner")
        .group_by(*KEYS, "t_s")
        .agg(
            (pl.col("Lp_band") - pl.col("Lp_band_bg")).median().alias("dL"),
            pl.len().alias("n"),
        )
        .sort(*KEYS, "t_s")
    )


def band_excess(intervals: pl.DataFrame, background: pl.DataFrame) -> pl.DataFrame:
    """Median excess over the background noise in each decidecade band, per event and
    hydrophone: KEYS, nominal_Hz, dL. Interval levels are the medians over
    each interval's segments (see `interpulse`); the excess is their median
    over intervals less the background noise median."""
    bands = [c for c in intervals.columns if is_band_label(c)]

    def long(df: pl.DataFrame, value: str) -> pl.DataFrame:
        return df.unpivot(index=KEYS, on=bands, variable_name="band", value_name=value)

    pulses = (
        intervals.filter(pl.col("n_segments") > 0)
        .group_by(KEYS)
        .agg(pl.col(c).median() for c in bands)
    )
    bg = (
        background.filter("valid").group_by(KEYS).agg(pl.col(c).median() for c in bands)
    )
    return (
        long(pulses, "Lp")
        .join(long(bg, "Lp_bg"), on=[*KEYS, "band"], how="inner")
        .with_columns(
            nominal_Hz=pl.col("band")
            .str.strip_prefix("Lp_")
            .str.strip_suffix("Hz")
            .cast(pl.Float64),
            dL=pl.col("Lp") - pl.col("Lp_bg"),
        )
        .select(*KEYS, "nominal_Hz", "dL")
        .sort(*KEYS, "nominal_Hz")
    )


HIST_BIN_DB = 0.5


def summarize_site(site, events: list) -> dict[str, pl.DataFrame]:
    """Site-wide inter-pulse summaries over the events' outputs, one event
    and sensor at a time (the segment files are large):

        excess       `excess_summary` per hydrophone, `definition` "interval"
                     (t95 to the next t5) or "guan"
        decay        `decay_curves` per hydrophone
        bands        `band_excess` per hydrophone
        hist         counts of summary-band segment levels in HIST_BIN_DB bins
                     (`Lp_band`, the bin's lower edge) per hydrophone, for
                     `kind` "interpulse" (inside the interval) and "background"
    """
    thresholds = site.interpulse.thresholds_db
    out: dict[str, list[pl.DataFrame]] = {
        k: [] for k in ("excess", "decay", "bands", "hist")
    }

    def hist(df: pl.DataFrame, kind: str) -> pl.DataFrame:
        return (
            df.with_columns(
                Lp_band=(pl.col("Lp_band") / HIST_BIN_DB).floor() * HIST_BIN_DB
            )
            .filter(pl.col("Lp_band").is_finite())
            .group_by(*KEYS, "Lp_band")
            .len("count")
            .with_columns(kind=pl.lit(kind))
        )

    for e in events:
        for s in site.sensors:
            if site.is_excluded(e.name, s.name) or too_few_strikes(
                site, e.name, s.name
            ):
                continue
            paths = [
                site.interpulse_segments_path(e.name, s.name),
                site.interpulse_intervals_path(e.name, s.name),
                site.interpulse_background_path(e.name, s.name),
            ]
            if not all(p.exists() for p in paths):
                continue
            seg, iv, bg = (pl.read_parquet(p) for p in paths)
            for definition, column in (
                ("interval", "in_interval"),
                ("guan", "in_guan"),
            ):
                out["excess"].append(
                    excess_summary(seg, bg, thresholds, column).with_columns(
                        definition=pl.lit(definition)
                    )
                )
            out["decay"].append(decay_curves(seg, bg))
            out["bands"].append(band_excess(iv, bg))
            out["hist"] += [
                hist(seg.filter("valid", "in_interval"), "interpulse"),
                hist(bg.filter("valid"), "background"),
            ]
    return {
        k: pl.concat(v, how="diagonal_relaxed") if v else pl.DataFrame()
        for k, v in out.items()
    }
