"""Background noise before each event, and power spectral densities of pulses."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.signal import welch

from soundscape.bands import band_sum, band_weights, decidecade_bands, exposure_spectrum
from soundscape.config import (
    BackgroundConfig,
    BandsConfig,
    PulsePSDConfig,
    SensorConfig,
)
from soundscape.events import Event, source_range
from soundscape.extract import (
    apply_filter,
    integration_bounds,
    iter_waveforms,
    read_pulses,
    trim,
)
from soundscape.kurtosis import beta2
from soundscape.metrics import _db_power, channel_valid, click_count
from soundscape.sources import read_segment

logger = logging.getLogger(__name__)


def background_window(
    event: Event, config: BackgroundConfig
) -> tuple[np.datetime64, np.datetime64]:
    start = event.start - np.timedelta64(int(config.offset_s * 1e6), "us")
    return start, start + np.timedelta64(int(config.duration_s * 1e6), "us")


def background_noise(
    sensor: SensorConfig,
    inventory: Path,
    event: Event,
    config: BackgroundConfig,
    segment_s: float,
    psd_config: PulsePSDConfig | None = None,
    bands: BandsConfig | None = None,
    click_threshold: float | None = None,
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Background noise (ISO 18406:2017, 3.4 and 6.1.4) in the window before
    an event.

    The window is cut into consecutive segments of `segment_s`, the averaging
    time of the pulse metrics it is compared with, and for each the
    mean-square sound pressure level, the mean-square sound pressure in each decidecade band
    (from the DFT, as for pulses), and the Welch PSD with the pulse settings
    are computed. A segment is invalid where it overlaps a recording gap or,
    with `click_threshold`, holds a click from the recording chain (see
    `metrics.click_count`). Kurtosis and mean-square sound pressure level are also computed
    over consecutive `config.interval_s` intervals.

    Returns:
        A tidy table of interval SPLs and kurtosis, and {"freqs", "psd"
        (n_channels, n_freqs) over the whole window, "segment_s",
        "segment_valid" and "segment_Lp" (dB re 1 uPa) (n_channels,
        n_segments), "segment_band_ms" (n_channels, n_segments, n_bands) in
        uPa^2, "band_{center,lower,upper,nominal}", "segment_freqs", and
        "segment_psd_{median,p5,p95}" (n_channels, n_freqs) in dB re
        1 uPa^2/Hz over valid segments}.
    """
    psd_config = psd_config or PulsePSDConfig()
    bands = bands or BandsConfig()
    start, end = background_window(event, config)
    seg = read_segment(sensor, inventory, start, end)
    fs = seg.sampling_rate

    def _blocks(x: np.ndarray, n: int) -> np.ndarray:
        return x[:, : (x.shape[1] // n) * n].reshape(x.shape[0], -1, n)

    # Gaps are zero-filled; detect them before filtering smears the edges.
    def _recorded(n: int) -> np.ndarray:
        return np.mean(_blocks(seg.data, n) == 0.0, axis=-1) < 0.01

    n_int = round(config.interval_s * fs)
    n_seg = round(segment_s * fs)
    recorded, seg_recorded = _recorded(n_int), _recorded(n_seg)
    data = apply_filter(seg.data, fs, config.filter)
    blocks = _blocks(data, n_int)
    n_full = blocks.shape[1]
    spl = _db_power(np.mean(blocks**2, axis=-1))
    kurt = np.where(recorded, beta2(blocks), np.nan)

    nperseg = round(config.psd_segment_s * fs)
    freqs, psd = welch(
        data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=int(nperseg * config.psd_overlap),
        scaling="density",
        axis=-1,
    )

    edges = decidecade_bands(bands.fmin, bands.upper_limit(fs))
    segments = _blocks(data, n_seg)
    n_ch, n_segments = segments.shape[:2]
    seg_valid = seg_recorded.copy()
    seg_lp = np.full((n_ch, n_segments), np.nan, dtype=np.float32)
    band_ms = np.full((n_ch, n_segments, len(edges["center"])), np.nan, np.float32)
    nper = min(psd_config.nperseg, n_seg)
    stats = {k: [] for k in ("median", "p5", "p95")}
    seg_freqs = np.fft.rfftfreq(nper, 1 / fs)
    weights = None
    # One channel at a time to bound memory use.
    for c in range(n_ch):
        x = segments[c]
        if click_threshold is not None:
            seg_valid[c] &= click_count(x, click_threshold) == 0
        seg_lp[c] = _db_power(np.mean(x**2, axis=-1))
        f_e, e = exposure_spectrum(x, fs)
        if weights is None:
            weights = band_weights(f_e, edges["lower"], edges["upper"])
        band_ms[c] = band_sum(e, weights) / segment_s
        if not seg_valid[c].any():
            for v in stats.values():
                v.append(np.full(len(seg_freqs), np.nan))
            continue
        _, p = welch(
            x[seg_valid[c]],
            fs=fs,
            window="hann",
            nperseg=nper,
            noverlap=int(nper * psd_config.overlap),
            scaling="density",
            axis=-1,
        )
        level = 10 * np.log10(p)
        for k, v in zip(("p5", "median", "p95"), np.percentile(level, [5, 50, 95], 0)):
            stats[k].append(v)

    interval_start = seg.time_init + np.arange(n_full) * seg.duration(n_int)
    table = pl.DataFrame(
        {
            "event": event.name,
            "sensor": sensor.name,
            "range_m": source_range(event, sensor),
            "channel": np.repeat(seg.channels, n_full),
            "interval_start": np.tile(interval_start, len(seg.channels)),
            "valid": recorded.reshape(-1),
            "Lp": spl.reshape(-1),
            "beta": kurt.reshape(-1),
        }
    )
    return table, {
        "freqs": freqs,
        "psd": psd,
        "segment_s": segment_s,
        "segment_valid": seg_valid,
        "segment_Lp": seg_lp,
        "segment_band_ms": band_ms,
        **{f"band_{k}": v for k, v in edges.items()},
        "segment_freqs": seg_freqs,
        **{f"segment_psd_{k}": np.array(v) for k, v in stats.items()},
    }


def write_background_psd(
    path: Path,
    event: Event,
    sensor: SensorConfig,
    spectra: dict,
    config: BackgroundConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        name = f"{event.name}/{sensor.name}"
        if name in f:
            del f[name]
        g = f.create_group(name)
        g.create_dataset("freqs", data=spectra["freqs"])
        g.create_dataset("psd", data=spectra["psd"])
        for k, v in spectra.items():
            if k.startswith(("segment_", "band_")) and k != "segment_s":
                g.create_dataset(k, data=v)
        start, end = background_window(event, config)
        g.attrs.update(
            {
                "channels": sensor.channels,
                "start": str(start),
                "end": str(end),
                "units": "uPa^2/Hz",
                "segment_s": spectra["segment_s"],
                "config": json.dumps(config.model_dump()),
            }
        )


def pulse_psd(
    pulses_path: Path,
    config: PulsePSDConfig,
    destination: Path,
    bands: BandsConfig | None = None,
    click_threshold: float | None = None,
) -> None:
    """Welch PSD and decidecade band exposure of every pulse and channel.

    Writes /freqs, /psd float32 (n_strikes, n_channels, n_freqs) in uPa^2/Hz;
    /band_exposure float32 (n_strikes, n_channels, n_bands) in uPa^2 s, the
    single-strike sound exposure in each band from the DFT of the window
    over the pulse's integration time (see `extract.integration_bounds`; so
    the bands of the full spectrum sum to the exposure behind LEp_ss); /band_center, /band_lower, /band_upper, /band_nominal in Hz; and
    /strike, /time, /valid copied from the pulse file; and /channel_valid
    bool (n_strikes, n_channels), also False where the channel's window has
    clicks (with `click_threshold`; see `metrics.click_count`).
    """
    bands = bands or BandsConfig()
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    fs = attrs["sampling_rate"]
    edges = decidecade_bands(bands.fmin, bands.upper_limit(fs))
    lo, hi = integration_bounds(pulses_path)
    psds, band_exposure, clicks = [], [], []
    freqs = weights = None
    i = 0
    for w in iter_waveforms(pulses_path):
        w = w.astype(np.float64)
        clicks.append(
            click_count(w, click_threshold)
            if click_threshold is not None
            else np.zeros(w.shape[:2], dtype=int)
        )
        w = trim(w, lo[i : i + len(w)], hi[i : i + len(w)])
        i += len(w)
        nperseg = min(config.nperseg, w.shape[-1])
        freqs, p = welch(
            w,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=int(nperseg * config.overlap),
            scaling="density",
            axis=-1,
        )
        psds.append(p.astype(np.float32))
        f_e, e = exposure_spectrum(w, fs)
        if weights is None:
            weights = band_weights(f_e, edges["lower"], edges["upper"])
        band_exposure.append(band_sum(e, weights).astype(np.float32))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as f:
        f.create_dataset("freqs", data=freqs)
        f.create_dataset("psd", data=np.concatenate(psds))
        f.create_dataset("band_exposure", data=np.concatenate(band_exposure))
        for k in ("center", "lower", "upper", "nominal"):
            f.create_dataset(f"band_{k}", data=edges[k])
        for k in ("strike", "valid"):
            f.create_dataset(k, data=arrays[k])
        f.create_dataset(
            "channel_valid",
            data=channel_valid(arrays["valid"], np.concatenate(clicks)),
        )
        f.create_dataset("time", data=arrays["time"].astype(np.int64))
        f.attrs.update(attrs)
        f.attrs.update(
            {"units": "uPa^2/Hz", "nperseg": nperseg, "overlap": config.overlap}
        )
