"""Ambient noise before each event, and power spectral densities of pulses."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.signal import welch

from soundscape.bands import band_sum, band_weights, decidecade_bands, exposure_spectrum
from soundscape.config import AmbientConfig, BandsConfig, PulsePSDConfig, SensorConfig
from soundscape.events import Event, source_range
from soundscape.extract import apply_filter, iter_waveforms, read_pulses
from soundscape.kurtosis import beta2
from soundscape.metrics import _db_power, channel_valid, spike_count
from soundscape.sources import read_segment

logger = logging.getLogger(__name__)


def ambient_window(
    event: Event, config: AmbientConfig
) -> tuple[np.datetime64, np.datetime64]:
    start = event.start - np.timedelta64(int(config.offset_s * 1e6), "us")
    return start, start + np.timedelta64(int(config.duration_s * 1e6), "us")


def ambient_noise(
    sensor: SensorConfig, inventory: Path, event: Event, config: AmbientConfig
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Mean-square SPL and kurtosis over consecutive intervals, and Welch PSDs
    of the whole window and of each interval.

    Returns:
        A tidy table of interval SPLs and kurtosis, and {"freqs", "psd"
        (n_channels, n_freqs), "psd_intervals" (n_channels, n_intervals,
        n_freqs), "interval_valid" (n_channels, n_intervals)}.
    """
    start, end = ambient_window(event, config)
    seg = read_segment(sensor, inventory, start, end)
    fs = seg.sampling_rate
    n_int = round(config.interval_s * fs)
    n_full = seg.data.shape[1] // n_int

    def _blocks(x: np.ndarray) -> np.ndarray:
        return x[:, : n_full * n_int].reshape(x.shape[0], n_full, n_int)

    # Gaps are zero-filled; detect them before filtering smears the edges.
    recorded = np.mean(_blocks(seg.data) == 0.0, axis=-1) < 0.01
    data = apply_filter(seg.data, fs, config.filter)
    blocks = _blocks(data)
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
    _, psd_intervals = welch(
        blocks,
        fs=fs,
        window="hann",
        nperseg=min(nperseg, n_int),
        noverlap=int(min(nperseg, n_int) * config.psd_overlap),
        scaling="density",
        axis=-1,
    )

    interval_start = seg.time_init + np.arange(n_full) * seg.duration(n_int)
    table = pl.DataFrame(
        {
            "event": event.name,
            "sensor": sensor.name,
            "range_m": source_range(event, sensor),
            "channel": np.repeat(seg.channels, n_full),
            "interval_start": np.tile(interval_start, len(seg.channels)),
            "valid": recorded.reshape(-1),
            "SPL_rms": spl.reshape(-1),
            "kurtosis": kurt.reshape(-1),
        }
    )
    return table, {
        "freqs": freqs,
        "psd": psd,
        "psd_intervals": psd_intervals,
        "interval_valid": recorded,
    }


def write_ambient_psd(
    path: Path, event: Event, sensor: SensorConfig, spectra: dict, config: AmbientConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        name = f"{event.name}/{sensor.name}"
        if name in f:
            del f[name]
        g = f.create_group(name)
        g.create_dataset("freqs", data=spectra["freqs"])
        g.create_dataset("psd", data=spectra["psd"])
        if "psd_intervals" in spectra:
            g.create_dataset(
                "psd_intervals", data=spectra["psd_intervals"].astype(np.float32)
            )
            g.create_dataset("interval_valid", data=spectra["interval_valid"])
        start, end = ambient_window(event, config)
        g.attrs.update(
            {
                "channels": sensor.channels,
                "start": str(start),
                "end": str(end),
                "units": "uPa^2/Hz",
                "config": json.dumps(config.model_dump()),
            }
        )


def pulse_psd(
    pulses_path: Path,
    config: PulsePSDConfig,
    destination: Path,
    bands: BandsConfig | None = None,
    spike_threshold: float | None = None,
) -> None:
    """Welch PSD and decidecade band exposure of every pulse and channel.

    Writes /freqs, /psd float32 (n_strikes, n_channels, n_freqs) in uPa^2/Hz;
    /band_exposure float32 (n_strikes, n_channels, n_bands) in uPa^2 s, the
    single-strike sound exposure in each band from the DFT of the whole
    window (so the bands of the full spectrum sum to the exposure behind
    SEL_ss); /band_center, /band_lower, /band_upper, /band_nominal in Hz; and
    /strike, /time, /valid copied from the pulse file; and /channel_valid
    bool (n_strikes, n_channels), also False where the channel's window has
    spikes (with `spike_threshold`; see `metrics.spike_count`).
    """
    bands = bands or BandsConfig()
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    fs = attrs["sampling_rate"]
    edges = decidecade_bands(bands.fmin, bands.upper_limit(fs))
    psds, band_exposure, spikes = [], [], []
    freqs = weights = None
    for w in iter_waveforms(pulses_path):
        w = w.astype(np.float64)
        spikes.append(
            spike_count(w, spike_threshold)
            if spike_threshold is not None
            else np.zeros(w.shape[:2], dtype=int)
        )
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
            data=channel_valid(arrays["valid"], np.concatenate(spikes)),
        )
        f.create_dataset("time", data=arrays["time"].astype(np.int64))
        f.attrs.update(attrs)
        f.attrs.update(
            {"units": "uPa^2/Hz", "nperseg": nperseg, "overlap": config.overlap}
        )
