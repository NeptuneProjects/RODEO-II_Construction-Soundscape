"""Ambient noise before each event, and power spectral densities of pulses."""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.signal import welch

from soundscape.config import AmbientConfig, PulsePSDConfig, SensorConfig
from soundscape.events import Event, source_range
from soundscape.extract import apply_filter, iter_waveforms, read_pulses
from soundscape.metrics import _db_power
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
    """SPL_rms over consecutive intervals and the Welch PSD of the whole window.

    Returns:
        A tidy table of interval SPLs, and {"freqs", "psd" (n_channels, n_freqs)}.
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
        }
    )
    return table, {"freqs": freqs, "psd": psd}


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


def pulse_psd(pulses_path: Path, config: PulsePSDConfig, destination: Path) -> None:
    """Welch PSD of every pulse and channel.

    Writes /freqs, /psd float32 (n_strikes, n_channels, n_freqs) in uPa^2/Hz,
    and /strike, /time, /valid copied from the pulse file.
    """
    _, arrays, attrs = read_pulses(pulses_path, waveforms=False)
    fs = attrs["sampling_rate"]
    psds = []
    freqs = None
    for w in iter_waveforms(pulses_path):
        nperseg = min(config.nperseg, w.shape[-1])
        freqs, p = welch(
            w.astype(np.float64),
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=int(nperseg * config.overlap),
            scaling="density",
            axis=-1,
        )
        psds.append(p.astype(np.float32))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as f:
        f.create_dataset("freqs", data=freqs)
        f.create_dataset("psd", data=np.concatenate(psds))
        for k in ("strike", "valid"):
            f.create_dataset(k, data=arrays[k])
        f.create_dataset("time", data=arrays["time"].astype(np.int64))
        f.attrs.update(attrs)
        f.attrs.update(
            {"units": "uPa^2/Hz", "nperseg": nperseg, "overlap": config.overlap}
        )
