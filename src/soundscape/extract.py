"""Window every channel of a sensor around each detected strike.

Pulses for one event and sensor are stored as fixed-length arrays in HDF5:

    /waveforms   float32 (n_strikes, n_channels, n_samples), uPa
    /strike      int64   (n_strikes,), strike number from the detections table
    /time        int64   (n_strikes,), detection time [us since epoch]
    /start       int64   (n_strikes,), time of first sample [us since epoch]
    /valid       bool    (n_strikes,), False if the window touches a data gap
    attrs: event, sensor, pile, range_m, sampling_rate, channels, pre_s, post_s,
           filter (JSON), units

Data are read and filtered in contiguous blocks of strikes, so filter
transients never fall inside a pulse window and each raw file is read once.
"""

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from scipy.signal import butter, sosfiltfilt
from tritonoa.data.reader import NoDataError

from soundscape.config import ExtractConfig, FilterConfig, SensorConfig
from soundscape.events import Event, source_range
from soundscape.sources import Segment, read_segment

logger = logging.getLogger(__name__)

_FILTER_PAD_S = 5.0
_MAX_GAP_FRACTION = 0.01


def apply_filter(
    data: np.ndarray, fs: float, config: FilterConfig | None
) -> np.ndarray:
    if config is None:
        return data
    sos = butter(config.order, config.freq, btype=config.type, fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def _blocks(times_us: np.ndarray, max_span_us: int) -> list[slice]:
    """Split sorted times into contiguous blocks spanning at most `max_span_us`."""
    blocks, start = [], 0
    for i in range(1, len(times_us) + 1):
        if i == len(times_us) or times_us[i] - times_us[start] > max_span_us:
            blocks.append(slice(start, i))
            start = i
    return blocks


def window_pulses(
    seg: Segment, times: np.ndarray, pre_s: float, post_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut fixed-length windows from a segment.

    Returns:
        waveforms (n, n_channels, n_samples), window start times, and a mask
        that is False where the window is out of range or mostly zero-filled.
    """
    n_pre = round(pre_s * seg.sampling_rate)
    n_samples = n_pre + round(post_s * seg.sampling_rate)
    out = np.zeros((len(times), seg.data.shape[0], n_samples), dtype=np.float32)
    starts = np.empty(len(times), dtype="datetime64[us]")
    valid = np.ones(len(times), dtype=bool)
    for k, t in enumerate(times):
        i0 = seg.index(t) - n_pre
        i1 = i0 + n_samples
        starts[k] = seg.time_init + seg.duration(i0)
        if i0 < 0 or i1 > seg.data.shape[1]:
            valid[k] = False
            continue
        out[k] = seg.data[:, i0:i1]
        if np.mean(out[k] == 0.0) > _MAX_GAP_FRACTION:
            valid[k] = False
    return out, starts, valid


def extract(
    sensor: SensorConfig,
    inventory: Path,
    event: Event,
    detections: pl.DataFrame,
    config: ExtractConfig,
    block_s: float,
    destination: Path,
) -> None:
    """Extract pulses for all detections of one sensor and event."""
    detections = detections.sort("strike")
    times = detections["time"].to_numpy().astype("datetime64[us]")
    strike_ids = detections["strike"].to_numpy()
    pad_s = max(config.pre_s, config.post_s) + _FILTER_PAD_S
    pad = np.timedelta64(int(pad_s * 1e6), "us")

    waveforms, strikes, peak_times, starts, valid = [], [], [], [], []
    fs = None
    for block in _blocks(times.astype(np.int64), int(block_s * 1e6)):
        bt = times[block]
        try:
            seg = read_segment(sensor, inventory, bt[0] - pad, bt[-1] + pad)
        except NoDataError:
            logger.warning(f"{sensor.name}: no data for strikes {block}")
            continue
        seg.data = apply_filter(seg.data, seg.sampling_rate, config.filter)
        w, s, v = window_pulses(seg, bt, config.pre_s, config.post_s)
        waveforms.append(w)
        strikes.append(strike_ids[block])
        peak_times.append(bt)
        starts.append(s)
        valid.append(v)
        fs = seg.sampling_rate

    if fs is None:
        logger.warning(f"{event.name}/{sensor.name}: no pulses extracted.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as f:
        wf = np.concatenate(waveforms)
        f.create_dataset("waveforms", data=wf, chunks=(1, *wf.shape[1:]))
        f.create_dataset("strike", data=np.concatenate(strikes))
        f.create_dataset("time", data=np.concatenate(peak_times).astype(np.int64))
        f.create_dataset("start", data=np.concatenate(starts).astype(np.int64))
        f.create_dataset("valid", data=np.concatenate(valid))
        f.attrs.update(
            {
                "event": event.name,
                "sensor": sensor.name,
                "pile": event.pile,
                "range_m": source_range(event, sensor),
                "sampling_rate": fs,
                "channels": sensor.channels,
                # HDF5 attributes cannot hold None: NaN and "" mean unknown.
                "depth_m": [
                    np.nan if (d := sensor.hydrophone(c).depth_m) is None else d
                    for c in sensor.channels
                ],
                "serial_number": [
                    sensor.hydrophone(c).serial_number or "" for c in sensor.channels
                ],
                "pre_s": config.pre_s,
                "post_s": config.post_s,
                "filter": json.dumps(
                    None if config.filter is None else config.filter.model_dump()
                ),
                "units": "uPa",
            }
        )
    logger.info(f"{event.name}/{sensor.name}: {len(wf)} pulses -> {destination}")


def read_pulses(
    path: Path, waveforms: bool = True
) -> tuple[np.ndarray | None, dict, dict]:
    """Read a pulse file.

    Returns:
        waveforms (or None), per-strike arrays {strike, time, start, valid},
        and attributes.
    """
    with h5py.File(path, "r") as f:
        wf = f["waveforms"][:] if waveforms else None
        arrays = {
            "strike": f["strike"][:],
            "time": f["time"][:].astype("datetime64[us]"),
            "start": f["start"][:].astype("datetime64[us]"),
            "valid": f["valid"][:],
        }
        attrs = dict(f.attrs)
    attrs["channels"] = [int(c) for c in attrs["channels"]]
    return wf, arrays, attrs


def iter_waveforms(path: Path, batch_size: int = 256):
    """Yield waveforms in batches of strikes to bound memory use."""
    with h5py.File(path, "r") as f:
        dset = f["waveforms"]
        for i in range(0, dset.shape[0], batch_size):
            yield dset[i : i + batch_size]
