# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from rodeo.detect import Detection

REF_PRESSURE = 1.0  # in uPa


@dataclass
class Metrics:
    index: int
    uid: str
    station: str
    channel: int
    time_init: np.datetime64
    time_end: np.datetime64
    unit: str
    SPL_rms: float
    SPL_pk: float
    SPL_pkpk: float
    SEL: float
    t_rise: float
    t_decay: float
    kurtosis: float
    SEL_cum: float = 0.0
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.metadata:
            [setattr(self, k, v) for k, v in self.metadata.items()]


def build_detection(det: h5py.Group) -> Detection:
    if "metadata" in det:
        mdgrp = det["metadata"]
        metadata = {k: mdgrp.attrs[k] for k in mdgrp.attrs.keys()}
    else:
        metadata = None
    return Detection(
        uid=det.attrs["uid"],
        station=det.attrs["station"],
        waveform=np.array(det["waveform"]),
        time_vector=np.array(det["time_vector"], dtype=np.int64).astype(
            "datetime64[us]"
        ),
        channel=det.attrs["channel"],
        time_init=det.attrs["time_init"],
        time_end=det.attrs["time_end"],
        sampling_rate=det.attrs["sampling_rate"],
        unit=det.attrs["unit"],
        metadata=metadata,
    )


def compute_metrics(dbpath: Path) -> list[Metrics]:
    with h5py.File(dbpath, "r") as fr:
        all_metrics = []
        for chan in fr.keys():
            running_sel_cum = 0.0
            for idx, k in enumerate(fr[chan].keys()):
                det = fr[chan][k]
                detection = build_detection(det)
                metrics = compute_pulse_metrics(idx, detection, running_sel_cum)
                running_sel_cum = metrics.SEL_cum
                all_metrics.append(metrics)

    return all_metrics


def compute_pulse_metrics(
    index: int, detection: Detection, running_sel_cum: float
) -> Metrics:
    x = detection.waveform.squeeze()
    t = detection.time_vector

    rise_time, decay_time = rise_decay_times(x, t)
    sound_exp_level = sel(x)
    running_sel_cum = sel_cum(np.array([running_sel_cum, sound_exp_level]))
    return Metrics(
        index=index,
        uid=detection.uid,
        station=detection.station,
        channel=detection.channel,
        time_init=detection.time_init,
        time_end=detection.time_end,
        unit=detection.unit,
        SPL_rms=spl_rms(x),
        SPL_pk=spl_pk(x),
        SPL_pkpk=spl_pkpk(x),
        SEL=sound_exp_level,
        t_rise=rise_time,
        t_decay=decay_time,
        kurtosis=kurtosis(x),
        SEL_cum=running_sel_cum,
        metadata=detection.metadata,
    )


def energy90(x: np.ndarray) -> tuple[int, int]:
    energy = x**2
    start_index, end_index = find_t5t95(energy)
    return 1 / (end_index - start_index) * np.sum(energy[start_index:end_index])


def find_t5t95(energy: np.ndarray) -> tuple[int, int]:
    cum_energy = np.cumsum(energy)
    norm_cum_energy = cum_energy / cum_energy[-1]

    lower_bound = 0.05
    upper_bound = 0.95

    start_index = np.searchsorted(norm_cum_energy, lower_bound)
    end_index = np.searchsorted(norm_cum_energy, upper_bound)
    return start_index, end_index


def p_msl(x: np.ndarray, axis: int = -1) -> float:
    """Returns mean square sound pressure in uPa."""
    return np.mean(x**2, axis=axis)


def p_pk(x: np.ndarray, axis: int = -1) -> float:
    """Returns zero-to-peak (Pk) sound pressure in uPa."""
    return np.abs(x).max(axis=axis)


def p_pkpk(x: np.ndarray, axis: int = -1) -> float:
    """Returns peak-to-peak sound pressure in uPa."""
    return x.max(axis=axis) - x.min(axis=axis)


def p_rms(x: np.ndarray, axis: int = -1) -> float:
    """Returns root mean square (RMS) sound pressure in uPa."""
    return np.sqrt(p_msl(x, axis=axis))


def rise_decay_times(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    energy = x**2
    start_index, end_index = find_t5t95(energy)
    peak_index = np.argmax(energy)
    rise_time = t[peak_index] - t[start_index]
    decay_time = t[end_index] - t[peak_index]
    # TODO: Handle conversion to seconds more programmatically
    return rise_time.astype(float) / 1e6, decay_time.astype(float) / 1e6


def save_metrics_to_csv(all_metrics: list[Metrics], destination: Path) -> None:
    df = pd.DataFrame([vars(m) for m in all_metrics]).drop("metadata", axis=1)
    df.to_csv(destination, index=False)


def sel(x: np.ndarray) -> float:
    """Returns single-strike sound exposure level (SEL) in dB re 1 uPa."""
    return 10 * np.log10(energy90(x) / (REF_PRESSURE**2))


def sel_cum(levels: np.ndarray) -> float:
    return 10 * np.log10(np.sum(10 ** (levels / 10)))


def spl(x: np.ndarray) -> np.ndarray:
    """Returns sound pressure level (SPL) of the time series in dB re 1 uPa."""
    return 20 * np.log10(np.abs(x) / REF_PRESSURE)


def spl_pk(x: np.ndarray, axis: int = -1) -> float:
    """Returns zero-to-peak sound pressure level (SPL) in dB re 1 uPa."""
    return spl(p_pk(x, axis=axis))


def spl_pkpk(x: np.ndarray, axis: int = -1) -> float:
    """Returns peak-to-peak sound pressure (SPL) in dB re 1 uPa."""
    return spl(p_pkpk(x, axis=axis))


def spl_rms(x: np.ndarray, axis: int = -1) -> float:
    """Returns root mean square (RMS) sound pressure level (SPL) of the
    time series in dB re 1 uPa.
    """
    return spl(p_rms(x, axis=axis))
