# -*- coding: utf-8 -*-

import argparse
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
    uid: str
    station: str
    channel: int
    time_init: np.datetime64
    time_end: np.datetime64
    unit: str
    RL_rms: float
    RL_pk: float
    RL_pkpk: float
    SEL: float
    t_rise: float
    t_decay: float
    kurtosis: float
    SEL_cum: float = 0.0
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.metadata:
            [setattr(self, k, v) for k, v in self.metadata.items()]


def save_metrics_to_csv(all_metrics: list[Metrics], destination: Path) -> None:
    df = pd.DataFrame([vars(m) for m in all_metrics]).drop("metadata", axis=1)
    df.to_csv(destination, index=False)


def sel_cum(levels: np.ndarray) -> float:
    return 10 * np.log10(np.sum(10 ** (levels / 10)))


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


def rise_decay_times(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    energy = x**2
    start_index, end_index = find_t5t95(energy)
    peak_index = np.argmax(energy)
    rise_time = t[peak_index] - t[start_index]
    decay_time = t[end_index] - t[peak_index]
    # TODO: Handle conversion to seconds more programmatically
    return rise_time.astype(float) / 1e6, decay_time.astype(float) / 1e6


def sel(x: np.ndarray) -> float:
    """Returns single-strike sound exposure level (SEL) in dB re 1 uPa."""
    return 10 * np.log10(energy90(x) / REF_PRESSURE)


def rl(x: np.ndarray) -> np.ndarray:
    """Returns received level (RL) of the time series in dB re 1 uPa."""
    return 20 * np.log10(np.abs(x) / REF_PRESSURE)


def rl_rms(x: np.ndarray) -> float:
    """Returns root mean square (RMS) received level (RL) of the
    time series in dB re 1 uPa.
    """
    return rl(np.sqrt(np.mean(x**2)))


def rl_pk(x: np.ndarray) -> float:
    """Returns zero-to-peak received level (RL) in dB re 1 uPa."""
    return rl(np.abs(x).max())


def rl_pkpk(x: np.ndarray) -> float:
    """Returns peak-to-peak received level (RL) in dB re 1 uPa."""
    return rl(x.max() - x.min())


def compute_pulse_metrics(detection: Detection, running_sel_cum: float) -> Metrics:
    x = detection.waveform
    t = detection.time_vector

    print(x.shape, t.shape)

    rise_time, decay_time = rise_decay_times(x, t)
    sound_exp_level = sel(x)
    running_sel_cum = sel_cum(np.array([running_sel_cum, sound_exp_level]))
    return Metrics(
        uid=detection.uid,
        station=detection.station,
        channel=detection.channel,
        time_init=detection.time_init,
        time_end=detection.time_end,
        unit=detection.unit,
        RL_rms=rl_rms(x),
        RL_pk=rl_pk(x),
        RL_pkpk=rl_pkpk(x),
        SEL=sound_exp_level,
        t_rise=rise_time,
        t_decay=decay_time,
        kurtosis=kurtosis(x),
        SEL_cum=running_sel_cum,
        metadata=detection.metadata,
    )


def compute_metrics(dbpath: Path) -> list[Metrics]:
    with h5py.File(dbpath, "r") as fr:
        all_metrics = []
        for chan in fr.keys():
            running_sel_cum = 0.0
            for k in fr[chan].keys():
                det = fr[chan][k]

                if "metadata" in det:
                    mdgrp = det["metadata"]
                    metadata = {k: mdgrp.attrs[k] for k in mdgrp.attrs.keys()}
                else:
                    metadata = None

                detection = Detection(
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

                metrics = compute_pulse_metrics(detection, running_sel_cum)
                running_sel_cum = metrics.SEL_cum
                all_metrics.append(metrics)

    return all_metrics


def main(args: argparse.Namespace) -> None:
    all_metrics = compute_metrics(args.dbpath)
    save_metrics_to_csv(all_metrics, args.destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dbpath",
        type=Path,
        default=Path(
            "/Users/wfjenkins/Research/Projects/RODEOII/data/detections/SHRU921_Tetra.h5"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(
            "/Users/wfjenkins/Research/Projects/RODEOII/data/metrics/SHRU921_Tetra.csv"
        ),
    )
    args = parser.parse_args()
    main(args)
