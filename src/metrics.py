#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import kurtosis

from detect import Detection

REF_PRESSURE = 1.0  # in uPa


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


def compute_metrics(detection: Detection):
    x = detection.waveform
    t = detection.time_vector

    rise_time, decay_time = rise_decay_times(x, t)

    metrics = {
        "RL_rms": rl_rms(x),
        "RL_pk": rl_pk(x),
        "RL_pkpk": rl_pkpk(x),
        "SEL": sel(x),
        "t_rise": rise_time,
        "t_decay": decay_time,
        "kurtosis": kurtosis(x),
    }
    return metrics


def save_metrics_to_hdf(metrics: dict):
    return


def main(args: argparse.Namespace) -> None:
    with h5py.File(args.dbpath, "r") as fr, h5py.File(args.destination, "a") as fw:
        for chan in fr.keys():
            running_sel_cum = 0.0
            for k in fr[chan].keys():
                det = fr[chan][k]

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
                )

                metrics = compute_metrics(detection)
                running_sel_cum = sel_cum(np.array([running_sel_cum, metrics["SEL"]]))
                metrics["SEL_cum"] = running_sel_cum

                


                print(metrics)
                return
    return


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
            "/Users/wfjenkins/Research/Projects/RODEOII/data/metrics/SHRU921_Tetra.h5"
        ),
    )
    args = parser.parse_args()
    main(args)
