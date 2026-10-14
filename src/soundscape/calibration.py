"""Convert raw recordings to acoustic pressure (uPa)."""

from functools import cache
from pathlib import Path

import numpy as np

from soundscape.config import (
    CalibrationConfig,
    FlatCalibration,
    FrequencyResponseCalibration,
    NoCalibration,
)


def calibrate(
    config: CalibrationConfig, data: np.ndarray, fs: float, channels: list[int]
) -> np.ndarray:
    """Calibrate `data` (n_channels, n_samples) whose rows are `channels`."""
    data = np.atleast_2d(data)
    match config:
        case NoCalibration():
            return data
        case FlatCalibration():
            return counts_to_upa(data, config, channels)
        case FrequencyResponseCalibration():
            return volts_to_upa(data, fs, config)
    raise TypeError(f"Unknown calibration {type(config).__name__}")


def counts_to_upa(
    data: np.ndarray, config: FlatCalibration, channels: list[int]
) -> np.ndarray:
    volts = data * (config.adc_halfscale_v / config.adc_max_count)
    total_db = np.array(
        [config.sensitivity[c] + config.fixed_gain[c] for c in channels]
    )
    return volts / 10 ** (total_db[:, np.newaxis] / 20)


@cache
def _load_response(file: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(file, skiprows=1, delimiter=",")
    return table[:, 0], table[:, 1]


def volts_to_upa(
    data: np.ndarray, fs: float, config: FrequencyResponseCalibration
) -> np.ndarray:
    """Deconvolve a tabulated sensitivity (dB re 1 V/uPa) in the frequency domain.

    The sensitivity is interpolated linearly in frequency and held constant
    beyond the tabulated range.
    """
    freq_cal, sens_db = _load_response(Path(config.file))
    sens_linear = 10 ** ((sens_db + config.correction_db) / 20)
    n = data.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    sens = np.interp(freqs, freq_cal, sens_linear)
    return np.fft.irfft(np.fft.rfft(data, axis=-1) / sens, n=n, axis=-1)
