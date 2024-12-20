# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, window_size):
    if data.ndim == 1:
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")
    return np.apply_along_axis(
        lambda m: np.convolve(m, np.ones(window_size) / window_size, mode="valid"),
        axis=0,
        arr=data,
    )


def safe_log10(x, eps=1e-10):
    result = np.where(x > eps, x, -10)
    np.log10(result, out=result, where=result > 0)
    return result


def shift_signal(signal: np.ndarray, lag: int) -> np.ndarray:
    print(lag)
    if lag < 0:
        # Add zeros at the end
        padded = np.pad(signal, (0, -lag), 'constant')
        return padded[-lag:]
    elif lag > 0:
        # Add zeros at the beginning
        padded = np.pad(signal, (lag, 0), 'constant')
        return padded[:-lag]
    else:
        return signal
