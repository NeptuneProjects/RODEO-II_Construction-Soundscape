#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import irfft
from scipy.signal.windows import hann
from tqdm import tqdm
from tritonoa.modeling.at.kraken import run_kraken
from tritonoa.modeling.physics import propagate_signal
from tritonoa.signal import waveforms
from tritonoa.signal.util import get_freqs_from_band

from rodeo import model as mdl


def create_signal(fs: float):
    factor = 2.0
    fc = 100.0 * factor
    # bw = 195.0
    T = 40 / fc * factor
    dt = 1 / fs
    
    t = np.arange(0, T - dt, dt)
    pulse = waveforms.generate_gaussian_pulse(T, fc, fs, sigma=0.01)

    window = hann(len(pulse))
    pulse *= window
    T = 1.0
    t = np.arange(0, T, 1 / fs)
    N = t.size
    x = np.concatenate([pulse, np.zeros(N - pulse.size)])
    x += np.random.normal(0, 1e-8, N)
    return t, x


def run_point_source_model(env, freqs):
    # Compute Green's Function
    p = np.zeros((len(freqs),), dtype=complex)
    for i, freq in tqdm(enumerate(freqs), total=len(freqs)):
        model = run_kraken(env | {"freq": freq})
        p[i] = model.modes.p.squeeze()
    
    return p


def main():
    # Define source function
    fs = 8192.0 * 2
    t, x = create_signal(fs)
    
    # Define model
    env = mdl.construct_environment()
    
    # Define source
    fl = 100
    fu = 200
    freqs, inds = get_freqs_from_band(len(x), [fl, fu], fs=fs)
    
    # Expected travel time
    c_max = 1500.0
    t_offset = env["rec_r"] * 1e3 / c_max

    # Run point source model
    p = run_point_source_model(env, freqs)
    # Propagate signal
    y = propagate_signal(x, p, freqs, inds, t_offset, nfft=t.size)
    h = irfft(np.abs(p) * np.conj(p), n=len(x))
    

    plt.figure()
    plt.plot(t, x)
    plt.plot(t, np.real(y) / np.max(np.real(y)), label="Real")
    plt.show()

    return


if __name__ == "__main__":
    main()
