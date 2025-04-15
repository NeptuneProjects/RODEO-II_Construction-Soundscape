#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.fft as fft
import scipy.linalg as linalg
import tritonoa.modeling.physics as physics
from tritonoa.signal.util import double_to_single_sided_fft, double_to_single_sided_ifft

from rodeo import model as mdl, plotting, utils

logging.basicConfig(**utils.logging_kwargs)


def offset_traces(
    arr: npt.NDArray[np.complex_], offset: float
) -> npt.NDArray[np.complex_]:
    """Space traces by adding a constant to each column."""
    return np.array([col + i * offset for i, col in enumerate(arr)])


def main(args: argparse.Namespace) -> None:
    # 1. Define an environment
    # TODO: Load actual sensor positions for given event.
    range_m = args.range_m
    ctd_df = utils.load_ssp(args.ssp)
    env = mdl.build_base_environment(ctd_df)
    env.update({"rec_r": range_m / 1e3, "rec_z": [16.2, 17.7, 21.3, 22.8]})
    # env.update({"rec_r": range_m / 1e3})

    # 2. Generate signal
    fs = args.fs
    time, source_signal_time, freq, source_signal_freq = mdl.generate_source_signal(
        type="point",
        num_signals=1,
        signal_duration=args.signal_duration,
        fs=fs,
        pulse_duration=args.pulse_duration,
        fc=args.fc,
        snr_db=args.snr_db,
    )
    num_samples = len(time)

    fig = plotting.plot_signal(
        time,
        source_signal_time,
        *double_to_single_sided_fft(freq, source_signal_freq),
        "Source Signal",
    )
    plt.draw()

    # 3. Compute Green's Function
    chan_resp_time, chan_resp_freq = mdl.compute_greens_function(
        env, (40, fs), fs, num_samples, range_m, freq
    )
    time_spacing = np.max(np.abs(chan_resp_time))
    freq_spacing = np.max(np.abs(chan_resp_freq))
    fig = plotting.plot_signal(
        *double_to_single_sided_ifft(time, offset_traces(chan_resp_time, time_spacing)),
        *double_to_single_sided_fft(
            freq, offset_traces(np.abs(chan_resp_freq), freq_spacing)
        ),
        "Channel Response",
    )
    plt.draw()

    # 4. Propagate signal
    time_offset = physics.expected_travel_time(
        range_m, env["layerdata"][0]["c_p"].mean()
    )
    # time_offset = 0.0
    received_signal_time = physics.propagate_signal(
        source_signal_time,
        chan_resp_freq,
        freq,
        t_offset=time_offset,
        nfft=num_samples,
    )
    received_signal_freq = fft.fft(received_signal_time, n=num_samples)
    time_spacing = np.max(np.abs(received_signal_time))
    freq_spacing = np.max(np.abs(received_signal_freq))

    fig = plotting.plot_signal(
        # *double_to_single_sided_ifft(time, offset_traces(received_signal_time, time_spacing)),
        time,
        offset_traces(received_signal_time, time_spacing),
        *double_to_single_sided_fft(
            freq, offset_traces(np.abs(received_signal_freq), freq_spacing)
        ),
        "Received Signal",
    )
    plt.draw()

    # 4. Filter data
    ## Form covariance matrices for each frequency:
    freq_ss, data = double_to_single_sided_fft(freq, received_signal_freq)
    data = data.T
    noise = source_signal_time[:, len(time) // 2 :].squeeze()
    print(data.shape)

    noise_f = np.vstack([fft.fft(noise, n=num_samples) for _ in range(data.shape[-1])])
    freq_ss, noise_f = double_to_single_sided_fft(freq, noise_f)
    noise_f = noise_f.T
    print(noise_f.shape)

    covar_signal = np.einsum("ij,ik->ijk", data, np.conj(data))
    print(covar_signal.shape)
    covar_noise = np.einsum("ij,ik->ijk", noise_f, np.conj(noise_f))
    print(covar_noise.shape)

    f_ind = np.argmin(np.abs(freq_ss - args.fc))
    print(f"Frequency index: {f_ind}")
    print(f"Frequency: {freq_ss[f_ind]}")

    # plt.figure()
    # plt.imshow(np.abs(covar_signal[f_ind]), aspect="auto")
    # plt.colorbar()
    # plt.title("Signal covariance matrix")
    # plt.draw()

    # plt.figure()
    # plt.imshow(np.abs(covar_noise[f_ind]), aspect="auto")
    # plt.colorbar()
    # plt.title("Noise covariance matrix")
    # plt.draw()


    U, s, Vh = linalg.svd(covar_signal)




    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the point source model")
    parser.add_argument(
        "--ssp",
        type=Path,
        default=Path(
            "data/ctd/sfw_leg2_2023-07-23/CTD_cast_south_fork_07_23_2023_data.txt"
        ),
    )
    parser.add_argument(
        "--range_m",
        type=float,
        default=5000.0,
        help="Range to the receiver (m)",
    )
    parser.add_argument(
        "--fs", type=float, default=4096.0, help="Sampling frequency (Hz)"
    )
    parser.add_argument(
        "--signal_duration",
        type=float,
        default=0.5,
        help="Duration of the signal (s)",
    )
    parser.add_argument(
        "--pulse_duration", type=float, default=0.25, help="Duration of the pulse (s)"
    )
    parser.add_argument(
        "--fc",
        type=float,
        default=475.0,
        help="Center frequency (Hz)",
    )
    parser.add_argument(
        "--snr_db",
        type=float,
        default=30.0,
        help="Signal-to-noise ratio (dB)",
    )
    parser.add_argument("--output", type=Path, default=Path("results/model"))
    args = parser.parse_args()
    main(args)
