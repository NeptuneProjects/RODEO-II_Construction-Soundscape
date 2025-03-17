#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.fft as fft
from tqdm import tqdm
from tritonoa.modeling.at.kraken import run_kraken
from tritonoa.signal import waveforms

from rodeo import model as mdl
from rodeo import utils

logging.basicConfig(**utils.logging_kwargs)


def plot_signal(
    time: np.ndarray,
    signal_t: np.ndarray,
    freq: np.ndarray,
    signal_f: np.ndarray,
    suptitle: str | None = None,
):
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    ax = axs[0]
    ax.plot(time, signal_t.T)
    # ax.plot(time, np.real(signal_t).T, "r--")
    # ax.plot(time, np.imag(signal_t).T, "g--")
    # ax.plot(time, np.abs(signal_t).T, "b--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time Domain")

    ax = axs[1]
    ax.plot(freq, np.abs(signal_f).T)
    # ax.plot(freq, np.real(signal_f.T), "r--")
    # ax.plot(freq, np.imag(signal_f.T), "g--")
    # ax.plot(freq, np.abs(signal_f.T), "b--")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Domain")

    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def build_base_environment(ctd_df: pd.DataFrame) -> dict:
    base_environment = mdl.construct_environment()
    layerdata = base_environment["layerdata"]
    z = ctd_df["Depth"].values
    c_p = ctd_df["Speed of sound"].values
    z = np.append(z, layerdata[0]["z"].max())
    c_p = np.append(c_p, c_p[-1])
    z = np.insert(z, 0, 0.0)
    c_p = np.insert(c_p, 0, layerdata[0]["c_p"])

    new_layerdata = [
        {"z": z, "c_p": c_p},
        layerdata[1],
    ]
    base_environment.update({"layerdata": new_layerdata})
    return base_environment


def double_to_single_sided_ifft(
    time_values: npt.NDArray[np.float_],
    ifft_values: npt.NDArray[np.complex_] | npt.NDArray[np.complexfloating],
    shift: bool = False,
) -> tuple[
    npt.NDArray[np.complex_] | npt.NDArray[np.complexfloating], npt.NDArray[np.float_]
]:
    """
    Extract the first half of the IFFT result for display or analysis purposes.
    Works with both single-channel and multi-channel data.

    Parameters:
    -----------
    time_values : array of float
        The time values corresponding to each IFFT sample
    ifft_values : array of complex or float
        The result of an inverse FFT (typically time-domain)
        Can be 1D array [n_time_points] for single-channel data
        or 2D array [n_channels, n_time_points] for multichannel data

    Returns:
    --------
    corresponding_times : array of float
        The time values corresponding to the first half
    first_half : array
        The first half of the IFFT result, with same dimensionality as input
    """
    # Get the total number of points (last dimension)
    n = ifft_values.shape[-1]

    if shift:
        ifft_values = fft.ifftshift(ifft_values, axes=-1)

    # Calculate the midpoint index
    # For even length, midpoint is n/2
    # For odd length, midpoint is (n+1)/2
    midpoint = (n + 1) // 2

    # Extract the first half of the data based on dimensionality
    if ifft_values.ndim == 1:
        first_half = ifft_values[:midpoint].copy()
    else:
        # For multi-channel data, keep all channels but take first half of time points
        first_half = ifft_values[..., :midpoint].copy()

    # Get corresponding time values
    corresponding_times = time_values[:midpoint]

    return corresponding_times, first_half


def double_to_single_sided_fft(
    freqs: npt.NDArray[np.float_],
    fft_values: npt.NDArray[np.complex_],
) -> tuple[npt.NDArray[np.complex_], npt.NDArray[np.float_]]:
    """
    Convert a double-sided FFT to a properly scaled single-sided FFT.
    Efficiently handles both single-channel and multichannel FFT data.

    Parameters:
    -----------
    freqs : array of float
        The corresponding frequencies for the FFT result (positive and negative)
    fft_values : array of complex
        The complex FFT result (double-sided)
        Can be 1D array [n_fft_points] for single-channel data
        or 2D array [n_channels, n_fft_points] for multichannel data

    Returns:
    --------
    positive_freqs : array of float
        The corresponding positive frequencies
    single_sided_fft : array of complex
        The single-sided FFT with proper scaling
        Maintains the same dimensionality as input fft_values
    """
    # Store original shape and dimensionality
    original_ndim = fft_values.ndim

    # Get number of FFT points (last dimension)
    n = fft_values.shape[-1]

    # Find indices where frequencies are >= 0
    positive_indices = freqs >= 0
    positive_freqs = freqs[positive_indices]

    # Create scaling factors array
    scale = np.ones(len(positive_freqs), dtype=float)

    # Set scaling for all frequencies except DC (and Nyquist if present)
    if n % 2 == 0:  # Even number of points
        # Scale all but first (DC) and last (Nyquist) elements
        scale[1:-1] = 2.0
    else:  # Odd number of points
        # Scale all but first (DC) element
        scale[1:] = 2.0

    # Handle indexing into the FFT values differently based on dimensionality
    # and apply scaling
    if original_ndim == 1:
        single_sided_fft = fft_values[positive_indices].copy() * scale
    else:
        # For multichannel data, we need to index the last dimension
        single_sided_fft = (
            fft_values[..., positive_indices].copy() * scale[np.newaxis, :]
        )

    return positive_freqs, single_sided_fft


def compute_greens_function(
    environment: dict,
    freq_bounds: tuple[float, float],
    fs: float,
    num_points: int,
    range_m: float,
    freq: npt.NDArray[np.float_],
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:

    def _enforce_symmetry(
        fft_data: npt.NDArray[np.complex_],
    ) -> npt.NDArray[np.complex_]:
        """Make FFT data conjugate symmetric to ensure real IFFT output"""
        n = len(fft_data)
        symmetric_fft = fft_data.copy()

        # DC component should be real
        symmetric_fft[0] = np.real(symmetric_fft[0])

        # Nyquist frequency component should be real (if n is even)
        if n % 2 == 0:
            symmetric_fft[n // 2] = np.real(symmetric_fft[n // 2])

        # Make all other frequencies conjugate symmetric
        for k in range(1, n // 2):
            symmetric_fft[n - k] = np.conj(symmetric_fft[k])

        return symmetric_fft

    def _greens_function(freq: float) -> npt.NDArray[np.complex_]:
        model = run_kraken(env | {"freq": freq})
        return model.modes.p.squeeze()

    def _get_freqs_from_band(
        num_points: int, freq_band: tuple[float, float], fs: float
    ):
        f = fft.fftfreq(num_points, d=1 / fs)
        inds = np.argwhere((f >= freq_band[0]) & (f <= freq_band[1])).squeeze()
        return f[inds], inds

    env = copy.deepcopy(environment)

    # Define frequencies
    freqs, _ = _get_freqs_from_band(num_points, freq_bounds, fs=fs)

    # Define the environment
    env.update({"rec_r": range_m / 1e3})

    # Compute Green's Function in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        chan_resp_f = list(
            tqdm(executor.map(_greens_function, freqs), total=len(freqs))
        )
    chan_resp_f_partial = np.array(chan_resp_f)
    padding = len(freq) - chan_resp_f_partial.shape[0]
    chan_resp_freq = fft.fftshift(
        np.concatenate([np.zeros(padding, dtype=complex), chan_resp_f_partial])
    )
    # Make chan_resp_freq conjugate symmetric:
    chan_resp_freq = _enforce_symmetry(chan_resp_freq)

    # Compute the IFFT of the channel response
    chan_resp_time = fft.ifft(chan_resp_freq * np.conj(chan_resp_freq), n=num_points)

    return chan_resp_time, chan_resp_freq


def point_source_signal(
    signal_duration: float,
    fs: float,
    pulse_duration: float,
    fc: float = 100.0,
    snr_db: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    pulse = waveforms.generate_gaussian_pulse(pulse_duration, fc, fs, sigma=0.005)

    t = np.arange(0, signal_duration, 1 / fs)
    num_samples = t.size
    x = np.concatenate([pulse, np.zeros(num_samples - pulse.size)])

    signal_power = np.mean(pulse**2)
    noise_rms = np.sqrt(signal_power / (10 ** (snr_db / 10)))
    noise = np.random.normal(0, 1, num_samples) * noise_rms
    return t, x + noise


def generate_point_source_signals(
    num_signals: int,
    signal_duration: float,
    fs: float,
    pulse_duration: float,
    fc: float = 100.0,
    snr_db: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_samples = int(signal_duration * fs)
    if isinstance(pulse_duration, float):
        pulse_duration = [pulse_duration] * num_signals
    if isinstance(fc, float):
        fc = [fc] * num_signals
    if isinstance(snr_db, float):
        snr_db = [snr_db] * num_signals

    source_signals_time = np.zeros((num_signals, num_samples))
    for i in range(num_signals):
        t, source_signals_time[i] = point_source_signal(
            signal_duration=signal_duration,
            fs=fs,
            pulse_duration=pulse_duration[i],
            fc=fc[i],
            snr_db=snr_db[i],
        )
    source_signals_freq = fft.fft(source_signals_time, axis=-1)
    freqvec = fft.fftfreq(num_samples, 1 / fs)

    return t, source_signals_time, freqvec, source_signals_freq


def generate_source_signal(
    type: str = "point", **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if type == "point":
        return generate_point_source_signals(**kwargs)
    else:
        raise ValueError(f"Invalid source type: {type}")


def load_ssp(ctd_file: Path) -> pd.DataFrame:
    ctd_df = utils.clean_ctd_data(
        pd.read_csv(ctd_file, parse_dates=["Time"]), min_depth=1.0
    )
    # ctd_df = ctd_df.iloc[700:1300].sort_values("Depth")
    # Find where the difference in depths changes sign:
    depth_diff = ctd_df["Depth"].diff()
    depth_diff_sign_change = depth_diff[depth_diff * depth_diff.shift(-1) < 0].index
    # Find the index of the first occurrence of the sign change:
    first_sign_change = depth_diff_sign_change[0]
    # Create a new DataFrame with the first sign change:
    ctd_df = ctd_df.iloc[:first_sign_change].sort_values("Depth")
    return ctd_df


def main(args: argparse.Namespace) -> None:
    ctd_df = load_ssp(args.ssp)
    env = build_base_environment(ctd_df)
    dr = 100
    ranges = np.arange(1000, 20000, dr)

    # Attributes shared by all pulses
    signal_duration = 0.5
    fs = 4096
    num_samples = int(signal_duration * fs)
    # Pulse-specific attributes
    num_signals = 4
    pulse_duration = 0.25
    starting_fc = [475.0 + 4 * i for i in range(num_signals)]
    snr_db = 30.0
    midpoint = num_samples // 2

    time, source_signals_time, freq, source_signals_freq = generate_source_signal(
        type="point",
        num_signals=num_signals,
        signal_duration=signal_duration,
        fs=fs,
        pulse_duration=pulse_duration,
        fc=starting_fc,
        snr_db=snr_db,
    )

    fig = plot_signal(
        time,
        source_signals_time,
        *double_to_single_sided_fft(freq, source_signals_freq),
        "Source Signals",
    )
    # plt.draw()
    fig.savefig(
        args.output / "source_signals.png", dpi=300, bbox_inches="tight", facecolor="w"
    )

    corr_coeff = []
    for r in ranges:
        logging.info(f"Processing range: {r} m")
        # Expected travel time:
        # c_max = 1500.0
        # t_offset = env["rec_r"] * 1e3 / c_max
        chan_resp_time, chan_resp_freq = compute_greens_function(
            env, (40, fs), fs, num_samples, r, freq
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(
                time,
                chan_resp_time,
            ),
            *double_to_single_sided_fft(freq, chan_resp_freq),
            "Channel Response",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"channel_response_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        comparison_pulse = -1

        s1 = source_signals_time[0]
        s2 = source_signals_time[comparison_pulse]

        S1 = source_signals_freq[0]
        S2 = source_signals_freq[comparison_pulse]

        Y1 = S1 * chan_resp_freq
        Y2 = S2 * chan_resp_freq
        y1 = fft.ifft(Y1, n=num_samples)
        y2 = fft.ifft(Y2, n=num_samples)

        # Compute auto-power spectra
        P11 = np.abs(S1) ** 2
        P22 = np.abs(S2) ** 2

        # Compute cross-power spectra
        P12 = S1 * np.conj(S2)
        Y12 = P12 * np.abs(chan_resp_freq) ** 2

        # Compute inverse FFTs
        p11 = np.real_if_close(fft.ifft(P11, n=num_samples))
        p22 = np.real_if_close(fft.ifft(P22, n=num_samples))
        p12 = np.real_if_close(fft.ifft(P12, n=num_samples))
        y12 = np.real_if_close(fft.ifft(Y12, n=num_samples))

        # Compute coherence
        R12 = np.abs(P12) ** 2 / np.sqrt(P11 * P22)
        RY12 = (
            np.abs(S1 * np.conj(S2)) ** 2
            * np.abs(chan_resp_freq) ** 2
            / (np.abs(S1) * np.abs(S2))
        )

        # Compute cross-correlation
        norm = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
        r12 = np.real_if_close(fft.ifft(R12, n=num_samples) / norm)
        normy = np.sqrt(np.sum(np.abs(y1) ** 2) * np.sum(np.abs(y2) ** 2))
        ry12 = np.real_if_close(fft.ifft(RY12, n=num_samples) / normy)

        print(
            f"Max cross-correlation at source: {np.max(r12)} | "
            f"Max cross-correlation at receiver: {np.max(ry12)}"
        )

        corr_coeff.append(np.max(ry12))

        fig = plot_signal(
            # *double_to_single_sided_ifft(
            #     time,
            #     y1,
            # ),
            time,
            y1,
            *double_to_single_sided_fft(freq, Y1),
            "Received Signal 1",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"received_signal_1_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(
                time,
                p11,
            ),
            *double_to_single_sided_fft(freq, P11),
            "Auto-spectral power of source signal 1",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"auto_spectral_power_1_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(
                time,
                p22,
            ),
            *double_to_single_sided_fft(freq, P22),
            "Auto-spectral power of source signal 2",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"auto_spectral_power_2_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(
                time,
                p12,
            ),
            *double_to_single_sided_fft(freq, P12),
            "Cross-spectral power of source signals",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"cross_spectral_power_source_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(
                time,
                y12,
            ),
            *double_to_single_sided_fft(freq, Y12),
            "Cross-spectral power of received signals",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"cross_spectral_power_received_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(time, r12),
            *double_to_single_sided_fft(freq, R12),
            "Cross-correlation of signals at source",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"cross_correlation_source_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plot_signal(
            *double_to_single_sided_ifft(time, ry12),
            *double_to_single_sided_fft(freq, RY12),
            "Cross-correlation of signals at receiver",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"cross_correlation_received_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )
        
        plt.close("all")

    corr_coeffs = np.array(corr_coeff)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(ranges, corr_coeffs)
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Correlation Coefficient")

    fig.savefig(
        args.output / "corrcoeff_v_range.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the point source model")
    parser.add_argument(
        "--ssp",
        type=Path,
        default=Path(
            "data/ctd/sfw_leg2_2023-07-23/CTD_cast_south_fork_07_23_2023_data.txt"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/model"))
    args = parser.parse_args()
    main(args)
