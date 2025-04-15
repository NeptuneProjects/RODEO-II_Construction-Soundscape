#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from tritonoa.signal.util import double_to_single_sided_fft, double_to_single_sided_ifft

from rodeo import model as mdl
from rodeo import plotting, utils

logging.basicConfig(**utils.logging_kwargs)


def main(args: argparse.Namespace) -> None:
    ctd_df = utils.load_ssp(args.ssp)
    env = mdl.build_base_environment(ctd_df)
    dr = 1000
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

    time, source_signals_time, freq, source_signals_freq = mdl.generate_source_signal(
        type="point",
        num_signals=num_signals,
        signal_duration=signal_duration,
        fs=fs,
        pulse_duration=pulse_duration,
        fc=starting_fc,
        snr_db=snr_db,
    )

    fig = plotting.plot_signal(
        time,
        source_signals_time,
        *double_to_single_sided_fft(freq, source_signals_freq),
        "Source Signals",
    )
    # plt.draw()
    fig.savefig(
        args.output / "source_signals.png", dpi=300, bbox_inches="tight", facecolor="w"
    )

    comparison_pulse = 1
    s1 = source_signals_time[0]
    s2 = source_signals_time[comparison_pulse]
    S1 = source_signals_freq[0]
    S2 = source_signals_freq[comparison_pulse]
    # Compute auto-power spectra
    P11 = np.abs(S1) ** 2
    P22 = np.abs(S2) ** 2
    # Compute cross-power spectra
    P12 = S1 * np.conj(S2)

    # Compute inverse FFTs
    p11 = np.real_if_close(fft.ifft(P11, n=num_samples))
    p22 = np.real_if_close(fft.ifft(P22, n=num_samples))
    p12 = np.real_if_close(fft.ifft(P12, n=num_samples))

    # Compute coherence
    R12 = np.abs(P12) ** 2 / np.sqrt(P11 * P22)
    norm = np.sqrt(np.sum(s1**2) * np.sum(s2**2))
    r12 = np.real_if_close(fft.ifft(R12, n=num_samples) / norm)

    fig = plotting.plot_signal(
        *double_to_single_sided_ifft(
            time,
            p11,
        ),
        *double_to_single_sided_fft(freq, P11),
        "Auto-spectral power of source signal 1",
    )
    # plt.draw()
    fig.savefig(
        args.output / f"auto_spectral_power_1.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )

    fig = plotting.plot_signal(
        *double_to_single_sided_ifft(
            time,
            p22,
        ),
        *double_to_single_sided_fft(freq, P22),
        "Auto-spectral power of source signal 2",
    )
    # plt.draw()
    fig.savefig(
        args.output / f"auto_spectral_power_2.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )

    fig = plotting.plot_signal(
        *double_to_single_sided_ifft(
            time,
            p12,
        ),
        *double_to_single_sided_fft(freq, P12),
        "Cross-spectral power of source signals",
    )
    # plt.draw()
    fig.savefig(
        args.output / f"cross_spectral_power_source.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )

    fig = plotting.plot_signal(
        *double_to_single_sided_ifft(time, r12),
        *double_to_single_sided_fft(freq, R12),
        "Cross-correlation of signals at source",
    )
    # plt.draw()
    fig.savefig(
        args.output / f"cross_correlation_source.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )

    corr_coeff = []
    for r in ranges:
        logging.info(f"Processing range: {r} m")

        chan_resp_time, chan_resp_freq = mdl.compute_greens_function(
            env, (40, fs), fs, num_samples, r, freq
        )

        fig = plotting.plot_signal(
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

        Y1 = S1 * chan_resp_freq
        Y2 = S2 * chan_resp_freq
        y1 = fft.ifft(Y1, n=num_samples)
        y2 = fft.ifft(Y2, n=num_samples)

        # Compute cross-power spectra
        Y12 = P12 * np.abs(chan_resp_freq) ** 2

        # Compute inverse FFTs
        y12 = np.real_if_close(fft.ifft(Y12, n=num_samples))

        # Compute coherence
        RY12 = (
            np.abs(S1 * np.conj(S2)) ** 2
            * np.abs(chan_resp_freq) ** 2
            / (np.abs(S1) * np.abs(S2))
        )

        # Compute cross-correlation
        normy = np.sqrt(np.sum(np.abs(y1) ** 2) * np.sum(np.abs(y2) ** 2))
        ry12 = np.real_if_close(fft.ifft(RY12, n=num_samples) / normy)

        print(
            f"Max cross-correlation at source: {np.max(r12)} | "
            f"Max cross-correlation at receiver: {np.max(ry12)}"
        )

        corr_coeff.append(np.max(ry12))

        fig = plotting.plot_signal(
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

        fig = plotting.plot_signal(
            # *double_to_single_sided_ifft(
            #     time,
            #     y1,
            # ),
            time,
            y2,
            *double_to_single_sided_fft(freq, Y2),
            "Received Signal 2",
        )
        # plt.draw()
        fig.savefig(
            args.output / f"received_signal_2_{r}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="w",
        )

        fig = plotting.plot_signal(
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

        fig = plotting.plot_signal(
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
