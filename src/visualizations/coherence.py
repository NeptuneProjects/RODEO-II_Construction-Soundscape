#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import itertools
from pathlib import Path
import warnings

from datautils.data import read
from datautils.query import CatalogueQuery
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

import helpers
import plotting


def main(args):
    time_start = np.datetime64(args.time_start, "us")
    time_end = np.datetime64(args.time_end, "us")
    if args.time_interval % 1 != 0:
        warnings.warn(
            "Time interval is not an integer. Rounding to the nearest integer."
        )
        args.time_interval = round(args.time_interval)
    time_interval = np.timedelta64(args.time_interval, args.time_units)
    if time_end - time_start < time_interval:
        time_interval = time_end - time_start
    num_intervals = int((time_end - time_start) / time_interval)
    channels = [int(c) for c in args.channels.split(",")]

    for i in range(num_intervals):
        time_start_i = time_start + i * time_interval
        time_end_i = time_start + (i + 1) * time_interval
        
        cat_query = CatalogueQuery(
            catalogue=args.catalogue,
            destination=args.destination,
            time_start=time_start_i,
            time_end=time_end_i,
            channels=channels,
        )
        ds = read(cat_query)
        ds = ds.decimate(factor=int(ds.stats.sampling_rate / 2000))
        ds = ds.filter(filt_type="lowpass", freq=50.0)
        
        time = ds.time_vector
        fs = ds.stats.sampling_rate

        for j, channel in enumerate([channels[1]]):
            sig = ds.waveform[:, j]
            all_peaks, _ = signal.find_peaks(sig, height=args.height, distance=2000)

            t_corr_front = 1.0
            t_corr_back = 0.5
            n_corr_front = int(t_corr_front * fs)
            n_corr_back = int(t_corr_back * fs)

            t_plot_front = 2.0
            t_plot_back = 2.0
            n_plot_front = int(t_plot_front * fs)
            n_plot_back = int(t_plot_back * fs)


            # Remove peaks that are too close to the edges
            # [peaks.remove(p) for p in peaks if p < n_plot_front or p > len(sig) - n_plot_back]
            peaks = all_peaks[(all_peaks > n_plot_front) & (all_peaks < len(sig) - n_plot_back)]

            # if peaks[0] < n_corr_front:
            #     peaks = peaks[1:]
            # if peaks[-1] > len(sig) - n_corr_back:
            #     peaks = peaks[:-1]

            num_to_stack = 20
            n_peaks = len(peaks) - num_to_stack
            print(n_peaks)

            corr_matrix = np.full((n_peaks, num_to_stack), np.nan)
            lag_matrix = np.full((n_peaks, num_to_stack), np.nan)
            time_matrix = np.full((n_peaks, num_to_stack), np.nan)

            for k in range(n_peaks):
                # print(f"Computing for pulse {k} of {n_peaks}...")
                if k == 0:
                    sig_matrix = np.full((num_to_stack, n_corr_front + n_corr_back), np.nan)
                for l in range(num_to_stack):
                    in1 = signal.detrend(sig[peaks[k] - n_corr_front : peaks[k] + n_corr_back])
                    in2 = signal.detrend(sig[peaks[k + l] - n_corr_front : peaks[k + l] + n_corr_back])

                    # an1 = signal.hilbert(in1)
                    # an2 = signal.hilbert(in2)
                    # env1 = np.abs(an1)
                    # env2 = np.abs(an2)

                    # in1 /= env1.std()
                    # in2 /= env2.std()

                    corr = signal.correlate(in1, in2, method="direct", mode="full")
                    
                    lags = signal.correlation_lags(len(in1), len(in2), mode="full")

                    max_corr_ind = np.argmax(corr)
                    max_corr = corr[max_corr_ind]
                    max_lag = lags[max_corr_ind]

                    corr_matrix[k, l] = max_corr / (in1.std() * in2.std() * len(in1))
                    lag_matrix[k, l] = max_lag
                    time_matrix[k, l] = (time[peaks[k + l]] - time[peaks[k]]) / 1e6
                    if k == 0:
                        sig_matrix[l, :] = in2

                    # plt.figure()
                    # plt.subplot(3, 1, 1)
                    # plt.plot(in1)
                    # plt.plot(in2)
                    # plt.subplot(3, 1, 2)
                    # plt.plot(corr)
                    # plt.title(f"Max Lag = {max_lag}")
                    # plt.tight_layout()
                    # plt.subplot(3, 1, 3)
                    # plt.plot(in1)
                    # plt.plot(helpers.shift_signal(in2, max_lag))
                    # plt.show()

            filename_append = (
                np.datetime_as_string(time_start_i, unit="s")
                .replace("-", "")
                .replace("T", "_")
                .replace(":", "")
                + "_"
                + args.title.replace("/", "_")
                + f"_ch{channel}"
            )
            # np.save(
            #     Path.cwd() / "data" / "coherence" / f"corr_matrix_{filename_append}.npy", corr_matrix
            # )
            # np.save(Path.cwd() / "data" / "coherence" / f"lag_matrix_{filename_append}.npy", lag_matrix)
            # np.save(
            #     Path.cwd() / "data" / "coherence" / f"time_matrix_{filename_append}.npy", time_matrix
            # )
            # np.save(Path.cwd() / "data" / "coherence" / f"sig_matrix_{filename_append}.npy", sig_matrix)

            title = (
                f"{args.title}\n"
                f"Reference Peak: {time[peaks[0]]}"
            )

            args.destination.mkdir(parents=True, exist_ok=True)
            fig = plotting.plot_pulse_coherence(
                time_matrix, corr_matrix, n_peaks, title, channel
            )
            # fig.savefig(
            #     args.destination
            #     / f"{filename_append}.png",
            #     bbox_inches="tight",
            #     dpi=300
            # )
            plt.draw()
            # plt.close()

            # Shift the signals
            shifted_matrix = np.array(
                [
                    helpers.shift_signal(sig_matrix[i], int(lag_matrix[0, i]))
                    for i in range(20)
                ]
            )

            fig = plotting.plot_pulse_stack(shifted_matrix, corr_matrix, fs, title, channel)
            # fig.savefig(
            #     args.destination.parent / "pulse_stack"
            #     / f"__{filename_append}.png",
            #     bbox_inches="tight",
            #     dpi=300
            # )
            plt.draw()
            # plt.close()

            fig, _ = plotting.plot_summed_pulses(shifted_matrix, fs=fs)
            # TODO: De-couple signal summation from plotting
            # fig.savefig(
            #     args.destination.parent / "pulse_sum"
            #     / f"__{filename_append}.png",
            #     bbox_inches="tight",
            #     dpi=300
            # )
            plt.draw()
            # plt.close()

        plt.show()

            # corr_matrix = np.full((n_peaks, n_peaks), np.nan)
            # lag_matrix = np.full((n_peaks, n_peaks), np.nan)

            # Pre-compute all signal segments
            # segments = [sig[pk - 1000:pk + 1001] for pk in peaks]

            # for k in range(n_peaks):
            #     for l in range(k+1, n_peaks):
            #         in1 = segments[k]
            #         in2 = segments[l]
            #         corr = signal.correlate(in1, in2)
            #         lags = signal.correlation_lags(len(in1), len(in2))
            #         max_corr_ind = np.argmax(corr)
            #         max_corr = corr[max_corr_ind]
            #         max_lag = lags[max_corr_ind]
            #         corr_matrix[k, l] = max_corr
            #         corr_matrix[l, k] = max_corr  # Use symmetry
            #         lag_matrix[k, l] = max_lag
            #         lag_matrix[l, k] = max_lag    # Use symmetry

            # np.save(f"lag_matrix_{i}_{j}.npy", lag_matrix)
            # np.save(f"corr_matrix_{i}_{j}.npy", corr_matrix)

            # plt.figure()
            # ax = plt.gca()
            # im = ax.imshow(corr_matrix)
            # ax.set_title("Correlation")
            # cbar = ax.figure.colorbar(im, ax=ax)

            # plt.figure()
            # ax = plt.gca()
            # im = ax.imshow(lag_matrix)
            # ax.set_title("Lag")
            # cbar = ax.figure.colorbar(im, ax=ax)

            # plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectrogram plotting routine")
    parser.add_argument(
        "--catalogue",
        help="Path to the catalogue file",
        type=Path,
        default="data/catalogues/SHRU920_VLA2_FileInfo.csv",
    )
    parser.add_argument(
        "--destination",
        help="Path to the destination directory",
        type=Path,
        default="reports/pulse_coherence",
    )
    parser.add_argument("--time_interval", help="Time interval", type=int, default=60)
    parser.add_argument("--time_units", help="Time units", type=str, default="m")
    parser.add_argument(
        "--time_start", help="Start time", type=str, default="2023-07-09T10:24:55"
    )
    parser.add_argument(
        # "--time_end",
        # help="End time",
        # type=str,
        # default="2023-07-09T10:26:10",
        "--time_end", help="End time", type=str, default="2023-07-09T10:30:00"
    )
    parser.add_argument(
        "--channels", help="Channels to be read", type=str, default="0,1,2,3"
    )
    parser.add_argument(
        "--title", help="Title of the plot", type=str, default="SHRU920/VLA2"
    )
    parser.add_argument("--tperseg", help="Window size in seconds", type=int, default=1)
    parser.add_argument(
        "--overlap", help="Overlap size in percentage", type=float, default=0.75
    )
    parser.add_argument(
        "--nfft", help="Number of data points used in each block", type=int
    )
    parser.add_argument(
        "--fmin", help="Minimum frequency in Hz", type=float, default=20.0
    )
    parser.add_argument(
        "--fmax", help="Maximum frequency in Hz", type=float, default=300.0
    )
    parser.add_argument(
        "--vmin", help="Minimum value for the color scale", type=float, default=60
    )
    parser.add_argument(
        "--vmax", help="Maximum value for the color scale", type=float, default=120
    )
    parser.add_argument(
        "--height", type=float, default=1.5e7
    )

    args = parser.parse_args()
    main(args)
