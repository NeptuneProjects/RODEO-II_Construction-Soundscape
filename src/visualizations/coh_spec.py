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
        title = (
            f"{args.title}\n"
            f"{np.datetime_as_string(time_start_i, unit='s')}"
            f" - {np.datetime_as_string(time_end_i, unit='s')}"
        )

        cat_query = CatalogueQuery(
            catalogue=args.catalogue,
            destination=args.destination,
            time_start=time_start_i,
            time_end=time_end_i,
            channels=channels,
        )
        ds = read(cat_query)
        time = ds.time_vector

        for j, channel in enumerate(channels):
            sig = ds.waveform[:, j]
            peaks, _ = signal.find_peaks(sig, height=args.height, distance=20000)

            # plt.plot(sig)
            # plt.plot(peaks, sig[peaks], "rx")
            # plt.show()

            if peaks[0] < 20000:
                peaks = peaks[1:]
            if peaks[-1] > len(sig) - 20000:
                peaks = peaks[:-1]

            num_to_stack = 20
            n_peaks = len(peaks) - num_to_stack
            print(n_peaks)

            corr_matrix = np.full((n_peaks, num_to_stack), np.nan)
            lag_matrix = np.full((n_peaks, num_to_stack), np.nan)
            time_matrix = np.full((n_peaks, num_to_stack), np.nan)

            for k in range(n_peaks):
                # print(f"Computing for pulse {k} of {n_peaks}...")
                if k == 0:
                    sig_matrix = np.full((20, 20000), np.nan)
                for l in range(20):
                    in1 = sig[peaks[k] - 10000 : peaks[k] + 10000]
                    in2 = sig[peaks[k + l] - 10000 : peaks[k + l] + 10000]
                    # in1 = signal.detrend(in1)
                    # in2 = signal.detrend(in2)
                    corr = signal.correlate(in1, in2)
                    lags = signal.correlation_lags(len(in1), len(in2))
                    max_corr_ind = np.argmax(corr)
                    max_corr = corr[max_corr_ind]
                    max_lag = lags[max_corr_ind]
                    corr_matrix[k, l] = max_corr / (in1.std() * in2.std() * len(in1))
                    lag_matrix[k, l] = max_lag
                    time_matrix[k, l] = (time[peaks[k + l]] - time[peaks[k]]) / 1e6
                    if k == 0:
                        sig_matrix[l, :] = in2

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
            #     Path.cwd()
            #     / "data"
            #     / "coherence"
            #     / f"corr_matrix_{filename_append}.npy",
            #     corr_matrix,
            # )
            # np.save(
            #     Path.cwd() / "data" / "coherence" / f"lag_matrix_{filename_append}.npy",
            #     lag_matrix,
            # )
            # np.save(
            #     Path.cwd()
            #     / "data"
            #     / "coherence"
            #     / f"time_matrix_{filename_append}.npy",
            #     time_matrix,
            # )
            # np.save(
            #     Path.cwd() / "data" / "coherence" / f"sig_matrix_{filename_append}.npy",
            #     sig_matrix,
            # )

            args.destination.mkdir(parents=True, exist_ok=True)
            # fig = plotting.plot_pulse_coherence(
            #     time_matrix, corr_matrix, n_peaks, title, channel
            # )
            # fig.savefig(
            #     args.destination / f"{filename_append}.png",
            #     bbox_inches="tight",
            #     dpi=300,
            # )
            # plt.close()

            # Shift the signals
            shifted_matrix = np.array(
                [
                    helpers.shift_signal(sig_matrix[i], int(lag_matrix[0, i]))[700:1500]
                    for i in range(10)
                ]
            )

            # fig = plotting.plot_pulse_stack(shifted_matrix, corr_matrix, title, channel)
            # fig.savefig(
            #     args.destination.parent / "pulse_stack" / f"{filename_append}.png",
            #     bbox_inches="tight",
            #     dpi=300,
            # )
            # plt.close()

            fig, summed_signal = plotting.plot_summed_pulses(shifted_matrix)
            # TODO: De-couple signal summation from plotting
            
            # plt.plot(summed_signal)
            
            fmin = 0
            fmax = 500

            vmin = 80
            vmax = 130

            window_sec = 1
            fs = ds.stats.sampling_rate
            nperseg = int(window_sec * fs)
            nfft = nperseg * 16
            overlap = 0.75
            noverlap = int(nperseg * overlap)

            print(
                nperseg,
                int(overlap * nperseg),
                noverlap,
                nfft,
                overlap,
                fs
            )

            f, t, Sxx = signal.spectrogram(
                summed_signal,
                fs=fs,
                nperseg=nperseg,
                noverlap=int(nperseg * overlap),
                # nfft=nfft,
            )
            if fmin and fmax:
                Sxx = Sxx[(f >= fmin) & (f <= fmax), :]
                f = f[(f >= fmin) & (f <= fmax)]
            elif fmin:
                Sxx = Sxx[f >= fmin, :]
                f = f[f >= fmin]
            elif fmax:
                Sxx = Sxx[f <= fmax, :]
                f = f[f <= fmax]
            
            fig = plt.figure()
            ax = plt.gca()
            im = ax.pcolormesh(t, f, 10 * np.log10(Sxx), cmap="jet", vmin=vmin, vmax=vmax)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Freq [Hz]")
            plt.show()
            

            
            # fig.savefig(
            #     args.destination.parent / "pulse_sum" / f"{filename_append}.png",
            #     bbox_inches="tight",
            #     dpi=300,
            # )
            # plt.close()

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
        "--time_start", help="Start time", type=str, default="2023-07-09T10:23:50"
    )
    parser.add_argument(
        "--time_end", help="End time", type=str, default="2023-07-09T10:26:10"
    )
    # parser.add_argument(
        # "--time_end", help="End time", type=str, default="2023-07-09T11:15:10"
    # )
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
    parser.add_argument("--height", type=float, default=1.5e7)

    args = parser.parse_args()
    main(args)
