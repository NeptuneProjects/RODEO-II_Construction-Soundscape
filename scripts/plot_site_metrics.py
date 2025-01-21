#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import dotenv
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

dotenv.load_dotenv()
plt.style.use(Path("config/jasa_fig_style.mplstyle"))

METRICS = {
    "SPL_rms": "$L_{p,\\text{rms}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
    "SPL_pk": "$L_{p,\\text{pk}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
    "SPL_pkpk": "$L_{p,\\text{pk-pk}}$\n(dB re 1 $\\mathrm{\\mu Pa}$)",
    "SEL": "$L_{E,p}$\n(dB re 1 $\\mathrm{\\mu Pa^2 s}$)",
    "SEL_cum": "cSEL\n(dB re 1 $\\mathrm{\\mu Pa^2 s}$)",
    "t_rise": "$t_\\mathrm{r}$ (s)",
    "t_decay": "$t_\\mathrm{d}$ (s)",
    "kurtosis": "$\\beta_2$",
}
YLIM_SCALE = 1.1

def plot_event_metrics(df: pl.DataFrame) -> None:
    events = sorted(df["event_name"].unique().to_list())
    stations = sorted(df["station"].unique().to_list(), key=lambda x: x.split("_")[-1])

    for k, event in enumerate(events):
        print(f"Plotting event: {event}")

        total_strikes = None

        fig, axs = plt.subplots(
            nrows=len(METRICS),
            ncols=len(stations),
            figsize=(8, 10),
            gridspec_kw={"hspace": 0.1, "wspace": 0.05},
        )

        event_start = (
            df.filter(pl.col("event_name") == event)["time_init"].unique().min()
        )
        event_end = df.filter(pl.col("event_name") == event)["time_end"].unique().max()

        for i, (metric_key, metric_name) in enumerate(METRICS.items()):

            min_val = np.inf
            max_val = 0.0

            for j, station in enumerate(stations):

                station_name = station.replace("_", "/")

                df_sta = df.filter(
                    (pl.col("event_name") == event) & (pl.col("station") == station)
                )
                if df_sta.is_empty():
                    ax = axs[i, j]
                    ax.text(
                        0.5,
                        0.5,
                        "No data",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                else:
                    distance = df_sta["range"].unique().item()
                    data = df_sta.select(["channel", "index", metric_key])

                    for l, ch in enumerate(sorted(data["channel"].unique().to_list())):
                        df_ch = df_sta.filter(pl.col("channel") == ch)
                        time = df_ch["time_init"].to_numpy()
                        data = df_ch[metric_key].to_numpy()
                        if not total_strikes:
                            total_strikes = len(time)

                        if max_val < max(data):
                            max_val = max(data)
                        if min_val > min(data):
                            min_val = min(data)

                        ax = axs[i, j]

                        ax.scatter(
                            time,
                            data,
                            alpha=0.5,
                            s=1.0,
                            color="k",
                            zorder=10 + l,
                            label=f"CH {ch}",
                        )

                ax.grid()

                # Control column titles
                if i == 0:
                    if df_sta.is_empty():
                        ax.set_title(station_name)
                    else:
                        ax.set_title(f"{station_name}, R={distance / 1000:.1f} km")

                # Control row titles & ytick labels
                if j == 0:
                    ax.set_ylabel(metric_name)
                else:
                    ax.set_yticklabels([])

                # Control x-axis ticks and labels
                if i == len(METRICS) - 1:
                    ax.set_xlabel("Strike Index")
                    # ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                else:
                    ax.set_xticklabels([])
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

            # Controlx x-axis limits:
            xlim_margin = 0.02 * (time.max() - time.min())
            xlim = (time.min() - xlim_margin, time.max() + xlim_margin)

            for j in range(len(stations)):
                axs[i, j].set_xlim(xlim)

            # Control y-axis limits
            for j in range(len(stations)):
                ylim_margin = 0.1 * (max_val - min_val)
                axs[i, j].set_ylim(min_val - ylim_margin, max_val + ylim_margin)

        fig.suptitle(
            f"Event {event} (all channels) | Total strikes: {total_strikes:,}\n{event_start.strftime("%d-%b-%Y %H:%M")} - "
            f"{event_end.strftime("%d-%b-%Y %H:%M %Z")}",
            y=0.935,
        )

        savepath = (
            Path(os.getenv("FIGPATH")) / "events" / f"{event}_{event_start.strftime("%Y-%m-%d")}" / "event_metrics.png"
        )
        # plt.draw()
        # if k > 0:
        #     return
        fig.savefig(savepath, dpi=600, bbox_inches="tight", facecolor="white")
        plt.close()


def main(args: argparse.Namespace) -> None:
    df = pl.read_csv(args.input, try_parse_dates=True)
    # with pl.Config(tbl_cols=-1):
    #     print(df)
    plot_event_metrics(df)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot site metrics from CSV file.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(os.getenv("METRICSFILE")),
        help="Path to the input CSV file.",
    )
    args = parser.parse_args()
    main(args)
