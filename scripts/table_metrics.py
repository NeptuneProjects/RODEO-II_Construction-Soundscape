#!/usr/bin/env python


import argparse
from itertools import product
import os
from pathlib import Path

import dotenv
import h5py
import polars as pl

dotenv.load_dotenv()

STATIONS = ["SHRU921_Tetra", "SHRU905_VLA1", "SHRU920_VLA2"]
OMIT = ["A01", "A04", "A07", "A12-A", "Z01-A", "Z01-B", "Z01-C", "Z01-D"]


def average_met_data(df_event: pl.DataFrame, df_met: pl.DataFrame) -> pl.DataFrame:
    avg_wind = []
    avg_wave = []

    for event in df_event.iter_rows(named=True):
        start = event["Start"]
        end = event["Stop"]

        subset = df_met.filter(
            (pl.col("datetime") >= start) & (pl.col("datetime") <= end)
        )
        if subset.is_empty():
            avg_wind.append(None)
            avg_wave.append(None)
        else:
            avg_wind.append(subset["WSPD"].mean())
            avg_wave.append(subset["WVHT"].mean())

    return df_event.with_columns(
        [
            pl.Series("Wind Speed (m/s)", avg_wind),
            pl.Series("Sig. Wave Height (m)", avg_wave),
        ]
    )


def create_event_table(event: dict, df: pl.DataFrame, df_amb: pl.DataFrame) -> pl.DataFrame:
    TABLE_ROWS_SITE = {
        "Wind Speed (m/s)": "Wind Speed",
        "Sig. Wave Height (m)": "$H_{1/3}$",
    }
    TABLE_ROWS_STA = {
        "SPL_rms": "$L_{p,\\textrm{rms}}$",
        "SPL_pk": "$L_{p,\\textrm{pk}}$",
        "SPL_pkpk": "$L_{p,\\textrm{pk-pk}}$",
        "SEL": "$L_{E,p}$",
        "SEL_cum": "cSEL",
        "t_rise": "$t_\\mathrm{r}$",
        "t_decay": "$t_\\mathrm{d}$",
        "kurtosis": "$\\beta_2$",
    }

    stations = sorted(df["station"].unique().to_list(), key=lambda x: STATIONS.index(x))
    channels = df["channel"].unique().to_list()

    stats = []
    for station in stations:
        stats.append(df_amb.filter(pl.col("station") == station).drop(["event", "station"]))
    for row, station in product(TABLE_ROWS_STA.keys(), stations):
        row_name = f"{TABLE_ROWS_STA[row]} ({station.split('_')[-1]})"

        means, meds, stds, mins, maxs = [], [], [], [], []
        for channel in channels:
            subset = df.filter(
                (pl.col("station") == station) & (pl.col("channel") == channel)
            )
            means.append(subset[row].mean())
            meds.append(subset[row].median())
            stds.append(subset[row].std())
            mins.append(subset[row].min())
            maxs.append(subset[row].max())

        stats.append(
            pl.DataFrame(
                {
                    "Metric": row_name,
                    "Mean": format_stats(means),
                    "Median": format_stats(meds),
                    "Std": format_stats(stds),
                    "Min": format_stats(mins),
                    "Max": format_stats(maxs),
                }
            )
        )

    for row in TABLE_ROWS_SITE.keys():
        stats.append(
            pl.DataFrame(
                {
                    "Metric": TABLE_ROWS_SITE[row],
                    "Mean": f"{event[row]:.1f}",
                    "Median": "",
                    "Std": "",
                    "Min": "",
                    "Max": "",
                }
            )
        )

    

    df_table = pl.concat(stats)
    return df_table


def format_stats(values: list[float]) -> str:
    return "/".join([f"{val:.1f}" for val in values])


def load_ambient_noise(file: Path) -> pl.DataFrame:
    with h5py.File(file, "r") as f:
        events = list(f.keys())
        stations = []
        for grp in f.values():
            stations.extend(list(grp.keys()))
        stations = list(set(stations))
        stats = []
        for event, station in product(events, stations):
            if station not in f[event]:
                continue
            grp = f[event][station]
            amb = grp["SPL_rms"][:].mean(axis=1)

            stats.append(
                pl.DataFrame(
                    {
                        "event": event,
                        "station": station,
                        "Metric": f"Ambient $L_{{p,\\textrm{{rms}}}}$ ({station.split("_")[-1]})",
                        "Mean": format_stats(amb),
                        "Median": "",
                        "Std": "",
                        "Min": "",
                        "Max": "",
                    }
                )
            )

    return pl.concat(stats)


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    eventconfig = Path(os.getenv("EVENTCONFIG"))
    metpath = Path(os.getenv("METPATH"))
    metricspath = Path(os.getenv("METRICSFILE"))

    df_event = pl.read_csv(eventconfig, try_parse_dates=True).filter(
        ~pl.col("Event").is_in(OMIT)
    )
    df_met = pl.read_csv(metpath, try_parse_dates=True)
    df_metrics = pl.read_csv(metricspath, try_parse_dates=True)
    df_amb = load_ambient_noise(Path(os.getenv("AMBFILE")))

    return df_event, df_met, df_metrics, df_amb


def main(args: argparse.Namespace) -> None:
    df_event, df_met, df_metrics, df_amb = load_data()
    df_event = average_met_data(df_event, df_met)
    for event in df_event.iter_rows(named=True):
        subset = df_metrics.filter(pl.col("event_name") == event["Event"])
        subset_amb = df_amb.filter(pl.col("event") == event["Event"])
        table = create_event_table(event, subset, subset_amb)
        file = args.output / f"{event['Event']}" / "metrics.md"
        with (
            open(file, "w") as f,
            pl.Config(
                tbl_formatting="MARKDOWN",
                tbl_hide_column_data_types=True,
                tbl_hide_dataframe_shape=True,
                tbl_rows=-1,
                fmt_str_lengths=1000,
            ),
        ):
            print(table, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metrics table.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path.",
        type=str,
        default=Path(os.getenv("EVENTRESULTS")),
    )
    args = parser.parse_args()
    main(args)
