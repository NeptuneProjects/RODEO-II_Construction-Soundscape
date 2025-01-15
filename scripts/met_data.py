#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os
from pathlib import Path
import re

import dotenv
import polars as pl

dotenv.load_dotenv()


DATETIME_FORMAT = "%Y-%m-%dT%H:%M"
SCHEMA = pl.Schema(
    {
        "#YY": pl.String(),
        "MM": pl.String(),
        "DD": pl.String(),
        "hh": pl.String(),
        "mm": pl.String(),
        "WDIR": pl.Float32(),
        "WSPD": pl.Float32(),
        "GST": pl.Float32(),
        "WVHT": pl.Float32(),
        "DPD": pl.Float32(),
        "APD": pl.Float32(),
        "MWD": pl.Float32(),
        "PRES": pl.Float32(),
        "ATMP": pl.Float32(),
        "WTMP": pl.Float32(),
        "DEWP": pl.Float32(),
        "VIS": pl.Float32(),
        "PTDY": pl.Float32(),
    }
)
WAVE_SELECTION = [
    "#YY",
    "MM",
    "DD",
    "hh",
    "mm",
    "WVHT",
    "DPD",
    "APD",
    "MWD",
    "WTMP",
]
WIND_SELECTION = [
    "#YY",
    "MM",
    "DD",
    "hh",
    "mm",
    "WDIR",
    "WSPD",
    "GST",
    "PRES",
    "ATMP",
]


def convert_literal_to_datetime(literal: str) -> datetime:
    return pl.lit(literal).str.to_datetime(DATETIME_FORMAT)


def convert_whitespace_to_single_space(file: Path) -> bytes:
    """Converts multiple whitespaces to a single whitespace in a file.

    Polars library does not support multiple whitespaces in a file. This
    function reads the file, replaces multiple whitespaces with a single
    whitespace in memory, and returns the file as a byte string.

    Args:
        file: The path to the file.

    Returns:
        The file as a byte string.
    """
    with open(file, "r") as file:
        lines = file.readlines()

    processed_lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    return "\n".join(processed_lines).encode()


def load_data(fname: Path, select: list[str]) -> pl.DataFrame:
    data = convert_whitespace_to_single_space(fname)
    return pl.read_csv(
        data,
        columns=select,
        separator=" ",
        schema=SCHEMA,
        skip_rows_after_header=1,
    )


def condition_data(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.with_columns(
            pl.concat_str(
                pl.col("#YY"),
                pl.lit("-"),
                pl.col("MM"),
                pl.lit("-"),
                pl.col("DD"),
                pl.lit("T"),
                pl.col("hh"),
                pl.lit(":"),
                pl.col("mm"),
            ).alias("datetime")
        )
        .with_columns(
            pl.col("datetime").str.to_datetime(DATETIME_FORMAT).alias("datetime")
        )
        .drop(["#YY", "MM", "DD", "hh", "mm"])
    )
    return (
        df.select(["datetime"] + [col for col in df.columns if col != "datetime"])
        .upsample(time_column="datetime", every="1m")
        .interpolate()
        .fill_null(strategy="forward")
    )


def join_data(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    df = (
        df1.join(df2, on="datetime", how="inner")
        .group_by_dynamic(
            "datetime",
            every="1h",
            closed="left",
        )
        .agg([pl.all().mean()])
    )
    return df


def main(args: argparse.Namespace) -> None:
    df_wv = condition_data(load_data(args.wavedata, WAVE_SELECTION))
    df_wx = condition_data(load_data(args.winddata, WIND_SELECTION))
    df = join_data(df_wv, df_wx)
    df = df.filter(
        (pl.col("datetime") >= convert_literal_to_datetime(args.start))
        & (pl.col("datetime") <= convert_literal_to_datetime(args.end))
    )
    df.write_csv(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loads meteorological data.")
    parser.add_argument(
        "--wavedata",
        type=Path,
        default=Path(os.getenv("WAVEPATH")),
        help="The path to the meteorological data file.",
    )
    parser.add_argument(
        "--winddata",
        type=Path,
        default=Path(os.getenv("WINDPATH")),
        help="The path to the meteorological data file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv("METPATH")),
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-07-01T00:00",
        help="The start date and time.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2023-08-06T00:00",
        help="The end date and time.",
    )
    args = parser.parse_args()
    main(args)
