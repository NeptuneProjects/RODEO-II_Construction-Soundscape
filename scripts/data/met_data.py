#!/usr/bin/env python3
"""Combine NDBC wind and wave records into one table for `[report] met`.

Reads NDBC standard meteorological files (e.g. `buzm3h2023.txt` for wind,
`44097h2023.txt` for waves), treats NDBC's missing-value codes (99, 999, 9999)
as missing, and writes one CSV of observations with columns `datetime`,
`source`, and the wind (WSPD, WDIR, GST) or wave (WVHT, DPD, APD, MWD, WTMP)
variables. Wind rows have no wave values and vice versa; the report averages
each variable over the observations within each event, ignoring missing
values.

Usage:
    python scripts/data/met_data.py --wind data/met/buzm3h2023.txt \\
        --wave data/met/44097h2023.txt --output data/met/met_2023.csv
"""

import argparse
from pathlib import Path

import polars as pl

WIND = ["WSPD", "WDIR", "GST"]
WAVE = ["WVHT", "DPD", "APD", "MWD", "WTMP"]


def load_ndbc(path: Path, columns: list[str]) -> pl.DataFrame:
    """One NDBC stdmet file as datetime plus `columns`, missing codes as null."""
    with open(path) as f:
        header = f.readline().lstrip("#").split()
        rows = [line.split() for line in f if not line.startswith("#")]
    df = pl.DataFrame(rows, schema=header, orient="row")
    time = pl.datetime(
        *(pl.col(c).cast(pl.Int32) for c in ("YY", "MM", "DD", "hh", "mm"))
    )
    values = []
    for c in columns:
        v = pl.col(c).cast(pl.Float64)
        # NDBC fills missing values with 99, 999, or 9999 (by column width).
        values.append(
            pl.when(v.is_in([99.0, 999.0, 9999.0])).then(None).otherwise(v).alias(c)
        )
    return df.select(time.alias("datetime"), *values)


def main(args: argparse.Namespace) -> None:
    wind = load_ndbc(args.wind, WIND).with_columns(
        pl.lit(args.wind.stem).alias("source")
    )
    wave = load_ndbc(args.wave, WAVE).with_columns(
        pl.lit(args.wave.stem).alias("source")
    )
    df = pl.concat([wind, wave], how="diagonal").sort("datetime")
    df = df.select("datetime", "source", *WIND, *WAVE)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.output)
    for name, part, cols in (("wind", wind, WIND), ("wave", wave, WAVE)):
        missing = {c: part[c].null_count() for c in cols if part[c].null_count()}
        print(f"{name}: {len(part):,} observations, missing {missing or 'none'}")
    print(f"-> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wind", type=Path, required=True, help="NDBC wind file")
    parser.add_argument("--wave", type=Path, required=True, help="NDBC wave file")
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
