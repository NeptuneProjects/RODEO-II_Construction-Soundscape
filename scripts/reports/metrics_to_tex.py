#!/usr/bin/env python3


import argparse
import os
from pathlib import Path

import dotenv
import pandas as pd
import polars as pl
from tqdm import tqdm

dotenv.load_dotenv()

COLUMNS = {
    "event_name": "Event",
    "station": "Station",
    "channel": "Chan.",
    "filter": "Filter",
    "filter_freq": r"\makecell{Freq\\(Hz)}",
    "range": r"\makecell{Range\\(km)}",
    "index": "Index",
    "time_init": "Time Init (UTC)",
    "time_end": "Time End (UTC)",
    "SPL_rms": r"\makecell{$L_{p,\textrm{rms}}$\\(dB re 1 $\mathrm{\mu Pa}$)}",
    "SPL_pk": r"\makecell{$L_{p,\textrm{pk}}$\\(dB re 1 $\mathrm{\mu Pa}$)}",
    "SPL_pkpk": r"\makecell{$L_{p,\textrm{pk-pk}}$\\(dB re 1 $\mathrm{\mu Pa}$)}",
    "SEL": r"\makecell{$L_{E,p}$\\(dB re 1 $\mathrm{\mu Pa^2 s}$)}",
    "SEL_cum": r"\makecell{cSEL\\(dB re 1 $\mathrm{\mu Pa^2 s}$)}",
    "t_rise": "$t_\\mathrm{r}$ (s)",
    "t_decay": "$t_\\mathrm{d}$ (s)",
    "kurtosis": "$\\beta_2$",
}


def write_to_latex(df: pd.DataFrame, output: Path) -> None:
    with open(output, "w") as f:
        f.write("\\begin{longtable}{llrlrrrllrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(df.columns) + " \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endfirsthead\n")
        f.write("\\toprule\n")
        f.write(" & ".join(df.columns) + " \\\\\n")
        f.write("\\midrule\n")
        f.write("\\endhead\n")
        f.write("\\midrule\n")
        f.write(
            f"\\multicolumn{{{len(df.columns)}}}{{r}}{{Continued on next page}} \\\\\n"
        )
        f.write("\\midrule\n")
        f.write("\\endfoot\n")
        f.write("\\bottomrule\n")
        f.write("\\endlastfoot\n")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            formatted_row = [
                f"{val:.2f}" if isinstance(val, float) else str(val)
                for val in row.values
            ]
            f.write(" & ".join(formatted_row) + " \\\\\n")

        f.write("\\end{longtable}\n")


def main(args: argparse.Namespace) -> None:
    df = (
        pl.read_csv(args.input)
        .drop(["uid", "unit", "event_site"])
        .with_columns((pl.col("range") / 1000).alias("range"))
        .with_columns(pl.col("filter_freq").cast(pl.Int32))
        .with_columns(pl.col("station").str.replace("_", "/"))
        .with_columns(pl.col("filter").str.replace("highpass", "Highpass"))
        .with_columns(pl.col("time_init").str.slice(0, 21))
        .with_columns(pl.col("time_end").str.slice(0, 21))
    )
    df = df.rename(COLUMNS).select(COLUMNS.values())
    write_to_latex(df.to_pandas(), args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the input CSV file containing the metrics",
        default=Path(os.getenv("METRICSFILE")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output PDF file",
        default=Path(os.getenv("METRICSTEXFILE")),
    )
    args = parser.parse_args()
    main(args)
