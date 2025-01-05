#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from pathlib import Path

import dotenv
import polars as pl

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> None:
    path = Path(os.getenv("METRICSPATH"))
    savepath = path / "metrics.csv"
    files = list(path.rglob("metrics_*.csv"))
    logging.info(f"Aggregating metrics found in {len(files)} files from {path.resolve()}")

    df = (
        pl.concat([pl.read_csv(f) for f in files]).sort("time_init")
    )
    df.write_csv(savepath)
    logging.info(f"{len(df):,} rows written to {savepath.resolve()}")


if __name__ == "__main__":
    main()
