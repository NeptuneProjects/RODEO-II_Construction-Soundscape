#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

from datautils.catalogue import build_and_save_catalogues
from datautils.query import load_file_query
from dotenv import load_dotenv

load_dotenv()

def main(args: argparse.Namespace):
    queries = load_file_query(args.config)
    build_and_save_catalogues(queries, fmt=["csv", "mat"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save catalogues")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("CATCONFIG")),
    )
    args = parser.parse_args()
    main(args)
