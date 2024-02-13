#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

from datautils.catalogue import build_and_save_catalogues
from datautils.query import load_file_query

config_path = Path(__file__).parent / "config"


def main():
    queries = load_file_query(config_path / "catalogue.toml")
    build_and_save_catalogues(queries, fmt=["csv", "mat"])


if __name__ == "__main__":
    main()
