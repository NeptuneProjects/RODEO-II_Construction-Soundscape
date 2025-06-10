#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rodeo.inventory import build_inventory

load_dotenv()


def main(args: argparse.Namespace) -> None:
    build_inventory(args.config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save inventories")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    args = parser.parse_args()
    main(args)
