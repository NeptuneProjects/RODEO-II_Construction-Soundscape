#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import tomllib

from dotenv import load_dotenv
import numpy as np
from tritonoa.data.inventory import Inventory
from tritonoa.data.signal import SignalParams
from tritonoa.data.time import ClockParameters

load_dotenv()


def main(args: argparse.Namespace) -> None:
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    for key in config:
        print(f"Building inventory for {key}...")
        
        data_cfg = config[key]["data"]
        clock_cfg = config[key]["clock"]
        hyd_cfg = config[key]["hydrophone"]

        time_check_0 = np.datetime64(clock_cfg["time_check_0"])
        time_check_1 = np.datetime64(clock_cfg["time_check_1"])

        clock_params = ClockParameters(
            time_check_0=time_check_0,
            time_check_1=time_check_1,
            offset_0=clock_cfg["offset_0"],
            offset_1=clock_cfg["offset_1"],
        )
        signal_params = SignalParams(
            gain=hyd_cfg["fixed_gain"],
            sensitivity=hyd_cfg["sensitivity"],
        )

        inv = Inventory()
        inv.build(
            dataset_path=data_cfg["directory"],
            glob_pattern=data_cfg["glob_pattern"],
            clock_params=clock_params,
            conditioner=signal_params,
        )
        savepath = Path(data_cfg["destination"]) / f"{key}_FileInfo.csv"
        inv.save(savepath)

        print(f"Inventory saved to {savepath.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save catalogues")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the configuration file",
        default=Path(os.getenv("INVCONFIG")),
    )
    args = parser.parse_args()
    main(args)
