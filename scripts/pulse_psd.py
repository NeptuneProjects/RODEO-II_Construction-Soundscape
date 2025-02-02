#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
from pathlib import Path

import dotenv
import h5py
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)


def compute_psd(files: list[Path], output: Path, nperseg: int, noverlap: int) -> None:
    logging.info(f"Creating PSD file at {output.resolve()}")
    with h5py.File(output, "w") as fw:
        db_attrs_set = False

        for i, file in enumerate(files):
            sta_attrs_set = False
            event_name = file.parent.name.split("_")[0]
            station_name = "_".join(file.stem.split("_")[1:])

            log_msg_top = (
                f"[{i + 1}/{len(files)}] Processing {event_name} | {station_name}"
            )
            logging.info(log_msg_top)

            grp_sta = fw.create_group(f"{event_name} / {station_name}")
            grp_event = grp_sta.parent

            with h5py.File(file, "r") as fr:

                for chankey in fr:
                    logging.info(f"{log_msg_top} | {chankey}")

                    grp_chan = grp_sta.create_group(chankey)
                    grp_chan.attrs.update({"channel": chankey.split("=")[-1]})

                    for detkey in tqdm(fr[chankey], desc="Detections"):
                        data = retrieve_data(fr, chankey, detkey)
                        f, Pxx = welch(
                            data["data"],
                            fs=data["sampling_rate"],
                            nperseg=nperseg,
                            noverlap=noverlap,
                        )

                        # Assign database-specific metadata
                        if not db_attrs_set:
                            fw.attrs.update(
                                {
                                    "sampling_rate": data["sampling_rate"],
                                    "unit": data["unit"],
                                    "filter": data["filter"],
                                    "filter_freq": data["filter_freq"],
                                    "nperseg": nperseg,
                                    "f": f,
                                }
                            )
                            db_attrs_set = True
                        if not sta_attrs_set:
                            grp_sta.attrs.update({"range": data["range"]})

                        # Assign event-specific metadata
                        grp_event.attrs.update(
                            {
                                "event_name": data["event_name"],
                                "event_site": data["event_site"],
                            }
                        )

                        # Assign station-specific metadata
                        grp_sta.attrs.update(
                            {
                                "station": station_name,
                            }
                        )

                        # Store the PSD in the new file
                        grp_det = grp_chan.create_group(detkey)
                        grp_det.attrs.update(
                            {
                                "time_init": str(data["time_init"]),
                                "time_end": str(data["time_end"]),
                            }
                        )
                        grp_det.create_dataset("Pxx", data=Pxx)

    logging.info(f"PSD file created at {output.resolve()}")


def retrieve_data(f: h5py.File, chankey: str, detkey: str) -> dict:
    return {
        "channel": f[chankey][detkey].attrs["channel"],
        "time_init": np.datetime64(f[chankey][detkey].attrs["time_init"][:-1]),
        "time_end": np.datetime64(f[chankey][detkey].attrs["time_end"][:-1]),
        "sampling_rate": f[chankey][detkey].attrs["sampling_rate"],
        "unit": f[chankey][detkey].attrs["unit"],
        "data": f[chankey][detkey]["waveform"][:].squeeze(),
        "event_name": f[chankey][detkey]["metadata"].attrs["event_name"],
        "event_site": f[chankey][detkey]["metadata"].attrs["event_site"],
        "filter": f[chankey][detkey]["metadata"].attrs["filter"],
        "filter_freq": f[chankey][detkey]["metadata"].attrs["filter_freq"],
        "range": f[chankey][detkey]["metadata"].attrs["range"],
    }


def main(args: argparse.Namespace) -> None:
    files = list(Path.rglob(args.dbpath, "*.hdf5"))
    compute_psd(files, args.psdfile, args.nperseg, args.noverlap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the Power Spectral Density (PSD) of detections."
    )
    parser.add_argument(
        "--dbpath",
        type=Path,
        default=Path(os.getenv("DBPATH")),
        help="Path to the input file.",
    )
    parser.add_argument(
        "--psdfile",
        type=Path,
        default=Path(os.getenv("PSDFILE")),
        help="Path to the output file.",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=4096,
        help="Length of each segment for Welch's method.",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=int(0.5 * 4096),
        help="Number of points to overlap between segments.",
    )
    args = parser.parse_args()
    main(args)
