# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional, Union
import uuid

from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION
from dotenv import load_dotenv
import h5py
import numpy as np
from scipy import signal
from tqdm import tqdm


load_dotenv()

CATPATH = Path(os.getenv("CATPATH"))
DBPATH = Path(os.getenv("DBPATH"))


@dataclass
class Detection:
    uid: str
    station: str
    waveform: np.ndarray
    time_vector: np.ndarray
    channel: int
    time_init: np.datetime64
    time_end: np.datetime64
    sampling_rate: float
    unit: str
    metadata: Optional[dict] = None


def apply_linear_taper(time_series: np.ndarray, taper_length: int) -> np.ndarray:
    """
    Apply a linear taper to the beginning and end of a time series.

    Parameters:
    - time_series: numpy array, the original time series data
    - taper_length: int, length of the taper in data points

    Returns:
    - tapered_series: numpy array, the tapered time series data
    """
    series_length = len(time_series)
    taper = np.ones(series_length)

    # Create the linear taper
    taper[:taper_length] = np.linspace(0, 1, taper_length)
    taper[-taper_length:] = np.linspace(1, 0, taper_length)

    # Apply the taper to the time series
    return time_series * taper


def characteristic_function(y: np.ndarray, stwin: int, ltwin: int) -> np.ndarray:
    y_rms = rms(y, stwin)
    y_long = signal.convolve(y_rms, np.ones(ltwin) / ltwin, mode="same")
    return y_rms / y_long


def detect_pulses(
    ds: DataStream,
    stwin: float = 0.025,
    ltwin: float = 5.0,
    threshold: float = 2.25,
    distance: float = 0.75,
    taper: float = 5.0,
) -> np.ndarray:
    fs = ds.stats.sampling_rate
    time = ds.time_vector

    chf = characteristic_function(
        apply_linear_taper(ds.waveform, int(fs * taper)),
        int(stwin * fs),
        int(ltwin * fs),
    )
    peaks, _ = signal.find_peaks(chf, height=threshold, distance=distance * fs)
    return time[peaks]


def get_station_from_catalogue_name(fname: Union[Path, str]) -> str:
    return fname.stem.split("FileInfo")[0].rstrip("_")


def rms(y: np.ndarray, win: int) -> np.ndarray:
    return np.sqrt(signal.convolve(y**2, np.ones(win) / win, mode="same"))


def save_detections_to_hdf5(
    channel: Union[int, str], detections: list[Detection], file_path: str
) -> None:
    with h5py.File(file_path, "a") as f:
        for i, detection in enumerate(detections):
            main_group = f"chan={str(channel)}"
            # grp = f.create_group(f"chan={str(channel)}_det={i:04d}")
            if main_group in f.keys():
                grp = f[main_group]
            else:
                main_grp = f.create_group(main_group)
            grp = main_grp.create_group(f"det={i:04d}")

            # Save string attributes
            grp.attrs["uid"] = detection.uid
            grp.attrs["station"] = detection.station
            grp.attrs["unit"] = detection.unit

            # Save numpy arrays
            grp.create_dataset("waveform", data=detection.waveform)
            grp.create_dataset(
                "time_vector", data=detection.time_vector.astype("int64")
            )

            # Save scalar attributes
            grp.attrs["channel"] = detection.channel
            grp.attrs["time_init"] = np.datetime_as_string(
                detection.time_init, timezone="UTC", unit=TIME_PRECISION
            ).encode("utf-8")
            grp.attrs["time_end"] = np.datetime_as_string(
                detection.time_end, timezone="UTC", unit=TIME_PRECISION
            ).encode("utf-8")
            grp.attrs["sampling_rate"] = detection.sampling_rate
            if detection.metadata:
                grp.create_group("metadata")
                for key, value in detection.metadata.items():
                    grp["metadata"].attrs[key] = value


def process_stream(
    ds: DataStream,
    pulse_start: Union[int, list[int]] = 500,
    pulse_end: Union[int, list[int]] = 500,
    stwin: Union[float, list[float]] = 0.025,
    ltwin: Union[float, list[float]] = 5.0,
    threshold: Union[float, list[float]] = 2.25,
    distance: Union[float, list[float]] = 0.75,
    taper: Union[float, list[float]] = 5.0,
    station: str = "unknown",
    dbpath: Path = Path.cwd() / "detections.h5",
    metadata: Optional[dict] = None,
) -> None:
    if not isinstance(pulse_start, list):
        pulse_start = [pulse_start] * ds.num_channels
    if not isinstance(pulse_end, list):
        pulse_end = [pulse_end] * ds.num_channels
    if not isinstance(stwin, list):
        stwin = [stwin] * ds.num_channels
    if not isinstance(ltwin, list):
        ltwin = [ltwin] * ds.num_channels
    if not isinstance(threshold, list):
        threshold = [threshold] * ds.num_channels
    if not isinstance(distance, list):
        distance = [distance] * ds.num_channels
    if not isinstance(taper, list):
        taper = [taper] * ds.num_channels

    for chan in range(ds.num_channels):
        dsc = ds[:, chan]
        pulse_times = detect_pulses(
            dsc,
            stwin=stwin[chan],
            ltwin=ltwin[chan],
            threshold=threshold[chan],
            distance=distance[chan],
            taper=taper[chan],
        )

        detections = []
        for pulse in tqdm(pulse_times):
            t_start = pulse - np.timedelta64(pulse_start[chan], "ms")
            t_end = pulse + np.timedelta64(pulse_end[chan], "ms")
            excerpt = dsc.slice(t_start, t_end)
            detections.append(
                Detection(
                    uid=uuid.uuid4().hex[:8],
                    station=station,
                    waveform=excerpt.waveform,
                    time_vector=excerpt.time_vector,
                    channel=excerpt.stats.channels,
                    time_init=excerpt.stats.time_init,
                    time_end=excerpt.stats.time_end,
                    sampling_rate=excerpt.stats.sampling_rate,
                    unit=excerpt.stats.units,
                    metadata=metadata,
                )
            )
        save_detections_to_hdf5(chan, detections, dbpath)


# def main(args: argparse.Namespace) -> None:
#     time_start = np.datetime64(args.time_start, TIME_PRECISION)
#     time_end = np.datetime64(args.time_end, TIME_PRECISION)
#     catalogue = CATPATH / args.catalogue
#     station = get_station_from_catalogue_name(catalogue)
#     dbpath = DBPATH / args.database
#     dbpath.parent.mkdir(exist_ok=True, parents=True)

#     cq = CatalogueQuery(
#         catalogue=catalogue,
#         destination="data/interrogation",
#         time_start=time_start,
#         time_end=time_end,
#         channels=[int(c) for c in args.channels.split(",")],
#     )
#     ds = read(cq)
#     ds = ds.filter("highpass", freq=10.0)
#     process_stream(
#         ds,
#         pulse_start=args.pulse_start,
#         pulse_end=args.pulse_end,
#         stwin=args.stwin,
#         ltwin=args.ltwin,
#         threshold=args.threshold,
#         distance=args.distance,
#         station=station,
#         dbpath=dbpath,
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--catalogue",
#         help="Catalogue file",
#         type=str,
#         default="catalogues/SHRU921_Tetra_FileInfo.csv",
#     )
#     parser.add_argument(
#         "--time_start", help="Start time", type=str, default="2023-07-09T10:00:00"
#     )
#     parser.add_argument(
#         "--time_end", help="End time", type=str, default="2023-07-09T11:30:00"
#     )
#     parser.add_argument(
#         "--channels", help="Channels  to be read", type=str, default="0,1,2,3"
#     )
#     parser.add_argument(
#         "--threshold", help="Threshold for peak detection", type=float, default=2.25
#     )
#     parser.add_argument(
#         "--stwin", help="Short time window for RMS", type=float, default=0.025
#     )
#     parser.add_argument(
#         "--ltwin", help="Long time window for mean RMS", type=float, default=5.0
#     )
#     parser.add_argument(
#         "--distance", help="Minimum distance between peaks", type=float, default=0.75
#     )
#     parser.add_argument(
#         "--database",
#         help="Database file",
#         type=str,
#         default="detections/SHRU921_Tetra.h5",
#     )
#     parser.add_argument(
#         "--pulse_start",
#         help="Time (ms) before pulse peak to retrieve",
#         type=int,
#         default=500,
#     )
#     parser.add_argument(
#         "--pulse_end",
#         help="Time (ms) after pulse peak to retrieve",
#         type=int,
#         default=500,
#     )
#     parser.add_argument("--metadata", help="", type=str)
#     args = parser.parse_args()
#     main(args)
