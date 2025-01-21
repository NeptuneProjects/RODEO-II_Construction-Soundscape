# -*- coding: utf-8 -*-

from dataclasses import dataclass
import logging
from pathlib import Path
import uuid

import h5py
import numpy as np
from scipy import signal
from tqdm import tqdm
from tritonoa.data.stream import DataStream
from tritonoa.data.time import TIME_PRECISION


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
    metadata: dict | None = None


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
    y_rms = np.sqrt(signal.convolve(y**2, np.ones(stwin) / stwin, mode="same"))
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
        apply_linear_taper(ds.data.squeeze(), int(fs * taper)),
        int(stwin * fs),
        int(ltwin * fs),
    )
    peaks, _ = signal.find_peaks(chf, height=threshold, distance=distance * fs)
    return time[peaks]


def station_from_catalogue_name(fname: Path) -> str:
    return fname.stem.split("FileInfo")[0].rstrip("_")


def save_detections_to_hdf5(
    channel: int | str, detections: list[Detection], file_path: str
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
    pulse_start: int | list[int] = 500,
    pulse_end: int | list[int] = 500,
    stwin: float | list[float] = 0.025,
    ltwin: float | list[float] = 5.0,
    threshold: float | list[float] = 2.25,
    distance: float | list[float] = 0.75,
    taper: float | list[float] = 5.0,
    pulse_filter_type: str | None = None,
    pulse_filter_freq: float | None = None,
    detector_filter_type: str | None = None,
    detector_filter_freq: float | None = None,
    detector_decimate_factor: int | None = None,
    station: str = "unknown",
    dbpath: Path = Path.cwd() / "detections.h5",
    metadata: dict | None = None,
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
        logging.info(f"--Processing channel {chan}.")
        dsc = ds[chan]
        if detector_decimate_factor is not None:
            dsc_det = dsc.copy()
            dsc_det.decimate(detector_decimate_factor)
            logging.info(f"--Detector data decimated by {detector_decimate_factor}.")
        else:
            dsc_det = dsc
        if detector_filter_type is not None:
            dsc_det.filter(filt_type=detector_filter_type, freq=detector_filter_freq)
            logging.info(
                f"--Detector data filtered with {detector_filter_type} "
                f"at {detector_filter_freq} Hz."
            )

        pulse_times = detect_pulses(
            dsc_det,
            stwin=stwin[chan],
            ltwin=ltwin[chan],
            threshold=threshold[chan],
            distance=distance[chan],
            taper=taper[chan],
        )

        if pulse_filter_type is not None:
            dsc.filter(filt_type=pulse_filter_type, freq=pulse_filter_freq)
            logging.info(
                f"--Pulse filtered with {pulse_filter_type} "
                f"at {pulse_filter_freq} Hz."
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
                    waveform=excerpt.data,
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
        logging.info(f"--Detections saved to {dbpath}.")
        logging.info(f"--Completed channel {chan}.")
