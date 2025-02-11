# -*- coding: utf-8 -*-

import h5py
import numpy as np
import pandas as pd
import pyproj


def get_welch() -> tuple[callable, bool]:
    try:
        import cupy

        if cupy.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("No GPU found.")

        from cupyx.scipy.signal import welch

        cuda_enabled = True

    except (ImportError, RuntimeError):
        from scipy.signal import welch

        cuda_enabled = False

    return welch, cuda_enabled


def clean_ctd_data(df: pd.DataFrame, min_depth: float = 1.0) -> pd.DataFrame:
    """Remove data points with depth less than 1.0 m.

    Args:
        df: CTD data.

    Returns:
        Cleaned CTD data.
    """
    # Round `Depth` column to two decimal points (for AT environment file)
    df["Depth"] = df["Depth"].round(2)
    # De-depulicate data:
    df.drop_duplicates(subset="Depth", keep="last", inplace=True)
    return df[df["Depth"] > min_depth]


def compute_range(
    point1_lon: float, point1_lat: float, point2_lon: float, point2_lat: float
) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist = geod.inv(point1_lon, point1_lat, point2_lon, point2_lat)
    return dist


def format_channel_text(
    channel: int,
    pulse_start: float,
    pulse_end: float,
    stwin: float,
    ltwin: float,
    threshold: float,
    distance: float,
) -> str:
    return (
        "\n\n[[event.sensors.channels]]"
        f"\nchannel = {channel}"
        f"\npulse_start = {pulse_start}"
        f"\npulse_end = {pulse_end}"
        f"\nstwin = {stwin}"
        f"\nltwin = {ltwin}"
        f"\nthreshold = {threshold}"
        f"\ndistance = {distance}"
    )


def format_event_text(event_name, event_site, event_start, event_stop) -> str:
    return (
        "\n[[event]]"
        f'\nname = "{event_name}"'
        f'\nsite = "{event_site}"'
        f'\nstart = "{event_start}"'
        f'\nend = "{event_stop}"'
    )


def format_sensor_text(
    sensor_name: str,
    dec_factor: int,
    filter_type: str,
    filter_freq: float,
    taper: float,
    channels: list,
    pulse_start: list,
    pulse_end: list,
    stwin: list,
    ltwin: list,
    threshold: list,
    distance: list,
) -> str:
    sensor_text = (
        "\n[[event.sensors]]"
        f'\nname = "{sensor_name}"'
        f"\ndec_factor = {dec_factor}"
        f'\nfilter = "{filter_type}"'
        f"\nfreq = {filter_freq}"
        f"\ntaper = {taper}"
    )
    channel_text = "".join(
        [
            format_channel_text(
                channel=c,
                pulse_start=ps,
                pulse_end=pe,
                stwin=st,
                ltwin=lt,
                threshold=th,
                distance=d,
            )
            for c, ps, pe, st, lt, th, d in zip(
                channels, pulse_start, pulse_end, stwin, ltwin, threshold, distance
            )
        ]
    )
    return sensor_text + channel_text


def h5_tree(grp: h5py.Group, prefix="", max_operations=None, current_operations=0):
    if max_operations is not None and current_operations >= max_operations:
        return current_operations

    items = len(grp)
    for key, grp in grp.items():
        current_operations += 1
        if max_operations is not None and current_operations >= max_operations:
            return current_operations

        items -= 1
        if items == 0:
            # the last item
            if isinstance(grp, h5py.Group):
                print(f"{prefix}└── {key} (Group)")
                for attr_key, attr_value in grp.attrs.items():
                    print(f"{prefix}    └── @{attr_key}: {attr_value}")
                current_operations = h5_tree(
                    grp, f"{prefix}    ", max_operations, current_operations
                )
            else:
                try:
                    print(f"{prefix}└── {key} ({len(grp)})")
                except TypeError:
                    print(f"{prefix}└── {key} (scalar)")
                for attr_key, attr_value in grp.attrs.items():
                    print(f"{prefix}    └── @{attr_key}: {attr_value}")
        else:
            if isinstance(grp, h5py.Group):
                print(f"{prefix}├── {key} (Group)")
                for attr_key, attr_value in grp.attrs.items():
                    print(f"{prefix}|   └── @{attr_key}: {attr_value}")
                current_operations = h5_tree(
                    grp, f"{prefix}|   ", max_operations, current_operations
                )
            else:
                try:
                    print(f"{prefix}├── {key} ({len(grp)})")
                except TypeError:
                    print(f"{prefix}├── {key} (scalar)")
                for attr_key, attr_value in grp.attrs.items():
                    print(f"{prefix}|   └── @{attr_key}: {attr_value}")

        if max_operations is not None and current_operations >= max_operations:
            return current_operations

    return current_operations


def spectrogram_args(
    window_func: callable, tperseg: float, fs: float, pcoverlap: float = 0.0
) -> tuple[int, np.ndarray, int]:
    nperseg = int(tperseg * fs)
    win = window_func(nperseg)
    hop = max(int(nperseg * (1 - pcoverlap)), 1)
    return nperseg, win, hop
