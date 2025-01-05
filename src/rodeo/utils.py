# -*- coding: utf-8 -*-

import pandas as pd
import pyproj


def clean_ctd_data(df: pd.DataFrame, min_depth: float = 1.0) -> pd.DataFrame:
    """Remove data points with depth less than 1.0 m.

    Args:
        df: CTD data.

    Returns:
        Cleaned CTD data.
    """
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
        "\n\n[[events.sensors.channels]]"
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
        "\n[[events]]"
        f'\nname = "{event_name}"'
        f'\nsite = "{event_site}"'
        f'\nstart = "{event_start}"'
        f'\nend = "{event_stop}"'
    )


def format_sensor_text(
    sensor_name: str,
    dec_factor: str,
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
        "\n[[events.sensors]]"
        f'\nname = "{sensor_name}"'
        f'\ndec_factor = "{dec_factor}"'
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
