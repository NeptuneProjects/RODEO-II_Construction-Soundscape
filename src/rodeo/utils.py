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
