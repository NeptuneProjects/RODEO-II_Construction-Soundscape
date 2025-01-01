# -*- coding: utf-8 -*-

import pyproj


def compute_range(
    point1_lon: float, point1_lat: float, point2_lon: float, point2_lat: float
) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    _, _, dist = geod.inv(point1_lon, point1_lat, point2_lon, point2_lat)
    return dist
