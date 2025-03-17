# -*- coding: utf-8 -*-
"""Utility functions for the RODEO package.

William F. Jenkins, Ph.D.
Scripps Institution of Oceanography
University of California San Diego
February 2025
"""
import importlib.resources as pkg_resources
import logging
from typing import Sequence

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyproj
import scipy.fft as fft

import rodeo

logging_kwargs = {
    "level": logging.INFO,
    "format": "%(asctime)s [%(levelname)s] %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}


def double_to_single_sided_ifft(
    time_values: npt.NDArray[np.float_],
    ifft_values: npt.NDArray[np.complex_] | npt.NDArray[np.complexfloating],
    shift: bool = False,
) -> tuple[
    npt.NDArray[np.complex_] | npt.NDArray[np.complexfloating], npt.NDArray[np.float_]
]:
    """
    Extract the first half of the IFFT result for display or analysis purposes.
    Works with both single-channel and multi-channel data.

    Parameters:
    -----------
    time_values : array of float
        The time values corresponding to each IFFT sample
    ifft_values : array of complex or float
        The result of an inverse FFT (typically time-domain)
        Can be 1D array [n_time_points] for single-channel data
        or 2D array [n_channels, n_time_points] for multichannel data

    Returns:
    --------
    corresponding_times : array of float
        The time values corresponding to the first half
    first_half : array
        The first half of the IFFT result, with same dimensionality as input
    """
    # Get the total number of points (last dimension)
    n = ifft_values.shape[-1]

    if shift:
        ifft_values = fft.ifftshift(ifft_values, axes=-1)

    # Calculate the midpoint index
    # For even length, midpoint is n/2
    # For odd length, midpoint is (n+1)/2
    midpoint = (n + 1) // 2

    # Extract the first half of the data based on dimensionality
    if ifft_values.ndim == 1:
        first_half = ifft_values[:midpoint].copy()
    else:
        # For multi-channel data, keep all channels but take first half of time points
        first_half = ifft_values[..., :midpoint].copy()

    # Get corresponding time values
    corresponding_times = time_values[:midpoint]

    return corresponding_times, first_half


def double_to_single_sided_fft(
    freqs: npt.NDArray[np.float_],
    fft_values: npt.NDArray[np.complex_],
) -> tuple[npt.NDArray[np.complex_], npt.NDArray[np.float_]]:
    """
    Convert a double-sided FFT to a properly scaled single-sided FFT.
    Efficiently handles both single-channel and multichannel FFT data.

    Parameters:
    -----------
    freqs : array of float
        The corresponding frequencies for the FFT result (positive and negative)
    fft_values : array of complex
        The complex FFT result (double-sided)
        Can be 1D array [n_fft_points] for single-channel data
        or 2D array [n_channels, n_fft_points] for multichannel data

    Returns:
    --------
    positive_freqs : array of float
        The corresponding positive frequencies
    single_sided_fft : array of complex
        The single-sided FFT with proper scaling
        Maintains the same dimensionality as input fft_values
    """
    # Store original shape and dimensionality
    original_ndim = fft_values.ndim

    # Get number of FFT points (last dimension)
    n = fft_values.shape[-1]

    # Find indices where frequencies are >= 0
    positive_indices = freqs >= 0
    positive_freqs = freqs[positive_indices]

    # Create scaling factors array
    scale = np.ones(len(positive_freqs), dtype=float)

    # Set scaling for all frequencies except DC (and Nyquist if present)
    if n % 2 == 0:  # Even number of points
        # Scale all but first (DC) and last (Nyquist) elements
        scale[1:-1] = 2.0
    else:  # Odd number of points
        # Scale all but first (DC) element
        scale[1:] = 2.0

    # Handle indexing into the FFT values differently based on dimensionality
    # and apply scaling
    if original_ndim == 1:
        single_sided_fft = fft_values[positive_indices].copy() * scale
    else:
        # For multichannel data, we need to index the last dimension
        single_sided_fft = (
            fft_values[..., positive_indices].copy() * scale[np.newaxis, :]
        )

    return positive_freqs, single_sided_fft


def get_welch() -> tuple[callable, bool]:
    """Get Welch's PSD method.

    If CuPy is installed and a GPU is available, the CUDA enabled flag is
    set to True and the CUDA version of Welch's PSD method is returned.

    Returns:
        Welch's PSD method.
        CUDA enabled flag.
    """
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


def compute_array_size(arrays: list[np.ndarray]) -> int:
    """Compute the total size of the arrays in bytes.

    Args:
        arrays: List of NumPy arrays.

    Returns:
        Total size of the arrays in bytes.
    """
    return sum([a.nbytes for a in arrays])


def decimation_factor(fs: float, target_fs: float) -> int:
    """Compute the decimation factor to downsample a signal.

    Args:
        fs: Original sampling rate.
        target_fs: Target sampling rate.

    Returns:
        Decimation factor.
    """
    return int(fs // target_fs)


def compute_range(
    point1_lon: float, point1_lat: float, point2_lon: float, point2_lat: float
) -> float:
    """Compute the distance between two points on the Earth's surface.

    Uses the WGS84 ellipsoid.

    Args:
        point1_lon: Longitude of the first point.
        point1_lat: Latitude of the first point.
        point2_lon: Longitude of the second point.
        point2_lat: Latitude of the second point.

    Returns:
        Distance between the two points in meters.
    """
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
    """Format the channel text for the configuration file.

    Args:
        channel: Channel number.
        pulse_start: Pulse start time.
        pulse_end: Pulse end time.
        stwin: Short time window.
        ltwin: Long time window.
        threshold: Threshold value.
        distance: Distance from the source.

    Returns:
        Formatted text.
    """
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
    """Format the event text for the configuration file.

    Args:
        event_name: Event name.
        event_site: Event site.
        event_start: Event start time.
        event_stop: Event stop time.

    Returns:
        Formatted text.
    """
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
    """Format the sensor text for the configuration file.

    Args:
        sensor_name: Sensor name.
        dec_factor: Decimation factor.
        filter_type: Filter type.
        filter_freq: Filter frequency.
        taper: Taper value.
        channels: List of channel numbers.
        pulse_start: List of pulse start times.
        pulse_end: List of pulse end times.
        stwin: List of short time windows.
        ltwin: List of long time windows.
        threshold: List of threshold values.
        distance: List of distances.

    Returns:
        Formatted text.
    """
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
    """Print the HDF5 tree structure.

    Args:
        grp: HDF5 group.
        prefix: Prefix for the tree structure.
        max_operations: Maximum number of operations.
        current_operations: Current number of operations.

    Returns:
        Number of operations.
    """
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


def initialize_julia(modules: str | Sequence[str]) -> object:
    """Initialize Julia and load the required modules.

    Args:
        modules: List of Julia modules to load.

    Returns:
        Julia object for name space.
    """
    from juliacall import Main as jl

    if isinstance(modules, str):
        modules = [modules]

    for module in modules:
        julia_path = pkg_resources.files(rodeo).joinpath(f"{module}.jl")
        jl.include(str(julia_path))

    return jl


def spectrogram_args(
    window_func: callable, tperseg: float, fs: float, pcoverlap: float = 0.0
) -> tuple[int, np.ndarray, int]:
    """Compute the spectrogram arguments.

    Args:
        window_func: Window function.
        tperseg: Time per segment (s).
        fs: Sampling rate (Hz).
        pcoverlap: Percent overlap.

    Returns:
        Segment length (samples).
        Window function.
        Hop size (samples).
    """
    nperseg = int(tperseg * fs)
    win = window_func(nperseg)
    hop = max(int(nperseg * (1 - pcoverlap)), 1)
    return nperseg, win, hop
