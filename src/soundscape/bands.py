"""Decidecade (base-10 one-third octave) frequency bands.

Bands follow IEC 61260-1 with base 10, called decidecades in ISO 18405: the
exact center frequencies are 10^(n/10) Hz for integer n, and each band spans
a factor 10^(1/10) between its edges. Bands are labeled by their nominal
center frequencies (10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80 Hz, ...).

Band values are sums of spectral values over frequency bins, each bin
weighted by the fraction of its width that lies in the band, so that bands
narrower than a few bins still get their share of the power.
"""

import numpy as np

_NOMINAL = [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80]


def nominal(center: float) -> float:
    """Nominal (rounded) label of an exact decidecade center frequency."""
    n = round(10 * np.log10(center))
    return _NOMINAL[n % 10] * 10.0 ** (n // 10 - 1)


def decidecade_bands(fmin: float, fmax: float) -> dict[str, np.ndarray]:
    """Bands whose lower edge is at least `fmin` and upper edge at most `fmax`.

    Returns {"center", "lower", "upper", "nominal"} arrays in Hz.
    """
    half = 10 ** (1 / 20)
    n0 = int(np.ceil(10 * np.log10(fmin * half) - 1e-9))
    n1 = int(np.floor(10 * np.log10(fmax / half) + 1e-9))
    n = np.arange(n0, n1 + 1)
    center = 10.0 ** (n / 10)
    return {
        "center": center,
        "lower": center / half,
        "upper": center * half,
        "nominal": np.array([nominal(c) for c in center]),
    }


def band_weights(freqs: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """(n_bands, n_freqs) fraction of each frequency bin inside each band.

    Bin k is taken to span freqs[k] ± df/2, with df the (uniform) bin spacing.
    """
    df = freqs[1] - freqs[0]
    lo = freqs - df / 2
    hi = freqs + df / 2
    overlap = np.minimum(hi[None, :], upper[:, None]) - np.maximum(
        lo[None, :], lower[:, None]
    )
    return np.clip(overlap, 0.0, None) / df


def band_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted sum over the last (frequency) axis: (..., n_freqs) ->
    (..., n_bands)."""
    return values @ weights.T


def exposure_spectrum(x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Sound exposure in each frequency bin of the one-sided DFT of `x` (time
    on the last axis), so that the bins sum to sum(x^2) / fs (Parseval).

    Returns (freqs, exposure) with exposure in the units of x^2 times s.
    """
    n = x.shape[-1]
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2 / (n * fs)
    # One-sided: double every bin except DC and (for even n) Nyquist.
    spec[..., 1 : (n + 1) // 2] *= 2
    return np.fft.rfftfreq(n, 1 / fs), spec
