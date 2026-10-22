import numpy as np
import pytest

from soundscape.bands import (
    band_sum,
    band_weights,
    decidecade_bands,
    exposure_spectrum,
    nominal,
)

FS = 2000.0


def test_nominal_labels():
    b = decidecade_bands(10.0, 1000.0)
    assert list(b["nominal"][:6]) == [12.5, 16, 20, 25, 31.5, 40]
    assert b["nominal"][-1] == 800
    assert nominal(1000.0) == 1000 and nominal(10**3.1) == 1250
    # Edges are a decidecade apart and bands are contiguous.
    assert np.allclose(b["upper"] / b["lower"], 10**0.1)
    assert np.allclose(b["upper"][:-1], b["lower"][1:])


def test_exposure_spectrum_conserves_energy():
    x = np.random.default_rng(0).standard_normal((3, 2401))
    _, e = exposure_spectrum(x, FS)
    assert e.sum(axis=-1) == pytest.approx((x**2).sum(axis=-1) / FS)


def test_tone_lands_in_its_band():
    t = np.arange(int(FS)) / FS
    x = np.sin(2 * np.pi * 100.0 * t)
    f, e = exposure_spectrum(x, FS)
    b = decidecade_bands(10.0, FS / 2)
    bands = band_sum(e, band_weights(f, b["lower"], b["upper"]))
    k = int(np.argmax(bands))
    assert b["nominal"][k] == 100
    assert bands[k] == pytest.approx(0.5, rel=1e-3)  # mean square 1/2 over 1 s


def test_weights_split_a_bin_between_bands():
    f = np.arange(0.0, 50.0, 1.0)
    w = band_weights(f, np.array([10.0, 10.5]), np.array([10.5, 11.0]))
    # The bin at 10 Hz spans 9.5-10.5 Hz: half of it lies in the first band.
    assert w[0, 10] == pytest.approx(0.5) and w[1, 11] == pytest.approx(0.5)
