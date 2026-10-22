import numpy as np
import pytest

from soundscape.kurtosis import beta2
from soundscape.metrics import cumulative_sel, pulse_metrics

FS = 10_000.0


def test_sine_levels():
    a = 1000.0
    t = np.arange(int(FS)) / FS
    x = a * np.sin(2 * np.pi * 100 * t)
    m = pulse_metrics(x, FS)
    assert m["SPL_pk"] == pytest.approx(20 * np.log10(a), abs=1e-3)
    assert m["SPL_pkpk"] == pytest.approx(20 * np.log10(2 * a), abs=1e-3)
    # Mean square over T90 of a long sine is a^2 / 2.
    assert m["SPL_rms90"] == pytest.approx(20 * np.log10(a / np.sqrt(2)), abs=0.05)
    # One second of signal: SEL_ss equals the mean-square level of the window.
    assert m["SEL_ss"] == pytest.approx(20 * np.log10(a / np.sqrt(2)), abs=1e-3)
    assert m["T90"] == pytest.approx(0.9, abs=0.01)


def test_sel_scales_with_duration():
    x = np.ones((2, int(2 * FS)))
    x[1, int(FS) :] = 0.0
    m = pulse_metrics(x, FS)
    assert m["SEL_ss"][0] - m["SEL_ss"][1] == pytest.approx(10 * np.log10(2), abs=1e-6)


def test_sel_ss_holds_all_pulse_energy():
    # SEL_ss integrates the whole window, so it exceeds the energy within T90
    # by 10 log10(1 / 0.9).
    x = np.zeros(int(FS))
    x[1000:3000] = np.random.default_rng(1).standard_normal(2000)
    m = pulse_metrics(x, FS)
    e90 = 10 ** (m["SPL_rms90"] / 10) * m["T90"]
    assert m["SEL_ss"] - 10 * np.log10(e90) == pytest.approx(
        10 * np.log10(1 / 0.9), abs=0.05
    )


def test_rise_and_decay():
    x = np.zeros(1000)
    x[100:200] = 1.0
    x[150] = 10.0
    m = pulse_metrics(x, FS)
    assert m["t_rise"] > 0 and m["t_decay"] > 0
    assert m["t_rise"] + m["t_decay"] == pytest.approx(m["T90"])


def test_gaussian_kurtosis_is_three():
    x = np.random.default_rng(0).standard_normal(200_000)
    assert beta2(x) == pytest.approx(3.0, abs=0.05)


def test_kurtosis_of_flat_window_is_nan(recwarn):
    k = beta2(np.zeros((2, 1000)))
    assert np.isnan(k).all()
    assert len(recwarn) == 0


def test_kurtosis_matches_scipy():
    from scipy.stats import kurtosis

    x = np.random.default_rng(2).standard_t(5, size=(3, 5000))
    assert beta2(x) == pytest.approx(kurtosis(x, axis=-1, fisher=False))


def test_impulsive_kurtosis_is_high():
    x = np.random.default_rng(0).standard_normal(60_000)
    x[::6000] += 200.0
    assert beta2(x) > 30


def test_batched_shapes():
    x = np.random.default_rng(0).standard_normal((5, 3, 1000))
    m = pulse_metrics(x, FS)
    assert all(v.shape == (5, 3) for v in m.values())


def test_cumulative_sel():
    sel = np.full((4, 2), 150.0)
    valid = np.array([True, True, False, True])
    out = cumulative_sel(sel, valid)
    assert out[-1, 0] == pytest.approx(150 + 10 * np.log10(3))
    assert out[2, 1] == out[1, 1]
