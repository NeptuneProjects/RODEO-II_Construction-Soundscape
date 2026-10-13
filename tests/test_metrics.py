import numpy as np
import pytest

from soundscape.metrics import cumulative_sel, pulse_metrics

FS = 10_000.0


def test_sine_levels():
    a = 1000.0
    t = np.arange(int(FS)) / FS
    x = a * np.sin(2 * np.pi * 100 * t)
    m = pulse_metrics(x, FS)
    assert m["SPL_pk"] == pytest.approx(20 * np.log10(a), abs=1e-3)
    assert m["SPL_pkpk"] == pytest.approx(20 * np.log10(2 * a), abs=1e-3)
    assert m["SPL_rms"] == pytest.approx(20 * np.log10(a / np.sqrt(2)), abs=1e-3)
    # One second of signal: SEL equals SPL_rms.
    assert m["SEL"] == pytest.approx(m["SPL_rms"], abs=1e-3)
    assert m["T90"] == pytest.approx(0.9, abs=0.01)


def test_sel_scales_with_duration():
    x = np.ones((2, int(2 * FS)))
    x[1, int(FS) :] = 0.0
    m = pulse_metrics(x, FS)
    assert m["SEL"][0] - m["SEL"][1] == pytest.approx(10 * np.log10(2), abs=1e-6)


def test_rise_and_decay():
    x = np.zeros(1000)
    x[100:200] = 1.0
    x[150] = 10.0
    m = pulse_metrics(x, FS)
    assert m["t_rise"] > 0 and m["t_decay"] > 0
    assert m["t_rise"] + m["t_decay"] == pytest.approx(m["T90"])


def test_gaussian_kurtosis_is_three():
    x = np.random.default_rng(0).standard_normal(200_000)
    assert pulse_metrics(x, FS)["kurtosis"] == pytest.approx(3.0, abs=0.05)


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
