import numpy as np
import pytest

from soundscape.kurtosis import beta2
from soundscape.metrics import channel_valid, click_count, cumulative_sel, pulse_metrics

FS = 10_000.0


def test_sine_levels():
    a = 1000.0
    t = np.arange(int(FS)) / FS
    x = a * np.sin(2 * np.pi * 100 * t)
    m = pulse_metrics(x, FS)
    assert m["Lp_pk"] == pytest.approx(20 * np.log10(a), abs=1e-3)
    assert m["Lp_pkpk"] == pytest.approx(20 * np.log10(2 * a), abs=1e-3)
    # Mean square over tau90 of a long sine is a^2 / 2.
    assert m["Lp_90"] == pytest.approx(20 * np.log10(a / np.sqrt(2)), abs=0.05)
    # One second of signal: LEp_ss equals the mean-square level of the window.
    assert m["LEp_ss"] == pytest.approx(20 * np.log10(a / np.sqrt(2)), abs=1e-3)
    assert m["tau90"] == pytest.approx(0.9, abs=0.01)


def test_sel_scales_with_duration():
    x = np.ones((2, int(2 * FS)))
    x[1, int(FS) :] = 0.0
    m = pulse_metrics(x, FS)
    assert m["LEp_ss"][0] - m["LEp_ss"][1] == pytest.approx(10 * np.log10(2), abs=1e-6)


def test_sel_ss_holds_all_pulse_energy():
    # LEp_ss integrates the whole window, so it exceeds the energy within tau90
    # by 10 log10(1 / 0.9).
    x = np.zeros(int(FS))
    x[1000:3000] = np.random.default_rng(1).standard_normal(2000)
    m = pulse_metrics(x, FS)
    e90 = 10 ** (m["Lp_90"] / 10) * m["tau90"]
    assert m["LEp_ss"] - 10 * np.log10(e90) == pytest.approx(
        10 * np.log10(1 / 0.9), abs=0.05
    )


def test_rise_and_decay():
    x = np.zeros(1000)
    x[100:200] = 1.0
    x[150] = 10.0
    m = pulse_metrics(x, FS)
    assert m["t_r"] > 0 and m["t_d"] > 0
    assert m["t_r"] + m["t_d"] == pytest.approx(m["tau90"])


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


def _burst(n=12_000, at=6000):
    """Noise with a 100 Hz tone burst starting at `at`."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    t = np.arange(2000) / FS
    x[at : at + 2000] += 100 * np.sin(2 * np.pi * 100 * t) * np.exp(-t / 0.05)
    return x


def test_rise_and_decay_are_non_negative_with_an_outlying_peak():
    x = _burst()
    x[100] = 500.0  # larger than the burst, well before it
    m = pulse_metrics(x, FS)
    assert m["Lp_pk"] == pytest.approx(20 * np.log10(500.0))
    assert m["t_r"] >= 0
    assert m["t_d"] >= 0


def test_spikes_are_counted_and_band_limited_sound_is_not():
    x = _burst()
    assert click_count(np.sin(np.arange(x.size) / 5), 30) == 0
    clean = np.convolve(x, np.ones(8) / 8, mode="same")  # band-limited
    spiky = clean.copy()
    spiky[[1000, 9000]] += 200.0
    assert click_count(np.stack([clean, spiky]), 30).tolist() == [0, 2]
    assert click_count(np.zeros((1, 100)), 30)[0] == 0


def test_spiky_channels_are_invalid_and_left_out_of_cumulative_sel():
    valid = channel_valid(
        np.array([True, True, False]), np.array([[0, 0], [0, 3], [0, 0]])
    )
    assert valid.tolist() == [[True, True], [True, False], [False, False]]
    out = cumulative_sel(np.full((3, 2), 150.0), valid)
    assert out[-1, 0] == pytest.approx(150 + 10 * np.log10(2))
    assert out[-1, 1] == pytest.approx(150)


def test_integration_stops_half_way_to_neighbouring_strikes(tmp_path):
    import h5py

    from soundscape.extract import integration_bounds, trim

    path = tmp_path / "pulses.h5"
    fs, n = 1000.0, 1200
    with h5py.File(path, "w") as f:
        f.create_dataset("waveforms", data=np.ones((3, 1, n), dtype=np.float32))
        f.create_dataset("time", data=np.array([0, 1_000_000, 3_000_000]))
        f.attrs.update({"sampling_rate": fs, "pre_s": 0.6})
    lo, hi = integration_bounds(path)
    # Strikes 1 s apart share the 0.2 s their 1.2 s windows overlap, split
    # at the midpoint; a strike 2 s from both neighbours keeps its window.
    np.testing.assert_array_equal(lo, [0, 100, 0])
    np.testing.assert_array_equal(hi, [1100, 1200, 1200])
    w = trim(np.ones((3, 1, n)), lo, hi)
    assert w.sum(axis=-1)[:, 0].tolist() == [1100, 1100, 1200]
