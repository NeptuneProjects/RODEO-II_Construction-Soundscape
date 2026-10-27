import numpy as np
import polars as pl
import pytest

from soundscape import detect
from soundscape.events import Event
from tests.conftest import T0, SyntheticRecording


def _event(duration_s: float) -> Event:
    return Event(
        name="E1",
        pile="P1",
        start=T0,
        end=T0 + np.timedelta64(int(duration_s * 1e6), "us"),
        latitude=41.09,
        longitude=-71.16,
    )


def _match(detected: np.ndarray, truth: np.ndarray, tol_s: float = 0.1):
    """Return (hits, false alarms) matching detections to true pulse times."""
    d = detected.astype(np.int64)
    t = truth.astype(np.int64)
    dist = np.abs(d[:, None] - t[None, :]).min(axis=1) * 1e-6
    hits = np.abs(t[:, None] - d[None, :]).min(axis=1) * 1e-6 <= tol_s
    return hits.sum(), (dist > tol_s).sum()


def _run(rec: SyntheticRecording, sensor, config, monkeypatch, duration_s=900.0):
    monkeypatch.setattr(detect, "read_segment", rec.read)
    candidates = detect.scan(sensor, None, _event(duration_s), config)
    return detect.select(candidates, config)


def test_detects_all_strikes_without_false_alarms(
    recording, sensor, detector, monkeypatch
):
    scored = _run(recording, sensor, detector, monkeypatch)
    times = scored.filter("is_strike")["time"].to_numpy()
    hits, false_alarms = _match(times, recording.peak_times)
    assert hits == len(recording.peak_times)
    assert false_alarms == 0


def test_strike_times_are_accurate(recording, sensor, detector, monkeypatch):
    scored = _run(recording, sensor, detector, monkeypatch)
    times = scored.filter("is_strike")["time"].to_numpy().astype(np.int64)
    truth = recording.peak_times.astype(np.int64)
    err_s = np.abs(times[:, None] - truth[None, :]).min(axis=1) * 1e-6
    # Pulse energy peaks within the first ~50 ms of the burst.
    assert np.median(err_s) < 0.05


def test_chunking_does_not_change_detections(recording, sensor, monkeypatch):
    configs = [detect.DetectorConfig(chunk_s=c) for c in (37.0, 120.0, 1000.0)]
    results = [
        _run(recording, sensor, c, monkeypatch).filter("is_strike")["time"].to_list()
        for c in configs
    ]
    assert results[0] == results[1] == results[2]


def test_global_normalization_misses_soft_start(recording, sensor, monkeypatch):
    """Vineyard-style normalization by the event maximum loses quiet strikes;
    the rolling window keeps them."""
    rolling = detect.DetectorConfig(chunk_s=120.0)
    global_ = rolling.model_copy(update={"normalization_window_s": 0.0})
    n_rolling = _run(recording, sensor, rolling, monkeypatch)["is_strike"].sum()
    n_global = _run(recording, sensor, global_, monkeypatch)["is_strike"].sum()
    assert n_global < n_rolling == len(recording.peak_times)


def test_no_detections_in_gap(sensor, detector, monkeypatch):
    rec = SyntheticRecording(gap=(300.0, 360.0))
    scored = _run(rec, sensor, detector, monkeypatch)
    times = scored.filter("is_strike")["time"].to_numpy()
    gap = (T0 + np.timedelta64(300, "s"), T0 + np.timedelta64(360, "s"))
    assert not np.any((times > gap[0]) & (times < gap[1]))
    hits, false_alarms = _match(times, rec.peak_times)
    assert hits == len(rec.peak_times) and false_alarms == 0


def test_noise_only_yields_no_strikes(sensor, detector, monkeypatch):
    rec = SyntheticRecording(duration_s=300.0)
    rec.data = np.random.default_rng(1).standard_normal(rec.data.shape)
    scored = _run(rec, sensor, detector, monkeypatch, duration_s=300.0)
    assert scored["is_strike"].sum() == 0


def test_suppress_close_keeps_strongest():
    times = np.array([0, 500_000, 2_000_000, 2_400_000])
    energy = np.array([1.0, 5.0, 3.0, 2.0])
    kept = detect.suppress_close(times, energy, 1_000_000)
    assert kept.tolist() == [1, 2]


@pytest.mark.parametrize("window", [0.0, 10.0])
def test_reference_energy(window):
    times = np.arange(0, 60_000_000, 1_000_000)
    energy = np.linspace(1, 2, len(times))
    ref = detect.reference_level(times, energy, window)
    if window == 0:
        assert np.all(ref == 2)
    else:
        assert np.all(ref >= energy)
        assert ref[0] == energy[5]


def test_count_neighbors():
    times = np.array([0.0, 1.5, 3.0, 10.0, 16.0]) * 1e6
    assert detect.count_neighbors(times.astype(np.int64), 2.0).tolist() == [
        1,
        2,
        1,
        0,
        0,
    ]


def test_neighbor_check_rejects_isolated_transients():
    """A soft-start group of four strikes 1.7 s apart and a full-energy train
    survive; knocks every 6 s (like mooring noise) do not."""
    soft = 10.0 + 1.7 * np.arange(4)
    train = 60.0 + 1.5 * np.arange(40)
    knocks = 200.0 + 6.0 * np.arange(10)
    t = np.concatenate([soft, train, knocks])
    candidates = pl.DataFrame(
        {
            "time": T0 + (t * 1e6).astype("timedelta64[us]"),
            "e": np.full(len(t), 100.0),
            "e_bg": 1.0,
        },
        schema=detect.CANDIDATE_SCHEMA,
    )
    config = detect.DetectorConfig(min_neighbors=2, neighbor_window_s=4.0)
    scored = detect.select(candidates, config)
    strike = scored["is_strike"].to_numpy()
    assert strike[: len(soft) + len(train)].all()
    assert not strike[len(soft) + len(train) :].any()

    off = detect.select(candidates, config.model_copy(update={"min_neighbors": 0}))
    assert off["is_strike"].all()
