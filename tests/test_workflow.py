"""End to end on a synthetic recording: detect -> extract -> metrics -> psd."""

import h5py
import numpy as np
import polars as pl
import pytest

from soundscape import detect, extract, spectra, workflow
from soundscape.config import Exclusion
from soundscape.events import load_events
from tests.conftest import SyntheticRecording


@pytest.fixture
def wf(site, monkeypatch):
    rec = SyntheticRecording(duration_s=900.0, gap=(300.5, 330.0))
    for module in (detect, extract, spectra):
        monkeypatch.setattr(module, "read_segment", rec.read)
    w = workflow.Workflow(site)
    w.recording = rec
    return w


def test_end_to_end(wf, site):
    wf.detect()
    event = "E1"
    detections = pl.read_csv(site.detections_path(event, "VLA1"), try_parse_dates=True)
    assert len(detections) == len(wf.recording.peak_times)
    assert (site.event_dir(event) / "detections_VLA1.png").exists()

    # Re-selecting from saved candidates gives the same answer without data.
    wf.detect(reselect=True, plot=False)
    again = pl.read_csv(site.detections_path(event, "VLA1"), try_parse_dates=True)
    assert again.equals(detections)

    wf.extract()
    with h5py.File(site.pulses_path(event, "VLA1")) as f:
        wf_shape = f["waveforms"].shape
        assert f.attrs["range_m"] > 0
        assert f["valid"][:].all()
    assert wf_shape == (len(detections), 2, int(1.2 * 2000))

    wf.metrics()
    m = pl.read_csv(site.metrics_path(event, "VLA1"))
    assert len(m) == 2 * len(detections)
    # Both channels carry the same pulses, so levels agree closely.
    by_channel = m.pivot(on="channel", index="strike", values="SPL_pk")
    np.testing.assert_allclose(by_channel["0"], by_channel["1"], atol=1.0)
    # Soft start: levels rise over the first strikes.
    first = m.filter(pl.col("channel") == 0).sort("strike")["SPL_pk"]
    assert first[60] - first[0] > 15
    # Cumulative SEL is non-decreasing.
    assert (m.filter(pl.col("channel") == 0)["SEL_cum"].diff().drop_nulls() >= 0).all()
    assert site.metrics_summary_path.exists()

    wf.psd()
    with h5py.File(site.pulse_psd_path(event, "VLA1")) as f:
        psd, freqs = f["psd"][:], f["freqs"][:]
    assert psd.shape[:2] == (len(detections), 2)
    # Pulse energy is centered at 200 Hz.
    mean_psd = psd[:, 0].mean(axis=0)
    assert abs(freqs[np.argmax(mean_psd)] - 200.0) < 10.0


def test_ambient(wf, site, monkeypatch):
    # Ambient window precedes the event; shift it into the synthetic record.
    site.ambient.offset_s = -600.0
    site.ambient.duration_s = 120.0
    wf.ambient()
    spl = pl.read_csv(site.ambient_spl_path)
    assert len(spl) == 2 * 2
    with h5py.File(site.ambient_path) as f:
        assert f["E1/VLA1/psd"].shape[0] == 2


def test_excluded_pairs_are_skipped(site):
    site = site.model_copy(
        update={"exclude": [Exclusion(event="E1", sensors=["VLA1"])]}
    )
    called = []
    w = workflow.Workflow(site, events=["E1"])
    w._each(lambda e, s: called.append((e.name, s.name)), "test")
    assert called == []
    assert load_events(site)
