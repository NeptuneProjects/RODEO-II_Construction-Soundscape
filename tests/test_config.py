from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from soundscape.calibration import calibrate
from soundscape.config import (
    DetectorOverride,
    FlatCalibration,
    FrequencyResponseCalibration,
    expand_env,
    load_site,
)
from soundscape.events import load_events, source_range
from soundscape.sources import neutralize_conditioning, remap_filenames

SITES = Path(__file__).parents[1] / "config" / "sites"


@pytest.mark.parametrize("name", ["southfork", "vineyard"])
def test_site_files_load(name):
    site = load_site(SITES / f"{name}.toml")
    events = load_events(site)
    assert events
    for e in events:
        assert e.start < e.end
        for s in site.sensors:
            assert 0 < source_range(e, s) < 50_000


def test_southfork_exclusions():
    site = load_site(SITES / "southfork.toml")
    names = {e.name for e in load_events(site)}
    assert "A04" not in names and "A08" in names
    assert site.is_excluded("A11", "Tetra")
    assert not site.is_excluded("A11", "VLA1")
    # Naming an excluded event explicitly includes it.
    assert [e.name for e in load_events(site, ["A04"])] == ["A04"]


def test_sensor_detector_override():
    site = load_site(SITES / "vineyard.toml")
    vla2 = site.sensor("VLA2")
    vla2.detector = DetectorOverride(threshold=0.02)
    assert site.detector_for(vla2).threshold == 0.02
    assert site.detector_for(vla2).min_neighbors == site.detector.min_neighbors
    assert site.detector_for(site.sensor("VLA1")).threshold == 0.05


def test_expand_env(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/mnt/data")
    assert expand_env("${DATA_ROOT}/x") == "/mnt/data/x"
    monkeypatch.delenv("DATA_ROOT")
    assert expand_env("${DATA_ROOT:-/project}/x") == "/project/x"
    with pytest.raises(KeyError):
        expand_env("${DATA_ROOT}/x")


def test_flat_calibration():
    cal = FlatCalibration(sensitivity=[-204.0, -200.0], fixed_gain=[6.0, 0.0])
    counts = np.full((2, 4), 2**23 - 1, dtype=float)
    p = calibrate(cal, counts, 1000.0, channels=[0, 1])
    assert p[0, 0] == pytest.approx(2.5 / 10 ** (-198 / 20))
    assert p[1, 0] == pytest.approx(2.5 / 10 ** (-200 / 20))
    # Channel selection picks the matching sensitivity.
    p1 = calibrate(cal, counts[:1], 1000.0, channels=[1])
    assert p1[0, 0] == pytest.approx(p[1, 0])


def test_flat_calibration_rejects_positive_sensitivity():
    with pytest.raises(ValidationError):
        FlatCalibration(sensitivity=[204.0], fixed_gain=[6.0])


def test_frequency_response_calibration(tmp_path):
    f = tmp_path / "cal.csv"
    f.write_text("frequency, sensitivity\n10,-180\n10000,-180\n")
    cal = FrequencyResponseCalibration(file=f)
    x = np.random.default_rng(0).standard_normal((1, 1000))
    p = calibrate(cal, x, 2000.0, channels=[0])
    np.testing.assert_allclose(p, x * 10**9, rtol=1e-9)


def test_inventory_rewrite():
    df = pl.DataFrame(
        {
            "filename": ["/project/SF/a.D23", "/other/b.D23"],
            "nch": ["4", "4"],
            "gain": ["6.0,6.0,6.0,6.0"] * 2,
            "sensitivity": ["204.0,204.1,204.0,204.2"] * 2,
        }
    )
    out = neutralize_conditioning(remap_filenames(df, {"/project/SF": "/Volumes/SF"}))
    assert out["filename"].to_list() == ["/Volumes/SF/a.D23", "/other/b.D23"]
    assert out["gain"][0] == "0.0,0.0,0.0,0.0"
    assert out["sensitivity"][0] == "0.0,0.0,0.0,0.0"
    assert out["adc_vref"][0] == "1.0,1.0,1.0,1.0"
