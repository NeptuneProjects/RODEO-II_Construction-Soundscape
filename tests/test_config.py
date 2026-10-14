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
    GlobSourceConfig,
    SensorConfig,
    load_site,
)
from soundscape.events import load_events, source_range
from soundscape.sources import (
    GlobSource,
    neutralize_conditioning,
    remap_filenames,
)

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


def test_hydrophones():
    site = load_site(SITES / "southfork.toml")
    vla1 = site.sensor("VLA1")
    h = vla1.hydrophone(2)
    assert (h.depth_m, h.serial_number) == (21.3, "722015")
    # Every analyzed channel at both sites is described.
    for name in ("southfork", "vineyard"):
        for sensor in load_site(SITES / f"{name}.toml").sensors:
            assert sensor.description
            for c in sensor.channels:
                assert sensor.hydrophone(c).description
    # Undescribed channels give an empty description rather than an error.
    assert vla1.hydrophone(9).depth_m is None


def test_hydrophone_channels_must_be_unique():
    with pytest.raises(ValidationError):
        SensorConfig(
            name="X",
            latitude=0.0,
            longitude=0.0,
            channels=[0],
            detect_channel=0,
            source={"type": "glob", "directory": ".", "pattern": "*"},
            hydrophones=[{"channel": 0}, {"channel": 0, "depth_m": 10.0}],
        )


def test_serial_numbers_identify_one_hydrophone():
    site = load_site(SITES / "vineyard.toml")
    data = site.model_dump()
    vla2 = next(s for s in data["sensors"] if s["name"] == "VLA2")
    vla2["hydrophones"][0]["serial_number"] = "722017"  # also on VLA1
    with pytest.raises(ValidationError, match="722017"):
        type(site).model_validate(data)


def test_depths_follow_serial_numbers_across_sites():
    """Each hydrophone has the same depth at both sites, whatever its mooring
    is called there."""
    depth = {}
    for name in ("southfork", "vineyard"):
        for sensor in load_site(SITES / f"{name}.toml").sensors:
            for h in sensor.hydrophones:
                if h.serial_number and h.serial_number.startswith("722"):
                    depth.setdefault(h.serial_number, set()).add(h.depth_m)
    assert depth and all(len(d) == 1 for d in depth.values())


def test_source_directory_is_relative_to_data_dir(tmp_path):
    relative = GlobSourceConfig(directory="Leg4/SHRU_905", pattern="*.D23")
    assert GlobSource(relative, tmp_path).directory() == tmp_path / "Leg4/SHRU_905"
    with pytest.raises(ValueError, match="--data-dir"):
        GlobSource(relative).directory()
    absolute = GlobSourceConfig(directory=str(tmp_path), pattern="*.D23")
    assert GlobSource(absolute).directory() == tmp_path
    assert GlobSource(absolute, Path("/elsewhere")).directory() == tmp_path


def test_site_files_hold_no_machine_paths():
    for name in ("southfork", "vineyard"):
        for sensor in load_site(SITES / f"{name}.toml").sensors:
            assert not Path(sensor.source.directory).is_absolute()
