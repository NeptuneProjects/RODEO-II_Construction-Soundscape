"""Inter-pulse sound field on a synthetic recording."""

import docx
import numpy as np
import polars as pl
import pytest

from soundscape import detect, extract, interpulse, kurtosis, spectra, workflow
from soundscape.compare import compare_sites
from soundscape.config import InterpulseConfig
from soundscape.interpulse import (
    band_excess,
    excess_summary,
    is_band_label,
    masking_factors,
    read_interpulse,
    strike_intervals,
)
from soundscape.metrics import read_table
from tests.conftest import SyntheticRecording


@pytest.fixture
def run(site, monkeypatch):
    rec = SyntheticRecording(duration_s=900.0, gap=(300.5, 330.0))
    for module in (detect, extract, kurtosis, spectra, interpulse):
        monkeypatch.setattr(module, "read_segment", rec.read)
    # Background noise from the 60 s pause in piling halfway through the record.
    site.background.offset_s = -455.0
    site.background.duration_s = 50.0
    w = workflow.Workflow(site)
    w.detect()
    w.extract()
    w.metrics()
    w.interpulse()
    return rec


def test_intervals_skip_pauses_and_gaps(run, site):
    m = read_table(site.metrics_path("E1", "VLA1"))
    iv = strike_intervals(m, site.interpulse)
    n_strikes = m["strike"].n_unique()
    # One interval per consecutive pair, less the pause and the gap.
    assert len(iv) == n_strikes - 3
    assert iv["spacing_s"].max() == pytest.approx(1.5, abs=0.01)


def test_t5_t95_offsets_match_t90(run, site):
    m = read_table(site.metrics_path("E1", "VLA1"))
    assert (m["t95"] - m["t5"] - m["tau90"]).abs().max() < 1e-9
    # The synthetic pulse starts at the detection time, within a few ms.
    assert m.filter("valid")["t5"].median() == pytest.approx(0.0, abs=0.01)


def test_segments_between_strikes(run, site):
    s = pl.read_parquet(site.interpulse_segments_path("E1", "VLA1"))
    assert {"Lp", "Lp_band", "in_interval", "valid"} <= set(s.columns)
    assert "in_guan" not in s.columns
    ok = s.filter("valid", pl.col("strike") > 100)
    by_t = (
        ok.group_by(pl.col("t_s").round(1))
        .agg(pl.col("Lp_band").median())
        .sort("t_s")["Lp_band"]
        .to_numpy()
    )
    # The tone burst dominates the first segment and the next strike's onset
    # the last; in between there is only noise.
    assert by_t[0] > by_t[5] + 20
    assert np.ptp(by_t[2:-1]) < 1.0
    # Unit-variance white noise through 20-707 Hz of a 1 kHz Nyquist band.
    noise = 10 * np.log10((707 - 20) / 1000)
    assert ok.filter("in_interval")["Lp_band"].median() == pytest.approx(noise, abs=0.5)
    # The interval follows tau90, so excludes the pulse at both ends.
    inside = ok.filter("in_interval")
    assert inside["t_s"].min() >= 0.1
    assert inside["Lp_band"].max() < noise + 10


def test_interval_medians(run, site):
    i = pl.read_parquet(site.interpulse_intervals_path("E1", "VLA1"))
    v = i.filter(pl.col("n_segments") > 0)
    assert len(v) > 0.9 * len(i)
    strong = v.filter(pl.col("strike") > 100)
    assert strong["duration_s"].median() == pytest.approx(1.5 - 0.045, abs=0.05)
    # Between pulses there is only the noise: about 0 dB above 10 Hz.
    assert strong["Lp"].median() == pytest.approx(0.0, abs=0.5)
    bands = [c for c in i.columns if is_band_label(c)]
    # For white noise the decidecade bands (11.2-707 Hz) hold their share of
    # the 1 kHz band.
    total = 10 * np.log10((10 ** (strong.select(bands).to_numpy() / 10)).sum(axis=1))
    share = 10 * np.log10((707 - 11.2) / 1000)
    assert np.median(total) == pytest.approx(strong["Lp"].median() + share, abs=0.5)


def test_summaries_against_background(run, site):
    events = ["E1"]
    seg = read_interpulse(site, events, "segments")
    bg = read_interpulse(site, events, "background")
    assert len(bg) == 2 * 500
    ex = excess_summary(seg, bg, [3.0, 10.0])
    assert len(ex) == 2
    # Between pulses there is mostly the noise the background noise window holds; only
    # the first segment after t95 (one in about 14) still holds the decaying
    # tail of the tone burst.
    assert ex["dL"].abs().max() < 0.5
    assert 0.05 < ex["above_3"].max() < 0.1
    tail = seg.filter("valid", "in_interval", pl.col("t_s") > 0.15)
    assert tail["Lp_band"].max() < ex["Lp_bg"].max() + 3
    assert ex["interval_share"].min() > 0.8
    b = band_excess(read_interpulse(site, events, "intervals"), bg)
    assert b["nominal_Hz"].min() == pytest.approx(12.5)
    assert b["dL"].abs().median() < 0.5


def test_summaries_and_report(run, site):
    for kind in ("excess", "bands", "hist"):
        assert site.interpulse_summary_path(kind).exists()
    ex = pl.read_csv(site.interpulse_summary_path("excess"))
    assert "definition" not in ex.columns
    workflow.Workflow(site).report()
    figs = site.report_dir / "figures"
    assert not (figs / "interpulse_decay.png").exists()
    assert (figs / "interpulse_cdf.png").exists()
    tex = (site.report_dir / "tables" / "E1_event.tex").read_text()
    assert "L_{p,\\mathrm{ip}}" in tex and "> 3 dB" in tex
    doc = docx.Document(str(site.report_dir / "report.docx"))
    assert any(p.text == "Inter-pulse sound field" for p in doc.paragraphs)
    event_table = next(
        t for t in doc.tables if any("ip" in c.text for c in t.rows[0].cells)
    )
    header = [c.text for c in event_table.rows[0].cells]
    assert any(h.startswith("ΔL") for h in header)
    assert header[-2:] == ["> 3 dB (%)", "> 10 dB (%)"]


def test_compare_masking(run, site, tmp_path):
    out = tmp_path / "compare"
    compare_sites([site], out)
    assert (out / "figures" / "interpulse_vs_range.png").exists()
    assert (out / "figures" / "interpulse_bands.png").exists()
    m = pl.read_csv(out / "tables" / "masking.csv")
    # Noise only between pulses: no excess, so the listening space is intact.
    assert m["dL_median"].abs().max() < 0.5
    assert m["area_left"].min() > 0.95
    mb = pl.read_csv(out / "tables" / "masking_bands.csv")
    assert mb.select("site", "sensor").unique().height == m.height
    assert mb["nominal_Hz"].n_unique() > 10
    # Under 1 dB of excess in any band: over 80% of the range is left with
    # cylindrical spreading.
    assert mb["dL_median"].max() < 1.0
    assert mb["range_left"].min() > 10 ** (-1.0 / 10)
    assert (out / "figures" / "masking_bands.png").exists()
    doc = docx.Document(str(out / "compare.docx"))
    assert any("Listening space" in p.text for p in doc.paragraphs)
    assert any("detected between strikes, as a percentage" in p.text for p in doc.paragraphs)


def test_events_with_few_strikes_are_dropped(run, site):
    from soundscape.events import load_events, too_few_strikes
    from soundscape.interpulse import summarize_site

    assert not too_few_strikes(site, "E1", "VLA1")
    site.min_strikes = 10_000
    assert too_few_strikes(site, "E1", "VLA1")
    assert all(t.is_empty() for t in summarize_site(site, load_events(site)).values())


def test_masking_factors():
    r, a = masking_factors([0.0, 10.0, -3.0], 15.0)
    assert r[0] == 1.0 and r[2] == 1.0
    assert r[1] == pytest.approx(10 ** (-10 / 15))
    assert a[1] == pytest.approx(0.0464, abs=1e-4)


def test_default_config():
    c = InterpulseConfig()
    assert c.segment_s == 0.1 and c.thresholds_db == [3.0, 10.0]


def test_background_vs_range_recovers_slope(site):
    from soundscape.compare import SiteData, background_vs_range

    r_km = np.array([2.0, 4.0, 8.0, 16.0])
    excess = pl.DataFrame(
        {
            "event": ["A", "B", "C", "D"],
            "sensor": "VLA1",
            "range_m": r_km * 1e3,
            "channel": site.sensors[0].detect_channel,
            "Lp_bg": 140.0 - 20.0 * np.log10(r_km),
        }
    )
    data = [SiteData(site, [], pl.DataFrame(), None, {"excess": excess})]
    pts, fits = background_vs_range(data)
    assert len(pts) == 4
    fit = fits.filter(pl.col("sensor") == "all").row(0, named=True)
    assert fit["a_dB"] == pytest.approx(140.0)
    assert fit["b_dB_per_decade"] == pytest.approx(20.0)
    assert fit["r"] == pytest.approx(-1.0)
