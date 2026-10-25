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
    decay_curves,
    excess_summary,
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
    # Ambient from the 60 s pause in piling halfway through the record.
    site.ambient.offset_s = -455.0
    site.ambient.duration_s = 50.0
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
    assert (m["t95_s"] - m["t5_s"] - m["T90"]).abs().max() < 1e-9
    # The synthetic pulse starts at the detection time, within a few ms.
    assert m.filter("valid")["t5_s"].median() == pytest.approx(0.0, abs=0.01)


def test_segments_decay_to_noise(run, site):
    s = pl.read_parquet(site.interpulse_segments_path("E1", "VLA1"))
    assert {"Lp", "Lp_band", "in_interval", "in_guan", "valid"} <= set(s.columns)
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
    # Guan's interval starts 0.1 s after the strike and ends 0.5 s before
    # the next; ours follows T90, so excludes the pulse at both ends.
    g = ok.filter("in_guan")
    assert g["t_s"].min() == pytest.approx(0.1)
    assert (g["t_s"] + 0.1).max() <= 1.0 + 1e-6
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
    assert strong["Lp_band_guan"].median() == pytest.approx(
        strong["Lp_band"].median(), abs=0.5
    )
    bands = [c for c in i.columns if c.startswith("L_")]
    # For white noise the decidecade bands (11.2-707 Hz) hold their share of
    # the 1 kHz band.
    total = 10 * np.log10((10 ** (strong.select(bands).to_numpy() / 10)).sum(axis=1))
    share = 10 * np.log10((707 - 11.2) / 1000)
    assert np.median(total) == pytest.approx(strong["Lp"].median() + share, abs=0.5)


def test_summaries_against_ambient(run, site):
    events = ["E1"]
    seg = read_interpulse(site, events, "segments")
    amb = read_interpulse(site, events, "ambient")
    assert len(amb) == 2 * 500
    ex = excess_summary(seg, amb, [3.0, 10.0])
    assert len(ex) == 2
    # Between pulses there is mostly the noise the ambient window holds; only
    # the first segment after t95 (one in about 14) still holds the decaying
    # tail of the tone burst.
    assert ex["dL"].abs().max() < 0.5
    assert 0.05 < ex["above_3"].max() < 0.1
    tail = seg.filter("valid", "in_interval", pl.col("t_s") > 0.15)
    assert tail["Lp_band"].max() < ex["amb_median"].max() + 3
    assert ex["interval_share"].min() > 0.8
    d = decay_curves(seg, amb).filter(pl.col("channel") == 0)
    assert d["dL"][0] > 20
    assert d.filter(pl.col("t_s").is_between(0.25, 1.25))["dL"].abs().max() < 1.0
    b = band_excess(read_interpulse(site, events, "intervals"), amb)
    assert b["nominal"].min() == pytest.approx(12.5)
    assert b["dL"].abs().median() < 0.5


def test_summaries_and_report(run, site):
    for kind in ("excess", "decay", "bands", "hist"):
        assert site.interpulse_summary_path(kind).exists()
    ex = pl.read_csv(site.interpulse_summary_path("excess"))
    assert set(ex["definition"]) == {"interval", "guan"}
    workflow.Workflow(site).report()
    figs = site.report_dir / "figures"
    assert (figs / "interpulse_decay.png").exists()
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
    assert m["area_left_N15"].min() > 0.95
    doc = docx.Document(str(out / "compare.docx"))
    assert any("Listening space" in p.text for p in doc.paragraphs)


def test_masking_factors():
    r, a = masking_factors([0.0, 10.0, -3.0], 15.0)
    assert r[0] == 1.0 and r[2] == 1.0
    assert r[1] == pytest.approx(10 ** (-10 / 15))
    assert a[1] == pytest.approx(0.0464, abs=1e-4)


def test_default_config():
    c = InterpulseConfig()
    assert c.segment_s == 0.1 and c.thresholds_db == [3.0, 10.0]
