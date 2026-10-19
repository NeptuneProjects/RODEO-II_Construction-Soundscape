"""End to end on a synthetic recording: detect -> extract -> metrics -> psd."""

import logging

import docx
import h5py
import numpy as np
import polars as pl
import pytest

from soundscape import detect, extract, spectra, workflow
from soundscape.compare import compare_sites
from soundscape.config import Exclusion
from soundscape.document import Document
from soundscape.events import load_events
from soundscape.metrics import read_table
from soundscape.report import write_coverage_figure
from soundscape.sources import coverage, recorded_intervals
from tests.conftest import FS, T0, SyntheticRecording


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
    m = read_table(site.metrics_path(event, "VLA1"))
    assert len(m) == 2 * len(detections)
    assert m.columns[
        m.columns.index("channel") + 1 : m.columns.index("channel") + 3
    ] == [
        "depth_m",
        "serial_number",
    ]
    assert m.filter(pl.col("channel") == 1)["serial_number"].unique().to_list() == [
        "722014"
    ]
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
    workflow.Workflow(site)._each(lambda e, s: called.append(s.name), "test")
    assert called == []
    assert load_events(site)
    # Naming the event overrides its exclusion.
    workflow.Workflow(site, events=["E1"])._each(
        lambda e, s: called.append(s.name), "test"
    )
    assert called == ["VLA1"]


def write_inventory(path, starts, npts=2000, fs=FS):
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "timestamp": [str(np.datetime64(t, "us")) for t in starts],
            "npts": [npts] * len(starts),
            "sampling_rate": [fs] * len(starts),
        }
    ).write_csv(path)


def test_coverage(tmp_path):
    path = tmp_path / "inv.csv"
    # Two 1 s records with a 1 s gap between them.
    write_inventory(path, [T0, T0 + np.timedelta64(2, "s")])
    s = np.timedelta64(1, "s")
    assert coverage(path, T0, T0 + 4 * s) == pytest.approx(0.5)
    assert coverage(path, T0, T0 + s) == pytest.approx(1.0)
    assert coverage(path, T0 + 10 * s, T0 + 20 * s) == 0.0


def test_recorded_intervals(tmp_path):
    path = tmp_path / "inv.csv"
    s = np.timedelta64(1, "s")
    # Records at 0, 1 and 5 s, each 1 s long: one join, one gap.
    write_inventory(path, [T0, T0 + s, T0 + 5 * s])
    assert recorded_intervals(path, max_gap_s=0.5) == [
        (T0, T0 + 2 * s),
        (T0 + 5 * s, T0 + 6 * s),
    ]
    assert recorded_intervals(path, max_gap_s=10.0) == [(T0, T0 + 6 * s)]


def test_coverage_figure(site, tmp_path):
    event = load_events(site)[0]
    write_inventory(site.inventory_path("VLA1"), [event.start], npts=int(600 * FS))
    path = tmp_path / "coverage.png"
    write_coverage_figure(site, path)
    assert path.stat().st_size > 0


def test_pairs_without_data_are_skipped(site):
    event = load_events(site)[0]
    write_inventory(site.inventory_path("VLA1"), [event.end + np.timedelta64(1, "h")])
    called = []
    workflow.Workflow(site)._each(lambda e, s: called.append(s.name), "test")
    assert called == []


def test_log_to_file(tmp_path):
    root = logging.getLogger()
    handlers = list(root.handlers)
    try:
        path = workflow.log_to_file(tmp_path / "logs", "detect")
        logging.getLogger("soundscape.test").info("hello")
    finally:
        for h in root.handlers[len(handlers) :]:
            h.close()
        root.handlers = handlers
    assert path.parent == tmp_path / "logs" and path.name.startswith("detect_")
    assert "|INFO|soundscape.test|hello" in path.read_text()


def test_report(wf, site):
    site.ambient.offset_s = -600.0
    site.ambient.duration_s = 120.0
    wf.detect(plot=False)
    wf.extract()
    wf.metrics()
    wf.psd()
    wf.ambient()
    wf.report()

    figs = site.report_dir / "figures"
    for name in ("metrics.png", "levels.png", "psd.png"):
        assert (figs / "E1" / name).stat().st_size > 0
    for metric in site.report.range_metrics:
        assert (figs / f"{metric}_vs_range.png").exists()

    tables = site.report_dir / "tables"
    event = (tables / "E1.tex").read_text()
    assert event.startswith("\\begin{tabular}") and "\\bottomrule" in event
    # One row per metric and sensor, plus ambient.
    # Header, depth and S/N, one row per metric (one sensor), ambient.
    assert event.count("\\\\") == 1 + 2 + len(site.report.metrics) + 1
    assert "Ambient" in event
    assert "Event total cSEL" in event
    assert "Hydrophone depth (m) & VLA1 & 16.2/17.7" in event
    assert "Hydrophone S/N & VLA1 & 722013/722014" in event
    detections = (tables / "detections.tex").read_text()
    assert "E1 & P1 & VLA1" in detections
    pulses = (tables / "pulses.tex").read_text()
    n_pulses = len(pl.read_csv(site.metrics_path("E1", "VLA1")).filter("valid"))
    assert (
        sum(line.startswith("E1 & VLA1 &") for line in pulses.splitlines()) == n_pulses
    )
    assert pulses.count("\\makecell") == 2 * (len(site.report.metrics) - 1)

    doc = docx.Document(str(site.report_dir / "report.docx"))
    # Detections and one event table, as native Word tables.
    assert len(doc.tables) == 2
    assert doc.tables[0].rows[1].cells[0].text == "E1"
    event_rows = doc.tables[1].rows
    assert event_rows[1].cells[0].text == "Hydrophone depth (m)"
    # Trailing empty cells are merged.
    assert event_rows[1].cells[2].text == event_rows[1].cells[-1].text
    # Range figures and three figures for the event (no inventory, so no
    # coverage figure).
    assert len(doc.inline_shapes) == len(site.report.range_metrics) + 3
    captions = [p for p in doc.paragraphs if p.style.name == "Caption"]
    assert len(captions) == len(doc.inline_shapes) + len(doc.tables)
    assert sum("SEQ Table" in p._p.xml for p in captions) == 2
    # The event table sits on a landscape page between portrait ones.
    widths = [s.page_width > s.page_height for s in doc.sections]
    assert widths == [False, True, False]


def test_report_without_titles(wf, site):
    site.report.titles = False
    wf.detect(plot=False)
    wf.extract()
    wf.metrics()
    wf.report()
    assert (site.report_dir / "figures" / "E1" / "metrics.png").exists()
    assert (site.report_dir / "report.docx").exists()


def test_rich_text(tmp_path):
    doc = Document()
    doc.paragraph("L_{p,pk} (dB re 1 µPa^{2} s)")
    runs = doc.doc.paragraphs[-1].runs
    assert [r.text for r in runs] == ["L", "p,pk", " (dB re 1 µPa", "2", " s)"]
    assert runs[1].font.subscript and runs[3].font.superscript


def test_compare(wf, site, tmp_path):
    site.ambient.offset_s = -600.0
    site.ambient.duration_s = 120.0
    wf.detect(plot=False)
    wf.extract()
    wf.metrics()
    wf.psd()
    wf.ambient()
    other = site.model_copy(update={"name": "Other"})
    out = tmp_path / "compare"
    compare_sites([site, other], out)

    summary = pl.read_csv(out / "tables" / "summary.csv")
    assert summary["site"].to_list() == [site.name, "Other"]
    valid = pl.read_csv(site.metrics_path("E1", "VLA1")).filter(
        "valid", pl.col("channel") == site.sensors[0].detect_channel
    )
    assert summary["pulses"][0] == len(valid)
    for name in ("levels_vs_range", "depth", "psd"):
        assert (out / "figures" / f"{name}.png").exists()
    doc = docx.Document(str(out / "compare.docx"))
    assert len(doc.tables) == 1 and len(doc.inline_shapes) == 3
