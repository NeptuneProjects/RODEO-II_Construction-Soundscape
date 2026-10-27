"""Report wording follows ISO 18405 and 18406: what is measured before each
event is background noise (ambient noise plus self-noise), so no report text,
caption, or table may call it "ambient"."""

import docx

from soundscape import detect, extract, interpulse, kurtosis, spectra, workflow
from soundscape.compare import compare_sites
from tests.conftest import SyntheticRecording


def test_no_ambient_in_report_wording(site, monkeypatch, tmp_path):
    rec = SyntheticRecording(duration_s=900.0, gap=(300.5, 330.0))
    for m in (detect, extract, kurtosis, spectra, interpulse):
        monkeypatch.setattr(m, "read_segment", rec.read)
    # Background noise from the 60 s pause in piling halfway through the record.
    site.background.offset_s, site.background.duration_s = -455.0, 50.0
    w = workflow.Workflow(site)
    for stage in (
        "detect",
        "extract",
        "metrics",
        "psd",
        "background",
        "kurtosis",
        "interpulse",
        "report",
    ):
        getattr(w, stage)()
    compare_sites([site], tmp_path / "compare")
    texts = []
    for path in (
        site.report_dir / "report.docx",
        tmp_path / "compare" / "compare.docx",
    ):
        d = docx.Document(str(path))
        texts += [p.text for p in d.paragraphs]
        texts += [c.text for t in d.tables for r in t.rows for c in r.cells]
    for f in (site.report_dir / "tables").glob("*"):
        texts.append(f.read_text())
    bad = [t for t in texts if "ambient" in t.lower()]
    assert not bad, bad[:5]
