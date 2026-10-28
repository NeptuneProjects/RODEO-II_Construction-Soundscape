"""Report wording follows ISO 18405 and 18406: what is measured before each
event is background noise (ambient noise plus self-noise), so no report text,
caption, or table may call it "ambient". Spelling is American throughout."""

import re
from pathlib import Path

import docx
import pytest

from soundscape import detect, extract, interpulse, kurtosis, spectra, workflow
from soundscape.compare import compare_sites
from tests.conftest import SyntheticRecording

# British spellings, each matched as the start of a word.
BRITISH = re.compile(
    r"\b(neighbour|colour|behaviour|favour|harbour|analogue|catalogue|centre|"
    r"metre|litre|fibre|modelling|modelled|labelled|labelling|travelled|"
    r"signalled|cancelled|grey|defence|licence|programme|aluminium|whilst|"
    r"amongst|towards|half-way|judgement|ageing|artefact|"
    r"[a-z]*(?:ise|ised|ising|isation)\b)",
    re.IGNORECASE,
)
# Words ending in -ise that are spelled that way in American English too.
AMERICAN_ISE = {
    "advertise", "advise", "arise", "chastise", "circumcise", "comprise",
    "compromise", "concise", "demise", "despise", "devise", "disguise",
    "enterprise", "excise", "exercise", "expertise", "franchise", "improvise",
    "incise", "merchandise", "noise", "otherwise", "poise", "precise",
    "premise", "promise", "raise", "revise", "rise", "supervise", "surmise",
    "surprise", "televise", "wise", "clockwise", "likewise", "pairwise",
    "stepwise", "elementwise",
}


def british_spellings(text: str) -> list[str]:
    """Words in `text` spelled the British way."""
    out = []
    for m in BRITISH.finditer(text):
        word = re.match(r"[A-Za-z-]+", text[m.start():]).group(0).lower()
        # The word and its base forms (raised -> raise, rising -> rise).
        forms = {word, word[:-1], word[:-2], word[:-3] + "e"}
        if not forms & AMERICAN_ISE:
            out.append(word)
    return out


# A number followed by a space (or a no-break or thin space) and a percent sign.
SPACED_PERCENT = re.compile("[0-9][ \u00a0\u2009\u202f]%")


def test_british_spellings_are_found():
    assert british_spellings("the neighbouring centre, labelled grey") == [
        "neighbouring", "centre", "labelled", "grey"
    ]
    assert british_spellings("normalised; summarise; organisation") == [
        "normalised", "summarise", "organisation"
    ]
    assert british_spellings("noise, otherwise, precise, raised, exercises") == []



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
    british = sorted({w for t in texts for w in british_spellings(t)})
    assert not british, british
    spaced = [t for t in texts if SPACED_PERCENT.search(t)]
    assert not spaced, spaced[:5]


ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "reports" / "hdr_report").glob("*.md"))]


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_documents_use_american_spelling(path):
    text = path.read_text()
    british = sorted(set(british_spellings(text)))
    assert not british, british
    spaced = SPACED_PERCENT.findall(text)
    assert not spaced, spaced
