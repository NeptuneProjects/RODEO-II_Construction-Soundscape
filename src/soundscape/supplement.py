"""Data supplement: a PDF table of every valid pulse on every hydrophone.

A stand-alone LaTeX document, separate from the report: a title page with a
contents table, a page defining the columns, and one `longtable` per event
(chronological), each starting on a new page, with rows ordered by sensor
(nearest first), hydrophone (shallowest first), and strike. Column headers
repeat on every page, the page header names the event, and PDF bookmarks
lead to each event and sensor. The machine-readable version of the same data
is `metrics.csv`.

The document is written to `<report>/supplement/supplement.tex` and compiled
with latexmk (or pdflatex run twice, for page references) if either is
installed; the PDF is copied to `<report>/supplement.pdf`.
"""

import logging
import shutil
import subprocess
from datetime import date
from pathlib import Path

import polars as pl

from soundscape.config import SiteConfig
from soundscape.report import (
    DIGITS,
    SCALE,
    _num,
    _tex_label,
    _window,
    load_metrics,
    sensors_by_range,
    tex_escape,
)

logger = logging.getLogger(__name__)

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,landscape,margin=0.5in,headheight=14pt,headsep=10pt,footskip=14pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{longtable,booktabs,makecell,array}
\usepackage{fancyhdr,lastpage}
\usepackage[hidelinks,bookmarksopen,bookmarksopenlevel=0]{hyperref}
\setlength{\LTpre}{0pt}
\setlength{\LTpost}{0pt}
\setlength{\tabcolsep}{4pt}
\newcommand{\eventmark}{}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textbf{%(site)s: per-pulse acoustic metrics}}
\fancyhead[C]{\eventmark}
\fancyhead[R]{Page \thepage\ of \pageref*{LastPage}}
\renewcommand{\headrulewidth}{0.4pt}
\hypersetup{pdftitle={%(site)s: per-pulse acoustic metrics (data supplement)}%(pdfauthor)s}
\begin{document}
"""

COLUMN_NOTES = {
    "sensor": "Sensor (mooring) name; names apply to this deployment only.",
    "serial_number": "Hydrophone serial number.",
    "depth_m": "Hydrophone depth below the sea surface.",
    "range_km": "Horizontal distance from the pile to the sensor.",
    "strike": "Strike number within the event, in time order.",
    "time": "Detected strike time (UTC).",
    "SPL_pk": "Peak sound pressure level.",
    "SPL_pkpk": "Peak-to-peak sound pressure level.",
    "SPL_rms90": "Mean-square sound pressure level over the pulse duration $T_{90}$.",
    "SEL_ss": "Single-strike sound exposure level, over the whole pulse window "
    "(100\\,\\% of the pulse energy).",
    "SEL_cum": "Cumulative sound exposure level: running energy sum of "
    "$\\mathrm{SEL}_\\mathrm{ss}$ over the valid strikes of the event so far.",
    "T90": "Pulse duration: the 90\\,\\% energy signal duration, from 5\\,\\% to "
    "95\\,\\% of the cumulative energy.",
    "t_rise": "Rise time: from the start of $T_{90}$ to the peak sound pressure.",
    "t_decay": "Decay time: from the peak sound pressure to the end of $T_{90}$.",
}

ID_COLUMNS = [
    ("sensor", "Sensor", "l"),
    ("serial_number", "Serial", "l"),
    ("depth_m", "\\makecell{Depth\\\\(m)}", "r"),
    ("range_km", "\\makecell{Range\\\\(km)}", "r"),
    ("strike", "Strike", "r"),
    ("time", "Time (UTC)", "l"),
]


def _event_rows(df: pl.DataFrame) -> pl.DataFrame:
    """One event's rows, ordered by sensor range, depth, channel, strike."""
    order = {s: k for k, (s, _) in enumerate(sensors_by_range(df))}
    by = ["_sensor", "depth_m", "channel", "strike"]
    return (
        df.with_columns(
            pl.col("sensor").replace_strict(order).alias("_sensor"),
            pl.col("time").dt.strftime("%Y-%m-%d %H:%M:%S%.3f").alias("time_str"),
        )
        .sort([c for c in by if c in df.columns])
        .drop("_sensor")
    )


def _longtable(event, rows: pl.DataFrame, metrics: list[str]) -> list[str]:
    header = " & ".join(
        [h for _, h, _ in ID_COLUMNS] + [_tex_label(m, stacked=True) for m in metrics]
    )
    n = len(ID_COLUMNS) + len(metrics)
    align = "".join(a for _, _, a in ID_COLUMNS) + "r" * len(metrics)
    out = [
        "\\clearpage",
        (
            f"\\renewcommand{{\\eventmark}}{{Event {tex_escape(event.name)} (pile "
            f"{tex_escape(event.pile)}), {_window(event)}}}"
        ),
        "\\phantomsection",
        f"\\pdfbookmark[0]{{Event {event.name}}}{{ev-{event.name}}}",
        f"\\label{{ev:{event.name}}}",
        "{\\fontsize{6.5}{7.6}\\selectfont",
        f"\\begin{{longtable}}{{{align}}}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        f"\\multicolumn{{{n}}}{{r}}{{\\emph{{Continued on next page}}}} \\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    previous = None
    for r in rows.iter_rows(named=True):
        sensor = tex_escape(r["sensor"])
        if r["sensor"] != previous:
            # Bookmark the first row of each sensor.
            sensor = (
                f"\\pdfbookmark[1]{{{r['sensor']}}}{{ev-{event.name}-{r['sensor']}}}"
                + sensor
            )
            previous = r["sensor"]
        cells = [
            sensor,
            tex_escape(r.get("serial_number") or "--"),
            _num(r.get("depth_m"), 1),
            f"{r['range_m'] / 1e3:.2f}",
            str(r["strike"]),
            r["time_str"],
        ] + [
            _num(None if r[m] is None else r[m] * SCALE.get(m, 1.0), DIGITS.get(m, 2))
            for m in metrics
        ]
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\end{longtable}", "}"]
    return out


def _front_matter(
    site: SiteConfig, parts, n_rows: int, metrics: list[str]
) -> list[str]:
    cfg = site.report
    who = [tex_escape(s) for s in (cfg.author, cfg.affiliation) if s]
    out = [
        "\\pdfbookmark[0]{Contents}{contents}",
        "\\begin{center}",
        "\\vspace*{3em}",
        f"{{\\LARGE\\bfseries {tex_escape(site.name)}: Per-Pulse Acoustic Metrics\\par}}",
        "\\vspace{0.6em}",
        "{\\large Data supplement\\par}",
        "\\vspace{1.2em}",
        *[f"{{\\large {w}\\par}}" for w in who],
        "\\vspace{0.8em}",
        f"Generated {date.today():%Y-%m-%d} from {n_rows:,} valid pulse records.\\par",
        "\\vspace{2em}",
        "\\begin{tabular}{llllrr}",
        "\\toprule",
        "Event & Pile & Window & Sensors & Rows & Page \\\\",
        "\\midrule",
    ]
    for e, rows in parts:
        sensors = ", ".join(tex_escape(s) for s, _ in sensors_by_range(rows))
        out.append(
            f"\\hyperref[ev:{e.name}]{{{tex_escape(e.name)}}} & {tex_escape(e.pile)} & "
            f"{_window(e)} & {sensors} & {len(rows):,} & \\pageref*{{ev:{e.name}}} \\\\"
        )
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}", "\\clearpage"]

    ex = site.extract
    items = [
        (
            "\\textbf{"
            + _plain_header(k, h)
            + "}"
            + (f" {u}" if (u := _unit(k)) else "")
            + f": {COLUMN_NOTES.get(k, '')}"
        )
        for k, h, _ in ID_COLUMNS
    ] + [f"\\textbf{{{_tex_label(m)}}}: {COLUMN_NOTES.get(m, '')}" for m in metrics]
    out += [
        "\\renewcommand{\\eventmark}{Notes}",
        "\\pdfbookmark[0]{Notes}{notes}",
        "\\section*{Notes}",
        (
            "Each row is one valid pile strike (pulse) recorded on one hydrophone. A "
            f"pulse is a window from {ex.pre_s:g}\\,s before to {ex.post_s:g}\\,s "
            "after the detected strike time, calibrated to sound pressure and "
            "high-pass filtered at 10\\,Hz; pulses whose window overlaps a gap in the "
            "recording are left out. Metrics follow ISO 18406:2017, with quantities "
            "and levels as defined in ISO 18405:2017 (reference values 1\\,$\\mu$Pa "
            "and 1\\,$\\mu$Pa$^2$\\,s)."
        ),
        "",
        (
            "Events are in chronological order, each starting on a new page. Within "
            "an event, rows are ordered by sensor (nearest the pile first), "
            "hydrophone (shallowest first), and strike number. Hydrophones are "
            "identified by sensor, serial number, and depth below the sea surface. "
            "The same data are provided in machine-readable form as "
            "\\texttt{metrics.csv}."
        ),
        "",
        "\\subsection*{Columns}",
        "\\begin{itemize}",
        *[f"\\item {i}" for i in items],
        "\\end{itemize}",
    ]
    return out


def _plain_header(key: str, header: str) -> str:
    return {"depth_m": "Depth", "range_km": "Range"}.get(key, header)


def _unit(key: str) -> str:
    return {"depth_m": "(m)", "range_km": "(km)"}.get(key, "")


def write_supplement(site: SiteConfig, events: list, build: bool = True) -> Path | None:
    """Write the data supplement for the given events and, if LaTeX is
    installed and `build`, compile it. Returns the PDF (or, without LaTeX,
    the .tex file)."""
    if not site.metrics_summary_path.exists():
        logger.warning(f"[supplement] missing {site.metrics_summary_path}")
        return None
    metrics = site.report.metrics
    events = sorted(events, key=lambda e: e.start)
    df = load_metrics(site).filter(pl.col("event").is_in([e.name for e in events]))
    for col in ("serial_number", "depth_m"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))
    parts = [
        (e, rows)
        for e in events
        if not (rows := _event_rows(df.filter(pl.col("event") == e.name))).is_empty()
    ]

    build_dir = site.report_dir / "supplement"
    build_dir.mkdir(parents=True, exist_ok=True)
    tex = build_dir / "supplement.tex"
    author = f",pdfauthor={{{site.report.author}}}" if site.report.author else ""
    with open(tex, "w") as f:
        f.write(PREAMBLE % {"site": tex_escape(site.name), "pdfauthor": author})
        f.write("\n".join(_front_matter(site, parts, len(df), metrics)) + "\n")
        for e, rows in parts:
            f.write("\n".join(_longtable(e, rows, metrics)) + "\n")
        f.write("\\end{document}\n")
    logger.info(f"[supplement] {len(df):,} rows -> {tex}")
    if not build:
        return tex
    pdf = compile_latex(tex)
    if pdf is None:
        return tex
    dest = site.report_dir / "supplement.pdf"
    shutil.copyfile(pdf, dest)
    logger.info(f"[supplement] {dest}")
    return dest


def compile_latex(tex: Path) -> Path | None:
    """Compile `tex` in its directory with latexmk, or pdflatex twice (for
    page references). None, with a warning, if neither is installed."""
    cwd = tex.parent
    flags = ["-interaction=nonstopmode", "-halt-on-error"]
    if shutil.which("latexmk"):
        commands = [["latexmk", "-pdf", *flags, tex.name]]
    elif shutil.which("pdflatex"):
        commands = [["pdflatex", *flags, tex.name]] * 2
    else:
        logger.warning(
            f"[supplement] no LaTeX found (latexmk or pdflatex); compile {tex} "
            "with pdflatex (twice) to make the PDF"
        )
        return None
    for cmd in commands:
        logger.info(f"[supplement] {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            log = tex.with_suffix(".log")
            tail = log.read_text(errors="replace")[-3000:] if log.exists() else ""
            raise RuntimeError(f"LaTeX failed on {tex}:\n{tail}")
    return tex.with_suffix(".pdf")
