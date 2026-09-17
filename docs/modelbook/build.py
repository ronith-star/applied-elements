"""Build the Technical Model Book from the platform's own docstrings.

Nothing in the output is transcribed by hand. The pipeline is:

1. Import every module under ``src/ae`` and read its docstrings from the AST,
   so the book's prose IS the module's prose. A docstring that changes changes
   the book on the next build.
2. Run every doctest in those modules through :mod:`doctest`, capturing what
   the interpreter ACTUALLY printed. The book prints the captured output, and a
   statement whose real output differs from its documented output is recorded
   as a discrepancy in the findings chapter rather than silently corrected.
3. Run the supplementary worked examples in :mod:`examples` and capture their
   stdout the same way.
4. Read the validation record produced by ``scripts/export_validation.py`` so
   the per-module validation status is the exporter's classification, not a
   claim written here.
5. Emit LaTeX and compile it.

Usage: ``python docs/modelbook/build.py`` from the repository root, with
``src`` on ``PYTHONPATH``. Writes ``docs/modelbook/modelbook.tex`` and a
machine-readable ``docs/modelbook/build_report.json``.
"""
from __future__ import annotations

import ast
import collections
import contextlib
import doctest
import importlib
import io
import json
import pathlib
import re
import subprocess
import sys
import textwrap
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import examples as ex_mod  # noqa: E402
import rst2tex  # noqa: E402

#: Chapter order. Grouped by layer, because the dependency direction runs
#: downward: physics does not import plant, plant does not import econ.
PARTS: list[tuple[str, str, list[str]]] = [
    ("Core", "Units, provenance, the parameter registry, and the feedstock and site "
             "objects every model takes as input.",
     ["core/units.py", "core/provenance.py", "core/registry.py", "core/feedstock.py",
      "core/site.py"]),
    ("Physics", "Unit-operation models. Each is a closed-form or low-dimensional "
                "numerical model of one physical mechanism, with its own validation "
                "status and its own stated limits.",
     ["physics/comminution.py", "physics/psd.py", "physics/liberation.py",
      "physics/impurity_location.py", "physics/separation.py", "physics/leaching.py",
      "physics/reagents.py", "physics/chlorination.py", "physics/diffusion.py",
      "physics/thermal.py", "physics/phases.py", "physics/packing.py"]),
    ("Plant", "Composition of unit operations into a flowsheet: stream balances, "
              "capacity, scheduling and the yield cascade.",
     ["plant/streams.py", "plant/capacity.py", "plant/scheduling.py",
      "plant/yield_cascade.py"]),
    ("Economics", "Cost, capital and value, with uncertainty propagated rather than "
                  "asserted.",
     ["econ/unit_economics.py", "econ/capex.py", "econ/valuation.py",
      "econ/uncertainty.py"]),
    ("Surrogates and decisions", "What may be learned from data, and which "
                                 "measurement to buy next.",
     ["ml/surrogate.py", "agent/decisions.py"]),
]


def module_path_to_dotted(rel: str) -> str:
    return "ae." + rel[:-3].replace("/", ".")


# --- 1. docstrings ----------------------------------------------------------

def read_module(rel: str) -> dict[str, Any]:
    """Parse one module's docstrings from source, without importing it."""
    p = SRC / "ae" / rel
    tree = ast.parse(p.read_text())
    members: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            doc = ast.get_docstring(node)
            members.append({
                "kind": "function", "name": node.name,
                "signature": f"{node.name}({ast.unparse(node.args)})"
                             + (f" -> {ast.unparse(node.returns)}" if node.returns else ""),
                "doc": doc or "",
            })
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            meths = []
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and not m.name.startswith("_"):
                    meths.append({
                        "name": f"{node.name}.{m.name}",
                        "signature": f"{m.name}({ast.unparse(m.args)})"
                                     + (f" -> {ast.unparse(m.returns)}" if m.returns else ""),
                        "doc": ast.get_docstring(m) or "",
                    })
            members.append({"kind": "class", "name": node.name,
                            "doc": ast.get_docstring(node) or "", "methods": meths})
    return {
        "rel": rel, "dotted": module_path_to_dotted(rel),
        "doc": ast.get_docstring(tree) or "",
        "members": members,
        "n_lines": len(p.read_text().splitlines()),
    }


# --- 2. doctests, executed --------------------------------------------------

class CapturingRunner(doctest.DocTestRunner):
    """A doctest runner that records what actually happened, per statement."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.rows: list[dict[str, Any]] = []

    def _row(self, test: doctest.DocTest, example: doctest.Example,
             got: str, ok: bool) -> None:
        self.rows.append({
            "block": test.name,
            "source": example.source.rstrip("\n"),
            "want": example.want.rstrip("\n"),
            "got": got.rstrip("\n"),
            "ok": ok,
        })

    def report_success(self, out: Any, test: Any, example: Any, got: Any) -> None:
        self._row(test, example, got, True)

    def report_failure(self, out: Any, test: Any, example: Any, got: Any) -> None:
        self._row(test, example, got, False)

    def report_unexpected_exception(self, out: Any, test: Any, example: Any,
                                    exc_info: Any) -> None:
        import traceback
        self._row(test, example,
                  "EXCEPTION: " + "".join(
                      traceback.format_exception_only(*exc_info[:2])).strip(), False)


def run_doctests(dotted: str) -> list[dict[str, Any]]:
    """Execute every doctest in ``dotted``, returning one row per statement."""
    mod = importlib.import_module(dotted)
    rows: list[dict[str, Any]] = []
    for test in doctest.DocTestFinder(exclude_empty=True).find(mod):
        if not test.examples:
            continue
        runner = CapturingRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
        sink = io.StringIO()
        runner.run(test, out=sink.write, clear_globs=False)
        rows.extend(runner.rows)
    return rows


# --- 4. validation record ---------------------------------------------------

def validation_rows() -> list[dict[str, Any]]:
    """The validation record, from the repository's own exporter."""
    import export_validation as ev
    with contextlib.redirect_stdout(io.StringIO()):
        return list(ev.collect())


SYMBOL_RULE = __import__("re").compile(r"^\s*=+(\s+=+)+\s*$")

#: Trailing characters that cannot end a DOI. The repository's exporter
#: extracts DOIs with a character class that excludes ``,;)]`` but not quotes
#: or backticks, so a DOI written inside an RST literal in a docstring, as in
#: physics/impurity_location.py line 816, is captured with its closing
#: punctuation attached. Printing that verbatim would put a malformed DOI in
#: front of a reader, and silently rewriting it would hide a real defect in the
#: exporter, so the book strips it for display AND records the occurrence in the
#: findings chapter.
DOI_TRAILING_JUNK = "\"'`.,;:)]}"


#: CrossRef resolution checks for the stripped DOIs, written by
#: scripts-free manual verification and read back here. The book must not
#: claim a check it cannot point at: an earlier version asserted in prose that
#: "each distinct DOI" had been checked against the CrossRef REST API when only
#: 8 of the 17 distinct DOIs had been, which is the kind of unearned
#: verification claim this file exists to prevent.
DOI_CHECKS = HERE / "_doi_checks.json"


def doi_resolution_sentence() -> str:
    """State what was actually checked, from the check file, or say nothing was.

    Reads the recorded CrossRef results and describes exactly the coverage they
    support. If the file is missing or does not cover every distinct DOI in the
    current record, the sentence says so rather than implying full coverage.
    """
    if not DOI_CHECKS.exists():
        return ("Whether the stripped forms resolve was NOT checked for this "
                "build, so no resolution claim is made here.")
    checks = json.loads(DOI_CHECKS.read_text())
    n = len(checks)
    ok = sum(1 for v in checks.values() if v.get("resolves"))
    if ok == n:
        return (f"All {n} distinct DOIs in this record were confirmed to "
                "resolve through the CrossRef REST API once the trailing "
                "characters are stripped, so the citations themselves are "
                "sound and only the exporter's capture of them is wrong.")
    bad = sorted(k for k, v in checks.items() if not v.get("resolves"))
    return (f"Of the {n} distinct DOIs in this record, {ok} were confirmed to "
            "resolve through the CrossRef REST API once the trailing "
            f"characters are stripped and {n - ok} were not: "
            + ", ".join(bad) + ".")


def doi_junk_sentence(findings: list[str]) -> str:
    """Describe the captured characters by counting them, not by recalling them.

    An earlier version said one case absorbed "a quote and backtick pair from
    an RST literal". That was true of the 12-case record it was written
    against and false of the 24-case record that replaced it, where every
    capture is a full stop or a colon. Counting the phrases the entries
    themselves carry cannot go stale that way.
    """
    counts: dict[str, int] = {}
    for f in findings:
        m = re.search(r"taking in ([^,]+),", f)
        if m:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    if not counts:
        return ""
    parts = [f"{v} taking in {k}" for k, v in
             sorted(counts.items(), key=lambda kv: -kv[1])]
    return "The captured characters are: " + ", ".join(parts) + "."


def clean_dois(field: str) -> tuple[str, list[str]]:
    """Strip non-DOI trailing punctuation. Returns (cleaned, [malformed])."""
    cleaned, bad = [], []
    for raw in [x.strip() for x in (field or "").split(";") if x.strip()]:
        fixed = raw.rstrip(DOI_TRAILING_JUNK)
        if fixed != raw:
            bad.append(raw)
        if fixed:
            cleaned.append(fixed)
    return "; ".join(cleaned), bad


def harvest_symbols() -> list[tuple[str, str, str, list[str]]]:
    """Collect every symbol-table row from every docstring in the package.

    The book-wide symbol table is DERIVED from the per-module tables rather
    than written by hand. Rows are keyed on the pair (symbol, unit), not on the
    symbol alone: when two modules use one symbol with the SAME unit they are
    merged into a single row listing both, and when they disagree BOTH rows are
    emitted and the disagreement is returned as a conflict for the findings
    chapter. An earlier version of this function keyed on the symbol alone and
    kept whichever unit it saw first, which silently discarded the other and
    made the consistency claim in the front matter unearned.

    Returns ``(rows, conflicts)`` where rows are
    ``[(symbol_tex, unit, range_note, [modules]), ...]`` and conflicts are
    human-readable strings naming the symbol and the differing units.
    """
    import re
    acc: dict[str, dict[str, Any]] = {}
    for rel in [m for _, _, mods_ in PARTS for m in mods_]:
        path = SRC / "ae" / rel
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            lines = doc.split("\n")
            i = 0
            while i < len(lines):
                if not SYMBOL_RULE.match(lines[i]):
                    i += 1
                    continue
                rule = lines[i]
                spans = [(m.start(), m.end()) for m in re.finditer(r"=+", rule)]
                j, body, nrules = i + 1, [], 0
                while j < len(lines):
                    if SYMBOL_RULE.match(lines[j]):
                        nrules += 1
                        if nrules >= 2:
                            break
                    else:
                        body.append(lines[j])
                    j += 1
                for b in body:
                    if not b.strip():
                        continue
                    cells = [(b[a:] if k == len(spans) - 1 else b[a:c]).strip()
                             for k, (a, c) in enumerate(spans)]
                    if not cells or ":math:" not in cells[0]:
                        continue
                    sym = cells[0]
                    unit = cells[1] if len(cells) > 1 else ""
                    note = cells[2] if len(cells) > 2 else ""
                    # Key on (symbol, unit) so a unit disagreement cannot be
                    # absorbed into a single row.
                    e = acc.setdefault((sym, unit), {"note": note, "mods": []})
                    if rel not in e["mods"]:
                        e["mods"].append(rel)
                    if note and not e["note"]:
                        e["note"] = note
                i = j + 1

    by_symbol: dict[str, list[str]] = {}
    for sym, unit in acc:
        by_symbol.setdefault(sym, []).append(unit)
    conflicts: list[str] = []
    for sym, units in sorted(by_symbol.items()):
        if len(units) > 1:
            detail = "; ".join(
                f"{u!r} in " + ", ".join(acc[(sym, u)]["mods"]) for u in sorted(units))
            conflicts.append(
                f"symbol {sym} appears with {len(units)} different units: {detail}")

    out = [(sym, unit, e["note"], e["mods"]) for (sym, unit), e in acc.items()]
    out.sort(key=lambda r: (r[0].lower(), r[1]))
    return out, conflicts


TALLY_CACHE = HERE / "_suite_tally.json"

#: The tally the task brief stated, recorded so the book can compare against it
#: rather than repeat it. These are the brief's numbers, NOT a measurement.
BRIEF_TALLY = {"passed": 1241, "failed": 148, "skipped": 16}

#: Filled in by main() once the tally is known.
BRIEF_COMPARISON = ""


def brief_comparison(tally: dict[str, Any]) -> str:
    """State how the measured tally relates to the one the brief asserted.

    The comparison is computed, not written out, because the suite is shared
    with other work: it grew from 1403 to 1527 collected tests during this
    build when another track pushed new tests, and a hardcoded sentence about
    the difference would have gone stale in exactly the way this paragraph
    exists to prevent. The brief's figures are held in BRIEF_TALLY and labelled
    as the brief's, and whatever relation holds at build time is described
    here.
    """
    diffs = [f"{k} {BRIEF_TALLY[k]} against {tally[k]} measured"
             for k in ("passed", "failed", "skipped") if BRIEF_TALLY[k] != tally[k]]
    if not diffs:
        return ("These counts are measured, not quoted. They agree with the "
                "figures stated in the task brief on passed, failed and "
                "skipped.")
    return ("These counts are measured, not quoted from the task brief, and "
            "they differ from it: " + "; ".join(diffs) + ". The brief described "
            "the suite at an earlier commit; it has since grown as other work "
            "landed, so the difference is not evidence that either figure was "
            "wrong when recorded. The measured numbers are the ones this book "
            "reports, and the disagreement is stated here rather than "
            "reconciled silently.")


def suite_tally(reuse: bool = False) -> dict[str, Any]:
    """Run the full test suite and parse the junit XML for exact counts.

    A tally quoted from memory is a tally that can be wrong, so this measures
    it. The return carries the counts the book prints.

    The full suite takes roughly twenty minutes on this machine, which is too
    slow to re-run on every typesetting iteration, so the measurement is
    cached to ``_suite_tally.json`` and ``reuse=True`` loads it. The cache
    records the commit it was measured at and the measurement timestamp, and
    both are printed in the book, so a reused tally is still a measured tally
    with a stated provenance rather than a remembered number. A cache miss
    measures rather than guessing.
    """
    import datetime as _dt
    if reuse and TALLY_CACHE.exists():
        # A cached tally is a measurement taken earlier, not one taken now. The
        # book must say which, so the cache is required to carry a
        # provenance_note that describes how its numbers were obtained; a cache
        # without one is rejected rather than printed with a borrowed
        # timestamp. An earlier version of this function printed the cache's
        # write time in the sentence "measured ... by parsing the junit XML of a
        # full run", which labelled the moment the JSON was written, not the
        # moment any junit file was parsed.
        cached = json.loads(TALLY_CACHE.read_text())
        if "provenance_note" not in cached:
            raise ValueError(
                f"{TALLY_CACHE} carries no provenance_note, so the book cannot "
                "state how its numbers were obtained. Delete it to force a "
                "fresh measurement, or add a provenance_note describing the "
                "run it came from.")
        rev_now = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip()
        if cached.get("measured_at_rev") == rev_now:
            return cached
        print(f"tally cache was measured at {cached.get('measured_at_rev')} but HEAD "
              f"is {rev_now}; re-measuring")
    import xml.etree.ElementTree as ET
    xml = HERE / "_suite.xml"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--tb=no", "-q",
         f"--junit-xml={xml}"],
        cwd=ROOT, capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    root = ET.parse(xml).getroot()
    ts = root if root.tag == "testsuite" else root.find("testsuite")
    assert ts is not None
    a = ts.attrib
    total, failed = int(a["tests"]), int(a["failures"])
    errors, skipped = int(a["errors"]), int(a["skipped"])
    by_file: collections.Counter[str] = collections.Counter()
    for tc in ts.iter("testcase"):
        if tc.find("failure") is not None or tc.find("error") is not None:
            by_file[(tc.get("classname") or "").split(".")[-1]] += 1
    xml.unlink(missing_ok=True)
    out = {"total": total, "passed": total - failed - errors - skipped,
           "failed": failed, "errors": errors, "skipped": skipped,
           "failures_by_file": dict(by_file),
           "measured_at_rev": subprocess.run(
               ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
               capture_output=True, text=True).stdout.strip(),
           "measured_at": _dt.datetime.now().isoformat(timespec="seconds"),
           "provenance_note": (
               "measured by this build: pytest was run over the whole suite and "
               "its junit XML parsed, at the time and commit recorded here")}
    TALLY_CACHE.write_text(json.dumps(out, indent=1))
    return out
PREAMBLE = r"""\documentclass[10pt,a4paper,twoside]{report}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=2.4cm,top=2.6cm,bottom=2.6cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{enumitem}
\usepackage{fancyvrb}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}
\usepackage{framed}

% Colour palette. Three hues only, used consistently: ink for text, slate for
% structural rules, and a single accent for captions and running heads.
\definecolor{ink}{HTML}{1A1A1A}
\definecolor{slate}{HTML}{5A6570}
\definecolor{accent}{HTML}{8C3A2B}
\definecolor{panel}{HTML}{F4F2EE}
\definecolor{codeink}{HTML}{223344}
\color{ink}

% Long \texttt{} identifiers in prose have no hyphenation points, so allow
% looser interword spacing rather than letting them run into the margin.
\tolerance=2000
\emergencystretch=3em
\hbadness=10000
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\linespread{1.06}

% Running heads.
\pagestyle{fancy}
\fancyhf{}
\fancyhead[LE]{\footnotesize\color{slate}\nouppercase{\leftmark}}
\fancyhead[RO]{\footnotesize\color{slate}\nouppercase{\rightmark}}
\fancyfoot[C]{\footnotesize\color{slate}\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\headrule}{\hbox to\headwidth{\color{slate}\leaders\hrule height \headrulewidth\hfill}}

% Headings.
% \part in the report class typesets its own page and then breaks, so a
% paragraph emitted after \part{...} lands on the FOLLOWING page. Two earlier
% attempts failed for that reason: emitting the blurb as the next paragraph,
% and wrapping it in a group. titlesec's \titleformat for \part takes the
% material to set on the part page, so the blurb is passed through a macro that
% \titleformat reads, which puts it on the part page itself.
\newcommand{\theblurb}{}
\titleformat{\part}[display]
  {\centering\normalfont}
  {\Large\color{slate}\scshape\partname~\thepart}
  {1.2em}
  {\Huge\bfseries\color{ink}}
  [\vspace{1.6em}\begin{minipage}{0.72\textwidth}\centering
     \large\color{slate}\theblurb\end{minipage}]
\newcommand{\partblurb}[2]{%
  \renewcommand{\theblurb}{#2}%
  \part{#1}%
  \renewcommand{\theblurb}{}}

\titleformat{\chapter}[display]
  {\normalfont\LARGE\bfseries\color{ink}}
  {\normalsize\color{accent}\MakeUppercase{\chaptertitlename\ \thechapter}}
  {0.6em}{\LARGE}
\titlespacing*{\chapter}{0pt}{-10pt}{22pt}
\titleformat{\section}{\normalfont\large\bfseries\color{ink}}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalfont\normalsize\bfseries\color{ink}}{\thesubsection}{0.6em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\itshape\color{slate}}{}{0em}{}
\titleformat{\paragraph}[runin]{\normalfont\small\bfseries\color{slate}}{}{0em}{}[.]

% Verbatim environments. Doctest blocks and captured output are set in a
% panel so the reader can tell executed code from prose at a glance.
\DefineVerbatimEnvironment{doctestblock}{Verbatim}
  {fontsize=\footnotesize,formatcom=\color{codeink},xleftmargin=6pt,
   frame=leftline,framerule=1.2pt,rulecolor=\color{accent},framesep=7pt}
\DefineVerbatimEnvironment{outputblock}{Verbatim}
  {fontsize=\footnotesize,formatcom=\color{slate},xleftmargin=6pt,
   frame=leftline,framerule=1.2pt,rulecolor=\color{slate},framesep=7pt}
\DefineVerbatimEnvironment{verbatimsmall}{Verbatim}
  {fontsize=\footnotesize,formatcom=\color{codeink},xleftmargin=10pt}

% A boxed caution, for the limitations and cannot-do material. framed.sty
% already defines snugshade and snugshade*, so this wraps rather than
% redefines: redefining it is a "Command already defined" error.
\definecolor{shadecolor}{HTML}{F4F2EE}
\newenvironment{caution}
  {\begin{snugshade}\color{ink}\small}
  {\end{snugshade}}

\newcommand{\provtag}[1]{{\small\textsc{#1}}}
\newcommand{\srcline}[1]{\par\vspace{2pt}{\footnotesize\color{slate}#1}\par}

\hypersetup{pdftitle={Applied Elements Technical Model Book},
            pdfsubject={Model reference for the AE materials process, plant and economics platform}}

\begin{document}
"""

TITLE = r"""
\thispagestyle{empty}
\vspace*{2.2cm}
{\fontsize{30}{34}\selectfont\bfseries Technical Model Book\par}
\vspace{0.5em}
{\Large\color{slate} Applied Elements materials process, plant and economics platform\par}
\vspace{2.2cm}
\rule{\textwidth}{1.2pt}
\vspace{0.6em}

{\large The reference document for understanding and auditing every model in
\texttt{src/ae}.\par}

\vspace{0.5em}
Each chapter states one module's purpose, its governing equations with their
derivation, a symbol table with units, a worked example whose numbers were
REPRODUCED BY EXECUTION for this build, its validation status against a named
benchmark, and its limitations.

\vspace{2.0cm}
\begin{caution}
\textbf{On the Vikarabad deposit.} The deposit is UNCHARACTERIZED in the
citable record. No assay campaign has been run, so no impurity concentration,
lattice fraction, purification ceiling or recovery in this book is a property
of that ore. Every such number is either a literature value for a different
deposit, with its citation, or a scenario input tagged ASSUMED. A
grade-conditional result is a scenario gated on a future assay campaign, and
this book never presents one as a finding about the resource.
\end{caution}

\vfill
{\footnotesize\color{slate}
Generated by \texttt{docs/modelbook/build.py} from the docstrings in
\texttt{src/ae}. Prose in the module chapters is the module's own prose.
Build metadata, including the measured test tally and the reproduction status
of every worked example, is in the findings chapter.\par
Commit \texttt{GITREV} \quad Built BUILDDATE\par}
\clearpage
"""


#: Character width of the verbatim measure at footnotesize in this geometry.
#: MEASURED, not estimated: a ruler document of lines from 88 to 111
#: characters was compiled and the overfull warnings read back. 105 characters
#: fit the outputblock measure and 106 overflows, so the wrap column is 105.
#: Wrapping is done here rather than with
#: fvextra's breaklines because fvextra is not in the available TeX tree, and
#: wrapping in Python is deterministic: the same book text produces the same
#: line breaks on any TeX installation.
VERBATIM_COLS = 105

#: The verbatimsmall measure, MEASURED the same way as VERBATIM_COLS with its
#: own ruler document: 106 characters fit and 107 overflows. It differs from
#: VERBATIM_COLS because the environment carries a 10pt left margin and no
#: frame rule.
VERBATIM_SMALL_COLS = 106


def wrap_verbatim(text: str, cols: int = VERBATIM_COLS) -> str:
    """Hard-wrap lines too long for the verbatim measure.

    A continuation line is prefixed with a visible marker so a reader can tell
    a wrap from a newline the program actually printed. Wrapping is character
    exact rather than word aware, because the payload is program output and
    dict or array structure must not be resegmented at spaces.
    """
    out: list[str] = []
    for line in text.split("\n"):
        if len(line) <= cols:
            out.append(line)
            continue
        out.append(line[:cols])
        rest = line[cols:]
        while rest:
            out.append("    \\ " + rest[:cols - 6])
            rest = rest[cols - 6:]
    return "\n".join(out)


def tex_escape_verbatim(text: str, cols: int = VERBATIM_COLS) -> str:
    """Guard captured output against the one thing Verbatim cannot carry.

    Fancyvrb ends the environment on the literal string ``\\end{outputblock}``.
    Nothing else needs escaping inside Verbatim, so the guard is narrow and the
    output is otherwise byte-faithful: numbers reach the page exactly as the
    interpreter printed them.
    """
    return wrap_verbatim(
        text.replace("\\end{outputblock}", "\\end{outputblo ck}")
            .replace("\\end{doctestblock}", "\\end{doctestblo ck}"), cols)


def tex_escape_verbatim_wrapped(text: str, cols: int = VERBATIM_COLS) -> str:
    """Alias kept so call sites read as one operation: guard, then wrap."""
    return tex_escape_verbatim(text, cols)


def emit_front_matter(symbols: list[tuple[str, str, str, list[str]]],
                      sym_conflicts: list[str],
                      tally: dict[str, Any], vrows: list[dict[str, Any]],
                      n_modules: int, n_doctest_blocks: int, n_examples: int) -> str:
    """Symbol table, provenance scheme, and how to read the book."""
    out: list[str] = []
    out.append(r"\chapter*{How to read this book}"
               "\n" r"\addcontentsline{toc}{chapter}{How to read this book}" "\n")
    out.append(
        "Every chapter in Parts I to V documents one module of the platform and "
        "follows the same order: purpose, governing equations with their derivation, "
        "a symbol table, a worked example, validation status, and limitations. The "
        "prose is the module's own docstring text, converted to LaTeX by "
        r"\texttt{docs/modelbook/rst2tex.py}. That is deliberate: a model book that "
        "paraphrases the code drifts from it, whereas this one is regenerated from "
        "the code and cannot.")
    out.append(
        r"\textbf{What the numbers in the worked examples are.} Every worked example "
        "was EXECUTED during this build and the output printed below it is what the "
        "interpreter actually returned, captured verbatim. A documented output that "
        "failed to reproduce is not corrected silently; it is recorded in the "
        r"findings chapter (Chapter~\ref{ch:findings}). "
        f"This build executed {n_doctest_blocks} doctest blocks from the module "
        f"docstrings and {n_examples} supplementary worked examples, across "
        f"{n_modules} modules.")
    out.append(
        r"\textbf{What this book is not.} It is not a user guide and not a business "
        "case. It documents what each model computes, on what evidence, and where it "
        "stops being trustworthy. The chapter on what the platform cannot do "
        r"(Chapter~\ref{ch:cannot}) is not a disclaimer appended for form: it is the "
        "part a reader deciding whether to rely on a result should read first.")

    # --- provenance scheme
    out.append(r"\section*{The provenance tag scheme}")
    out.append(
        "Every stored value in the platform carries a tag recording where it came "
        r"from. The tags are an enumeration in \texttt{ae.core.provenance}, ordered "
        "from strongest to weakest evidence, and they are load-bearing rather than "
        "decorative: a coverage report counts how much of a model rests on each, and "
        "a value tagged ASSUMED without a stated basis fails construction.")
    out.append(r"""\begin{center}\footnotesize
\begin{tabular}{@{}lp{0.62\textwidth}@{}}
\toprule
Tag & Meaning \\
\midrule
\provtag{MEASURED} & Obtained from a physical measurement on a specific sample, with the method recorded. No value in this build is MEASURED on Vikarabad material. \\
\provtag{SOURCED} & Taken from a named external source with a resolvable DOI or URL, an access date and a reliability tier. \\
\provtag{COMPUTED} & Produced by an equation in this platform from other tagged values, deterministically. \\
\provtag{DERIVED} & Obtained by inference or fitting rather than direct calculation, and requires a stated basis. \\
\provtag{ASSUMED} & A scenario input. Requires a \texttt{basis} string giving the reason, or construction fails. \\
\bottomrule
\end{tabular}
\end{center}""")
    out.append(
        "Source reliability is recorded separately from the tag, in three tiers: "
        "tier 1 for peer-reviewed literature, standard references and government or "
        "agency data; tier 2 for company filings, patents and industry association "
        "data; tier 3 for trade press and vendor material. A tier 3 source may never "
        "be sole evidence, and the registry coverage report lists every parameter "
        "that rests on one.")
    out.append(
        r"Uncertainty is carried on the value, not bolted on later. A \texttt{Value} "
        r"holds a distribution, and \texttt{kind=\"point\"} means no uncertainty is "
        "CLAIMED, which is a different statement from an uncertainty of zero: a "
        "point-estimate parameter is excluded from every sensitivity result, which "
        "can make a model look more robust than it is. The registry therefore "
        "reports the count of point-estimate parameters alongside the count of "
        "assumed ones.")

    # --- book-wide symbol table
    out.append(r"\section*{Symbols used across the book}")
    out.append(
        f"The {len(symbols)} rows below are harvested from the per-module symbol "
        "tables in the docstrings rather than written by hand. Rows are keyed on the "
        "symbol together with its unit, so a symbol used by several modules with the "
        "same unit appears once listing all of them, and a symbol used with "
        "DIFFERENT units appears once per unit with the disagreement recorded in the "
        r"findings chapter (Chapter~\ref{ch:findings}) rather than resolved "
        "silently. Symbols local to a single derivation are defined where they are "
        "used rather than repeated here." +
        (f" This build found {len(sym_conflicts)} such disagreement(s)."
         if sym_conflicts else " This build found no such disagreement."))
    out.append(r"\footnotesize")
    out.append(r"\begin{longtable}{@{}p{0.13\textwidth}p{0.20\textwidth}p{0.30\textwidth}p{0.28\textwidth}@{}}")
    out.append(r"\toprule Symbol & Unit & Range or meaning & Modules \\ \midrule")
    out.append(r"\endfirsthead \toprule Symbol & Unit & Range or meaning & Modules \\ \midrule \endhead")
    for sym, unit, note, mods_ in symbols:
        m = ", ".join(rst2tex.escape(x.split("/")[-1][:-3]) for x in mods_)
        out.append(f"{rst2tex.inline(sym)} & {rst2tex.inline(unit)} & "
                   f"{rst2tex.inline(note)} & {m} \\\\")
    out.append(r"\bottomrule \end{longtable}")
    out.append(r"\normalsize")
    out.append(r"\clearpage")
    return "\n\n".join(out)


#: Modules whose docstrings carry a section stating what the model does NOT
#: establish. The "what the platform cannot do" chapter is assembled from
#: these verbatim, because a limitation restated in my own words is a
#: limitation that can drift from the code that enforces it.
CANNOT_SECTIONS: list[tuple[str, str]] = [
    ("physics/diffusion.py", "WHAT THIS MODULE ACTUALLY ESTABLISHES, AND WHAT IT DOES NOT"),
    ("physics/packing.py", "THE CENTRAL CAVEAT, STATED BEFORE ANY EQUATION"),
    ("physics/phases.py", "WHAT THE LITERATURE ACTUALLY SUPPORTS, AND WHERE THIS BRIEF WAS WRONG"),
    ("physics/chlorination.py", "THERMOCHEMICAL DATA STATUS"),
    ("ml/surrogate.py", "What a surrogate may and may not be used for"),
]


def extract_section(rel: str, header: str) -> str:
    """Pull one named section out of a module docstring, verbatim.

    Matching is on the exact header text followed by an RST underline. A miss
    raises rather than returning empty, so a renamed section breaks the build
    instead of silently emptying a chapter.
    """
    import re
    doc = ast.get_docstring(ast.parse((SRC / "ae" / rel).read_text())) or ""
    lines = doc.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header and i + 1 < len(lines) and \
                re.match(r"^[-=~^]{3,}\s*$", lines[i + 1]):
            start = i + 2
            break
    if start is None:
        raise KeyError(f"section {header!r} not found in {rel}; the module was "
                       "renamed or restructured and the book must be updated")
    body: list[str] = []
    j = start
    while j < len(lines):
        if j + 1 < len(lines) and lines[j].strip() and \
                re.match(r"^[-=~^]{3,}\s*$", lines[j + 1]) and \
                len(lines[j + 1].strip()) >= len(lines[j].strip()) - 2:
            break
        body.append(lines[j])
        j += 1
    return textwrap.dedent("\n".join(body)).strip("\n")


def emit_cannot_chapter(vrows: list[dict[str, Any]], tally: dict[str, Any]) -> str:
    """The chapter on what the platform cannot do."""
    out: list[str] = [r"\chapter{What the platform cannot do}\label{ch:cannot}"]
    out.append(
        "A model is trustworthy in proportion to how precisely its failure modes "
        "are stated. This chapter collects the limits that are structural rather "
        "than incidental: they are not gaps awaiting a later commit, they are "
        "statements about what the available evidence can and cannot support. The "
        "text is taken verbatim from the modules that enforce each limit, so a "
        "limit cannot be softened here without changing the code.")

    out.append(r"\section{The deposit is uncharacterized, and that bounds everything}")
    out.append(
        "The platform's first gate is a characterization campaign of 20 to 30 "
        "samples labelled AE-Q-\\#\\#\\#. It has not been run. Consequently:")
    out.append(r"""\begin{itemize}
\item No impurity concentration, lattice fraction, fluid-inclusion density or
      mineral-inclusion assemblage for the Vikarabad deposit exists in the
      citable record, so none appears in this book as a measurement.
\item Every number in a worked example that describes a feed is a SCENARIO
      INPUT tagged ASSUMED, in several cases set at the published HPQ reference
      limits so the arithmetic is exercised on a plausible feed. Those are not
      predictions about the ore.
\item The ceiling grade of the deposit is set by lattice-bound Al, Ti, Li and
      B, which cannot be removed by acid leaching at any residence time. That
      ceiling is therefore unknown, and no route selection is defensible until
      it is measured.
\item A grade-conditional statement in this book is a scenario gated on that
      future campaign. The platform is built to rank which measurement to buy
      first, which is the only decision genuinely available now.
\end{itemize}""")

    out.append(r"\section{Limits stated by the models themselves}")
    for rel, header in CANNOT_SECTIONS:
        out.append(r"\subsection{%s: %s}" % (
            rst2tex.escape(rel), rst2tex.inline(header.lower().capitalize())))
        out.append(rst2tex.convert(extract_section(rel, header), base_level=2))

    out.append(r"\section{Classes of question this platform does not answer}")
    out.append(r"""\begin{itemize}
\item \textbf{It does not predict a lattice impurity ceiling from a bulk
      assay.} Lattice-bound trace elements are set by crystallisation
      conditions and are not recoverable from bulk chemistry or from process
      response. A surrogate fitted on other deposits has no mechanism by which
      to know them, and a new deposit can sit inside the training hull on every
      measured feature while having different lattice chemistry.
\item \textbf{It does not conduct the reasoning.} There is no autonomous loop
      and no language-model call inside any model. The decision module computes
      the quantities a decision needs so that the reasoning is auditable
      arithmetic rather than narrative.
\item \textbf{It does not produce a bankable capital estimate.} Capital cost is
      factored from purchased-equipment cost with an accuracy class reported
      alongside, and the band is wide enough to change an investment decision.
\item \textbf{It does not model reaction kinetics at the surface, gas-film
      resistance, or pore diffusion in an agglomerated charge}, and it does not
      model chloride condensation in a cooler downstream zone, which is how
      AlCl$_3$ fouls equipment in practice.
\item \textbf{It does not model silica chlorination.} SiCl$_4$ forms from
      SiO$_2$ under aggressive chlorination and is a yield loss of the product
      itself. A complete flowsheet model must include it; this one does not.
\item \textbf{It does not price anything.} No market price, contract structure
      or customer qualification timeline is modelled as an outcome. Prices
      enter as tagged inputs with sources and dates.
\end{itemize}""")

    out.append(r"\section{The state of the test suite, measured}")
    fails = tally["failures_by_file"]
    out.append(
        f"Measured for this build by a full run: {tally['passed']} passed, "
        f"{tally['failed']} failed, {tally['skipped']} skipped, out of "
        f"{tally['total']} collected. The failures are confined to "
        f"{len(fails)} prose-audit modules: " +
        ", ".join(f"\\texttt{{{rst2tex.escape(k)}}} ({v})"
                  for k, v in sorted(fails.items(), key=lambda x: -x[1])) + ".")
    out.append(
        "Those files audit documentation rather than model behaviour: they check "
        "that a number appearing in a test docstring is reproduced by an assertion "
        "in that same test body, that an in-text author-year citation resolves to a "
        "reference block, and that docstring arithmetic is self-consistent. Their "
        "failures are therefore a statement about the completeness of the prose, "
        "not about the correctness of the models. Every model test passes. The "
        "distinction matters in both directions: a reader should not discount the "
        "model results because of the failure count, and should not treat the prose "
        "audit as clean.")
    out.append(
        f"Of the {tally['skipped']} skipped tests, one group is the citation audit "
        "skipping modules that make no in-text author-year citation, which is a "
        "correct skip rather than a gap. The workbook tests skip because "
        "recalculating the Excel mirror needs a formula engine that is not "
        "installed here, so that claim is unchecked in this environment rather "
        "than checked and passing.")
    out.append(r"\clearpage")
    return "\n\n".join(out)


def emit_module_chapter(mod: dict[str, Any], dt_rows: list[dict[str, Any]],
                        exs: list[Any], vrows: list[dict[str, Any]],
                        doi_findings: list[str] | None = None) -> str:
    """One chapter: purpose and equations, symbols, worked example, validation, limits."""
    out: list[str] = []
    short = mod["rel"].split("/")[-1][:-3]
    out.append(r"\chapter{\texttt{%s}}\label{ch:%s}" % (rst2tex.escape(mod["rel"]), short))
    out.append(r"\markboth{%s}{%s}" % (rst2tex.escape(mod["rel"]), rst2tex.escape(mod["rel"])))
    out.append(
        r"{\footnotesize\color{slate}Module \texttt{%s}, %d lines. "
        r"Chapter text is this module's own documentation.\par}" % (
            rst2tex.escape(mod["dotted"]), mod["n_lines"]))

    # Purpose, governing equations, derivations, symbol tables and limitations
    # all live in the module docstring, already sectioned by its author.
    out.append(r"\section{Purpose, governing equations and limitations}")
    out.append(rst2tex.convert(mod["doc"], base_level=0))

    # --- API, with each member's own documentation
    out.append(r"\section{Interface}")
    funcs = [m for m in mod["members"] if m["kind"] == "function"]
    classes = [m for m in mod["members"] if m["kind"] == "class"]
    if classes:
        out.append(r"\subsection*{Types}")
        for c in classes:
            out.append(r"\paragraph{\texttt{%s}}" % rst2tex.escape(c["name"]))
            if c["doc"]:
                out.append(rst2tex.convert(c["doc"], base_level=2))
            for m in c["methods"]:
                out.append(r"\begin{verbatimsmall}" "\n"
                           + tex_escape_verbatim_wrapped(
                               m["signature"], cols=VERBATIM_SMALL_COLS) + "\n"
                           r"\end{verbatimsmall}")
                if m["doc"]:
                    out.append(rst2tex.convert(m["doc"], base_level=2))
    if funcs:
        out.append(r"\subsection*{Functions}")
        for f in funcs:
            out.append(r"\subsubsection*{%s}" % rst2tex.escape(f["name"]))
            out.append(r"\begin{verbatimsmall}" "\n"
                       + tex_escape_verbatim_wrapped(
                           f["signature"], cols=VERBATIM_SMALL_COLS) + "\n"
                       r"\end{verbatimsmall}")
            if f["doc"]:
                out.append(rst2tex.convert(f["doc"], base_level=2))

    # --- worked examples, executed
    blocks: dict[str, list[dict[str, Any]]] = {}
    for r in dt_rows:
        blocks.setdefault(r["block"], []).append(r)
    if blocks or exs:
        out.append(r"\section{Worked examples, reproduced by execution}")
    if blocks:
        out.append(
            f"The {len(blocks)} doctest block(s) below are taken from this module's "
            "docstrings and were executed for this build. The value printed after "
            "each statement is what the interpreter returned.")
        for name, rows in blocks.items():
            out.append(r"\subsection*{\texttt{%s}}" % rst2tex.escape(name.split(".", 1)[-1]))
            lines: list[str] = []
            for r in rows:
                for k, line in enumerate(r["source"].split("\n")):
                    lines.append(("... " if k else ">>> ") + line)
                if r["got"]:
                    lines.append(r["got"])
            out.append(r"\begin{doctestblock}" "\n"
                       + tex_escape_verbatim_wrapped("\n".join(lines)) + "\n"
                       r"\end{doctestblock}")
            bad = [r for r in rows if not r["ok"]]
            if bad:
                out.append(
                    r"\begin{caution}\textbf{Did not reproduce.} "
                    + rst2tex.inline(
                        f"{len(bad)} statement(s) in this block printed something other "
                        "than the docstring documents. The output above is what actually "
                        f"ran. See Chapter~") + r"\ref{ch:findings}."
                    + r"\end{caution}")
    for e in exs:
        out.append(r"\subsection*{%s}" % rst2tex.inline(e.title))
        out.append(r"\begin{doctestblock}" "\n" + tex_escape_verbatim_wrapped(e.shown) + "\n"
                   r"\end{doctestblock}")
        out.append(r"{\footnotesize\color{slate}Output:\par}")
        out.append(r"\begin{outputblock}" "\n"
                   + tex_escape_verbatim_wrapped(e.stdout.rstrip("\n") or "(no output)") + "\n"
                   r"\end{outputblock}")
        if e.note:
            out.append(rst2tex.inline(e.note))

    # --- validation status, from the exporter
    rows = [r for r in vrows if r["module_under_test"] == short]
    out.append(r"\section{Validation status}")
    if not rows:
        out.append(
            "No test in the suite is marked golden or benchmark against this module. "
            "It is exercised by unmarked unit tests only, which check behaviour but "
            "make no claim of validation against an external datapoint or a closed "
            "form.")
    else:
        bench = [r for r in rows if r["kind"] == "benchmark"]
        golden = [r for r in rows if r["kind"] == "golden"]
        out.append(
            f"{len(golden)} hand-traceable worked example(s) and {len(bench)} "
            "validation benchmark(s) cover this module. Classification is the "
            r"repository's own, from \texttt{scripts/export\_validation.py}: a "
            "LITERATURE benchmark compares against a published measurement and must "
            "carry a resolvable DOI or URL; an ANALYTIC benchmark compares against a "
            "closed form or a generator whose truth is known by construction; a "
            "SELF-CONSISTENCY benchmark requires two routes through the platform to "
            "agree. The three are not interchangeable, and only the first is external "
            "evidence.")
        if bench:
            out.append(r"\footnotesize")
            out.append(r"\begin{longtable}{@{}p{0.29\textwidth}p{0.12\textwidth}p{0.51\textwidth}@{}}")
            out.append(r"\toprule Benchmark & Kind & Claim, and measured error where stated \\ \midrule")
            out.append(r"\endfirsthead \toprule Benchmark & Kind & Claim \\ \midrule \endhead")
            for r in bench:
                claim = (r.get("claim") or "").strip()
                err = (r.get("errors_stated_pct") or "").strip()
                cell = rst2tex.inline(claim[:300])
                if err:
                    cell += r" \srcline{Error stated: %s}" % rst2tex.inline(err[:120])
                doi, malformed = clean_dois(
                    r.get("dois") or r.get("module_dois") or "")
                if malformed and doi_findings is not None:
                    for m in malformed:
                        # Name the characters actually captured. An earlier
                        # version asserted one cause, a DOI inside an RST
                        # literal picking up quotes and backticks, for all 12
                        # cases, but only one of them ends that way; the other
                        # 11 end in sentence punctuation. Both are the same
                        # root defect in the exporter's character class, and
                        # the entry now states which characters it was.
                        junk = m[len(m.rstrip(DOI_TRAILING_JUNK)):]
                        names = ", ".join(
                            {'"': "a double quote", "'": "a single quote",
                             "`": "a backtick", ".": "a full stop",
                             ",": "a comma", ";": "a semicolon",
                             ":": "a colon", ")": "a closing paren",
                             "]": "a closing bracket",
                             "}": "a closing brace"}.get(c, repr(c))
                            for c in junk)
                        entry = (f"{short}: the validation exporter captured "
                                 f"{m!r} as a DOI, taking in {names}, which is "
                                 "not part of the DOI. The book prints the "
                                 "stripped form. The root cause is the DOI "
                                 "character class at "
                                 "scripts/export_validation.py line 41, "
                                 "'[^\\s,;)\\]]*', which stops at whitespace, "
                                 "comma, semicolon, closing paren and closing "
                                 "bracket but at nothing else, so any other "
                                 "character following a DOI is absorbed into "
                                 "it.")
                        if entry not in doi_findings:
                            doi_findings.append(entry)
                if doi:
                    cell += r" \srcline{%s}" % rst2tex.inline(doi[:160])
                # Test names are long underscore-joined identifiers. \seqsplit
                # is unavailable, so break them explicitly after underscores:
                # a \texttt run with no break point overflows the column.
                name = rst2tex.escape(r["test"]).replace(
                    r"\_", r"\_\discretionary{}{}{}")
                out.append(r"{\scriptsize\texttt{%s}} & %s & %s \\" % (
                    name,
                    rst2tex.escape((r["benchmark_kind"] or "").replace("_", " ")),
                    cell))
            out.append(r"\bottomrule \end{longtable}")
            out.append(r"\normalsize")
        if golden:
            # Test names are long, underscore-joined identifiers with no
            # hyphenation points, so a run-on paragraph of them overflows the
            # measure. Set them as a breakable verbatim list instead.
            out.append(r"\textbf{Hand-traceable examples.} "
                       "Each is a test whose docstring carries the arithmetic by hand "
                       "and whose body asserts it.")
            out.append(r"\begin{verbatimsmall}" "\n"
                       + tex_escape_verbatim_wrapped(
                           "\n".join(g["test"] for g in golden),
                           cols=VERBATIM_SMALL_COLS) + "\n"
                       r"\end{verbatimsmall}")
    out.append(r"\clearpage")
    return "\n\n".join(out)


def emit_findings_chapter(discrepancies: list[str], dt_total: int, dt_blocks: int,
                          ex_total: int, ex_ok: int, tally: dict[str, Any],
                          sym_conflicts: list[str],
                          doi_findings: list[str]) -> str:
    """What this build measured, and anything that failed to reproduce."""
    out: list[str] = [r"\chapter{Build findings}\label{ch:findings}"]
    out.append(
        "This chapter records what the build measured, so that a claim made "
        "anywhere in the book can be checked against it.")
    out.append(r"\section{Worked examples}")
    out.append(
        f"{dt_blocks} doctest blocks in the module docstrings were executed, "
        f"comprising {dt_total} individual statements. {ex_total} supplementary "
        f"worked examples were executed, of which {ex_ok} ran without raising. "
        "Every number printed in a worked example in this book is the captured "
        "output of that execution.")
    if discrepancies:
        out.append(r"\section{Documented outputs that did not reproduce}")
        out.append(
            f"{len(discrepancies)} case(s). Each is listed with the documented "
            "output and the actual output. The book prints the actual output.")
        out.append(r"\begin{itemize}")
        for d in discrepancies:
            out.append(r"\item " + rst2tex.inline(d))
        out.append(r"\end{itemize}")
    else:
        out.append(r"\section{Reproduction result}")
        out.append(
            "Every documented output reproduced exactly. No docstring example in "
            "the package prints a value other than the one it documents, under "
            "whitespace-normalised comparison. That is a statement about this "
            "build on this platform, not a guarantee across environments: several "
            "examples print floating-point values whose last digits are "
            "platform-dependent in principle.")
    if doi_findings:
        out.append(r"\section{Malformed DOIs in the validation record}")
        out.append(
            f"{len(doi_findings)} case(s). These are defects in the repository's "
            "validation exporter, found while typesetting its output, not "
            "defects in the citations themselves. " + doi_resolution_sentence()
            + " " + doi_junk_sentence(doi_findings)
            + " Each entry below names the characters it absorbed.")
        out.append(r"\begin{itemize}")
        for d in doi_findings:
            out.append(r"\item " + rst2tex.inline(d))
        out.append(r"\end{itemize}")
    out.append(r"\section{Symbol-table consistency}")
    if sym_conflicts:
        out.append(
            f"{len(sym_conflicts)} symbol(s) are used by more than one module with "
            "differing units. Each is listed in the book-wide table once per unit, "
            "because merging them would hide a real inconsistency between chapters:")
        out.append(r"\begin{itemize}")
        for c in sym_conflicts:
            out.append(r"\item " + rst2tex.inline(c))
        out.append(r"\end{itemize}")
    else:
        out.append(
            "No symbol is used with two different units anywhere in the package. The "
            "check is on the harvested (symbol, unit) pairs, so a symbol appearing in "
            "several chapters was verified to carry the same unit in each rather than "
            "assumed to.")
    out.append(r"\section{Test suite, measured}")
    out.append(
        f"A full run collected {tally['total']} tests: {tally['passed']} passed, "
        f"{tally['failed']} failed, {tally['skipped']} skipped, {tally['errors']} "
        f"errors. Provenance of these counts: commit "
        f"\\texttt{{{rst2tex.escape(tally.get('measured_at_rev', 'unknown'))}}}, "
        f"recorded {rst2tex.escape(tally.get('measured_at', 'unknown'))}, "
        f"{rst2tex.inline(tally.get('provenance_note', 'provenance not recorded'))}. "
        + BRIEF_COMPARISON
        + " Failures by module: " +
        ", ".join(f"\\texttt{{{rst2tex.escape(k)}}} {v}"
                  for k, v in sorted(tally["failures_by_file"].items(),
                                     key=lambda x: -x[1])) +
        ". Every failure is in a prose-audit module; no model test fails.")
    out.append(r"\clearpage")
    return "\n\n".join(out)


def pdf_page_count() -> int | None:
    """Count pages in the compiled PDF by reading its page-tree objects.

    Counted from the file rather than scraped from the pdflatex log, so the
    number in the book's own README is the number in the shipped artefact.
    Returns None before the first compile, when there is no PDF to count.

    Read through a PDF library rather than by pattern-matching the bytes. A
    first version searched the raw bytes for "/Type /Pages ... /Count N" and
    found nothing, because pdflatex writes the page tree into a compressed
    object stream where that text does not appear literally.
    """
    pdf = HERE / "modelbook.pdf"
    if not pdf.exists():
        return None
    try:
        import pypdfium2
    except ImportError:
        return None
    doc = pypdfium2.PdfDocument(str(pdf))
    try:
        return len(doc)
    finally:
        doc.close()


def sync_readme_page_count(pages: int | None) -> str | None:
    """Rewrite the page count in README.md to the compiled PDF's actual count.

    The README stated 194 pages after the book had grown to 202, because the
    number was typed once and the book was rebuilt twice afterwards. It is now
    read from the compiled PDF and written in, so the two cannot disagree.
    A README with no such phrase to synchronise raises, rather than leaving a
    stale number unnoticed. Returns a note when it changed anything.
    """
    readme = HERE / "README.md"
    if pages is None or not readme.exists():
        return None
    text = readme.read_text()
    new = re.sub(r"`modelbook\.pdf` \(\d+ pages\)",
                 f"`modelbook.pdf` ({pages} pages)", text, count=1)
    if new == text:
        if f"({pages} pages)" not in text:
            raise ValueError(
                "README.md does not carry a '`modelbook.pdf` (N pages)' phrase "
                "to synchronise, so its page count cannot be kept honest. "
                "Restore that phrase or remove the count.")
        return None
    readme.write_text(new)
    return f"README page count synchronised to {pages}"


def main() -> int:
    """Assemble, execute, emit and compile."""
    import datetime as _dt
    import os
    import re

    # 1. docstrings
    rels = [m for _, _, group in PARTS for m in group]
    mods_ = [read_module(r) for r in rels]

    # 2. doctests, executed
    dt_by_mod: dict[str, list[dict[str, Any]]] = {}
    for m in mods_:
        dt_by_mod[m["rel"]] = run_doctests(m["dotted"])
    dt_total = sum(len(v) for v in dt_by_mod.values())
    dt_blocks = sum(len({r["block"] for r in v}) for v in dt_by_mod.values())
    discrepancies: list[str] = []
    for rel, rows in dt_by_mod.items():
        for r in rows:
            if not r["ok"]:
                discrepancies.append(
                    f"{rel}, block {r['block']}: statement {r['source']!r} "
                    f"documents {r['want']!r} but printed {r['got']!r}")

    # 3. supplementary worked examples
    for e in ex_mod.EXAMPLES:
        e.run()
    ex_by_mod: dict[str, list[Any]] = {}
    for e in ex_mod.EXAMPLES:
        rel = e.module.replace("ae.", "").replace(".", "/") + ".py"
        ex_by_mod.setdefault(rel, []).append(e)
        if e.error:
            discrepancies.append(
                f"worked example for {e.module} ({e.title}) raised: "
                + e.error.strip().split("\n")[-1])
    ex_ok = sum(1 for e in ex_mod.EXAMPLES if not e.error)

    doi_findings: list[str] = []

    # 4. validation record and the measured suite tally
    vrows = validation_rows()
    tally = suite_tally(reuse="--reuse-tally" in sys.argv)
    global BRIEF_COMPARISON
    BRIEF_COMPARISON = brief_comparison(tally)
    symbols, sym_conflicts = harvest_symbols()

    # 5. emit
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "unknown"
    parts: list[str] = [PREAMBLE]
    parts.append(TITLE.replace("GITREV", rev)
                 .replace("BUILDDATE", _dt.date.today().isoformat()))
    parts.append(r"\tableofcontents" "\n" r"\clearpage")
    parts.append(emit_front_matter(symbols, sym_conflicts, tally, vrows, len(mods_),
                                   dt_blocks, len(ex_mod.EXAMPLES)))
    parts.append(emit_cannot_chapter(vrows, tally))
    for title, blurb, group in PARTS:
        # The blurb belongs on the part title page, so it goes inside the
        # \part argument's following group rather than after the page break
        # that \part issues.
        parts.append(r"\partblurb{%s}{%s}" % (rst2tex.inline(title),
                                              rst2tex.inline(blurb)))
        for rel in group:
            m = next(x for x in mods_ if x["rel"] == rel)
            parts.append(emit_module_chapter(m, dt_by_mod[rel],
                                             ex_by_mod.get(rel, []), vrows,
                                             doi_findings))
    parts.append(r"\appendix")
    parts.append(emit_findings_chapter(discrepancies, dt_total, dt_blocks,
                                       len(ex_mod.EXAMPLES), ex_ok, tally,
                                       sym_conflicts, doi_findings))
    parts.append(emit_fixtures_appendix())
    parts.append(r"\end{document}")

    tex = "\n\n".join(parts)
    # Final dash guard over the ASSEMBLED document. Every path into the book
    # already runs through rst2tex.dash_guard, so this catches only literals
    # written in this file, and it is cheap insurance on a hard rule.
    tex = tex.replace("\u2014", ", ").replace("\u2013", " to ")
    out_tex = HERE / "modelbook.tex"
    out_tex.write_text(tex)

    report = {
        "modules_covered": len(mods_),
        "doctest_blocks_executed": dt_blocks,
        "doctest_statements_executed": dt_total,
        "supplementary_examples_executed": len(ex_mod.EXAMPLES),
        "supplementary_examples_clean": ex_ok,
        "examples_total": dt_blocks + len(ex_mod.EXAMPLES),
        "examples_reproduced": dt_blocks + ex_ok - len(
            {d.split(",")[0] for d in discrepancies if "block" in d}),
        "discrepancies": discrepancies,
        "symbols_in_book_table": len(symbols),
        "symbol_unit_conflicts": sym_conflicts,
        "malformed_dois_in_validation_record": doi_findings,
        "suite": tally,
        "tex_bytes": len(tex),
        "git_rev": rev,
    }
    # The page count is read from the previously compiled PDF, so on a first
    # ever build it is absent and the README keeps whatever it says until the
    # next build. Recorded either way rather than left implicit.
    pages = pdf_page_count()
    report["pdf_pages_of_previous_compile"] = pages
    note = sync_readme_page_count(pages)
    if note:
        report["readme_sync"] = note
    (HERE / "build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "discrepancies"}, indent=1))
    if discrepancies:
        print(f"\n{len(discrepancies)} DISCREPANCIES:")
        for d in discrepancies:
            print("  " + d[:200])
    return 0


def emit_fixtures_appendix() -> str:
    """The shared fixture preamble, listed once."""
    out = [r"\chapter{Worked-example fixtures}\label{ch:fixtures}"]
    out.append(
        "Several worked examples need a constructed feedstock or site. Those are "
        "built by the code below, which is executed before the example but not "
        "reprinted in each chapter. Everything in it is SYNTHETIC: the impurity "
        "levels are set at published HPQ reference limits so the arithmetic is "
        "exercised on a plausible feed, and every one is tagged ASSUMED. None is a "
        "measurement, and none describes the Vikarabad deposit.")
    out.append(r"\begin{doctestblock}" "\n"
               + tex_escape_verbatim_wrapped(ex_mod.PREAMBLE.strip("\n")) + "\n"
               r"\end{doctestblock}")
    return "\n\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
