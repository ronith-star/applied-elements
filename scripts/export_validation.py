"""Export the validation record: every benchmark and golden test, with sources.

Run: python scripts/export_validation.py [outdir]

A BENCHMARK test compares a model output against an external reference and
REPORTS the error. A GOLDEN test is a hand-traceable worked example with an
expected value and a tolerance. Both are pytest markers, so this script reads
the suite rather than a hand-maintained list: a benchmark that is deleted or
renamed cannot linger in the report.

BENCHMARKS COME IN TWO KINDS AND THE DISTINCTION MATTERS. A LITERATURE
benchmark compares against a published measurement, so it needs a citation
with a DOI or URL. An SELF-CONSISTENCY benchmark checks that two routes through the platform agree,
or that a documented methodological property holds end to end (a tornado and a
Sobol ranking disagreeing on an interacting chain is the property being
demonstrated, not a measurement being reproduced). An ANALYTIC benchmark
compares against a closed-form result
that is exact by derivation: the M/M/1 waiting time, the M/D/1 half-relation,
Sobol indices of the Ishigami function, a permutation importance whose true
variance shares are known because the generator is known. An analytic
benchmark has no DOI to carry, and demanding one would be a category error;
conversely, reporting an analytic check as though it were literature
validation would overstate the external evidence. The classifier below assigns
each benchmark to one kind, and the counts are reported separately.

Errors are parsed from each test's docstring where stated. Where a docstring
names a percentage error, it is extracted verbatim rather than recomputed,
because the docstring is the claim under audit.
"""
from __future__ import annotations

import ast
import csv
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

DOI = re.compile(r"(?:doi[:\s]*|https?://(?:dx\.)?doi\.org/)(10\.\d{4,}[^\s,;)\]]*)",
                 re.I)
ERR = re.compile(r"([-+]?\d+\.?\d*)\s*(?:percent|%)", re.I)
# "Author et al. YEAR" or "Author and Author YEAR" or "Author YEAR"
CITE = re.compile(r"\b([A-Z][A-Za-z\-']+(?:\s+(?:et al\.|and\s+[A-Z][A-Za-z\-']+))?)\s+"
                  r"(1[89]\d\d|20\d\d)")


#: Markers of an analytic benchmark: a closed form, an identity, or a
#: generator whose truth is known by construction. Matched against the test
#: name and docstring.
_ANALYTIC_HINTS = (
    "analytic", "closed form", "closed-form", "identity", "ishigami",
    "mm1", "m/m/1", "m/d/1", "mm_d_1", "littles_law", "little's law",
    "by construction", "known mechanism", "synthetic", "exact by",
    "monte_carlo_10k", "per_fold", "baselines",
)

#: Phrases by which a docstring declares it reproduces REPORTED VALUES. A
#: keyword scan alone cannot separate these cases: a benchmark against Xia et
#: al. 2024 describes an "accounting identity" over the paper's measured
#: 128.86 and 24.23 ug/g, so the word "identity" pulled it into the analytic
#: bucket even though the reference values are experimental. Conversely a
#: partition-curve check states its reference is "exact rather than
#: experimental" while citing a textbook chapter. The claim's own words about
#: WHERE ITS NUMBERS COME FROM outrank both the keyword scan and the presence
#: of a DOI.
_REPORTED_VALUE_HINTS = (
    "reported:", "measured value", "published value",
    "the paper's", "experimental data",
)
# A bare "reported " was too loose and mattered: a benchmark REPORTS its own
# error ("Error is REPORTED, not just bounded"), which is a property of the
# test, not evidence of an external measurement. It pulled the M/D/1 queueing
# check and a tornado-versus-Sobol comparison into the literature bucket, and
# both then showed up as literature rows with no source anywhere, which is how
# the looseness was caught.

#: The mirror image: a docstring saying in terms that its reference value is
#: not experimental. This outranks a DOI, because citing the source of a
#: DEFINITION is not validating against a measurement.
_EXACT_REFERENCE_HINTS = (
    "exact rather than experimental", "reference value is exact",
    "known indices", "known in closed form", "true value is known",
)


#: Markers of a self-consistency benchmark: two routes through the platform
#: must agree, or a documented methodological property must hold end to end.
#: These have no external datapoint by construction, so demanding a DOI would
#: be a category error, and counting them as literature validation would
#: overstate the external evidence.
_SELF_CONSISTENCY_HINTS = (
    "full_chain", "full chain", "end to end", "end-to-end",
    "can_disagree", "ranks_what_to_measure", "two ways", "agree",
)


def _benchmark_kind(name: str, doc: str, has_source: bool,
                    module_dois: list[str]) -> str:
    """LITERATURE if a published measurement backs it, else ANALYTIC.

    A test docstring often names the paper in prose ("Ore A of Arellano-Pina
    et al. 2023") while the DOI and access date live on the ``Source`` object
    in the module under test, which is the right place for them: the citation
    belongs to the PARAMETER, not to the test that checks it. So a benchmark
    counts as literature-backed when either the test docstring carries the
    citation or the module under test does. Classifying on the test docstring
    alone reported thirteen well-sourced benchmarks as unclassified.
    """
    hay_pre = (name + " " + doc).lower()
    # A docstring that names its reference as exact is analytic whatever it
    # cites; one that quotes reported values is literature whatever it is
    # called. These two rules are checked before anything else.
    if any(h in hay_pre for h in _EXACT_REFERENCE_HINTS):
        return "analytic"
    if any(h in hay_pre for h in _REPORTED_VALUE_HINTS):
        return "literature"

    # ANALYTIC IS TESTED NEXT, and that order relative to DOIs is the point. Making a
    # module-level DOI sufficient for "literature" reclassified two analytic
    # checks as external validation: test_sobol_reproduces_ishigami_analytic_
    # indices carries no citation of its own and validates against a function
    # whose indices are known in closed form, but the module it tests cites
    # Saltelli and Sobol METHOD papers, so a DOI-first rule labelled it
    # literature-backed. A method citation is not a measurement to reproduce.
    # The reported literature count was inflated by at least two as a result.
    hay = (name + " " + doc).lower()
    if any(h in hay for h in _ANALYTIC_HINTS):
        return "analytic"
    if any(h in hay for h in _SELF_CONSISTENCY_HINTS):
        return "self_consistency"
    if has_source or module_dois:
        return "literature"
    if any(h in hay for h in _SELF_CONSISTENCY_HINTS):
        return "self_consistency"
    return "unclassified"


def _module_dois(module_stem: str) -> list[str]:
    """DOIs declared in the module under test, if it can be located."""
    for cand in (ROOT / "src" / "ae").rglob(f"{module_stem}.py"):
        return sorted(set(DOI.findall(cand.read_text())))
    return []


def _marker(fn: ast.FunctionDef) -> str | None:
    for d in fn.decorator_list:
        t = ast.unparse(d)
        if "benchmark" in t:
            return "benchmark"
        if "golden" in t:
            return "golden"
    return None


def collect() -> list[dict[str, str]]:
    rows = []
    for f in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(f.read_text())
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            kind = _marker(fn)
            if kind is None:
                continue
            doc = (ast.get_docstring(fn) or "").strip()
            one_line = " ".join(doc.split())
            dois = sorted(set(DOI.findall(doc)))
            errs = ERR.findall(doc)
            cites = sorted({f"{a} {y}" for a, y in CITE.findall(doc)})
            has_source = bool(dois or cites)
            stem = f.name.replace("test_", "").replace(".py", "")
            mdois = _module_dois(stem) if kind == "benchmark" else []
            sub = (_benchmark_kind(fn.name, doc, has_source, mdois)
                   if kind == "benchmark" else "")
            rows.append({
                "kind": kind,
                "benchmark_kind": sub,
                "module_under_test": f.name.replace("test_", "").replace(".py", ""),
                "test_file": f.name,
                "test": fn.name,
                "line": str(fn.lineno),
                "claim": one_line[:400],
                "errors_stated_pct": "; ".join(errs[:4]),
                "citations_in_docstring": "; ".join(cites[:4]),
                "dois": "; ".join(dois[:4]),
                "module_dois": "; ".join(mdois[:6]),
            })
    return rows


def main() -> None:
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "registry"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = collect()
    out = outdir / "validation_record.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    kinds = Counter(r["kind"] for r in rows)
    bm = [r for r in rows if r["kind"] == "benchmark"]
    lit = [r for r in bm if r["benchmark_kind"] == "literature"]
    ana = [r for r in bm if r["benchmark_kind"] == "analytic"]
    slf = [r for r in bm if r["benchmark_kind"] == "self_consistency"]
    unc = [r for r in bm if r["benchmark_kind"] == "unclassified"]
    with_err = sum(1 for r in bm if r["errors_stated_pct"])
    mods = Counter(r["module_under_test"] for r in bm)
    print(f"validation_record.csv  {len(rows)} marked tests: {dict(kinds)}")
    print(f"  literature benchmarks (cite a published measurement): {len(lit)}")
    print(f"  analytic benchmarks (exact closed form or known generator): {len(ana)}")
    print(f"  self-consistency benchmarks (two routes must agree): {len(slf)}")
    print(f"  UNCLASSIFIED, needing a source or an analytic basis: {len(unc)}")
    for r in unc:
        print(f"    {r['test_file']}::{r['test']}")
    print(f"  benchmarks stating a numeric error: {with_err}/{len(bm)}")
    print(f"  modules with at least one benchmark: {len(mods)}")


if __name__ == "__main__":
    main()
