"""The validation record must stay complete: no unclassified benchmark.

Every test marked ``benchmark`` claims a model has been validated against
something. The exporter in scripts/export_validation.py assigns each to one of
three kinds, and the distinction is the substance of the claim:

  literature       compares against a published measurement, so it must carry
                   a citation with a DOI or URL, on the test or on the module
                   whose parameter it checks
  analytic         compares against a closed form or a generator whose truth
                   is known by construction (M/M/1, Ishigami, a synthetic
                   dataset). No DOI exists to carry and none is required.
  self_consistency two routes through the platform must agree, or a
                   methodological property must hold end to end. No external
                   datapoint by construction.

An UNCLASSIFIED benchmark is a claim of validation with no stated basis. It is
the failure this guard exists to prevent, because it is the one that would
reach a reader as though it were external evidence.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_validation as ev


def test_every_benchmark_has_a_stated_basis() -> None:
    rows = ev.collect()
    bm = [r for r in rows if r["kind"] == "benchmark"]
    assert bm, "the suite must contain benchmark-marked tests"
    unclassified = [f"{r['test_file']}::{r['test']}" for r in bm
                    if r["benchmark_kind"] == "unclassified"]
    assert not unclassified, (
        "a benchmark with no stated basis reads as external validation it does "
        "not have. Give it a citation, or state the closed form or identity it "
        "checks:\n  " + "\n  ".join(unclassified)
    )


def test_literature_benchmarks_carry_a_resolvable_source() -> None:
    """A literature claim needs a DOI or URL, not just an author-year."""
    rows = ev.collect()
    lit = [r for r in rows
           if r["kind"] == "benchmark" and r["benchmark_kind"] == "literature"]
    assert lit, "the platform must have literature-backed benchmarks"
    bare = [f"{r['test_file']}::{r['test']}" for r in lit
            if not (r["dois"] or r["module_dois"])]
    assert not bare, (
        "these cite an author and year with no DOI anywhere in the test or the "
        "module under test, so a reader cannot check them:\n  "
        + "\n  ".join(bare)
    )


def test_the_three_kinds_are_the_only_kinds() -> None:
    """Reflection guard: the docstring above enumerates the kinds exactly."""
    rows = ev.collect()
    kinds = {r["benchmark_kind"] for r in rows if r["kind"] == "benchmark"}
    documented = {"literature", "analytic", "self_consistency"}
    assert kinds <= documented, (
        f"exporter produced kinds this module does not document: "
        f"{sorted(kinds - documented)}"
    )


def test_no_analytic_check_is_reported_as_literature() -> None:
    """A method citation is not a measurement to reproduce.

    The classifier once treated a DOI anywhere in the module under test as
    sufficient for "literature". That reclassified two analytic checks as
    external validation: a Sobol test against the Ishigami function, whose
    indices are known in closed form, counted as literature-backed because
    uncertainty.py cites Saltelli and Sobol METHOD papers. That rule, together
    with a keyword scan that read "identity" and "reported" too loosely,
    inflated the reported literature count.

    NO ABSOLUTE COUNTS ARE STATED HERE, deliberately. Earlier revisions of
    this docstring named 27 and 36; the measured values later moved to 28 and
    37 when a demoted Xia benchmark was restored and a new guard was added to
    this module, and because the assertions on those literals had by then
    been replaced with a property check, nothing caught the drift. A reader
    would have taken 27 as current. Numbers that change when a benchmark is
    added do not belong in prose; the body below measures both figures at run
    time and asserts only the relation between them, which is what the claim
    actually is.

    The invariant: a benchmark whose docstring declares its reference value
    exact, or names a closed form, is analytic no matter what its module
    cites.
    """
    rows = ev.collect()
    bm = [r for r in rows if r["kind"] == "benchmark"]

    # The inflation the DOI-first rule produced, MEASURED by re-running that
    # rule rather than recalled as a literal. Absolute counts are deliberately
    # not pinned here: adding a benchmark changes them, and this module's own
    # guards are themselves benchmark-marked, so a hardcoded total makes the
    # test fail for the wrong reason. Earlier revisions pinned 27 then 28 and
    # broke on exactly that. What is stable and what the docstring claims is
    # that the DOI-first rule OVERCOUNTS, by the number of analytic and
    # self-consistency checks whose module happens to cite a method paper.
    n_lit = sum(1 for r in bm if r["benchmark_kind"] == "literature")
    inflated = sum(
        1 for r in bm
        if r["dois"] or r["module_dois"] or r["citations_in_docstring"]
    )
    assert inflated > n_lit, (
        f"the DOI-first rule classified {inflated} as literature against the "
        f"corrected {n_lit}; if it no longer overcounts, the distinction this "
        "module exists to enforce has gone away and the test needs rewriting"
    )
    # Each overcounted row must be one the corrected rule calls analytic or
    # self-consistency, which is what makes the delta an inflation rather
    # than an unexplained discrepancy.
    over = [r for r in bm
            if (r["dois"] or r["module_dois"] or r["citations_in_docstring"])
            and r["benchmark_kind"] != "literature"]
    assert len(over) == inflated - n_lit, (
        "the overcount does not decompose into non-literature rows, so the "
        "two rules disagree for some reason other than method citations"
    )
    assert all(r["benchmark_kind"] in ("analytic", "self_consistency")
               for r in over), (
        "a row the DOI-first rule would call literature is neither literature "
        f"nor analytic nor self-consistency: "
        f"{[(r['test'], r['benchmark_kind']) for r in over]}"
    )

    exact_words = ("exact rather than experimental", "known indices",
                   "known in closed form", "reference value is exact")
    mislabelled = [
        f"{r['test_file']}::{r['test']}"
        for r in bm
        if r["benchmark_kind"] == "literature"
        and any(w in r["claim"].lower() for w in exact_words)
    ]
    assert not mislabelled, (
        "these declare an exact reference value yet are counted as literature "
        f"validation:\n  " + "\n  ".join(mislabelled)
    )
    # The 27 to 36 inflation the DOI-first rule produced. Absolute counts are
    # not pinned (the comment above explains why), so the historical pair is
    # used only as a floor on the MEASURED overcount: the analytic and
    # self-consistency checks that caused it are all still present, so the
    # measured inflation cannot have fallen below the 9 rows it was then, and
    # every overcounted row must be one the corrected rule reclassifies.
    historical_inflation = 36 - 27
    assert inflated - n_lit >= historical_inflation
    assert len(over) == inflated - n_lit
    assert len(over) >= historical_inflation


@pytest.mark.benchmark
def test_no_literature_benchmark_is_demoted_by_a_fixture_word() -> None:
    """A synthetic INPUT must not decide the provenance of a REFERENCE.

    This is the direction the earlier guard missed. That one checked only the
    literature-direction mislabel (an analytic check reported as literature),
    so a real regression in the other direction went undetected:
    test_benchmark_xia_2024_residual_is_lattice quotes Xia et al. 2024's
    measured 128.86 and 24.23 ug/g, and was silently classified `analytic`
    because its docstring also says "For the synthetic fixture". The word
    described where the input came from, not where the reference came from.

    The invariant asserted here is general, not a patch for one row: a
    benchmark that cites a DOI and is classified analytic must say IN TERMS
    that its reference value is exact, via an _EXACT_REFERENCE_HINT. A DOI
    plus an analytic label with no such statement is the signature of a
    keyword collision.
    """
    rows = [r for r in ev.collect() if r["kind"] == "benchmark"]
    offenders = []
    for r in rows:
        if r["benchmark_kind"] != "analytic":
            continue
        if not (r["dois"] or r["module_dois"]):
            continue
        hay = (r["test"] + " " + r["claim"]).lower()
        if not any(h in hay for h in ev._EXACT_REFERENCE_HINTS):
            offenders.append(r["test"])
    assert not offenders, (
        "these benchmarks cite a DOI, are labelled analytic, and never state "
        f"that their reference value is exact: {offenders}. Either the "
        "reference is a published measurement (so the label is wrong) or the "
        "docstring must say the value is exact rather than experimental."
    )


def test_all_six_xia_benchmarks_are_literature() -> None:
    """The specific regression, pinned by name.

    Six benchmarks reproduce values from Xia et al. 2024 (doi
    10.3390/min14070727). All six must be literature-classified; one of them
    regressed to analytic once and was not caught because the verification
    after that change re-printed only three of the affected rows.
    """
    rows = {r["test"]: r for r in ev.collect() if r["kind"] == "benchmark"}
    xia = {t: r["benchmark_kind"] for t, r in rows.items() if "xia" in t.lower()}
    assert len(xia) == 6, f"expected 6 Xia benchmarks, found {len(xia)}: {xia}"
    wrong = {t: k for t, k in xia.items() if k != "literature"}
    assert not wrong, f"Xia benchmarks misclassified: {wrong}"


def test_no_docstring_in_this_module_states_an_unasserted_count() -> None:
    """Counts in prose must be asserted, or not stated.

    This module's own docstrings drifted: they named a literature count of 27
    and a DOI-first count of 36, both correct when written. Restoring a
    demoted benchmark moved them to 28 and 37, and because the assertions on
    those literals had been replaced by a property check, nothing failed and
    the prose silently became wrong. A reader takes a number in a docstring as
    current.

    The rule enforced here is narrow and mechanical: any integer in the range
    where these counts live, appearing in a docstring in this file, must also
    appear in that function's body, OR appear in a sentence that marks it as
    historical. Anything else is a live claim with nothing behind it.

    The range is bounded to plausible benchmark counts rather than all
    integers, because years, DOIs and section numbers are not counts and
    flagging them would make the guard unusable.
    """
    import ast
    import re

    src = pathlib.Path(__file__).read_text()
    tree = ast.parse(src)
    HIST = ("earlier", "once", "named", "later moved", "was inflated",
            "historical", "previously", "drifted", "no longer",
            "would have taken", "moved them to")

    # A first version of this guard tokenised on \b\d{1,3}\b, which split
    # decimals and identifiers into spurious "counts": 128.86 ug/g became 128
    # and 86, and the DOI 10.3390/min14070727 contributed 10. Those are not
    # counts and flagging them makes the guard unusable, so measurements,
    # DOIs, versions and dates are masked out before tokenising. A number is
    # only a candidate count when it stands alone.
    def candidates(sentence: str) -> list[str]:
        masked = re.sub(r"\d+\.\d+", " ", sentence)          # decimals
        masked = re.sub(r"\b10\.\d{4,}/\S+", " ", masked)     # DOIs
        masked = re.sub(r"\bv?\d+(\.\d+)+\b", " ", masked)    # versions
        masked = re.sub(r"\b(19|20)\d{2}\b", " ", masked)     # years
        masked = re.sub(r"\b[0-9a-f]{7,40}\b", " ", masked)   # git hashes
        return re.findall(r"(?<![\w.])(\d{1,3})(?![\w.])", masked)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        doc = ast.get_docstring(node) or ""
        if not doc:
            continue
        body_src = "".join(
            ast.unparse(s) for s in node.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        )
        for sentence in re.split(r"(?<=[.!?])\s+", doc):
            if any(h in sentence.lower() for h in HIST):
                continue
            for lit in candidates(sentence):
                if not (5 <= int(lit) <= 200):
                    continue
                if lit not in body_src:
                    offenders.append(f"{node.name}: {lit} in {sentence.strip()[:70]!r}")

    assert not offenders, (
        "these docstring numbers are not asserted anywhere in their test "
        "body and are not marked historical, so they are live claims with "
        "nothing behind them:\n  " + "\n  ".join(offenders)
    )
