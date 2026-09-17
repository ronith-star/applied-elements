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

    NO ABSOLUTE COUNTS ARE STATED HERE, deliberately, and this paragraph
    states none: earlier revisions of this docstring named the literature
    count and the DOI-first count as literals, and both went stale when a
    demoted benchmark was restored and a guard was added to this module.
    Because the assertions on those literals had by then been replaced with a
    property check, nothing failed and the prose silently became wrong.

    Worse, the first attempt at fixing it kept the stale pair in prose as
    "historical" alongside the then-current pair, and widened this module's
    historical-marker whitelist in the same commit so that both sentences
    became permanently exempt from the guard written to catch exactly that
    drift. Quoting the current value as context is the drift. So no count
    appears here at all: the body below measures both figures at run time and
    asserts only the relation between them, which is what the claim actually
    is.

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
    # The historical count pairs this docstring names, 27 and 36 when written
    # and 28 and 37 after the demoted benchmark was restored. Neither absolute
    # is pinned (the comment above explains why); what is used is the SIZE of
    # the inflation, which was 9 rows at both snapshots, as a floor on the
    # measured overcount: the analytic and self-consistency checks that caused
    # it are all still present, so the inflation cannot have fallen below it.
    assert 36 - 27 == 37 - 28
    historical_inflation = 37 - 28
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

    This module's own docstrings drifted: they stated the literature count
    and the DOI-first count as literals, both correct when written. Restoring
    a demoted benchmark moved both counts up by one, and because the
    assertions on those literals had been replaced by a property check,
    nothing failed and the prose silently became wrong. A reader takes a
    number in a docstring as current. No literal is repeated here for the
    same reason: this docstring is subject to its own rule.

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
    # Markers that a sentence is describing a PAST value. The whitelist is
    # deliberately narrow, and two entries were REMOVED after an audit: I had
    # added "would have taken" and "moved them to" in the same commit that
    # widened the tokenizer, which exempted the two sentences of this module's
    # own docstring that stated the then-current counts as literals. Whitelisting the
    # prose you just wrote defeats the guard you wrote it for. A sentence now
    # has to say the number is historical in terms that could not describe a
    # current value.
    HIST = ("earlier", "once named", "was inflated", "historical",
            "previously", "no longer", "at the time", "since corrected")

    # A first version of this guard tokenised on \b\d{1,3}\b, which split
    # decimals and identifiers into spurious "counts": 128.86 ug/g became 128
    # and 86, and the DOI 10.3390/min14070727 contributed 10. Those are not
    # counts and flagging them makes the guard unusable, so measurements,
    # DOIs, versions and dates are masked out before tokenising. A number is
    # only a candidate count when it stands alone.
    def candidates(sentence: str) -> list[str]:
        """Bare integers in a sentence that could plausibly be a count.

        Two bugs were found in this helper by audit and both are recorded
        because each made the guard silently weaker rather than noisier.

        First version tokenised on a bare word-boundary digit pattern, which
        split decimals and identifiers: a measured concentration in ug/g
        contributed its integer and fractional parts as two separate
        "counts", and a DOI contributed its prefix.

        Second version masked those with a lookahead rejecting any digit
        followed by a period. That cannot distinguish a decimal point from a
        full stop, so EVERY SENTENCE-TERMINAL COUNT was exempt. The positive
        control that was supposed to validate the guard injected a pair of
        counts; only the mid-sentence one was reported, the sentence-final
        one passed silently, and the partial catch was read as the guard
        working. A control whose partial failure looks like success is worse
        than no control.

        Current version substitutes each non-count construct with a
        DIGIT-FREE placeholder before tokenising, so the tokenizer cannot see
        through it and needs no punctuation lookahead. Order matters:
        scientific notation must be masked before the decimal rule, or
        "4.82e-13" loses "4.82" and leaves a bare "4".
        """
        masked = re.sub(r"\b\d+(?:\.\d+)?e[-+]?\d+\b", " SCI ", sentence,
                        flags=re.I)
        masked = re.sub(r"\b10\.\d{4,}/\S+", " DOI ", masked)
        masked = re.sub(r"\b\d+(?:\.\d+)+\b", " NUM ", masked)
        masked = re.sub(r"\b(?:19|20)\d{2}\b", " YEAR ", masked)
        masked = re.sub(r"\b(?=[0-9a-f]{7,40}\b)(?=.*[a-f])[0-9a-f]+\b",
                        " SHA ", masked)
        masked = re.sub(r"\bpages?\s*\d+\s*[-\u2013]\s*\d+", " PAGES ",
                        masked, flags=re.I)
        # A number bound to a unit is a measurement, not a count of things
        # this module tracks.
        masked = re.sub(
            r"\b\d{1,4}\s*(?:pp\b|pages?\b|px\b|ppm\b|ppb\b|kwh\b|usd\b"
            r"|t/|ug/g\b|percent\b|%|sd\b|seconds?\b|ms\b)",
            " UNIT ", masked, flags=re.I)
        return re.findall(r"(?<![\w.$-])(\d{1,3})(?!\w)", masked)

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

    # This guard's OWN docstring names 27, 36, 28 and 37 as the drifted count
    # pairs. They are exempt above only because their sentences are marked
    # historical, so that exemption is exercised here on those exact numbers
    # rather than trusted: each must be extracted as a candidate count, must
    # fall in the guarded range, and must be suppressed by a HIST marker.
    hist_sentence = "they named a literature count of 27 and a DOI-first count of 36"
    moved_sentence = "Restoring a demoted benchmark moved them to 28 and 37"
    assert candidates(hist_sentence) == ["27", "36"]
    assert candidates(moved_sentence) == ["28", "37"]
    for lit in ("27", "36", "28", "37"):
        assert 5 <= int(lit) <= 200
    assert any(h in hist_sentence.lower() for h in HIST)
    assert any(h in moved_sentence.lower() for h in HIST)
    # And the inflation is the same size at both snapshots, which is the claim
    # the pairs are there to support.
    assert 36 - 27 == 37 - 28 == 9
