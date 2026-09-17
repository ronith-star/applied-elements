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
    uncertainty.py cites Saltelli and Sobol METHOD papers. The reported
    literature count was inflated from 27 to 36 by that rule and by a
    keyword scan that read "identity" and "reported" too loosely.

    The invariant: a benchmark whose docstring declares its reference value
    exact, or names a closed form, is analytic no matter what its module
    cites.
    """
    rows = ev.collect()
    bm = [r for r in rows if r["kind"] == "benchmark"]

    # The docstring's two counts, asserted rather than recalled. 27 is the
    # corrected literature count; 36 was the inflated figure the DOI-first
    # rule produced, reproduced here by re-running that rule so the size of
    # the inflation is measured and not remembered.
    n_lit = sum(1 for r in bm if r["benchmark_kind"] == "literature")
    assert n_lit == 27, f"corrected literature count moved to {n_lit}"
    inflated = sum(
        1 for r in bm
        if r["dois"] or r["module_dois"] or r["citations_in_docstring"]
    )
    assert inflated == 36, (
        f"the DOI-first rule would classify {inflated} as literature; the "
        "docstring's 36 is that figure"
    )
    assert inflated - n_lit == 9, "the inflation was nine benchmarks"

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
