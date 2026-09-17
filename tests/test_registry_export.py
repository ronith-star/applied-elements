"""The two registry files must partition the platform's constants exactly.

A parameter is a quantity that could have been measured differently, so it
carries a Value with provenance and uncertainty. A definitional constant is a
definition (a molar mass, a unit conversion, a published classification band)
and carries none. Every hardcoded number must be in exactly one of the two
files, and the classification must be decided by RUNTIME CONTENT rather than
by the text of the assignment.

Both defects this module guards against were real and were reported as fine.

First, the walker traversed only dicts and sequences, so Values held inside
record objects were invisible: PARTITION_PRIORS is a dict of ElementPartition
objects each holding _assumed Values, and none of its parameters reached the
registry. Coverage was reported as 28 parameters when the true figure is 203, a
seven-fold understatement, and the missed constants were then filed as
"definitional" despite carrying explicit ASSUMED bases.

Second, classification tested the assignment source for the literal "Value(",
so any constant built through a helper was misfiled. FORMATION_ENTHALPY
appeared in BOTH files at once, which is precisely the condition the second
file was introduced to make impossible.
"""
from __future__ import annotations

import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_registry as er  # noqa: E402


def test_the_two_files_are_disjoint() -> None:
    """No constant may appear as both a parameter and a definition."""
    vals = er.collect_values()
    defs = er.collect_definitional()
    # collect_values reports the dotted module name; collect_definitional
    # appends ".py". Normalise both to the dotted form before comparing.
    pk = {(r["module"], r["container"]) for r in vals}
    dk = {(r["module"].removesuffix(".py"), r["name"]) for r in defs}
    overlap = pk & dk
    assert not overlap, (
        "these carry provenance AND are reported as definitional, which is "
        "the exact contradiction the second file exists to prevent: "
        f"{sorted(overlap)}"
    )


def test_no_definitional_constant_carries_provenance() -> None:
    """Decided by walking the object, not by reading its source text."""
    defs = er.collect_definitional()
    assert defs, "the platform must have definitional constants"
    unresolved = [r["name"] for r in defs
                  if r["provenance_values_found"] == "UNRESOLVED"]
    assert not unresolved, (
        f"could not import to classify, so provenance is unknown: {unresolved}"
    )
    assert all(r["provenance_values_found"] == "0" for r in defs)


def test_values_inside_record_objects_are_reachable() -> None:
    """Regression: the walk must descend through objects, not stop at them.

    PARTITION_PRIORS is the specific constant the object-blind walker missed.
    It is a dict of ElementPartition records, each holding Values, so a walker
    that stops at the object boundary finds nothing in it.
    """
    import ae.physics.impurity_location as il

    found = list(er._iter_values(il.PARTITION_PRIORS))
    assert found, (
        "PARTITION_PRIORS holds Values inside ElementPartition objects; "
        "finding none means the walker stops at the object boundary again"
    )
    assert any(v.tag.value == "ASSUMED" for v in found), (
        "its priors are explicitly estimates and must be tagged ASSUMED"
    )


def test_every_sourced_parameter_has_a_resolvable_reference() -> None:
    """A SOURCED or MEASURED tag requires a DOI or URL, with no exceptions."""
    vals = er.collect_values()
    bad = [f"{r['module']}.{r['container']} ({r['quantity']})" for r in vals
           if r["tag"] in ("SOURCED", "MEASURED")
           and not (r["doi"] or r["url"])]
    assert not bad, (
        "SOURCED without a DOI or URL is unverifiable; tag it ASSUMED with a "
        f"basis instead:\n  " + "\n  ".join(bad)
    )


def test_no_parameter_is_a_bare_point_estimate() -> None:
    """Every parameter must state an uncertainty, even a wide one."""
    vals = er.collect_values()
    bare = [f"{r['module']}.{r['container']}" for r in vals
            if r["uncertainty"].startswith("point estimate")]
    assert not bare, (
        "a point estimate with no distribution cannot be sampled by the Monte "
        f"Carlo layer, so it silently becomes certain:\n  " + "\n  ".join(bare)
    )


def test_definitional_roles_are_reported_honestly() -> None:
    """Bibliography and prose must not be counted as constants.

    Of 92 rows in the definitional file, 50 are Source objects or access dates
    and 2 are prose strings documenting a basis. Reporting 92 "definitional
    constants" overstated what the platform hardcodes by more than double.
    """
    defs = er.collect_definitional()
    roles = {r["role"] for r in defs}
    known = {"numeric_constant", "numeric_table", "bibliography", "prose"}
    unknown = {r for r in roles if r.startswith("other:")}
    assert not unknown, f"unclassified roles: {sorted(unknown)}"
    assert roles <= known, f"undocumented roles: {sorted(roles - known)}"
    n_const = sum(1 for r in defs
                  if r["role"] in ("numeric_constant", "numeric_table"))
    assert 0 < n_const < len(defs), (
        "the role split must be meaningful: every row a constant, or none, "
        "means the classifier is not discriminating"
    )
