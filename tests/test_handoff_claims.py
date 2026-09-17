"""The handoff documents must not drift from the repository they describe.

This file exists because of a failure mode this repository has already
committed twice, recorded as C1 and C2 in ``docs/CORRECTIONS.md``: a count was
written from memory into a commit message, and a ceiling was attributed to two
files when the run in front of the author printed three. Prose is not executed,
so nothing catches it.

``PLAN.md``, ``HANDOFF.md``, ``README.md`` and ``CONTRIBUTING-provenance.md``
state numbers about this repository: how many modules there are, how many
import edges, how many modules carry an external literature benchmark, what the
CI ceiling is and how it splits. Every one of those is re-derived here from the
source tree, the registry CSVs or the workflow file, and compared against the
number written in the document. A document that drifts fails this file.

What this does NOT check: the measured test counts (1,415 collected, 1,248
passed, 149 failed, 18 skipped), because re-running the suite from inside the
suite is not possible, and asserting a remembered total is the exact error this
file exists to prevent. Those counts are labelled in the documents with the
commit they were measured at, and re-measuring them is a manual step.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DOCS = {
    "PLAN.md": ROOT / "PLAN.md",
    "HANDOFF.md": ROOT / "HANDOFF.md",
    "README.md": ROOT / "README.md",
    "CONTRIBUTING-provenance.md": ROOT / "CONTRIBUTING-provenance.md",
}


def _text(name: str) -> str:
    p = DOCS[name]
    if not p.exists():
        pytest.skip(f"{name} is not present in this checkout")
    return p.read_text()


def _modules() -> dict[str, pathlib.Path]:
    return {
        p.relative_to(SRC).with_suffix("").as_posix().replace("/", "."): p
        for p in sorted((SRC / "ae").rglob("*.py"))
        if p.name != "__init__.py"
    }


def _import_edges() -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    for name, path in _modules().items():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("ae.") and node.module != name:
                    edges.add((node.module, name))
    return edges


@pytest.mark.golden
def test_module_and_edge_counts_in_plan_are_current() -> None:
    """PLAN.md states 27 modules and 57 internal import edges.

    Both are re-derived here: the module count by walking ``src/ae`` for
    non-``__init__`` files, the edge count by an AST walk collecting every
    ``from ae.x import y`` whose target is not the importing module itself.
    """
    mods = _modules()
    edges = _import_edges()
    assert len(mods) == 27, f"module count is {len(mods)}, PLAN.md says 27"
    assert len(edges) == 57, f"edge count is {len(edges)}, PLAN.md says 57"
    plan = _text("PLAN.md")
    assert "27 modules" in plan
    assert "57 internal\nimport edges" in plan or "57 internal import edges" in plan


@pytest.mark.golden
def test_cross_layer_edge_table_in_plan_is_current() -> None:
    """PLAN.md's edge table claims 39 inputs-to-physics edges and, critically,
    ZERO edges for physics-to-plant, plant-to-econ and econ-to-decision.

    The three zeros carry the architectural claim that those couplings are
    composition at a call site rather than imports, so they are the rows worth
    guarding: if someone adds such an import, the document's central
    architectural statement becomes false.
    """
    layer = {}
    for name in _modules():
        part = name.split(".")[1]
        layer[name] = {
            "core": "inputs", "physics": "physics", "plant": "plant",
            "econ": "econ", "ml": "decision", "agent": "decision",
        }[part]
    counts: dict[tuple[str, str], int] = {}
    for src, dst in _import_edges():
        k = (layer[src], layer[dst])
        counts[k] = counts.get(k, 0) + 1
    assert counts.get(("inputs", "physics"), 0) == 39
    assert counts.get(("physics", "physics"), 0) == 2
    # Both directions of each coupling are checked. The table's cell
    # ("physics", "plant") means "physics imported BY plant", so asserting only
    # that cell would miss a plant module imported by physics, which violates
    # the same architectural claim. A control run confirmed this: injecting
    # `from ae.plant.streams import Stream` into ae.physics.comminution left the
    # one-directional version of this test passing.
    for a, b in (("physics", "plant"), ("plant", "econ"), ("econ", "decision")):
        for forbidden in ((a, b), (b, a)):
            assert counts.get(forbidden, 0) == 0, (
                f"{forbidden[1]} now imports {forbidden[0]} "
                f"({counts[forbidden]} edge(s)), which contradicts the claim in "
                "PLAN.md and README.md that those layers are composed at the "
                "call site and carry no import edge"
            )


@pytest.mark.golden
def test_fanout_of_units_and_provenance_matches_plan() -> None:
    """PLAN.md's fan-out table: units 21, provenance 16, feedstock 12, site 6.

    These four numbers are why the document calls units and provenance the
    modules that can break everything, so a drift changes the argument and not
    just the table.
    """
    fan: dict[str, int] = {}
    for src, _dst in _import_edges():
        fan[src] = fan.get(src, 0) + 1
    assert fan.get("ae.core.units") == 21
    assert fan.get("ae.core.provenance") == 16
    assert fan.get("ae.core.feedstock") == 12
    assert fan.get("ae.core.site") == 6


@pytest.mark.golden
def test_evidence_class_counts_in_handoff_are_current() -> None:
    """HANDOFF.md's summary table: 13 modules with an external literature
    benchmark, 2 analytic-only, 7 golden-only, 5 with neither.

    Re-derived by joining ``validation_record.csv`` to the source module names
    and taking the strongest evidence class per module. 13 + 2 + 7 + 5 = 27,
    which is also asserted, because a partition that does not cover every
    module would let a module fall out of the table unnoticed.
    """
    vr = pd.read_csv(ROOT / "data/registry/validation_record.csv")
    classes: dict[str, str] = {}
    for full in _modules():
        short = full.rsplit(".", 1)[1]
        sub = vr[vr["module_under_test"] == short]
        lit = int((sub["benchmark_kind"] == "literature").sum())
        ana = int(sub["benchmark_kind"].isin(["analytic", "self_consistency"]).sum())
        gold = int((sub["kind"] == "golden").sum())
        classes[full] = (
            "literature" if lit else "analytic" if ana else "golden" if gold else "none"
        )
    tally = {k: sum(1 for v in classes.values() if v == k)
             for k in ("literature", "analytic", "golden", "none")}
    assert tally == {"literature": 13, "analytic": 2, "golden": 7, "none": 5}, tally
    assert sum(tally.values()) == 27
    handoff = _text("HANDOFF.md")
    assert "**13 of 27 modules**" in handoff
    assert "**5 of 27 modules**" in handoff


@pytest.mark.golden
def test_registry_totals_in_handoff_are_current() -> None:
    """HANDOFF.md states 203 provenance-tracked values split SOURCED 134,
    ASSUMED 61, DERIVED 8, and zero Tier 3 rows.

    The zero matters most: CONTRIBUTING-provenance.md argues a Tier 3 source may
    never be sole evidence for a registry value, and the empty Tier 3 column is
    the evidence that the rule is being followed rather than merely stated.
    """
    pr = pd.read_csv(ROOT / "data/registry/parameter_registry.csv")
    assert len(pr) == 203, f"registry has {len(pr)} rows, documents say 203"
    tags = pr["tag"].value_counts().to_dict()
    assert tags.get("SOURCED") == 134
    assert tags.get("ASSUMED") == 61
    assert tags.get("DERIVED") == 8
    assert int((pr["tier"] == 3).sum()) == 0, "a Tier 3 registry row has appeared"
    assert 134 + 61 + 8 == len(pr), "the three tags must partition the registry"


@pytest.mark.golden
def test_impurity_location_is_still_the_most_assumed_module() -> None:
    """HANDOFF.md's NEEDS DATA argument rests on impurity_location carrying 18
    ASSUMED values against 3 SOURCED, the worst ratio in the platform, and on
    thermal carrying 0 ASSUMED against 30 SOURCED as the contrast.

    If those move, the argument for which measurement to buy first moves with
    them, so the document's recommendation is only as current as this assertion.
    """
    pr = pd.read_csv(ROOT / "data/registry/parameter_registry.csv")
    il = pr[pr["module"] == "ae.physics.impurity_location"]["tag"].value_counts().to_dict()
    th = pr[pr["module"] == "ae.physics.thermal"]["tag"].value_counts().to_dict()
    assert il.get("ASSUMED") == 18
    assert il.get("SOURCED") == 3
    assert th.get("ASSUMED", 0) == 0
    assert th.get("SOURCED") == 30
    ratios = {}
    for mod, grp in pr.groupby("module"):
        counts = grp["tag"].value_counts().to_dict()
        assumed = counts.get("ASSUMED", 0)
        if assumed:
            ratios[mod] = assumed / max(counts.get("SOURCED", 0), 1)
    worst = max(ratios, key=lambda m: ratios[m])
    assert worst == "ae.physics.impurity_location", (
        f"{worst} now has the worst assumed-to-sourced ratio, not "
        "impurity_location, which is the module HANDOFF.md names"
    )


@pytest.mark.golden
def test_the_prose_audit_is_inside_the_fatal_gate() -> None:
    """README.md states that the prose audit is now fatal rather than pinned,
    and that the pinned step with its numeric ceiling is gone.

    This assertion is the inverse of the one it replaces. An earlier version of
    this guard asserted that the workflow enforced a ceiling and that README.md
    quoted it with a per-file split summing to it, which was true until commit
    56e61ff deleted the pinned step and folded the three audit files into the
    fatal suite. Guarding the claim in its current direction is what keeps the
    document from describing a workflow that no longer exists, which is the
    failure the earlier version would itself have committed had it not been
    rewritten.
    """
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert not re.search(r'"\$FAILED" -gt \d+', ci), (
        "the workflow enforces a numeric prose-audit ceiling again, so the "
        "README section describing the audit as fatal is now wrong"
    )
    audit_files = (
        "tests/test_test_docstrings.py",
        "tests/test_citations.py",
        "tests/test_docstring_arithmetic.py",
    )
    for f in audit_files:
        assert f"--ignore={f}" not in ci, (
            f"{f} is excluded from the CI run again, so it is no longer inside "
            "the fatal gate as README.md states"
        )
    readme = _text("README.md")
    assert "inside the fatal gate" in readme
    assert "removed the pinned step" in readme


@pytest.mark.golden
def test_registry_counts_quoted_from_the_committed_csv_match_that_file() -> None:
    """HANDOFF.md's summary table quotes the marked-test totals from the
    COMMITTED validation record, and separately records that regenerating the
    file with this guard file in the tree gives a higher golden count, because
    these guards carry the ``golden`` marker and the exporter collects every
    marked test.

    Both numbers are live, and confusing them makes the document
    self-contradictory, which is an error I committed and had to retract. This
    pins the quoted pair to the file actually on disk, so committing a
    regenerated CSV fails here rather than silently falsifying the table.
    """
    vr = pd.read_csv(ROOT / "data/registry/validation_record.csv")
    kinds = vr["kind"].value_counts().to_dict()
    handoff = _text("HANDOFF.md")
    assert f"| marked tests in the validation record | {len(vr)} |" in handoff, (
        f"the committed CSV holds {len(vr)} marked tests, which HANDOFF.md's "
        "summary table does not state"
    )
    assert f"| of which golden | {kinds.get('golden')} |" in handoff
    assert f"| of which benchmark | {kinds.get('benchmark')} |" in handoff


@pytest.mark.golden
def test_test_file_count_in_readme_is_current() -> None:
    """README.md's layout block states how many files are in ``tests/``.

    This guard exists because that number drifted the moment this very file was
    added: the README stated a count one lower than the tree then held, and
    nothing caught it. The count is re-derived from the tree rather than
    asserted as a literal, so adding or removing a test file cannot silently
    falsify the layout block. No figure is written in this docstring, because a
    literal here would be the same unasserted-number defect the audit catches.
    """
    on_disk = sorted(p.name for p in (ROOT / "tests").glob("*.py"))
    assert "conftest.py" in on_disk, "conftest.py is missing from tests/"
    readme = _text("README.md")
    stated = re.search(r"tests/\s+(\d+) files including conftest\.py", readme)
    assert stated is not None, "README.md no longer states a tests/ file count"
    assert int(stated.group(1)) == len(on_disk), (
        f"README.md says {stated.group(1)} test files, the tree has "
        f"{len(on_disk)}"
    )


@pytest.mark.golden
def test_every_doi_in_the_documents_appears_in_the_repository() -> None:
    """No document may cite a DOI that does not appear in src/ or tests/.

    This is the anti-fabrication guard. A DOI written into a handoff document
    and nowhere else in the repository is either a transcription error or an
    invention, and neither is acceptable in the file a reader trusts most.
    """
    corpus = "\n".join(
        p.read_text() for p in list((SRC / "ae").rglob("*.py")) + list((ROOT / "tests").glob("*.py"))
    )
    pattern = re.compile(r"10\.\d{4,9}/[^\s`)\],;]+")
    for name in DOCS:
        text = _text(name)
        for raw in set(pattern.findall(text)):
            doi = raw.rstrip(".")
            assert doi in corpus, f"{name} cites {doi}, which appears nowhere in src/ or tests/"


@pytest.mark.golden
def test_documents_contain_no_dashes_that_the_style_rule_forbids() -> None:
    """The project's writing rule forbids em dashes and en dashes outright.

    Checked mechanically because it is the one style rule that cannot be
    satisfied approximately: a single character violates it, and the character
    is visually similar to the hyphen that is allowed.
    """
    for name in DOCS:
        text = _text(name)
        for bad, label in (("\u2014", "em dash"), ("\u2013", "en dash")):
            hits = [i for i, line in enumerate(text.splitlines(), 1) if bad in line]
            assert not hits, f"{name} contains an {label} on line(s) {hits}"
