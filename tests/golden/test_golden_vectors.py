"""Parametrised runner for the golden and regression vector suite.

Each file in data/golden/ is a standalone auditable document: inputs, the full
arithmetic chain, expected outputs, and a per-output tolerance with a stated
reason. This module loads each file, dispatches its inputs to the matching
function in checks.py, and compares every expected output against what the
platform returns.

What this module does NOT do is derive any expected value. Every number in
data/golden/ was computed by hand or by a script that imports nothing from ae,
except where a file declares itself kind: regression and says why. That
separation is the point of the suite: a snapshot of current behaviour cannot
detect a wrong formula, because the snapshot changes with the formula.

Four structural properties are enforced here, each by its own test, because a
loader that silently skipped a malformed vector would be worse than no loader:

1. Every expected output key must be present in the dispatch result. A vector
   naming an output the check function does not return fails on the missing key
   rather than passing vacuously.
2. Every expected output must carry a tolerance, and every tolerance must carry
   a non-trivial reason. A tolerance with no stated reason is a fudge factor.
3. Every file must declare kind as golden or regression, and a regression file
   must state a regression_reason. The two kinds are counted separately.
4. A golden file must declare derivation as independent. A file whose expected
   values came from running the code cannot be golden, whatever it is called.
"""

from __future__ import annotations

import math
import os
import pathlib
import re
import subprocess
from typing import Any

import checks
import pytest
import yaml

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "golden"

_VECTOR_FILES = sorted(DATA.glob("*.yaml"))


def _load(path: pathlib.Path) -> dict[str, Any]:
    with path.open() as fh:
        loaded: dict[str, Any] = yaml.safe_load(fh)
    return loaded


_VECTORS = [(p.name, _load(p)) for p in _VECTOR_FILES]
_IDS = [name for name, _ in _VECTORS]


def _tolerance_of(spec: object) -> tuple[str, float]:
    """Return (mode, value) for a tolerance spec, which is a mapping.

    Mode is 'abs' or 'rel'. A spec carrying both is rejected: a tolerance that
    is whichever of two bounds happens to pass is not a tolerance.

    The bound must already be a number. It is not coerced here, because
    coercion is what hid a real defect during construction: repr(1e-09) is
    '1e-09', which YAML 1.1 parses as a STRING rather than a float because it
    carries no decimal point in the mantissa, and 174 of the tolerances in this
    suite were silently strings until a float() call in this function was
    removed and the types checked instead.
    """
    assert isinstance(spec, dict), f"tolerance must be a mapping, got {type(spec)}"
    has_abs = "abs" in spec
    has_rel = "rel" in spec
    assert has_abs != has_rel, f"tolerance must give exactly one of abs or rel: {spec}"
    mode = "abs" if has_abs else "rel"
    bound = spec[mode]
    assert isinstance(bound, (int, float)) and not isinstance(bound, bool), (
        f"tolerance bound must be a number, got {bound!r} of type "
        f"{type(bound).__name__}. Write 1.0e-9, not 1e-9: YAML 1.1 needs a "
        f"decimal point in the mantissa or the resolver reads it as a string."
    )
    return mode, float(bound)


def _within(actual: float, expected: float, mode: str, tol: float) -> tuple[bool, float]:
    if mode == "abs":
        err = abs(actual - expected)
        return err <= tol, err
    denom = abs(expected)
    if denom == 0.0:
        err = abs(actual)
        return err <= tol, err
    err = abs(actual - expected) / denom
    return err <= tol, err


@pytest.mark.golden
@pytest.mark.parametrize("name,vec", _VECTORS, ids=_IDS)
def test_vector_outputs_match_expected(name: str, vec: dict[str, Any]) -> None:
    """Every expected output of every vector agrees within its stated tolerance.

    This is the whole suite in one test. The expected values live in the data
    files, were derived independently of the code for every file declaring
    derivation: independent, and each carries a tolerance whose reason is
    recorded beside it.
    """
    result = checks.run(vec["check"], vec["inputs"])
    expected = vec["expected"]
    tolerances = vec["tolerances"]
    failures = []
    for key, exp in expected.items():
        assert key in result, (
            f"{name}: vector expects output {key!r} but the check function returned "
            f"only {sorted(result)}. A vector naming an output the code does not "
            f"produce must fail here, not pass silently."
        )
        mode, tol = _tolerance_of(tolerances[key])
        actual = float(result[key])
        ok, err = _within(actual, float(exp), mode, tol)
        if not ok:
            failures.append(
                f"{key}: expected {exp!r}, got {actual!r}, {mode} error {err!r} "
                f"exceeds {tol!r}"
            )
    assert not failures, f"{name}: " + "; ".join(failures)


@pytest.mark.golden
@pytest.mark.parametrize("name,vec", _VECTORS, ids=_IDS)
def test_every_expected_output_has_a_reasoned_tolerance(name: str, vec: dict[str, Any]) -> None:
    """No expected output may lack a tolerance, and none may lack a reason.

    The reason must be at least 40 characters, because a bare "floating point"
    does not say which bound was chosen or why one order tighter was rejected.
    40 is a floor on effort, not a measure of quality.
    """
    expected = vec["expected"]
    tolerances = vec["tolerances"]
    missing = sorted(set(expected) - set(tolerances))
    assert not missing, f"{name}: outputs with no tolerance: {missing}"
    extra = sorted(set(tolerances) - set(expected))
    assert not extra, f"{name}: tolerances for outputs that do not exist: {extra}"
    for key, spec in tolerances.items():
        _tolerance_of(spec)
        reason = spec.get("reason", "")
        assert isinstance(reason, str) and len(reason.strip()) >= 40, (
            f"{name}: tolerance for {key!r} has no stated reason (got "
            f"{reason!r}). A tolerance with no stated reason is a fudge factor."
        )


@pytest.mark.golden
@pytest.mark.parametrize("name,vec", _VECTORS, ids=_IDS)
def test_vector_declares_its_kind_and_provenance(name: str, vec: dict[str, Any]) -> None:
    """A vector must declare kind, derivation, an arithmetic chain and provenance.

    A regression vector must additionally state why its expected values could
    only come from the code, and must not claim derivation: independent.
    """
    assert vec["kind"] in ("golden", "regression"), f"{name}: bad kind {vec.get('kind')!r}"
    assert vec["derivation"] in ("independent", "from_code", "mixed"), (
        f"{name}: bad derivation {vec.get('derivation')!r}"
    )
    assert len(vec["arithmetic"].strip()) >= 200, (
        f"{name}: the arithmetic block is {len(vec.get('arithmetic', ''))} characters. "
        f"A reader with a calculator must be able to reproduce the outputs from it."
    )
    assert isinstance(vec["provenance"], dict) and vec["provenance"], (
        f"{name}: every vector must tag its inputs MEASURED / SOURCED / COMPUTED / "
        f"DERIVED / ASSUMED in a provenance block."
    )
    if vec["kind"] == "golden":
        assert vec["derivation"] == "independent", (
            f"{name}: a golden vector must be derived independently of the code. "
            f"This one declares derivation {vec['derivation']!r}, so it is a "
            f"regression vector whatever the filename says."
        )
        assert "regression_reason" not in vec
    else:
        assert len(vec.get("regression_reason", "").strip()) >= 80, (
            f"{name}: a regression vector must state why its expected values "
            f"could only come from running the code."
        )


@pytest.mark.golden
@pytest.mark.parametrize("name,vec", _VECTORS, ids=_IDS)
def test_every_tolerance_bound_is_a_number_and_not_a_yaml_string(
    name: str, vec: dict[str, Any]
) -> None:
    """Every tolerance bound parses as a float, not as a string.

    This guard exists because of a defect in this suite's own construction: the
    bounds were written with repr(), which emits an exponent form with no
    decimal point in the mantissa, and PyYAML's float resolver requires one, so
    every exponent-form bound in the suite parsed as a string. A float() coercion in
    _tolerance_of made every test pass anyway, which is exactly the shape of a
    guard that looks like it works. The coercion was removed and this test
    added; the companion test below counts the exponent-form bounds in the
    suite, which is how many were affected.
    """
    for key, spec in vec["tolerances"].items():
        mode, bound = _tolerance_of(spec)
        assert isinstance(vec["tolerances"][key][mode], (int, float)), key
        assert bound >= 0.0, f"{name}: negative tolerance on {key}"


@pytest.mark.golden
def test_the_suite_writes_174_exponent_form_bounds_all_parsing_as_floats() -> None:
    """Of the 224 tolerance bounds in the suite, 174 are in exponent form.

    That count is the blast radius of the mantissa-form defect described
    above, so it is measured rather than asserted from memory. Every one of the
    174 must parse as a float, which is what the per-vector test checks; this
    test pins how many there are, so a future bound written as 1e-9 rather than
    1.0e-9 changes a count that is under assertion.
    """
    exponent_form = 0
    total = 0
    for _, vec in _VECTORS:
        for spec in vec["tolerances"].values():
            total += 1
            mode, bound = _tolerance_of(spec)
            raw = spec[mode]
            assert isinstance(raw, (int, float))
            if bound != 0.0 and "e" in repr(float(bound)).lower():
                exponent_form += 1
    assert total == 224
    assert exponent_form == 174


@pytest.mark.golden
@pytest.mark.parametrize("name,vec", _VECTORS, ids=_IDS)
def test_no_tolerance_reason_defers_to_another_output(
    name: str, vec: dict[str, Any]
) -> None:
    """No reason may point at another output instead of stating its own basis.

    This guard exists because of a defect in this suite's own construction. Many
    reasons were first written as bare cross-references of the form "As <other
    key>", and a script then expanded them mechanically by prepending the
    referenced output's text. The expansion produced reasons that describe the
    WRONG output: the median and ninetieth percentile in the seeded Monte Carlo
    vector acquired the sentence about the sample mean, and the bootstrap
    standard error at the median acquired the tenth-percentile standard error,
    contradicting that same file's own analytic reference block. The two
    figures are asserted against each other in the companion test below. Every
    affected reason was rewritten by hand against its own expected value. The
    loader now rejects the cross-reference form outright, because a reason a
    reader must follow to a second row is not a reason, and an automatic
    expansion of one is worse than the cross-reference it replaced.
    """
    offenders = []
    for key, spec in vec["tolerances"].items():
        reason = " ".join(str(spec["reason"]).split())
        if (
            reason.startswith("As ")
            or "Same basis as" in reason
            or "Additionally for this output" in reason
        ):
            offenders.append(key)
    assert not offenders, (
        f"{name}: tolerance reasons deferring to another output: {offenders}. "
        f"State the basis for THIS output, including its own magnitude."
    )


@pytest.mark.golden
@pytest.mark.parametrize("name,vec", _VECTORS, ids=_IDS)
def test_no_title_claims_a_mechanism_its_own_body_denies(
    name: str, vec: dict[str, Any]
) -> None:
    """A vector's title may not assert a mechanism its own text rules out.

    This guard exists because the EVPI regression vector's title read "showing
    the nested-Monte-Carlo bias" while its provenance said that bias "does NOT
    arise here" and its arithmetic concluded the whole deviation is outer-loop
    sampling error. The title had been carried over from a pre-rewrite version
    whose body did claim the bias; the rewrite corrected the body and left the
    label. A reviewer found it. Titles are what a reader sees first and what
    the generated document uses as its heading and index entry, so a title
    contradicting its own derivation misleads before any number is read.

    The check is narrow by design: for each phrase a vector's body explicitly
    negates, the title must not assert it. A general contradiction detector is
    not attempted, because a guard that claims more than it checks is the
    failure mode this suite exists to prevent.
    """
    title = str(vec["title"]).lower()
    body = " ".join(
        [
            str(vec.get("regression_reason", "")),
            str(vec["arithmetic"]),
            " ".join(str(t) for t in vec["provenance"].values()),
        ]
    ).lower()
    denied = [
        ("nesting bias", ("no upward nesting bias", "bias does not arise")),
        ("nested-monte-carlo bias", ("no upward nesting bias", "does not arise here")),
        ("upward bias", ("no upward nesting bias", "does not arise here")),
    ]
    for phrase, denials in denied:
        if phrase not in title:
            continue
        hit = [d for d in denials if d in body]
        assert not hit, (
            f"{name}: the title claims {phrase!r} while the body states {hit[0]!r}. "
            f"Correct the title or the body, whichever is wrong."
        )


@pytest.mark.golden
def test_the_document_states_the_measured_residual_reason_properties() -> None:
    """docs/golden-vectors.md's claims about the reason texts are re-derived.

    The document's defect record states two properties of the committed reason
    texts: that none defers to another output, and that the shortest is 41
    characters. Both are recomputed here from the files rather than trusted,
    because this entry exists precisely because a count in that record was
    written from memory and was wrong twice, once as a figure that was never
    measured at all and once as an audit that omitted the reasons beginning
    with the deferral prefix rather than containing it. The extent of the
    original defect is a fact about a diff between two committed revisions, so
    it is checked by the companion test below, which reads those revisions.
    What this test checks is the end state, which is what a reader of the
    document verifies.
    """
    doc = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "docs"
        / "golden-vectors.md"
    )
    text = doc.read_text()
    shortest = min(
        len(" ".join(str(spec["reason"]).split()))
        for _, vec in _VECTORS
        for spec in vec["tolerances"].values()
    )
    assert shortest == 41, f"shortest reason is {shortest} characters"
    assert f"is now {shortest} characters" in text, (
        "the document states a shortest-reason length that the files contradict"
    )
    deferring = [
        f"{name}:{key}"
        for name, vec in _VECTORS
        for key, spec in vec["tolerances"].items()
        if " ".join(str(spec["reason"]).split()).startswith("As ")
    ]
    assert not deferring, deferring
    total_outputs = sum(len(vec["expected"]) for _, vec in _VECTORS)
    assert total_outputs == 224
    assert f"{total_outputs} checked outputs" in text, (
        "the document states an output count the files contradict"
    )


@pytest.mark.golden
def test_the_control_table_has_one_row_per_injection_the_script_performs() -> None:
    """The document's control table lists exactly as many defects as are injected.

    Stated because the paragraph under that table used to describe its rows by
    POSITION ("the last two rows"), and inserting rows above it silently made
    the description point at the wrong guards. Both paragraphs now name the
    guards, and the row count is checked against the injection count parsed out
    of control.py, so a defect added to the script without a table row, or a
    row added without an injection, fails here.
    """
    root = pathlib.Path(__file__).resolve().parent.parent.parent
    doc = (root / "docs" / "golden-vectors.md").read_text()
    start = doc.index("| defect injected | guard that reported it |")
    table = doc[start : doc.index("\n\n", start)]
    rows = [
        line
        for line in table.split("\n")
        if line.startswith("| ") and "| ---" not in line
    ]
    data_rows = len(rows) - 1
    # Count the distinct defect TAGS the script reports, not its comments: two
    # of the document-count injections share a loop and therefore one comment,
    # which a comment count misses. The tags are what the operator reads.
    script = (root / "tests" / "golden" / "control.py").read_text()
    tags = sorted(set(re.findall(r'"(D\d+) [^"]+"', script)))
    assert data_rows == len(tags), (
        f"the control table lists {data_rows} defects but control.py reports "
        f"{len(tags)} tags {tags}. Every injection needs a row and every row an "
        f"injection."
    )
    assert data_rows >= 17, f"only {data_rows} control rows"


@pytest.mark.golden
def test_the_document_s_stated_defect_extent_matches_the_committed_diff() -> None:
    """The document says 42 reasons across 17 files carried borrowed text.

    Both figures are re-derived from the two committed revisions of
    data/golden/ rather than recalled, because this is the entry where a
    recalled count was wrong twice. The repair commit is located by its subject
    line, its parent supplies the before state, and the reasons are compared
    text by text. Skipped, not failed, when the git history is unavailable (a
    shallow clone or an export), because the end-state properties are checked
    by the companion test above and do not depend on history.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    subject = "Rewrite 37 tolerance reasons that described the wrong output"
    try:
        sha = subprocess.run(
            ["git", "log", "--format=%H", "-1", "--grep", subject],
            cwd=repo, capture_output=True, text=True, check=True,
            env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
                 "GIT_TERMINAL_PROMPT": "0", "PATH": os.environ.get("PATH", "")},
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        pytest.skip("git history unavailable")
    if not sha:  # pragma: no cover
        pytest.skip("the repair commit is not in this history")

    def read(rev: str, name: str) -> dict[str, Any]:
        blob = subprocess.run(
            ["git", "show", f"{rev}:data/golden/{name}"],
            cwd=repo, capture_output=True, text=True, check=True,
            env={"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
                 "GIT_TERMINAL_PROMPT": "0", "PATH": os.environ.get("PATH", "")},
        ).stdout
        loaded: dict[str, Any] = yaml.safe_load(blob)
        return loaded

    def flat(text: object) -> str:
        return " ".join(str(text).split())

    borrowed = 0
    files = set()
    for name, _ in _VECTORS:
        before = read(f"{sha}^", name)
        after = read(sha, name)
        for key, spec in before["tolerances"].items():
            rb = flat(spec["reason"])
            deferred = (
                rb.startswith("As ")
                or "Same basis as" in rb
                or "Additionally for this output" in rb
            )
            if deferred:
                borrowed += 1
                files.add(name)
                assert flat(after["tolerances"][key]["reason"]) != rb, (
                    f"{name}:{key} carried borrowed text and was not rewritten"
                )
    assert borrowed == 42, f"measured {borrowed} borrowed reasons, document says 42"
    assert len(files) == 17, f"measured {len(files)} files, document says 17"
    doc = (repo / "docs" / "golden-vectors.md").read_text()
    assert f"{borrowed} reasons across {len(files)} of" in doc, (
        "the document's stated defect extent does not match the committed diff"
    )


@pytest.mark.golden
def test_the_two_bootstrap_standard_errors_are_different_numbers() -> None:
    """The p10 and p50 order-statistic standard errors are not interchangeable.

    Stated because a mechanical expansion put one under the other's output in
    RV-01, so the file contradicted its own analytic reference block. The
    median's error is larger by the square root of the ratio of the p(1-p)
    terms: 0.25 / 0.09 = 2.7777777777777777, root 1.6666666666666667, and
    0.1341640786499874 times that is 0.223606797749979.
    """
    rv01 = dict(_VECTORS)["RV-01-monte-carlo-percentiles.yaml"]
    ref = rv01["analytic_reference"]
    se10 = float(ref["order_statistic_se_p10"])
    se50 = float(ref["order_statistic_se_p50"])
    assert se10 != se50, "the two standard errors must not be the same number"
    assert se10 == pytest.approx(0.1341640786499874, abs=1.0e-9)
    assert se50 == pytest.approx(0.223606797749979, abs=1.0e-9)
    assert 0.25 / 0.09 == pytest.approx(2.7777777777777777, abs=1.0e-12)
    assert math.sqrt(0.25 / 0.09) == pytest.approx(1.6666666666666667, abs=1.0e-12)
    assert se10 * math.sqrt(0.25 / 0.09) == pytest.approx(se50, rel=1.0e-9)
    p50_reason = " ".join(str(rv01["tolerances"]["bootstrap_se_p50"]["reason"]).split())
    assert "0.2236067977" in p50_reason, (
        "the p50 bootstrap reason must name its OWN analytic standard error"
    )
    assert "0.1341640786" not in p50_reason, (
        "the p50 bootstrap reason must not name the p10 standard error"
    )


@pytest.mark.golden
def test_no_two_outputs_in_a_file_share_a_verbatim_reason_unless_identical() -> None:
    """Outputs sharing a reason verbatim must have identical expected values.

    Two outputs may legitimately share wording when they are the same kind of
    quantity at the same magnitude (two booleans, two equal integer counts).
    They may not when their magnitudes differ, because the reason names a
    magnitude and would then be wrong for one of them. This is the residue of
    the same copy defect, checked rather than trusted.
    """
    offenders = []
    for name, vec in _VECTORS:
        by_reason: dict[str, list[str]] = {}
        for key, spec in vec["tolerances"].items():
            by_reason.setdefault(" ".join(str(spec["reason"]).split()), []).append(key)
        for keys in by_reason.values():
            if len(keys) < 2:
                continue
            values = {float(vec["expected"][k]) for k in keys}
            if len(values) > 1:
                offenders.append(f"{name}:{keys} values {sorted(values)}")
    assert not offenders, (
        "outputs sharing a verbatim tolerance reason while holding different "
        f"expected values: {offenders}"
    )


@pytest.mark.golden
def test_every_provenance_tag_is_one_of_the_five() -> None:
    """Every provenance entry names one of the five tags.

    The tags are MEASURED, SOURCED, COMPUTED, DERIVED and ASSUMED. An entry
    naming none of them is an untagged parameter, which is the condition this
    check exists to prevent.
    """
    tags = ("MEASURED", "SOURCED", "COMPUTED", "DERIVED", "ASSUMED", "NOT SOURCED")
    untagged = []
    for name, vec in _VECTORS:
        for key, text in vec["provenance"].items():
            if not any(t in str(text) for t in tags):
                untagged.append(f"{name}:{key}")
    assert not untagged, f"provenance entries with no tag: {untagged}"


@pytest.mark.golden
def test_the_suite_has_the_declared_shape() -> None:
    """The suite holds 30 vectors: 27 golden and 3 regression, 27 independent.

    The counts are asserted rather than reported so that a vector added without
    a kind, or silently reclassified, breaks this test. 27 of the 30 declare
    derivation: independent; the three that do not are exactly the three
    regression vectors, two of which are mixed (RV-02 and RV-03 derive most of
    their values by hand) and one wholly from code (RV-01).
    """
    kinds = [vec["kind"] for _, vec in _VECTORS]
    derivations = [vec["derivation"] for _, vec in _VECTORS]
    assert len(_VECTORS) == 30
    assert kinds.count("golden") == 27
    assert kinds.count("regression") == 3
    assert derivations.count("independent") == 27
    assert derivations.count("mixed") == 2
    assert derivations.count("from_code") == 1


@pytest.mark.golden
def test_a_vector_naming_a_missing_output_fails_rather_than_passing() -> None:
    """A vector that expects an output the check does not return must fail.

    VERIFY BY CONTROL. This is the guard on the loader itself: a loader that
    iterated over the dispatch result rather than over the expected block would
    pass such a vector vacuously, which is the failure mode that makes a test
    suite worse than no suite. The defect is injected here rather than left to
    a reviewer's imagination.
    """
    good = dict(_load(DATA / "GV-01-bond-specific-energy.yaml"))
    broken = dict(good)
    broken["expected"] = dict(good["expected"])
    broken["expected"]["specific_energy_on_the_moon"] = 1.0
    broken["tolerances"] = dict(good["tolerances"])
    broken["tolerances"]["specific_energy_on_the_moon"] = {
        "abs": 1.0,
        "reason": "Injected defect for the control test, forty characters of reason.",
    }
    with pytest.raises(AssertionError, match="did not produce|does not produce|returned"):
        test_vector_outputs_match_expected("control", broken)
    test_vector_outputs_match_expected("GV-01-bond-specific-energy.yaml", good)


@pytest.mark.golden
def test_a_wrong_expected_value_fails_by_more_than_its_tolerance() -> None:
    """Perturbing an expected value by ten times its tolerance must fail.

    VERIFY BY CONTROL, second direction. This shows the tolerances are doing
    work: GV-01 expects 11.1759843719 kWh/short_ton at an absolute tolerance of
    1e-9, and adding 1e-8 to that expected value must be rejected.
    """
    good = dict(_load(DATA / "GV-01-bond-specific-energy.yaml"))
    perturbed = dict(good)
    perturbed["expected"] = dict(good["expected"])
    key = "specific_energy_kwh_per_short_ton"
    assert good["expected"][key] == pytest.approx(11.1759843719, abs=1e-10)
    assert _tolerance_of(good["tolerances"][key]) == ("abs", 1e-9)
    perturbed["expected"][key] = good["expected"][key] + 1.0e-8
    with pytest.raises(AssertionError, match="exceeds"):
        test_vector_outputs_match_expected("control", perturbed)
    test_vector_outputs_match_expected("GV-01-bond-specific-energy.yaml", good)


@pytest.mark.golden
def test_a_tolerance_with_both_abs_and_rel_is_rejected() -> None:
    """A tolerance giving both abs and rel must be rejected, not silently used.

    VERIFY BY CONTROL, third direction. A spec carrying both bounds would pass
    whenever either passed, which is not a tolerance but a disjunction.
    """
    with pytest.raises(AssertionError, match="exactly one of abs or rel"):
        _tolerance_of({"abs": 1.0e-9, "rel": 1.0e-9, "reason": "x" * 40})
    with pytest.raises(AssertionError, match="exactly one of abs or rel"):
        _tolerance_of({"reason": "x" * 40})
    assert _tolerance_of({"abs": 1.0e-9, "reason": "x" * 40}) == ("abs", 1.0e-9)
    assert _tolerance_of({"rel": 1.0e-6, "reason": "x" * 40}) == ("rel", 1.0e-6)


@pytest.mark.golden
def test_relative_tolerance_at_zero_expected_falls_back_to_absolute() -> None:
    """A relative tolerance against an expected zero compares the raw magnitude.

    Dividing by zero would give inf or nan and pass everything, so the helper
    treats a zero expected value as an absolute comparison. Checked because the
    fallback is the kind of branch that is written once and never exercised.
    """
    ok, err = _within(0.0, 0.0, "rel", 1.0e-9)
    assert ok and err == 0.0
    ok, err = _within(1.0e-12, 0.0, "rel", 1.0e-9)
    assert ok and err == 1.0e-12
    ok, err = _within(1.0e-6, 0.0, "rel", 1.0e-9)
    assert not ok and err == 1.0e-6


@pytest.mark.golden
def test_rv01_seeded_percentiles_sit_within_the_order_statistic_error() -> None:
    """RV-01's seeded percentiles lie within 3 order-statistic standard errors.

    The regression pin fixes the bit stream; this test checks the ESTIMATOR.
    Analytic P10 is 110.0 with se sqrt(0.10*0.90/50000)/0.01 = 0.1341640786, and
    the measured 109.7887651789 sits 1.574 se low. Analytic P50 is 150.0 with se
    0.2236067977 and the measured 150.0565756103 sits 0.253 se high. Analytic
    P90 is 190.0 with se 0.1341640786 and the measured 189.7903414569 sits
    1.563 se low.
    """
    vec = _load(DATA / "RV-01-monte-carlo-percentiles.yaml")
    ref = vec["analytic_reference"]
    exp = vec["expected"]
    for p, se_key in (("p10", "order_statistic_se_p10"),
                      ("p50", "order_statistic_se_p50"),
                      ("p90", "order_statistic_se_p90")):
        se = float(ref[se_key])
        dev = abs(float(exp[p]) - float(ref[p])) / se
        assert dev <= 3.0, f"{p} sits {dev} standard errors from analytic"
    assert float(exp["p10"]) == pytest.approx(109.7887651789, abs=1e-10)
    assert float(exp["p50"]) == pytest.approx(150.0565756103, abs=1e-10)
    assert float(exp["p90"]) == pytest.approx(189.7903414569, abs=1e-10)
    assert abs(float(exp["p10"]) - 110.0) / 0.1341640786 == pytest.approx(1.574, abs=0.005)
    assert abs(float(exp["p50"]) - 150.0) / 0.2236067977 == pytest.approx(0.253, abs=0.005)
    assert abs(float(exp["p90"]) - 190.0) / 0.1341640786 == pytest.approx(1.563, abs=0.005)
    assert float(ref["order_statistic_se_p10"]) == pytest.approx(
        math.sqrt(0.10 * 0.90 / 50000) / 0.01, rel=1e-9
    )
    assert float(ref["order_statistic_se_p50"]) == pytest.approx(
        math.sqrt(0.25 / 50000) / 0.01, rel=1e-9
    )


@pytest.mark.golden
def test_rv01_bootstrap_standard_error_agrees_with_the_closed_form() -> None:
    """The module's bootstrap standard errors match the closed form within 6 percent.

    Measured ratios: 0.1325916214/0.1341640786 = 0.988 at p10 and
    0.2366398887/0.2236067977 = 1.058 at p50. Both within 6 percent, which is
    what a 400-resample bootstrap delivers; asserting a tighter band would be
    asserting a precision the bootstrap does not have.
    """
    vec = _load(DATA / "RV-01-monte-carlo-percentiles.yaml")
    exp, ref = vec["expected"], vec["analytic_reference"]
    assert vec["inputs"]["n_boot"] == 400
    assert float(exp["bootstrap_se_p10"]) == pytest.approx(0.1325916214, abs=1e-10)
    assert float(exp["bootstrap_se_p50"]) == pytest.approx(0.2366398887, abs=1e-10)
    assert float(ref["order_statistic_se_p10"]) == pytest.approx(0.1341640786, abs=1e-10)
    assert float(ref["order_statistic_se_p50"]) == pytest.approx(0.2236067977, abs=1e-10)
    r10 = float(exp["bootstrap_se_p10"]) / float(ref["order_statistic_se_p10"])
    r50 = float(exp["bootstrap_se_p50"]) / float(ref["order_statistic_se_p50"])
    assert r10 == pytest.approx(0.988, abs=0.002)
    assert r50 == pytest.approx(1.058, abs=0.002)
    assert 0.94 <= r10 <= 1.06
    assert 0.94 <= r50 <= 1.06


@pytest.mark.golden
def test_rv03_evpi_estimate_sits_within_three_paired_standard_errors_of_one_sixteenth() -> None:
    """RV-03's seeded EVPI lies 0.796 paired standard errors from the exact 1/16.

    The closed form is 0.5625 - 0.5 = 0.0625. The module pairs the baseline with
    the resolved value on common random numbers, so at these sample sizes the
    estimator reduces to one sample mean of max(0, -theta), whose per-draw
    variance is 1/48 - (1/16)^2 = 13/768 = 0.016927083333333332, giving
    sd 0.1301041249666333 and a standard error at 2048 draws of
    0.0028749221570077625. The measured 0.064788758648048 deviates by
    0.002288758648048006, which is 0.796 of that error. The switch fraction
    0.2548828125 deviates from the exact 0.25 by 0.0048828125, which is 0.510 of
    the binomial standard error sqrt(0.25*0.75/2048) = 0.0095683193.

    The earlier form of this test used a combined standard error of
    0.014196590161039404 built from two INDEPENDENT samples, which was the right
    model for the unpaired estimator this vector was first pinned against and is
    the wrong model now. The paired bound is 4.94 times tighter, so keeping the
    old one would have admitted a fivefold error.
    """
    vec = _load(DATA / "RV-03-evpi.yaml")
    exp, ref = vec["expected"], vec["analytic_reference"]
    assert float(ref["resolved_value"]) == pytest.approx(0.5625, abs=1e-12)
    assert float(exp["evpi"]) == pytest.approx(0.064788758648048, abs=1e-15)
    assert float(exp["switch_fraction"]) == pytest.approx(0.2548828125, abs=1e-12)
    # The paired per-draw quantity max(0,-theta) and its exact moments.
    e_pos = 0.5 * (0.5**2) / 2
    assert e_pos == pytest.approx(0.0625, abs=1e-15)
    e_pos_sq = 0.5 * (0.5**3) / 3
    assert e_pos_sq == pytest.approx(1.0 / 48.0, abs=1e-15)
    var = e_pos_sq - e_pos**2
    assert var == pytest.approx(13.0 / 768.0, abs=1e-15)
    assert var == pytest.approx(0.016927083333333332, abs=1e-15)
    assert float(ref["paired_per_draw_variance"]) == pytest.approx(var, abs=1e-15)
    sd = math.sqrt(var)
    assert sd == pytest.approx(0.1301041249666333, abs=1e-12)
    assert float(ref["paired_per_draw_sd"]) == pytest.approx(sd, abs=1e-12)
    paired_se = sd / math.sqrt(2048)
    assert paired_se == pytest.approx(0.0028749221570077625, abs=1e-15)
    assert float(ref["paired_standard_error"]) == pytest.approx(paired_se, abs=1e-15)
    dev = float(exp["evpi"]) - float(ref["evpi"])
    assert dev == pytest.approx(0.002288758648048006, abs=1e-15)
    assert abs(dev) / paired_se == pytest.approx(0.796, abs=0.001)
    assert abs(dev) <= 3.0 * paired_se
    # The superseded unpaired model, kept so the tightening is under assertion.
    unpaired = math.sqrt(0.24609375 / 2048 + (4.0 / 12.0) / 4096)
    assert unpaired == pytest.approx(0.014196590161039404, abs=1e-15)
    assert float(ref["superseded_unpaired_combined_standard_error"]) == pytest.approx(
        unpaired, abs=1e-15
    )
    assert unpaired / paired_se == pytest.approx(4.9380781063704715, rel=1e-9)
    assert round(unpaired / paired_se, 2) == 4.94
    binom_se = math.sqrt(0.25 * 0.75 / 2048)
    assert binom_se == pytest.approx(0.0095683193, abs=1e-9)
    sw_dev = float(exp["switch_fraction"]) - float(ref["switch_fraction"])
    assert sw_dev == pytest.approx(0.0048828125, abs=1e-12)
    assert abs(sw_dev) / binom_se == pytest.approx(0.510, abs=0.002)


@pytest.mark.golden
def test_rv03_inner_sample_size_does_not_change_the_estimate_in_this_problem() -> None:
    """Raising n_inner from 256 to 2048 changes the EVPI by 2.22e-16, not more.

    The nested Monte Carlo estimator is biased upward when the outer maximum is
    taken over noisy inner estimates. In THIS problem there is no residual
    uncertainty once theta is resolved, so the inner loop averages a constant
    and no bias arises. Measured: 0.064788758648048 at n_inner 256 against
    0.06478875864804778 at n_inner 2048, a difference of 2.220446049250313e-16,
    which is 2 ulp at this magnitude. The claim is checked rather than asserted
    because it is the reason the vector's deviation can be attributed wholly to
    outer-loop sampling error.
    """
    vec = _load(DATA / "RV-03-evpi.yaml")
    inputs = dict(vec["inputs"])
    at_256 = checks.run("evpi", inputs)["evpi"]
    inputs_2048 = dict(inputs)
    inputs_2048["n_inner"] = 2048
    at_2048 = checks.run("evpi", inputs_2048)["evpi"]
    assert at_256 == pytest.approx(0.064788758648048, abs=1e-12)
    assert at_2048 == pytest.approx(0.06478875864804778, abs=1e-12)
    assert abs(at_256 - at_2048) == pytest.approx(2.220446049250313e-16, abs=1e-18)
    assert abs(at_256 - at_2048) < 1e-14


@pytest.mark.golden
def test_gv27_sobol_indices_match_the_exact_variance_decomposition() -> None:
    """The Sobol estimate reproduces 1/14, 4/14 and 9/14 to within 0.005.

    Exact for y = x1 + 2 x2 + 3 x3 with xi uniform on (0,1): the variances are
    1/12, 4/12 and 9/12, total 14/12 = 1.1666666667, so the first-order indices
    are 0.0714285714, 0.2857142857 and 0.6428571429 and they sum to exactly 1
    because the model has no interactions. The measured maximum deviation of any
    index from its analytic value across five seeds at n_base 4096 is what sets
    the 0.005 tolerance in the vector file, rather than any floating point
    bound. That worst-case figure belongs to seed 4, not to seed 0, and is
    measured and asserted in the companion test below.
    """
    vec = _load(DATA / "GV-27-sobol-additive.yaml")
    assert vec["inputs"]["n_base"] == 4096
    result = checks.run("sobol_additive", vec["inputs"])
    exact = {"x1": 1.0 / 14.0, "x2": 4.0 / 14.0, "x3": 9.0 / 14.0}
    assert (14.0 / 12.0) == pytest.approx(1.1666666667, abs=1e-10)
    assert exact["x1"] == pytest.approx(0.0714285714, abs=1e-9)
    assert exact["x2"] == pytest.approx(0.2857142857, abs=1e-9)
    assert exact["x3"] == pytest.approx(0.6428571429, abs=1e-9)
    assert sum(exact.values()) == pytest.approx(1.0, abs=1e-12)
    worst = 0.0
    for key, truth in exact.items():
        for order in ("first", "total"):
            worst = max(worst, abs(float(result[f"{order}_{key}"]) - truth))
    assert worst <= 0.005, f"worst index deviation {worst}"
    # Seed 0's own worst deviation is 1.8216682789187755e-05. The 0.000313
    # figure quoted in the vector file is the worst across seeds 0 to 4 and
    # belongs to seed 4 (0.000313270921173725); a first draft asserted it
    # against seed 0 and failed, which is recorded rather than quietly loosened.
    assert worst == pytest.approx(1.8216682789187755e-05, rel=1e-6)
    assert float(result["output_variance"]) == pytest.approx(14.0 / 12.0, abs=0.002)
    assert float(result["output_variance"]) == pytest.approx(1.1667327924321855, rel=1e-9)


@pytest.mark.golden
def test_gv27_worst_sobol_deviation_across_five_seeds_is_the_quoted_figure() -> None:
    """Across seeds 0 to 4 the worst index deviation is 0.000313270921173725.

    This is the measurement the 0.005 tolerance in GV-27 rests on, so it is
    asserted rather than left in prose. Per-seed worst deviations, measured in
    this session at n_base 4096: seed 0 1.8216682789187755e-05, seed 1
    3.070839859953267e-06, seed 2 4.6773696027457845e-12, seed 3
    7.663454326678476e-08, seed 4 0.000313270921173725. The tolerance is 16
    times the worst of those, and every seed passes it by at least an order.
    """
    vec = _load(DATA / "GV-27-sobol-additive.yaml")
    assert vec["inputs"]["n_base"] == 4096
    exact = {"x1": 1.0 / 14.0, "x2": 4.0 / 14.0, "x3": 9.0 / 14.0}
    per_seed = []
    for seed in range(5):
        inputs = dict(vec["inputs"])
        inputs["seed"] = seed
        result = checks.run("sobol_additive", inputs)
        per_seed.append(
            max(
                abs(float(result[f"{order}_{key}"]) - truth)
                for key, truth in exact.items()
                for order in ("first", "total")
            )
        )
    assert per_seed[0] == pytest.approx(1.8216682789187755e-05, rel=1e-6)
    assert per_seed[1] == pytest.approx(3.070839859953267e-06, rel=1e-6)
    assert per_seed[2] == pytest.approx(4.6773696027457845e-12, rel=1e-6)
    assert per_seed[3] == pytest.approx(7.663454326678476e-08, rel=1e-6)
    assert per_seed[4] == pytest.approx(0.000313270921173725, rel=1e-6)
    worst = max(per_seed)
    assert worst == pytest.approx(0.000313270921173725, rel=1e-6)
    tol = _tolerance_of(vec["tolerances"]["first_x1"])[1]
    assert tol == 0.005
    assert tol / worst == pytest.approx(15.960, abs=0.01)


@pytest.mark.golden
def test_rv02_mean_baseline_rmse_is_the_hand_derived_value() -> None:
    """The fold baselines are sqrt(226), 1.0 and sqrt(226), derived by hand.

    Fold A trains on y = [20,22,30,32] whose mean is 26.0 and tests on [10,12],
    giving errors of -16 and -14, MSE (256+196)/2 = 226.0 and RMSE
    sqrt(226) = 15.033296378372908. Fold B trains on [10,12,30,32], mean 21.0,
    tests on [20,22], errors -1 and +1, MSE 1.0, RMSE 1.0. Fold C mirrors A.
    Averaging the three gives 10.355530918915273, which describes none of them,
    which is why the per-fold spread is reported.
    """
    vec = _load(DATA / "RV-02-surrogate-lodo.yaml")
    exp = vec["expected"]
    y = vec["inputs"]["y"]
    assert sum(y[2:]) / 4.0 == 26.0
    assert (y[0] - 26.0) ** 2 == 256.0
    assert (y[1] - 26.0) ** 2 == 196.0
    assert sum(y[:4]) / 4.0 == 16.0
    assert (256.0 + 196.0) / 2.0 == 226.0
    assert y[2] == 20.0
    assert float(exp["global_mean"]) == 21.0
    assert float(exp["baseline_mean_rmse_A"]) == pytest.approx(math.sqrt(226.0), abs=1e-9)
    assert math.sqrt(226.0) == pytest.approx(15.033296378372908, abs=1e-12)
    assert float(exp["baseline_mean_rmse_B"]) == pytest.approx(1.0, abs=1e-12)
    assert float(exp["baseline_mean_rmse_C"]) == pytest.approx(math.sqrt(226.0), abs=1e-9)
    mean_of_three = (
        float(exp["baseline_mean_rmse_A"])
        + float(exp["baseline_mean_rmse_B"])
        + float(exp["baseline_mean_rmse_C"])
    ) / 3.0
    assert mean_of_three == pytest.approx(10.355530918915273, abs=1e-9)
    assert float(exp["global_mean"]) == pytest.approx(126.0 / 6.0, abs=1e-12)
    assert float(exp["target_iqr"]) == pytest.approx(28.0 - 14.0, abs=1e-12)
