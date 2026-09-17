"""VERIFY BY CONTROL for the golden vector loader.

Run from the repository root: python tests/golden/control.py


Injects each defect the loader is supposed to catch, one at a time, and
reports whether the loader failed. Then removes the defect and reports that
it passes. Both directions are printed for every defect, and the number of
defects reported is checked against the number injected, because a control
whose partial failure looks like success is worse than none.
"""
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
D = REPO / "data" / "golden"
sys.path.insert(0, str(REPO / "tests" / "golden"))
sys.path.insert(0, str(REPO / "src"))

import test_golden_vectors as T

GV1 = D / "GV-01-bond-specific-energy.yaml"
GV20 = D / "GV-20-capacity-oee.yaml"
GV27 = D / "GV-27-sobol-additive.yaml"
RV1 = D / "RV-01-monte-carlo-percentiles.yaml"


def main() -> int:
    """Run every injection, report both directions, return an exit code.

    Wrapped in a function and guarded below because the test suite runs with
    --doctest-modules, which imports every module in tests/. At module level
    this body would inject defects during collection.
    """

    def load(p: pathlib.Path) -> dict[str, Any]:
        loaded: dict[str, Any] = yaml.safe_load(p.open())
        return loaded

    def ran(fn: Callable[..., None], *a: Any) -> tuple[bool, str]:
        """Return (failed, message)."""
        try:
            fn(*a)
            return False, ""
        except AssertionError as e:
            return True, str(e).split("\n")[0][:160]

    results = []

    # D1: expected value perturbed by 10x its tolerance
    v = load(GV1); v["expected"]["specific_energy_kwh_per_short_ton"] += 1.0e-8
    results.append(("D1 wrong expected value, +1e-8 on a 1e-9 tolerance",
                    *ran(T.test_vector_outputs_match_expected, "D1", v)))

    # D2: vector expects an output the check does not return
    v = load(GV1); v["expected"]["energy_on_the_moon"] = 1.0
    v["tolerances"]["energy_on_the_moon"] = {"abs": 1.0, "reason": "x"*50}
    results.append(("D2 vector names an output the dispatch does not produce",
                    *ran(T.test_vector_outputs_match_expected, "D2", v)))

    # D3: a tolerance with no reason
    v = load(GV1); v["tolerances"]["specific_energy_kwh_per_short_ton"] = {"abs": 1e-9, "reason": "fp"}
    results.append(("D3 tolerance whose reason is 2 characters",
                    *ran(T.test_every_expected_output_has_a_reasoned_tolerance, "D3", v)))

    # D4: an expected output with no tolerance at all
    v = load(GV1); del v["tolerances"]["specific_energy_kwh_per_metric_tonne"]
    results.append(("D4 expected output with no tolerance entry",
                    *ran(T.test_every_expected_output_has_a_reasoned_tolerance, "D4", v)))

    # D5: a golden vector claiming derivation from_code
    v = load(GV1); v["derivation"] = "from_code"
    results.append(("D5 golden vector declaring derivation from_code",
                    *ran(T.test_vector_declares_its_kind_and_provenance, "D5", v)))

    # D6: a regression vector with no regression_reason
    v = load(RV1); v["regression_reason"] = "too short"
    results.append(("D6 regression vector with a 9-character reason",
                    *ran(T.test_vector_declares_its_kind_and_provenance, "D6", v)))

    # D7: an untagged provenance entry
    v = load(GV1); v["provenance"]["work_index"] = "a round mid-range figure"
    saved = T._VECTORS
    T._VECTORS = [("D7", v)]
    results.append(("D7 provenance entry naming none of the five tags",
                    *ran(T.test_every_provenance_tag_is_one_of_the_five)))
    T._VECTORS = saved

    # D8: an arithmetic block too short to reproduce by hand
    v = load(GV1); v["arithmetic"] = "W = 10 * Wi * (1/sqrt(P80) - 1/sqrt(F80))"
    results.append(("D8 arithmetic block of 41 characters",
                    *ran(T.test_vector_declares_its_kind_and_provenance, "D8", v)))

    # D9: a tolerance giving both abs and rel
    def d9() -> None:
        T._tolerance_of({"abs": 1e-9, "rel": 1e-9, "reason": "x"*50})
    results.append(("D9 tolerance giving both abs and rel",
                    *ran(d9)))

    # D10: bottleneck named wrongly (the missing-key path on a real vector)
    v = load(GV20)
    v["expected"]["bottleneck_is_kiln"] = v["expected"].pop("bottleneck_is_leach")
    v["tolerances"]["bottleneck_is_kiln"] = v["tolerances"].pop("bottleneck_is_leach")
    results.append(("D10 vector naming the wrong bottleneck",
                    *ran(T.test_vector_outputs_match_expected, "D10", v)))

    # D12: a tolerance reason that defers to another output instead of stating
    # its own basis. This is the defect a reviewer found in the committed files:
    # a mechanical expansion of "As <other key>" pasted the referenced output's
    # text under a different output, including a standard error that
    # contradicted the same file's own analytic_reference block.
    v = load(GV1)
    v["tolerances"]["specific_energy_kwh_per_short_ton"]["reason"] = (
        "As specific_energy_kwh_per_metric_tonne. The same floating point "
        "basis applies here, so nothing further is stated for this output."
    )
    results.append(("D12 tolerance reason defers to another output",
                    *ran(T.test_no_tolerance_reason_defers_to_another_output,
                         "D12", v)))

    # D13: two outputs at different magnitudes sharing a verbatim reason, which
    # is what the bad expansion produced: a reason naming the p10 standard
    # error 0.1342 sitting on an output whose value is 0.2366. This guard is
    # suite-level, so the injection replaces the loaded vector list.
    v = load(RV1)
    v["tolerances"]["bootstrap_se_p50"]["reason"] = (
        v["tolerances"]["bootstrap_se_p10"]["reason"]
    )
    saved = T._VECTORS
    T._VECTORS = [("RV-01-monte-carlo-percentiles.yaml", v)]
    results.append((
        "D13 differing outputs share a verbatim reason",
        *ran(T.test_no_two_outputs_in_a_file_share_a_verbatim_reason_unless_identical),
    ))
    T._VECTORS = saved

    # D14 and D15: the document states a property of the reason texts that the
    # files contradict. These exist because a count written into
    # docs/golden-vectors.md was wrong twice (thirty-seven, then 35, against a
    # measured 42), so a figure in the document is now itself under assertion.
    # The document is edited on disk and restored, so these two injections are
    # checked for byte-identical restoration rather than dict mutation.
    doc = REPO / "docs" / "golden-vectors.md"
    original = doc.read_text()
    for tag, before, after in (
        ("D14 document states a wrong shortest-reason length",
         "is now 41 characters", "is now 55 characters"),
        ("D15 document states a wrong checked-output count",
         "224 checked outputs", "300 checked outputs"),
    ):
        assert before in original, f"{tag}: anchor text not present in the document"
        doc.write_text(original.replace(before, after))
        try:
            results.append((
                tag,
                *ran(T.test_the_document_states_the_measured_residual_reason_properties),
            ))
        finally:
            doc.write_text(original)

    # D16: the document's stated extent of the original defect (42 reasons
    # across 17 files) contradicts what the committed diff actually shows. This
    # is the count that was wrong twice, so it is injected as well as asserted.
    assert "42 reasons across 17 of" in original, "D16: anchor text not present"
    doc.write_text(original.replace("42 reasons across 17 of", "50 reasons across 17 of"))
    try:
        results.append((
            "D16 document extent contradicts the committed diff",
            *ran(T.test_the_document_s_stated_defect_extent_matches_the_committed_diff),
        ))
    finally:
        doc.write_text(original)
    assert doc.read_text() == original, "the document was not restored byte-identically"

    # D11: suite shape claim contradicted
    v = load(GV27)
    saved = T._VECTORS
    T._VECTORS = [("a", v)]
    results.append(("D11 suite shape assertion against a 1-vector suite",
                    *ran(T.test_the_suite_has_the_declared_shape)))
    T._VECTORS = saved

    injected = len(results)
    caught = sum(1 for _, failed, _ in results if failed)
    print(f"defects injected: {injected}")
    print(f"defects reported by the guard: {caught}")
    for name, failed, msg in results:
        print(("CAUGHT   " if failed else "MISSED   ") + name)
        if failed:
            print("           -> " + msg)
    print()

    # Both directions: with every defect removed, the same tests pass.
    clean = []
    for p in (GV1, GV20, GV27, RV1):
        v = load(p)
        for fn in (T.test_vector_outputs_match_expected,
                   T.test_every_expected_output_has_a_reasoned_tolerance,
                   T.test_vector_declares_its_kind_and_provenance,
                   T.test_no_tolerance_reason_defers_to_another_output):
            failed, msg = ran(fn, p.name, v)
            clean.append((p.name, fn.__name__, failed, msg))
    for fn in (T.test_every_provenance_tag_is_one_of_the_five,
               T.test_the_suite_has_the_declared_shape,
               T.test_the_document_states_the_measured_residual_reason_properties,
               T.test_the_document_s_stated_defect_extent_matches_the_committed_diff,
               T.test_no_two_outputs_in_a_file_share_a_verbatim_reason_unless_identical):
        failed, msg = ran(fn)
        clean.append(("suite", fn.__name__, failed, msg))
    bad = [c for c in clean if c[2]]
    print(f"clean checks run: {len(clean)}, failures: {len(bad)}")
    for c in bad:
        print("UNEXPECTED FAILURE", c)
    print()
    print("VERDICT:", "all injected defects reported and clean state passes"
          if caught == injected and not bad else "INCOMPLETE")
    return 0 if caught == injected and not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
