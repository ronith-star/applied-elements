"""Guards on the memo's number-generating script, scripts/memo_run.py.

These do NOT re-run the Monte Carlo or the EVPI loops, which take minutes.
They guard the things that can silently corrupt the memo's traceability: the
fast value function diverging from the audited chain, and the CSV attaching a
provenance a cost does not have.
"""
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import memo_run  # noqa: E402

#: Parameters whose cost is a commercial study, not a laboratory service. No
#: fee schedule sells these, so a fee-schedule note against them would be a
#: fabricated provenance.
NON_LAB_PARAMS = ("price", "capex_musd", "power_price", "reagent_price",
                  "discount_rate")
#: Laboratories whose fee schedules were read in this session. A note naming
#: one of these is a claim that the number came off a price list.
LAB_NAMES = ("actlabs", "hazen")


def test_cost_provenance_covers_every_measurement_cost():
    """Every cost must carry its own tag, or the CSV falls back to a blanket."""
    assert set(memo_run.COST_PROVENANCE) == set(memo_run.MEASUREMENT_COSTS)


def test_non_laboratory_costs_are_assumed_and_cite_no_fee_schedule():
    """The load-bearing guard the auditor's finding was about.

    A cost the script declares non-laboratory must be tagged ASSUMED and must
    NOT name a laboratory in its note. Attaching 'Actlabs 2026 and Hazen July
    2026 fee schedules' to the price or capex cost tells a reader those
    figures came off a price list, and they did not.
    """
    for p in NON_LAB_PARAMS:
        tag, note = memo_run.COST_PROVENANCE[p]
        assert tag == "ASSUMED", f"{p} is not a laboratory service, got {tag}"
        low = note.lower()
        for lab in LAB_NAMES:
            assert lab not in low, (
                f"{p} is ASSUMED but its note names {lab}, which claims a fee "
                f"schedule provenance it does not have: {note!r}"
            )


def test_flotation_cost_is_assumed_not_derived():
    """A SOURCED dollar figure standing in for a different test is judgement.

    The 1,000 USD is a real Hazen July 2026 list price, but for a Bond ball
    mill grindability test, not a bench flotation mass-yield test. DERIVED
    would claim the number is an algebraic transform of a sourced value for
    the same quantity. The weaker tag governs.
    """
    tag, note = memo_run.COST_PROVENANCE["mass_yield_flot"]
    assert tag == "ASSUMED"
    assert "proxy" in note.lower()


def test_sourced_and_derived_costs_name_their_schedule():
    """The converse: a cost that IS off a price list must say which one."""
    for p, (tag, note) in memo_run.COST_PROVENANCE.items():
        if tag in ("SOURCED", "DERIVED"):
            assert any(lab in note.lower() for lab in LAB_NAMES), (
                f"{p} is {tag} but names no fee schedule: {note!r}"
            )


def test_al_mean_cost_equals_its_stated_components():
    """The arithmetic in the note must reproduce the number in the dict.

    Actlabs 2026: RX1 prep 12.40 USD, Code 4B2-Std 52.20 USD at the 11+
    sample price. Both read from the fee schedule PDF in this session.
    """
    assert memo_run.MEASUREMENT_COSTS["al_mean_ppm"] == pytest.approx(
        12.40 + 52.20)
    assert memo_run.MEASUREMENT_COSTS["al_sigma_ppm"] == pytest.approx(
        12.40 + 5 * 52.20)
    assert memo_run.MEASUREMENT_COSTS["mass_yield_leach"] == pytest.approx(
        6 * (10.0 + 155.0 + 20.0) + 30.0)


def test_sigma_costs_more_than_mean_because_it_needs_replicates():
    """Lot-to-lot variance is not measurable from one assay.

    This is the mechanism behind the memo's recommendation, so it is asserted
    rather than asserted in prose: a sigma estimate needs replicate lots, so
    it must cost strictly more than a single mean determination.
    """
    assert (memo_run.MEASUREMENT_COSTS["al_sigma_ppm"]
            > memo_run.MEASUREMENT_COSTS["al_mean_ppm"])


def test_fast_value_function_equals_the_audited_chain():
    """Control: npv_musd must equal plant_npv exactly, or the memo's EVPI and
    Sobol sections describe two different models.

    Raises inside verify_npv_matches_plant_npv on any disagreement, so
    reaching the assertion at all means every one of the 200 draws matched.
    """
    diffs = memo_run.verify_npv_matches_plant_npv()
    assert len(diffs) == 200
    assert max(diffs) == 0.0


def test_polish_action_cannot_remove_all_aluminium():
    """Lattice-bound Al is not removable by a gas-phase or surface treatment.

    POLISH_AL_FACTOR is ASSUMED, but it is bounded away from zero on physical
    grounds, and a later edit setting it to 0 would silently assert that a
    polishing stage can drive Al to zero.
    """
    assert 0.0 < memo_run.POLISH_AL_FACTOR < 1.0


def test_csv_written_by_the_script_carries_no_fabricated_provenance():
    """End-to-end on the CSV writer, on a minimal synthetic record.

    Guards the output file rather than the dict: the auditor's finding was in
    the writer, not in COST_PROVENANCE.
    """
    rec = _minimal_record()
    path = Path(__file__).parent / "_memo_csv_guard.csv"
    try:
        memo_run.write_csv(rec, path)
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
    finally:
        path.unlink(missing_ok=True)
    cost_rows = [r for r in rows if r["section"] == "measurement_costs_usd"]
    assert len(cost_rows) == len(memo_run.MEASUREMENT_COSTS)
    for r in cost_rows:
        if r["parameter"] in NON_LAB_PARAMS:
            assert r["tag"] == "ASSUMED"
            for lab in LAB_NAMES:
                assert lab not in r["note"].lower(), (
                    f"CSV row for {r['parameter']} claims {lab} provenance"
                )


def _minimal_record():
    """The smallest record write_csv accepts, for the CSV guard above."""
    ins = [{"name": "price", "low": 2800.0, "high": 5200.0,
            "kind": "triangular", "mode": 3500.0}]
    mc_summary = {"mean": 1.0, "sd": 1.0, "min": 0.0, "max": 2.0, "P10": 0.5,
                  "P50": 1.0, "P90": 1.5, "P90_minus_P10": 1.0}
    sob = {"n_base": 8, "n_evaluations": 88, "output_variance": 1.0,
           "first_order": {"price": 0.4}, "total_order": {"price": 0.45},
           "first_conf": {"price": 0.01}, "total_conf": {"price": 0.01},
           "interaction_share": {"price": 0.05}, "additive_fraction": 0.4,
           "ranking": [("price", 0.45)], "negligible": [],
           "diagnostics": {"sum_total_order": 1.0, "converged": True}}
    return {
        "value_function_control": {"n_checked": 200, "max_abs_diff_musd": 0.0},
        "inputs": ins,
        "mc": {"n_draws": 10, "n_failed": 0,
               "summary": {"npv_musd": mc_summary},
               "p_npv_below_zero": 0.1,
               "percentile_se": {"npv_P10": 0.2}},
        "sobol": {"npv_musd": sob},
        "sobol_convergence": {"levels": [8, 4, 2], "top_input": {"8": "price"},
                              "ranking_stable": True, "max_total_drift": 0.01,
                              "diagnostics": {"8": True},
                              "total_order_by_level": {"8": {"price": 0.45}}},
        "tornado": {"price": {"low": -1.0, "high": 2.0, "base": 0.5,
                              "swing": 3.0, "low_delta": -1.5,
                              "high_delta": 1.5}},
        "tornado_nominal": {"price": 3500.0},
        "decision": {"actions": ["walk_away", "build_base"],
                     "expected_values_musd": {"walk_away": 0.0,
                                              "build_base": 1.0},
                     "prior_best_action": "build_base",
                     "prior_best_value_musd": 1.0,
                     "assumed_polish": {"capex_musd": 9.0}},
        "evpi": [{"parameter": "price", "evpi_musd": 0.1,
                  "baseline_value_musd": 1.0, "resolved_value_musd": 1.1,
                  "switch_fraction": 0.2, "prior_best_action": "build_base",
                  "evpi_fraction": 0.1, "n_outer": 8, "n_inner": 8}],
        "evpi_per_cost": [{"parameter": p, "evpi": 0.1, "cost": c,
                           "evpi_per_cost": 0.1 / c, "switch_fraction": 0.2,
                           "prior_best_action": "build_base"}
                          for p, c in memo_run.MEASUREMENT_COSTS.items()],
        "measurement_costs_usd": memo_run.MEASUREMENT_COSTS,
        "evpi_convergence": {"parameter": "price", "series": [(8, 0.2)]},
    }
