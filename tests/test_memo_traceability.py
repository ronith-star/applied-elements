"""Guards on the memo's traceability claim, which was false once already.

docs/decision-memo.md asserts that every figure it quotes appears in
docs/memo_numbers.csv. That claim was FALSE when first written: the CSV was
produced by scripts/memo_run.py alone, so it covered the Monte Carlo, Sobol
and tornado results but contained no row for the paired EVPI ceiling
(0.4030 MUSD), the group EVPI table, the estimator null floor (0.1233 MUSD),
the seed replication spread, or the campaign costs, which are exactly the
numbers the RECOMMENDATION rests on. scripts/memo_append_csv.py and
scripts/memo_falsification.py now write those rows.

These tests fail if a load-bearing number is quoted in the memo without a CSV
row behind it, or if the value in the CSV drifts from the value in the JSON
record that produced it.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CSV_PATH = DOCS / "memo_numbers.csv"
MEMO = DOCS / "decision-memo.md"

#: Every (section, quantity) the memo's argument depends on. A missing row
#: here means the memo quotes a number with nothing behind it.
REQUIRED_ROWS = [
    ("evpi_ceiling", "evpi_all_mean"),
    ("evpi_ceiling", "within_run_standard_error"),
    ("evpi_ceiling", "sd_across_seeds"),
    ("evpi_ceiling", "seed_sd_over_within_run_se"),
    ("evpi_ceiling", "ceiling_usd"),
    ("evpi_ceiling", "ceiling_pct_of_prior_value"),
    ("evpi_ceiling", "prior_best_value"),
    ("evpi_null_calibration", "evpi_shipped"),
    ("evpi_null_calibration", "evpi_common_baseline"),
    ("evpi_group_control", "evpi_closed_form"),
    ("evpi_group_control", "nested_estimate"),
    ("evpi_group_control", "abs_gap"),
    ("evpi_group_control", "tolerance_3se"),
    ("assay_campaign", "per_sample_cost"),
    ("assay_campaign", "campaign_cost_low"),
    ("assay_campaign", "campaign_cost_high"),
    ("assay_campaign", "ceiling_over_cost_ratio_low"),
    ("assay_campaign", "ceiling_over_cost_ratio_high"),
    ("falsification", "overall_yield_at_zero_npv"),
    ("falsification", "npv_at_mc_overall_yield_p10"),
    ("falsification", "combined_mass_yield_total_order_sobol"),
]


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    return list(csv.DictReader(CSV_PATH.open()))


def test_every_load_bearing_quantity_has_a_csv_row(rows):
    have = {(r["section"], r["quantity"]) for r in rows}
    missing = [k for k in REQUIRED_ROWS if k not in have]
    assert not missing, f"memo quotes these with no CSV row: {missing}"


def test_ceiling_in_csv_matches_the_ceiling_json(rows):
    ce = json.loads((DOCS / "memo_evpi_ceiling.json").read_text())
    row = [r for r in rows if r["section"] == "evpi_ceiling"
           and r["quantity"] == "evpi_all_mean"][0]
    assert float(row["value"]) == pytest.approx(
        ce["across_seeds"]["mean_musd"], abs=1e-12)
    usd = [r for r in rows if r["section"] == "evpi_ceiling"
           and r["quantity"] == "ceiling_usd"][0]
    assert float(usd["value"]) == pytest.approx(
        ce["across_seeds"]["mean_musd"] * 1e6, rel=1e-9)


def test_null_floor_in_csv_matches_the_control_json(rows):
    c = json.loads((DOCS / "memo_evpi_control.json").read_text())
    row = [r for r in rows if r["section"] == "evpi_null_calibration"
           and r["parameter"] == "dummy"
           and r["quantity"] == "evpi_common_baseline"][0]
    assert float(row["value"]) == pytest.approx(
        c["null_calibration"]["evpi_common_baseline"], abs=1e-12)
    assert "NOISE FLOOR" in row["note"] or "noise floor" in row["note"].lower()


def test_group_evpi_rows_match_the_joint_json(rows):
    j = json.loads((DOCS / "memo_evpi_joint.json").read_text())
    for g in j["groups"]:
        row = [r for r in rows if r["section"] == "evpi_group"
               and r["parameter"] == g["group"] and r["quantity"] == "evpi"]
        assert row, f"no CSV row for group {g['group']}"
        assert float(row[0]["value"]) == pytest.approx(g["evpi"], abs=1e-12)


def test_campaign_cost_is_the_sample_count_times_the_per_sample_cost(rows):
    per = float([r for r in rows if r["section"] == "assay_campaign"
                 and r["quantity"] == "per_sample_cost"][0]["value"])
    lo = float([r for r in rows if r["section"] == "assay_campaign"
                and r["quantity"] == "campaign_cost_low"][0]["value"])
    hi = float([r for r in rows if r["section"] == "assay_campaign"
                and r["quantity"] == "campaign_cost_high"][0]["value"])
    n_lo = float([r for r in rows if r["section"] == "assay_campaign"
                  and r["quantity"] == "n_samples_low"][0]["value"])
    n_hi = float([r for r in rows if r["section"] == "assay_campaign"
                  and r["quantity"] == "n_samples_high"][0]["value"])
    assert lo == pytest.approx(per * n_lo)
    assert hi == pytest.approx(per * n_hi)


def test_sample_count_rows_are_tagged_assumed_and_say_where_they_came_from(rows):
    for q in ("n_samples_low", "n_samples_high"):
        row = [r for r in rows if r["section"] == "assay_campaign"
               and r["quantity"] == q][0]
        assert row["tag"] == "ASSUMED", (
            f"{q} is tagged {row['tag']}; the sample count has no power "
            "calculation behind it in this analysis")
        assert "project brief" in row["note"].lower()
        assert "power calculation" in row["note"].lower()


def test_yield_threshold_is_a_root_not_a_percentile(rows):
    """The memo once equated the yield P10 with the NPV P10. It does not now."""
    f = json.loads((DOCS / "memo_falsification.json").read_text())
    root = f["overall_yield_at_zero_npv"]
    assert abs(root["npv_at_threshold"]) < 1e-3, (
        "the stated threshold must be where NPV is zero")
    # The root and the Monte Carlo yield P10 are different numbers, and the
    # memo must not conflate them: the P10 maps to a clearly positive NPV.
    assert f["mc_yield_p10_maps_to_npv"]["npv_musd"] > 5.0
    assert root["overall_yield"] < f["mc_yield_p10_maps_to_npv"]["overall_yield_p10"]


def test_memo_does_not_claim_the_yield_p10_is_the_npv_p10():
    text = MEMO.read_text()
    assert "below about 0.66 puts the project at the modelled P10" not in text
    assert "0.5385" in text, "the measured zero-NPV yield root must be quoted"


def test_band_relative_notes_are_computed_not_asserted(rows):
    """A note claiming a threshold is below its band must actually be below it.

    An earlier version hardcoded a "below the ASSUMED band low end" note for
    the implied flotation yield, while the solved value is INSIDE that band.
    The note is now derived from the band, and this test compares every such
    note against the input_band rows in the same CSV, so the verdict in the
    prose is checked against the numbers rather than asserted alongside them.
    """
    bands = {r["parameter"]: float(r["value"]) for r in rows
             if r["section"] == "input_band" and r["quantity"] == "low"}
    assert bands, "no input_band low rows found in the CSV"
    checked = 0
    for r in rows:
        if r["section"] != "falsification":
            continue
        param, note, val = r["parameter"], r["note"], r["value"]
        if param not in bands or not note.strip() or val in ("", "None"):
            continue
        low = bands[param]
        v = float(val)
        if "BELOW the ASSUMED band" in note:
            assert v < low, (
                f"{r['quantity']} note says BELOW the band but {v} >= {low}")
            checked += 1
        if "INSIDE the ASSUMED band" in note:
            assert v >= low, (
                f"{r['quantity']} note says INSIDE the band but {v} < {low}")
            checked += 1
    assert checked >= 3, f"expected several band-relative notes, checked {checked}"


def test_memo_does_not_claim_both_yields_fall_below_their_bands():
    text = MEMO.read_text()
    assert "both below the low ends of their ASSUMED bands" not in text
    assert "0.7053" in text and "INSIDE its ASSUMED band" in text


def test_memo_quotes_one_value_for_the_yield_p10_npv():
    """The memo said +10.58 in one place and +10.57 in another for one number."""
    f = json.loads((DOCS / "memo_falsification.json").read_text())
    true = f["mc_yield_p10_maps_to_npv"]["npv_musd"]
    assert round(true, 2) == 10.57, f"recorded value moved to {true}"
    text = MEMO.read_text()
    assert "+10.58 MUSD" not in text, "memo mis-rounds 10.574 as 10.58"
    assert text.count("+10.57 MUSD") >= 2


def test_memo_csv_line_count_claim_matches_the_file(rows):
    text = MEMO.read_text()
    m = re.search(r"`docs/memo_numbers\.csv` \((\d+) data rows", text)
    assert m, "memo must state the CSV size as 'N data rows'"
    assert int(m.group(1)) == len(rows), (
        f"memo claims {m.group(1)} data rows, file has {len(rows)}")


def test_memo_reproduction_block_lists_every_script_that_writes_the_csv():
    text = MEMO.read_text()
    for script in ("memo_run.py", "memo_evpi_control.py", "memo_evpi_joint.py",
                   "memo_evpi_ceiling.py", "memo_append_csv.py",
                   "memo_falsification.py"):
        assert script in text, f"reproduction block omits {script}"
        assert (ROOT / "scripts" / script).exists(), f"{script} does not exist"


def test_memo_quotes_no_pytest_wall_time():
    """A wall time cannot be guarded, so quoting one invites fabrication.

    The memo once quoted a pytest line as verbatim run output when no run had
    produced it: the count came from a guard failure and the runtime beside it
    was typed because it looked plausible. Counts are checkable against the
    files; runtimes are a property of the machine and are not.
    """
    text = MEMO.read_text()
    hits = re.findall(r"\d+ (?:passed|failed) in [\d.]+\s*s", text)
    assert not hits, f"memo quotes unguardable pytest wall times: {hits}"


def test_memo_quoted_test_count_matches_the_files_it_names():
    """The memo quotes a pytest total. It went stale twice; this catches that."""
    text = MEMO.read_text()
    m = re.search(r"`(\d+) passed`", text)
    assert m, "memo must quote a measured pytest count as `N passed`"
    claimed = int(m.group(1))
    counted = 0
    for f in ("test_memo_numbers.py", "test_memo_traceability.py",
              "test_memo_pdf.py"):
        src = (ROOT / "tests" / f).read_text()
        counted += len(re.findall(r"^def test_", src, flags=re.M))
        # parametrize multiplies the collected count; add the extra cases
        for m2 in re.finditer(r"@pytest\.mark\.parametrize\([^\[]*\[(.*?)\]\)",
                              src, flags=re.S):
            counted += len(re.findall(r"\(", m2.group(1))) - 1
    assert claimed == counted, (
        f"memo quotes {claimed} passing tests; the three named files define "
        f"{counted}")


def test_memo_has_no_em_or_en_dashes():
    text = MEMO.read_text()
    bad = re.findall(r"[\u2012\u2013\u2014\u2015\u2212]", text)
    assert not bad, f"{len(bad)} dash characters in the memo"
