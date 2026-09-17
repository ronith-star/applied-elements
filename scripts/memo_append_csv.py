"""Append the control, group and ceiling results to docs/memo_numbers.csv.

WHY THIS EXISTS. scripts/memo_run.py writes the CSV from its own JSON record,
so the CSV covered only the headline campaign: Monte Carlo, Sobol, tornado,
and the per-parameter EVPI that three later controls showed is not reportable.
The numbers the memo's RECOMMENDATION actually rests on were produced by three
later scripts and had no rows at all: the paired EVPI ceiling (0.4030 MUSD),
the group EVPI table, the estimator null floor (0.1233 MUSD), the seed
replication spread, and the campaign costs. The memo asserted that every
figure it quotes appears in the CSV, which was false for exactly the
load-bearing figures.

This script reads the three JSON records and appends rows in the same schema
(section, quantity, parameter, output, value, unit, tag, note) so the
traceability claim holds for every number the memo quotes. It is idempotent:
it rewrites the appended sections rather than duplicating them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CSV_PATH = DOCS / "memo_numbers.csv"
FIELDS = ["section", "quantity", "parameter", "output", "value", "unit",
          "tag", "note"]
#: Sections this script owns. Existing rows in these sections are dropped
#: before the new ones are written, so re-running does not duplicate.
OWNED = {"evpi_seed_replication", "evpi_null_calibration", "evpi_group",
         "evpi_group_control", "evpi_ceiling", "assay_campaign"}


def rows_from_control(c: dict) -> list[dict]:
    out = []
    base = c["baseline_common"]
    out.append(dict(section="evpi_null_calibration", quantity="baseline",
                    parameter="", output="prior_best_value",
                    value=base["prior_best_value_musd"], unit="MUSD",
                    tag="COMPUTED",
                    note=f"shared baseline, {base['n_draws']} draws, best action "
                         f"{base['prior_best_action']}"))
    for p, s in c["control_a_summary"].items():
        for q, v in (("evpi_mean", s["evpi_mean"]), ("evpi_sd", s["evpi_sd"]),
                     ("evpi_max", s["evpi_max"]),
                     ("baseline_sd", s["baseline_sd"]),
                     ("switch_fraction_max", s["switch_fraction_max"])):
            out.append(dict(section="evpi_seed_replication", quantity=q,
                            parameter=p, output="npv_musd", value=v,
                            unit="MUSD" if "evpi" in q or "baseline" in q else "fraction",
                            tag="COMPUTED",
                            note="ae.agent.decisions.evpi, n_outer=256, "
                                 "n_inner=256, seeds 11/23/37/53/71/97"))
    n = c["null_calibration"]
    out.append(dict(section="evpi_null_calibration", quantity="evpi_shipped",
                    parameter="dummy", output="npv_musd",
                    value=n["evpi_shipped"], unit="MUSD", tag="COMPUTED",
                    note="dummy input drawn from a prior and discarded by the "
                         "value function; TRUE EVPI is exactly 0"))
    out.append(dict(section="evpi_null_calibration",
                    quantity="evpi_common_baseline", parameter="dummy",
                    output="npv_musd", value=n["evpi_common_baseline"],
                    unit="MUSD", tag="COMPUTED",
                    note="same dummy, re-formed against the shared baseline; "
                         "this is the estimator NOISE FLOOR"))
    for r in c["control_bc_common_baseline"]:
        out.append(dict(section="evpi_null_calibration",
                        quantity="evpi_common_baseline",
                        parameter=r["parameter"], output="npv_musd",
                        value=r["evpi_common_baseline"], unit="MUSD",
                        tag="COMPUTED",
                        note="single-parameter EVPI against one shared "
                             "high-precision baseline, no per-call baseline"))
    return out


def rows_from_joint(j: dict) -> list[dict]:
    out = []
    fi = j["full_information"]
    out.append(dict(section="evpi_group_control", quantity="evpi_closed_form",
                    parameter="all_nine", output="npv_musd",
                    value=fi["evpi_closed_form_musd"], unit="MUSD",
                    tag="COMPUTED",
                    note=f"E[max_a V] - max_a E[V], no nesting, "
                         f"{fi['n_draws']} draws"))
    cf = j["all_nine_vs_closed_form"]
    for q, v, u in (("nested_estimate", cf["nested_musd"], "MUSD"),
                    ("abs_gap", cf["abs_gap_musd"], "MUSD"),
                    ("nested_standard_error", cf["nested_standard_error_musd"], "MUSD"),
                    ("tolerance_3se", cf["tolerance_3se_musd"], "MUSD")):
        out.append(dict(section="evpi_group_control", quantity=q,
                        parameter="all_nine", output="npv_musd", value=v,
                        unit=u, tag="COMPUTED",
                        note="definitional control: nested group EVPI over all "
                             "parameters must reproduce the closed form to "
                             "within 3 standard errors"))
    for g in j["groups"]:
        note = ("nested MC, n_outer=512, n_inner=512, seed 11, no clamping at "
                "zero; params: " + ", ".join(g["parameters"]))
        for q, v, u in (("evpi", g["evpi"], "MUSD"),
                        ("standard_error", g["standard_error"], "MUSD"),
                        ("ci95_low", g["evpi_ci95_low"], "MUSD"),
                        ("ci95_high", g["evpi_ci95_high"], "MUSD"),
                        ("switch_fraction", g["switch_fraction"], "fraction"),
                        ("cost", g["cost_usd"], "USD")):
            tag = "COMPUTED"
            if q == "cost":
                tag = "DERIVED"
                note_c = ("sum of the per-parameter measurement costs in "
                          "memo_run.MEASUREMENT_COSTS; upper bound, shared "
                          "sample preparation not netted out")
                out.append(dict(section="evpi_group", quantity=q,
                                parameter=g["group"], output="", value=v,
                                unit=u, tag=tag, note=note_c))
                continue
            out.append(dict(section="evpi_group", quantity=q,
                            parameter=g["group"], output="npv_musd", value=v,
                            unit=u, tag=tag, note=note))
    return out


def rows_from_ceiling(ce: dict) -> list[dict]:
    out = []
    a = ce["across_seeds"]
    note = ("paired estimator d_k = max_a V(a,theta_k) - V(a*,theta_k), "
            "200,000 draws x 4 seeds; cancels the shared NPV variance")
    for q, v, u in (("evpi_all_mean", a["mean_musd"], "MUSD"),
                    ("sd_across_seeds", a["sd_across_seeds_musd"], "MUSD"),
                    ("within_run_standard_error", a["mean_standard_error_musd"], "MUSD"),
                    ("min", a["min_musd"], "MUSD"),
                    ("max", a["max_musd"], "MUSD"),
                    ("seed_sd_over_within_run_se", ce["seed_sd_over_within_run_se"], "ratio"),
                    ("ceiling_pct_of_prior_value", ce["ceiling_pct_of_prior_value"], "percent"),
                    ("prior_best_value", ce["prior_best_value_musd"], "MUSD")):
        out.append(dict(section="evpi_ceiling", quantity=q,
                        parameter="all_nine", output="npv_musd", value=v,
                        unit=u, tag="COMPUTED", note=note))
    for r in ce["runs"]:
        out.append(dict(section="evpi_ceiling", quantity="evpi_all_by_seed",
                        parameter=f"seed_{r['seed']}", output="npv_musd",
                        value=r["evpi_all_musd"], unit="MUSD", tag="COMPUTED",
                        note=note))
        out.append(dict(section="evpi_ceiling",
                        quantity="fraction_of_draws_where_information_binds",
                        parameter=f"seed_{r['seed']}", output="npv_musd",
                        value=r["fraction_of_draws_where_information_binds"],
                        unit="fraction", tag="COMPUTED", note=note))
    out.append(dict(section="evpi_ceiling", quantity="ceiling_usd",
                    parameter="all_nine", output="npv_musd",
                    value=a["mean_musd"] * 1e6, unit="USD", tag="COMPUTED",
                    note="the same ceiling expressed in USD, for comparison "
                         "against campaign cost"))
    oc = ce["ore_campaign"]
    out.append(dict(section="assay_campaign", quantity="per_sample_cost",
                    parameter="ore_all_four", output="", value=oc["per_sample_cost_usd"],
                    unit="USD", tag="DERIVED",
                    note="sum of the four ore parameter costs: 64.60 SOURCED, "
                         "273.40 and 1140.00 DERIVED, 1000.00 ASSUMED proxy"))
    for q, v in (("n_samples_low", oc["n_samples_low"]),
                 ("n_samples_high", oc["n_samples_high"])):
        out.append(dict(section="assay_campaign", quantity=q,
                        parameter="ore_all_four", output="", value=v,
                        unit="samples", tag="ASSUMED",
                        note="20 to 30 samples labelled AE-Q-### comes from the "
                             "Quartz Foundry project brief, NOT from a power "
                             "calculation performed here"))
    for q, v in (("campaign_cost_low", oc["campaign_cost_usd_low"]),
                 ("campaign_cost_high", oc["campaign_cost_usd_high"])):
        out.append(dict(section="assay_campaign", quantity=q,
                        parameter="ore_all_four", output="", value=v,
                        unit="USD", tag="DERIVED",
                        note="per-sample cost times the ASSUMED sample count"))
    for q, v in (("ceiling_over_cost_ratio_low", oc["ceiling_over_cost_ratio_low"]),
                 ("ceiling_over_cost_ratio_high", oc["ceiling_over_cost_ratio_high"])):
        out.append(dict(section="assay_campaign", quantity=q,
                        parameter="ore_all_four", output="npv_musd", value=v,
                        unit="ratio", tag="COMPUTED",
                        note="information ceiling divided by campaign cost; "
                             "cover ratio, not a return on investment"))
    for q, v in (("campaign_pct_of_ceiling_low",
                  100.0 * oc["campaign_cost_usd_low"] / (ce["ceiling_musd"] * 1e6)),
                 ("campaign_pct_of_ceiling_high",
                  100.0 * oc["campaign_cost_usd_high"] / (ce["ceiling_musd"] * 1e6))):
        out.append(dict(section="assay_campaign", quantity=q,
                        parameter="ore_all_four", output="npv_musd", value=v,
                        unit="percent", tag="COMPUTED",
                        note="campaign cost as a percentage of the information "
                             "ceiling"))
    return out


def main() -> None:
    existing = list(csv.DictReader(CSV_PATH.open()))
    kept = [r for r in existing if r["section"] not in OWNED]
    new = (rows_from_control(json.loads((DOCS / "memo_evpi_control.json").read_text()))
           + rows_from_joint(json.loads((DOCS / "memo_evpi_joint.json").read_text()))
           + rows_from_ceiling(json.loads((DOCS / "memo_evpi_ceiling.json").read_text())))
    with CSV_PATH.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(kept)
        w.writerows(new)
    print(f"[csv] kept {len(kept)} rows, appended {len(new)} rows, total "
          f"{len(kept) + len(new)} -> {CSV_PATH}")
    secs: dict[str, int] = {}
    for r in kept + new:
        secs[r["section"]] = secs.get(r["section"], 0) + 1
    for s in sorted(secs):
        print(f"  {s:26s} {secs[s]:4d}")


if __name__ == "__main__":
    main()
