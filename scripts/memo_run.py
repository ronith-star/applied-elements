"""Run every number that appears in docs/decision-memo.md, and write them to CSV.

This script exists so the memo is traceable to a single executed file rather
than to numbers typed by hand. It runs, in order:

1. Monte Carlo over the integrated ore-to-NPV chain (tests/test_integration_
   uncertainty.py::plant_npv), 20,000 draws.
2. Sobol variance decomposition on four outputs of that chain, plus a
   convergence check at N, N/2 and N/4 on the headline output.
3. One-at-a-time tornado at the nominal point, for presentation only.
4. EVPI per parameter and EVPI per unit measurement cost over a three-action
   decision problem (walk away, build the base flowsheet, build the base
   flowsheet plus a polishing stage).

WHY A THREE-ACTION PROBLEM. src.ae.agent.decisions.DecisionProblem rejects a
single-action problem, correctly: EVPI against one available action is
identically zero because information cannot change a decision already made.
The action set below is therefore part of the model, and it is ASSUMED
structure, not a measurement. Its parameters (polish capex, polish operating
cost, polish Al reduction factor) are ASSUMED and declared as module constants
so the memo can state them.

The Vikarabad deposit is UNCHARACTERIZED in the citable record. Every input
band here is a scenario band taken from chain_inputs() in the integration
test, not an assay result, and every EVPI below is computed against those
ASSUMED priors.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from ae.agent.decisions import (  # noqa: E402
    Action,
    DecisionProblem,
    evpi_convergence,
    measurement_priority,
    rank_measurements,
)
from ae.econ.uncertainty import (  # noqa: E402
    convergence_check,
    monte_carlo,
    sobol_analysis,
    tornado,
)
from ae.econ.valuation import Project  # noqa: E402
from ae.plant.yield_cascade import cascade_yield, off_spec_fraction  # noqa: E402
from test_integration_uncertainty import (  # noqa: E402
    FEED_TONNES,
    USL_AL,
    chain_inputs,
    plant_npv,
)

# --- ASSUMED structure of the polishing action ---------------------------
#: Extra capital for a polishing stage (hot chlorination), MUSD. ASSUMED.
POLISH_CAPEX_MUSD = 9.0
#: Extra operating cost of that stage, USD per tonne of product. ASSUMED.
POLISH_OPEX_USD_PER_T = 35.0
#: Multiplier applied to the product Al mean by that stage. ASSUMED. Set above
#: zero deliberately: lattice-bound Al is not removable by any surface or gas
#: phase treatment, so a polishing stage cannot drive Al to zero.
POLISH_AL_FACTOR = 0.60

N_MC_DRAWS = 20_000
N_SOBOL_BASE = 4096
N_SOBOL_BASE_SECONDARY = 2048
N_EVPI_OUTER = 256
N_EVPI_INNER = 256

OUTPUTS = ("npv_musd", "breakeven", "overall_yield", "cash_cost")

NOMINAL = {
    "mass_yield_leach": 0.92,
    "mass_yield_flot": 0.85,
    "al_mean_ppm": 18.0,
    "al_sigma_ppm": 3.0,
    "price": 3500.0,
    "power_price": 0.07,
    "reagent_price": 1.80,
    "capex_musd": 38.0,
    "discount_rate": 0.14,
}


def npv_musd(polish: bool = False, **k: float) -> float:
    """NPV in MUSD for one parameter vector, optionally with the polish stage.

    With ``polish=False`` this reproduces ``plant_npv(**k)["npv_musd"]``
    exactly, which ``verify_npv_matches_plant_npv`` asserts. It exists
    separately because ``plant_npv`` also computes a breakeven price by root
    finding, which costs roughly sixteen times the NPV alone and would make
    the nested EVPI loop below intractable.
    """
    al_mean = k["al_mean_ppm"] * (POLISH_AL_FACTOR if polish else 1.0)
    mass_y = cascade_yield([k["mass_yield_flot"], k["mass_yield_leach"]])
    spec_y = 1.0 - off_spec_fraction(al_mean, k["al_sigma_ppm"], USL_AL,
                                     model="lognormal")
    overall = mass_y * spec_y
    feed_cost_per_t_feed = 250.0 * k["power_price"] + 12.0 * k["reagent_price"]
    cash = feed_cost_per_t_feed / overall + 180.0
    if polish:
        cash += POLISH_OPEX_USD_PER_T
    capex = k["capex_musd"] + (POLISH_CAPEX_MUSD if polish else 0.0)
    p = Project(
        capex_schedule=[capex * 1e6 / 2] * 2,
        construction_periods=2,
        ramp_fractions=[0.35, 0.75, 1.0],
        nameplate_tonnes=FEED_TONNES * overall,
        price=k["price"],
        cash_cost_per_tonne=cash,
        life_periods=15,
        fixed_cost_per_period=3.0e6,
        tax_rate=0.21,
    )
    return float(p.cash_flows().npv(k["discount_rate"]) / 1e6)


def verify_npv_matches_plant_npv() -> list[float]:
    """Control check: the fast value function must equal the audited chain.

    Returns the absolute differences so the caller can record them. Raises if
    any draw disagrees, because an EVPI computed on a different model than the
    Sobol indices would make the memo internally inconsistent.
    """
    ins = chain_inputs()
    rng = np.random.default_rng(7)
    u = rng.random((200, len(ins)))
    diffs = []
    for r in range(u.shape[0]):
        kw = {inp.name: float(inp.ppf(u[r:r + 1, i])[0])
              for i, inp in enumerate(ins)}
        a = plant_npv(**kw)["npv_musd"]
        b = npv_musd(polish=False, **kw)
        diffs.append(abs(a - b))
    worst = max(diffs)
    if worst != 0.0:
        raise AssertionError(
            f"npv_musd disagrees with plant_npv by up to {worst:.3e} MUSD; the "
            "EVPI and Sobol sections would then describe different models"
        )
    return diffs


ACTIONS = [
    Action("walk_away", cost=0.0),
    Action("build_base", cost=0.0),
    Action("build_polish", cost=0.0),
]


def value(action: str, params) -> float:
    """Decision payoff in MUSD. Walking away is worth exactly zero.

    Action cost is zero for all three because capital is already inside the
    NPV through ``capex_musd``; charging it twice via ``Action.cost`` would
    double count it.
    """
    if action == "walk_away":
        return 0.0
    return npv_musd(polish=(action == "build_polish"), **dict(params))


def priors_from_chain():
    """Samplers matching the Sobol and Monte Carlo input bands exactly."""
    ins = chain_inputs()

    def make(inp):
        def sampler(rng, n):
            return inp.ppf(rng.random(n))
        return sampler

    return {inp.name: make(inp) for inp in ins}


# --- Measurement costs, USD per sample ----------------------------------
# SOURCED entries are per-sample list prices read from two primary fee
# schedules downloaded in this session:
#   Actlabs (Activation Laboratories Ltd.), "Geochemistry Schedule of
#   Services & Fees, International 2026", dated January 22, 2026, USD.
#   https://actlabs.com/wp-content/uploads/2026/01/Actlabs_Geochemistry_Schedule_of_Services_2026-01-22_USD.pdf
#   Hazen Research, Inc., "Analytical Laboratory Services Fee Schedule",
#   effective July 1, 2026, USD.
#   https://www.hazenresearch.com/sites/default/files/hazen_analytical_fee_schedule_july_2026.pdf
# ASSUMED entries are NOT assay prices. Both laboratories quote bench
# metallurgical testing and LA-ICP-MS "by request" or "by quote", and market,
# power, reagent and EPC figures are commercial studies rather than laboratory
# services, so no list price exists to cite. Those four are ASSUMED and the
# memo says so.
MEASUREMENT_COSTS = {
    # Actlabs RX1 prep 12.40 + Code 4B2-Std fusion ICP-MS trace element
    # package 52.20 at the 11+ sample price. SOURCED.
    "al_mean_ppm": 12.40 + 52.20,
    # Lot-to-lot variance needs replicate lots, not one assay. Priced as five
    # of the same package on one prep: 12.40 + 5 x 52.20. DERIVED from the
    # same SOURCED unit price; the replicate count 5 is ASSUMED.
    "al_sigma_ppm": 12.40 + 5 * 52.20,
    # Hazen Bond ball mill grindability 1,000 as the cheapest published proxy
    # for a flotation-circuit mass-yield test. Bench flotation itself is "By
    # Quote" at both laboratories. ASSUMED proxy on a SOURCED price.
    "mass_yield_flot": 1000.0,
    # Hazen "Leach" dissolution 10 plus prep 30 plus an ICP-OES 4-acid solids
    # scan 155 plus 20 digestion, per leach condition, times an ASSUMED six
    # conditions to build a leach response curve. DERIVED from SOURCED prices.
    "mass_yield_leach": 6 * (10.0 + 155.0 + 20.0) + 30.0,
    # ASSUMED. Not a laboratory service.
    "price": 45000.0,
    "capex_musd": 120000.0,
    "power_price": 15000.0,
    "reagent_price": 8000.0,
    "discount_rate": 25000.0,
}

#: Per-parameter (tag, note) for every entry in MEASUREMENT_COSTS, so the CSV
#: carries the provenance of THAT cost rather than one blanket note. The four
#: commercial parameters are ASSUMED and must not carry a fee-schedule note:
#: no laboratory sells them, so citing a fee schedule against them would be a
#: fabricated provenance. mass_yield_flot is ASSUMED, not DERIVED: the dollar
#: figure is a SOURCED list price but the claim that a grindability test
#: substitutes for a flotation mass-yield test is judgement, and the weaker
#: tag governs.
COST_PROVENANCE: dict[str, tuple[str, str]] = {
    "al_mean_ppm": (
        "SOURCED",
        "Actlabs 2026 fee schedule: RX1 prep 12.40 plus Code 4B2-Std fusion "
        "ICP-MS 52.20 at the 11+ sample price"),
    "al_sigma_ppm": (
        "DERIVED",
        "Actlabs 2026 SOURCED unit prices, 12.40 plus 5 x 52.20; the "
        "replicate count of 5 lots is ASSUMED"),
    "mass_yield_flot": (
        "ASSUMED",
        "proxy: Hazen July 2026 Bond ball mill grindability 1,000 USD is "
        "SOURCED, but standing in for a bench flotation test is judgement. "
        "Bench flotation is quoted By Quote at both laboratories"),
    "mass_yield_leach": (
        "DERIVED",
        "Hazen July 2026 SOURCED prices: 6 x (leach 10 plus ICP-OES solids "
        "scan 155 plus 4-acid digestion 20) plus prep 30; the 6 leach "
        "conditions are ASSUMED"),
    "price": (
        "ASSUMED",
        "commercial market study, not a laboratory service; no list price "
        "exists to cite"),
    "capex_musd": (
        "ASSUMED",
        "EPC or engineering cost study, not a laboratory service; no list "
        "price exists to cite"),
    "power_price": (
        "ASSUMED",
        "tariff negotiation or power market study, not a laboratory service; "
        "no list price exists to cite"),
    "reagent_price": (
        "ASSUMED",
        "reagent supply quotation, not a laboratory service; no list price "
        "exists to cite"),
    "discount_rate": (
        "ASSUMED",
        "cost of capital is set by financing terms, not measurable by assay; "
        "the figure is the cost of establishing terms"),
}
if set(COST_PROVENANCE) != set(MEASUREMENT_COSTS):
    raise AssertionError(
        "COST_PROVENANCE and MEASUREMENT_COSTS must cover the same "
        f"parameters; differences: "
        f"{set(COST_PROVENANCE) ^ set(MEASUREMENT_COSTS)}"
    )


def main() -> None:
    t0 = time.perf_counter()
    rec: dict[str, object] = {}

    diffs = verify_npv_matches_plant_npv()
    rec["value_function_control"] = {
        "n_checked": len(diffs), "max_abs_diff_musd": max(diffs)}
    print(f"[control] npv_musd vs plant_npv over {len(diffs)} draws: "
          f"max abs diff {max(diffs):.3e} MUSD", flush=True)

    ins = chain_inputs()
    rec["inputs"] = [
        {"name": i.name, "low": i.low, "high": i.high, "kind": i.kind,
         "mode": i.mode} for i in ins]

    # 1. Monte Carlo ------------------------------------------------------
    print(f"[mc] {N_MC_DRAWS} draws ...", flush=True)
    mc = monte_carlo(plant_npv, ins, n_draws=N_MC_DRAWS, seed=42)
    rec["mc"] = {
        "n_draws": mc.n_draws, "n_failed": mc.n_failed,
        "summary": {o: mc.summary(o) for o in OUTPUTS},
        "p_npv_below_zero": mc.probability_below("npv_musd", 0.0),
        "percentile_se": {
            f"npv_P{p}": mc.percentile_standard_error("npv_musd", p)
            for p in (10, 50, 90)},
    }
    s = mc.summary("npv_musd")
    print(f"[mc] NPV P10 {s['P10']:+.2f} P50 {s['P50']:+.2f} "
          f"P90 {s['P90']:+.2f} MUSD, P(NPV<0) "
          f"{rec['mc']['p_npv_below_zero']:.4f}", flush=True)

    # 2. Sobol ------------------------------------------------------------
    rec["sobol"] = {}
    for out in OUTPUTS:
        n = N_SOBOL_BASE if out == "npv_musd" else N_SOBOL_BASE_SECONDARY
        print(f"[sobol] {out} at N={n} ...", flush=True)
        r = sobol_analysis(plant_npv, ins, n_base=n, seed=43, output=out)
        rec["sobol"][out] = {
            "n_base": r.n_base, "n_evaluations": r.n_evaluations,
            "output_variance": r.output_variance,
            "first_order": r.first_order, "total_order": r.total_order,
            "first_conf": r.first_conf, "total_conf": r.total_conf,
            "interaction_share": r.interaction_share,
            "additive_fraction": r.additive_fraction,
            "ranking": r.ranking(), "negligible": r.negligible(),
            "diagnostics": r.diagnostics(),
        }
        print(f"[sobol] {out} top3 {r.ranking()[:3]}", flush=True)

    print(f"[sobol] convergence at N={N_SOBOL_BASE} ...", flush=True)
    cc = convergence_check(plant_npv, ins, n_base=N_SOBOL_BASE, seed=43,
                           output="npv_musd")
    rec["sobol_convergence"] = {
        "levels": cc["levels"],
        "top_input": {str(k): v for k, v in cc["top_input"].items()},
        "ranking_stable": cc["ranking_stable"],
        "max_total_drift": cc["max_total_drift"],
        "diagnostics": {str(k): v for k, v in cc["diagnostics"].items()},
        "total_order_by_level": {
            str(n): res.total_order for n, res in cc["results"].items()},
    }

    # 3. Tornado ----------------------------------------------------------
    print("[tornado] ...", flush=True)
    rec["tornado"] = tornado(plant_npv, ins, NOMINAL, output="npv_musd")
    rec["tornado_nominal"] = NOMINAL

    # 4. EVPI -------------------------------------------------------------
    problem = DecisionProblem(actions=ACTIONS, value=value,
                              priors=priors_from_chain())
    ev = problem.expected_values(n_draws=4096, seed=11)
    best, best_v = problem.best_action_now(n_draws=4096, seed=11)
    rec["decision"] = {
        "actions": [a.name for a in ACTIONS],
        "expected_values_musd": ev,
        "prior_best_action": best,
        "prior_best_value_musd": best_v,
        "assumed_polish": {"capex_musd": POLISH_CAPEX_MUSD,
                           "opex_usd_per_t": POLISH_OPEX_USD_PER_T,
                           "al_factor": POLISH_AL_FACTOR},
    }
    print(f"[evpi] prior expected values {ev}", flush=True)

    print(f"[evpi] ranking, outer={N_EVPI_OUTER} inner={N_EVPI_INNER} ...",
          flush=True)
    ivs = rank_measurements(problem, n_outer=N_EVPI_OUTER,
                            n_inner=N_EVPI_INNER, seed=11)
    rec["evpi"] = [
        {"parameter": iv.parameter, "evpi_musd": iv.evpi,
         "baseline_value_musd": iv.baseline_value,
         "resolved_value_musd": iv.resolved_value,
         "switch_fraction": iv.switch_fraction,
         "prior_best_action": iv.prior_best_action,
         "evpi_fraction": iv.evpi_fraction,
         "n_outer": iv.n_outer, "n_inner": iv.n_inner} for iv in ivs]
    for iv in ivs:
        print(f"[evpi] {iv.parameter:18s} {iv.evpi:8.4f} MUSD  switch "
              f"{iv.switch_fraction:.3f}", flush=True)

    print("[evpi] per unit cost ...", flush=True)
    pri = measurement_priority(problem, MEASUREMENT_COSTS,
                               n_outer=N_EVPI_OUTER, n_inner=N_EVPI_INNER,
                               seed=11)
    rec["evpi_per_cost"] = pri
    # Control: measurement_priority calls rank_measurements internally with
    # the same seed, so the two must return identical EVPI values. If they
    # drift, one of the two rankings in the memo is from a different sample
    # and the ratio column would not correspond to the EVPI column.
    by_param = {r["parameter"]: r["evpi"] for r in pri}
    worst = max(abs(by_param[r["parameter"]] - r["evpi_musd"])
                for r in rec["evpi"])
    if worst > 0.0:
        raise AssertionError(
            f"rank_measurements and measurement_priority disagree by up to "
            f"{worst:.3e} MUSD at the same seed"
        )
    rec["evpi_consistency_control_max_abs_diff_musd"] = worst
    print(f"[control] EVPI two-entry-point agreement: max abs diff "
          f"{worst:.3e} MUSD", flush=True)
    rec["measurement_costs_usd"] = MEASUREMENT_COSTS

    top_evpi_param = rec["evpi"][0]["parameter"]
    print(f"[evpi] convergence on {top_evpi_param} ...", flush=True)
    rec["evpi_convergence"] = {
        "parameter": top_evpi_param,
        "series": evpi_convergence(problem, top_evpi_param,
                                   inner_sizes=(16, 64, 256), n_outer=128,
                                   seed=11),
    }

    rec["runtime_seconds"] = time.perf_counter() - t0
    rec["n_mc_draws"] = N_MC_DRAWS

    out_json = ROOT / "docs" / "memo_numbers.json"
    out_json.parent.mkdir(exist_ok=True)
    out_json.write_text(json.dumps(rec, indent=1, default=str))
    write_csv(rec, ROOT / "docs" / "memo_numbers.csv")
    print(f"[done] {rec['runtime_seconds']:.1f} s -> {out_json}", flush=True)


def write_csv(rec: dict, path: Path) -> None:
    """Flat long-format CSV: one row per reported quantity.

    Long format rather than one wide table per section, because the memo cites
    single quantities and a reader checking one figure should be able to grep
    for it.
    """
    rows = []

    def add(section, quantity, param, output, value, unit, tag, note=""):
        rows.append({"section": section, "quantity": quantity,
                     "parameter": param, "output": output, "value": value,
                     "unit": unit, "tag": tag, "note": note})

    add("control", "max_abs_diff_npv_musd", "", "npv_musd",
        rec["value_function_control"]["max_abs_diff_musd"], "MUSD", "MEASURED",
        f"fast value function vs plant_npv over "
        f"{rec['value_function_control']['n_checked']} draws")

    for i in rec["inputs"]:
        add("input_band", "low", i["name"], "", i["low"], "native", "ASSUMED",
            f"{i['kind']} scenario band from chain_inputs()")
        add("input_band", "high", i["name"], "", i["high"], "native",
            "ASSUMED", f"{i['kind']} scenario band from chain_inputs()")
        if i["mode"] is not None:
            add("input_band", "mode", i["name"], "", i["mode"], "native",
                "ASSUMED", "triangular mode from chain_inputs()")

    for out, d in rec["mc"]["summary"].items():
        unit = {"npv_musd": "MUSD", "breakeven": "USD/t",
                "overall_yield": "fraction", "cash_cost": "USD/t"}[out]
        for k, v in d.items():
            add("monte_carlo", k, "", out, v, unit, "COMPUTED",
                f"{rec['mc']['n_draws']} draws, seed 42")
    add("monte_carlo", "P(NPV<0)", "", "npv_musd",
        rec["mc"]["p_npv_below_zero"], "fraction", "COMPUTED",
        f"{rec['mc']['n_draws']} draws, seed 42")
    for k, v in rec["mc"]["percentile_se"].items():
        add("monte_carlo", f"bootstrap_se_{k}", "", "npv_musd", v, "MUSD",
            "COMPUTED", "400 bootstrap resamples")

    for out, d in rec["sobol"].items():
        for p in d["total_order"]:
            add("sobol", "total_order_ST", p, out, d["total_order"][p],
                "variance share", "COMPUTED",
                f"Saltelli N={d['n_base']}, {d['n_evaluations']} evaluations")
            add("sobol", "first_order_S1", p, out, d["first_order"][p],
                "variance share", "COMPUTED",
                f"Saltelli N={d['n_base']}, {d['n_evaluations']} evaluations")
            add("sobol", "total_order_conf95", p, out, d["total_conf"][p],
                "variance share", "COMPUTED", "SALib bootstrap CI")
        add("sobol", "additive_fraction_sum_S1", "", out,
            d["additive_fraction"], "variance share", "COMPUTED", "")
        add("sobol", "sum_total_order", "", out,
            d["diagnostics"]["sum_total_order"], "variance share", "COMPUTED",
            f"converged={d['diagnostics']['converged']}")
        add("sobol", "output_variance", "", out, d["output_variance"],
            "native squared", "COMPUTED", "")

    sc = rec["sobol_convergence"]
    add("sobol_convergence", "max_total_drift", "", "npv_musd",
        sc["max_total_drift"], "variance share", "COMPUTED",
        f"levels {sc['levels']}, ranking_stable={sc['ranking_stable']}")
    for lvl, tos in sc["total_order_by_level"].items():
        for p, v in tos.items():
            add("sobol_convergence", f"total_order_ST_at_N{lvl}", p,
                "npv_musd", v, "variance share", "COMPUTED", "")

    for p, d in rec["tornado"].items():
        for k in ("low", "high", "swing", "low_delta", "high_delta"):
            add("tornado", k, p, "npv_musd", d[k], "MUSD", "COMPUTED",
                "one-at-a-time at nominal; blind to interaction")
    add("tornado", "base", "", "npv_musd",
        next(iter(rec["tornado"].values()))["base"], "MUSD", "COMPUTED",
        "nominal point NPV")

    for a, v in rec["decision"]["expected_values_musd"].items():
        add("decision", "prior_expected_value", a, "npv_musd", v, "MUSD",
            "COMPUTED", "4096 joint prior draws, seed 11")
    for k, v in rec["decision"]["assumed_polish"].items():
        add("decision", f"polish_{k}", "build_polish", "", v, "native",
            "ASSUMED", "structural assumption of the action set")

    for r in rec["evpi"]:
        add("evpi", "evpi", r["parameter"], "npv_musd", r["evpi_musd"],
            "MUSD", "COMPUTED",
            f"nested MC outer={r['n_outer']} inner={r['n_inner']}")
        add("evpi", "switch_fraction", r["parameter"], "npv_musd",
            r["switch_fraction"], "fraction", "COMPUTED", "")
    for r in rec["evpi_per_cost"]:
        add("evpi_per_cost", "measurement_cost", r["parameter"], "", r["cost"],
            "USD", COST_PROVENANCE[r["parameter"]][0],
            "see measurement_costs_usd section for this cost's basis")
        add("evpi_per_cost", "evpi_per_usd", r["parameter"], "npv_musd",
            r["evpi_per_cost"], "MUSD per USD", "COMPUTED", "")
    for p, c in rec["measurement_costs_usd"].items():
        tag, note = COST_PROVENANCE[p]
        add("measurement_costs_usd", "unit_cost", p, "", c, "USD", tag, note)

    ecv = rec["evpi_convergence"]
    for n_inner, v in ecv["series"]:
        add("evpi_convergence", f"evpi_at_inner_{n_inner}", ecv["parameter"],
            "npv_musd", v, "MUSD", "COMPUTED",
            "nested MC bias falls as inner sample grows")

    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "quantity", "parameter",
                                           "output", "value", "unit", "tag",
                                           "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {len(rows)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
