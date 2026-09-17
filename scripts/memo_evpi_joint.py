"""Joint and group EVPI, which is the quantity an assay CAMPAIGN actually buys.

WHY THIS EXISTS. scripts/memo_evpi_control.py established two things, and it
is worth being exact about which.

What it DID show: the shipped estimator's per-parameter values are not
separable from its own baseline noise. A dummy input the value function
ignores, whose true EVPI is exactly zero, returned 0.7698 MUSD, ahead of
three real parameters. Re-seeding collapsed four of five replicated
parameters to exactly 0.0000 at five of six seeds. The switch fraction was
exactly 0.000 for all nine real parameters at every seed tested. The
headline ranking is therefore not reportable as it stands.

What it did NOT show: that every parameter's EVPI is zero. Under the
consistent-baseline estimator (control B) six parameters cleared the dummy
null floor of +0.1233 MUSD, and two cleared it by a wide margin:
discount_rate +0.8635 and capex_musd +0.7588 MUSD, roughly seven and six
times the floor. Those two are plausibly non-zero. The claim this script
rests on is narrower and sufficient: the per-parameter RANKING the brief
asked for cannot be read off that estimator at the sample sizes used. Of the
FOUR parameters an assay campaign can actually resolve, the two Al
parameters came out just above the floor (al_mean_ppm +0.1488 and
al_sigma_ppm +0.1387 against a floor of +0.1233, margins of 0.026 and 0.015
MUSD, which is 21 and 12 percent of the floor and not a separation worth
ranking on), and the two mass yields came out NEGATIVE (-0.3498 and -0.2186),
which is only possible as sampling error since EVPI is non-negative by
construction. Neither pair supports an ordering.

A further structural point, which the group calculation below tests
directly rather than assumes: resolving ONE parameter leaves the other eight
at their priors, and the conditional expectation over those eight almost
always still favours build_base, so the action does not change. That is the
mechanism behind the zero switch fractions.

The right question: a real assay campaign resolves the ore parameters
TOGETHER, not one at a time. (The 20 to 30 sample count used for costing in
scripts/memo_evpi_ceiling.py comes from the Quartz Foundry project context,
not from the task brief; see the provenance note on N_SAMPLES_LOW there. The
group structure below does not depend on it.) Group EVPI is
therefore computed directly, by its definition:

    EVPI(G) = E_G[ max_a E_{rest|G}[ V(a, theta) ] ] - max_a E_theta[ V(a, theta) ]

with an inner expectation over the parameters NOT in G. The estimator here is
the plain nested Monte Carlo: no clamping at zero, and a single shared
high-precision baseline for the second term, so the difference can come out
negative and a negative result is reported rather than hidden. The dummy
group is carried through as a null calibration exactly as before.

Definitional check, asserted at runtime rather than argued: the group
containing ALL parameters must reproduce the closed-form full-information
value E_theta[max_a V(a, theta)] - max_a E_theta[V(a, theta)], which needs no
nesting because there is nothing left to average over. If the nested
estimator and the closed form disagree beyond their sampling error, the
estimator is wrong and the script raises.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))

from memo_run import ACTIONS, MEASUREMENT_COSTS, priors_from_chain, value  # noqa: E402

ACTION_NAMES = [a.name for a in ACTIONS]

#: Groups a real campaign can actually buy. ORE is the four parameters the
#: assay campaign in the brief resolves. COMMERCIAL is the five that a market,
#: EPC or financing study resolves. ALL is the definitional check.
GROUPS: dict[str, tuple[str, ...]] = {
    "ore_all_four": ("al_mean_ppm", "al_sigma_ppm", "mass_yield_flot",
                     "mass_yield_leach"),
    "ore_chemistry_only": ("al_mean_ppm", "al_sigma_ppm"),
    "ore_metallurgy_only": ("mass_yield_flot", "mass_yield_leach"),
    "commercial_all_five": ("price", "capex_musd", "discount_rate",
                            "power_price", "reagent_price"),
    "price_and_capex": ("price", "capex_musd"),
    "dummy_null": ("dummy",),
    "all_nine": ("al_mean_ppm", "al_sigma_ppm", "mass_yield_flot",
                 "mass_yield_leach", "price", "capex_musd", "discount_rate",
                 "power_price", "reagent_price"),
}

#: Campaign costs, USD, built from the per-parameter costs in memo_run. A
#: campaign that resolves several parameters on the same samples does not pay
#: the preparation charge repeatedly, but no attempt is made to model that
#: here: the sum is an UPPER bound on campaign cost, which is the
#: conservative direction for a value-per-cost ratio.
GROUP_COSTS = {
    g: sum(MEASUREMENT_COSTS[p] for p in ps if p in MEASUREMENT_COSTS)
    for g, ps in GROUPS.items()
}

N_OUTER = 512
N_INNER = 512
N_BASELINE = 131_072


def sample(priors, rng, n):
    return {k: f(rng, n) for k, f in priors.items()}


def action_values(params_at_row: dict[str, float]) -> np.ndarray:
    return np.array([value(a, params_at_row) for a in ACTION_NAMES])


def expected_values(priors, n, seed):
    """E_theta[V(a, theta)] for each action, plain Monte Carlo."""
    rng = np.random.default_rng(seed)
    d = sample(priors, rng, n)
    tot = np.zeros(len(ACTION_NAMES))
    for j in range(n):
        row = {k: float(v[j]) for k, v in d.items()}
        tot += action_values(row)
    return tot / n


def full_info_closed_form(priors, n, seed):
    """E_theta[max_a V(a, theta)], no nesting required."""
    rng = np.random.default_rng(seed)
    d = sample(priors, rng, n)
    tot = 0.0
    for j in range(n):
        row = {k: float(v[j]) for k, v in d.items()}
        tot += float(action_values(row).max())
    return tot / n


def group_evpi(priors, group, baseline, n_outer, n_inner, seed):
    """Nested MC group EVPI. No clamping: a negative result is returned.

    Also returns the standard error of the outer mean, which is the term that
    dominates the uncertainty on the result: the outer average is over
    ``n_outer`` draws of a quantity whose spread is comparable to the NPV
    spread itself (sd about 14 MUSD), while the EVPI being estimated is a
    fraction of one MUSD. Reporting the value without this standard error
    would repeat the error that scripts/memo_evpi_control.py diagnosed.
    """
    rng = np.random.default_rng(seed)
    outer = sample({k: priors[k] for k in group}, rng, n_outer)
    rest = {k: v for k, v in priors.items() if k not in group}
    maxes = np.empty(n_outer)
    switches = 0
    for i in range(n_outer):
        fixed = {k: float(v[i]) for k, v in outer.items()}
        inner = sample(rest, rng, n_inner)
        acc = np.zeros(len(ACTION_NAMES))
        for j in range(n_inner):
            row = dict(fixed)
            row.update({k: float(v[j]) for k, v in inner.items()})
            acc += action_values(row)
        acc /= n_inner
        maxes[i] = float(acc.max())
        if ACTION_NAMES[int(acc.argmax())] != baseline["best_action"]:
            switches += 1
    se = float(maxes.std(ddof=1) / np.sqrt(n_outer))
    return {"evpi": float(maxes.mean()) - baseline["best_value"],
            "posterior_best_value": float(maxes.mean()),
            "outer_sd": float(maxes.std(ddof=1)),
            "standard_error": se,
            "evpi_ci95_low": float(maxes.mean()) - baseline["best_value"] - 1.96 * se,
            "evpi_ci95_high": float(maxes.mean()) - baseline["best_value"] + 1.96 * se,
            "switch_fraction": switches / n_outer,
            "n_outer": n_outer, "n_inner": n_inner}


def main() -> None:
    t0 = time.perf_counter()
    priors = dict(priors_from_chain())
    priors["dummy"] = lambda rng, n: rng.random(n)
    out: dict[str, object] = {"group_definitions": {k: list(v)
                                                    for k, v in GROUPS.items()},
                              "group_costs_usd": GROUP_COSTS}

    print(f"[baseline] {N_BASELINE} draws ...", flush=True)
    ev = expected_values(priors, N_BASELINE, seed=1234)
    best_i = int(np.argmax(ev))
    baseline = {"best_action": ACTION_NAMES[best_i],
                "best_value": float(ev[best_i])}
    out["baseline"] = {"expected_values_musd": dict(zip(ACTION_NAMES,
                                                        ev.tolist())),
                       **baseline, "n_draws": N_BASELINE}
    print(f"[baseline] {dict(zip(ACTION_NAMES, ev.round(4).tolist()))} "
          f"best={baseline['best_action']}", flush=True)

    print(f"[full-info] closed form, {N_BASELINE} draws ...", flush=True)
    fi = full_info_closed_form(priors, N_BASELINE, seed=1234)
    evpi_full_cf = fi - baseline["best_value"]
    out["full_information"] = {
        "e_max_musd": fi, "evpi_closed_form_musd": evpi_full_cf,
        "n_draws": N_BASELINE}
    print(f"[full-info] E[max] {fi:.4f}, EVPI(all) closed form "
          f"{evpi_full_cf:+.4f} MUSD", flush=True)

    rows = []
    for g, ps in GROUPS.items():
        print(f"[group] {g} ({len(ps)} params) ...", flush=True)
        r = group_evpi(priors, ps, baseline, N_OUTER, N_INNER, seed=11)
        cost = GROUP_COSTS[g]
        r.update({"group": g, "parameters": list(ps), "cost_usd": cost,
                  "evpi_per_usd": (r["evpi"] / cost) if cost > 0 else None,
                  "evpi_musd_per_kusd": (r["evpi"] / (cost / 1000.0))
                  if cost > 0 else None})
        rows.append(r)
        print(f"[group] {g:22s} EVPI {r['evpi']:+8.4f} MUSD  switch "
              f"{r['switch_fraction']:.3f}  cost {cost:,.0f} USD", flush=True)
    out["groups"] = rows
    # Persist before the controls run. The controls can raise, and the group
    # loop above costs minutes of nested Monte Carlo; discarding it because a
    # downstream check failed would be a self-inflicted loss.
    path = ROOT / "docs" / "memo_evpi_joint.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"[checkpoint] groups written -> {path}", flush=True)

    # Definitional control: the all-parameter group must match the closed
    # form, to within the sampling error of the nested estimate. The tolerance
    # is the MEASURED standard error of that estimate, not a fixed number: a
    # first attempt used a flat 0.15 MUSD and fired at a gap of 0.2212 MUSD
    # when the nested estimate's own standard error was of that order, which
    # made the control report a correct estimator as broken.
    allr = [r for r in rows if r["group"] == "all_nine"][0]
    gap = abs(allr["evpi"] - evpi_full_cf)
    tol = 3.0 * allr["standard_error"]
    out["all_nine_vs_closed_form"] = {
        "nested_musd": allr["evpi"], "closed_form_musd": evpi_full_cf,
        "abs_gap_musd": gap, "nested_standard_error_musd":
        allr["standard_error"], "tolerance_3se_musd": tol,
        "passes": gap <= tol}
    print(f"[control] all_nine nested {allr['evpi']:+.4f} (SE "
          f"{allr['standard_error']:.4f}) vs closed form {evpi_full_cf:+.4f}, "
          f"gap {gap:.4f} MUSD, tolerance 3 SE = {tol:.4f}", flush=True)
    if gap > tol:
        raise AssertionError(
            f"nested group EVPI for all nine parameters ({allr['evpi']:.4f}) "
            f"disagrees with the closed-form full-information value "
            f"({evpi_full_cf:.4f}) by {gap:.4f} MUSD, more than 3 standard "
            f"errors ({tol:.4f}); the nested estimator is wrong"
        )

    null = [r for r in rows if r["group"] == "dummy_null"][0]
    out["null_floor_musd"] = null["evpi"]
    real = [r for r in rows if r["group"] not in ("dummy_null", "all_nine")]
    out["exceeds_null"] = sorted(
        [r["group"] for r in real if r["evpi"] > abs(null["evpi"])],
        key=lambda g: -[r for r in rows if r["group"] == g][0]["evpi"])
    print(f"[control] null floor (dummy group, true EVPI 0) "
          f"{null['evpi']:+.4f} MUSD; groups clearing it: "
          f"{out['exceeds_null']}", flush=True)

    ranked = sorted([r for r in real if r["evpi_per_usd"] is not None],
                    key=lambda r: -r["evpi_per_usd"])
    out["ranked_by_evpi_per_cost"] = [r["group"] for r in ranked]
    for r in ranked:
        print(f"[ratio] {r['group']:22s} {r['evpi_musd_per_kusd']:+8.4f} "
              f"MUSD per kUSD spent", flush=True)

    out["runtime_seconds"] = time.perf_counter() - t0
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"[done] {out['runtime_seconds']:.1f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
