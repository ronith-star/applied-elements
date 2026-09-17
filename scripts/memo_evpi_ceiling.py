"""The one information number in this problem that IS precisely estimable.

Both prior scripts hit the same wall. Single-parameter EVPI (memo_run.py) is
swamped by per-call baseline noise, and group EVPI (memo_evpi_joint.py)
carries a standard error of 0.6142 MUSD on values that are all smaller than
that, because the outer average is over a quantity whose spread is the NPV
spread (sd 14.3 MUSD) while the EVPI is a fraction of one MUSD. Neither
supports a ranking.

One quantity escapes this. The value of perfect information on ALL
parameters at once needs no nesting:

    EVPI(all) = E_theta[ max_a V(a, theta) ] - max_a E_theta[ V(a, theta) ]

Both terms are plain expectations over the same draws, so both converge at
the usual Monte Carlo rate and, better, the difference can be formed
PER DRAW and averaged, which cancels the shared NPV variance that destroys
the nested estimator:

    d_k = max_a V(a, theta_k) - V(a*, theta_k)

where a* is the prior-best action. E[d] = EVPI(all) exactly, and d_k is zero
on every draw where the prior-best action is already optimal, so its
variance is tiny compared with either term separately. This script uses that
paired form, reports its standard error, and treats EVPI(all) as what it is:
a CEILING. No measurement of any subset can be worth more, because no
information is worth more than perfect information on everything.

That ceiling is what the memo's recommendation rests on, since it does not
require resolving the per-parameter ranking the estimators cannot deliver.
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
N_DRAWS = 200_000
SEEDS = (1234, 4321, 999, 20260917)

#: Cost of the ore assay campaign, USD: the four ore parameters from
#: MEASUREMENT_COSTS. Upper bound, since shared sample preparation is not
#: netted out.
ORE_PARAMS = ("al_mean_ppm", "al_sigma_ppm", "mass_yield_flot",
              "mass_yield_leach")
ORE_CAMPAIGN_COST_USD = sum(MEASUREMENT_COSTS[p] for p in ORE_PARAMS)
#: Sample count used to scale the per-sample costs into a campaign cost.
#: PROVENANCE, stated precisely because this number multiplies the
#: load-bearing cost figure: 20 to 30 samples labelled AE-Q-### comes from the
#: Quartz Foundry PROJECT context ("a representative characterization campaign
#: of 20 to 30 samples across the deposit, labeled AE-Q-###"), NOT from the
#: task brief for this memo and NOT from anything in this repository. Grepping
#: the repo finds AE-Q-### only as test fixture sample_ids (tests/test_phases.py,
#: tests/test_leaching.py), never as a campaign size. Treat the count as
#: ASSUMED for costing purposes: no statistical justification for 20 to 30 has
#: been established here, and the sample size a capability estimate actually
#: requires is itself an open question (ae.plant.yield_cascade.capability
#: returns a confidence interval that narrows with lot count).
N_SAMPLES_LOW, N_SAMPLES_HIGH = 20, 30


def paired_evpi(n_draws: int, seed: int) -> dict[str, float]:
    """EVPI(all) by the paired estimator, with its standard error."""
    priors = priors_from_chain()
    rng = np.random.default_rng(seed)
    d = {k: f(rng, n_draws) for k, f in priors.items()}
    vals = np.empty((n_draws, len(ACTION_NAMES)))
    for j in range(n_draws):
        row = {k: float(v[j]) for k, v in d.items()}
        vals[j, :] = [value(a, row) for a in ACTION_NAMES]
    ev = vals.mean(axis=0)
    star = int(np.argmax(ev))
    diff = vals.max(axis=1) - vals[:, star]
    se = float(diff.std(ddof=1) / np.sqrt(n_draws))
    frac_binding = float((diff > 0).mean())
    return {
        "seed": seed, "n_draws": n_draws,
        "prior_best_action": ACTION_NAMES[star],
        "prior_best_value_musd": float(ev[star]),
        "expected_values_musd": dict(zip(ACTION_NAMES, ev.tolist())),
        "evpi_all_musd": float(diff.mean()),
        "standard_error_musd": se,
        "ci95_low": float(diff.mean()) - 1.96 * se,
        "ci95_high": float(diff.mean()) + 1.96 * se,
        "fraction_of_draws_where_information_binds": frac_binding,
        "evpi_as_pct_of_prior_value": 100.0 * float(diff.mean()) / float(ev[star]),
    }


def main() -> None:
    t0 = time.perf_counter()
    runs = []
    for sd in SEEDS:
        r = paired_evpi(N_DRAWS, sd)
        runs.append(r)
        print(f"[paired] seed {sd:>9d}  EVPI(all) {r['evpi_all_musd']:.4f} "
              f"+/- {r['standard_error_musd']:.4f} MUSD  "
              f"({r['evpi_as_pct_of_prior_value']:.2f}% of prior value, "
              f"binds on {100*r['fraction_of_draws_where_information_binds']:.2f}% "
              f"of draws)", flush=True)

    vals = np.array([r["evpi_all_musd"] for r in runs])
    across = {"mean_musd": float(vals.mean()),
              "sd_across_seeds_musd": float(vals.std(ddof=1)),
              "min_musd": float(vals.min()), "max_musd": float(vals.max()),
              "mean_standard_error_musd": float(
                  np.mean([r["standard_error_musd"] for r in runs]))}
    print(f"[paired] across {len(SEEDS)} seeds: mean {across['mean_musd']:.4f} "
          f"MUSD, sd {across['sd_across_seeds_musd']:.4f}, within-run SE "
          f"{across['mean_standard_error_musd']:.4f}", flush=True)

    # Control: the sd across independent seeds must be consistent with the
    # within-run standard error. If the seed spread were much larger, the
    # reported SE would be understating the true uncertainty, which is exactly
    # the failure mode diagnosed in the single-parameter estimator.
    ratio = (across["sd_across_seeds_musd"]
             / across["mean_standard_error_musd"])
    print(f"[control] seed sd / within-run SE = {ratio:.2f} "
          f"(consistent if of order 1)", flush=True)

    ceiling = across["mean_musd"]
    base = float(np.mean([r["prior_best_value_musd"] for r in runs]))
    camp_lo = ORE_CAMPAIGN_COST_USD * N_SAMPLES_LOW
    camp_hi = ORE_CAMPAIGN_COST_USD * N_SAMPLES_HIGH
    out = {
        "runs": runs, "across_seeds": across,
        "seed_sd_over_within_run_se": ratio,
        "ceiling_musd": ceiling,
        "prior_best_value_musd": base,
        "ceiling_pct_of_prior_value": 100.0 * ceiling / base,
        "ore_campaign": {
            "parameters": list(ORE_PARAMS),
            "per_sample_cost_usd": ORE_CAMPAIGN_COST_USD,
            "n_samples_low": N_SAMPLES_LOW, "n_samples_high": N_SAMPLES_HIGH,
            "campaign_cost_usd_low": camp_lo,
            "campaign_cost_usd_high": camp_hi,
            "ceiling_over_cost_ratio_low": ceiling * 1e6 / camp_hi,
            "ceiling_over_cost_ratio_high": ceiling * 1e6 / camp_lo,
        },
        "runtime_seconds": time.perf_counter() - t0,
    }
    print(f"[ratio] ore campaign {camp_lo:,.0f} to {camp_hi:,.0f} USD "
          f"({N_SAMPLES_LOW} to {N_SAMPLES_HIGH} samples at "
          f"{ORE_CAMPAIGN_COST_USD:,.2f} USD each) against an information "
          f"CEILING of {ceiling*1e6:,.0f} USD: ratio "
          f"{out['ore_campaign']['ceiling_over_cost_ratio_low']:.1f}x to "
          f"{out['ore_campaign']['ceiling_over_cost_ratio_high']:.1f}x",
          flush=True)

    path = ROOT / "docs" / "memo_evpi_ceiling.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"[done] {out['runtime_seconds']:.1f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
