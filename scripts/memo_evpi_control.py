"""Noise control on the EVPI ranking, because the first run looked like noise.

WHAT PROMPTED THIS. The headline run (scripts/memo_run.py) returned positive
EVPI for five of nine parameters: mass_yield_flot 0.8731, al_sigma_ppm 0.7640,
al_mean_ppm 0.6292, discount_rate 0.5566, and power_price 0.0329 MUSD. The
remaining four (capex_musd, mass_yield_leach, price, reagent_price) came back
at exactly 0.0000. The switch fraction was EXACTLY 0.000 for all nine. Those
two results are in tension. If no resolved value of a parameter ever changes
the chosen action, then the action taken is the same one, its expected payoff
is the prior expected payoff, and EVPI is zero by construction. A positive
number alongside a zero switch fraction is therefore a property of the
estimator, not of the decision, unless something else explains it.

Inspection of the returned fields shows the mechanism. Each call to
ae.agent.decisions.evpi estimates its own baseline with
``max(n_outer * 2, 512)`` draws and a per-parameter seed, so the nine calls
reported baselines spanning 18.11 to 19.63 MUSD. That 1.5 MUSD spread is
larger than every EVPI in the ranking. The estimator then clamps the
difference at zero, which makes the error one-sided: a baseline that happens
to come out low produces an apparently positive EVPI, and a baseline that
comes out high is clamped to 0.0000 and looks like a confident null.

THREE CONTROLS, each a direct measurement rather than an argument:

A. Seed replication. Call the shipped estimator on the same parameter at
   several seeds. If the spread across seeds is comparable to the values
   themselves, the ranking carries no information.

B. Consistent baseline. Re-form the estimator as
   ``resolved_value - baseline_common``, where ``baseline_common`` is one
   high-precision estimate of the prior-best value shared by every parameter,
   instead of nine independent low-precision ones. This removes the
   between-parameter baseline noise without touching the resolved values,
   which the shipped estimator computes at 65,536 evaluations each and are
   the precise half of the calculation.

C. Null calibration. Run control B on a parameter the decision provably
   cannot use. ``dummy`` is drawn from a prior and passed to the value
   function, which ignores it, so its true EVPI is exactly zero. Whatever
   control B returns for ``dummy`` is the estimator's noise floor, and any
   real parameter must clear it to be called non-zero.

Nothing here replaces the shipped estimator. Controls A and C call
ae.agent.decisions.evpi unmodified; control B reuses its own
``resolved_value`` output.
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

from ae.agent.decisions import DecisionProblem, evpi  # noqa: E402
from memo_run import (  # noqa: E402
    ACTIONS,
    N_EVPI_INNER,
    N_EVPI_OUTER,
    priors_from_chain,
    value,
)

SEEDS = (11, 23, 37, 53, 71, 97)
#: Draws for the shared high-precision baseline. 65,536 matches the number of
#: evaluations the shipped estimator spends on ONE resolved value, so the
#: baseline stops being the weakest term in the difference.
N_BASELINE = 65_536


def problem_with_dummy() -> DecisionProblem:
    """The same problem plus a parameter the value function ignores.

    Its true EVPI is exactly zero, which is what makes it a calibration
    target: it is not an approximation of a small effect, it is a known null.
    """
    priors = dict(priors_from_chain())

    def dummy_sampler(rng, n):
        return rng.random(n)

    priors["dummy"] = dummy_sampler

    def value_ignoring_dummy(action: str, params) -> float:
        p = {k: v for k, v in dict(params).items() if k != "dummy"}
        return value(action, p)

    return DecisionProblem(actions=ACTIONS, value=value_ignoring_dummy,
                           priors=priors)


def main() -> None:
    t0 = time.perf_counter()
    out: dict[str, object] = {}
    problem = DecisionProblem(actions=ACTIONS, value=value,
                              priors=priors_from_chain())

    # --- shared high-precision baseline ---------------------------------
    print(f"[baseline] {N_BASELINE} draws ...", flush=True)
    ev_hi = problem.expected_values(n_draws=N_BASELINE, seed=1234)
    best_hi = max(ev_hi, key=lambda k: ev_hi[k])
    out["baseline_common"] = {
        "n_draws": N_BASELINE, "expected_values_musd": ev_hi,
        "prior_best_action": best_hi, "prior_best_value_musd": ev_hi[best_hi]}
    print(f"[baseline] {ev_hi}", flush=True)

    # --- control A: seed replication on the shipped estimator -----------
    params_a = ("mass_yield_flot", "al_sigma_ppm", "al_mean_ppm",
                "discount_rate", "price")
    rep: dict[str, list[dict[str, float]]] = {}
    for p in params_a:
        rows = []
        for sd in SEEDS:
            iv = evpi(problem, p, n_outer=N_EVPI_OUTER, n_inner=N_EVPI_INNER,
                      seed=sd)
            rows.append({"seed": sd, "evpi": iv.evpi,
                         "baseline": iv.baseline_value,
                         "resolved": iv.resolved_value,
                         "switch_fraction": iv.switch_fraction})
            print(f"[A] {p:18s} seed {sd:3d}  evpi {iv.evpi:7.4f}  base "
                  f"{iv.baseline_value:8.4f}  resolved "
                  f"{iv.resolved_value:8.4f}", flush=True)
        rep[p] = rows
        e = np.array([r["evpi"] for r in rows])
        b = np.array([r["baseline"] for r in rows])
        print(f"[A] {p:18s} evpi mean {e.mean():.4f} sd {e.std(ddof=1):.4f} "
              f"range [{e.min():.4f}, {e.max():.4f}] | baseline sd "
              f"{b.std(ddof=1):.4f}", flush=True)
    out["control_a_seed_replication"] = rep
    out["control_a_summary"] = {
        p: {"evpi_mean": float(np.mean([r["evpi"] for r in rows])),
            "evpi_sd": float(np.std([r["evpi"] for r in rows], ddof=1)),
            "evpi_min": float(np.min([r["evpi"] for r in rows])),
            "evpi_max": float(np.max([r["evpi"] for r in rows])),
            "baseline_sd": float(np.std([r["baseline"] for r in rows],
                                        ddof=1)),
            "switch_fraction_max": float(
                np.max([r["switch_fraction"] for r in rows]))}
        for p, rows in rep.items()}

    # --- controls B and C: consistent baseline, incl. a known null ------
    pd = problem_with_dummy()
    base_hi = ev_hi[best_hi]
    rows_b = []
    for p in sorted(pd.priors):
        iv = evpi(pd, p, n_outer=N_EVPI_OUTER, n_inner=N_EVPI_INNER, seed=11)
        rows_b.append({
            "parameter": p,
            "evpi_shipped": iv.evpi,
            "baseline_shipped": iv.baseline_value,
            "resolved": iv.resolved_value,
            "evpi_common_baseline": iv.resolved_value - base_hi,
            "switch_fraction": iv.switch_fraction,
        })
        print(f"[B] {p:18s} shipped {iv.evpi:7.4f}  common-baseline "
              f"{iv.resolved_value - base_hi:+8.4f}  switch "
              f"{iv.switch_fraction:.3f}", flush=True)
    rows_b.sort(key=lambda r: -r["evpi_common_baseline"])
    out["control_bc_common_baseline"] = rows_b
    null = [r for r in rows_b if r["parameter"] == "dummy"][0]
    out["null_calibration"] = null
    print(f"[C] null (dummy, true EVPI exactly 0) common-baseline estimate "
          f"{null['evpi_common_baseline']:+.4f} MUSD", flush=True)

    real = [r for r in rows_b if r["parameter"] != "dummy"]
    out["exceeds_null"] = [
        r["parameter"] for r in real
        if r["evpi_common_baseline"] > abs(null["evpi_common_baseline"])]
    print(f"[C] parameters exceeding the null floor: {out['exceeds_null']}",
          flush=True)

    out["runtime_seconds"] = time.perf_counter() - t0
    path = ROOT / "docs" / "memo_evpi_control.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(f"[done] {out['runtime_seconds']:.1f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
