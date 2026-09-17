"""Which measurement to buy next, ranked by what it is worth.

The platform's purpose is not to produce an NPV. It is to decide what to
measure, in what order, on a finite characterization budget. A model whose
output is a single number invites the reader to believe the number; a model
that reports WHICH INPUT IS DRIVING THE UNCERTAINTY tells them what to do
next, which is the only decision available before any ore has been assayed.

The construct here is the expected value of perfect information (EVPI) on a
single parameter, and the definition is the standard one:

.. math::

   \\mathrm{EVPI}_i = \\mathbb{E}_{x_i}\\!\\left[\\max_a
   \\mathbb{E}[V \\mid a, x_i]\\right] - \\max_a \\mathbb{E}[V \\mid a]

In words: what the decision is worth if you learn :math:`x_i` before choosing,
minus what it is worth if you must choose now. The difference is the most any
measurement of :math:`x_i` can be worth, and it is an UPPER BOUND on the value
of a real assay, which is never perfect.

WHY EVPI AND NOT SOBOL, given the platform already computes Sobol indices.
They answer different questions and the distinction is the reason this module
exists. A Sobol total-order index says how much of the OUTPUT VARIANCE a
parameter accounts for. EVPI says how much a DECISION would improve if the
parameter were known. These come apart in a case that matters here: a
parameter can dominate the variance while having zero EVPI, because the
decision is the same at every value it takes. Spending a characterization
budget on that parameter buys a narrower forecast and no better decision. The
converse also occurs, where a low-variance parameter sits exactly on a
decision boundary and knowing it flips the choice.

WHAT THIS MODULE DOES NOT DO. It does not conduct the reasoning. There is no
LLM call here and no autonomous loop. It computes the quantities a decision
loop needs, so that the reasoning is auditable arithmetic rather than a
narrative. The value of information is only as good as the prior it is
computed against, and with no ore characterized those priors are ASSUMED; the
output is therefore a RANKING to argue with, not a budget to execute.

References
----------
Both entries below were written from memory in the first version of this
module and neither was verified at the time, which is a provenance failure of
exactly the kind the platform's tagging scheme exists to prevent: a citation
in committed source carries more weight than one in conversation, because the
next reader has no way to tell it was never checked. Both have since been
resolved against primary metadata and the fields corrected.

Raiffa, H. and Schlaifer, R. (1961) *Applied Statistical Decision Theory*.
Division of Research, Graduate School of Business Administration, Harvard
University, Boston. xxviii + 356 pp. The EVPI construct. No DOI: the 1961
edition predates DOI assignment. Verified by publisher record; the imprint is
the Division of Research rather than the Harvard Business School as first
written.

Howard, R. A. (1966) 'Information value theory'. *IEEE Transactions on
Systems Science and Cybernetics* 2(1):22-26. doi:10.1109/TSSC.1966.300074
Verified against Crossref metadata: author Ronald Howard, 1966, volume 2,
issue 1, pages 22-26, IEEE. Closed access, so no full text was retrieved and
the derivations here are not quoted from it.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Action",
    "DecisionProblem",
    "InformationValue",
    "evpi",
    "rank_measurements",
    "measurement_priority",
]


@dataclass(frozen=True)
class Action:
    """One course of action, and what it costs to take.

    ``name`` is the decision, not a scenario: "build the leach circuit",
    "walk away", "buy Spruce Pine feedstock". ``cost`` is a one-off cost
    incurred on choosing it, in the same units as the value function's output,
    so that an action can be worth taking only if its upside exceeds it.
    """

    name: str
    cost: float = 0.0


@dataclass
class DecisionProblem:
    """A set of actions, a value function, and priors over the unknowns.

    Parameters
    ----------
    actions
        The feasible choices. Must include at least two: an EVPI against a
        single available action is identically zero, because information
        cannot change a decision that has already been made. A problem with
        one action is a forecast, not a decision, and is rejected rather than
        silently returning zeros.
    value
        ``value(action_name, params) -> float``. The decision-maker's payoff,
        typically NPV. Called many times, so it should be cheap.
    priors
        ``name -> sampler``, each ``sampler(rng, n) -> ndarray`` of draws.
        These are the beliefs EVPI is computed against, and they carry the
        whole result: EVPI against a confident prior is small no matter how
        important the parameter is physically, because a confident prior says
        there is little left to learn.
    """

    actions: Sequence[Action]
    value: Callable[[str, Mapping[str, float]], float]
    priors: Mapping[str, Callable[[np.random.Generator, int], np.ndarray]]
    _names: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if len(self.actions) < 2:
            raise ValueError(
                "EVPI needs at least two actions: information has no value "
                "when there is nothing to choose between. Got "
                f"{[a.name for a in self.actions]}"
            )
        names = [a.name for a in self.actions]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate action names: {names}")
        if not self.priors:
            raise ValueError("no priors given, so there is nothing to learn")
        object.__setattr__(self, "_names", names) if False else None
        self._names = names

    # -- baseline -----------------------------------------------------------
    def expected_values(self, n_draws: int = 4096,
                        seed: int = 0) -> dict[str, float]:
        """E[V | a] for each action, under the joint prior."""
        rng = np.random.default_rng(seed)
        draws = {k: np.asarray(f(rng, n_draws), dtype=float)
                 for k, f in self.priors.items()}
        out: dict[str, float] = {}
        for a in self.actions:
            vals = np.empty(n_draws)
            for j in range(n_draws):
                params = {k: float(v[j]) for k, v in draws.items()}
                vals[j] = self.value(a.name, params)
            out[a.name] = float(vals.mean() - a.cost)
        return out

    def best_action_now(self, n_draws: int = 4096,
                        seed: int = 0) -> tuple[str, float]:
        """The action with the highest prior expected value, and that value."""
        ev = self.expected_values(n_draws, seed)
        name = max(ev, key=lambda k: ev[k])
        return name, ev[name]


@dataclass(frozen=True)
class InformationValue:
    """What learning one parameter is worth, and whether it changes anything.

    ``evpi`` is in value units. ``switch_fraction`` is the fraction of
    resolved states in which the best action DIFFERS from the prior-best, and
    it is reported alongside because the two together say something neither
    says alone: an EVPI near zero with a switch fraction near zero means the
    parameter is decision-irrelevant, while an EVPI near zero with a high
    switch fraction means the actions are nearly equal in value, so the choice
    barely matters however it goes.
    """

    parameter: str
    evpi: float
    baseline_value: float
    resolved_value: float
    switch_fraction: float
    prior_best_action: str
    n_outer: int
    n_inner: int

    @property
    def evpi_fraction(self) -> float:
        """EVPI as a fraction of the baseline value, guarded at zero."""
        b = abs(self.baseline_value)
        return self.evpi / b if b > 1e-12 else float("nan")


def evpi(problem: DecisionProblem, parameter: str, *,
         n_outer: int = 256, n_inner: int = 256,
         seed: int = 0) -> InformationValue:
    """Expected value of perfect information on one parameter.

    The estimator is the standard NESTED Monte Carlo: draw the parameter to be
    resolved from its prior (outer loop), and for each such value average the
    payoff over the remaining uncertainty (inner loop) to find the action that
    would then be chosen. The nesting is not an implementation detail that
    could be flattened away: the inner expectation must be conditional on the
    resolved value, and a single flat sample cannot express that.

    Nested Monte Carlo is BIASED UPWARD at small ``n_inner``, and the
    mechanism is worth stating because it is easy to mistake for a real
    result. The outer loop takes a maximum over actions of NOISY inner
    estimates, and the maximum of noisy estimates exceeds the maximum of their
    true values by an amount that grows with the noise. So a small inner
    sample inflates EVPI. ``n_inner`` is therefore a convergence parameter,
    not a speed knob, and :func:`evpi_convergence` below reports the trend so
    the bias is visible rather than assumed away.

    THE BASELINE IS PAIRED WITH THE RESOLVED SAMPLE, and an earlier version
    was not, which was a larger error than the bias above and was misdiagnosed
    as that bias. The baseline came from ``best_action_now`` drawing its own
    independent sample, so the subtraction ``resolved - baseline`` differenced
    two quantities estimated on different draws and did not cancel their
    common sampling error. On a problem with a parameter that shifts every
    action's payoff equally (true EVPI exactly zero, since a constant added to
    all actions cannot move the argmax), with that parameter distributed
    N(0, 1000), the raw estimate was +73.1086 at ``n_outer`` 256, -49.7340 at
    1024 and -27.5788 at 4096: falling as 1/sqrt(n_outer) and completely
    unresponsive to ``n_inner``, which is the signature of outer-sample noise
    rather than of the inner-maximum bias. The clamp at zero made it worse
    than a visible error: negative excursions were hidden as a correct-looking
    zero while positive ones were reported as a large information value on a
    parameter that cannot matter.

    Both terms are now accumulated over the SAME outer and inner draws (common
    random numbers), so the per-action expectation and the max-over-actions
    expectation share every source of noise and the difference is a paired
    estimator. Measured on that test problem AFTER this rewrite, at
    ``n_inner`` 16: 0.167969 at ``n_outer`` 256, 0.168945 at 1024 and 0.164062
    at 4096. Two things changed. The magnitude fell by more than two orders of
    magnitude, and it stopped scaling with ``n_outer``, because what remains
    is no longer outer-sample noise. It is the inner-maximum bias this
    docstring opens by describing, and it responds to ``n_inner`` as that bias
    must: at ``n_outer`` 512, seed 0, the residual falls 2.964844, 0.671875,
    0.136719, 0.009766, 0.000000 across ``n_inner`` 1, 4, 16, 64, 256. When
    every action has an identical payoff the two terms agree exactly, giving
    resolved 66796.4468022350 against baseline 66796.4468022350 and an EVPI of
    0.0 rather than a small residual.

    (A hand-written prototype of the pairing, not this code, gave 0.017578,
    0.007080 and 0.009094 on the same problem at ``n_inner`` 1. Those figures
    are the prototype's and are recorded as such: an earlier draft of this
    docstring presented them as this function's own output, which they were
    not.)

    ``baseline_value`` is therefore the paired baseline, which differs from
    ``problem.best_action_now()`` by sampling error. ``prior_best_action`` is
    the argmax of the paired per-action means, for the same reason.

    Raises
    ------
    KeyError
        If ``parameter`` is not among the priors. Silently returning zero for
        a misspelled name would read as "this measurement is worthless".
    """
    if parameter not in problem.priors:
        raise KeyError(
            f"{parameter!r} is not a prior in this problem; known parameters "
            f"are {sorted(problem.priors)}. A typo must not read as an EVPI "
            "of zero."
        )

    rng = np.random.default_rng(seed)
    others = [k for k in problem.priors if k != parameter]
    costs = {a.name: a.cost for a in problem.actions}

    focus = np.asarray(problem.priors[parameter](rng, n_outer), dtype=float)
    resolved_total = 0.0
    # Per-action totals accumulated over the SAME draws as resolved_total, so
    # the baseline is a paired estimate rather than an independent one.
    action_totals = {a.name: 0.0 for a in problem.actions}
    per_outer_best: list[str] = []

    for i in range(n_outer):
        inner = {k: np.asarray(problem.priors[k](rng, n_inner), dtype=float)
                 for k in others}
        best_v = -np.inf
        best_a = ""
        for a in problem.actions:
            acc = 0.0
            for j in range(n_inner):
                params = {k: float(v[j]) for k, v in inner.items()}
                params[parameter] = float(focus[i])
                acc += problem.value(a.name, params)
            v = acc / n_inner - costs[a.name]
            action_totals[a.name] += v
            if v > best_v:
                best_v, best_a = v, a.name
        resolved_total += best_v
        per_outer_best.append(best_a)

    resolved = resolved_total / n_outer
    action_means = {k: v / n_outer for k, v in action_totals.items()}
    prior_best = max(action_means, key=lambda k: action_means[k])
    baseline = action_means[prior_best]
    switches = sum(1 for a in per_outer_best if a != prior_best)
    # EVPI is non-negative by construction: knowing more cannot hurt a
    # decision-maker who may ignore what they learn. A negative estimate is
    # Monte Carlo noise, so it is clamped, and the clamp is recorded here
    # rather than hidden because a large negative raw value would mean the
    # estimator is misconfigured, not that information is harmful.
    raw = resolved - baseline
    return InformationValue(
        parameter=parameter,
        evpi=max(raw, 0.0),
        baseline_value=baseline,
        resolved_value=resolved,
        switch_fraction=switches / n_outer,
        prior_best_action=prior_best,
        n_outer=n_outer,
        n_inner=n_inner,
    )


def rank_measurements(problem: DecisionProblem, *,
                      n_outer: int = 256, n_inner: int = 256,
                      seed: int = 0) -> list[InformationValue]:
    """EVPI for every parameter, highest first.

    The ranking is the deliverable, not the absolute values: a nested Monte
    Carlo estimate of EVPI carries real uncertainty, but the ORDER is stable
    long before the magnitudes are, and the order is what determines which
    assay to buy first.
    """
    out = [evpi(problem, p, n_outer=n_outer, n_inner=n_inner, seed=seed + i)
           for i, p in enumerate(sorted(problem.priors))]
    return sorted(out, key=lambda iv: iv.evpi, reverse=True)


def measurement_priority(problem: DecisionProblem,
                         costs: Mapping[str, float], *,
                         n_outer: int = 256, n_inner: int = 256,
                         seed: int = 0) -> list[dict[str, float | str]]:
    """Rank measurements by EVPI PER UNIT COST, which is the real question.

    A budget does not buy EVPI, it buys assays, and assays differ by more than
    an order of magnitude in price: a whole-rock XRF is a few tens of dollars,
    a full LA-ICP-MS trace-element suite with fluid-inclusion microthermometry
    is thousands. Ranking by raw EVPI systematically favours the expensive
    measurement, so the ratio is what a characterization plan should be built
    on.

    A measurement with no stated cost is REJECTED rather than defaulted, since
    a default of zero would put it at the top of a ratio ranking.
    """
    missing = sorted(set(problem.priors) - set(costs))
    if missing:
        raise ValueError(
            f"no measurement cost given for {missing}; a missing cost would "
            "default to zero and rank first on any ratio, which is the "
            "opposite of the truth"
        )
    bad = {k: v for k, v in costs.items() if v <= 0}
    if bad:
        raise ValueError(f"measurement costs must be positive, got {bad}")

    ivs = rank_measurements(problem, n_outer=n_outer, n_inner=n_inner,
                            seed=seed)
    rows: list[dict[str, float | str]] = []
    for iv in ivs:
        c = float(costs[iv.parameter])
        rows.append({
            "parameter": iv.parameter,
            "evpi": iv.evpi,
            "cost": c,
            "evpi_per_cost": iv.evpi / c,
            "switch_fraction": iv.switch_fraction,
            "prior_best_action": iv.prior_best_action,
        })
    rows.sort(key=lambda r: float(r["evpi_per_cost"]), reverse=True)
    return rows


def evpi_convergence(problem: DecisionProblem, parameter: str, *,
                     inner_sizes: Sequence[int] = (16, 64, 256),
                     n_outer: int = 128,
                     seed: int = 0) -> list[tuple[int, float]]:
    """EVPI against inner sample size, so the upward bias is visible.

    Returns ``[(n_inner, evpi), ...]``. A decreasing sequence is the expected
    signature of the max-of-noisy-estimates bias converging away. Reporting it
    is the honest alternative to picking one ``n_inner`` and presenting the
    result as a point estimate.
    """
    return [(n, evpi(problem, parameter, n_outer=n_outer, n_inner=n,
                     seed=seed).evpi)
            for n in inner_sizes]
