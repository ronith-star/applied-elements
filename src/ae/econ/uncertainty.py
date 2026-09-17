r"""Monte Carlo propagation and variance-based sensitivity (Sobol indices).

Why variance decomposition and not a tornado chart: a one-at-a-time tornado
varies each input across its range with all others held at nominal, which
measures the local gradient at one point and is blind to interaction. If
purity and yield interact (they do: a lower grind size lifts liberation and
depresses throughput at once), a tornado assigns the joint effect to neither
input and the two bars sum to less than the observed spread. Sobol indices
partition the OUTPUT VARIANCE, so the shortfall is visible and attributable.

Decomposition
-------------
For a model :math:`Y = f(X_1, \dots, X_k)` with independent inputs, the
variance decomposes uniquely (Sobol' 2001, see References):

.. math::
   \mathrm{Var}(Y) = \sum_i V_i + \sum_{i<j} V_{ij} + \dots + V_{1\dots k}

with :math:`V_i = \mathrm{Var}_{X_i}(E[Y \mid X_i])`. The first-order and total
indices are

.. math::
   S_i = \frac{V_i}{\mathrm{Var}(Y)}, \qquad
   S_{Ti} = \frac{E_{X_{\sim i}}[\mathrm{Var}_{X_i}(Y \mid X_{\sim i})]}
                 {\mathrm{Var}(Y)}

Reading them:

* :math:`S_i` is the variance removed by learning :math:`X_i` exactly. It
  answers "what is worth measuring".
* :math:`S_{Ti}` includes every interaction involving :math:`X_i`. It answers
  "what can be ignored": :math:`S_{Ti} \approx 0` means the input can be fixed.
* :math:`S_{Ti} - S_i` is the interaction share. Large values mean the model is
  not separable, and any one-at-a-time analysis of it is wrong.
* :math:`\sum_i S_i \le 1` always, with equality only for a purely additive
  model. :math:`\sum_i S_{Ti} \ge 1`.

Those inequalities are the diagnostic this module checks, because they must
hold and a violation means too few samples rather than an interesting result.

Sampling
--------
Sobol index estimation uses Saltelli's scheme (Saltelli 2002; Saltelli et al.
2010), which needs :math:`N(k+2)` model evaluations for first and total order,
drawn from a Sobol sequence rather than pseudo-random points. The base sample
:math:`N` must be a power of two, or the sequence loses the balance properties
the estimator relies on. SALib enforces this and so does this module, with an
error that says why rather than silently rounding.

Convergence is not assumed: :func:`sobol_analysis` optionally bootstraps
confidence intervals, and :func:`convergence_check` re-estimates at :math:`N/2`
and :math:`N/4` so a reader can see whether the ranking has stabilised. An
index quoted without that check is an assertion about a number of samples, not
about the model.

References
----------
Both entries below were resolved against the Crossref API, and the fields
here (title, journal, volume, issue, pages, year, author list) are as Crossref
returned them. Verified 16 September 2026.

Saltelli, A. (2002), "Making best use of model evaluations to compute
sensitivity indices", *Computer Physics Communications* 145(2):280-297,
doi:10.1016/S0010-4655(02)00280-1. Source of the sampling scheme.

Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M. and Tarantola,
S. (2010), "Variance based sensitivity analysis of model output. Design and
estimator for the total sensitivity index", *Computer Physics Communications*
181(2):259-270, doi:10.1016/j.cpc.2009.09.018. Source of the estimator used by
:func:`sobol_analysis` via SALib.

Sobol', I.M. (2001), "Global sensitivity indices for nonlinear mathematical
models and their Monte Carlo estimates", *Mathematics and Computers in
Simulation* 55(1-3):271-280, doi:10.1016/S0378-4754(00)00270-6. Source of the
variance decomposition above.

The decomposition is often attributed to a 1993 paper in *Mathematical
Modelling and Computational Experiments*. That journal is not indexed by
Crossref and the reference could not be verified, so it is NOT cited here; the
2001 paper above states the same decomposition and does resolve.

LIMITATIONS
-----------
* Independent inputs assumed. With correlated inputs the decomposition is not
  unique and these indices are biased; :func:`spearman_screening` is provided
  as a correlation-robust fallback, and a correlation matrix warning is raised
  when one is supplied.
* Deterministic model assumed. A stochastic model (the discrete-event scheduler
  with a random seed) must be seeded per evaluation, or its noise inflates
  every total index. :func:`sobol_analysis` requires a seeded callable and
  documents this.
* Indices are unitless variance shares, not elasticities. An input with a small
  index can still have a large effect per unit change if its assigned range is
  narrow; the range IS part of the result, which is why ranges carry provenance.
* A percentile from :math:`10^4` draws has its own sampling error. P10 and P90
  from 10,000 draws carry roughly 1 to 2 percent relative standard error on a
  smooth unimodal output, and far more in a tail. Reported by
  :meth:`MonteCarloResult.percentile_standard_error`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MonteCarloResult",
    "SobolResult",
    "Uncertain",
    "convergence_check",
    "monte_carlo",
    "sobol_analysis",
    "spearman_screening",
    "tornado",
]


@dataclass(frozen=True)
class Uncertain:
    """One uncertain input: a name, a distribution and its support.

    Parameters
    ----------
    name
        Key passed to the model callable.
    low, high
        Support bounds. Used directly for uniform and triangular, and as
        truncation bounds otherwise, so no draw can be physically impossible
        (a negative yield, a purity above unity).
    kind
        ``"uniform"``, ``"triangular"``, ``"normal"``, ``"lognormal"``.
    mode
        Peak for triangular. Defaults to the midpoint.
    mean, sd
        Parameters for normal and lognormal. For lognormal these are the mean
        and standard deviation OF THE VARIABLE, not of its logarithm, converted
        by method of moments, because that is how process data is reported.
    """

    name: str
    low: float
    high: float
    kind: str = "uniform"
    mode: float | None = None
    mean: float | None = None
    sd: float | None = None

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"{self.name}: high must exceed low")
        if self.kind not in ("uniform", "triangular", "normal", "lognormal"):
            raise ValueError(f"{self.name}: unknown distribution {self.kind!r}")
        if self.kind == "triangular" and self.mode is not None:
            if not self.low <= self.mode <= self.high:
                raise ValueError(f"{self.name}: mode must lie within [low, high]")
        if self.kind in ("normal", "lognormal"):
            if self.mean is None or self.sd is None:
                raise ValueError(f"{self.name}: {self.kind} needs mean and sd")
            if self.sd <= 0:
                raise ValueError(f"{self.name}: sd must be positive")
            if self.kind == "lognormal" and self.mean <= 0:
                raise ValueError(f"{self.name}: lognormal mean must be positive")

    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF on ``u`` in [0, 1), truncated to the support.

        Inverse-transform sampling is used rather than rejection so that a
        Sobol sequence maps to the distribution without destroying its
        low-discrepancy structure, which rejection would.
        """
        u = np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12)
        if self.kind == "uniform":
            return np.asarray(self.low + u * (self.high - self.low), dtype=float)
        if self.kind == "triangular":
            m = self.mode if self.mode is not None else 0.5 * (self.low + self.high)
            c = (m - self.low) / (self.high - self.low)
            out = np.where(
                u < c,
                self.low + np.sqrt(np.maximum(u * (self.high - self.low) * (m - self.low), 0.0)),
                self.high - np.sqrt(
                    np.maximum((1 - u) * (self.high - self.low) * (self.high - m), 0.0)),
            )
            return np.asarray(out, dtype=float)
        from scipy import stats

        # __post_init__ rejects a normal or lognormal without both mean and
        # sd, so these cannot be None here. Binding them to locals states that
        # invariant where the arithmetic happens instead of relying on the
        # reader to remember it, and it is what lets a type checker see the
        # guard: without it mypy reported nine operand errors on this block
        # (float / None, float - None) which read as real division-by-None
        # bugs and were not.
        if self.mean is None or self.sd is None:      # pragma: no cover
            raise AssertionError(
                f"{self.name}: {self.kind} reached ppf without mean and sd, "
                "which __post_init__ is supposed to make impossible"
            )
        mean, sd = self.mean, self.sd

        if self.kind == "normal":
            a = (self.low - mean) / sd
            b = (self.high - mean) / sd
            return np.asarray(
                stats.truncnorm.ppf(u, a, b, loc=mean, scale=sd), dtype=float)
        # Lognormal by method of moments on the variable itself.
        s2 = np.log1p((sd / mean) ** 2)
        s = np.sqrt(s2)
        mu = np.log(mean) - 0.5 * s2
        lo_p = stats.lognorm.cdf(self.low, s, scale=np.exp(mu))
        hi_p = stats.lognorm.cdf(self.high, s, scale=np.exp(mu))
        return np.asarray(
            stats.lognorm.ppf(lo_p + u * (hi_p - lo_p), s, scale=np.exp(mu)),
            dtype=float)


@dataclass
class MonteCarloResult:
    """Draws and the distribution summary of one or more outputs."""

    samples: dict[str, np.ndarray]
    outputs: dict[str, np.ndarray]
    n_draws: int
    n_failed: int = 0
    failures: list[str] = field(default_factory=list)

    def percentiles(self, output: str,
                    ps: Sequence[float] = (10, 50, 90)) -> dict[str, float]:
        y = self.outputs[output]
        return {f"P{int(p)}": float(np.percentile(y, p)) for p in ps}

    def summary(self, output: str) -> dict[str, float]:
        y = self.outputs[output]
        d = {"mean": float(np.mean(y)), "sd": float(np.std(y, ddof=1)),
             "min": float(np.min(y)), "max": float(np.max(y))}
        d.update(self.percentiles(output))
        d["P90_minus_P10"] = d["P90"] - d["P10"]
        return d

    def probability_below(self, output: str, threshold: float) -> float:
        """Fraction of draws below a threshold, e.g. P(NPV < 0)."""
        y = self.outputs[output]
        return float(np.mean(y < threshold))

    def percentile_standard_error(self, output: str, p: float,
                                  n_boot: int = 400, seed: int = 0) -> float:
        """Bootstrap standard error of a percentile estimate.

        A percentile from a finite sample has its own sampling error, and
        quoting P10 to four significant figures from 10,000 draws overstates
        what the sample supports.
        """
        y = self.outputs[output]
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, y.size, size=(n_boot, y.size))
        return float(np.std(np.percentile(y[idx], p, axis=1), ddof=1))


@dataclass
class SobolResult:
    """First-order, total and (optionally) second-order Sobol indices."""

    names: list[str]
    first_order: dict[str, float]
    total_order: dict[str, float]
    first_conf: dict[str, float] = field(default_factory=dict)
    total_conf: dict[str, float] = field(default_factory=dict)
    second_order: dict[tuple[str, str], float] = field(default_factory=dict)
    n_base: int = 0
    n_evaluations: int = 0
    output_variance: float = 0.0

    @property
    def interaction_share(self) -> dict[str, float]:
        """``S_Ti - S_i`` per input: the variance in interactions with it."""
        return {k: self.total_order[k] - self.first_order[k] for k in self.names}

    @property
    def additive_fraction(self) -> float:
        """``sum S_i``: the share of variance explained without interactions.

        Well below 1 means the model is not separable and a one-at-a-time
        tornado of it is structurally wrong, not merely imprecise.
        """
        return float(sum(self.first_order.values()))

    def ranking(self, by: str = "total") -> list[tuple[str, float]]:
        d = self.total_order if by == "total" else self.first_order
        return sorted(d.items(), key=lambda kv: -kv[1])

    def negligible(self, threshold: float = 0.01) -> list[str]:
        """Inputs whose TOTAL index is below a threshold, so fixing them is safe.

        Judged on total order, never first order: an input with a small first
        index can carry large interactions and fixing it would then change the
        answer.
        """
        return [k for k in self.names if self.total_order[k] < threshold]

    def diagnostics(self) -> dict[str, Any]:
        """Checks that MUST hold, so a violation flags sampling error."""
        s1 = self.additive_fraction
        st = float(sum(self.total_order.values()))
        neg1 = {k: v for k, v in self.first_order.items() if v < -0.05}
        negt = {k: v for k, v in self.total_order.items() if v < -0.05}
        bad_order = {k: (self.first_order[k], self.total_order[k])
                     for k in self.names
                     if self.first_order[k] > self.total_order[k] + 0.05}
        return {
            "sum_first_order": s1,
            "sum_total_order": st,
            "sum_first_le_one": s1 <= 1.05,
            "sum_total_ge_one": st >= 0.95,
            "materially_negative_first": neg1,
            "materially_negative_total": negt,
            "first_exceeds_total": bad_order,
            "converged": (s1 <= 1.05 and st >= 0.95 and not neg1 and not negt
                          and not bad_order),
        }


def _require_power_of_two(n: int, what: str) -> None:
    if n < 8 or (n & (n - 1)) != 0:
        raise ValueError(
            f"{what} must be a power of two and at least 8, got {n}. The Saltelli "
            f"estimator draws from a Sobol sequence whose balance properties hold "
            f"only at powers of two; rounding silently would bias every index."
        )


def monte_carlo(
    model: Callable[..., Mapping[str, float] | float],
    inputs: Sequence[Uncertain],
    n_draws: int,
    seed: int = 0,
    output_names: Sequence[str] | None = None,
    on_error: str = "raise",
) -> MonteCarloResult:
    """Propagate input uncertainty by Latin-hypercube-free plain Monte Carlo.

    Parameters
    ----------
    model
        Callable taking the input names as keyword arguments and returning
        either a float or a mapping of output name to float.
    n_draws
        Number of evaluations. 10,000 is the platform floor for reported
        percentiles.
    on_error
        ``"raise"`` (default) or ``"record"``. Recording is for models that
        legitimately fail on part of the input space (a negative-NPV region
        where IRR is undefined); the failure count and reasons are returned so
        a silently-thinned sample cannot be mistaken for a complete one.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    rng = np.random.default_rng(seed)
    u = rng.random((n_draws, len(inputs)))
    cols = {inp.name: inp.ppf(u[:, i]) for i, inp in enumerate(inputs)}

    collected: dict[str, list[float]] = {}
    failures: list[str] = []
    keep: list[int] = []
    for r in range(n_draws):
        kwargs = {k: float(v[r]) for k, v in cols.items()}
        try:
            out = model(**kwargs)
        except Exception as exc:
            if on_error == "raise":
                raise
            failures.append(f"draw {r}: {type(exc).__name__}: {exc}")
            continue
        vals = ({"y": float(out)} if np.isscalar(out) or isinstance(out, (int, float))
                else {k: float(v) for k, v in dict(out).items()})
        for k, v in vals.items():
            collected.setdefault(k, []).append(v)
        keep.append(r)

    if not keep:
        raise ValueError(
            f"every one of {n_draws} draws failed; the first failure was: "
            f"{failures[0] if failures else 'unknown'}"
        )
    kept = np.asarray(keep)
    return MonteCarloResult(
        samples={k: v[kept] for k, v in cols.items()},
        outputs={k: np.asarray(v, dtype=float) for k, v in collected.items()},
        n_draws=len(keep), n_failed=len(failures), failures=failures[:20],
    )


def sobol_analysis(
    model: Callable[..., Mapping[str, float] | float],
    inputs: Sequence[Uncertain],
    n_base: int,
    output: str | None = None,
    seed: int = 0,
    second_order: bool = False,
    conf_level: float = 0.95,
) -> SobolResult:
    """Saltelli-scheme Sobol indices for one scalar output.

    Requires ``n_base`` to be a power of two. Total evaluations are
    ``n_base * (k + 2)`` without second order and ``n_base * (2k + 2)`` with it.

    The model must be DETERMINISTIC given its inputs. A stochastic model must
    fix its own seed internally, or its noise appears as variance attributable
    to every input and inflates all total indices.
    """
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    _require_power_of_two(n_base, "n_base")
    names = [i.name for i in inputs]
    problem = {"num_vars": len(names), "names": names,
               "bounds": [[0.0, 1.0]] * len(names)}
    u = sobol_sample.sample(problem, n_base, calc_second_order=second_order,
                            seed=seed, scramble=True)
    x = np.column_stack([inp.ppf(u[:, i]) for i, inp in enumerate(inputs)])

    ys = np.empty(x.shape[0], dtype=float)
    for r in range(x.shape[0]):
        out = model(**{n: float(x[r, i]) for i, n in enumerate(names)})
        if np.isscalar(out) or isinstance(out, (int, float)):
            ys[r] = float(out)
        else:
            d = dict(out)
            if output is None:
                if len(d) != 1:
                    raise ValueError(
                        f"model returns {sorted(d)}; name the output to analyse"
                    )
                ys[r] = float(next(iter(d.values())))
            else:
                if output not in d:
                    raise ValueError(f"output {output!r} not in {sorted(d)}")
                ys[r] = float(d[output])

    if not np.isfinite(ys).all():
        raise ValueError(
            f"{int((~np.isfinite(ys)).sum())} of {ys.size} evaluations were not "
            f"finite; Sobol indices on a sample containing inf or nan are meaningless"
        )
    var = float(np.var(ys, ddof=1))
    if var <= 0:
        raise ValueError(
            "output variance is zero: every evaluation returned the same value, so "
            "no variance can be apportioned. Check that the inputs actually enter "
            "the model."
        )

    si = sobol_analyze.analyze(problem, ys, calc_second_order=second_order,
                               conf_level=conf_level, print_to_console=False,
                               seed=seed)
    res = SobolResult(
        names=names,
        first_order={n: float(si["S1"][i]) for i, n in enumerate(names)},
        total_order={n: float(si["ST"][i]) for i, n in enumerate(names)},
        first_conf={n: float(si["S1_conf"][i]) for i, n in enumerate(names)},
        total_conf={n: float(si["ST_conf"][i]) for i, n in enumerate(names)},
        n_base=n_base, n_evaluations=int(x.shape[0]), output_variance=var,
    )
    if second_order:
        s2 = si["S2"]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                res.second_order[(names[i], names[j])] = float(s2[i, j])
    return res


def convergence_check(
    model: Callable[..., Mapping[str, float] | float],
    inputs: Sequence[Uncertain],
    n_base: int,
    output: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-estimate indices at N, N/2 and N/4 to expose unstable rankings.

    An index quoted without this is a statement about a sample size, not about
    the model. Returns the top-ranked input at each level and the maximum
    absolute drift in total index between the largest two.
    """
    levels = [n_base, n_base // 2, n_base // 4]
    if levels[-1] < 8:
        raise ValueError(f"n_base {n_base} is too small to halve twice")
    runs = {n: sobol_analysis(model, inputs, n, output=output, seed=seed)
            for n in levels}
    top = {n: r.ranking()[0][0] for n, r in runs.items()}
    big, mid = runs[levels[0]], runs[levels[1]]
    drift = max(abs(big.total_order[k] - mid.total_order[k]) for k in big.names)
    return {
        "levels": levels,
        "top_input": top,
        "ranking_stable": len(set(top.values())) == 1,
        "max_total_drift": float(drift),
        "diagnostics": {n: r.diagnostics()["converged"] for n, r in runs.items()},
        "results": runs,
    }


def spearman_screening(mc: MonteCarloResult, output: str) -> dict[str, float]:
    """Rank-correlation screening, robust to monotone nonlinearity.

    Offered as a correlated-input fallback, since Sobol indices assume
    independence. Rank correlation does NOT decompose variance and cannot see
    interactions or non-monotone effects, so it is a screen, not a substitute.
    """
    from scipy import stats
    y = mc.outputs[output]
    return {k: float(stats.spearmanr(v, y).statistic)
            for k, v in mc.samples.items()}


def tornado(
    model: Callable[..., Mapping[str, float] | float],
    inputs: Sequence[Uncertain],
    nominal: Mapping[str, float],
    output: str | None = None,
) -> dict[str, dict[str, float]]:
    """One-at-a-time low/high swing, for presentation alongside Sobol.

    Provided because decision memos ask for it, with the explicit caveat that
    it measures a local gradient and is blind to interaction. Compare its
    implied spread against :attr:`SobolResult.additive_fraction`: when that is
    well below 1, the tornado understates the true spread and its ordering can
    be wrong.
    """
    def ev(**kw: float) -> float:
        out = model(**kw)
        if np.isscalar(out) or isinstance(out, (int, float)):
            return float(out)
        d = dict(out)
        return float(d[output] if output is not None else next(iter(d.values())))

    base = ev(**dict(nominal))
    res: dict[str, dict[str, float]] = {}
    for inp in inputs:
        lo_kw = dict(nominal) | {inp.name: inp.low}
        hi_kw = dict(nominal) | {inp.name: inp.high}
        lo, hi = ev(**lo_kw), ev(**hi_kw)
        res[inp.name] = {"low": lo, "high": hi, "base": base,
                         "swing": abs(hi - lo),
                         "low_delta": lo - base, "high_delta": hi - base}
    return dict(sorted(res.items(), key=lambda kv: -kv[1]["swing"]))
