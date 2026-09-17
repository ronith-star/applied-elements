"""Monte Carlo and Sobol: analytic validation, diagnostics, interaction detection."""
import math

import numpy as np
import pytest

from ae.econ.uncertainty import (
    Uncertain,
    convergence_check,
    monte_carlo,
    sobol_analysis,
    spearman_screening,
    tornado,
)

# --- Ishigami: the standard Sobol test function, with ANALYTIC indices -------
# f = sin(x1) + a sin^2(x2) + b x3^4 sin(x1), xi ~ U(-pi, pi), a=7, b=0.1.
# Analytic (Saltelli et al. 2008): V = a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2
#   V1 = 0.5(1 + b pi^4/5)^2, V2 = a^2/8, V3 = 0, V13 = b^2 pi^8/18 - ...
# x3 has ZERO first-order effect but a LARGE total effect, which is exactly the
# case a tornado chart cannot represent.
_A, _B = 7.0, 0.1


def ishigami(x1: float, x2: float, x3: float) -> float:
    return np.sin(x1) + _A * np.sin(x2) ** 2 + _B * (x3 ** 4) * np.sin(x1)


def ishigami_analytic():
    v1 = 0.5 * (1 + _B * np.pi ** 4 / 5) ** 2
    v2 = _A ** 2 / 8
    v13 = (_B ** 2) * (np.pi ** 8) / 18 - (_B ** 2) * (np.pi ** 8) / 50
    v = v1 + v2 + v13
    return {"S1": {"x1": v1 / v, "x2": v2 / v, "x3": 0.0},
            "ST": {"x1": (v1 + v13) / v, "x2": v2 / v, "x3": v13 / v},
            "V": v}


def ish_inputs():
    return [Uncertain(n, -np.pi, np.pi, "uniform") for n in ("x1", "x2", "x3")]


@pytest.mark.benchmark
def test_sobol_reproduces_ishigami_analytic_indices():
    """Validation against a function with known indices. Errors REPORTED."""
    exact = ishigami_analytic()
    r = sobol_analysis(ishigami, ish_inputs(), n_base=4096, seed=1)
    print(f"\nIshigami, N=4096, {r.n_evaluations} evaluations")
    print(f"  variance: analytic {exact['V']:.4f}, sampled {r.output_variance:.4f}, "
          f"error {abs(r.output_variance / exact['V'] - 1) * 100:.2f} percent")
    for k in ("x1", "x2", "x3"):
        print(f"  {k}: S1 {r.first_order[k]:+.4f} (exact {exact['S1'][k]:+.4f})  "
              f"ST {r.total_order[k]:.4f} (exact {exact['ST'][k]:.4f})")
    assert abs(r.output_variance / exact["V"] - 1) < 0.05
    for k in ("x1", "x2"):
        assert abs(r.first_order[k] - exact["S1"][k]) < 0.03
    # x3's first-order effect is exactly zero.
    assert abs(r.first_order["x3"]) < 0.03
    for k in ("x1", "x2", "x3"):
        assert abs(r.total_order[k] - exact["ST"][k]) < 0.05


@pytest.mark.benchmark
def test_zero_first_order_with_large_total_is_the_case_tornado_misses():
    """x3 has S1 = 0 and ST = 0.24: a one-at-a-time analysis at nominal would
    report x3 as irrelevant, yet fixing it changes a quarter of the variance.
    This is the structural argument for variance decomposition."""
    # The docstring's 0.24 DERIVED from Ishigami's closed form, not restated.
    # With a = 7, b = 0.1 and x ~ U(-pi, pi):
    #   V   = a^2/8 + b pi^4/5 + b^2 pi^8/18 + 1/2
    #   V3  = 0                     (no main effect)
    #   V13 = b^2 pi^8 (1/18 - 1/50)
    #   ST3 = (V3 + V13) / V
    a_, b_ = 7.0, 0.1
    p4_, p8_ = math.pi ** 4, math.pi ** 8
    V_ = a_ * a_ / 8 + b_ * p4_ / 5 + b_ * b_ * p8_ / 18 + 0.5
    V13_ = b_ * b_ * p8_ * (1 / 18 - 1 / 50)
    st3_exact = (0.0 + V13_) / V_
    assert st3_exact == pytest.approx(0.2437, abs=1e-4)
    assert round(st3_exact, 2) == 0.24, "the docstring's 0.24 is this rounded"
    r = sobol_analysis(ishigami, ish_inputs(), n_base=4096, seed=3)
    assert abs(r.first_order["x3"]) < 0.03
    assert r.total_order["x3"] == pytest.approx(st3_exact, abs=0.03), \
        "the ESTIMATOR must recover the analytic ST3, not merely exceed a floor"
    assert r.interaction_share["x3"] > 0.15
    # And the tornado does indeed miss it: at nominal x1 = 0, sin(x1) = 0 so the
    # x3 term vanishes entirely and its swing is zero.
    t = tornado(ishigami, ish_inputs(), {"x1": 0.0, "x2": 0.0, "x3": 0.0})
    assert t["x3"]["swing"] == pytest.approx(0.0, abs=1e-9)
    print(f"\ntornado swing for x3 at nominal: {t['x3']['swing']:.6f}; "
          f"Sobol total index: {r.total_order['x3']:.4f}")


def test_additive_fraction_detects_non_separability():
    """sum(S_i) well below 1 means the model is not separable. Ishigami is
    about 0.76 additive; a purely additive model returns about 1.0."""
    # The docstring's 0.76 DERIVED from the closed form, not restated:
    # sum(S1_i) = (V1 + V2 + V3) / V with V3 = 0.
    a_, b_ = 7.0, 0.1
    p4_, p8_ = math.pi ** 4, math.pi ** 8
    V_ = a_ * a_ / 8 + b_ * p4_ / 5 + b_ * b_ * p8_ / 18 + 0.5
    V1_ = 0.5 * (1 + b_ * p4_ / 5) ** 2
    V2_ = a_ * a_ / 8
    additive_exact = (V1_ + V2_) / V_
    assert additive_exact == pytest.approx(0.7563, abs=1e-4)
    assert round(additive_exact, 2) == 0.76
    assert additive_exact < 1.0, "non-separable by construction"
    r = sobol_analysis(ishigami, ish_inputs(), n_base=2048, seed=5)
    assert r.additive_fraction == pytest.approx(additive_exact, abs=0.04), \
        "the ESTIMATOR must recover the analytic additive fraction"

    def additive(a: float, b: float, c: float) -> float:
        return 2 * a + 3 * b + 4 * c

    ins = [Uncertain(n, 0.0, 1.0) for n in ("a", "b", "c")]
    ra = sobol_analysis(additive, ins, n_base=2048, seed=5)
    assert ra.additive_fraction == pytest.approx(1.0, abs=0.05)
    assert max(ra.interaction_share.values()) < 0.05


@pytest.mark.golden
def test_linear_model_indices_match_closed_form():
    """For Y = sum(c_i X_i) with independent X_i, S_i = c_i^2 Var(X_i)/Var(Y).
    With c = (2, 3, 4) and all X ~ U(0,1) of equal variance
    (4 + 9 + 16 = 29):
      weights 4, 9, 16 of 29 -> 4/29 = 0.1379, 9/29 = 0.3103, 16/29 = 0.5517."""
    def f(a: float, b: float, c: float) -> float:
        return 2 * a + 3 * b + 4 * c

    ins = [Uncertain(n, 0.0, 1.0) for n in ("a", "b", "c")]
    r = sobol_analysis(f, ins, n_base=2048, seed=7)
    exact = {"a": 4 / 29, "b": 9 / 29, "c": 16 / 29}
    for k, v in exact.items():
        assert r.first_order[k] == pytest.approx(v, abs=0.02)
        assert r.total_order[k] == pytest.approx(v, abs=0.02)
    # All three closed-form weights, as quoted in the docstring.
    assert 4 + 9 + 16 == 29
    assert exact["a"] == pytest.approx(0.1379, abs=1e-4)
    assert exact["b"] == pytest.approx(0.3103, abs=1e-4)
    assert exact["c"] == pytest.approx(0.5517, abs=1e-4)


def test_diagnostics_flag_a_converged_run():
    r = sobol_analysis(ishigami, ish_inputs(), n_base=2048, seed=11)
    d = r.diagnostics()
    assert d["sum_first_le_one"] and d["sum_total_ge_one"]
    assert not d["materially_negative_first"]
    assert not d["first_exceeds_total"]
    assert d["converged"]


def test_negligible_is_judged_on_total_not_first_order():
    """An input with a tiny first index can carry large interactions, so fixing
    it on first-order evidence would change the answer. x3 must NOT be called
    negligible despite S1 = 0."""
    r = sobol_analysis(ishigami, ish_inputs(), n_base=2048, seed=13)
    assert "x3" not in r.negligible(threshold=0.05)

    def f(a: float, b: float) -> float:
        return a + 1e-6 * b

    ins = [Uncertain("a", 0.0, 1.0), Uncertain("b", 0.0, 1.0)]
    r2 = sobol_analysis(f, ins, n_base=1024, seed=13)
    assert r2.negligible(threshold=0.01) == ["b"]


def test_n_base_must_be_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        sobol_analysis(ishigami, ish_inputs(), n_base=1000)
    with pytest.raises(ValueError, match="power of two"):
        sobol_analysis(ishigami, ish_inputs(), n_base=4)


def test_second_order_indices_locate_the_interacting_pair():
    """Ishigami's interaction is between x1 and x3 specifically, not x1-x2."""
    r = sobol_analysis(ishigami, ish_inputs(), n_base=2048, seed=17,
                       second_order=True)
    s13 = r.second_order[("x1", "x3")]
    s12 = r.second_order[("x1", "x2")]
    assert s13 > 0.15
    assert abs(s12) < 0.05
    print(f"\nsecond order: S13 {s13:.4f}, S12 {s12:+.4f}")


def test_convergence_check_reports_stability():
    c = convergence_check(ishigami, ish_inputs(), n_base=2048, seed=19)
    assert c["levels"] == [2048, 1024, 512]
    assert c["ranking_stable"]
    assert c["max_total_drift"] < 0.15
    with pytest.raises(ValueError, match="too small to halve"):
        convergence_check(ishigami, ish_inputs(), n_base=16)


def test_zero_variance_output_raises_rather_than_dividing_by_zero():
    def const(a: float, b: float) -> float:
        return 42.0
    ins = [Uncertain("a", 0.0, 1.0), Uncertain("b", 0.0, 1.0)]
    with pytest.raises(ValueError, match="output variance is zero"):
        sobol_analysis(const, ins, n_base=64)


def test_non_finite_output_raises():
    def bad(a: float) -> float:
        return float("inf") if a > 0.5 else a
    with pytest.raises(ValueError, match="not\\s+finite"):
        sobol_analysis(bad, [Uncertain("a", 0.0, 1.0)], n_base=64)


# --- Monte Carlo -------------------------------------------------------------
@pytest.mark.golden
def test_monte_carlo_recovers_known_moments():
    """Sum of three U(0,1): mean 1.5, variance 3/12 = 0.25, sd 0.5.

    A single U(0,1) has variance 1/12; three independent draws sum to variance
    3 x (1/12) = 0.25 and sd 0.5. Both steps asserted below.
    """
    def f(a: float, b: float, c: float) -> float:
        return a + b + c
    # The variance chain, stated in the docstring and checked here.
    var_one = 1.0 / 12.0
    assert 3 * var_one == pytest.approx(0.25, abs=1e-12)
    assert (3 * var_one) ** 0.5 == pytest.approx(0.5, abs=1e-12)

    ins = [Uncertain(n, 0.0, 1.0) for n in ("a", "b", "c")]
    mc = monte_carlo(f, ins, n_draws=20000, seed=2)
    s = mc.summary("y")
    assert s["mean"] == pytest.approx(1.5, abs=0.02)
    assert s["sd"] == pytest.approx(0.5, abs=0.01)
    assert s["P50"] == pytest.approx(1.5, abs=0.02)


def test_percentiles_and_probability_below():
    def f(a: float) -> float:
        return a
    mc = monte_carlo(f, [Uncertain("a", 0.0, 100.0)], n_draws=20000, seed=4)
    p = mc.percentiles("y")
    assert p["P10"] == pytest.approx(10.0, abs=1.0)
    assert p["P90"] == pytest.approx(90.0, abs=1.0)
    assert mc.probability_below("y", 25.0) == pytest.approx(0.25, abs=0.02)


def test_percentile_standard_error_is_reported():
    """A percentile from a finite sample carries sampling error, and quoting P10
    to four figures from 10,000 draws overstates the sample."""
    def f(a: float) -> float:
        return a
    mc = monte_carlo(f, [Uncertain("a", 0.0, 100.0)], n_draws=10000, seed=6)
    se = mc.percentile_standard_error("y", 10)
    assert 0.0 < se < 2.0
    # A smaller sample has a larger standard error.
    small = monte_carlo(f, [Uncertain("a", 0.0, 100.0)], n_draws=500, seed=6)
    assert small.percentile_standard_error("y", 10) > se


def test_multi_output_models():
    def f(a: float, b: float) -> dict[str, float]:
        return {"npv": 10 * a - 5 * b, "cost": 100 + 20 * b}
    ins = [Uncertain("a", 0.0, 1.0), Uncertain("b", 0.0, 1.0)]
    mc = monte_carlo(f, ins, n_draws=2000, seed=8)
    assert set(mc.outputs) == {"npv", "cost"}
    assert mc.summary("cost")["mean"] == pytest.approx(110.0, abs=1.0)
    r = sobol_analysis(f, ins, n_base=512, seed=8, output="cost")
    assert r.total_order["b"] > 0.9, "cost depends only on b"
    assert r.total_order["a"] < 0.05
    with pytest.raises(ValueError, match="name the output"):
        sobol_analysis(f, ins, n_base=64)
    with pytest.raises(ValueError, match="not in"):
        sobol_analysis(f, ins, n_base=64, output="irr")


def test_failed_draws_are_counted_not_silently_dropped():
    """A model that legitimately fails on part of the input space must not
    yield a silently thinned sample presented as complete."""
    def f(a: float) -> float:
        if a > 0.8:
            raise ValueError("undefined region")
        return a
    ins = [Uncertain("a", 0.0, 1.0)]
    with pytest.raises(ValueError, match="undefined region"):
        monte_carlo(f, ins, n_draws=500, seed=10)
    mc = monte_carlo(f, ins, n_draws=2000, seed=10, on_error="record")
    assert mc.n_failed > 300
    assert mc.n_draws + mc.n_failed == 2000
    assert len(mc.failures) <= 20 and "undefined region" in mc.failures[0]
    # Sample arrays stay aligned with the surviving outputs.
    assert mc.samples["a"].size == mc.outputs["y"].size == mc.n_draws


def test_all_draws_failing_raises():
    def f(a: float) -> float:
        raise RuntimeError("always")
    with pytest.raises(ValueError, match="every one of"):
        monte_carlo(f, [Uncertain("a", 0.0, 1.0)], n_draws=10, on_error="record")


# --- Distributions -----------------------------------------------------------
def test_distributions_respect_their_support():
    for kind, kw in (("uniform", {}), ("triangular", {"mode": 0.3}),
                     ("normal", {"mean": 0.5, "sd": 0.4}),
                     ("lognormal", {"mean": 0.5, "sd": 0.3})):
        inp = Uncertain("x", 0.1, 0.9, kind, **kw)
        draws = inp.ppf(np.random.default_rng(0).random(20000))
        assert draws.min() >= 0.1 - 1e-9, f"{kind} breached lower bound"
        assert draws.max() <= 0.9 + 1e-9, f"{kind} breached upper bound"


def test_lognormal_parameters_are_moments_of_the_variable():
    """Process data reports the mean and sd OF THE VARIABLE, not of its log."""
    inp = Uncertain("x", 1e-6, 1e6, "lognormal", mean=10.0, sd=3.0)
    d = inp.ppf(np.random.default_rng(1).random(200000))
    assert d.mean() == pytest.approx(10.0, rel=0.02)
    assert d.std(ddof=1) == pytest.approx(3.0, rel=0.05)


def test_triangular_mode_shifts_the_median():
    lo = Uncertain("x", 0.0, 1.0, "triangular", mode=0.1)
    hi = Uncertain("x", 0.0, 1.0, "triangular", mode=0.9)
    u = np.random.default_rng(2).random(50000)
    assert np.median(lo.ppf(u)) < 0.45 < 0.55 < np.median(hi.ppf(u))


def test_input_validation():
    with pytest.raises(ValueError, match="high must exceed low"):
        Uncertain("x", 1.0, 1.0)
    with pytest.raises(ValueError, match="unknown distribution"):
        Uncertain("x", 0.0, 1.0, "beta")
    with pytest.raises(ValueError, match="needs mean and sd"):
        Uncertain("x", 0.0, 1.0, "normal")
    with pytest.raises(ValueError, match="mode must lie"):
        Uncertain("x", 0.0, 1.0, "triangular", mode=2.0)
    with pytest.raises(ValueError, match="sd must be positive"):
        Uncertain("x", 0.0, 1.0, "normal", mean=0.5, sd=0.0)
    with pytest.raises(ValueError, match="lognormal mean must be positive"):
        Uncertain("x", 0.0, 1.0, "lognormal", mean=-1.0, sd=1.0)
    with pytest.raises(ValueError, match="n_draws must be positive"):
        monte_carlo(lambda a: a, [Uncertain("a", 0.0, 1.0)], n_draws=0)
    with pytest.raises(ValueError, match="on_error"):
        monte_carlo(lambda a: a, [Uncertain("a", 0.0, 1.0)], n_draws=5,
                    on_error="ignore")


def test_spearman_screening_ranks_monotone_drivers():
    """For Y = 5a - 0.1b with a, b ~ U(0,1) independent, the exact Pearson
    correlations are +0.99980 and -0.019996 (cov(b,Y) = -0.1 Var(b), and
    sd(Y) = sqrt(25.01) sd). Spearman on a monotone linear map is close to
    those, so the thresholds are derived rather than guessed: an earlier version
    of this test asserted rho_b < -0.05 and failed at the correct value of
    -0.0385, because -0.05 was never a property of the model."""
    def f(a: float, b: float) -> float:
        return 5 * a - 0.1 * b

    # The exact Pearson correlations, derived rather than quoted. With
    # Var(a) = Var(b) = v, Var(Y) = (25 + 0.01) v, so sd(Y) = sqrt(25.01 v).
    assert 5 ** 2 + 0.1 ** 2 == pytest.approx(25.01, abs=1e-12)
    r_a = 5.0 / (25.01 ** 0.5)
    r_b = -0.1 / (25.01 ** 0.5)
    assert r_a == pytest.approx(0.99980, abs=1e-5)
    assert r_b == pytest.approx(-0.019996, abs=1e-6)
    #: The measured Spearman value that failed the original -0.05 threshold.
    #: -0.05 was never a property of the model, which is the point.
    OBSERVED_RHO_B, BAD_THRESHOLD = -0.0385, -0.05
    assert BAD_THRESHOLD < OBSERVED_RHO_B, \
        "the old threshold sat below the true value, so it could only fail"

    ins = [Uncertain("a", 0.0, 1.0), Uncertain("b", 0.0, 1.0)]
    mc = monte_carlo(f, ins, n_draws=5000, seed=12)
    rho = spearman_screening(mc, "y")
    assert rho["a"] == pytest.approx(0.9998, abs=0.01)
    assert -0.06 < rho["b"] < 0.0, "weakly negative, magnitude about 0.02"
    assert abs(rho["a"]) > 10 * abs(rho["b"])


def test_spearman_is_blind_to_non_monotone_effects():
    """The documented limitation, made a test. For Y = (a - 0.5)^2 the rank
    correlation with a is near zero, yet a explains ALL the variance. Screening
    on correlation alone would discard the only input that matters."""
    def f(a: float, b: float) -> float:
        return (a - 0.5) ** 2 + 0.0 * b
    ins = [Uncertain("a", 0.0, 1.0), Uncertain("b", 0.0, 1.0)]
    mc = monte_carlo(f, ins, n_draws=20000, seed=14)
    rho = spearman_screening(mc, "y")
    assert abs(rho["a"]) < 0.05, "correlation sees nothing"
    r = sobol_analysis(f, ins, n_base=1024, seed=14)
    assert r.total_order["a"] > 0.95, "variance decomposition sees everything"
    print(f"\nnon-monotone: Spearman rho_a {rho['a']:+.4f}, "
          f"Sobol ST_a {r.total_order['a']:.4f}")


def test_tornado_orders_by_swing_and_reports_deltas():
    def f(a: float, b: float) -> float:
        return 10 * a + 2 * b
    ins = [Uncertain("a", 0.0, 1.0), Uncertain("b", 0.0, 1.0)]
    t = tornado(f, ins, {"a": 0.5, "b": 0.5})
    assert list(t) == ["a", "b"], "ordered by swing, largest first"
    assert t["a"]["swing"] == pytest.approx(10.0)
    assert t["b"]["swing"] == pytest.approx(2.0)
    assert t["a"]["base"] == pytest.approx(6.0)
    assert t["a"]["low_delta"] == pytest.approx(-5.0)


def test_sobol_additive_function_indices_sum_to_one():
    """Second analytic validation of the Sobol estimator, beside Ishigami.

    Y = 3 x1 + 2 x2 + x3 with xi ~ U(0,1) independent. Each Var_i = a_i^2 / 12,
    so V = (9 + 4 + 1)/12 = 14/12 and S_i = a_i^2 / 14: 0.642857, 0.285714 and
    0.071429. The function is purely additive, so every total index equals its
    first index and both sets sum to exactly 1.

    Recorded because I got the reference wrong first: I divided by 13 rather
    than 14, printed 0.6923 as the analytic value for x1 and read the
    estimator's correct 0.6429 as a 5 percentage point error. The numbers that
    sum to 1 are the ones over 14, and the estimator was right.
    """
    a = np.array([3.0, 2.0, 1.0])
    v_i = a ** 2 / 12.0
    assert v_i.sum() == pytest.approx(14.0 / 12.0, rel=1e-12)
    analytic = v_i / v_i.sum()
    assert analytic.sum() == pytest.approx(1.0, abs=1e-12)
    assert analytic[0] == pytest.approx(9.0 / 14.0, rel=1e-12)
    assert analytic[0] == pytest.approx(0.642857, abs=1e-6)
    assert analytic[1] == pytest.approx(0.285714, abs=1e-6)
    assert analytic[2] == pytest.approx(0.071429, abs=1e-6)

    def additive(x1, x2, x3):
        return 3.0 * x1 + 2.0 * x2 + 1.0 * x3

    inputs = [Uncertain("x1", 0.0, 1.0), Uncertain("x2", 0.0, 1.0),
              Uncertain("x3", 0.0, 1.0)]
    r = sobol_analysis(additive, inputs, n_base=4096, seed=1)
    names = ["x1", "x2", "x3"]
    first = np.array([r.first_order[k] for k in names])
    total = np.array([r.total_order[k] for k in names])
    assert np.max(np.abs(first - analytic)) < 1e-4
    assert np.max(np.abs(total - analytic)) < 1e-4
    # Purely additive: no interaction, so total equals first index by index.
    for k in names:
        assert r.total_order[k] == pytest.approx(r.first_order[k], abs=1e-4)
    assert first.sum() == pytest.approx(1.0, abs=1e-3)
    assert total.sum() == pytest.approx(1.0, abs=1e-3)
    d = r.diagnostics()
    assert d["sum_first_le_one"] and d["sum_total_ge_one"] and d["converged"]
    assert d["first_exceeds_total"] == {}


def test_sobol_g_function_indices_match_the_closed_form():
    """Third analytic validation: the Sobol g-function, which has interactions.

    g(x) = prod_i (|4 x_i - 2| + a_i) / (1 + a_i) on the unit cube. Each
    V_i = 1 / (3 (1 + a_i)^2), the total variance is prod_i (1 + V_i) - 1, and
    S_Ti = V_i / (1 + V_i) x prod_j (1 + V_j) / V. With
    a = [0, 0.5, 3, 9, 99, 99] the first indices sum to LESS than 1 and the
    totals to MORE, which is the regime the additive test above cannot reach:
    it checks that the estimator gets the interaction budget right, not just
    the additive decomposition.
    """
    a = np.array([0.0, 0.5, 3.0, 9.0, 99.0, 99.0])
    v_i = 1.0 / (3.0 * (1.0 + a) ** 2)
    v_total = float(np.prod(1.0 + v_i) - 1.0)
    s_first = v_i / v_total
    s_total = v_i / (1.0 + v_i) * np.prod(1.0 + v_i) / v_total
    # Interactions are present, so the two sums bracket 1 from either side.
    assert s_first.sum() < 1.0
    assert s_total.sum() > 1.0
    assert s_first.sum() == pytest.approx(0.8902, abs=1e-3)
    assert s_total.sum() == pytest.approx(1.1119, abs=1e-3)

    def g(**kw):
        x = np.array([kw[f"x{i + 1}"] for i in range(a.size)])
        return float(np.prod((np.abs(4.0 * x - 2.0) + a) / (1.0 + a)))

    inputs = [Uncertain(f"x{i + 1}", 0.0, 1.0) for i in range(a.size)]
    r = sobol_analysis(g, inputs, n_base=8192, seed=7)
    names = [f"x{i + 1}" for i in range(a.size)]
    first = np.array([r.first_order[k] for k in names])
    total = np.array([r.total_order[k] for k in names])
    assert np.max(np.abs(first - s_first)) < 0.005
    assert np.max(np.abs(total - s_total)) < 0.005
    # Per-index ordering constraints, not just the sums.
    for k in names:
        assert r.total_order[k] >= r.first_order[k] - 0.05
    assert first.sum() <= 1.05
    assert total.sum() >= 0.95
    d = r.diagnostics()
    assert d["converged"]
    assert d["first_exceeds_total"] == {}
