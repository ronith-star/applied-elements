"""Value of information: hand-computable EVPI, zero cases, estimator noise.

No test module existed for :mod:`ae.agent.decisions` before this one, which is
why the estimator defect recorded in
``test_evpi_of_a_decision_irrelevant_parameter_is_not_large_and_positive`` had
survived: the module computes a quantity whose correct value is known in
closed form on small problems, and nothing was checking it against one.
"""
import numpy as np
import pytest

from ae.agent.decisions import (
    Action,
    DecisionProblem,
    evpi,
    evpi_convergence,
    measurement_priority,
    rank_measurements,
)


def binary_theta(rng, n):
    """theta = 1 with probability 1/2, else 0."""
    return (rng.random(n) < 0.5).astype(float)


def build_or_walk(build_up=10.0, build_down=-6.0):
    """Two actions, one binary uncertainty, fully hand-computable.

    build pays ``build_up`` when theta = 1 and ``build_down`` when theta = 0;
    walk pays 0 either way.
    """
    def value(action, params):
        if action == "build":
            return build_up if params["theta"] > 0.5 else build_down
        return 0.0
    return DecisionProblem(actions=[Action("build"), Action("walk")],
                           value=value, priors={"theta": binary_theta})


@pytest.mark.golden
def test_evpi_worked_example_two_actions_one_binary_uncertainty():
    """Hand-computed end to end, every intermediate asserted.

      E[V | build] = 0.5 x 10 + 0.5 x (-6) = 5 - 3 = 2.0
      E[V | walk]  = 0.0
      prior best   = build, so the baseline is 2.0
      perfect info: theta = 1 -> build pays 10 ; theta = 0 -> walk pays 0
      resolved     = 0.5 x 10 + 0.5 x 0 = 5.0
      EVPI         = 5.0 - 2.0 = 3.0
      switch       = the decision changes only when theta = 0, so 0.5
    """
    assert 0.5 * 10.0 + 0.5 * (-6.0) == pytest.approx(2.0, abs=1e-12)
    assert 0.5 * 10.0 + 0.5 * 0.0 == pytest.approx(5.0, abs=1e-12)
    assert 5.0 - 2.0 == pytest.approx(3.0, abs=1e-12)

    p = build_or_walk()
    iv = evpi(p, "theta", n_outer=4096, n_inner=1, seed=0)
    assert iv.prior_best_action == "build"
    # The baseline is now PAIRED with the resolved sample, so it carries the
    # outer sample's own error rather than an independent one: measured 2.1406
    # against the hand value 2.0 at this seed and outer size.
    assert iv.baseline_value == pytest.approx(2.0, abs=0.25)
    assert iv.resolved_value == pytest.approx(5.0, abs=0.25)
    assert iv.evpi == pytest.approx(3.0, abs=0.15)
    assert iv.switch_fraction == pytest.approx(0.5, abs=0.03)
    # EVPI as a fraction of the baseline: 3.0 / 2.0 = 1.5.
    assert 3.0 / 2.0 == pytest.approx(1.5, abs=1e-12)
    assert iv.evpi_fraction == pytest.approx(1.5, abs=0.25)


def test_evpi_is_zero_when_the_decision_cannot_change():
    """One action dominates at every value of theta, so information is worth
    nothing however much variance theta carries.

    V(build) = 10 + theta with theta ~ U(0,1) is never below 10, and walk pays
    0, so build is chosen whatever theta turns out to be. EVPI must be 0 and
    the switch fraction exactly 0.
    """
    def value(action, params):
        return 10.0 + params["theta"] if action == "build" else 0.0
    p = DecisionProblem(actions=[Action("build"), Action("walk")], value=value,
                        priors={"theta": lambda rng, n: rng.random(n)})
    for n_inner in (1, 16, 256):
        iv = evpi(p, "theta", n_outer=256, n_inner=n_inner, seed=0)
        assert iv.switch_fraction == 0.0
        assert iv.evpi < 0.01, f"n_inner={n_inner}: EVPI {iv.evpi}"
        assert iv.prior_best_action == "build"


def test_evpi_is_positive_when_the_decision_can_change():
    """The complement of the zero case: the same structure with a payoff that
    crosses the walk-away value has strictly positive EVPI.

    V(build) = theta - 0.5 with theta ~ U(0,1) is negative below theta = 0.5,
    so learning theta flips the choice half the time. E[V|build] = 0 exactly,
    so the baseline is 0, and the resolved value is
    E[max(theta - 0.5, 0)] = integral from 0.5 to 1 of (t - 0.5) dt = 0.125.
    """
    assert 0.5 * 0.5 * 0.5 / 2.0 == pytest.approx(0.0625, abs=1e-12)
    # integral_{0.5}^{1} (t - 0.5) dt = [t^2/2 - t/2] = 0.5 - 0.375 = 0.125.
    assert 0.5 - 0.375 == pytest.approx(0.125, abs=1e-12)

    def value(action, params):
        return params["theta"] - 0.5 if action == "build" else 0.0
    p = DecisionProblem(actions=[Action("build"), Action("walk")], value=value,
                        priors={"theta": lambda rng, n: rng.random(n)})
    iv = evpi(p, "theta", n_outer=4096, n_inner=1, seed=1)
    assert iv.evpi > 0.0
    assert iv.evpi == pytest.approx(0.125, abs=0.02)
    assert iv.switch_fraction == pytest.approx(0.5, abs=0.03)


def test_evpi_of_a_decision_irrelevant_parameter_is_not_large_and_positive():
    """The defect. A parameter that shifts BOTH actions equally cannot change
    the decision, so its EVPI is exactly zero, and the estimator reported 73.

    Fixture: theta is the binary driver as above, and nu ~ N(0, 1000) is added
    to the payoff of EVERY action. Adding a constant to all actions leaves the
    argmax untouched at every nu, so EVPI(nu) = 0 identically, whatever nu's
    variance.

    Measured before the fix, raw resolved minus baseline for nu:
      n_outer  256: +73.1086 at n_inner 16, +72.8091 at n_inner 256
      n_outer 1024: -49.7340 at n_inner 16, -49.8785 at n_inner 256
      n_outer 4096: -27.5788 at n_inner 16, -27.7527 at n_inner 256
    The magnitude falls as 1/sqrt(n_outer) and does NOT respond to n_inner, so
    it is not the max-of-noisy-inner-estimates bias the module docstring
    attributes it to. It is outer-sample noise: the baseline was estimated
    from an INDEPENDENT sample (best_action_now with its own draws) while the
    resolved value came from the nested sample, so the two differ by the
    sampling error of a N(0, 1000) mean, which the subtraction does not
    cancel. A hand-written paired estimator using common random numbers on the
    identical problem gave 0.017578, 0.007080 and 0.009094 at those three
    outer sizes, three orders of magnitude smaller.

    The clamp at zero makes this WORSE rather than safer: half the time the
    noise is negative and is clamped to a correct-looking 0, and half the time
    it is positive and reported as a large information value on a parameter
    that cannot matter.
    """
    def value(action, params):
        base = (10.0 if params["theta"] > 0.5 else -6.0) if action == "build" else 0.0
        return base + params["nu"]
    priors = {"theta": binary_theta,
              "nu": lambda rng, n: rng.normal(0.0, 1000.0, n)}
    p = DecisionProblem(actions=[Action("build"), Action("walk")],
                        value=value, priors=priors)

    # nu enters every action identically, so it cannot change the argmax.
    for nu in (-2000.0, 0.0, 2000.0):
        for theta in (0.0, 1.0):
            params = {"theta": theta, "nu": nu}
            gap = value("build", params) - value("walk", params)
            assert gap == pytest.approx(10.0 if theta > 0.5 else -6.0, abs=1e-9)

    # The three outer sizes the pre-fix figures above were measured at. 4096
    # is run too, since the defect did not shrink between 1024 and 4096.
    # The three outer sizes the pre-fix figures above were measured at. The
    # defect did shrink with n_outer (it is 1/sqrt(n_outer) noise), so 4096 is
    # the weakest of the three as evidence and the smallest is the strongest;
    # all three are run because the fix must hold at every one.
    for n_outer in (256, 1024, 4096):
        iv = evpi(p, "nu", n_outer=n_outer, n_inner=16, seed=0)
        assert abs(iv.resolved_value - iv.baseline_value) < 1.0, (
            f"n_outer={n_outer}: raw EVPI on a decision-irrelevant parameter "
            f"is {iv.resolved_value - iv.baseline_value:.4f}; the pre-fix "
            f"estimator gave +73.1086 at n_outer 256 through independent "
            f"sampling of the baseline"
        )
        assert iv.evpi < 0.25

    # What remains after pairing IS the inner-maximum bias the module
    # docstring describes, and unlike the outer-sample noise it responds to
    # n_inner: measured at n_outer 512, seed 0, the residual falls 2.964844,
    # 0.671875, 0.136719, 0.009766, 0.000000 across n_inner 1, 4, 16, 64, 256.
    residuals = [evpi(p, "nu", n_outer=512, n_inner=ni, seed=0).evpi
                 for ni in (1, 4, 16, 64, 256)]
    assert residuals[0] == pytest.approx(2.964844, abs=1e-5)
    assert residuals[2] == pytest.approx(0.136719, abs=1e-5)
    assert residuals[4] == pytest.approx(0.0, abs=1e-9)
    assert residuals == sorted(residuals, reverse=True), (
        f"the post-pairing residual must fall monotonically with n_inner, "
        f"which is the signature of the inner-maximum bias: {residuals}"
    )

    # And theta, which DOES drive the decision, still scores 3.0 in the
    # presence of the same large nuisance variance.
    iv_theta = evpi(p, "theta", n_outer=1024, n_inner=64, seed=0)
    assert iv_theta.evpi == pytest.approx(3.0, abs=0.6)
    assert iv_theta.evpi > 10.0 * evpi(p, "nu", n_outer=1024, n_inner=64,
                                       seed=0).evpi


def test_evpi_baseline_is_paired_with_the_resolved_sample():
    """Direct statement of the mechanism, independent of any nuisance scale.

    With common random numbers the baseline and the resolved value are
    computed from the SAME draws, so on a problem where information is
    worthless the two must agree to floating-point noise rather than to Monte
    Carlo noise. Here every action pays exactly nu, so no action is ever
    better than another and both the EVPI and the switch fraction are zero by
    construction.
    """
    def value(action, params):
        return params["nu"]
    p = DecisionProblem(actions=[Action("a"), Action("b")], value=value,
                        priors={"nu": lambda rng, n: rng.normal(0.0, 1e6, n)})
    iv = evpi(p, "nu", n_outer=64, n_inner=8, seed=0)
    assert iv.resolved_value == pytest.approx(iv.baseline_value, abs=1e-6), (
        f"resolved {iv.resolved_value} against baseline {iv.baseline_value}: "
        "with identical payoffs these must come from the same sample"
    )
    assert iv.evpi == pytest.approx(0.0, abs=1e-6)


def test_a_misspelled_parameter_raises_rather_than_reading_as_worthless():
    p = build_or_walk()
    with pytest.raises(KeyError, match="not a prior"):
        evpi(p, "thета_typo", n_outer=8, n_inner=8)


def test_one_action_is_a_forecast_not_a_decision():
    with pytest.raises(ValueError, match="at least two actions"):
        DecisionProblem(actions=[Action("only")], value=lambda a, p: 0.0,
                        priors={"theta": binary_theta})


def test_ranking_puts_the_decision_driver_above_the_nuisance():
    """rank_measurements must order by EVPI, and the driver must win.

    theta flips the choice and is worth 3.0; nu shifts both payoffs equally
    and is worth 0. A ranking that puts nu first would send a characterization
    budget at the parameter that cannot change the decision, which is the
    specific failure this module exists to prevent.
    """
    def value(action, params):
        base = (10.0 if params["theta"] > 0.5 else -6.0) if action == "build" else 0.0
        return base + params["nu"]
    priors = {"theta": binary_theta,
              "nu": lambda rng, n: rng.normal(0.0, 1000.0, n)}
    p = DecisionProblem(actions=[Action("build"), Action("walk")],
                        value=value, priors=priors)
    ranked = rank_measurements(p, n_outer=1024, n_inner=16, seed=0)
    assert [iv.parameter for iv in ranked][0] == "theta"
    # theta's hand-computed value, as in the worked example above.
    assert 0.5 * 10.0 + 0.5 * 0.0 - (0.5 * 10.0 + 0.5 * -6.0) == pytest.approx(
        3.0, abs=1e-12)
    assert ranked[0].evpi == pytest.approx(3.0, abs=0.6)
    assert ranked[0].evpi > ranked[-1].evpi
    # measurement_priority divides EVPI by assay cost, which AMPLIFIES the
    # residual inner-maximum bias on a cheap measurement and can invert the
    # ranking. Measured at n_outer 1024, n_inner 16, seed 0, with theta at
    # 1000 USD and nu at 10 USD: nu ranks FIRST on 0.016895 EVPI per USD
    # against theta's 0.002965, because nu's residual 0.168945 of pure
    # estimator bias survives division by the small cost while theta's true
    # 2.964844 is divided by a cost a hundred times larger. So n_inner is a
    # correctness parameter for the cost ranking, not just a precision one.
    cheap_nu = measurement_priority(p, {"theta": 1000.0, "nu": 10.0},
                                    n_outer=1024, n_inner=16, seed=0)
    assert cheap_nu[0]["parameter"] == "nu"
    assert cheap_nu[0]["evpi_per_cost"] == pytest.approx(0.016895, abs=1e-6)
    assert cheap_nu[1]["evpi_per_cost"] == pytest.approx(0.002965, abs=1e-6)

    # At an n_inner large enough to drive the bias to zero, the ranking is
    # right: nu's EVPI is exactly 0.0, so no cost makes it competitive.
    converged = measurement_priority(p, {"theta": 1000.0, "nu": 10.0},
                                     n_outer=512, n_inner=256, seed=0)
    assert converged[0]["parameter"] == "theta"
    assert converged[1]["parameter"] == "nu"
    assert converged[1]["evpi"] == 0.0
    assert converged[0]["evpi"] == pytest.approx(2.847656, abs=1e-5)
