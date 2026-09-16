"""Yield cascade and SPC: mass yield vs spec yield, capability, off-spec tails."""
import numpy as np
import pytest
from scipy import stats
from ae.plant.yield_cascade import (
    cascade_yield, stage_throughput_factors, capability, off_spec_fraction,
    off_spec_fraction_empirical, required_process_mean, joint_off_spec_fraction,
)


@pytest.mark.golden
def test_cascade_and_throughput_worked_example():
    """Stages 0.98, 0.90, 0.95.
       Y = 0.98 x 0.90 x 0.95 = 0.8379
       Stage 3 handles 1/0.95            = 1.0526 t per t shipped
       Stage 2 handles 1/(0.90 x 0.95)   = 1.1696
       Stage 1 handles 1/0.8379          = 1.1935
    """
    ys = [0.98, 0.90, 0.95]
    assert cascade_yield(ys) == pytest.approx(0.8379, abs=1e-9)
    f = stage_throughput_factors(ys)
    assert f == pytest.approx([1.19345, 1.16959, 1.05263], abs=1e-5)
    # The first stage's factor is exactly the reciprocal of the cascade yield.
    assert f[0] == pytest.approx(1.0 / cascade_yield(ys), rel=1e-12)
    # Factors decrease downstream: later stages carry less mass.
    assert f[0] > f[1] > f[2] > 1.0


def test_cascade_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one stage"):
        cascade_yield([])
    with pytest.raises(ValueError):
        cascade_yield([0.9, 1.4])
    with pytest.raises(ValueError, match="zero yield"):
        stage_throughput_factors([0.9, 0.0])


def test_mass_yield_and_spec_yield_are_different_quantities():
    """A lot off spec is a full lot of mass that cannot be sold into that grade,
    not a mass loss. Multiplying them together would double-count."""
    mass_y = cascade_yield([0.98, 0.90, 0.95])
    spec_off = off_spec_fraction(18.0, 3.0, 30.0)
    assert mass_y == pytest.approx(0.8379, abs=1e-9)
    assert spec_off < 1e-4
    # They are independent numbers: saleable fraction of FEED is the product,
    # but the spec loss applies to lots, not to tonnes lost in the flowsheet.
    assert mass_y * (1 - spec_off) < mass_y


@pytest.mark.golden
def test_required_process_mean_worked_example():
    """USL 30 ppm Al, Cpk 1.33.
       sigma 3 ppm -> mu = 30 - 3(1.33)(3) = 30 - 11.97 = 18.03 ppm
       sigma 5 ppm -> mu = 30 - 3(1.33)(5) = 30 - 19.95 = 10.05 ppm
    Halving sigma is worth more than a large shift in mean, which is why
    lot-to-lot control is the qualification-relevant quantity."""
    assert required_process_mean(30.0, 3.0, 1.33) == pytest.approx(18.03, abs=1e-9)
    assert required_process_mean(30.0, 5.0, 1.33) == pytest.approx(10.05, abs=1e-9)
    # Headroom gained by halving sigma from 6 to 3 ppm:
    assert (required_process_mean(30.0, 3.0, 1.33)
            - required_process_mean(30.0, 6.0, 1.33)) == pytest.approx(11.97, abs=1e-9)


def test_required_mean_raises_when_variability_eats_the_spec():
    """At sigma 8 ppm and Cpk 1.33 the required mean is negative: not attainable."""
    with pytest.raises(ValueError, match="not physically attainable"):
        required_process_mean(30.0, 8.0, 1.33)


@pytest.mark.golden
def test_capability_and_off_spec_are_the_same_number_two_ways():
    """f = 1 - Phi(3 Cpk) exactly.

    At mu = 18 ppm, sigma = 3 ppm against a 30 ppm limit:
      Cpk = (30 - 18) / (3 x 3) = 12/9 = 4/3 = 1.33333
      3 Cpk = 4.0  ->  f = 1 - Phi(4) = 3.1671e-5

    Note the 1.33 that capability tables quote is NOT 4/3: at Cpk = 1.33
    exactly, 3 Cpk = 3.99 and f = 3.3037e-5, 4 percent higher. The two are
    routinely conflated, and this test pins both so the distinction survives.
    """
    assert (30.0 - 18.0) / (3.0 * 3.0) == pytest.approx(4.0 / 3.0, rel=1e-12)
    assert float(stats.norm.sf(4.0)) == pytest.approx(3.1671e-5, rel=1e-3)
    assert float(stats.norm.sf(3.0 * 1.33)) == pytest.approx(3.3037e-5, rel=1e-3)
    # The module's own parametric tail must agree with the closed form.
    assert off_spec_fraction(18.0, 3.0, 30.0) == pytest.approx(float(stats.norm.sf(4.0)),
                                                               rel=1e-12)


def test_capability_estimate_converges_to_the_true_value():
    """With enough lots the estimator recovers Cpk = 4/3. A modest sample does
    NOT: the seed-7 draw of 400 lots returns 1.481, 11 percent high, because a
    favourable mean and a low sample sd compound. That is the reason
    is_estimated_from_few_lots and the confidence interval exist."""
    rng = np.random.default_rng(7)
    big = capability(rng.normal(18.0, 3.0, 40000), usl=30.0)
    assert big.cpk == pytest.approx(4.0 / 3.0, rel=0.03)
    assert big.ci_low < 4.0 / 3.0 < big.ci_high
    assert big.off_spec_fraction < 1e-3
    small = capability(np.random.default_rng(7).normal(18.0, 3.0, 400), usl=30.0)
    assert small.cpk == pytest.approx(1.481, abs=0.001)
    assert small.cpk > big.cpk, "the small sample is optimistic here"
    # The interval is what protects the reader from that point estimate.
    assert small.ci_high - small.ci_low > 5 * (big.ci_high - big.ci_low)


def test_capability_interval_widens_with_fewer_lots():
    rng = np.random.default_rng(11)
    wide = capability(rng.normal(18.0, 3.0, 8), usl=30.0)
    narrow = capability(rng.normal(18.0, 3.0, 400), usl=30.0)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)
    assert wide.is_estimated_from_few_lots
    assert not narrow.is_estimated_from_few_lots


def test_capability_guards():
    with pytest.raises(ValueError, match="at least 2 lots"):
        capability([20.0], usl=30.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        capability([20.0, -1.0], usl=30.0)
    with pytest.raises(ValueError, match="must be positive"):
        capability([20.0, 21.0], usl=0.0)
    with pytest.raises(ValueError, match="resolution artefact"):
        capability([20.0, 20.0, 20.0], usl=30.0)


def test_lognormal_tail_exceeds_normal_tail():
    """The central reason the normal model is the wrong default for an impurity:
    it UNDERSTATES the upper tail, which is the tail that fails lots."""
    n = off_spec_fraction(18.0, 3.0, 30.0, model="normal")
    ln = off_spec_fraction(18.0, 3.0, 30.0, model="lognormal")
    assert ln > n
    assert ln / n > 10
    assert n == pytest.approx(3.1671e-5, rel=1e-2)
    assert ln == pytest.approx(7.65e-4, rel=1e-2)


def test_lognormal_matches_requested_moments():
    """Method of moments: the fitted lognormal must reproduce the mean and sd
    it was given, otherwise the tail is being computed for a different process."""
    mean, sigma = 18.0, 3.0
    cv2 = (sigma / mean) ** 2
    s2 = np.log1p(cv2)
    dist = stats.lognorm(s=np.sqrt(s2), scale=np.exp(np.log(mean) - 0.5 * s2))
    assert dist.mean() == pytest.approx(mean, rel=1e-12)
    assert dist.std() == pytest.approx(sigma, rel=1e-12)


def test_off_spec_guards():
    for kw in [dict(mean=18.0, sigma=0.0, usl=30.0),
               dict(mean=-1.0, sigma=3.0, usl=30.0),
               dict(mean=18.0, sigma=3.0, usl=0.0)]:
        with pytest.raises(ValueError):
            off_spec_fraction(**kw)


def test_empirical_off_spec_makes_no_shape_assumption():
    assays = [12.0, 18.0, 22.0, 31.0, 29.9, 45.0]
    f, n_off, n = off_spec_fraction_empirical(assays, usl=30.0)
    assert (n_off, n) == (2, 6)
    assert f == pytest.approx(2 / 6)
    with pytest.raises(ValueError, match="no assays"):
        off_spec_fraction_empirical([], usl=30.0)


@pytest.mark.golden
def test_joint_bounds_bracket_the_truth():
    """Al 1 percent, Na 2 percent, K 2 percent off spec individually.
       Independent: 1 - (0.99)(0.98)(0.98) = 0.049204
       Perfect:     max = 0.02
    The same fluid inclusions carry Na and K together, so independence is
    wrong; both bounds are returned so neither can be adopted silently."""
    r = joint_off_spec_fraction({"Al": 0.01, "Na": 0.02, "K": 0.02})
    assert r["independent"] == pytest.approx(1 - 0.99 * 0.98 * 0.98, abs=1e-12)
    assert r["independent"] == pytest.approx(0.049204, abs=1e-6)
    assert r["perfect"] == pytest.approx(0.02)
    assert r["spread"] == pytest.approx(r["independent"] - r["perfect"])
    assert r["perfect"] < r["independent"], "perfect correlation is the optimistic bound"


def test_joint_single_element_bounds_coincide():
    r = joint_off_spec_fraction({"Al": 0.03})
    assert r["independent"] == pytest.approx(0.03)
    assert r["perfect"] == pytest.approx(0.03)
    assert r["spread"] == pytest.approx(0.0, abs=1e-15)


def test_joint_guards_and_modes():
    with pytest.raises(ValueError, match="no per-element"):
        joint_off_spec_fraction({})
    with pytest.raises(ValueError):
        joint_off_spec_fraction({"Al": 1.5})
    assert set(joint_off_spec_fraction({"Al": 0.01}, "independent")) == {"independent"}
    assert set(joint_off_spec_fraction({"Al": 0.01}, "perfect")) == {"perfect"}
