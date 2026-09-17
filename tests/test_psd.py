"""Tests for ae.physics.psd, including the number-area-volume weighting relationship.

The weighting-conversion test is pinned deliberately and explicitly: mixing a
number-weighted d50 with a volume-weighted one is the commonest quantitative error in
particle technology, and a test that pins the exact relationship is the only thing that
stops it recurring silently.
"""

from __future__ import annotations

import math

import pytest

from ae.core.feedstock import Feedstock, OreType
from ae.core.provenance import Tag, Tier, Value
from ae.core.units import Q_, DimensionalityError
from ae.physics import psd as psd_mod
from ae.physics.psd import (
    Z_10,
    Z_90,
    LogNormalPSD,
    PSDSummary,
    RosinRammlerPSD,
    Weighting,
    convert_lognormal_median,
    feedstock_sphericity_assumption,
    specific_surface_area,
)

QUARTZ_DENSITY = Q_(2650.0, "kg/m**3")


@pytest.fixture
def ore() -> Feedstock:
    return Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="Vikarabad", country="IN")


@pytest.fixture
def ln() -> LogNormalPSD:
    return LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY)


@pytest.fixture
def rr() -> RosinRammlerPSD:
    return RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=3.5, density=QUARTZ_DENSITY)


# --------------------------------------------------------------------------------------
# dimensional analysis
# --------------------------------------------------------------------------------------

def test_bare_float_diameter_rejected() -> None:
    """A bare float is not a diameter. Pydantic surfaces the guard as a ValidationError."""
    with pytest.raises((TypeError, ValueError)):
        LogNormalPSD(d_gn=10.0, sigma_g=1.5, density=QUARTZ_DENSITY)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        specific_surface_area(15.0, QUARTZ_DENSITY)  # type: ignore[arg-type]


@pytest.mark.parametrize("wrong", [Q_(10.0, "kg"), Q_(10.0, "s"), Q_(10.0, "J")])
def test_wrong_dimension_diameter_rejected(wrong) -> None:
    with pytest.raises(DimensionalityError):
        LogNormalPSD(d_gn=wrong, sigma_g=1.5, density=QUARTZ_DENSITY)


def test_wrong_dimension_density_rejected() -> None:
    with pytest.raises(DimensionalityError):
        LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=Q_(2650.0, "kg"))


def test_specific_surface_area_dimensionality(ln) -> None:
    """S_m must come back as area per unit mass, from equation (8)."""
    s = ln.specific_surface_area()
    assert s.dimensionality == Q_(1.0, "m**2/kg").dimensionality


def test_moment_dimensionality(ln) -> None:
    """The k-th moment of a length distribution has dimensions of length^k."""
    assert ln.moment(1.0).dimensionality == Q_(1.0, "m").dimensionality
    assert ln.moment(2.0).dimensionality == Q_(1.0, "m**2").dimensionality
    assert ln.moment(3.0).dimensionality == Q_(1.0, "m**3").dimensionality


def test_sauter_mean_is_the_ratio_of_the_third_to_second_moment(ln) -> None:
    """d32 is DEFINED as M3/M2, which the closed form must reproduce exactly.

    This is both a dimensional and an algebraic check: M3/M2 has dimensions of length and
    must equal d_gn exp(5 s^2 / 2).
    """
    m3 = float(ln.moment(3.0).to("m**3").magnitude)
    m2 = float(ln.moment(2.0).to("m**2").magnitude)
    ratio = m3 / m2
    closed = float(ln.sauter_d32().to("m").magnitude)
    assert ratio == pytest.approx(closed, rel=1e-12)
    exponent_coeff = 3.0**2 - 2.0**2
    assert exponent_coeff == pytest.approx(5.0, rel=1e-12)


def test_diameters_are_lengths(ln, rr) -> None:
    for q in (ln.median(Weighting.VOLUME), ln.sauter_d32(), ln.quantile(0.1),
              rr.median(), rr.sauter_d32(), rr.volume_weighted_mean(), rr.quantile(0.9)):
        assert q.dimensionality == Q_(1.0, "m").dimensionality


def test_cumulative_and_span_are_dimensionless(ln, rr) -> None:
    assert isinstance(rr.cumulative_undersize(Q_(20.0, "um")), float)
    assert isinstance(ln.span(), float)


# --------------------------------------------------------------------------------------
# golden tests
# --------------------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_lognormal_weighted_medians() -> None:
    """Hatch-Choate conversions for d_gn = 10 um, sigma_g = 1.5.

    s = ln(1.5) = 0.40546511, so s^2 = 0.16440195.

      d_ga = d_gn exp(2 s^2) = 10 x exp(0.32880391) = 10 x 1.38930540 = 13.89305 um
      d_gv = d_gn exp(3 s^2) = 10 x exp(0.49320586) = 10 x 1.63755760 = 16.37558 um
      d_32 = d_gn exp(2.5 s^2) = 10 x exp(0.41100488) = 10 x 1.50833272 = 15.08333 um

    The volume median is 64 percent larger than the count median for this perfectly
    ordinary width, which is the whole reason the weighting basis must be stated.

    The 2 s^2 and 3 s^2 exponents were written 0.32880390 and 0.49320585 before
    this revision, each low by one in the eighth decimal. Both came from
    multiplying the ROUNDED s^2 = 0.16440195 rather than s^2 itself: 2 and 3
    times ln(1.5)^2 are 0.32880391 and 0.49320586. The exp() values they give
    differ in the eighth digit (exp(0.49320585) = 1.63755758 against
    1.63755760 for the exact exponent), so the quoted 1.63755760 belongs to
    the corrected exponent. The diameters at five decimals, 13.89305 and
    16.37558 um, are unaffected either way. Both exponents and both exp values
    are asserted below.
    """
    s = math.log(1.5)
    assert s == pytest.approx(0.40546511, abs=1e-8)
    assert s * s == pytest.approx(0.16440195, abs=1e-8)
    assert math.exp(2.0 * s * s) == pytest.approx(1.38930540, abs=1e-8)
    assert math.exp(3.0 * s * s) == pytest.approx(1.63755760, abs=1e-8)
    assert math.exp(2.5 * s * s) == pytest.approx(1.50833272, abs=1e-8)

    p = LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY)
    assert float(p.median(Weighting.NUMBER).to("um").magnitude) == pytest.approx(
        10.0, abs=1e-9)
    assert float(p.median(Weighting.AREA).to("um").magnitude) == pytest.approx(
        13.89305, abs=1e-5)
    assert float(p.median(Weighting.VOLUME).to("um").magnitude) == pytest.approx(
        16.37558, abs=1e-5)
    assert float(p.sauter_d32().to("um").magnitude) == pytest.approx(15.08333, abs=1e-5)
    assert 16.37558 / 10.0 == pytest.approx(1.637558, abs=1e-6)
    assert 2.0 * s * s == pytest.approx(0.32880391, abs=5e-9)
    assert 3.0 * s * s == pytest.approx(0.49320586, abs=5e-9)
    # The superseded exponents came from multiplying the ROUNDED s^2, and
    # each is low by 1e-8; the exp() values they give are unchanged at the
    # eight digits the docstring quotes, which is why only the exponents moved.
    assert 3.0 * 0.16440195 == pytest.approx(0.49320585, abs=5e-9)
    assert math.exp(0.49320585) == pytest.approx(1.63755758, abs=5e-9)
    assert math.exp(3.0 * s * s) == pytest.approx(1.63755760, abs=5e-9)
    assert 2.5 * s * s == pytest.approx(0.41100488, abs=1e-8)


@pytest.mark.golden
def test_golden_weighting_conversion_relationship_pinned() -> None:
    """The number-area-volume relationship, pinned in both directions.

    For a log-normal the three medians are in fixed geometric relation:
      d_gv / d_ga = exp(3 s^2) / exp(2 s^2) = exp(s^2) = exp(0.16440195) = 1.17868800
      d_gv / d_gn = exp(3 s^2) = 1.63755760
      d_ga / d_gn = exp(2 s^2) = 1.38930540

    and the Sauter mean sits between the area and volume medians:
      d_ga = 13.89305 < d_32 = 15.08333 < d_gv = 16.37558 um

    Round-tripping any median through two conversions must return the original, which is
    the property a unit-conversion helper has to guarantee.
    """
    s2 = math.log(1.5) ** 2
    assert math.exp(s2) == pytest.approx(1.17868800, abs=1e-8)

    p = LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY)
    d_gn = float(p.median(Weighting.NUMBER).to("um").magnitude)
    d_ga = float(p.median(Weighting.AREA).to("um").magnitude)
    d_gv = float(p.median(Weighting.VOLUME).to("um").magnitude)
    d32 = float(p.sauter_d32().to("um").magnitude)
    assert d_gv / d_ga == pytest.approx(1.17868800, abs=1e-8)
    assert d_gv / d_gn == pytest.approx(1.63755760, abs=1e-8)
    assert d_ga / d_gn == pytest.approx(1.38930540, abs=1e-8)
    assert d_ga < d32 < d_gv

    back = convert_lognormal_median(Q_(d_gv, "um"), 1.5, Weighting.VOLUME,
                                    Weighting.NUMBER)
    assert float(back.to("um").magnitude) == pytest.approx(10.0, abs=1e-9)
    fwd = convert_lognormal_median(Q_(10.0, "um"), 1.5, Weighting.NUMBER,
                                   Weighting.VOLUME)
    assert float(fwd.to("um").magnitude) == pytest.approx(d_gv, rel=1e-12)
    area = convert_lognormal_median(Q_(d_gv, "um"), 1.5, Weighting.VOLUME, Weighting.AREA)
    assert float(area.to("um").magnitude) == pytest.approx(d_ga, rel=1e-12)
    assert s2 == pytest.approx(0.16440195, abs=1e-8)
    assert d_ga == pytest.approx(13.89305, abs=1e-5)
    assert d32 == pytest.approx(15.08333, abs=1e-5)
    assert d_gv == pytest.approx(16.37558, abs=1e-5)


@pytest.mark.golden
def test_golden_lognormal_quantiles_and_span() -> None:
    """Volume-basis quantiles and span for d_gn = 10 um, sigma_g = 1.5.

    On the volume basis the median is d_gv = 16.375576 um and
      d10 = d_gv exp(z10 s) = 16.375576 x exp(-1.28155157 x 0.40546511)
          = 16.375576 x exp(-0.51962444) = 16.375576 x 0.59474387 = 9.73927 um
      d90 = d_gv exp(z90 s) = 16.375576 x exp(+0.51962444)
          = 16.375576 x 1.68139607 = 27.53383 um
      span = (27.53383 - 9.73927) / 16.375576 = 17.79456 / 16.375576 = 1.086652
    """
    s = math.log(1.5)
    assert Z_10 * s == pytest.approx(-0.51962444, abs=1e-8)
    assert math.exp(-0.51962444) == pytest.approx(0.59474387, abs=1e-8)
    assert math.exp(0.51962444) == pytest.approx(1.68139607, abs=1e-8)
    assert Z_90 == pytest.approx(-Z_10, abs=1e-12)

    p = LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY)
    assert float(p.quantile(0.1).to("um").magnitude) == pytest.approx(9.73927, abs=1e-5)
    assert float(p.quantile(0.9).to("um").magnitude) == pytest.approx(27.53383, abs=1e-5)
    assert p.span() == pytest.approx(1.086652, abs=1e-6)
    assert (27.53383 - 9.73927) / 16.375576 == pytest.approx(1.086652, abs=1e-6)
    assert s == pytest.approx(0.40546511, abs=1e-8)
    assert -Z_10 == pytest.approx(1.28155157, abs=1e-8)
    d10 = float(p.quantile(0.1).to("um").magnitude)
    d90 = float(p.quantile(0.9).to("um").magnitude)
    assert (d90 - d10) == pytest.approx(17.79456, abs=1e-5)


@pytest.mark.golden
def test_golden_specific_surface_area() -> None:
    """Geometric specific surface area from the Sauter mean, equation (8).

    S_m = 6 / (rho d_32 psi) with rho = 2650 kg/m^3, d_32 = 15.083327e-6 m, psi = 1:
      rho x d_32 = 2650 x 15.083327e-6 = 0.039970817 kg/m^2
      S_m        = 6 / 0.039970817 = 150.1095 m^2/kg = 0.1501 m^2/g

    Compare with the 0.38 to 0.47 m^2/g that Ringdalen 2015 measured by BET on
    heat-treated natural quartz: the geometric figure for a 15 um powder is three times
    SMALLER than a BET measurement on lump. That is the surface roughness and internal
    cracking that equation (8) cannot see, and it is why LIMITATIONS item 1 forbids
    substituting one for the other.
    """
    d32 = 15.083327243425664e-6
    assert 2650.0 * d32 == pytest.approx(0.039970817, abs=1e-9)
    assert 6.0 / 0.039970817 == pytest.approx(150.1095, abs=1e-4)
    s = specific_surface_area(Q_(d32, "m"), QUARTZ_DENSITY)
    assert float(s.to("m**2/kg").magnitude) == pytest.approx(150.1095, abs=1e-4)
    assert float(s.to("m**2/g").magnitude) == pytest.approx(0.1501, abs=1e-4)
    geometric_estimate = float(s.to("m**2/g").magnitude)
    assert 0.38 / 3.0 <= geometric_estimate <= 0.47 / 3.0


@pytest.mark.golden
def test_golden_rosin_rammler_diameters() -> None:
    """Rosin-Rammler derived diameters for d' = 20 um, n = 3.5.

      d50 = d' (ln 2)^{1/n} = 20 x 0.69314718^{0.28571429}
          = 20 x 0.90057847 = 18.011569 um
      E_v[d] = d' Gamma(1 + 1/n) = 20 x Gamma(1.28571429) = 20 x 0.89974718
          = 17.994944 um
      d32 = d' / Gamma(1 - 1/n) = 20 / Gamma(0.71428571) = 20 / 1.27599268
          = 15.674072 um

    And at d = d' the cumulative volume undersize is 1 - 1/e = 0.6321206 by construction,
    which is the definition of the location parameter.
    """
    assert math.log(2.0) ** (1.0 / 3.5) == pytest.approx(0.90057847, abs=1e-8)
    assert math.gamma(1.0 + 1.0 / 3.5) == pytest.approx(0.89974718, abs=1e-8)
    assert math.gamma(1.0 - 1.0 / 3.5) == pytest.approx(1.27599268, abs=1e-8)

    p = RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=3.5, density=QUARTZ_DENSITY)
    assert float(p.median().to("um").magnitude) == pytest.approx(18.011569, abs=1e-6)
    assert float(p.volume_weighted_mean().to("um").magnitude) == pytest.approx(
        17.994944, abs=1e-6)
    assert float(p.sauter_d32().to("um").magnitude) == pytest.approx(15.674072, abs=1e-6)
    assert p.cumulative_undersize(Q_(20.0, "um")) == pytest.approx(0.6321206, abs=1e-7)
    assert 1.0 - math.exp(-1.0) == pytest.approx(0.6321206, abs=1e-7)
    assert 1.0 / 3.5 == pytest.approx(0.28571429, abs=1e-8)
    assert math.log(2.0) == pytest.approx(0.69314718, abs=1e-8)
    assert 1.0 - 1.0 / 3.5 == pytest.approx(0.71428571, abs=1e-8)
    assert 1.0 + 1.0 / 3.5 == pytest.approx(1.28571429, abs=1e-8)


# --------------------------------------------------------------------------------------
# benchmark tests
# --------------------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_geometric_ssa_against_ringdalen_bet() -> None:
    """Geometric surface area against Ringdalen 2015 BET measurements.

    Literature: Ringdalen 2015 (doi 10.1007/s11837-014-1149-y) reports BET specific
    surface areas of 0.38 to 0.47 m^2/g on heat-treated natural quartz.

    Model: to produce 0.42 m^2/g (the middle of that band) by geometry alone, equation (8)
    requires a Sauter mean of
      d_32 = 6 / (rho S_m) = 6 / (2650 x 420) = 5.39 um

    Ringdalen's samples were lump quartz of millimetre to centimetre size, whose geometric
    surface area is of order 1e-3 m^2/g. The geometric model therefore understates the
    measured BET area by three to four ORDERS OF MAGNITUDE for that material.

    The error is reported rather than asserted tight because it is not a model defect: it
    quantifies exactly how much of a calcined quartz surface is roughness, microcrack and
    decrepitated inclusion cavity rather than external envelope. Reported this way it is a
    usable number; asserted loosely it would be a false validation.
    """
    bet_mid = Q_(0.42, "m**2/g")
    d32_equiv = 6.0 / (2650.0 * float(bet_mid.to("m**2/kg").magnitude))
    assert d32_equiv == pytest.approx(5.39e-6, rel=5e-3)

    for d_mm, label in [(1.0, "1 mm lump"), (10.0, "10 mm lump")]:
        geom = specific_surface_area(Q_(d_mm, "mm"), QUARTZ_DENSITY)
        geom_m2_g = float(geom.to("m**2/g").magnitude)
        ratio = float(bet_mid.to("m**2/g").magnitude) / geom_m2_g
        error_pct = 100.0 * (geom_m2_g - float(bet_mid.to("m**2/g").magnitude)) / float(
            bet_mid.to("m**2/g").magnitude)
        print(f"\n[benchmark] geometric vs BET specific surface area, {label}"
              f"\n  literature (Ringdalen 2015 BET, midpoint of 0.38 to 0.47): "
              f"{float(bet_mid.to('m**2/g').magnitude):.3f} m^2/g"
              f"\n  model (equation 8, geometric):                             "
              f"{geom_m2_g:.6f} m^2/g"
              f"\n  error: {error_pct:+.2f} percent (BET is {ratio:.0f}x larger)"
              f"\n  equivalent Sauter mean that WOULD give the BET value: "
              f"{d32_equiv * 1e6:.2f} um")
        assert ratio > 100.0, (
            "the geometric model must understate BET by orders of magnitude on lump; if "
            "it did not, equation (8) would be being misread as a BET predictor"
        )
    assert float(bet_mid.to("m**2/kg").magnitude) == pytest.approx(420.0, rel=1e-3)
    geom_1mm = specific_surface_area(Q_(1.0, "mm"), QUARTZ_DENSITY)
    geom_1mm_m2_g = float(geom_1mm.to("m**2/g").magnitude)
    order_of_magnitude = math.floor(math.log10(geom_1mm_m2_g))
    assert order_of_magnitude == -3


@pytest.mark.benchmark
def test_benchmark_rosin_rammler_against_lognormal_sauter_mean() -> None:
    """Cross-validate the two distribution families on a matched volume median.

    Two independent functional forms fitted to the same volume median should give Sauter
    means that agree to within the difference in their tail shapes, which is a real
    consistency check on both closed-form derivations.

    Matched at a volume median of 18.011569 um:
      Rosin-Rammler (d' = 20 um, n = 3.5): d_32 = 15.674072 um
      log-normal (sigma_g chosen so the volume median matches): d_32 computed below

    Error is reported. A disagreement of tens of percent would be expected for very
    different widths; a disagreement of orders of magnitude would mean one of the two
    closed forms is wrong.
    """
    rr = RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=3.5, density=QUARTZ_DENSITY)
    target_median = float(rr.median().to("um").magnitude)
    target_span = rr.span()

    lo, hi = 1.0001, 3.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        trial = LogNormalPSD(d_gn=Q_(1.0, "um"), sigma_g=mid, density=QUARTZ_DENSITY)
        if trial.span() < target_span:
            lo = mid
        else:
            hi = mid
    sigma_g = 0.5 * (lo + hi)
    s = math.log(sigma_g)
    d_gn = target_median / math.exp(3.0 * s * s)
    ln = LogNormalPSD(d_gn=Q_(d_gn, "um"), sigma_g=sigma_g, density=QUARTZ_DENSITY)

    rr_d32 = float(rr.sauter_d32().to("um").magnitude)
    ln_d32 = float(ln.sauter_d32().to("um").magnitude)
    error_pct = 100.0 * (ln_d32 - rr_d32) / rr_d32
    print(f"\n[benchmark] Sauter mean, two distribution families matched on volume "
          f"median and span"
          f"\n  matched volume median: {target_median:.4f} um, span {target_span:.4f}"
          f"\n  Rosin-Rammler (d'=20 um, n=3.5): d_32 = {rr_d32:.4f} um"
          f"\n  log-normal (sigma_g={sigma_g:.4f}):  d_32 = {ln_d32:.4f} um"
          f"\n  error: {error_pct:+.2f} percent")
    assert abs(error_pct) < 15.0, (
        "two closed-form Sauter means matched on median and span must agree to well "
        "within a factor of two; a larger gap indicates an algebra error"
    )
    assert target_median == pytest.approx(18.011569, abs=1e-6)
    assert rr_d32 == pytest.approx(15.674072, abs=1e-6)


# --------------------------------------------------------------------------------------
# physical sanity
# --------------------------------------------------------------------------------------

def test_quantiles_are_monotonic(ln, rr) -> None:
    for p in (ln, rr):
        prev = 0.0
        for frac in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
            d = float(p.quantile(frac).to("m").magnitude)
            assert d > prev
            prev = d


def test_monodisperse_limit(ln) -> None:
    """At sigma_g = 1 every weighted median collapses to the same diameter, span zero."""
    mono = LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.0, density=QUARTZ_DENSITY)
    for w in Weighting:
        assert float(mono.median(w).to("um").magnitude) == pytest.approx(10.0)
    assert float(mono.sauter_d32().to("um").magnitude) == pytest.approx(10.0)
    assert mono.span() == pytest.approx(0.0, abs=1e-12)
    expected = 6.0 / (2650.0 * 10e-6)
    assert float(mono.specific_surface_area().to("m**2/kg").magnitude) == pytest.approx(
        expected, rel=1e-9)


def test_sigma_below_one_rejected() -> None:
    with pytest.raises(ValueError):
        LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=0.9, density=QUARTZ_DENSITY)


def test_negative_diameter_rejected() -> None:
    with pytest.raises(ValueError):
        LogNormalPSD(d_gn=Q_(-10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY)
    with pytest.raises(ValueError):
        RosinRammlerPSD(d_prime=Q_(-20.0, "um"), n=3.5, density=QUARTZ_DENSITY)


def test_cumulative_stays_in_range(rr) -> None:
    for d in (0.0, 0.1, 1.0, 20.0, 100.0, 1e6):
        assert 0.0 <= rr.cumulative_undersize(Q_(d, "um")) <= 1.0


def test_cumulative_is_monotonic(rr) -> None:
    prev = -1.0
    for d in (0.5, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0):
        q = rr.cumulative_undersize(Q_(d, "um"))
        assert q >= prev
        prev = q


def test_quantile_inverts_the_cumulative(rr) -> None:
    """quantile and cumulative_undersize must be exact inverses of each other."""
    for p in (0.05, 0.2, 0.5, 0.8, 0.95):
        d = rr.quantile(p)
        assert rr.cumulative_undersize(d) == pytest.approx(p, abs=1e-12)


def test_sauter_mean_lies_between_the_extreme_quantiles(ln, rr) -> None:
    for p in (ln, rr):
        d32 = float(p.sauter_d32().to("m").magnitude)
        assert float(p.quantile(0.01).to("m").magnitude) < d32
        assert d32 < float(p.quantile(0.99).to("m").magnitude)


def test_surface_area_scales_inversely_with_size(ln) -> None:
    """Halving every diameter must double the specific surface area exactly."""
    fine = LogNormalPSD(d_gn=Q_(5.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY)
    coarse_ssa = float(ln.specific_surface_area().to("m**2/kg").magnitude)
    fine_ssa = float(fine.specific_surface_area().to("m**2/kg").magnitude)
    assert fine_ssa == pytest.approx(2.0 * coarse_ssa, rel=1e-12)


def test_sphericity_below_one_raises_surface_area(ln) -> None:
    """An angular particle has MORE area per unit mass than a sphere of equal volume."""
    psi = Value(quantity=Q_(0.7, "dimensionless"), tag=Tag.ASSUMED,
                basis="test value, not a measurement")
    angular = LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY,
                           sphericity=psi)
    assert (float(angular.specific_surface_area().to("m**2/kg").magnitude)
            > float(ln.specific_surface_area().to("m**2/kg").magnitude))
    assert float(angular.specific_surface_area().to("m**2/kg").magnitude) == pytest.approx(
        float(ln.specific_surface_area().to("m**2/kg").magnitude) / 0.7, rel=1e-12)


def test_sphericity_above_one_rejected() -> None:
    """A sphericity above unity is geometrically impossible."""
    bad = Value(quantity=Q_(1.3, "dimensionless"), tag=Tag.ASSUMED, basis="impossible")
    with pytest.raises(ValueError):
        LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=QUARTZ_DENSITY,
                     sphericity=bad)


def test_rosin_rammler_sauter_requires_n_above_one() -> None:
    """n <= 1 gives a divergent surface area, which must raise rather than return a number."""
    p = RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=0.8, density=QUARTZ_DENSITY)
    with pytest.raises(ValueError, match="requires n > 1"):
        p.sauter_d32()


def test_spread_warning_below_three() -> None:
    """Alderliesten 2013: n < 3 distributions cannot exist, and must be flagged."""
    assert RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=2.0,
                           density=QUARTZ_DENSITY).spread_warning is not None
    assert RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=3.5,
                           density=QUARTZ_DENSITY).spread_warning is None
    warning = RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=2.0,
                              density=QUARTZ_DENSITY).spread_warning
    assert warning is not None and "10.1002/ppsc.201200021" in warning


def test_summary_rejects_non_monotonic_quantiles() -> None:
    with pytest.raises(ValueError, match="not monotonic"):
        PSDSummary(
            weighting=Weighting.VOLUME, d10=Q_(30.0, "um"), d50=Q_(20.0, "um"),
            d90=Q_(10.0, "um"), span=1.0, sauter_d32=Q_(15.0, "um"),
            specific_surface_area=Q_(150.0, "m**2/kg"),
        )


def test_summary_round_trips(ln, rr) -> None:
    for p in (ln.summary(), rr.summary()):
        assert isinstance(p, PSDSummary)
        assert float(p.specific_surface_area.to("m**2/kg").magnitude) > 0.0
        assert p.span > 0.0


def test_span_is_independent_of_weighting_basis(ln) -> None:
    """For a log-normal the span depends only on sigma_g, which is worth pinning."""
    spans = [ln.span(w) for w in Weighting]
    assert max(spans) - min(spans) < 1e-12


def test_negative_density_rejected() -> None:
    with pytest.raises(ValueError):
        specific_surface_area(Q_(15.0, "um"), Q_(-2650.0, "kg/m**3"))


# --------------------------------------------------------------------------------------
# provenance and documentation
# --------------------------------------------------------------------------------------

def test_sphericity_assumption_is_honestly_tagged(ore) -> None:
    """No sphericity measurement exists, so the value must be ASSUMED with a basis."""
    v = feedstock_sphericity_assumption(ore)
    assert v.tag is Tag.ASSUMED
    assert v.basis is not None and "ESTIMATE" in v.basis
    assert "LOWER bound" in v.basis
    assert v.confidence == "low"
    assert float(v.quantity.to("dimensionless").magnitude) == 1.0


def test_sphericity_assumption_requires_a_feedstock() -> None:
    with pytest.raises(TypeError, match="Feedstock"):
        feedstock_sphericity_assumption("Vikarabad")  # type: ignore[arg-type]


def test_sources_have_dois() -> None:
    for src in (psd_mod.SRC_HATCH_CHOATE, psd_mod.SRC_ALDERLIESTEN,
                psd_mod.SRC_RINGDALEN_SSA):
        assert src.doi
        assert src.accessed is not None
        assert src.tier is Tier.T1


def test_limitations_flags_the_bet_gap() -> None:
    doc = psd_mod.__doc__ or ""
    assert "LIMITATIONS" in doc
    assert "NOT A BET SURFACE AREA" in doc
    assert "UNDERSTATE" in doc
