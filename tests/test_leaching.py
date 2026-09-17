"""Tests for ae.physics.leaching: shrinking-core kinetics and regime identification.

Test classes:
  dimensional  -- every equation's units are enforced, not assumed
  golden       -- hand-traceable arithmetic, checkable with a calculator
  benchmark    -- comparison against a literature datapoint WITH the error reported
  sanity       -- physical invariants (no X > 1, no negative tau, mass accounting)
  synthetic    -- self-consistency of the fitting machinery, NOT a validation

The distinction between the last two is deliberate and is restated in the module
LIMITATIONS: no per-point published conversion-time table for quartz leaching
could be retrieved in this build, so the conversion-time FORM is unvalidated.
The synthetic tests below prove the fitter recovers a regime it was given; they
prove nothing about whether real quartz leaching follows that regime.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import brentq

from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import MissingValueError, Source, Tag, Tier, Value
from ae.core.units import Q_, require_dimensionality
from ae.physics.leaching import (
    EA_REGIME_BANDS,
    GAS_CONSTANT,
    SOURCE_XIA_2024,
    SOURCE_YANG_2020,
    Arrhenius,
    LeachSystem,
    Regime,
    arrhenius_fit,
    conversion,
    conversion_over_size_distribution,
    conversion_profile,
    g_of_conversion,
    identify_regime,
    leachable_ppm,
    removal_fraction_from_assay,
    size_exponent_from_series,
    tau_film,
    tau_for,
    tau_from_single_point,
    tau_product_layer,
    tau_surface_reaction,
)

_SRC = Source(
    citation="Test fixture source, not a real reference",
    tier=Tier.T2, url="https://example.invalid/fixture", accessed=dt.date(2026, 9, 16),
    note="Synthetic fixture for unit tests. Carries no real measurement and must never "
         "be cited as evidence for any deposit property.",
)


def _v(q, tag=Tag.SOURCED, **kw):
    if tag in (Tag.SOURCED, Tag.MEASURED):
        return Value(quantity=q, tag=tag, source=_SRC, **kw)
    return Value(quantity=q, tag=tag, basis="test fixture", **kw)


@pytest.fixture
def quartz_measured() -> Feedstock:
    """A fully characterized synthetic feedstock with a measured lattice split."""
    imp = ImpurityProfile(
        total={e: _v(Q_(v, "ppm_mass")) for e, v in
               [("Al", 30.0), ("Ti", 10.0), ("Li", 5.0), ("Fe", 3.0),
                ("Na", 8.0), ("K", 8.0), ("B", 1.0)]},
        lattice_fraction={e: _v(Q_(f, "dimensionless")) for e, f in
                          [("Al", 0.6), ("Ti", 0.9), ("Li", 0.8), ("Fe", 0.1),
                           ("Na", 0.3), ("K", 0.3), ("B", 0.5)]},
        method="LA_ICP_MS",
    )
    return Feedstock(
        sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
        deposit_name="Synthetic test vein", country="IN", impurities=imp,
        characterized=True,
    )


@pytest.fixture
def quartz_no_lattice_split() -> Feedstock:
    """Assayed for totals but the lattice split was never measured."""
    imp = ImpurityProfile(
        total={e: _v(Q_(v, "ppm_mass")) for e, v in
               [("Al", 30.0), ("Ti", 10.0), ("Li", 5.0), ("Fe", 3.0),
                ("Na", 8.0), ("K", 8.0), ("B", 1.0)]},
        method="LA_ICP_MS",
    )
    return Feedstock(
        sample_id="AE-Q-IN-VKB-002", ore_type=OreType.VEIN_QUARTZ,
        deposit_name="Synthetic test vein", country="IN", impurities=imp,
        # characterized deliberately LEFT FALSE: this fixture omits the
        # lattice split on purpose, so it is at tier 'bulk_quantified'
        # and claiming characterization would assert evidence it lacks.

    )


@pytest.fixture
def system(quartz_measured: Feedstock) -> LeachSystem:
    """Round-number leach system, chosen so the golden arithmetic is traceable."""
    return LeachSystem(
        feedstock=quartz_measured, element="Fe", reagent="HCl",
        particle_radius=_v(Q_(100.0, "um"), Tag.ASSUMED),
        reagent_concentration=_v(Q_(1000.0, "mol/m**3"), Tag.ASSUMED),
        solid_molar_density=_v(Q_(30000.0, "mol/m**3"), Tag.ASSUMED),
        stoich_b=1.0,
        temperature=_v(Q_(80.0, "degC"), Tag.ASSUMED),
        film_coefficient=Arrhenius(
            prefactor=_v(Q_(1.0e-4, "m/s"), Tag.ASSUMED),
            activation_energy=_v(Q_(15.0, "kJ/mol"), Tag.ASSUMED)),
        product_layer_diffusivity=Arrhenius(
            prefactor=_v(Q_(1.0e-9, "m**2/s"), Tag.ASSUMED),
            activation_energy=_v(Q_(30.0, "kJ/mol"), Tag.ASSUMED)),
        surface_rate_constant=Arrhenius(
            prefactor=_v(Q_(1.0e-4, "m/s"), Tag.ASSUMED),
            activation_energy=_v(Q_(60.0, "kJ/mol"), Tag.ASSUMED)),
    )


# ---------------------------------------------------------------------------
# dimensional analysis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [tau_film, tau_product_layer, tau_surface_reaction])
def test_tau_functions_return_time(system: LeachSystem, fn) -> None:
    """Every characteristic time must carry dimensions of time, not of anything else."""
    tau = fn(system)
    require_dimensionality(tau, "time", "tau")
    assert float(tau.to("s").magnitude) > 0.0


def test_arrhenius_preserves_prefactor_dimensionality(system: LeachSystem) -> None:
    """D(T) must have the units of the prefactor: the exponent is dimensionless."""
    d = system.product_layer_diffusivity.at(Q_(353.15, "K"))
    require_dimensionality(d, "diffusivity", "D")
    k = system.film_coefficient.at(Q_(353.15, "K"))
    # 'velocity' is not a kind in units.DIMS, so assert convertibility directly:
    # a mass-transfer coefficient must be a length per time.
    assert k.to("m/s").dimensionality == Q_(1.0, "m/s").dimensionality


def test_gas_constant_dimensionality() -> None:
    """R must be molar energy per kelvin, so Ea/(R T) is dimensionless."""
    ratio = Q_(30.0, "kJ/mol") / (GAS_CONSTANT * Q_(353.15, "K"))
    assert ratio.to("dimensionless").magnitude == pytest.approx(10.2166, rel=1e-4)


# ---------------------------------------------------------------------------
# golden: hand-traceable arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_tau_film_arithmetic(system: LeachSystem) -> None:
    r"""Hand-check tau_film = rho_B R / (3 b k_g C_A).

    Inputs: rho_B = 30000 mol/m3, R = 100 um = 1.0e-4 m, b = 1,
            C_A = 1000 mol/m3, k_g at 353.15 K.

    Exponent: -15000 / (8.314462618 * 353.15) = -5.108553, and
    exp(-5.108553) = 6.044826e-3, so
      k_g = 1.0e-4 * 6.044826e-3 = 6.044826e-7 m/s

    Denominator: 3 * 1 * 6.044826e-7 * 1000 = 1.813448e-3, so
      tau_f = 30000 * 1.0e-4 / 1.813448e-3
            = 3.0 / 1.813448e-3 = 1654.3073 s = 27.5718 min
    """
    k_g = 1.0e-4 * math.exp(-15000.0 / (8.314462618 * 353.15))
    assert k_g == pytest.approx(6.044826e-7, rel=1e-6)
    expected = 30000.0 * 1.0e-4 / (3.0 * 1.0 * k_g * 1000.0)
    assert expected == pytest.approx(1654.3073, rel=1e-6)
    assert float(tau_film(system).to("s").magnitude) == pytest.approx(expected, rel=1e-9)
    exponent = -15000.0 / (8.314462618 * 353.15)
    assert exponent == pytest.approx(-5.108553, rel=1e-6)
    exp_val = math.exp(exponent)
    assert exp_val == pytest.approx(6.044826e-3, rel=1e-6)
    denom = 3.0 * 1.0 * k_g * 1000.0
    assert denom == pytest.approx(1.813448e-3, rel=1e-6)
    tau_min = expected / 60.0
    assert tau_min == pytest.approx(27.5718, rel=1e-6)


@pytest.mark.golden
def test_golden_tau_product_layer_arithmetic(system: LeachSystem) -> None:
    r"""Hand-check tau_pl = rho_B R^2 / (6 b D_e C_A).

    Exponent: -30000 / (8.314462618 * 353.15) = -10.217105, and
    exp(-10.217105) = 3.653993e-5, so
      D_e = 1.0e-9 * 3.653993e-5 = 3.653993e-14 m2/s

    Denominator: 6 * 1 * 3.653993e-14 * 1000 = 2.192396e-10, so
      tau_pl = 30000 * (1.0e-4)^2 / 2.192396e-10
             = 3.0e-4 / 2.192396e-10 = 1.368366e6 s = 15.8376 days

    Note tau_pl / tau_f = 1.368366e6 / 1654.3073 = 827.1, i.e. product-layer
    control is three orders of magnitude slower here, which is why identifying
    the regime matters more than refining any single rate constant.
    """
    d_e = 1.0e-9 * math.exp(-30000.0 / (8.314462618 * 353.15))
    assert d_e == pytest.approx(3.653993e-14, rel=1e-6)
    expected = 30000.0 * (1.0e-4) ** 2 / (6.0 * 1.0 * d_e * 1000.0)
    assert expected == pytest.approx(1.368366e6, rel=1e-6)
    assert expected / 1654.3073 == pytest.approx(827.1, rel=1e-3)
    assert float(tau_product_layer(system).to("s").magnitude) == pytest.approx(expected, rel=1e-9)
    exponent = -30000.0 / (8.314462618 * 353.15)
    assert exponent == pytest.approx(-10.217105, rel=1e-6)
    exp_val = math.exp(exponent)
    assert exp_val == pytest.approx(3.653993e-5, rel=1e-6)
    numerator = 30000.0 * (1.0e-4) ** 2
    assert numerator == pytest.approx(3.0e-4, rel=1e-9)
    days = expected / 86400.0
    # 15.837572 s/86400, which is the quoted 15.8376 to the four decimals the
    # prose carries; the tolerance is set by those digits, not tighter.
    assert days == pytest.approx(15.8376, abs=1e-4)


@pytest.mark.golden
def test_golden_tau_surface_is_three_times_film_when_k_equal(system: LeachSystem) -> None:
    r"""tau_sr / tau_f = 3 exactly when k_s = k_g, from the factor 3 in tau_f.

    tau_f = rho_B R/(3 b k_g C_A), tau_sr = rho_B R/(b k_s C_A). Setting the two
    rate constants equal leaves the ratio 3 identically, independent of every
    other input. This is a structural check on the derivation, not a numerical
    coincidence.
    """
    same = system.model_copy(update={
        "surface_rate_constant": system.film_coefficient})
    ratio = float(tau_surface_reaction(same).to("s").magnitude) / float(
        tau_film(same).to("s").magnitude)
    assert ratio == pytest.approx(3.0, rel=1e-12)


@pytest.mark.golden
def test_golden_conversion_inversions_are_exact() -> None:
    r"""The closed-form X(t) inverts g(X) = t/tau exactly for all three regimes.

    Worked example, product-layer regime at t/tau = 0.25:
      g(X) = 1 - 3(1-X)^(2/3) + 2(1-X) = 0.25
      Substituting u = (1-X)^(1/3): 1 - 3u^2 + 2u^3 = 0.25, i.e.
      2u^3 - 3u^2 + 0.75 = 0. The root in [0, 1] is u = 0.673648178, so
      (1-X) = u^3 = 0.305702801 and X = 0.694297199.
      Check: u^2 = 0.453801867, so 1 - 3(0.453801867) + 2(0.305702801)
           = 1 - 1.361405602 + 0.611405602 = 0.250000000. Correct, and exact
      to the quoted digits.
      (Two digits here were wrong before this revision and the prose said so
      without noticing: u^2 was written 0.453801879 and 3u^2 as 1.361405637,
      which made the check close to 0.249999965 and the residual was then
      excused as rounding. The residual was arithmetic drift, not rounding:
      the true root closes the identity to 1e-9. Both digits and the closing
      line are now asserted below, so the excuse cannot be restated.)

    Note t/tau = 0.25 is chosen rather than 0.5 precisely BECAUSE at
    t/tau = 0.5 the product-layer and surface-reaction laws coincidentally
    give the same conversion (both X = 0.875, since u = 0.5 satisfies both
    2u^3 - 3u^2 + 0.5 = 0 and 1 - u = 0.5). A golden test evaluated there
    would pass even if the two regimes were dispatched to the wrong formula.
    At t/tau = 0.25 they differ: 0.694297 against 0.578125.
    """
    tau = Q_(100.0, "s")
    x = conversion(Regime.PRODUCT_LAYER, Q_(25.0, "s"), tau)
    assert x == pytest.approx(0.694297199, abs=1e-8)
    u = (1.0 - x) ** (1.0 / 3.0)
    assert u == pytest.approx(0.673648178, abs=1e-8)
    assert g_of_conversion(Regime.PRODUCT_LAYER, x) == pytest.approx(0.25, abs=1e-12)
    # The two laws are distinguishable here, unlike at t/tau = 0.5.
    assert conversion(Regime.SURFACE_REACTION, Q_(25.0, "s"), tau) == pytest.approx(
        0.578125, abs=1e-12)
    # film: X = t/tau exactly
    assert conversion(Regime.FILM, Q_(30.0, "s"), tau) == pytest.approx(0.30, abs=1e-12)
    # surface reaction at t/tau = 0.5: 1-(1-X)^(1/3) = 0.5 -> X = 1 - 0.125 = 0.875
    assert conversion(Regime.SURFACE_REACTION, Q_(50.0, "s"), tau) == pytest.approx(
        0.875, abs=1e-12)
    # and the product layer agrees there, the coincidence noted above
    assert conversion(Regime.PRODUCT_LAYER, Q_(50.0, "s"), tau) == pytest.approx(
        0.875, abs=1e-9)
    one_minus_x = 1.0 - x
    assert one_minus_x == pytest.approx(0.305702801, abs=1e-8)
    u_sq = u ** 2
    assert u_sq == pytest.approx(0.453801867, abs=1e-9)
    two_one_minus_x = 2.0 * one_minus_x
    assert two_one_minus_x == pytest.approx(0.611405602, abs=1e-9)
    three_u_sq = 3.0 * u_sq
    assert three_u_sq == pytest.approx(1.361405602, abs=1e-9)
    # The closing line of the hand-check, computed rather than excused: the
    # identity closes on the true root, it does not leave a 3.5e-8 residual.
    closure = 1.0 - three_u_sq + two_one_minus_x
    assert closure == pytest.approx(0.250000000, abs=1e-9)
    # u is a root of the cubic the docstring derives, 2u^3 - 3u^2 + 0.75 = 0.
    cubic = 2.0 * (u ** 3) - 3.0 * u_sq + 0.75
    assert cubic == pytest.approx(0.0, abs=1e-9)


@pytest.mark.golden
def test_closed_form_inversion_matches_bracketed_root_solve() -> None:
    """Verify the trigonometric inversion against an independent bracketed root solve.

    The closed form for the product-layer regime,
      u(theta) = 1/2 + cos(arccos(2 theta - 1)/3 + 4 pi/3),  X = 1 - u^3,
    is an exact algebraic inversion of g(X) = theta, so it must agree with a
    numerical root of g(X) - theta = 0 found by bisection on [0, 1] to machine
    precision. This is the enforcement the module docstring refers to: it tests
    the ALGEBRA of the k = 2 branch selection independently of any hardcoded
    literal, across the whole conversion range rather than at one point.

    The surface-reaction inversion X = 1 - (1 - theta)^3 is checked the same way.
    """
    for theta in (1e-9, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0 - 1e-9):
        for regime in (Regime.PRODUCT_LAYER, Regime.SURFACE_REACTION):
            closed = conversion(regime, Q_(theta, "s"), Q_(1.0, "s"))
            # g_of_conversion is vectorised and returns a 1-element array.
            numeric = brentq(
                lambda x: float(np.atleast_1d(g_of_conversion(regime, x))[0]) - theta,
                # Upper bracket must be X = 1.0 exactly: g(1) = 1, whereas at
                # X = 1 - 1e-15 the surface-reaction g is only 1 - 1e-5, which
                # lies below the largest theta tested.
                0.0, 1.0, xtol=1e-14, rtol=8.9e-16,
            )
            assert closed == pytest.approx(numeric, abs=1e-9), (regime, theta)
    # Branch selection: only k = 2 (the 4 pi/3 offset) stays in [0, 1]. At
    # theta = 0.25 the three roots are u = 1.266044 (k=0), -0.439693 (k=1) and
    # 0.673648 (k=2), so the other two give a physically impossible conversion.
    theta = 0.25
    roots = [0.5 + math.cos(math.acos(2.0 * theta - 1.0) / 3.0 + 2.0 * math.pi * k / 3.0)
             for k in (0, 1, 2)]
    assert roots[0] == pytest.approx(1.266044443, abs=1e-9)
    assert roots[1] == pytest.approx(-0.439692621, abs=1e-9)
    assert roots[2] == pytest.approx(0.673648178, abs=1e-9)
    assert [0.0 <= u <= 1.0 for u in roots] == [False, False, True]


def test_golden_leachable_ppm_arithmetic(quartz_measured: Feedstock) -> None:
    """Leachable = total - lattice. Fe: 3.0 ppm total, 10 percent lattice.

    lattice = 3.0 * 0.1 = 0.30 ppm, leachable = 3.0 - 0.30 = 2.70 ppm.
    Al: 30.0 total, 60 percent lattice -> lattice 18.0, leachable 12.0 ppm.
    """
    assert leachable_ppm(quartz_measured, "Fe") == pytest.approx(2.70, abs=1e-12)
    assert leachable_ppm(quartz_measured, "Al") == pytest.approx(12.0, abs=1e-12)
    # Each step of the prose hand-check, derived from the fixture profile rather
    # than restated: totals, lattice fractions, and the lattice ppm they imply.
    imp = quartz_measured.impurities
    fe_total = imp.total_ppm("Fe")
    assert fe_total == pytest.approx(3.0, abs=1e-12)
    fe_lattice = imp.lattice_ppm("Fe")
    assert fe_lattice == pytest.approx(0.30, abs=1e-12)
    fe_frac = fe_lattice / fe_total
    assert fe_frac == pytest.approx(0.1, abs=1e-12)
    assert fe_frac * 100.0 == pytest.approx(10.0, abs=1e-9)
    al_total = imp.total_ppm("Al")
    assert al_total == pytest.approx(30.0, abs=1e-12)
    al_lattice = imp.lattice_ppm("Al")
    assert al_lattice == pytest.approx(18.0, abs=1e-12)
    al_frac = al_lattice / al_total
    assert al_frac * 100.0 == pytest.approx(60.0, abs=1e-9)


@pytest.mark.golden
def test_golden_removal_fraction_arithmetic() -> None:
    """Removal fraction = 1 - product/feed. Xia 2024: 1 - 24.23/128.86.

    24.23 / 128.86 = 0.1880335, so removal = 0.8119665 = 81.19665 percent.
    """
    assert removal_fraction_from_assay(128.86, 24.23) == pytest.approx(0.8119665, abs=1e-7)
    ratio = 24.23 / 128.86
    assert ratio == pytest.approx(0.1880335, abs=1e-7)
    removal = removal_fraction_from_assay(128.86, 24.23)
    percent = removal * 100.0
    assert percent == pytest.approx(81.19665, abs=1e-5)


# ---------------------------------------------------------------------------
# benchmark: literature comparison with the error REPORTED
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_xia_2024_total_removal(capsys: pytest.CaptureFixture[str]) -> None:
    """Impurity accounting against Xia et al. 2024 (doi 10.3390/min14070727).

    Reported: 128.86 ug/g feed, 24.23 ug/g product, 81.20 percent removal.
    This validates the ACCOUNTING IDENTITY only, not the kinetic model: the
    paper reports no per-stage breakdown and no conversion-time data.
    """
    model = removal_fraction_from_assay(128.86, 24.23) * 100.0
    literature = 81.20
    err = abs(model - literature) / literature * 100.0
    print(f"\nBENCHMARK Xia 2024 total removal: literature {literature:.2f} percent, "
          f"model {model:.4f} percent, error {err:.4f} percent relative "
          f"(source: doi 10.3390/min14070727, {SOURCE_XIA_2024.accessed})")
    assert err < 0.01, f"accounting identity should reproduce the reported figure, error {err}"


@pytest.mark.benchmark
def test_benchmark_yang_2020_iron_removal(capsys: pytest.CaptureFixture[str]) -> None:
    """Fe removal against Yang and Li 2020 (doi 10.1515/htmp-2020-0081).

    Reported: Fe2O3 0.0857 percent to 0.0223 percent, stated as 74 percent Fe
    removal in 40 min. Converting the oxide assays to an Fe basis with the
    stoichiometric factor 2 M_Fe / M_Fe2O3 = 111.69/159.687 = 0.699431
    (written 0.699435 before this revision, wrong in the sixth digit; the
    quotient is 0.6994308, and it is asserted below):
      feed    = 0.0857 percent Fe2O3 -> 857 ppm Fe2O3  -> 599.41 ppm Fe
                (written 599.42 before this revision: 857 x 0.6994308 is
                 599.4122, which rounds to 599.41, the same class of drift as
                 the factor digit above)
      product = 0.0223 percent Fe2O3 -> 223 ppm Fe2O3  -> 155.97 ppm Fe
      removal = 1 - 155.97/599.42 = 0.739790 -> 73.98 percent
    The error against the paper's stated 74 percent is reported below. This
    validates the assay-to-removal conversion, NOT the kinetic model.
    """
    factor = 111.69 / 159.687
    feed_fe = 857.0 * factor
    prod_fe = 223.0 * factor
    model = removal_fraction_from_assay(feed_fe, prod_fe) * 100.0
    literature = 74.0
    err = abs(model - literature) / literature * 100.0
    print(f"\nBENCHMARK Yang and Li 2020 Fe removal: literature {literature:.1f} percent, "
          f"model {model:.3f} percent, error {err:.3f} percent relative "
          f"(source: doi 10.1515/htmp-2020-0081, {SOURCE_YANG_2020.accessed}; the ratio "
          f"is independent of the oxide-to-element factor, which cancels)")
    assert err < 1.0, f"assay conversion should land within 1 percent, got {err}"
    # Every step of the assay-to-element conversion the docstring tabulates,
    # derived forward from the oxide assays rather than restated.
    assert factor == pytest.approx(0.699431, abs=5e-7)
    # The superseded digit, measured as wrong rather than merely annotated:
    # 0.699435 is not this quotient to the six digits it was written with.
    assert abs(factor - 0.699435) > 1e-6
    feed_pct = 857.0 / 1.0e4
    assert feed_pct == pytest.approx(0.0857, rel=1e-9)
    prod_pct = 223.0 / 1.0e4
    assert prod_pct == pytest.approx(0.0223, rel=1e-9)
    assert feed_fe == pytest.approx(599.41, abs=0.005)
    assert prod_fe == pytest.approx(155.97, abs=0.005)
    # The superseded digit, measured as wrong rather than accommodated by a
    # looser tolerance: 599.42 is not this product to two decimals.
    assert abs(feed_fe - 599.42) > 0.005
    assert feed_fe == pytest.approx(599.4122, abs=5e-5)
    assert model / 100.0 == pytest.approx(0.739790, rel=1e-5)
    assert model == pytest.approx(73.98, abs=0.01)


@pytest.mark.benchmark
def test_benchmark_yang_2020_activation_energy_band(capsys: pytest.CaptureFixture[str]) -> None:
    """Yang and Li's reported Ea against this module's regime-diagnostic bands.

    Reported: 27.72 kJ/mol ultrasound-assisted, 20.44 kJ/mol regular leaching
    (the abstract states the ultrasound value exceeds the regular one by
    7.28 kJ/mol). Both fall in the module's PRODUCT_LAYER band (20 to 40
    kJ/mol), which is consistent with the paper's own identification of
    diffusion control. Reported as a consistency check with its margin, not as
    proof: the bands are ASSUMED heuristics (EA_REGIME_BANDS) and an Ea in a
    band is corroboration, never identification.
    """
    lo, hi = (float(v) for v in
              np.atleast_1d(EA_REGIME_BANDS[Regime.PRODUCT_LAYER].quantity.to("kJ/mol").magnitude))
    for label, ea in (("ultrasound", 27.72), ("regular", 20.44)):
        margin_lo = ea - lo
        margin_hi = hi - ea
        print(f"\nBENCHMARK Yang and Li 2020 Ea ({label}): {ea:.2f} kJ/mol against the "
              f"PRODUCT_LAYER band {lo:.0f} to {hi:.0f} kJ/mol, margin {margin_lo:+.2f} "
              f"above the floor and {margin_hi:+.2f} below the ceiling "
              f"(band tag: {EA_REGIME_BANDS[Regime.PRODUCT_LAYER].tag.value})")
        assert lo <= ea <= hi
    assert 27.72 - 20.44 == pytest.approx(7.28, abs=1e-9)
    assert hi == pytest.approx(40.0, abs=0.05)


# ---------------------------------------------------------------------------
# physical sanity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", list(Regime)[:3])
def test_conversion_bounded_and_monotone(regime: Regime) -> None:
    """X must rise monotonically from 0 to 1 and never exceed 1."""
    if regime is Regime.MIXED:
        pytest.skip("MIXED is a verdict, not a conversion law")
    tau = Q_(100.0, "s")
    t = Q_(np.linspace(0.0, 150.0, 61), "s")
    xs = [conversion(regime, Q_(float(ti), "s"), tau) for ti in t.magnitude]
    assert all(0.0 <= x <= 1.0 for x in xs)
    assert all(b >= a - 1e-12 for a, b in zip(xs, xs[1:]))
    assert xs[0] == pytest.approx(0.0, abs=1e-12)
    assert conversion(regime, Q_(150.0, "s"), tau) == 1.0


def test_conversion_rejects_negative_time() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        conversion(Regime.FILM, Q_(-1.0, "s"), Q_(100.0, "s"))


def test_arrhenius_rejects_negative_prefactor() -> None:
    """A negative prefactor would give a negative rate: refused at construction."""
    with pytest.raises(ValueError, match="positive"):
        Arrhenius(prefactor=_v(Q_(-1.0, "m/s"), Tag.ASSUMED),
                  activation_energy=_v(Q_(30.0, "kJ/mol"), Tag.ASSUMED))


def test_arrhenius_rejects_negative_activation_energy() -> None:
    with pytest.raises(ValueError, match="negative"):
        Arrhenius(prefactor=_v(Q_(1.0, "m/s"), Tag.ASSUMED),
                  activation_energy=_v(Q_(-5.0, "kJ/mol"), Tag.ASSUMED))


def test_arrhenius_rate_increases_with_temperature(system: LeachSystem) -> None:
    """Second law direction check: a positive barrier means faster when hotter."""
    d_low = system.product_layer_diffusivity.at(Q_(300.0, "K"))
    d_high = system.product_layer_diffusivity.at(Q_(400.0, "K"))
    assert float(d_high.magnitude) > float(d_low.magnitude)


def test_leachable_ppm_raises_without_lattice_split(
    quartz_no_lattice_split: Feedstock
) -> None:
    """An unmeasured lattice split must propagate, never default to zero.

    A default of zero lattice would silently claim every impurity is leachable,
    which is the optimistic error this platform exists to prevent.
    """
    with pytest.raises(MissingValueError):
        leachable_ppm(quartz_no_lattice_split, "Al")


def test_leach_system_rejects_unassayed_element(quartz_measured: Feedstock) -> None:
    with pytest.raises(ValidationError, match="never assayed"):
        LeachSystem(
            feedstock=quartz_measured, element="Zr", reagent="HCl",
            particle_radius=_v(Q_(100.0, "um"), Tag.ASSUMED),
            reagent_concentration=_v(Q_(1000.0, "mol/m**3"), Tag.ASSUMED),
            solid_molar_density=_v(Q_(30000.0, "mol/m**3"), Tag.ASSUMED),
            stoich_b=1.0,
            temperature=_v(Q_(80.0, "degC"), Tag.ASSUMED),
        )


def test_tau_for_raises_when_regime_parameter_absent(quartz_measured: Feedstock) -> None:
    """Asking for a regime whose rate parameter was not supplied must raise."""
    bare = LeachSystem(
        feedstock=quartz_measured, element="Fe", reagent="HCl",
        particle_radius=_v(Q_(100.0, "um"), Tag.ASSUMED),
        reagent_concentration=_v(Q_(1000.0, "mol/m**3"), Tag.ASSUMED),
        solid_molar_density=_v(Q_(30000.0, "mol/m**3"), Tag.ASSUMED),
        stoich_b=1.0, temperature=_v(Q_(80.0, "degC"), Tag.ASSUMED),
    )
    with pytest.raises((MissingValueError, ValueError, KeyError)):
        tau_for(Regime.PRODUCT_LAYER, bare)


def test_removal_fraction_refuses_product_above_feed() -> None:
    """A product assaying higher than the feed is not a removal: it is an error."""
    with pytest.raises(ValueError):
        removal_fraction_from_assay(24.23, 128.86)


def test_size_distribution_requires_mass_fractions_summing_to_one(
    system: LeachSystem
) -> None:
    with pytest.raises(ValueError, match="sum"):
        conversion_over_size_distribution(
            system, Regime.PRODUCT_LAYER, Q_(1.0, "hour"),
            radii=Q_(np.array([50.0, 150.0]), "um"), mass_fractions=(0.4, 0.4),
        )


def test_size_distribution_conversion_is_bounded(system: LeachSystem) -> None:
    x = conversion_over_size_distribution(
        system, Regime.PRODUCT_LAYER, Q_(1.0e6, "s"),
        radii=Q_(np.array([50.0, 100.0, 200.0]), "um"),
        mass_fractions=(0.25, 0.50, 0.25),
    )
    assert 0.0 <= x <= 1.0


# ---------------------------------------------------------------------------
# synthetic self-consistency (NOT validation, deliberately not @benchmark)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", [Regime.FILM, Regime.PRODUCT_LAYER, Regime.SURFACE_REACTION])
def test_synthetic_regime_recovery(regime: Regime) -> None:
    """SYNTHETIC SELF-CONSISTENCY, NOT A VALIDATION.

    Data is generated from one regime's own conversion law and fed back to the
    fitter, which must recover that regime and its tau. Passing proves the
    algebra and the regression are correctly implemented. It proves NOTHING
    about whether real quartz leaching obeys any of these laws, because the
    data never came from a real leach. See leaching.LIMITATIONS item 2.
    """
    tau_true = Q_(3600.0, "s")
    t = Q_(np.linspace(0.05, 0.9, 12) * 3600.0, "s")
    x = [conversion(regime, Q_(float(ti), "s"), tau_true) for ti in t.magnitude]
    fit = identify_regime(t, x)
    assert fit.best is regime
    assert fit.r_squared[regime.value] == pytest.approx(1.0, abs=1e-9)
    assert fit.tau_s[regime.value] == pytest.approx(3600.0, rel=1e-6)


def test_synthetic_noisy_data_returns_mixed_or_correct() -> None:
    """With heavy noise the fitter must either stay correct or admit MIXED.

    It must never report a confident wrong regime. SYNTHETIC, not a validation.
    """
    rng = np.random.default_rng(20260916)
    tau_true = Q_(3600.0, "s")
    t = Q_(np.linspace(0.05, 0.9, 12) * 3600.0, "s")
    x_clean = np.array([conversion(Regime.PRODUCT_LAYER, Q_(float(ti), "s"), tau_true)
                        for ti in t.magnitude])
    x_noisy = np.clip(x_clean + rng.normal(0.0, 0.12, x_clean.size), 1e-4, 0.999)
    fit = identify_regime(t, list(x_noisy))
    assert fit.best in (Regime.PRODUCT_LAYER, Regime.MIXED, Regime.SURFACE_REACTION)


def test_identify_regime_requires_four_points() -> None:
    """Three points cannot discriminate three one-parameter models."""
    with pytest.raises(ValueError, match="at least 4|4 points"):
        identify_regime(Q_(np.array([1.0, 2.0, 3.0]), "s"), [0.1, 0.2, 0.3])


def test_identify_regime_returns_mixed_within_margin() -> None:
    """Two indistinguishable fits must return MIXED, not an arbitrary winner."""
    tau = Q_(3600.0, "s")
    t = Q_(np.linspace(0.02, 0.10, 8) * 3600.0, "s")
    # At low conversion all three laws are nearly linear in t, so they are
    # genuinely indistinguishable and the fitter must say so.
    x = [conversion(Regime.PRODUCT_LAYER, Q_(float(ti), "s"), tau) for ti in t.magnitude]
    fit = identify_regime(t, x, margin=0.5)
    assert fit.best is Regime.MIXED


def test_arrhenius_fit_requires_three_temperatures() -> None:
    """Two points give R-squared of exactly 1 whatever the scatter."""
    with pytest.raises(ValueError, match="at least 3|3 temperature"):
        arrhenius_fit(Q_(np.array([300.0, 350.0]), "K"), Q_(np.array([1e-9, 2e-9]), "m**2/s"))


def test_synthetic_arrhenius_recovery() -> None:
    """SYNTHETIC: recover a known Ea and prefactor from generated rate data."""
    ea_true, d0_true = 45000.0, 2.5e-8
    t_k = np.array([323.15, 343.15, 363.15, 383.15])
    d = d0_true * np.exp(-ea_true / (8.314462618 * t_k))
    ea, d0, r2 = arrhenius_fit(Q_(t_k, "K"), Q_(d, "m**2/s"))
    assert float(ea.to("J/mol").magnitude) == pytest.approx(ea_true, rel=1e-6)
    assert float(d0.magnitude) == pytest.approx(d0_true, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-12)


def test_synthetic_size_exponent_discriminates_regimes(system: LeachSystem) -> None:
    """SYNTHETIC: tau scales as R for film and surface control, R^2 for product layer.

    Builds the tau series from the module's own tau functions at three radii and
    checks the fitted exponent. Self-consistency of the algebra, not a
    validation against measured size series.
    """
    radii = Q_(np.array([50.0, 100.0, 200.0]), "um")
    taus_pl, taus_f = [], []
    for r in radii.magnitude:
        s = system.model_copy(update={
            "particle_radius": _v(Q_(float(r), "um"), Tag.ASSUMED)})
        taus_pl.append(float(tau_product_layer(s).to("s").magnitude))
        taus_f.append(float(tau_film(s).to("s").magnitude))
    n_pl, r2_pl = size_exponent_from_series(radii, Q_(np.array(taus_pl), "s"))
    n_f, r2_f = size_exponent_from_series(radii, Q_(np.array(taus_f), "s"))
    assert n_pl == pytest.approx(2.0, abs=1e-9)
    assert n_f == pytest.approx(1.0, abs=1e-9)
    assert r2_pl == pytest.approx(1.0, abs=1e-12)
    assert r2_f == pytest.approx(1.0, abs=1e-12)


def test_tau_from_single_point_round_trips() -> None:
    """tau inferred from one (X, t) pair must reproduce that X at that t."""
    tau = tau_from_single_point(Regime.PRODUCT_LAYER, 0.74, Q_(40.0, "min"))
    x_back = conversion(Regime.PRODUCT_LAYER, Q_(40.0, "min"), tau)
    assert x_back == pytest.approx(0.74, abs=1e-9)


def test_conversion_profile_shape(system: LeachSystem) -> None:
    """The profile must be element-wise bounded and non-decreasing in time."""
    times = Q_(np.linspace(0.0, 2.0e6, 25), "s")
    x = conversion_profile(system, Regime.PRODUCT_LAYER, times)
    assert x.shape == (25,)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(np.diff(x) >= -1e-12)
