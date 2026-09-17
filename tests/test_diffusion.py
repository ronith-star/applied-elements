"""Tests for ae.physics.diffusion: Fick transport and the lattice purity ceiling.

The central assertion under test is NOT "lattice impurities are immobile". It is
the narrower, defensible claim the module actually makes: at atmospheric acid
leaching temperatures, and for Ti at chlorination-roasting temperature, the
diffusion length falls short of a grain radius by orders of magnitude across the
entire plausible parameter range, while the Al case at roasting temperature is
NOT decidable from a diffusion-length argument with the data available. Tests
below pin both the positive and the negative half of that, so a later change
that quietly turns the undecided case into a confident one will fail here.
"""

from __future__ import annotations

import datetime as dt
import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import MissingValueError, Source, Tag, Tier, Value
from ae.core.units import Q_, require_dimensionality
from ae.physics.diffusion import (
    D0_SWEEP_M2_S,
    DIFFUSIVITY_STATUS,
    ERF_HALF,
    FO_SHORT_TIME_SWITCH,
    GAS_CONSTANT,
    LIU_2026_EA_RANGE_KJ,
    MOST_MOBILE_BOUND,
    TI_LATTICE_BOUND,
    ArrheniusDiffusivity,
    Verdict,
    critical_activation_energy,
    diffusion_length,
    diffusion_length_table,
    diffusivity,
    erfc_profile,
    fourier_number,
    fractional_extraction_sphere,
    lattice_ceiling_ppm,
    lattice_removal_verdict,
    limiting_grain_radius,
    time_to_deplete_grain,
)

_SRC = Source(
    citation="Test fixture source, not a real reference",
    tier=Tier.T2, url="https://example.invalid/fixture", accessed=dt.date(2026, 9, 16),
    note="Synthetic fixture for unit tests. Carries no real measurement.",
)


def _v(q, tag=Tag.SOURCED, **kw):
    if tag in (Tag.SOURCED, Tag.MEASURED):
        return Value(quantity=q, tag=tag, source=_SRC, **kw)
    return Value(quantity=q, tag=tag, basis="test fixture", **kw)


@pytest.fixture
def quartz_measured() -> Feedstock:
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


# ---------------------------------------------------------------------------
# dimensional analysis
# ---------------------------------------------------------------------------

def test_diffusion_length_is_a_length() -> None:
    length = diffusion_length(Q_(1e-14, "m**2/s"), Q_(6.0, "hour"))
    require_dimensionality(length, "length", "L")


def test_arrhenius_diffusivity_dimensionality() -> None:
    d = MOST_MOBILE_BOUND.at(Q_(353.15, "K"))
    require_dimensionality(d, "diffusivity", "D")


def test_fourier_number_is_dimensionless() -> None:
    """Fo = D t / R^2 must reduce to a bare number: it is the only group that matters."""
    fo = fourier_number(Q_(1e-14, "m**2/s"), Q_(6.0, "hour"), Q_(100.0, "um"))
    assert isinstance(fo, float)


def test_critical_activation_energy_is_molar_energy() -> None:
    ea = critical_activation_energy(
        Q_(1e-4, "m**2/s"), Q_(100.0, "um"), Q_(1200.0, "degC"), Q_(6.0, "hour"))
    require_dimensionality(ea.to("J/mol"), "molar_energy", "Ea")


def test_gas_constant_makes_exponent_dimensionless() -> None:
    ratio = Q_(90.0, "kJ/mol") / (GAS_CONSTANT * Q_(1473.15, "K"))
    assert ratio.to("dimensionless").magnitude == pytest.approx(7.347868, rel=1e-6)


# ---------------------------------------------------------------------------
# golden: hand-traceable arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_diffusion_length_arithmetic() -> None:
    r"""Hand-check L = sqrt(D t) with the most generous bound at 80 degC, 6 h.

    D(353.15 K) = 1.0e-4 * exp(-90000 / (8.314462618 * 353.15))
      R T       = 8.314462618 * 353.15 = 2936.2525 J/mol
      exponent  = -90000 / 2936.2525 = -30.651315
      exp(...)  = 4.878687e-14
      D         = 1.0e-4 * 4.878687e-14 = 4.878687e-18 m2/s

    t = 6 h = 21600 s, so
      D t = 4.878687e-18 * 21600 = 1.053797e-13 m2
      L   = sqrt(1.053797e-13) = 3.246223e-7 m = 324.62 nm

    Against a 100 um = 1.0e-4 m grain radius, R/L = 308.05, i.e. 2.4886 orders
    of magnitude. The depleted skin is 0.32 um on a 100 um grain.
    """
    expo = -90000.0 / (8.314462618 * 353.15)
    assert expo == pytest.approx(-30.651315, rel=1e-7)
    d = 1.0e-4 * math.exp(expo)
    assert d == pytest.approx(4.878687e-18, rel=1e-6)
    l_m = math.sqrt(d * 21600.0)
    assert l_m == pytest.approx(3.246223e-7, rel=1e-6)
    model = diffusion_length(MOST_MOBILE_BOUND.at(Q_(80.0, "degC")), Q_(6.0, "hour"))
    assert float(model.to("m").magnitude) == pytest.approx(l_m, rel=1e-9)
    assert 1.0e-4 / l_m == pytest.approx(308.05, rel=1e-4)
    assert math.log10(1.0e-4 / l_m) == pytest.approx(2.4886, abs=1e-4)
    rt_product = 8.314462618 * 353.15
    assert rt_product == pytest.approx(2936.2525, rel=1e-7)
    l_nm = l_m * 1.0e9
    assert l_nm == pytest.approx(324.62, abs=0.01)
    grain_radius_um = 1.0e-4 * 1.0e6
    assert grain_radius_um == pytest.approx(100.0, rel=1e-9)
    skin_um = l_m * 1.0e6
    assert skin_um == pytest.approx(0.32, abs=0.01)


@pytest.mark.golden
def test_golden_critical_activation_energy_arithmetic() -> None:
    r"""Hand-check Ea_crit = -R T ln(R_grain^2/(D0 t)) at 1200 degC, 6 h, 100 um.

    Required diffusivity to span the grain: D_req = R^2/t
      = (1.0e-4)^2 / 21600 = 1.0e-8 / 21600 = 4.629630e-13 m2/s

    With D0 = 1.0e-4 m2/s:
      ln(D_req/D0) = ln(4.629630e-13 / 1.0e-4) = ln(4.629630e-9) = -19.19079
      (written -19.19082 before this revision, wrong in the seventh digit;
      math.log gives -19.190789, and it is asserted below)
      Ea_crit = -8.314462618 * 1473.15 * (-19.19082) J/mol
              = 12248.4506 * 19.19082 = 235057.4 J/mol = 235.0574 kJ/mol

    With D0 = 1.0e-10 m2/s:
      ln(4.629630e-13 / 1.0e-10) = ln(4.629630e-3) = -5.375278
      Ea_crit = 12248.4506 * 5.375278 = 65838.8 J/mol = 65.8388 kJ/mol

    The high bracket (235.06 kJ/mol) falls INSIDE the 90 to 400 kJ/mol range
    Liu et al. (2026) report; the low bracket (65.84 kJ/mol) falls BELOW that
    range. The two intervals therefore OVERLAP over 90 to 235.06 kJ/mol, and it
    is that overlap, not containment, which makes the Al roast case undecidable:
    a true Al barrier anywhere in the overlap permits depletion at some
    prefactor in the sweep and forbids it at others.
    """
    d_req = (1.0e-4) ** 2 / 21600.0
    assert d_req == pytest.approx(4.629630e-13, rel=1e-6)
    rt = 8.314462618 * 1473.15
    assert rt == pytest.approx(12248.4506, rel=1e-7)
    ea_high = -rt * math.log(d_req / 1.0e-4) / 1000.0
    ea_low = -rt * math.log(d_req / 1.0e-10) / 1000.0
    assert ea_high == pytest.approx(235.0574, rel=1e-6)
    assert ea_low == pytest.approx(65.8388, rel=1e-6)
    m_high = critical_activation_energy(
        Q_(1e-4, "m**2/s"), Q_(100.0, "um"), Q_(1200.0, "degC"), Q_(6.0, "hour"))
    m_low = critical_activation_energy(
        Q_(1e-10, "m**2/s"), Q_(100.0, "um"), Q_(1200.0, "degC"), Q_(6.0, "hour"))
    assert float(m_high.magnitude) == pytest.approx(ea_high, rel=1e-9)
    assert float(m_low.magnitude) == pytest.approx(ea_low, rel=1e-9)
    ea_band = np.atleast_1d(LIU_2026_EA_RANGE_KJ.quantity.to("kJ/mol").magnitude)
    # The high bracket is inside the reported barrier range.
    assert ea_band.min() <= ea_high <= ea_band.max()
    # The low bracket is strictly BELOW it, so the brackets are not contained.
    assert ea_low < ea_band.min()
    # What makes the case undecidable is that the two intervals overlap and
    # neither contains the other: the overlap is 90.0 to 235.0574 kJ/mol.
    overlap_lo = max(ea_low, float(ea_band.min()))
    overlap_hi = min(ea_high, float(ea_band.max()))
    assert overlap_lo == pytest.approx(90.0, abs=1e-9)
    assert overlap_hi == pytest.approx(235.0574, rel=1e-6)
    assert overlap_hi > overlap_lo
    sq_r = (1.0e-4) ** 2
    assert sq_r == pytest.approx(1.0e-8, rel=1e-9)
    ratio_high = d_req / 1.0e-4
    assert ratio_high == pytest.approx(4.629630e-9, rel=1e-6)
    ln_ratio_high = math.log(ratio_high)
    assert ln_ratio_high == pytest.approx(-19.19079, abs=5e-6)
    # The superseded digit, measured as wrong rather than merely annotated:
    # -19.19082 is not this logarithm to the seven digits it was written with.
    assert abs(ln_ratio_high - (-19.19082)) > 1e-5
    ratio_low = d_req / 1.0e-10
    assert ratio_low == pytest.approx(4.629630e-3, rel=1e-6)
    ln_ratio_low = math.log(ratio_low)
    assert ln_ratio_low == pytest.approx(-5.375278, rel=1e-6)
    ea_high_j = -rt * ln_ratio_high
    assert ea_high_j == pytest.approx(235057.4, rel=1e-6)
    assert ea_high_j / 1000.0 == pytest.approx(235.06, abs=5e-3)
    ea_low_j = -rt * ln_ratio_low
    assert ea_low_j == pytest.approx(65838.8, rel=1e-6)
    assert ea_low_j / 1000.0 == pytest.approx(65.84, abs=5e-3)
    assert float(ea_band.max()) == pytest.approx(400.0, rel=1e-9)


@pytest.mark.golden
def test_golden_sphere_extraction_short_time_arithmetic() -> None:
    r"""Hand-check the short-time sphere limit Mt/Minf = (6/sqrt(pi)) sqrt(Fo).

    At Fo = 1.0e-6:
      sqrt(Fo)   = 1.0e-3
      6/sqrt(pi) = 6 / 1.7724539 = 3.3851375
      Mt/Minf    = 3.3851375 * 1.0e-3 = 3.385137e-3

    Sanity on the geometry: for a thin depleted skin the extracted fraction is
    approximately 3 L / R, and Fo = 1e-6 means L/R = 1e-3, giving 3.0e-3, which
    agrees with 3.385e-3 to within the shape factor. The skin term never
    vanishes, which is why the infeasibility threshold in
    lattice_removal_verdict is a few percent rather than zero.
    """
    coeff = 6.0 / math.sqrt(math.pi)
    assert coeff == pytest.approx(3.3851375, rel=1e-6)
    val = coeff * math.sqrt(1.0e-6)
    assert val == pytest.approx(3.385137e-3, rel=1e-6)
    assert fractional_extraction_sphere(1.0e-6) == pytest.approx(val, rel=1e-12)
    assert val / 3.0e-3 == pytest.approx(1.128, rel=1e-3)
    sqrt_pi = math.sqrt(math.pi)
    assert sqrt_pi == pytest.approx(1.7724539, rel=1e-6)
    assert val == pytest.approx(3.385e-3, rel=1e-3)


@pytest.mark.golden
def test_golden_erf_half_defines_the_diffusion_length() -> None:
    r"""At x = sqrt(Dt) the erf solution retains erf(1/2) = 0.5205 of C0.

    C/C0 = erf(x / (2 sqrt(Dt))), so at x = sqrt(Dt) the argument is 1/2 and
    erf(0.5) = 0.52049988. This is why L = sqrt(Dt) is the skin thickness scale
    rather than an arbitrary convention: the depletion at that depth is about
    48 percent.
    """
    from scipy.special import erf
    assert float(erf(0.5)) == pytest.approx(ERF_HALF, rel=1e-12)
    d, t = Q_(1e-14, "m**2/s"), Q_(6.0, "hour")
    l_m = diffusion_length(d, t)
    retained = erfc_profile(d, t, l_m)
    assert float(retained[0]) == pytest.approx(ERF_HALF, rel=1e-9)
    assert float(erf(0.5)) == pytest.approx(0.5205, abs=1e-4)
    depletion_fraction = 1.0 - float(retained[0])
    depletion_pct = depletion_fraction * 100.0
    assert depletion_pct == pytest.approx(48.0, abs=1.0)


# ---------------------------------------------------------------------------
# benchmark: literature comparison with the error REPORTED
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_ti_barrier_against_liu_2026(capsys: pytest.CaptureFixture[str]) -> None:
    """Ti removal verdict against the Liu et al. 2026 activation energy.

    Liu et al. (2026, doi 10.3390/min16080836) report about 400 kJ/mol for Ti4+
    lattice diffusion in SiO2 and identify solid-state lattice diffusion as the
    likely rate-determining step for deep impurity removal. This test checks
    that the module's transport calculation is CONSISTENT with that finding:
    at 1200 degC and 6 h the Ti diffusion length must fall far short of a
    100 um grain, and it must do so across the whole prefactor sweep, since no
    Ti prefactor is available.

    The reported quantity is the deficit in orders of magnitude at both sweep
    ends. There is no numeric literature value to difference against, because
    the paper reports no diffusion length, so the error reported is the spread
    across the unmeasured prefactor, which is the honest uncertainty here.
    """
    ea = TI_LATTICE_BOUND.activation_energy.quantity.to("J/mol")
    assert float(ea.to("kJ/mol").magnitude) == pytest.approx(400.0, abs=1e-9)
    r_grain = 1.0e-4
    results = {}
    for d0 in (1e-10, 1e-4):
        d = d0 * math.exp(-float(ea.magnitude) / (8.314462618 * 1473.15))
        l_m = math.sqrt(d * 21600.0)
        results[d0] = math.log10(r_grain / l_m)
    print(f"\nBENCHMARK Ti lattice transport at 1200 degC, 6 h, 100 um grain "
          f"(Ea = 400 kJ/mol from doi 10.3390/min16080836): deficit "
          f"{results[1e-4]:.2f} orders of magnitude at D0 = 1e-4 m2/s and "
          f"{results[1e-10]:.2f} at D0 = 1e-10 m2/s. Prefactor spread "
          f"{results[1e-10] - results[1e-4]:.2f} orders, which is the full "
          f"uncertainty from the unmeasured prefactor. No literature diffusion "
          f"length exists to difference against; the paper reports Ea only.")
    assert results[1e-4] > 2.0
    assert results[1e-10] > results[1e-4]


@pytest.mark.benchmark
def test_benchmark_xia_2024_residual_is_lattice(
    quartz_measured: Feedstock, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lattice-ceiling calculation against the Xia et al. 2024 endpoint.

    Xia et al. (2024, doi 10.3390/min14070727) took a vein quartz from
    128.86 ug/g total trace impurities to 24.23 ug/g using a flowsheet that
    included both hot pressure acid leaching AND chlorination roasting, and
    identified the residual as lattice-bound Al, Ti and Li. The model claim
    under test is structural: the achievable floor equals the lattice
    inventory, so for a feedstock with a measured lattice split the predicted
    floor must equal the sum of lattice Al, Ti, Li and B and nothing else.

    For the synthetic fixture: Al 30*0.6 = 18.0, Ti 10*0.9 = 9.0,
    Li 5*0.8 = 4.0, B 1*0.5 = 0.5, sum 31.5 ppm. Compared against the Xia
    residual of 24.23 ug/g, the fixture floor is 30 percent higher, which is
    expected and not an error: the fixture is not their ore. The comparison
    reported is of MAGNITUDE, establishing that a lattice-split-derived floor
    lands in the same tens-of-ppm range as the measured residual of a real
    purification campaign rather than near zero.
    """
    ceiling = lattice_ceiling_ppm(quartz_measured)
    assert ceiling["Al"] == pytest.approx(18.0, abs=1e-12)
    assert ceiling["Ti"] == pytest.approx(9.0, abs=1e-12)
    assert ceiling["Li"] == pytest.approx(4.0, abs=1e-12)
    assert ceiling["B"] == pytest.approx(0.5, abs=1e-12)
    assert ceiling["_sum"] == pytest.approx(31.5, abs=1e-12)
    xia_residual = 24.23
    err = abs(ceiling["_sum"] - xia_residual) / xia_residual * 100.0
    print(f"\nBENCHMARK lattice floor magnitude: synthetic fixture floor "
          f"{ceiling['_sum']:.2f} ppm (Al {ceiling['Al']:.1f} + Ti {ceiling['Ti']:.1f} + "
          f"Li {ceiling['Li']:.1f} + B {ceiling['B']:.1f}) against the Xia et al. 2024 "
          f"measured residual of {xia_residual:.2f} ug/g after leaching AND "
          f"chlorination, difference {err:.1f} percent. NOT a validation of the "
          f"fixture's lattice split (which is synthetic), only a check that a "
          f"lattice-derived floor is of the same order as a real measured residual "
          f"(source: doi 10.3390/min14070727).")
    assert 0.3 < ceiling["_sum"] / xia_residual < 3.0
    # The lattice fractions the prose multiplies out, recovered from the profile
    # itself so the arithmetic is checked against the fixture and not retyped.
    imp = quartz_measured.impurities
    al_frac = ceiling["Al"] / imp.total_ppm("Al")
    assert al_frac == pytest.approx(0.6, abs=1e-12)
    ti_frac = ceiling["Ti"] / imp.total_ppm("Ti")
    assert ti_frac == pytest.approx(0.9, abs=1e-12)
    li_frac = ceiling["Li"] / imp.total_ppm("Li")
    assert li_frac == pytest.approx(0.8, abs=1e-12)
    # The stated 30 percent excess of the fixture floor over the Xia residual is
    # the quantity already computed above, so it is asserted rather than printed.
    assert err == pytest.approx(30.0, abs=0.5)
    # The Xia et al. 2024 feed total, closed by the removal it implies.
    xia_feed = 128.86
    xia_removal_pct = (xia_feed - xia_residual) / xia_feed * 100.0
    assert xia_removal_pct == pytest.approx(81.2, abs=0.1)


# ---------------------------------------------------------------------------
# the negative result, and the limits of the negative result
# ---------------------------------------------------------------------------

def test_atmospheric_leach_is_robustly_infeasible(quartz_measured: Feedstock) -> None:
    """80 degC, 6 h, 100 um: infeasible across the whole parameter range."""
    v = lattice_removal_verdict(
        quartz_measured, "Al", Q_(100.0, "um"), Q_(80.0, "degC"), Q_(6.0, "hour"))
    assert v["verdict"] is Verdict.ROBUST_INFEASIBLE
    assert v["orders_of_magnitude"] == pytest.approx(2.4885, abs=1e-3)
    assert v["diffusivity_is_assumed_bound"] is True


def test_al_roast_case_is_explicitly_undecided(quartz_measured: Feedstock) -> None:
    """1200 degC Al must return UNDECIDED, never a confident negative.

    This is the adversarial test on the module's own headline claim. The
    generous mobility bound gives a diffusion length far LARGER than the grain
    at roasting temperature, and the critical barrier band (65.8 to 235.1
    kJ/mol) OVERLAPS the 90 to 400 kJ/mol range Liu et al. report over 90 to
    235.1 kJ/mol, so a diffusion-length argument cannot exclude Al removal in a
    roast. If a later
    change makes this return ROBUST_INFEASIBLE, it is claiming more than the
    data supports and this test must fail.
    """
    v = lattice_removal_verdict(
        quartz_measured, "Al", Q_(100.0, "um"), Q_(1200.0, "degC"), Q_(6.0, "hour"))
    assert v["verdict"] is Verdict.UNDECIDED_BY_DIFFUSION
    assert "NOT DECIDED" in str(v["interpretation"])
    assert "Do not report this as infeasibility" in str(v["interpretation"])
    crit = v["critical_Ea_kJ_per_mol"]
    lo, hi = crit["d0_low_1e-10"], crit["d0_high_1e-4"]
    assert lo == pytest.approx(65.84, rel=1e-3)
    assert hi == pytest.approx(235.06, rel=1e-3)
    liu_band = np.atleast_1d(LIU_2026_EA_RANGE_KJ.quantity.to("kJ/mol").magnitude)
    liu_lo = float(liu_band.min())
    liu_hi = float(liu_band.max())
    assert liu_lo == pytest.approx(90.0, rel=1e-3)
    assert liu_hi == pytest.approx(400.0, rel=1e-3)
    overlap_hi = min(hi, liu_hi)
    assert overlap_hi == pytest.approx(235.1, rel=1e-3)


def test_ti_roast_case_is_robustly_infeasible(quartz_measured: Feedstock) -> None:
    """Ti at 1200 degC IS decidable, because its barrier is sourced at 400 kJ/mol."""
    v = lattice_removal_verdict(
        quartz_measured, "Ti", Q_(100.0, "um"), Q_(1200.0, "degC"), Q_(6.0, "hour"),
        pair=TI_LATTICE_BOUND)
    assert v["verdict"] is Verdict.ROBUST_INFEASIBLE
    assert v["diffusion_length_nm"] == pytest.approx(119.07, rel=1e-3)
    assert v["fraction_extracted"] < 0.01
    assert v["barrier_interval_kJ_per_mol"] == (400.0, 400.0)


def test_verdict_reports_ppm_removable(quartz_measured: Feedstock) -> None:
    """The process-relevant output is ppm moved, comparable against a spec."""
    v = lattice_removal_verdict(
        quartz_measured, "Al", Q_(100.0, "um"), Q_(80.0, "degC"), Q_(6.0, "hour"))
    # lattice Al is 18.0 ppm; extraction is about 1.1 percent of it
    assert isinstance(v["ppm_removable_from_lattice"], float)
    assert v["ppm_removable_from_lattice"] == pytest.approx(
        18.0 * float(v["fraction_extracted"]), rel=1e-9)
    assert v["ppm_removable_from_lattice"] < 0.5


def test_verdict_handles_unmeasured_lattice_split(
    quartz_no_lattice_split: Feedstock
) -> None:
    """Without a lattice split the ppm figure is unavailable, not zero."""
    v = lattice_removal_verdict(
        quartz_no_lattice_split, "Al", Q_(100.0, "um"), Q_(80.0, "degC"), Q_(6.0, "hour"))
    assert isinstance(v["ppm_removable_from_lattice"], str)
    assert "unavailable" in v["ppm_removable_from_lattice"]


def test_lattice_ceiling_raises_without_measurement(
    quartz_no_lattice_split: Feedstock
) -> None:
    """An unmeasured lattice split means the purity ceiling is UNKNOWN."""
    with pytest.raises(MissingValueError):
        lattice_ceiling_ppm(quartz_no_lattice_split)


# ---------------------------------------------------------------------------
# missing-data discipline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("element", ["Al", "Ti", "Li", "Na", "B"])
def test_no_builtin_diffusivity_for_any_element(element: str) -> None:
    """Every element's diffusivity must raise, naming the measurement needed.

    This is the discipline test: a silent default would make the whole negative
    result unverifiable, and an invented Arrhenius pair is exactly the failure
    mode the platform brief prohibits.
    """
    with pytest.raises(MissingValueError, match="would be supplied by|supplied by"):
        diffusivity(element, Q_(1200.0, "degC"))
    assert DIFFUSIVITY_STATUS[element]["would_be_supplied_by"]


def test_diffusivity_accepts_explicit_pair() -> None:
    """A caller may supply a pair, making the assumption visible at the call site."""
    d = diffusivity("Ti", Q_(1200.0, "degC"), pair=TI_LATTICE_BOUND)
    require_dimensionality(d, "diffusivity", "D")
    assert float(d.magnitude) > 0.0


def test_diffusivity_rejects_mismatched_pair() -> None:
    with pytest.raises(ValueError, match="is for Ti, not Al"):
        diffusivity("Al", Q_(1200.0, "degC"), pair=TI_LATTICE_BOUND)


def test_diffusivity_rejects_untracked_element() -> None:
    with pytest.raises(KeyError, match="not tracked"):
        diffusivity("Zr", Q_(1200.0, "degC"))


def test_bounds_are_tagged_assumed_with_a_basis() -> None:
    """Every bracketing value must be ASSUMED and must say what it rests on."""
    assert MOST_MOBILE_BOUND.d0.tag is Tag.ASSUMED
    assert MOST_MOBILE_BOUND.activation_energy.tag is Tag.ASSUMED
    assert "Estimate" in MOST_MOBILE_BOUND.d0.basis
    assert "OVERSTATE" in MOST_MOBILE_BOUND.d0.basis
    assert D0_SWEEP_M2_S.tag is Tag.ASSUMED
    # the Ti barrier IS sourced, unlike its prefactor
    assert TI_LATTICE_BOUND.activation_energy.tag is Tag.SOURCED
    assert TI_LATTICE_BOUND.activation_energy.source is not None
    assert TI_LATTICE_BOUND.activation_energy.source.doi == "10.3390/min16080836"
    assert TI_LATTICE_BOUND.d0.tag is Tag.ASSUMED


def test_liu_range_is_sourced_with_a_real_doi() -> None:
    assert LIU_2026_EA_RANGE_KJ.tag is Tag.SOURCED
    assert LIU_2026_EA_RANGE_KJ.source is not None
    assert LIU_2026_EA_RANGE_KJ.source.doi == "10.3390/min16080836"
    band = np.atleast_1d(LIU_2026_EA_RANGE_KJ.quantity.to("kJ/mol").magnitude)
    assert band.min() == pytest.approx(90.0)
    assert band.max() == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# numerical correctness of the series solution
# ---------------------------------------------------------------------------

def test_sphere_series_matches_converged_sum() -> None:
    """The adaptive term count must reproduce a brute-force 200k-term sum."""
    for fo in (1e-4, 1e-3, 1e-2, 0.1):
        n = np.arange(1, 200001, dtype=float)
        converged = 1.0 - 6.0 / math.pi**2 * float(
            np.sum(np.exp(-(n**2) * math.pi**2 * fo) / n**2))
        assert fractional_extraction_sphere(fo) == pytest.approx(converged, abs=1e-12)


def test_sphere_series_matches_finite_difference_solution() -> None:
    """Verify the series against an independent numerical solve of the same PDE.

    The series is derived in the module docstring rather than quoted from a
    textbook equation number, so it is checked here against a finite-difference
    solution of the spherical diffusion equation. Substituting u = r C reduces
    it to the 1-D heat equation with u(0) = u(R) = 0, which is integrated
    explicitly on 4000 nodes to Fo = 1e-3, and the remaining mass fraction is
    obtained by volume-weighted integration of the resulting profile.
    """
    n_nodes, radius, d = 4000, 1.0, 1.0
    r = np.linspace(0.0, radius, n_nodes + 1)
    dr = float(r[1] - r[0])
    u = r.copy()                      # uniform C = 1 means u = r
    target_fo = 1.0e-3
    n_steps = int(round(target_fo * radius**2 / d / (0.2 * dr**2 / d)))
    dt = target_fo * radius**2 / d / n_steps
    for _ in range(n_steps):
        lap = np.zeros_like(u)
        lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dr**2
        u = u + d * dt * lap
        u[0] = 0.0
        u[-1] = 0.0
    conc = np.zeros_like(u)
    conc[1:] = u[1:] / r[1:]
    conc[0] = conc[1]
    remaining = float(np.trapezoid(r**2 * conc, r) / np.trapezoid(r**2, r))
    numeric = 1.0 - remaining
    series = fractional_extraction_sphere(target_fo)
    rel = abs(numeric - series) / series
    assert rel < 1.0e-4, f"series and finite-difference disagree by {rel*100:.3f} percent"


def test_sphere_series_is_continuous_at_the_switch() -> None:
    """The switch to the short-time form must not introduce a visible jump."""
    below = fractional_extraction_sphere(FO_SHORT_TIME_SWITCH * 0.9999)
    above = fractional_extraction_sphere(FO_SHORT_TIME_SWITCH * 1.0001)
    rel = abs(above - below) / below
    assert rel < 0.01, f"discontinuity of {rel*100:.2f} percent at the switch"


def test_fixed_truncation_would_overstate_extraction() -> None:
    """Demonstrate the error the adaptive term count exists to avoid.

    A fixed term count that is too small for the Fourier number leaves a large
    last retained term and overstates extraction. Exercised here with the
    extreme case n_terms = 5 at Fo = 2.0e-4, where the adaptive rule would
    select max(50, ceil(sqrt(28/(pi^2 * 2.0e-4)))) = 120 terms, i.e. 24 times
    as many as are supplied here. The function
    asserts on the magnitude of the last retained term, so the call must raise
    rather than return a wrong number.
    """
    assert max(50, math.ceil(math.sqrt(28.0 / (math.pi**2 * 2.0e-4)))) == 120
    with pytest.raises(AssertionError, match="under-converged"):
        fractional_extraction_sphere(2.0e-4, n_terms=5)
    adaptive_terms = max(50, math.ceil(math.sqrt(28.0 / (math.pi**2 * 2.0e-4))))
    ratio = adaptive_terms / 5
    assert ratio == pytest.approx(24.0, rel=1e-9)


@pytest.mark.parametrize("fo,expected", [(0.0, 0.0), (10.0, 1.0)])
def test_sphere_extraction_limits(fo: float, expected: float) -> None:
    assert fractional_extraction_sphere(fo) == pytest.approx(expected, abs=1e-9)


def test_sphere_extraction_is_monotone_in_fourier_number() -> None:
    fos = np.logspace(-8, 1, 60)
    vals = [fractional_extraction_sphere(float(f)) for f in fos]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert all(0.0 <= v <= 1.0 for v in vals)


def test_time_to_deplete_grain_round_trips() -> None:
    """The bisection inverse must reproduce the target extraction."""
    d = Q_(1e-16, "m**2/s")
    t = time_to_deplete_grain(d, Q_(100.0, "um"), target_extraction=0.5)
    fo = fourier_number(d, t, Q_(100.0, "um"))
    assert fractional_extraction_sphere(fo) == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# physical sanity
# ---------------------------------------------------------------------------

def test_arrhenius_rejects_nonpositive_prefactor() -> None:
    with pytest.raises(ValidationError, match="must be positive|uphill"):
        ArrheniusDiffusivity(
            element="X", d0=_v(Q_(-1e-4, "m**2/s"), Tag.ASSUMED),
            activation_energy=_v(Q_(90.0, "kJ/mol"), Tag.ASSUMED))


def test_arrhenius_rejects_nonpositive_barrier() -> None:
    with pytest.raises(ValidationError, match="positive"):
        ArrheniusDiffusivity(
            element="X", d0=_v(Q_(1e-4, "m**2/s"), Tag.ASSUMED),
            activation_energy=_v(Q_(0.0, "kJ/mol"), Tag.ASSUMED))


def test_diffusivity_rises_with_temperature() -> None:
    """Positive barrier means faster when hotter: second-law direction check."""
    lo = MOST_MOBILE_BOUND.at(Q_(300.0, "K"))
    hi = MOST_MOBILE_BOUND.at(Q_(1000.0, "K"))
    assert float(hi.magnitude) > float(lo.magnitude)


def test_diffusion_length_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        diffusion_length(Q_(-1e-14, "m**2/s"), Q_(1.0, "hour"))
    with pytest.raises(ValueError, match="non-negative"):
        diffusion_length(Q_(1e-14, "m**2/s"), Q_(-1.0, "hour"))


def test_fourier_number_rejects_zero_radius() -> None:
    with pytest.raises(ValueError, match="positive"):
        fourier_number(Q_(1e-14, "m**2/s"), Q_(1.0, "hour"), Q_(0.0, "um"))


def test_erfc_profile_is_bounded_and_monotone() -> None:
    """Retained fraction rises from 0 at the surface to 1 in the interior."""
    d, t = Q_(1e-14, "m**2/s"), Q_(6.0, "hour")
    depths = Q_(np.linspace(0.0, 50.0, 40), "um")
    prof = erfc_profile(d, t, depths)
    assert np.all(prof >= 0.0) and np.all(prof <= 1.0)
    assert np.all(np.diff(prof) >= -1e-12)
    assert prof[0] == pytest.approx(0.0, abs=1e-12)
    # At 50 um, i.e. 3.40 diffusion lengths (L = 14.697 um here), the retained
    # fraction is erf(50/(2*14.697)) = erf(1.7010) = 0.98386, not 1: the erf
    # solution approaches the interior value asymptotically, and quoting it as
    # exactly 1 at finite depth would be wrong.
    l_um = float(diffusion_length(d, t).to("um").magnitude)
    assert l_um == pytest.approx(14.696938, rel=1e-6)
    assert float(prof[-1]) == pytest.approx(0.9838552, rel=1e-6)
    # Five diffusion lengths is where it is within 0.05 percent of the interior.
    deep = erfc_profile(d, t, Q_(5.0 * l_um, "um"))
    assert float(deep[0]) == pytest.approx(0.99959305, rel=1e-6)


def test_critical_activation_energy_raises_when_prefactor_too_small() -> None:
    """If D0 itself is below the required D, no barrier permits depletion."""
    with pytest.raises(ValueError, match="already exceeds the prefactor"):
        critical_activation_energy(
            Q_(1e-20, "m**2/s"), Q_(100.0, "um"), Q_(1200.0, "degC"), Q_(6.0, "hour"))


def test_limiting_grain_radius_equals_diffusion_length() -> None:
    d, t = Q_(1e-14, "m**2/s"), Q_(6.0, "hour")
    assert float(limiting_grain_radius(d, t).to("m").magnitude) == pytest.approx(
        float(diffusion_length(d, t).to("m").magnitude), rel=1e-12)


def test_diffusion_length_table_is_self_consistent() -> None:
    """Every row's L, Fo and extraction must agree with each other."""
    rows = diffusion_length_table(MOST_MOBILE_BOUND, grain_radius=Q_(100.0, "um"))
    assert len(rows) == 15
    for r in rows:
        assert r["L_m"] == pytest.approx(
            math.sqrt(r["D_m2_per_s"] * r["time_h"] * 3600.0), rel=1e-9)
        assert r["fourier_number"] == pytest.approx(
            r["D_m2_per_s"] * r["time_h"] * 3600.0 / r["grain_radius_m"] ** 2, rel=1e-9)
        assert 0.0 <= r["fraction_extracted"] <= 1.0
        assert r["L_over_R"] == pytest.approx(r["L_m"] / r["grain_radius_m"], rel=1e-9)


def test_out_of_calibration_warns() -> None:
    """Extrapolating a calibrated pair must warn, not fail silently."""
    pair = ArrheniusDiffusivity(
        element="X", d0=_v(Q_(1e-6, "m**2/s"), Tag.ASSUMED),
        activation_energy=_v(Q_(200.0, "kJ/mol"), Tag.ASSUMED),
        calibration_T=(1000.0, 1200.0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pair.at(Q_(1500.0, "K"))
    assert any("outside its calibration interval" in str(x.message) for x in w)
