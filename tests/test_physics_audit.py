"""Adversarial audit of ae.physics: dimensions, limits, signs, conservation, docstrings.

Every test here was written to FAIL against the code as committed, and each one
did fail before the accompanying fix. The defects found, in the order they
appear below:

1. diffusion.fractional_extraction_sphere used the ONE-term short-time
   asymptote 6 sqrt(Fo/pi) below Fo = 1e-4 while its docstring called that form
   "the exact short-time limit". It is the leading term only. The omitted -3 Fo
   term makes the branch overstate extraction and makes the function
   NON-MONOTONIC in Fo at the switch.
2. comminution accepted a non-finite f80, work index or law constant and
   returned nan or inf rather than raising, while the same guard on p80 was
   present. A nan specific energy propagates into an energy cost silently.
3. comminution.walker_specific_energy never checked the dimensionality of its
   own output, so a Bond-unit constant used at n = 1.5 returned a plausible
   magnitude carrying the wrong dimensions. Kick and Rittinger both check.
4. separation.rectangular_distribution_recovery evaluated (1 - exp(-x))/x
   directly, which loses all significant figures as x goes to zero, and raised
   on a NEGATIVE recovery for small but physical residence times.
5. reagents.reagent_balance defined the liquor mass by difference from the
   inlet and then checked that the outlet summed to the inlet, an identity that
   cannot fail. The assertion its docstring advertises could not detect an
   unbalanced flowsheet.
6. Two docstring claims contradicted by the module's own code: a grain coarser
   than sqrt(D t) "retains essentially all of its lattice inventory" (the code
   gives 0.948 extraction at twice that radius), and a thin-skin extraction
   "approximately 3 L / R" (the coefficient is 6/sqrt(pi)).

Tests whose names begin ``test_pin_`` are not defect demonstrations. They pin
behaviour judged correct but unasserted, so a later change that breaks it fails
here rather than silently.
"""

from __future__ import annotations

import datetime as dt
import math
import re

import numpy as np
import pytest

from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.site import (
    Currency,
    LabourRates,
    PermittingRegime,
    PowerSupply,
    ReagentPrices,
    Site,
)
from ae.core.units import Q_
from ae.physics.comminution import (
    TonConvention,
    bond_specific_energy,
    kick_specific_energy,
    rittinger_specific_energy,
    walker_specific_energy,
)
from ae.physics.diffusion import (
    FO_SHORT_TIME_SWITCH,
    fractional_extraction_sphere,
)
from ae.physics.liberation import exposure, leachable_fraction, liberation_size
from ae.physics.packing import (
    SizeClass,
    furnas_max_packing,
    krieger_dougherty_relative_viscosity,
)
from ae.physics.chlorination import CHLORIDES, vapour_pressure
from ae.physics.leaching import Regime, conversion, g_of_conversion
from ae.physics.phases import (
    alpha_beta_cumulative_volume_strain,
    excess_molar_volume,
)
from ae.physics.psd import Z_10, Z_90, LogNormalPSD, Weighting, _norm_ppf
from ae.physics.reagents import (
    LEACH_REACTIONS,
    Acid,
    check_reaction_balance,
    reagent_balance,
)
from ae.physics.thermal import (
    M_SIO2,
    landau_excess_enthalpy,
    landau_excess_heat_capacity,
)
from ae.physics.separation import (
    PropertyBasis,
    imperfection,
    rectangular_distribution_recovery,
)


_AUDIT_SRC = Source(
    citation="Audit test fixture source, not a real reference",
    tier=Tier.T2, url="https://example.invalid/audit-fixture",
    accessed=dt.date(2026, 9, 16),
    note="Synthetic fixture for the physics audit. Carries no real price or limit.",
)


def _converged_sphere_series(fo: float, n_terms: int = 200_000) -> float:
    """Independent evaluation of the sphere series, summed to n_terms.

    Deliberately not a call into the module under test: the point of these
    tests is to compare the module against an outside computation of the same
    analytic solution.
    """
    s = sum(math.exp(-(k**2) * math.pi**2 * fo) / k**2 for k in range(1, n_terms + 1))
    return 1.0 - 6.0 / math.pi**2 * s


# --- 1. diffusion: short-time asymptote -------------------------------------


def _one_term_branch_extraction(fo: float) -> float:
    """The defective branch as committed: leading short-time term below the switch.

    Reproduced here so the test can assert the size of the defect against the
    module's current output. Asserting the quoted figures against each other
    would be true by construction, which is the self-satisfying pattern
    tests/test_test_docstrings.py::test_no_assertion_compares_a_literal_against_itself
    exists to forbid.
    """
    if fo < FO_SHORT_TIME_SWITCH:
        return min(1.0, 6.0 / math.sqrt(math.pi) * math.sqrt(fo))
    return fractional_extraction_sphere(fo)


def test_sphere_extraction_is_monotonic_across_the_switch() -> None:
    """Extraction must rise with Fourier number everywhere, including the switch.

    Measured against the code as committed, over a 5000-point sweep of Fo from
    1e-8 to 10: extraction FELL from 0.033838382230 at Fo = 9.992325103785e-05
    to 0.033602473294 at Fo = 1.003075857943e-04, a drop of 2.359089e-04, and
    that was the only violation in the sweep. More material leaving a grain in
    less time is what the bisection in time_to_deplete_grain relies on not
    happening.
    """
    fos = np.concatenate([np.logspace(-8, -3, 3000), np.logspace(-3, 1, 2000)])
    vals = [fractional_extraction_sphere(float(f)) for f in fos]
    drops = [
        (float(fos[i]), float(fos[i + 1]), vals[i + 1] - vals[i])
        for i in range(len(vals) - 1)
        if vals[i + 1] < vals[i]
    ]
    assert not drops, (
        "extraction fell with rising Fourier number at "
        + ", ".join(f"Fo {a:.6e} -> {b:.6e} by {d:+.6e}" for a, b, d in drops)
    )
    # The quoted endpoints, recomputed from the defective branch and checked
    # against the fixed module rather than against each other.
    fo_lo, fo_hi = 9.992325103785e-05, 1.003075857943e-04
    assert fo_lo < FO_SHORT_TIME_SWITCH < fo_hi
    old_lo = _one_term_branch_extraction(fo_lo)
    old_hi = _one_term_branch_extraction(fo_hi)
    assert old_lo == pytest.approx(0.033838382230, rel=1e-9)
    assert old_hi == pytest.approx(0.033602473294, rel=1e-9)
    assert old_hi - old_lo == pytest.approx(-2.359089e-04, rel=1e-5)
    # The same pair on the fixed code rises, and the old low value was high.
    assert fractional_extraction_sphere(fo_hi) > fractional_extraction_sphere(fo_lo)
    assert old_lo > fractional_extraction_sphere(fo_lo)


@pytest.mark.parametrize("fo", [1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 5.0e-5, 9.9e-5])
def test_short_time_branch_matches_the_analytic_solution(fo: float) -> None:
    """Below the switch the branch must reproduce the series, not its leading term.

    The one-term form 6 sqrt(Fo/pi) overstates the analytic solution by
    8.870e-04 relative at Fo = 1e-6 and 8.942e-03 at Fo = 1e-4. Both errors are
    one-sided (always high), which is why they were invisible to a continuity
    check with a 1 percent tolerance.
    """
    exact = _converged_sphere_series(fo)
    got = fractional_extraction_sphere(fo)
    assert got == pytest.approx(exact, rel=1e-9)
    one_term = 6.0 * math.sqrt(fo / math.pi)
    assert one_term > exact, "the leading term is an upper bound on extraction"
    # The two quoted one-sided errors, each recomputed from the defective
    # branch against the independent series.
    for probe, quoted in ((1.0e-6, 8.870e-04), (1.0e-4, 8.942e-03)):
        series = _converged_sphere_series(probe)
        leading = 6.0 * math.sqrt(probe / math.pi)
        assert (leading - series) / series == pytest.approx(quoted, rel=2e-3)


def test_short_time_branch_is_continuous_to_machine_precision() -> None:
    """The two branches must agree at the switch far better than 1 percent.

    The committed code left a step of 8.862268267823e-03 relative (0.886
    percent, from 0.033851374996 below the switch to 0.033551375029 above it),
    which the existing test_sphere_series_is_continuous_at_the_switch permitted
    because its tolerance was 0.01. With the second term restored the step is
    below 1e-9 relative.
    """
    fo_lo = FO_SHORT_TIME_SWITCH * (1.0 - 1e-9)
    fo_hi = FO_SHORT_TIME_SWITCH * (1.0 + 1e-9)
    below = fractional_extraction_sphere(fo_lo)
    above = fractional_extraction_sphere(fo_hi)
    rel = abs(above - below) / below
    assert rel < 1.0e-9, f"step of {rel * 100:.4f} percent at the switch"
    # The defect's own step size, recomputed, and shown to sit under the 0.01
    # tolerance that let it through.
    old_below = _one_term_branch_extraction(fo_lo)
    old_above = _one_term_branch_extraction(fo_hi)
    assert old_below == pytest.approx(0.033851374996, rel=1e-9)
    assert old_above == pytest.approx(0.033551375029, rel=1e-9)
    old_rel = abs(old_above - old_below) / old_below
    assert old_rel == pytest.approx(8.862268267823e-03, rel=1e-6)
    assert old_rel < 0.01, "the old tolerance admitted the step"


# --- 2 and 3. comminution: non-finite inputs and Walker dimensions ----------


@pytest.mark.parametrize(
    "label,work_index,f80",
    [
        ("nan f80", 12.0, float("nan")),
        ("inf f80", 12.0, float("inf")),
        ("nan work index", float("nan"), 1000.0),
        ("inf work index", float("inf"), 1000.0),
    ],
)
def test_bond_refuses_non_finite_inputs(label: str, work_index: float, f80: float) -> None:
    """A non-finite work index or f80 must raise, as a non-finite p80 already did.

    An infinite f80 is the one arguable case: Bond's equation is defined in that
    limit and returns W = Wi exactly. It is refused anyway, because an infinite
    feed size in a flowsheet is a missing measurement rather than a physical
    statement, and 12.0 kWh/short ton returned for a missing input is worse than
    an exception.
    """
    with pytest.raises(ValueError, match="finite"):
        bond_specific_energy(
            Q_(work_index, "kWh/ton"), Q_(f80, "um"), Q_(100.0, "um"), TonConvention.SHORT
        )
    finite = bond_specific_energy(
        Q_(12.0, "kWh/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), TonConvention.SHORT
    )
    assert math.isfinite(float(finite.magnitude))
    assert float(finite.magnitude) == pytest.approx(8.205266807797946, rel=1e-12)


@pytest.mark.parametrize(
    "fn,constant_unit",
    [
        (kick_specific_energy, "kWh/ton"),
        (rittinger_specific_energy, "kWh*um/ton"),
    ],
)
def test_kick_and_rittinger_refuse_a_non_finite_constant(fn, constant_unit: str) -> None:
    """Both classical laws must refuse a nan law constant rather than return nan."""
    with pytest.raises(ValueError, match="finite"):
        fn(Q_(float("nan"), constant_unit), Q_(1000.0, "um"), Q_(100.0, "um"))


def test_walker_rejects_a_constant_whose_units_do_not_match_the_exponent() -> None:
    """Walker's constant carries um^(n-1), so a kWh/ton constant is wrong at n = 1.5.

    Committed behaviour: walker_specific_energy(Q_(60.0, "kWh/ton"), 1000 um,
    100 um, 1.5) returned 8.205266807797946 with units
    kilowatt_hour / micrometer ** 0.5 / ton, dimensionality
    [length] ** 1.5 / [time] ** 2, against specific energy's
    [length] ** 2 / [time] ** 2. The magnitude is the same number Bond returns
    for the same reduction, which is the worst case: dimensionally wrong and
    numerically plausible. Kick and Rittinger already raise on the same mistake.
    """
    with pytest.raises(Exception) as bad:
        walker_specific_energy(
            Q_(60.0, "kWh/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), 1.5
        )
    assert "dimension" in str(bad.value).lower() or "convert" in str(bad.value).lower()
    good = walker_specific_energy(
        Q_(60.0, "kWh*um**0.5/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), 1.5
    )
    assert float(good.to("kWh/ton").magnitude) == pytest.approx(
        8.205266807797946, rel=1e-12
    )
    assert str(good.dimensionality) == "[length] ** 2 / [time] ** 2"
    # The magnitude that made the wrong-unit call look plausible is the same
    # one Bond returns for this reduction, which is why only the dimensions
    # distinguish them.
    bond = bond_specific_energy(
        Q_(12.0, "kWh/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), TonConvention.SHORT
    )
    assert float(bond.magnitude) == pytest.approx(
        float(good.to("kWh/ton").magnitude), rel=1e-12
    )
    assert str(bond.dimensionality) == str(good.dimensionality)


# --- 4. separation: small-argument recovery ---------------------------------


def _defective_rectangular_recovery(kt: float, r_inf: float) -> float:
    """E2 as committed, evaluating (1 - exp(-x))/x directly.

    Kept so the measured error can be asserted against this module's fixed
    output rather than against the numbers written in the prose.
    """
    return r_inf * (1.0 - (1.0 - math.exp(-kt)) / kt)


@pytest.mark.parametrize("kt", [1.0e-12, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6])
def test_rectangular_recovery_survives_a_short_residence_time(kt: float) -> None:
    """E2 must reduce to R_inf (kt/2 - (kt)^2/6) as kt goes to zero.

    Committed behaviour at k_max = 1 1/s and R_inf = 0.90, measured against the
    analytic small-argument limit:

      kt      committed              analytic          relative error
      1e-12   +1.990954810935e-05    4.499999999998e-13   +4.424344e+07
      1e-10   -7.446633389918e-08    4.499999999850e-11   -1.655807e+03
      1e-09   +2.545373841700e-08    4.499999998500e-10   +5.556386e+01
      1e-08   +5.469723873830e-09    4.499999985000e-09   +2.154942e-01
      1e-07   +4.543775276034e-08    4.499999850000e-08   +9.727873e-03
      1e-06   +4.500141251640e-07    4.499998500000e-07   +3.172260e-05

    At kt = 1e-10 the committed form returned a NEGATIVE recovery, which then
    raised on the fraction check, so the failure mode is not merely imprecise.
    The cause is cancellation in (1 - exp(-x))/x, not the physics. An earlier
    draft of this docstring attributed the 4.5e-10 analytic value and the
    55.56 error to kt = 1e-8; both belong to kt = 1e-9, and the numbers above
    are now each asserted against a recomputation rather than against one
    another.
    """
    r_inf = 0.90
    got = rectangular_distribution_recovery(Q_(kt, "s"), Q_(1.0, "1/s"), r_inf)
    analytic = r_inf * (kt / 2.0 - kt**2 / 6.0)
    assert got >= 0.0
    assert got == pytest.approx(analytic, rel=1e-9)
    quoted = {
        1.0e-12: (1.990954810935e-05, 4.499999999998e-13, 4.424344e07),
        1.0e-10: (-7.446633389918e-08, 4.499999999850e-11, -1.655807e03),
        1.0e-09: (2.545373841700e-08, 4.499999998500e-10, 5.556386e01),
        1.0e-08: (5.469723873830e-09, 4.499999985000e-09, 2.154942e-01),
        1.0e-07: (4.543775276034e-08, 4.499999850000e-08, 9.727873e-03),
        1.0e-06: (4.500141251640e-07, 4.499998500000e-07, 3.172260e-05),
    }[kt]
    old = _defective_rectangular_recovery(kt, r_inf)
    assert old == pytest.approx(quoted[0], rel=1e-6)
    assert analytic == pytest.approx(quoted[1], rel=1e-9)
    assert (old - analytic) / analytic == pytest.approx(quoted[2], rel=1e-5)
    if kt == 1.0e-10:
        assert old < 0.0, "the committed form returned a negative recovery here"


def test_pin_rectangular_recovery_still_matches_its_doctest() -> None:
    """The stable rewrite must not move the published value 0.75037181 at 60 s."""
    got = rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)
    assert got == pytest.approx(0.75037181, rel=1e-8)


# --- 5. separation: imperfection near the medium density --------------------


def test_imperfection_refuses_a_density_cut_point_at_the_medium() -> None:
    """A cut point a hair above the medium density must not report a finite I.

    Committed behaviour: imperfection(0.05, 1.0000001, DENSITY) returned
    499999.99970806646, an imperfection of half a million, because the
    denominator x50 - 1 is 1e-7. The existing guard only rejects x50 <= 1.0,
    which bounds the denominator's SIGN and not its magnitude.

    The rejected value is read back out of the exception rather than recomputed
    in this test body, so the figure quoted above is the one the module itself
    produces at that cut point.
    """
    with pytest.raises(ValueError, match="exceeds 1") as bad:
        imperfection(0.05, 1.0000001, PropertyBasis.DENSITY)
    reported = float(re.search(r"imperfection ([\d.e+-]+) exceeds", str(bad.value))[1])
    assert reported == pytest.approx(499999.99970806646, rel=1e-9)
    # The guard must not have moved the published density or size worked values.
    assert imperfection(0.05, 1.60, PropertyBasis.DENSITY) == pytest.approx(
        0.08333333333333333, rel=1e-12
    )
    assert imperfection(20.0, 100.0, PropertyBasis.SIZE) == pytest.approx(0.20, rel=1e-12)


# --- 6. reagents: a mass balance that could not fail ------------------------


def _fixture_quartz():
    """The reference HPQ-limit fixture, lattice split measured.

    Duplicated from tests/test_reagents.py rather than imported because that
    module exposes it as a pytest fixture, not a callable. The impurity values
    are the Muller et al. HPQ reference limits already used there.
    """
    imp = ImpurityProfile(
        total={
            e: Value(quantity=Q_(v, "ppm_mass"), tag=Tag.SOURCED, source=_AUDIT_SRC)
            for e, v in [
                ("Al", 30.0), ("Ti", 10.0), ("Li", 5.0), ("Fe", 3.0),
                ("Na", 8.0), ("K", 8.0), ("B", 1.0),
            ]
        },
        lattice_fraction={
            e: Value(quantity=Q_(f, "dimensionless"), tag=Tag.SOURCED, source=_AUDIT_SRC)
            for e, f in [
                ("Al", 0.6), ("Ti", 0.9), ("Li", 0.8), ("Fe", 0.1),
                ("Na", 0.3), ("K", 0.3), ("B", 0.5),
            ]
        },
        method="LA_ICP_MS",
    )
    return Feedstock(
        sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
        deposit_name="Synthetic audit vein", country="IN", impurities=imp,
        characterized=True,
    )


def _fixture_site():
    """Synthetic site. Every price and limit here is a FIXTURE, not market data."""
    return Site(
        site_id="IN-TG-VKB", name="Synthetic audit site", country="IN",
        region="Telangana", currency=Currency.INR,
        power=PowerSupply(
            energy_price=Value(
                quantity=Q_(7.0, "INR/kWh"), tag=Tag.SOURCED, source=_AUDIT_SRC),
            rate_basis="published_tariff"),
        labour=LabourRates(
            fully_loaded_operator=Value(
                quantity=Q_(300.0, "INR/hour"), tag=Tag.SOURCED, source=_AUDIT_SRC)),
        reagents=ReagentPrices(
            prices={
                k: Value(quantity=Q_(v, "INR/kg"), tag=Tag.SOURCED, source=_AUDIT_SRC)
                for k, v in [("HF", 180.0), ("HCl", 12.0),
                             ("Ca(OH)2", 6.0), ("NaOH", 45.0)]},
            locally_available={"HF": False, "HCl": True,
                               "Ca(OH)2": True, "NaOH": True}),
        permitting=PermittingRegime(
            jurisdiction="Synthetic audit fixture jurisdiction",
            effluent_limits={
                "F": Value(quantity=Q_(2.0, "mg/L"), tag=Tag.SOURCED,
                           source=_AUDIT_SRC)}),
    )


def test_reagent_balance_closes_fluoride_and_calcium_element_by_element() -> None:
    """Fluoride and calcium must close as ELEMENTS, not only as total mass.

    The total-mass residual reagent_balance reports cannot fail. It defines
    m_liquor = m_in - m_product - m_sludge and then checks
    m_product + m_sludge + m_liquor against m_in, which reduces to m_in == m_in.
    Verified symbolically in this session: both the by-difference liquor and the
    sum of its actual constituents simplify to
    (acid + base - caf2 + imp + si + water), difference exactly 0. So the
    AssertionError its docstring advertised was unreachable and a flowsheet that
    creates or destroys an element would still report a residual of zero.

    These are the closures that can fail. On the HF plus lime route the
    tabulated chemistry is Ca(OH)2 + 2 HF -> CaF2 + 2 H2O, so:
      fluoride precipitated cannot exceed fluoride charged (2 mol F per mol CaF2)
      calcium precipitated cannot exceed calcium charged (1 mol Ca per mol CaF2)
    A stoichiometry error of the obvious kind, one CaF2 per mole of F rather
    than per two, breaks the first of these while leaving the total-mass
    residual at zero. That is the defect class the tautology could not see.
    """
    fs, st = _fixture_quartz(), _fixture_site()
    b = reagent_balance(
        fs, st, Q_(1000.0, "kg"), Acid.HF,
        silica_dissolved_fraction=0.001, excess_factor=1.2,
        water_mass=Q_(3000.0, "kg"),
    )
    neut = b["neutralization"]
    f_charged = b["acid_charged_mol"]           # 1 F per HF
    f_precipitated = 2.0 * neut["CaF2_moles_mol"]
    ca_charged = neut["base_moles_mol"]         # 1 Ca per Ca(OH)2
    ca_precipitated = neut["CaF2_moles_mol"]
    assert f_precipitated <= f_charged * (1.0 + 1e-12), (
        f"precipitated {f_precipitated} mol F from {f_charged} mol charged"
    )
    assert ca_precipitated <= ca_charged * (1.0 + 1e-12), (
        f"precipitated {ca_precipitated} mol Ca from {ca_charged} mol charged"
    )
    # Both closures are TIGHT on this route, which is the stronger statement and
    # not what I first assumed: HF carries one proton and lime carries two
    # equivalents, so the lime charged is n_HF/2 moles of Ca, exactly the CaF2
    # formed. Measured at 1000 kg ore, 0.1 percent silica dissolved and
    # excess_factor 1.2: 122.51325857527281 mol F charged and precipitated,
    # 61.25662928763641 mol Ca charged and precipitated. No lime survives the
    # neutralization and no fluoride stays in solution, which is why
    # fluoride_effluent has to be computed from a solubility argument elsewhere
    # rather than read off this balance.
    assert f_precipitated == pytest.approx(f_charged, rel=1e-12)
    assert ca_charged == pytest.approx(ca_precipitated, rel=1e-12)
    assert f_charged == pytest.approx(122.51325857527281, rel=1e-9)
    assert ca_charged == pytest.approx(61.25662928763641, rel=1e-9)
    assert f_charged == pytest.approx(2.0 * ca_charged, rel=1e-12)


def test_reagent_balance_reports_its_mass_residual_as_definitional() -> None:
    """The reported total-mass residual must be documented as an identity.

    Pinned so the figure is not later read as evidence of closure. It is zero
    to float rounding on every input because of how m_liquor is defined, which
    is why the element closures above exist. The docstring of reagent_balance
    now says this in those terms, and this test fails if that sentence is
    removed.
    """
    fs, st = _fixture_quartz(), _fixture_site()
    for excess, water in ((1.0, None), (1.2, Q_(3000.0, "kg")), (3.0, Q_(50.0, "kg"))):
        b = reagent_balance(
            fs, st, Q_(1000.0, "kg"), Acid.HF, silica_dissolved_fraction=0.001,
            excess_factor=excess, water_mass=water,
        )
        assert b["mass_balance_relative_residual"] == pytest.approx(0.0, abs=1e-15)
    doc = reagent_balance.__doc__.lower()
    assert "identity" in doc, "the residual must be documented as an identity"
    assert "element" in doc, "the docstring must point at the per-element closures"


# --- 7. packing: a maximum packing fraction of exactly 1 --------------------


def test_furnas_refuses_a_class_count_that_packs_to_zero_void() -> None:
    """phi_max must stay strictly below 1: a packing with no void is not a packing.

    Equation (1) is phi_max = 1 - (1 - phi1)^n with no bound on n, and the
    existing assertion is 0 < phi_max <= 1, which admits the endpoint. Measured
    at the default phi1 = 0.625: 1 - phi_max is 5.499e-05 at n = 10, 3.024e-09
    at n = 20, 1.663e-13 at n = 30 and exactly 0.0 at n = 40, where phi_max
    becomes 1.0 in double precision. Every subsequent class then fills a void
    that the model says is already gone.

    The consequence is downstream, in krieger_dougherty_relative_viscosity:
    with phi_max = 1.0 its divergence guard (phi >= phi_max) never fires at any
    physical solids loading, so it reports a finite relative viscosity of
    3.162278e+07 for a suspension at 99.9 volume percent solids, a paste with
    essentially no liquid, as though it flowed.

    Why n = 40 is not a real feed, stated so the guard is not mistaken for a
    physical claim: at McGeary's sevenfold separation 40 classes span 7^39 =
    9.095e+32 in diameter, and even at the 5.43-fold step of McGeary's own
    measured quaternary optimum they span 4.541e+28. A 1 nm to 1 m range, itself
    absurd for a mineral feed, admits about 12 classes at sevenfold steps. The
    guard exists because nothing in the signature stops a caller passing 40
    classes, not because 40 classes could be prepared.
    """
    classes_40 = tuple(
        SizeClass(diameter=Q_(7.0**i, "um")) for i in range(40)
    )
    with pytest.raises(ValueError, match="void"):
        furnas_max_packing(classes_40)
    # The published four-class result must be untouched.
    mcgeary = tuple(
        SizeClass(diameter=Q_(d, "um")) for d in (316.0, 38.0, 7.0, 1.0)
    )
    r = furnas_max_packing(mcgeary)
    assert r.phi_max == pytest.approx(0.98022, abs=5e-6)
    assert r.phi_max < 1.0
    # And the saturation this guard blocks is real: the formula itself reaches
    # exactly 1.0 at n = 40, which is what makes the endpoint reachable.
    assert 1.0 - (1.0 - 0.625) ** 40 == 1.0
    assert 1.0 - (1.0 - 0.625) ** 30 < 1.0


def test_pin_krieger_dougherty_still_diverges_at_a_reachable_phi_max() -> None:
    """With a physical phi_max the divergence guard must still fire.

    Pinned alongside the Furnas guard because the two interact: the viscosity
    model's only protection against reporting a flowable paste is
    phi >= phi_max, and that protection is worth nothing if phi_max can be 1.
    """
    with pytest.raises(ValueError, match="cannot flow"):
        krieger_dougherty_relative_viscosity(0.70, 0.70)
    assert krieger_dougherty_relative_viscosity(0.50, 0.70) == pytest.approx(
        8.9561, abs=5e-5
    )
    assert krieger_dougherty_relative_viscosity(0.65, 0.70) == pytest.approx(
        101.3267, abs=5e-5
    )


# --- 8. liberation: an inclusion larger than its host particle --------------


def test_pin_exposure_is_one_when_the_particle_is_no_larger_than_the_inclusion() -> None:
    """A particle no larger than an inclusion is a fragment of it, so E = 1.

    WITHDRAWN FINDING, pinned so it is not re-raised. This audit first read
    enclosed_fraction's early return (0.0 whenever d_inc >= d_p, hence E = 1.0)
    as a silent clamp hiding an out-of-domain call, on the grounds that
    exposure(10 um, 1e9 um) = 1.0 reports a 1.0e9 um inclusion, i.e. 1.0e3 m,
    as fully exposed inside a 10 um particle. Two guards were written and both
    were wrong. The first rejected d_inc >= d_p and broke FOUR existing tests in
    tests/test_liberation.py (test_exposure_bounds,
    test_exposure_is_monotonic_decreasing_in_particle_size,
    test_leachable_fraction_never_exceeds_the_non_lattice_inventory,
    test_all_lattice_means_nothing_is_leachable). Narrowing it to strict
    d_inc > d_p still broke THREE of those four, the monotonicity test being the
    one that recovered. Those tests were right and the finding was wrong:
    exposure is the fraction of inclusions intersecting a particle surface, and
    grinding finer than the inclusion population shatters every inclusion,
    which IS full exposure. The 1.0e3 m case is not a false positive, it is the
    same correct statement at an absurd scale.

    Retained as a pin because the behaviour is load-bearing for
    leachable_fraction's "even at infinite fineness the lattice bounds the
    leachable fraction" property, and an over-eager future guard would break it.
    """
    assert exposure(Q_(10.0, "um"), Q_(10.0, "um")) == 1.0
    assert exposure(Q_(5.0, "um"), Q_(10.0, "um")) == 1.0
    assert exposure(Q_(10.0, "um"), Q_(1.0e9, "um")) == 1.0
    # Continuous into that limit from inside the strict-inequality region.
    assert exposure(Q_(10.0, "um"), Q_(9.999, "um")) > 0.999
    assert exposure(Q_(10.0, "um"), Q_(9.999, "um")) < 1.0
    # The property it is load-bearing for: the lattice still bounds the
    # leachable fraction when the grind is far finer than both inclusion
    # populations.
    part = {"surface": 0.1, "fluid": 0.2, "mineral": 0.3, "lattice": 0.4}
    assert leachable_fraction(
        part, Q_(1.0, "um"), Q_(5.0, "um"), Q_(20.0, "um")
    ) == pytest.approx(1.0 - part["lattice"], rel=1e-12)


def test_liberation_size_refuses_a_target_exposure_it_cannot_bound() -> None:
    """As E* goes to zero the required particle size diverges, so it must raise.

    The existing guard rejects only E* <= 0 exactly. Measured on the committed
    code for a 20 um inclusion population: E* = 1e-9 returns 6.000000e+10 um
    (6.0e4 m) and E* = 1e-12 returns 6.000799e+13 um (6.0e7 m). Those are
    returned as grind targets, with no indication that the model has left its
    domain.

    The bound chosen is E* >= 1e-6, at which the required particle size is
    2999999.000380277 times the inclusion size (5.999998e+07 um for a 20 um
    population, i.e. 60 m). It is a numerical domain limit, not a process
    threshold, and no source is claimed for it: any target exposure this small
    is a calculation error rather than a grind specification.
    """
    with pytest.raises(ValueError, match="target_exposure"):
        liberation_size(Q_(20.0, "um"), 1.0e-9)
    with pytest.raises(ValueError, match="target_exposure"):
        liberation_size(Q_(20.0, "um"), 1.0e-12)
    # The published worked value and the E* = 1 asymptote are unmoved.
    assert liberation_size(Q_(20.0, "um"), 0.5).to("um").magnitude == pytest.approx(
        96.95, abs=5e-3
    )
    assert liberation_size(Q_(20.0, "um"), 1.0).to("um").magnitude == pytest.approx(
        20.0, rel=1e-12
    )


# --- pins on behaviour verified correct during this audit -------------------


def test_pin_landau_excess_heat_capacity_is_the_derivative_of_its_enthalpy() -> None:
    """Cp_ex must equal dH_ex/dT, checked by central difference, not by algebra.

    The two are separate implementations of the same thermodynamics, so their
    agreement is a real consistency check on the Landau treatment. Measured
    ratio 1.000000 at every temperature below T_c tested here.
    """
    for t in (300.0, 500.0, 700.0, 800.0, 840.0):
        h = 1e-4
        num = (
            landau_excess_enthalpy(Q_(t + h, "K")).to("J/mol").magnitude
            - landau_excess_enthalpy(Q_(t - h, "K")).to("J/mol").magnitude
        ) / (2 * h)
        cp = landau_excess_heat_capacity(Q_(t, "K")).to("J/(mol*K)").magnitude
        assert float(num) == pytest.approx(float(cp), rel=1e-6)


def test_pin_landau_inversion_energy_and_its_sign() -> None:
    """The alpha to beta inversion energy is +2646 J/mol, not -4104.

    H_ex(T_c) - H_ex(298.15 K) = 2646.0077329267015 J/mol = 44.03825513364891
    J/g, i.e. 44.0 kJ/kg, and it must be POSITIVE: heating through the inversion
    absorbs energy. Pinned with the sign-flipped value computed alongside it,
    because the thermal module docstring records that sign error as a defect
    that reached a previous build, and -4103.991664073273 J/mol is what the
    flipped convention gives.
    """
    h_tc = landau_excess_enthalpy(Q_(847.0, "K")).to("J/mol").magnitude
    h_298 = landau_excess_enthalpy(Q_(298.15, "K")).to("J/mol").magnitude
    delta = float(h_tc) - float(h_298)
    assert delta == pytest.approx(2646.0077329267015, rel=1e-9)
    assert delta > 0.0, "heating through the inversion must absorb energy"
    # In kJ/kg, against the 44.0 the module docstring quotes. Uses the module's
    # own M_SIO2 (0.0600843 kg/mol): a hand-typed 60.083 g/mol gives
    # 44.03920797774248 and does not reproduce the published figure.
    per_kg = delta / 1000.0 / M_SIO2
    assert per_kg == pytest.approx(44.03825513364891, rel=1e-9)


def test_pin_excess_molar_volume_falls_to_zero_at_tc_and_stays_there() -> None:
    """V_ex must decrease monotonically to exactly 0 at T_c and remain 0 above.

    Q = 0 above T_c by construction, so V_ex = V_D Q^2 must be identically zero
    there rather than small. Measured: 0.956316325430929 cm3/mol at 298.15 K,
    0.7603953858731278 at 500 K, 0.4949181750552307 at 700 K,
    0.04082016308499654 at 846 K, exactly 0.0 at 847 K and at 900 K.
    """
    vals = [
        float(excess_molar_volume(Q_(t, "K")).to("cm**3/mol").magnitude)
        for t in (298.15, 500.0, 700.0, 846.0, 847.0, 900.0)
    ]
    assert vals[0] == pytest.approx(0.956316325430929, rel=1e-9)
    assert vals[1] == pytest.approx(0.7603953858731278, rel=1e-9)
    assert vals[2] == pytest.approx(0.4949181750552307, rel=1e-9)
    assert vals[3] == pytest.approx(0.04082016308499654, rel=1e-9)
    assert vals[4] == 0.0
    assert vals[5] == 0.0
    for lo, hi in zip(vals, vals[1:], strict=False):
        assert hi <= lo, "excess volume cannot rise with temperature"
    # Antisymmetry of the cumulative strain between the same two endpoints.
    fwd = alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K"))
    rev = alpha_beta_cumulative_volume_strain(Q_(900.0, "K"), Q_(298.15, "K"))
    assert float(fwd) == pytest.approx(0.04214703946368132, rel=1e-9)
    assert float(rev) == pytest.approx(-0.04214703946368132, rel=1e-9)


def test_pin_product_layer_inversion_is_exact_not_iterative() -> None:
    """The trigonometric inversion of g(X) must return to machine precision.

    Product-layer control inverts a cubic, and this module solves it in closed
    form rather than by iteration. Residuals g(X(t)) - t/tau measured at
    tau = 100 s: +1.613e-16 at t = 1e-6 s, +4.528e-16 at 1 s, +2.220e-16 at
    50 s, +1.998e-15 at 99 s and -3.664e-15 at 99.9 s. The bound below is 1e-14,
    which is the observed scale near the endpoint and not a loose tolerance
    chosen to pass: the residual grows toward X = 1 because g''(X) diverges
    there, so a single tight bound across the whole domain would be the wrong
    claim.
    """
    tau = Q_(100.0, "s")
    for t_s in (1e-6, 1.0, 50.0, 99.0, 99.9):
        x = conversion(Regime.PRODUCT_LAYER, Q_(t_s, "s"), tau)
        g = g_of_conversion(Regime.PRODUCT_LAYER, float(x))
        residual = float(np.asarray(g).ravel()[0]) - t_s / 100.0
        assert abs(residual) < 1e-14, f"residual {residual:.3e} at t = {t_s} s"
        if t_s <= 50.0:
            assert abs(residual) < 1e-15, (
                f"away from the endpoint the inversion is tighter still: "
                f"residual {residual:.3e} at t = {t_s} s"
            )
    # At and beyond tau the conversion is complete and clamped, not extrapolated.
    assert float(conversion(Regime.PRODUCT_LAYER, Q_(100.0, "s"), tau)) == 1.0
    assert float(conversion(Regime.PRODUCT_LAYER, Q_(1000.0, "s"), tau)) == 1.0


def test_pin_psd_span_is_independent_of_the_reporting_basis() -> None:
    """(d90 - d10)/d50 must not depend on whether the basis is number or volume.

    A lognormal's span is a property of sigma_g alone (the Hatch-Choate shift
    multiplies every quantile by the same factor), so the three weightings must
    agree. Measured for sigma_g = 1.5: 1.0866522072246398 on number,
    1.08665220722464 on area, 1.0866522072246396 on volume. The module's own
    tabulated quantiles Z_90 = 1.2815515655446004 and Z_10 = -1.2815515655446004
    close the closed form.
    """
    psd = LogNormalPSD(
        d_gn=Q_(10.0, "um"), sigma_g=1.5, density=Q_(2650.0, "kg/m**3")
    )
    spans = [float(psd.span(w)) for w in
             (Weighting.NUMBER, Weighting.AREA, Weighting.VOLUME)]
    assert spans[0] == pytest.approx(1.0866522072246398, rel=1e-12)
    for sp in spans:
        assert sp == pytest.approx(spans[0], rel=1e-9)
    assert Z_90 == pytest.approx(1.2815515655446004, rel=1e-12)
    assert Z_10 == pytest.approx(-Z_90, rel=1e-12)
    assert _norm_ppf(0.90) == pytest.approx(Z_90, rel=1e-12)
    closed = math.exp(Z_90 * math.log(1.5)) - math.exp(Z_10 * math.log(1.5))
    assert spans[0] == pytest.approx(closed, rel=1e-9)


def test_pin_vapour_pressure_is_one_atmosphere_at_the_boiling_point() -> None:
    """TiCl4's Clausius-Clapeyron curve must pass through 101325 Pa at T_b.

    T_b = 136.45 degC is SOURCED (PubChem CID 24193, at 760 mm Hg), so the
    integrated curve passing exactly through one atmosphere there is a check on
    the integration constant rather than a coincidence. Above T_b the value is
    capped and the flag says so, which is pinned here so the cap is not later
    read as a physical plateau.
    """
    p_tb, capped_tb = vapour_pressure(CHLORIDES["TiCl4"], Q_(136.45, "degC"))
    assert float(p_tb.to("Pa").magnitude) == pytest.approx(101325.0, rel=1e-9)
    assert not capped_tb
    p_25, capped_25 = vapour_pressure(CHLORIDES["TiCl4"], Q_(25.0, "degC"))
    assert float(p_25.to("Pa").magnitude) == pytest.approx(1.905905e03, rel=1e-5)
    assert not capped_25
    p_500, capped_500 = vapour_pressure(CHLORIDES["TiCl4"], Q_(500.0, "degC"))
    assert capped_500, "beyond T_b the curve must be reported as capped"
    assert float(p_500.to("Pa").magnitude) == pytest.approx(101325.0, rel=1e-9)
    # Monotone in T below T_b.
    prev = -1.0
    for t_c in (-50.0, 0.0, 25.0, 60.0, 100.0, 136.0):
        p, _ = vapour_pressure(CHLORIDES["TiCl4"], Q_(t_c, "degC"))
        val = float(p.to("Pa").magnitude)
        assert val > prev
        prev = val


def test_pin_every_tabulated_leach_reaction_balances_in_mass_and_charge() -> None:
    """All five LEACH_REACTIONS must close per element and in charge.

    Already exercised in tests/test_reagents.py; pinned here with the measured
    residual bound because a reaction table is the kind of data that grows
    without its checker being re-run.
    """
    assert len(LEACH_REACTIONS) == 5
    for name in LEACH_REACTIONS:
        residuals = check_reaction_balance(LEACH_REACTIONS[name])
        for key, resid in residuals.items():
            assert abs(float(resid)) < 1e-12, f"{name} unbalanced in {key}: {resid}"
