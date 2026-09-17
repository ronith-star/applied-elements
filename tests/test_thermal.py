"""Tests for ae.physics.thermal: dimensionality, golden arithmetic, benchmarks, sanity.

Golden tests carry their hand arithmetic in the docstring so a reader can check every
number with a calculator. Benchmark tests REPORT the error against a literature datapoint
rather than asserting a loose tolerance and calling it validation.
"""

from __future__ import annotations

import math

import pytest

from ae.core.feedstock import Feedstock, OreType
from ae.core.provenance import Tag, Tier, Value
from ae.core.site import Currency, LabourRates, PowerSupply, Site
from ae.core.units import Q_, DimensionalityError
from ae.physics import thermal
from ae.physics.phases import Polymorph
from ae.physics.thermal import (
    CP_COEFFICIENTS,
    FORMATION_ENTHALPY,
    M_SIO2,
    T_REF,
    EnergyBalance,
    ThermalStep,
    calcination_energy,
    electricity_cost,
    fusion_energy,
    integrated_enthalpy,
    landau_excess_enthalpy,
    landau_excess_heat_capacity,
    molar_heat_capacity,
    phase_enthalpy,
    quench_heat_rejection,
    specific_heat_capacity,
    specific_transition_enthalpy,
)


@pytest.fixture
def ore() -> Feedstock:
    """An uncharacterized Vikarabad feedstock, which is the only honest default."""
    return Feedstock(
        sample_id="AE-Q-IN-VKB-001",
        ore_type=OreType.VEIN_QUARTZ,
        deposit_name="Vikarabad",
        country="IN",
    )


@pytest.fixture
def eta_unity() -> Value:
    return Value(
        quantity=Q_(1.0, "dimensionless"),
        tag=Tag.ASSUMED,
        basis="unit efficiency, used to isolate the thermodynamic demand in a test",
    )


@pytest.fixture
def site() -> Site:
    power = PowerSupply(
        energy_price=Value(
            quantity=Q_(7.0, "INR/kWh"),
            tag=Tag.ASSUMED,
            basis="round number for a test, not a sourced tariff",
        ),
        rate_basis="state_average",
    )
    labour = LabourRates(
        fully_loaded_operator=Value(
            quantity=Q_(300.0, "INR/hour"), tag=Tag.ASSUMED,
            basis="round number for a test, not a sourced wage"),
    )
    return Site(site_id="IN-TG-VKB", name="Vikarabad test site", country="IN",
                region="Telangana", currency=Currency.INR, power=power, labour=labour)


# --------------------------------------------------------------------------------------
# dimensional analysis
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "func,args",
    [
        (molar_heat_capacity, (Polymorph.QUARTZ, 800.0)),
        (specific_heat_capacity, (Polymorph.QUARTZ, 800.0)),
        (landau_excess_heat_capacity, (800.0,)),
        (landau_excess_enthalpy, (800.0,)),
    ],
)
def test_bare_float_temperature_rejected(func, args) -> None:
    """A bare float is not a temperature and must not be silently interpreted as kelvin."""
    with pytest.raises(TypeError):
        func(*args)


@pytest.mark.parametrize(
    "wrong", [Q_(800.0, "J"), Q_(800.0, "kg"), Q_(800.0, "Pa"), Q_(800.0, "m")]
)
def test_wrong_dimension_temperature_rejected(wrong) -> None:
    """Only a temperature may be passed where a temperature is required."""
    with pytest.raises(DimensionalityError):
        molar_heat_capacity(Polymorph.QUARTZ, wrong)


def test_cp_dimensionality() -> None:
    """Cp must come back as energy per mole per kelvin, and per kilogram per kelvin."""
    cp = molar_heat_capacity(Polymorph.QUARTZ, Q_(500.0, "K"))
    assert cp.dimensionality == Q_(1.0, "J/(mol*K)").dimensionality
    cs = specific_heat_capacity(Polymorph.QUARTZ, Q_(500.0, "K"))
    assert cs.dimensionality == Q_(1.0, "J/(kg*K)").dimensionality


def test_enthalpy_dimensionality() -> None:
    """Integrated Cp dT has dimensions of molar energy, and the specific form of J/kg."""
    dh = integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(900.0, "K"))
    assert dh.dimensionality == Q_(1.0, "J/mol").dimensionality
    ds = specific_transition_enthalpy(
        Polymorph.QUARTZ, Polymorph.CRISTOBALITE, Q_(1743.15, "K"),
        allow_extrapolation=True)
    assert ds.dimensionality == Q_(1.0, "J/kg").dimensionality


def test_cp_coefficient_units_are_consistent() -> None:
    """Each Cp coefficient must carry the unit that makes a + bT + c/T^2 + d/sqrt(T) work.

    Dimensional analysis of equation (1): every term must reduce to J/(mol K). With T in
    kelvin, b must be J/(mol K^2), c must be J K/mol, and d must be J/(mol K^0.5).
    """
    for phase, coeffs in CP_COEFFICIENTS.items():
        a, b, c, d = coeffs.raw
        t = Q_(700.0, "K")
        term_a = Q_(a, "J/(mol*K)")
        term_b = Q_(b, "J/(mol*K**2)") * t
        term_c = Q_(c, "J*K/mol") / (t ** 2)
        term_d = Q_(d, "J/(mol*K**0.5)") / (t ** 0.5)
        for term in (term_b, term_c, term_d):
            assert term.dimensionality == term_a.dimensionality, phase


def test_formation_enthalpy_dimensionality() -> None:
    for phase, value in FORMATION_ENTHALPY.items():
        assert value.quantity.dimensionality == Q_(1.0, "J/mol").dimensionality, phase


def test_electricity_cost_dimensionality(ore, eta_unity, site) -> None:
    """Cost must be currency per mass, in the site's own currency, with no conversion."""
    step = ThermalStep(name="calcine", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.QUARTZ, t_start=Q_(298.15, "K"),
                       t_end=Q_(1173.15, "K"), eta_thermal=eta_unity)
    cost = electricity_cost(calcination_energy(ore, step), site)
    assert "INR" in str(cost.units)
    assert cost.dimensionality == Q_(1.0, "INR/kg").dimensionality


# --------------------------------------------------------------------------------------
# provenance discipline
# --------------------------------------------------------------------------------------

def test_every_cp_coefficient_is_sourced() -> None:
    """No Cp coefficient may be a bare number: each carries a Source with a real DOI."""
    for phase, coeffs in CP_COEFFICIENTS.items():
        for name in ("a", "b", "c", "d"):
            v: Value = getattr(coeffs, name)
            assert v.tag is Tag.SOURCED, f"{phase} {name}"
            assert v.source is not None
            assert v.source.doi, f"{phase} {name} has no DOI"
            assert v.source.accessed is not None
            assert v.source.tier is Tier.T1


def test_glass_is_flagged_as_a_liquid_proxy() -> None:
    """Silica glass Cp is the liquid endmember, and that substitution must be visible."""
    assert CP_COEFFICIENTS[Polymorph.SILICA_GLASS].is_liquid_proxy_for_glass is True
    assert CP_COEFFICIENTS[Polymorph.SILICA_LIQUID].is_liquid_proxy_for_glass is False


def test_limitations_section_exists_and_names_the_blocked_source() -> None:
    """LIMITATIONS must state where the model is untrustworthy, including the Cp gap."""
    doc = thermal.__doc__ or ""
    assert "LIMITATIONS" in doc
    assert "webbook.nist.gov" in doc
    assert "10.1016/0016-7037(82)90383-0" in doc  # Richet et al. 1982, paywalled


# --------------------------------------------------------------------------------------
# golden tests: hand-traceable arithmetic
# --------------------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_cp_alpha_quartz_at_298() -> None:
    """Alpha-quartz Cp at 298.15 K from the ds62 polynomial, term by term.

    Cp = a + bT + c/T^2 + d/sqrt(T) with a = 92.9, b = -6.42e-4, c = -714900,
    d = -716.1 and T = 298.15 K:

      a            = 92.9
      bT           = -6.42e-4 x 298.15      = -0.19141230
      c/T^2        = -714900 / 88893.4225   = -8.04221482
      d/sqrt(T)    = -716.1 / 17.2670206    = -41.47212290
      -----------------------------------------------------
      Cp                                    = 43.19424998 J/(mol K)

    Per unit mass: 43.19424998 / 0.0600843 = 718.894 J/(kg K).

    sqrt(298.15) was written 17.2670496 before this revision. It is 17.2670206,
    wrong in the sixth digit, and it does not reproduce the quotient on the
    same line: -716.1/17.2670496 is -41.47205322, not -41.47212290. The
    corrected root gives -41.47212290 exactly as written, so the quoted term
    was right and only the intermediate was drifted. Both are asserted below.
    """
    cp = molar_heat_capacity(Polymorph.ALPHA_QUARTZ, Q_(298.15, "K"), include_landau=False)
    assert float(cp.magnitude) == pytest.approx(43.19424998, abs=1e-7)

    a, b, c, d = CP_COEFFICIENTS[Polymorph.QUARTZ].raw
    t = 298.15
    assert b * t == pytest.approx(-0.19141230, abs=1e-8)
    assert c / (t * t) == pytest.approx(-8.04221482, abs=1e-8)
    assert d / math.sqrt(t) == pytest.approx(-41.47212290, abs=1e-8)
    assert a + b * t + c / (t * t) + d / math.sqrt(t) == pytest.approx(43.19424998, abs=1e-7)

    cs = specific_heat_capacity(Polymorph.ALPHA_QUARTZ, Q_(298.15, "K"),
                                include_landau=False)
    assert float(cs.to("J/(kg*K)").magnitude) == pytest.approx(718.894, abs=1e-3)
    assert a == pytest.approx(92.9, abs=1e-6)
    assert b == pytest.approx(-6.42e-4, rel=1e-9)
    assert c == pytest.approx(-714900, rel=1e-9)
    assert d == pytest.approx(-716.1, abs=1e-6)
    assert t * t == pytest.approx(88893.4225, abs=1e-4)
    assert math.sqrt(t) == pytest.approx(17.2670206, abs=5e-8)
    # The superseded root, measured as wrong rather than annotated: it does not
    # reproduce the d/sqrt(T) term the same docstring line quotes.
    assert abs(math.sqrt(t) - 17.2670496) > 1e-6
    assert d / 17.2670496 == pytest.approx(-41.47205322, abs=5e-8)
    assert d / math.sqrt(t) == pytest.approx(-41.47212290, abs=5e-8)
    molar_mass = float(cp.magnitude) / float(cs.to("J/(kg*K)").magnitude)
    assert molar_mass == pytest.approx(0.0600843, rel=1e-6)


@pytest.mark.golden
def test_golden_landau_excess_heat_capacity_at_298() -> None:
    """Landau excess Cp of the quartz inversion at 298.15 K.

    Q_0 = ((Tc_0 - T_0)/Tc_0)^{1/4} = ((847 - 298.15)/847)^{0.25}
        = (548.85/847)^{0.25} = (0.64799291)^{0.25} = 0.89720682
    Q_0^2 = 0.80498007

    Cp_ex = T S_D / (2 Tc_0 Q^2)
          = 298.15 x 4.95 / (2 x 847 x 0.80498007)
          = 1475.8425 / 1363.63624
          = 1.08228460 J/(mol K)
    (the denominator was written 1363.63627 before this revision. The exact
    product 2 x 847 x Q_0^2 is 1363.636252, and the quotients are NOT equal at
    the eight digits quoted: 1475.8425/1363.63627 gives 1.08228457 against
    1.08228460 for the corrected denominator, which is the value the model
    returns and the test asserts. Both quotients are computed below.)

    So the ordering transition contributes 2.5 percent of the total Cp at room
    temperature, and the total is 43.19424998 + 1.08228460 = 44.27653458 J/(mol K).
    """
    q0 = ((847.0 - 298.15) / 847.0) ** 0.25
    assert q0 == pytest.approx(0.89720682, abs=1e-8)
    assert q0 * q0 == pytest.approx(0.80498007, abs=1e-8)

    cp_ex = landau_excess_heat_capacity(Q_(298.15, "K"))
    assert float(cp_ex.magnitude) == pytest.approx(1.08228460, abs=1e-8)
    assert 298.15 * 4.95 / (2.0 * 847.0 * q0 * q0) == pytest.approx(1.08228460, abs=1e-8)

    total = molar_heat_capacity(Polymorph.ALPHA_QUARTZ, Q_(298.15, "K"))
    assert float(total.magnitude) == pytest.approx(44.27653458, abs=1e-7)
    diff = 847.0 - 298.15
    assert diff == pytest.approx(548.85, abs=1e-6)
    ratio = diff / 847.0
    assert ratio == pytest.approx(0.64799291, abs=1e-8)
    numerator = 298.15 * 4.95
    assert numerator == pytest.approx(1475.8425, abs=1e-4)
    denominator = 2.0 * 847.0 * q0 * q0
    assert denominator == pytest.approx(1363.63624, abs=5e-5)
    # The superseded denominator, measured: it misses the asserted Cp_ex in the
    # eighth digit, which is why the digit was corrected rather than tolerated.
    assert numerator / denominator == pytest.approx(1.08228460, abs=5e-9)
    assert numerator / 1363.63627 == pytest.approx(1.08228457, abs=5e-9)
    baseline = float(total.magnitude) - float(cp_ex.magnitude)
    assert baseline == pytest.approx(43.19424998, abs=1e-7)
    percent_contribution = float(cp_ex.magnitude) / float(total.magnitude) * 100.0
    assert percent_contribution == pytest.approx(2.5, abs=0.1)


@pytest.mark.golden
def test_golden_landau_excess_diverges_at_tc_and_vanishes_above() -> None:
    """The excess Cp peaks just below Tc and is exactly zero above it.

    At 846 K, one kelvin below Tc = 847 K:
      Q = ((847 - 846)/847)^{0.25} = (0.00118064)^{0.25} = 0.18536560
      Q^2 = 0.03436041
      Cp_ex = 846 x 4.95 / (2 x 847 x 0.03436041) = 4187.7 / 58.20653 = 71.9455 J/(mol K)

    That is larger than the base Cp of 66.738 J/(mol K) at the same temperature, which is
    the physical content of the divergence: within a kelvin of the inversion most of the
    heat goes into disordering rather than into vibrational modes.
    """
    q = ((847.0 - 846.0) / 847.0) ** 0.25
    assert q == pytest.approx(0.18536560, abs=1e-8)
    cp_ex = float(landau_excess_heat_capacity(Q_(846.0, "K")).magnitude)
    assert cp_ex == pytest.approx(71.9455, abs=1e-4)
    base = float(molar_heat_capacity(Polymorph.QUARTZ, Q_(846.0, "K"),
                                     include_landau=False).magnitude)
    assert base == pytest.approx(66.7380, abs=1e-4)
    assert cp_ex > base
    assert float(landau_excess_heat_capacity(Q_(847.0, "K")).magnitude) == 0.0
    assert float(landau_excess_heat_capacity(Q_(1000.0, "K")).magnitude) == 0.0
    num = (847.0 - 846.0) / 847.0
    assert num == pytest.approx(0.00118064, abs=1e-8)
    q_sq = q ** 2
    assert q_sq == pytest.approx(0.03436041, abs=1e-8)
    denom = 2.0 * 847.0 * q_sq
    assert denom == pytest.approx(58.20653, abs=1e-5)
    numerator = cp_ex * denom
    assert numerator == pytest.approx(4187.7, abs=1e-1)
    landau_a = numerator / 846.0
    assert landau_a == pytest.approx(4.95, abs=5e-3)


@pytest.mark.golden
def test_golden_integrated_enthalpy_quartz_298_to_1000() -> None:
    """Sensible enthalpy of quartz from 298.15 to 1000 K, base polynomial only.

    Antiderivative F(T) = aT + bT^2/2 - c/T + 2 d sqrt(T), with a = 92.9,
    b/2 = -3.21e-4, -c = +714900, 2d = -1432.2:

      F(1000)   = 92900.0000 - 321.0000 + 714.9000 - 45290.1406 = 48003.7594
      F(298.15) = 27698.1350 -  28.5348 + 2397.7863 - 24729.8269 =  5337.5597
      ----------------------------------------------------------------------
      int Cp dT                                                  = 42666.1996 J/mol

    The Landau term adds H_ex(1000) - H_ex(298.15) = 2646.0077 J/mol, for a total of
    45312.2074 J/mol. Per tonne of quartz:
      45312.2074 / 0.0600843 = 754143.9 J/kg = 0.7541439 MJ/kg
      0.7541439 MJ/kg x 1000 kg/t / 3.6 MJ/kWh = 209.484 kWh/tonne
    """
    dh_base = integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(1000.0, "K"),
                                  include_landau=False)
    assert float(dh_base.magnitude) == pytest.approx(42666.1996, abs=1e-3)

    a, b, c, d = CP_COEFFICIENTS[Polymorph.QUARTZ].raw

    def f(t: float) -> float:
        return a * t + 0.5 * b * t * t - c / t + 2.0 * d * math.sqrt(t)

    assert f(1000.0) == pytest.approx(48003.7594, abs=1e-3)
    assert f(298.15) == pytest.approx(5337.5597, abs=1e-3)

    landau = (landau_excess_enthalpy(Q_(1000.0, "K"))
              - landau_excess_enthalpy(Q_(298.15, "K")))
    assert float(landau.magnitude) == pytest.approx(2646.0077, abs=1e-3)

    dh_total = integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(1000.0, "K"))
    assert float(dh_total.magnitude) == pytest.approx(45312.2074, abs=1e-3)
    kwh_per_t = float(dh_total.magnitude) / M_SIO2 / 3.6e6 * 1000.0
    assert kwh_per_t == pytest.approx(209.484, abs=1e-3)
    # Each term of the antiderivative at both limits, taken from the sourced Cp
    # coefficients rather than retyped, so the hand-check verifies the table.
    assert a == pytest.approx(92.9, abs=1e-9)
    assert 0.5 * b == pytest.approx(-3.21e-4, abs=1e-9)
    assert -c == pytest.approx(714900.0, abs=1e-6)
    assert 2.0 * d == pytest.approx(-1432.2, abs=1e-9)
    assert a * 1000.0 == pytest.approx(92900.0000, abs=1e-4)
    assert 0.5 * b * 1000.0 ** 2 == pytest.approx(-321.0000, abs=1e-4)
    assert -c / 1000.0 == pytest.approx(714.9000, abs=1e-4)
    assert 2.0 * d * math.sqrt(1000.0) == pytest.approx(-45290.1406, abs=1e-3)
    assert a * 298.15 == pytest.approx(27698.1350, abs=1e-3)
    assert 0.5 * b * 298.15 ** 2 == pytest.approx(-28.5348, abs=1e-4)
    assert -c / 298.15 == pytest.approx(2397.7863, abs=1e-4)
    assert 2.0 * d * math.sqrt(298.15) == pytest.approx(-24729.8269, abs=1e-3)
    # The per-mass conversion the docstring closes on, through the molar mass the
    # module actually uses.
    assert M_SIO2 == pytest.approx(0.0600843, abs=1e-9)
    j_per_kg = float(dh_total.magnitude) / M_SIO2
    assert j_per_kg == pytest.approx(754143.9, abs=1.0)
    assert j_per_kg / 1.0e6 == pytest.approx(0.7541439, abs=1e-6)


@pytest.mark.golden
def test_golden_inversion_enthalpy() -> None:
    """Whole energy cost of the alpha to beta inversion, 298.15 K to Tc.

    H_ex(847) - H_ex(298.15) = 2646.0077 J/mol. Per unit mass:
      2646.0077 / 0.0600843 = 44038.2 J/kg = 44.04 kJ/kg = 12.23 kWh/tonne

    That is small against the 209 kWh/tonne needed to reach 1000 K, which is the
    quantitative reason calcination is a sensible-heat problem and not a latent-heat one.
    """
    dh = float((landau_excess_enthalpy(Q_(847.0, "K"))
                - landau_excess_enthalpy(Q_(298.15, "K"))).magnitude)
    assert dh == pytest.approx(2646.0077, abs=1e-3)
    per_kg = dh / M_SIO2
    assert per_kg == pytest.approx(44038.2, abs=1.0)
    assert per_kg / 3.6e6 * 1000.0 == pytest.approx(12.23, abs=0.01)
    assert M_SIO2 == pytest.approx(0.0600843, abs=1e-6)
    kj_per_kg = per_kg / 1000.0
    assert kj_per_kg == pytest.approx(44.04, abs=0.01)
    dh_sensible = float(integrated_enthalpy(
        Polymorph.QUARTZ, Q_(298.15, "K"), Q_(1000.0, "K")).magnitude)
    per_kg_sensible = dh_sensible / M_SIO2
    kwh_per_tonne_sensible = per_kg_sensible / 3.6e6 * 1000.0
    assert kwh_per_tonne_sensible == pytest.approx(209.0, rel=0.05)


@pytest.mark.golden
def test_golden_fusion_path_enthalpy(ore, eta_unity) -> None:
    """Total enthalpy to take alpha-quartz at 298.15 K to silica liquid at 1996 K.

    Built from absolute phase enthalpies, both referenced to 298.15 K:
      H(qL, 1996 K)  = -921080 + 82.5 x (1996 - 298.15)
                     = -921080 + 82.5 x 1697.85 = -921080 + 140072.6 = -781007.4 J/mol
      H(q, 298.15 K) = -910720 + 0 + 0 = -910720 J/mol
      -------------------------------------------------------------------------
      dH             = -781007.4 - (-910720) = 129712.6 J/mol = 129.71 kJ/mol

    Per unit mass and in industry units:
      129712.6 / 0.0600843 = 2158843 J/kg = 2158.8 kJ/kg = 2.159 MJ/kg
      (2.159 MJ/kg x 1000 kg/t) / 3.6 MJ/kWh = 599.7 kWh/tonne

    That 599.7 kWh/tonne is the thermodynamic floor for melting silica from cold. A real
    furnace needs several times it; see the LBNL benchmark test.
    """
    h_liq = phase_enthalpy(Polymorph.SILICA_LIQUID, Q_(1996.0, "K"),
                           allow_extrapolation=True)
    assert float(h_liq.magnitude) == pytest.approx(-781007.4, abs=1.0)
    h_q = phase_enthalpy(Polymorph.QUARTZ, Q_(T_REF, "K"))
    assert float(h_q.magnitude) == pytest.approx(-910720.0, abs=1e-6)
    assert 82.5 * (1996.0 - 298.15) == pytest.approx(140072.6, abs=0.1)

    step = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
                       t_end=Q_(1996.0, "K"), transition_temperature=Q_(1996.0, "K"),
                       eta_thermal=eta_unity)
    bal = fusion_energy(ore, step)
    assert bal.theoretical_kwh_per_tonne == pytest.approx(599.7, abs=0.1)
    assert float(bal.theoretical.to("MJ/kg").magnitude) == pytest.approx(2.159, abs=1e-3)
    # The temperature span and the absolute-enthalpy arithmetic the docstring
    # tabulates, derived from the phase enthalpies the model returns.
    span_k = 1996.0 - 298.15
    assert span_k == pytest.approx(1697.85, abs=1e-9)
    assert float(h_q.magnitude) == pytest.approx(-921080.0 + 10360.0, abs=1.0)
    dh_j = float(h_liq.magnitude) - float(h_q.magnitude)
    assert dh_j == pytest.approx(129712.6, abs=1.0)
    assert dh_j / 1000.0 == pytest.approx(129.71, abs=0.01)
    # Per unit mass, then into industry units, through the module's molar mass.
    assert M_SIO2 == pytest.approx(0.0600843, abs=1e-9)
    j_per_kg = dh_j / M_SIO2
    assert j_per_kg == pytest.approx(2158843.0, abs=20.0)
    # 2158.8435 kJ/kg, which is the quoted 2158.8 to the five significant
    # figures the prose carries, so the tolerance follows that rounding.
    assert j_per_kg / 1000.0 == pytest.approx(2158.8, abs=0.05)
    # kWh/tonne from MJ/kg: 1000 kg per tonne over 3.6 MJ per kWh.
    kwh_per_t = j_per_kg / 1.0e6 * 1000.0 / 3.6
    assert kwh_per_t == pytest.approx(bal.theoretical_kwh_per_tonne, rel=1e-6)


@pytest.mark.golden
def test_golden_quench_heat_rejection(ore) -> None:
    """Heat rejected quenching quartz from 900 degC to 25 degC, and the mean rate.

    int Cp dT from 298.15 to 1173.15 K (including the Landau term) divided by the molar
    mass. From the antiderivative of equation (2):
      F(1173.15) = 108985.635 - 441.786 + 609.385 - 49054.679 = 60098.554
      F(298.15)  =                                                5337.560
      base       =                                               54760.995 J/mol
      Landau     =                                                2646.008 J/mol
                   (the inversion is fully crossed below 1173.15 K)
      total      =                                               57407.003 J/mol
                 = 57407.003 / 0.0600843 = 955441 J/kg = 955.4 kJ/kg

    Mean cooling rate over 60 s:
      (1173.15 - 298.15) / 60 = 875 / 60 = 14.5833 K/s
    """
    q, rate = quench_heat_rejection(ore, Polymorph.QUARTZ, Q_(1173.15, "K"),
                                    Q_(298.15, "K"), Q_(60.0, "s"))
    assert float(q.to("kJ/kg").magnitude) == pytest.approx(955.4, abs=0.1)
    assert float(rate.to("K/s").magnitude) == pytest.approx(14.5833, abs=1e-4)
    assert (1173.15 - 298.15) / 60.0 == pytest.approx(14.5833, abs=1e-4)
    # Every term of the antiderivative at the hot limit, plus the Landau term and
    # the sum, recomputed from the sourced coefficients rather than restated.
    a, b, c, d = CP_COEFFICIENTS[Polymorph.QUARTZ].raw
    assert a * 1173.15 == pytest.approx(108985.635, abs=1e-3)
    assert 0.5 * b * 1173.15 ** 2 == pytest.approx(-441.786, abs=1e-3)
    assert -c / 1173.15 == pytest.approx(609.385, abs=1e-3)
    assert 2.0 * d * math.sqrt(1173.15) == pytest.approx(-49054.679, abs=1e-3)
    f_hot = (a * 1173.15 + 0.5 * b * 1173.15 ** 2 - c / 1173.15
             + 2.0 * d * math.sqrt(1173.15))
    assert f_hot == pytest.approx(60098.554, abs=1e-3)
    f_cold = (a * 298.15 + 0.5 * b * 298.15 ** 2 - c / 298.15
              + 2.0 * d * math.sqrt(298.15))
    assert f_cold == pytest.approx(5337.560, abs=1e-3)
    base_j = f_hot - f_cold
    assert base_j == pytest.approx(54760.995, abs=1e-3)
    landau_j = float((landau_excess_enthalpy(Q_(1173.15, "K"))
                      - landau_excess_enthalpy(Q_(298.15, "K"))).magnitude)
    assert landau_j == pytest.approx(2646.008, abs=1e-3)
    total_j = base_j + landau_j
    assert total_j == pytest.approx(57407.003, abs=1e-2)
    # Per unit mass, through the module's molar mass.
    assert M_SIO2 == pytest.approx(0.0600843, abs=1e-9)
    assert total_j / M_SIO2 == pytest.approx(955441.0, abs=200.0)
    # The quench span in degC and K, and the mean rate over 60 s.
    span_k = 1173.15 - 298.15
    assert span_k == pytest.approx(875.0, abs=1e-9)
    assert 1173.15 - 273.15 == pytest.approx(900.0, abs=1e-9)
    assert 298.15 - 273.15 == pytest.approx(25.0, abs=1e-9)


@pytest.mark.golden
def test_golden_efficiency_scaling(ore) -> None:
    """Supplied energy is the theoretical demand divided by the overall efficiency.

    With eta_thermal = 0.40 (the top of the LBNL 33 to 40 percent band for a continuous
    glass furnace) and no extra loss deduction, a theoretical 599.7 kWh/tonne becomes
      599.7 / 0.40 = 1499.3 kWh/tonne
    and the losses are the difference, 1499.3 - 599.7 = 899.6 kWh/tonne.
    """
    eta = Value(quantity=Q_(0.40, "dimensionless"), tag=Tag.SOURCED,
                source=thermal.SRC_LBNL_GLASS,
                basis="upper end of the stated 33 to 40 percent of continuous-furnace "
                      "energy that goes toward melting the glass")
    step = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
                       t_end=Q_(1996.0, "K"), transition_temperature=Q_(1996.0, "K"),
                       eta_thermal=eta)
    bal = fusion_energy(ore, step)
    assert bal.overall_efficiency == pytest.approx(0.40)
    assert bal.theoretical_kwh_per_tonne == pytest.approx(599.7, abs=0.1)
    assert bal.kwh_per_tonne == pytest.approx(1499.3, abs=0.3)
    assert float(bal.losses.to("kWh/tonne").magnitude) == pytest.approx(899.6, abs=0.3)


# --------------------------------------------------------------------------------------
# benchmark tests: literature comparison with the error reported
# --------------------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_theoretical_melt_energy_against_lbnl(ore, eta_unity) -> None:
    """Model fusion floor against the DOE theoretical melting energy for glass.

    Literature: Galitsky and Worrell 2008 (LBNL, doi 10.2172/927883) state that
    "theoretically, 2.2 MMBtu are required to melt one short ton of glass". Converting:
      2.2 MMBtu = 2.2 x 1.05505585262e9 J = 2.3211229e9 J
      1 short ton = 907.18474 kg
      2.3211229e9 / 907.18474 = 2558600 J/kg = 2.5586 MJ/kg
      (2.5586 MJ/kg x 1000 kg/t) / 3.6 MJ/kWh = 710.7 kWh/tonne

    Model: 599.7 kWh/tonne for pure SiO2 from 298.15 K to liquid at 1996 K.

    Error: (599.7 - 710.7) / 710.7 = -15.6 percent.

    This is the RIGHT SIGN and the right size. The DOE figure is for soda-lime glass batch,
    which includes the endothermic decomposition of soda ash and limestone (releasing CO2)
    that pure silica does not have, so the pure-SiO2 floor must sit below it. The 15.6
    percent gap is therefore a consistency check, not a defect, and it is the reason this
    test reports the error instead of asserting a tight tolerance.
    """
    mmbtu_j = 1.05505585262e9
    short_ton_kg = 907.18474
    lit_kwh_per_t = 2.2 * mmbtu_j / short_ton_kg / 3.6e6 * 1000.0
    assert lit_kwh_per_t == pytest.approx(710.7, abs=0.1)

    step = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
                       t_end=Q_(1996.0, "K"), transition_temperature=Q_(1996.0, "K"),
                       eta_thermal=eta_unity)
    model_kwh_per_t = fusion_energy(ore, step).theoretical_kwh_per_tonne
    error_pct = 100.0 * (model_kwh_per_t - lit_kwh_per_t) / lit_kwh_per_t

    print(f"\n[benchmark] theoretical melt energy"
          f"\n  literature (LBNL 2008, 2.2 MMBtu/short ton, soda-lime batch): "
          f"{lit_kwh_per_t:.1f} kWh/tonne"
          f"\n  model (pure SiO2, 298.15 K -> liquid at 1996 K):              "
          f"{model_kwh_per_t:.1f} kWh/tonne"
          f"\n  error: {error_pct:+.2f} percent (expected negative: no carbonate "
          f"decomposition in pure silica)")

    assert -20.0 < error_pct < 0.0, (
        f"model floor {model_kwh_per_t:.1f} kWh/tonne should sit modestly below the "
        f"soda-lime batch figure {lit_kwh_per_t:.1f}; got {error_pct:+.2f} percent"
    )
    step_j = 2.2 * mmbtu_j
    assert step_j == pytest.approx(2.3211229e9, rel=1e-6)
    j_per_kg = step_j / short_ton_kg
    assert j_per_kg == pytest.approx(2558600, rel=1e-5)
    mj_per_kg = j_per_kg / 1e6
    assert mj_per_kg == pytest.approx(2.5586, abs=1e-4)
    assert model_kwh_per_t == pytest.approx(599.7, abs=0.1)
    assert error_pct == pytest.approx(-15.6, abs=0.1)


@pytest.mark.benchmark
def test_benchmark_electric_melter_efficiency_against_lbnl(ore) -> None:
    """Implied furnace efficiency of a specialty-glass electric melter.

    Literature: Galitsky and Worrell 2008 Table 7 gives a specialty-glass electric melter
    at 10.3 MMBtu per short ton of electricity (range 8.9 to 11.6). Converting as above:
      10.3 MMBtu/short ton = 3327.5 kWh/tonne   (range 2875.2 to 3747.4)
    They also report state-of-the-art electric melters at 780 to 800 kWh per short ton for
    soda-lime and sodium borate glass, which is 859.8 to 881.8 kWh per tonne.

    Model: dividing the 599.7 kWh/tonne thermodynamic floor by the reported melter
    consumption gives the implied efficiency:
      100 x 599.7 / 3327.5 = 18.0 percent   (specialty glass electric melter)
      100 x 599.7 / 859.8  = 69.7 percent   (state-of-the-art electric melter,
                                             780 kWh/short ton)

    Both bracket the LBNL statement that "only about 33-40% of the energy consumed by a
    continuous furnace goes toward melting the glass": specialty melters are worse than
    that band, state-of-the-art electric melters better, which is exactly what those two
    categories mean. The model's floor is therefore consistent with the published range,
    and this test reports the implied efficiencies rather than asserting one.
    """
    mmbtu_j = 1.05505585262e9
    short_ton_kg = 907.18474
    specialty = 10.3 * mmbtu_j / short_ton_kg / 3.6e6 * 1000.0
    soa_780 = 780.0 / (short_ton_kg / 1000.0)
    assert specialty == pytest.approx(3327.5, abs=0.1)
    assert soa_780 == pytest.approx(859.8, abs=0.1)

    eta_unity_v = Value(quantity=Q_(1.0, "dimensionless"), tag=Tag.ASSUMED,
                        basis="unit efficiency to isolate the thermodynamic floor")
    step = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
                       t_end=Q_(1996.0, "K"), transition_temperature=Q_(1996.0, "K"),
                       eta_thermal=eta_unity_v)
    floor = fusion_energy(ore, step).theoretical_kwh_per_tonne

    eta_specialty = 100.0 * floor / specialty
    eta_soa = 100.0 * floor / soa_780
    print(f"\n[benchmark] implied melter efficiency from the model floor "
          f"({floor:.1f} kWh/tonne)"
          f"\n  specialty glass electric melter, 10.3 MMBtu/short ton = "
          f"{specialty:.1f} kWh/tonne -> {eta_specialty:.1f} percent"
          f"\n  state-of-the-art electric melter, 780 kWh/short ton  = "
          f"{soa_780:.1f} kWh/tonne -> {eta_soa:.1f} percent"
          f"\n  LBNL stated band for continuous furnaces: 33 to 40 percent")

    assert 10.0 < eta_specialty < 33.0
    assert 40.0 < eta_soa < 100.0
    low_range = 8.9 * mmbtu_j / short_ton_kg / 3.6e6 * 1000.0
    high_range = 11.6 * mmbtu_j / short_ton_kg / 3.6e6 * 1000.0
    assert low_range == pytest.approx(2875.2, abs=0.1)
    assert high_range == pytest.approx(3747.4, abs=0.1)
    soa_800 = 800.0 / (short_ton_kg / 1000.0)
    assert soa_800 == pytest.approx(881.8, abs=0.1)
    assert floor == pytest.approx(599.7, abs=0.1)
    assert eta_specialty == pytest.approx(18.0, abs=0.1)
    assert eta_soa == pytest.approx(69.7, abs=0.1)


@pytest.mark.benchmark
def test_benchmark_quartz_cp_against_dulong_petit() -> None:
    """High-temperature Cp of quartz against the Dulong-Petit classical limit.

    Dulong-Petit: a solid with N atoms per formula unit approaches 3 N R at high
    temperature. SiO2 has N = 3, so
      3 x 3 x 8.31446 = 74.83 J/(mol K)

    Model at 1000 K (above the inversion, so no Landau term): 68.898 J/(mol K).

    Error against the classical limit: (68.898 - 74.83) / 74.83 = -7.93 percent.

    A real solid approaches the limit from BELOW and reaches it only well above its Debye
    temperature, so a value 8 percent short at 1000 K is physically correct rather than a
    discrepancy. This is an independent sanity check on the polynomial: a fitted Cp that
    OVERSHOT 3NR at 1000 K would indicate the fit had gone wrong.
    """
    r = 8.31446261815324
    dulong_petit = 9.0 * r
    assert dulong_petit == pytest.approx(74.83, abs=0.01)
    model = float(molar_heat_capacity(Polymorph.QUARTZ, Q_(1000.0, "K")).magnitude)
    assert model == pytest.approx(68.898, abs=1e-3)
    error_pct = 100.0 * (model - dulong_petit) / dulong_petit
    print(f"\n[benchmark] quartz Cp at 1000 K vs Dulong-Petit 3NR"
          f"\n  classical limit 3 x 3 x R = {dulong_petit:.3f} J/(mol K)"
          f"\n  model (ds62 polynomial)   = {model:.3f} J/(mol K)"
          f"\n  error: {error_pct:+.2f} percent (must be negative: a solid approaches "
          f"the limit from below)")
    assert -15.0 < error_pct < 0.0
    assert error_pct == pytest.approx(-7.93, abs=0.01)


@pytest.mark.benchmark
def test_benchmark_transition_enthalpies_are_small_and_positive() -> None:
    """Silica polymorph transition enthalpies against the accepted order of magnitude.

    Computed from the ds62 endmembers at the sourced transition temperatures:
      quartz -> tridymite      at 1143.15 K:  2.756 kJ/mol =  45.9 kJ/kg
      quartz -> cristobalite   at 1743.15 K:  3.059 kJ/mol =  50.9 kJ/kg
      cristobalite -> liquid   at 1996.00 K:  9.069 kJ/mol = 150.9 kJ/kg

    The literature consensus is that the SiO2 polymorph transitions have enthalpies of a
    few kJ/mol, roughly an order of magnitude below the enthalpy of fusion, which is what
    these numbers show: the reconstructive transitions cost little energy but are slow,
    which is precisely why they are kinetically controlled rather than energy controlled.
    Every value is positive on heating, as the second law requires for a transition to a
    higher-entropy phase at increasing temperature.
    """
    rows = [
        ("quartz -> tridymite", Polymorph.QUARTZ, Polymorph.TRIDYMITE, 1143.15, 2.756),
        ("quartz -> cristobalite", Polymorph.QUARTZ, Polymorph.CRISTOBALITE, 1743.15,
         3.059),
        ("cristobalite -> liquid", Polymorph.CRISTOBALITE, Polymorph.SILICA_LIQUID,
         1996.0, 9.069),
    ]
    print("\n[benchmark] silica transition enthalpies (ds62/ds633 endmembers)")
    for label, p1, p2, t_k, expected in rows:
        dh = specific_transition_enthalpy(p1, p2, Q_(t_k, "K"), allow_extrapolation=True)
        per_mol = float(dh.to("J/kg").magnitude) * M_SIO2 / 1000.0
        print(f"  {label:26s} at {t_k:7.2f} K: {per_mol:6.3f} kJ/mol = "
              f"{float(dh.to('kJ/kg').magnitude):6.1f} kJ/kg")
        assert per_mol == pytest.approx(expected, abs=5e-3)
        assert per_mol > 0.0, f"{label} must absorb heat on heating (second law)"
        assert per_mol < 15.0, f"{label} should be a few kJ/mol, not a fusion-scale value"
    dh_qt = specific_transition_enthalpy(rows[0][1], rows[0][2], Q_(rows[0][3], "K"), allow_extrapolation=True)
    kjkg_qt = float(dh_qt.to("kJ/kg").magnitude)
    assert kjkg_qt == pytest.approx(45.9, abs=0.05)
    dh_qc = specific_transition_enthalpy(rows[1][1], rows[1][2], Q_(rows[1][3], "K"), allow_extrapolation=True)
    kjkg_qc = float(dh_qc.to("kJ/kg").magnitude)
    assert kjkg_qc == pytest.approx(50.9, abs=0.05)
    dh_cl = specific_transition_enthalpy(rows[2][1], rows[2][2], Q_(rows[2][3], "K"), allow_extrapolation=True)
    kjkg_cl = float(dh_cl.to("kJ/kg").magnitude)
    assert kjkg_cl == pytest.approx(150.9, abs=0.05)


# --------------------------------------------------------------------------------------
# physical sanity, enforced in the code and pinned here
# --------------------------------------------------------------------------------------

def test_energy_balance_closes(ore) -> None:
    """theoretical + losses == supplied, and sensible + transition == theoretical."""
    eta = Value(quantity=Q_(0.35, "dimensionless"), tag=Tag.SOURCED,
                source=thermal.SRC_LBNL_GLASS,
                basis="mid-range of the 33 to 40 percent melting share")
    step = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
                       t_end=Q_(1996.0, "K"), transition_temperature=Q_(1996.0, "K"),
                       eta_thermal=eta)
    bal = fusion_energy(ore, step)
    theo = float(bal.theoretical.to("J/kg").magnitude)
    sup = float(bal.supplied.to("J/kg").magnitude)
    loss = float(bal.losses.to("J/kg").magnitude)
    sens = float(bal.sensible.to("J/kg").magnitude)
    trans = float(bal.transition.to("J/kg").magnitude)
    assert theo + loss == pytest.approx(sup, rel=1e-12)
    assert sens + trans == pytest.approx(theo, rel=1e-12)


def test_efficiency_above_one_rejected(ore) -> None:
    """An efficiency above unity violates the first law and must not be constructible."""
    with pytest.raises(ValueError):
        ThermalStep(
            name="impossible", from_phase=Polymorph.QUARTZ, to_phase=Polymorph.QUARTZ,
            t_start=Q_(298.15, "K"), t_end=Q_(900.0, "K"),
            eta_thermal=Value(quantity=Q_(1.4, "dimensionless"), tag=Tag.ASSUMED,
                              basis="deliberately impossible"),
        )


def test_broken_energy_balance_rejected() -> None:
    """The EnergyBalance model refuses to hold an unclosed balance."""
    with pytest.raises(ValueError, match="does not close"):
        EnergyBalance(
            step_name="broken", theoretical=Q_(100.0, "J/kg"), supplied=Q_(500.0, "J/kg"),
            losses=Q_(100.0, "J/kg"), overall_efficiency=0.5,
            sensible=Q_(100.0, "J/kg"), transition=Q_(0.0, "J/kg"), is_scenario=True,
        )


def test_cp_positive_across_the_range() -> None:
    """Cp must be positive at every temperature in every phase's assumed range."""
    for phase, coeffs in CP_COEFFICIENTS.items():
        t_lo = float(coeffs.t_min.to("K").magnitude)
        t_hi = float(coeffs.t_max.to("K").magnitude)
        for i in range(41):
            t = t_lo + (t_hi - t_lo) * i / 40.0
            cp = float(molar_heat_capacity(phase, Q_(t, "K")).magnitude)
            assert cp > 0.0, f"{phase} Cp = {cp} at {t} K"


def test_heating_enthalpy_is_monotonic() -> None:
    """Heating always absorbs heat: the integral must increase with the upper limit."""
    prev = -math.inf
    for t in (400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0):
        dh = float(integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(t, "K"))
                   .magnitude)
        assert dh > prev
        prev = dh


def test_enthalpy_antisymmetric() -> None:
    """Reversing the integration limits must negate the enthalpy change."""
    fwd = float(integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(900.0, "K"))
                .magnitude)
    rev = float(integrated_enthalpy(Polymorph.QUARTZ, Q_(900.0, "K"), Q_(298.15, "K"))
                .magnitude)
    assert fwd == pytest.approx(-rev, rel=1e-12)


def test_negative_absolute_temperature_rejected() -> None:
    with pytest.raises(ValueError):
        molar_heat_capacity(Polymorph.QUARTZ, Q_(-5.0, "K"))


def test_extrapolation_refused_by_default() -> None:
    """Beyond the assumed validity range the model raises rather than extrapolate."""
    with pytest.raises(ValueError, match="outside the"):
        molar_heat_capacity(Polymorph.QUARTZ, Q_(2500.0, "K"))
    cp = molar_heat_capacity(Polymorph.QUARTZ, Q_(2500.0, "K"), allow_extrapolation=True)
    assert float(cp.magnitude) > 0.0


def test_phase_change_requires_an_explicit_transition_temperature() -> None:
    """The datasets cannot locate the 1 bar boundaries, so the caller must supply one."""
    with pytest.raises(ValueError, match="no transition_temperature"):
        ThermalStep(
            name="fuse", from_phase=Polymorph.QUARTZ, to_phase=Polymorph.SILICA_LIQUID,
            t_start=Q_(298.15, "K"), t_end=Q_(1996.0, "K"),
            eta_thermal=Value(quantity=Q_(1.0, "dimensionless"), tag=Tag.ASSUMED,
                              basis="test"),
        )


def test_transition_outside_the_traversed_interval_rejected(ore, eta_unity) -> None:
    """A phase change cannot occur at a temperature the schedule never reaches."""
    step = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
                       t_end=Q_(1500.0, "K"), transition_temperature=Q_(1996.0, "K"),
                       eta_thermal=eta_unity)
    with pytest.raises(ValueError, match="outside the traversed interval"):
        fusion_energy(ore, step)


def test_cooling_step_rejected_by_heating_functions(ore, eta_unity) -> None:
    step = ThermalStep(name="cool", from_phase=Polymorph.QUARTZ, to_phase=Polymorph.QUARTZ,
                       t_start=Q_(1000.0, "K"), t_end=Q_(400.0, "K"),
                       eta_thermal=eta_unity)
    with pytest.raises(ValueError, match="ends below its start"):
        calcination_energy(ore, step)


def test_quench_must_cool(ore) -> None:
    with pytest.raises(ValueError, match="must cool"):
        quench_heat_rejection(ore, Polymorph.QUARTZ, Q_(400.0, "K"), Q_(900.0, "K"),
                              Q_(60.0, "s"))


def test_quench_duration_must_be_positive(ore) -> None:
    with pytest.raises(ValueError, match="duration must be positive"):
        quench_heat_rejection(ore, Polymorph.QUARTZ, Q_(900.0, "K"), Q_(400.0, "K"),
                              Q_(0.0, "s"))


# --------------------------------------------------------------------------------------
# ore-agnosticism and scenario labelling
# --------------------------------------------------------------------------------------

def test_feedstock_is_required(eta_unity) -> None:
    """Every model function takes a Feedstock. Passing anything else must fail loudly."""
    step = ThermalStep(name="calcine", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.QUARTZ, t_start=Q_(298.15, "K"),
                       t_end=Q_(900.0, "K"), eta_thermal=eta_unity)
    with pytest.raises(TypeError, match="Feedstock"):
        calcination_energy("Vikarabad", step)  # type: ignore[arg-type]


def test_site_is_required(ore, eta_unity) -> None:
    step = ThermalStep(name="calcine", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.QUARTZ, t_start=Q_(298.15, "K"),
                       t_end=Q_(900.0, "K"), eta_thermal=eta_unity)
    bal = calcination_energy(ore, step)
    with pytest.raises(TypeError, match="Site"):
        electricity_cost(bal, "Telangana")  # type: ignore[arg-type]


def test_uncharacterized_ore_marks_every_result_a_scenario(ore, eta_unity) -> None:
    """An uncharacterized deposit can only produce scenarios, never findings."""
    assert ore.characterized is False
    step = ThermalStep(name="calcine", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.QUARTZ, t_start=Q_(298.15, "K"),
                       t_end=Q_(1173.15, "K"), eta_thermal=eta_unity)
    bal = calcination_energy(ore, step)
    assert bal.is_scenario is True
    assert any("SCENARIO" in n for n in bal.notes)


def test_energy_is_independent_of_the_deposit(eta_unity) -> None:
    """Cp is a property of SiO2, so two different ores must give the same demand.

    This pins ore-agnosticism from the other side: the deposit changes the labelling and
    the downstream economics, not the thermodynamics of the silica itself.
    """
    a = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="Vikarabad", country="IN")
    b = Feedstock(sample_id="AE-Q-US-SPRUCE-001", ore_type=OreType.ALASKITE,
                  deposit_name="Spruce Pine", country="US")
    step = ThermalStep(name="calcine", from_phase=Polymorph.QUARTZ,
                       to_phase=Polymorph.QUARTZ, t_start=Q_(298.15, "K"),
                       t_end=Q_(1173.15, "K"), eta_thermal=eta_unity)
    assert (calcination_energy(a, step).kwh_per_tonne
            == pytest.approx(calcination_energy(b, step).kwh_per_tonne))


def test_glass_use_is_flagged_in_the_result(ore, eta_unity) -> None:
    """Any result that leaned on the liquid-for-glass substitution must say so."""
    step = ThermalStep(name="anneal glass", from_phase=Polymorph.SILICA_GLASS,
                       to_phase=Polymorph.SILICA_GLASS, t_start=Q_(298.15, "K"),
                       t_end=Q_(1400.0, "K"), eta_thermal=eta_unity)
    bal = calcination_energy(ore, step)
    assert bal.uses_liquid_cp_for_glass is True
    assert any("LIQUID endmember" in n for n in bal.notes)


def test_alpha_and_beta_quartz_share_the_endmember() -> None:
    """Both sides of the displacive inversion resolve to the same Cp coefficients."""
    for phase in (Polymorph.ALPHA_QUARTZ, Polymorph.BETA_QUARTZ, Polymorph.QUARTZ):
        cp = molar_heat_capacity(phase, Q_(700.0, "K"))
        assert float(cp.magnitude) == pytest.approx(
            float(molar_heat_capacity(Polymorph.QUARTZ, Q_(700.0, "K")).magnitude))


def test_amorphous_resolves_to_glass() -> None:
    """AMORPHOUS is an alias for the glass coefficients, which is deliberate.

    It must therefore NOT raise, and it must inherit the liquid-proxy flag so the
    substitution stays visible however the phase was named.
    """
    cp = molar_heat_capacity(Polymorph.AMORPHOUS, Q_(700.0, "K"))
    assert float(cp.magnitude) == pytest.approx(82.5)
    assert CP_COEFFICIENTS[Polymorph.SILICA_GLASS].is_liquid_proxy_for_glass is True


def test_phase_without_coefficients_raises(monkeypatch) -> None:
    """A polymorph with no Cp entry must raise rather than fall back to a default.

    Tridymite is removed from the table for the duration of the test, which proves the
    lookup has no silent default: an unmapped phase is an error, not a guess.
    """
    patched = {k: v for k, v in CP_COEFFICIENTS.items() if k is not Polymorph.TRIDYMITE}
    monkeypatch.setattr(thermal, "CP_COEFFICIENTS", patched)
    with pytest.raises(KeyError, match="no Cp coefficients"):
        molar_heat_capacity(Polymorph.TRIDYMITE, Q_(1200.0, "K"))
