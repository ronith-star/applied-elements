"""Tests for ae.physics.phases: transition windows, Landau strain, kinetics, risk screen."""

from __future__ import annotations

import math

import pytest

from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import Tag, Value
from ae.core.units import DimensionalityError, Q_
from ae.physics import phases
from ae.physics.phases import (
    QUARTZ_LANDAU,
    TRANSITIONS,
    ContactMaterial,
    LandauParameters,
    Polymorph,
    RiskLevel,
    ScheduleSegment,
    ThermalSchedule,
    Transition,
    TransitionCharacter,
    alpha_beta_cumulative_volume_strain,
    arrhenius_rate_ratio,
    cristobalite_onset_temperature,
    crossed_transitions,
    devitrification_risk,
    excess_molar_volume,
    jmak_fraction,
    landau_critical_temperature,
    order_parameter,
    require_molar_volume,
)


@pytest.fixture
def ore() -> Feedstock:
    return Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="Vikarabad", country="IN")


@pytest.fixture
def characterized_ore() -> Feedstock:
    """A synthetic characterized ore, used ONLY to prove the scenario flag flips.

    The per-element values below are INVENTED test numbers, not measurements and not
    attributable to any publication. Xia et al. 2024 (doi 10.3390/min14070727) report a
    TOTAL of 128.86 ug/g across Al, K, Ca, Na, Ti, Fe and Li with no per-element
    breakdown available from this sandbox (only the abstract was retrievable), so no
    source can back an element-by-element profile and none is claimed: every value is
    Tag.ASSUMED with a basis saying so. The numbers are chosen only to satisfy the
    ``characterized=True`` validator, which requires Al, Ti, Li, Fe, Na, K and B present
    by a single-grain-capable method. No test in this file asserts anything about their
    magnitudes.
    """
    def v(ppm: float) -> Value:
        return Value(
            quantity=Q_(ppm, "ppm_mass"), tag=Tag.ASSUMED,
            basis="INVENTED test fixture value, not a measurement and not sourced to any "
                  "publication. Present only so characterized=True can be constructed; no "
                  "assertion depends on its magnitude.",
            confidence="low",
        )

    prof = ImpurityProfile(
        total={"Al": v(60.0), "Ti": v(3.0), "Li": v(2.0), "Fe": v(5.0), "Na": v(10.0),
               "K": v(20.0), "B": v(0.5)},
        method="LA_ICP_MS",
    )
    return Feedstock(sample_id="AE-Q-PK-VEIN-001", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="synthetic test fixture, not a real deposit",
                     country="PK", impurities=prof, characterized=True)


def _schedule(peak_k: float, hold_h: float = 1.0, **kw) -> ThermalSchedule:
    return ThermalSchedule(
        start_temperature=Q_(298.15, "K"),
        segments=(ScheduleSegment(target_temperature=Q_(peak_k, "K"),
                                  hold_time=Q_(hold_h, "hour")),),
        **kw,
    )


# --------------------------------------------------------------------------------------
# dimensional analysis
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("func", [order_parameter, excess_molar_volume])
def test_bare_float_rejected(func) -> None:
    with pytest.raises(TypeError):
        func(500.0)


@pytest.mark.parametrize("wrong", [Q_(500.0, "J"), Q_(500.0, "kg"), Q_(500.0, "m")])
def test_wrong_dimension_rejected(wrong) -> None:
    with pytest.raises(DimensionalityError):
        order_parameter(wrong)


def test_molar_volume_guard() -> None:
    """The local molar-volume check must mirror the core dimensionality contract."""
    assert require_molar_volume(Q_(2.269e-5, "m**3/mol"), "v") is not None
    with pytest.raises(TypeError):
        require_molar_volume(2.269e-5, "v")
    with pytest.raises(DimensionalityError):
        require_molar_volume(Q_(2.269e-5, "m**3"), "v")


def test_excess_volume_dimensionality() -> None:
    v = excess_molar_volume(Q_(298.15, "K"))
    assert v.dimensionality == Q_(1.0, "m**3/mol").dimensionality


def test_order_parameter_and_strain_are_dimensionless() -> None:
    assert isinstance(order_parameter(Q_(500.0, "K")), float)
    assert isinstance(
        alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K")), float)


def test_critical_temperature_dimensionality() -> None:
    tc = landau_critical_temperature()
    assert tc.dimensionality == Q_(1.0, "K").dimensionality


def test_arrhenius_ratio_is_dimensionless_and_activation_energy_is_molar() -> None:
    """E_a must be molar energy; a bare energy or a temperature must be refused."""
    r = arrhenius_rate_ratio(Q_(1500.0, "K"), Q_(1550.0, "K"))
    assert isinstance(r, float)
    bad = Value(quantity=Q_(555.0, "kJ"), tag=Tag.ASSUMED, basis="wrong dimension")
    with pytest.raises(DimensionalityError):
        arrhenius_rate_ratio(Q_(1500.0, "K"), Q_(1550.0, "K"), activation_energy=bad)


def test_jmak_rate_constant_must_be_a_rate() -> None:
    bad = Value(quantity=Q_(1.0, "s"), tag=Tag.ASSUMED, basis="wrong dimension")
    with pytest.raises(DimensionalityError):
        jmak_fraction(Q_(1.0, "hour"), bad)


def test_landau_parameters_reject_wrong_dimensions() -> None:
    with pytest.raises(DimensionalityError):
        LandauParameters(
            tc_0=Value(quantity=Q_(847.0, "K"), tag=Tag.ASSUMED, basis="t"),
            s_d=Value(quantity=Q_(4.95, "J/(mol*K)"), tag=Tag.ASSUMED, basis="t"),
            v_d=Value(quantity=Q_(1.188e-6, "m**3"), tag=Tag.ASSUMED, basis="t"),
            v_0=Value(quantity=Q_(2.269e-5, "m**3/mol"), tag=Tag.ASSUMED, basis="t"),
            p_0=Value(quantity=Q_(1.0e5, "Pa"), tag=Tag.ASSUMED, basis="t"),
        )


# --------------------------------------------------------------------------------------
# provenance and documentation discipline
# --------------------------------------------------------------------------------------

def test_every_transition_temperature_is_sourced() -> None:
    for t in TRANSITIONS:
        assert t.temperature.tag is Tag.SOURCED, t.name
        assert t.temperature.source is not None
        assert t.temperature.source.doi, t.name
        assert t.temperature.source.accessed is not None


def test_every_landau_parameter_is_sourced() -> None:
    for name in ("tc_0", "s_d", "v_d", "v_0", "p_0"):
        v: Value = getattr(QUARTZ_LANDAU, name)
        assert v.tag is Tag.SOURCED
        assert v.source is not None and v.source.doi


def test_synthetic_fixture_values_are_not_tagged_sourced(characterized_ore) -> None:
    """A fixture's invented numbers must never claim a publication as their source.

    No per-element impurity breakdown for any vein quartz was retrievable from this
    sandbox, so an element-by-element profile cannot be SOURCED. This test pins that the
    synthetic fixture declares itself ASSUMED, which is the rule that stops a fabricated
    number acquiring a real DOI by being convenient.
    """
    for element, value in characterized_ore.impurities.total.items():
        assert value.tag is Tag.ASSUMED, element
        assert value.source is None, element
        assert value.basis is not None and "INVENTED" in value.basis, element


def test_limitations_names_the_missing_pre_exponential() -> None:
    doc = phases.__doc__ or ""
    assert "LIMITATIONS" in doc
    assert "NO ABSOLUTE KINETICS" in doc
    assert "10.1111/jace.12889" in doc or "Zhang" in doc


def test_module_documents_the_brief_correction() -> None:
    """The 3.7 vol% versus 0.4 vol% conflation must be documented, not silently resolved."""
    doc = phases.__doc__ or ""
    assert "3.7 vol%" in doc
    assert "0.4%" in doc


# --------------------------------------------------------------------------------------
# golden tests
# --------------------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_order_parameter_at_room_temperature() -> None:
    """Landau order parameter of quartz at 298.15 K and 1 bar.

    Q_0 = ((Tc_0 - T)/Tc_0)^{1/4} with Tc_0 = 847 K:
      847 - 298.15            = 548.85
      548.85 / 847            = 0.64799291
      0.64799291^{1/2}        = 0.80498007
      0.80498007^{1/2}        = 0.89720682

    (taking the fourth root as two successive square roots, which is how it can be
    checked on a plain calculator)
    """
    assert (847.0 - 298.15) / 847.0 == pytest.approx(0.64799291, abs=1e-8)
    assert math.sqrt(0.64799291) == pytest.approx(0.80498007, abs=1e-8)
    assert math.sqrt(0.80498007) == pytest.approx(0.89720682, abs=1e-8)
    assert order_parameter(Q_(298.15, "K")) == pytest.approx(0.89720682, abs=1e-8)


@pytest.mark.golden
def test_golden_cumulative_volume_strain_across_the_inversion() -> None:
    """Cumulative transition-related expansion of quartz, 298.15 K to above Tc.

    V_ex(T) = V_D Q^2, and above Tc the order parameter is zero, so the whole ordering
    volume is released:
      V_ex(298.15) = 1.188e-6 x 0.80498007 = 9.5631633e-7 m^3/mol
      V_ex(900)    = 0
      strain       = 9.5631633e-7 / 2.269e-5 = 0.0421470 = 4.2147 vol%

    This is the quantity the trade literature quotes as 3.5 to 4.5 vol% for the alpha to
    beta inversion, and the platform brief's 3.7 vol% sits inside that band. It is NOT the
    0.4 vol% step at the inversion that Ringdalen 2015 measured; the two differ by an order
    of magnitude and this module returns them separately.
    """
    assert 1.188e-6 * 0.80498007 == pytest.approx(9.5631633e-7, abs=1e-14)
    v = excess_molar_volume(Q_(298.15, "K"))
    assert float(v.to("m**3/mol").magnitude) == pytest.approx(9.5631633e-7, abs=1e-14)
    assert float(excess_molar_volume(Q_(900.0, "K")).to("m**3/mol").magnitude) == 0.0
    strain = alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K"))
    assert 9.5631633e-7 / 2.269e-5 == pytest.approx(0.0421470, abs=1e-7)
    assert strain == pytest.approx(0.0421470, abs=1e-7)
    assert 0.035 < strain < 0.045, "must bracket the trade-literature 3.5 to 4.5 vol% band"


@pytest.mark.golden
def test_golden_step_strain_is_an_order_of_magnitude_smaller() -> None:
    """The STEP at the inversion, from the transition table, is 0.4 vol%.

    Ringdalen 2015 measured "an increase in volume of around 0.4%" at 573 degC. Against
    the cumulative 4.2147 vol% computed from the Landau volume, the ratio is
      0.042147 / 0.004 = 10.5

    So conflating the two overstates the strain by a factor of about ten. That is the
    error this module exists to prevent.
    """
    inv = next(t for t in TRANSITIONS if t.name == "alpha_beta_quartz_inversion")
    assert inv.volume_strain is not None
    step = float(inv.volume_strain.quantity.to("dimensionless").magnitude)
    assert step == pytest.approx(0.004, abs=1e-12)
    cumulative = alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K"))
    assert cumulative / step == pytest.approx(10.5, abs=0.1)


@pytest.mark.golden
def test_golden_arrhenius_rate_ratio() -> None:
    """Rate ratio for cristobalite formation between 1200 and 1250 degC.

    k(T2)/k(T1) = exp[-(E_a/R)(1/T2 - 1/T1)] with E_a = 555000 J/mol,
    R = 8.31446262 J/(mol K), T1 = 1473.15 K, T2 = 1523.15 K:
      E_a/R      = 555000 / 8.31446262 = 66751.157
      1/T2       = 1 / 1523.15 = 6.565342e-4
      1/T1       = 1 / 1473.15 = 6.788175e-4
      difference = -2.228334e-5
      exponent   = -66751.157 x (-2.228334e-5) = 1.487439
      ratio      = exp(1.487439) = 4.425747

    A 50 K rise near 1200 degC multiplies the rate by about 4.43, which is the quantitative
    statement of why cristobalite control in a crucible is a temperature-control problem.
    """
    ea, r = 555000.0, 8.31446261815324
    assert ea / r == pytest.approx(66751.157, abs=0.01)
    assert 1.0 / 1523.15 - 1.0 / 1473.15 == pytest.approx(-2.228334e-5, abs=1e-11)
    ratio = arrhenius_rate_ratio(Q_(1473.15, "K"), Q_(1523.15, "K"))
    assert math.exp(-(ea / r) * (1.0 / 1523.15 - 1.0 / 1473.15)) == pytest.approx(
        4.425747, abs=1e-5)
    assert ratio == pytest.approx(4.425747, abs=1e-5)


@pytest.mark.golden
def test_golden_jmak_fraction() -> None:
    """JMAK transformed fraction at the characteristic time.

    X = 1 - exp[-(k t)^n]. With k t = 1 and n = 3:
      (k t)^n = 1
      X       = 1 - exp(-1) = 1 - 0.36787944 = 0.63212056

    With k t = 2 and n = 3:
      (k t)^n = 8
      X       = 1 - exp(-8) = 1 - 0.00033546 = 0.99966454
    """
    k = Value(quantity=Q_(1.0, "1/hour"), tag=Tag.ASSUMED,
              basis="unit rate constant for a hand-traceable test; no published "
                    "pre-exponential factor exists for this transformation")
    assert jmak_fraction(Q_(1.0, "hour"), k) == pytest.approx(0.63212056, abs=1e-8)
    assert 1.0 - math.exp(-1.0) == pytest.approx(0.63212056, abs=1e-8)
    assert jmak_fraction(Q_(2.0, "hour"), k) == pytest.approx(0.99966454, abs=1e-8)
    assert 1.0 - math.exp(-8.0) == pytest.approx(0.99966454, abs=1e-8)


# --------------------------------------------------------------------------------------
# benchmark tests
# --------------------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_inversion_temperature() -> None:
    """Landau Tc_0 against the textbook alpha to beta quartz inversion temperature.

    Literature: the inversion is universally quoted at 573 degC = 846.15 K, and the
    platform brief states 573 C.
    Model: the ds62 Landau parameter Tc_0 = 847 K = 573.85 degC.
    Error: (847 - 846.15) / 846.15 = +0.10 percent on the absolute scale, i.e. 0.85 K.

    A sub-kelvin agreement on an independently fitted thermodynamic dataset is a genuine
    validation of the Landau parameterisation, and it is worth noting the real transition
    temperature in natural quartz shifts by a few kelvin with Al and Li content, which is
    larger than this model error.
    """
    tc_model = float(landau_critical_temperature().to("K").magnitude)
    tc_lit = 846.15
    error_pct = 100.0 * (tc_model - tc_lit) / tc_lit
    print(f"\n[benchmark] alpha to beta quartz inversion temperature"
          f"\n  literature: 573 degC = {tc_lit:.2f} K"
          f"\n  model (ds62 Landau Tc_0): {tc_model:.2f} K = "
          f"{tc_model - 273.15:.2f} degC"
          f"\n  error: {error_pct:+.3f} percent ({tc_model - tc_lit:+.2f} K)")
    assert abs(error_pct) < 0.5


@pytest.mark.benchmark
def test_benchmark_cumulative_strain_against_the_trade_figure() -> None:
    """Landau expansion against the 3.7 vol% figure quoted in the platform brief.

    Literature (trade figure, as stated in the platform brief): roughly 3.7 vol% expansion
    associated with the alpha to beta quartz inversion.
    Model (ds62 Landau volume of disordering): 4.2147 vol%.
    Error: (4.2147 - 3.7) / 3.7 = +13.9 percent.

    The agreement is to within 14 percent, which for a quantity quoted without a source or
    an uncertainty is about as close as it can be checked. The much more important finding
    is that BOTH numbers describe the cumulative expansion and NEITHER describes the step at
    the inversion, which Ringdalen 2015 measured at 0.4 vol%.
    """
    model = 100.0 * alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K"))
    lit = 3.7
    error_pct = 100.0 * (model - lit) / lit
    print(f"\n[benchmark] alpha to beta cumulative volume expansion"
          f"\n  trade figure (platform brief): {lit:.2f} vol%"
          f"\n  model (ds62 V_D Q_0^2 / V_0):  {model:.4f} vol%"
          f"\n  error: {error_pct:+.2f} percent"
          f"\n  note: Ringdalen 2015 measures the STEP at the inversion as 0.4 vol%, a "
          f"different quantity")
    assert abs(error_pct) < 20.0


@pytest.mark.benchmark
def test_benchmark_tridymite_boundary_from_gibbs_energies() -> None:
    """Independently locate the quartz-tridymite boundary from ds62 Gibbs energies.

    This is a real validation because the transition temperature is NOT an input to the
    dataset: it falls out of equating G(quartz) and G(tridymite), each built from its own
    H_0, S_0 and Cp polynomial.

    Literature: beta-quartz to tridymite near 870 degC (Ringdalen 2015, quoting the SiO2
    phase diagram).
    Model: root of G(q) - G(trd), found by bisection below.
    Error is reported.

    The same calculation places tridymite-cristobalite above 2600 K against the accepted
    1470 degC, which is why ae.physics.thermal LIMITATIONS item 3 says to take transition
    temperatures from this module's sourced table and not from the dataset. Reporting the
    one that works and the one that does not is the point.
    """
    from ae.physics.thermal import CP_COEFFICIENTS, FORMATION_ENTHALPY, landau_excess_enthalpy

    s0 = {Polymorph.QUARTZ: 41.43, Polymorph.TRIDYMITE: 44.10}

    def gibbs(phase: Polymorph, t_k: float) -> float:
        a, b, c, d = CP_COEFFICIENTS[phase].raw
        t0 = 298.15

        def int_cp(t: float) -> float:
            return a * t + 0.5 * b * t * t - c / t + 2.0 * d * math.sqrt(t)

        def int_cp_over_t(t: float) -> float:
            return a * math.log(t) + b * t - c / (2.0 * t * t) - 2.0 * d / math.sqrt(t)

        h0 = float(FORMATION_ENTHALPY[phase].quantity.to("J/mol").magnitude)
        g = (h0 + (int_cp(t_k) - int_cp(t0))
             - t_k * (s0[phase] + int_cp_over_t(t_k) - int_cp_over_t(t0)))
        if phase is Polymorph.QUARTZ:
            q = order_parameter(Q_(t_k, "K"))
            q_0 = order_parameter(Q_(t0, "K"))
            tc0, s_d = 847.0, 4.95
            g += (tc0 * s_d * (q_0 ** 2 - q_0 ** 6 / 3.0)
                  - s_d * (tc0 * q ** 2 - tc0 * q ** 6 / 3.0)
                  - t_k * s_d * (q_0 ** 2 - q ** 2))
        return g

    lo, hi = 800.0, 1500.0
    f_lo = gibbs(Polymorph.QUARTZ, lo) - gibbs(Polymorph.TRIDYMITE, lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = gibbs(Polymorph.QUARTZ, mid) - gibbs(Polymorph.TRIDYMITE, mid)
        if (f_mid < 0.0) == (f_lo < 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    t_model_k = 0.5 * (lo + hi)
    t_model_c = t_model_k - 273.15
    t_lit_c = 870.0
    error_pct = 100.0 * (t_model_c - t_lit_c) / t_lit_c
    print(f"\n[benchmark] quartz to tridymite equilibrium from ds62 Gibbs energies"
          f"\n  literature (SiO2 phase diagram, via Ringdalen 2015): {t_lit_c:.1f} degC"
          f"\n  model (root of G_q = G_trd): {t_model_c:.2f} degC ({t_model_k:.2f} K)"
          f"\n  error: {error_pct:+.2f} percent"
          f"\n  the same dataset places tridymite-cristobalite above 2600 K against the "
          f"accepted 1470 degC, documented in thermal.LIMITATIONS item 3")
    assert t_model_c == pytest.approx(866.11, abs=0.05)
    assert abs(error_pct) < 2.0
    assert landau_excess_enthalpy is not None


# --------------------------------------------------------------------------------------
# physical sanity
# --------------------------------------------------------------------------------------

def test_order_parameter_bounds_and_monotonicity() -> None:
    """Q must lie in [0, 1] and fall monotonically with temperature."""
    prev = math.inf
    for t in (100.0, 300.0, 500.0, 700.0, 800.0, 846.0, 847.0, 1000.0):
        q = order_parameter(Q_(t, "K"))
        assert 0.0 <= q <= 1.0, t
        assert q <= prev + 1e-12, t
        prev = q
    assert order_parameter(Q_(847.0, "K")) == 0.0
    assert order_parameter(Q_(1200.0, "K")) == 0.0


def test_excess_volume_never_negative() -> None:
    for t in (100.0, 500.0, 846.9, 847.0, 2000.0):
        assert float(excess_molar_volume(Q_(t, "K")).magnitude) >= 0.0


def test_strain_sign_convention() -> None:
    """Heating expands, cooling contracts, and the two must be exact negatives."""
    up = alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K"))
    down = alpha_beta_cumulative_volume_strain(Q_(900.0, "K"), Q_(298.15, "K"))
    assert up > 0.0
    assert down == pytest.approx(-up, rel=1e-12)


def test_arrhenius_rate_always_positive_and_ordered() -> None:
    """Rates are strictly positive, and a higher temperature must give a higher rate."""
    for t2 in (1480.0, 1520.0, 1560.0, 1600.0):
        r = arrhenius_rate_ratio(Q_(1473.15, "K"), Q_(t2, "K"))
        assert r > 0.0
        assert r > 1.0
    assert arrhenius_rate_ratio(Q_(1550.0, "K"), Q_(1500.0, "K")) < 1.0


def test_arrhenius_self_ratio_is_unity() -> None:
    assert arrhenius_rate_ratio(Q_(1500.0, "K"), Q_(1500.0, "K")) == pytest.approx(1.0)


def test_negative_activation_energy_rejected() -> None:
    bad = Value(quantity=Q_(-100.0, "kJ/mol"), tag=Tag.ASSUMED, basis="unphysical")
    with pytest.raises(ValueError, match="activation energy must be positive"):
        arrhenius_rate_ratio(Q_(1500.0, "K"), Q_(1550.0, "K"), activation_energy=bad)


def test_extrapolation_outside_the_calibration_band_refused() -> None:
    """The default E_a must not be silently extrapolated past its rate maximum."""
    with pytest.raises(ValueError, match="outside the 1473 to 1623 K band"):
        arrhenius_rate_ratio(Q_(1200.0, "K"), Q_(1300.0, "K"))
    r = arrhenius_rate_ratio(Q_(1200.0, "K"), Q_(1300.0, "K"), allow_extrapolation=True)
    assert r > 1.0


def test_jmak_bounds() -> None:
    """The transformed fraction must stay within [0, 1] for any positive time."""
    k = Value(quantity=Q_(1.0, "1/hour"), tag=Tag.ASSUMED, basis="test")
    assert jmak_fraction(Q_(0.0, "hour"), k) == 0.0
    for t in (0.01, 0.5, 1.0, 5.0, 100.0):
        x = jmak_fraction(Q_(t, "hour"), k)
        assert 0.0 <= x <= 1.0


def test_jmak_rejects_negative_inputs() -> None:
    k = Value(quantity=Q_(1.0, "1/hour"), tag=Tag.ASSUMED, basis="test")
    with pytest.raises(ValueError, match="hold time cannot be negative"):
        jmak_fraction(Q_(-1.0, "hour"), k)
    bad = Value(quantity=Q_(-1.0, "1/hour"), tag=Tag.ASSUMED, basis="test")
    with pytest.raises(ValueError, match="rate constant cannot be negative"):
        jmak_fraction(Q_(1.0, "hour"), bad)


def test_jmak_requires_an_explicit_rate_constant() -> None:
    """There is no default k, because no published pre-exponential factor exists."""
    with pytest.raises(TypeError):
        jmak_fraction(Q_(1.0, "hour"))  # type: ignore[call-arg]


def test_transitions_are_ordered_and_have_positive_temperatures() -> None:
    temps = [t.temperature_k for t in TRANSITIONS]
    assert temps == sorted(temps)
    assert all(t > 0.0 for t in temps)


def test_schedule_requires_a_segment() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        ThermalSchedule(start_temperature=Q_(298.15, "K"), segments=())


def test_negative_hold_time_rejected() -> None:
    with pytest.raises(ValueError, match="hold time cannot be negative"):
        ScheduleSegment(target_temperature=Q_(1000.0, "K"), hold_time=Q_(-1.0, "hour"))


# --------------------------------------------------------------------------------------
# process windows
# --------------------------------------------------------------------------------------

def test_calcination_crosses_only_the_inversion(ore) -> None:
    hit = crossed_transitions(_schedule(1073.15), ore)
    assert [t.name for t in hit] == ["alpha_beta_quartz_inversion"]
    assert hit[0].character is TransitionCharacter.DISPLACIVE
    assert hit[0].reversible_on_cooling is True


def test_fusion_schedule_crosses_everything(ore) -> None:
    hit = crossed_transitions(_schedule(2100.0), ore)
    assert [t.name for t in hit] == [
        "alpha_beta_quartz_inversion", "beta_quartz_to_tridymite",
        "tridymite_to_cristobalite", "silica_melting",
    ]


def test_tridymite_boundary_is_marked_equilibrium_only(ore) -> None:
    """Crossing 870 degC permits tridymite; Ringdalen 2015 did not observe it forming."""
    hit = crossed_transitions(_schedule(1200.0), ore)
    trd = next(t for t in hit if t.name == "beta_quartz_to_tridymite")
    assert trd.equilibrium_only is True
    assert trd.character is TransitionCharacter.RECONSTRUCTIVE
    assert "not observe" in (trd.note or "")


def test_low_temperature_schedule_crosses_nothing(ore) -> None:
    assert crossed_transitions(_schedule(500.0), ore) == ()


def test_crossed_transitions_requires_a_feedstock() -> None:
    with pytest.raises(TypeError, match="Feedstock"):
        crossed_transitions(_schedule(1073.15), "Vikarabad")  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# cristobalite onset and the SiC correction
# --------------------------------------------------------------------------------------

def test_crystalline_onset_is_sourced_at_1470c() -> None:
    v = cristobalite_onset_temperature(Polymorph.ALPHA_QUARTZ)
    assert v.tag is Tag.SOURCED
    assert float(v.quantity.to("degC").magnitude) == pytest.approx(1470.0, abs=0.01)


def test_glass_onset_is_sourced_at_1000c() -> None:
    v = cristobalite_onset_temperature(Polymorph.SILICA_GLASS)
    assert v.tag is Tag.SOURCED
    assert float(v.quantity.to("degC").magnitude) == pytest.approx(1000.0, abs=0.01)


@pytest.mark.parametrize("contact", [ContactMaterial.GRAPHITE, ContactMaterial.ALUMINA,
                                     ContactMaterial.SILICON_CARBIDE])
def test_contact_material_onsets_are_honestly_assumed(contact) -> None:
    """No source gives a contact-shifted onset, so those values must be ASSUMED, not SOURCED."""
    v = cristobalite_onset_temperature(Polymorph.SILICA_GLASS, contact)
    assert v.tag is Tag.ASSUMED
    assert v.basis is not None and "ESTIMATE" in v.basis
    assert v.confidence == "low"


def test_sic_correction_is_recorded(ore) -> None:
    """The brief groups SiC with graphite and alumina; Warden et al. 2024 contradict it."""
    sched = _schedule(1773.15, hold_h=3.0, starting_phase=Polymorph.SILICA_GLASS,
                      contact=ContactMaterial.SILICON_CARBIDE)
    drivers = " ".join(devitrification_risk(sched, ore).drivers)
    assert "THINNEST" in drivers
    assert "contradicting" in drivers


def test_graphite_scores_higher_risk_than_sic(ore) -> None:
    """Graphite must rank as riskier than SiC, per the measured layer thicknesses."""
    order = {RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MODERATE: 2,
             RiskLevel.HIGH: 3, RiskLevel.SEVERE: 4}
    common = dict(hold_h=3.0, starting_phase=Polymorph.SILICA_GLASS, atmosphere="inert")
    g = devitrification_risk(_schedule(1773.15, contact=ContactMaterial.GRAPHITE,
                                       **common), ore)
    s = devitrification_risk(_schedule(1773.15, contact=ContactMaterial.SILICON_CARBIDE,
                                       **common), ore)
    assert order[g.level] > order[s.level]


# --------------------------------------------------------------------------------------
# devitrification screen
# --------------------------------------------------------------------------------------

def test_no_risk_below_the_onset(ore) -> None:
    a = devitrification_risk(_schedule(1073.15), ore)
    assert a.level is RiskLevel.NONE
    assert float(a.margin_above_onset.to("K").magnitude) < 0.0


def test_risk_rises_with_temperature_and_time(ore) -> None:
    order = {RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MODERATE: 2,
             RiskLevel.HIGH: 3, RiskLevel.SEVERE: 4}
    levels = [
        order[devitrification_risk(
            _schedule(peak, hold_h=hold, starting_phase=Polymorph.SILICA_GLASS), ore).level]
        for peak, hold in [(1280.0, 0.1), (1300.0, 1.0), (1600.0, 1.0), (1773.15, 7.0)]
    ]
    assert levels == sorted(levels)
    assert levels[0] < levels[-1]


def test_reducing_atmosphere_is_not_scored_as_enhancing(ore) -> None:
    """Ringdalen 2015: cristobalite forms more readily in inert than reducing atmospheres."""
    order = {RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.MODERATE: 2,
             RiskLevel.HIGH: 3, RiskLevel.SEVERE: 4}
    common = dict(hold_h=1.0, starting_phase=Polymorph.SILICA_GLASS)
    inert = devitrification_risk(_schedule(1600.0, atmosphere="inert", **common), ore)
    red = devitrification_risk(_schedule(1600.0, atmosphere="reducing", **common), ore)
    assert order[inert.level] >= order[red.level]
    assert any("reducing atmosphere retards" in d for d in red.drivers)


def test_assessment_reports_its_drivers(ore) -> None:
    a = devitrification_risk(_schedule(1773.15, hold_h=3.0,
                                       starting_phase=Polymorph.SILICA_GLASS,
                                       contact=ContactMaterial.GRAPHITE), ore)
    assert len(a.drivers) >= 3
    assert "NOT a predicted cristobalite fraction" in a.note


def test_uncharacterized_ore_is_a_scenario(ore, characterized_ore) -> None:
    """The scenario flag must track the ore, not the schedule."""
    sched = _schedule(1773.15, hold_h=3.0, starting_phase=Polymorph.SILICA_GLASS)
    assert devitrification_risk(sched, ore).is_scenario is True
    assert devitrification_risk(sched, characterized_ore).is_scenario is False


def test_devitrification_requires_a_feedstock() -> None:
    with pytest.raises(TypeError, match="Feedstock"):
        devitrification_risk(_schedule(1773.15), "Vikarabad")  # type: ignore[arg-type]


def test_hold_above_onset_accounting(ore) -> None:
    """Only segments at or above the onset count toward the exposure time."""
    sched = ThermalSchedule(
        start_temperature=Q_(298.15, "K"),
        starting_phase=Polymorph.SILICA_GLASS,
        segments=(
            ScheduleSegment(target_temperature=Q_(900.0, "K"), hold_time=Q_(5.0, "hour")),
            ScheduleSegment(target_temperature=Q_(1600.0, "K"), hold_time=Q_(2.0, "hour")),
        ),
    )
    a = devitrification_risk(sched, ore)
    assert float(a.hold_above_onset.to("hour").magnitude) == pytest.approx(2.0)


def test_transition_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError):
        Transition(
            name="impossible", from_phase=Polymorph.QUARTZ, to_phase=Polymorph.TRIDYMITE,
            temperature=Value(quantity=Q_(-10.0, "K"), tag=Tag.ASSUMED, basis="test"),
            character=TransitionCharacter.RECONSTRUCTIVE, reversible_on_cooling=False,
        )
