"""Tests for ae.physics.chlorination: chloride formation, volatility and the screen.

The structural claim under test is that the three removal conditions
(thermodynamic, volatility, transport) are kept SEPARATE, so that a pass on
volatility can never be mistaken for removal. The reductant rule from Liu et al.
(2026) is tested as a hard gate: Ti, Al and B must come back blocked without a
carbonaceous reductant at every temperature.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest
from pydantic import ValidationError

from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.units import Q_, require_dimensionality
from ae.physics.chlorination import (
    CHLORIDES,
    GAS_CONSTANT,
    P_REFERENCE,
    REDUCTANT_REQUIRED,
    SOURCE_LIU_2026,
    SOURCE_PUBCHEM,
    THERMO,
    THERMO_MISSING,
    TROUTON_ENTROPY,
    ChlorideSpecies,
    PhaseChange,
    gibbs_of_reaction,
    hertz_knudsen_flux,
    removal_screen,
    requires_reductant,
    screen_feedstock,
    trouton_enthalpy,
    vapour_pressure,
)

_SRC = Source(
    citation="Test fixture source, not a real reference",
    tier=Tier.T2, url="https://example.invalid/fixture", accessed=dt.date(2026, 9, 16),
    note="Synthetic fixture for unit tests.",
)


def _v(q, tag=Tag.SOURCED, **kw):
    if tag in (Tag.SOURCED, Tag.MEASURED):
        return Value(quantity=q, tag=tag, source=_SRC, **kw)
    kw.setdefault("basis", "test fixture")
    return Value(quantity=q, tag=tag, **kw)


@pytest.fixture
def quartz() -> Feedstock:
    imp = ImpurityProfile(
        total={e: _v(Q_(v, "ppm_mass")) for e, v in
               [("Al", 30.0), ("Ti", 10.0), ("Li", 5.0), ("Fe", 3.0),
                ("Na", 8.0), ("K", 8.0), ("B", 1.0)]},
        lattice_fraction={el: Value(
            quantity=Q_(0.5, "dimensionless"), tag=Tag.ASSUMED,
            basis="INVENTED test fixture split, not a measurement. Present only "
                  "so the 'located' tier can be constructed; no assertion "
                  "depends on its magnitude.",
            confidence="low") for el in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")},
        method="LA_ICP_MS",
    )
    return Feedstock(
        sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
        deposit_name="Synthetic test vein", country="IN", impurities=imp,
        characterized=True,
    )


# ---------------------------------------------------------------------------
# dimensional analysis
# ---------------------------------------------------------------------------

def test_vapour_pressure_is_a_pressure() -> None:
    p, _ = vapour_pressure(CHLORIDES["TiCl4"], Q_(100.0, "degC"))
    require_dimensionality(p, "pressure", "p_sat")


def test_hertz_knudsen_dimensionality() -> None:
    r"""J = alpha (p_sat - p_bulk)/sqrt(2 pi M R T) must reduce to mol m^-2 s^-1.

    Pa / sqrt(kg mol^-1 * J mol^-1 K^-1 * K) = Pa / sqrt(kg J mol^-2)
      = (kg m^-1 s^-2) / (kg m s^-1 mol^-1) = mol m^-2 s^-1.
    Checked here by requiring the pint reduction rather than by trusting the
    algebra in the docstring.
    """
    j = hertz_knudsen_flux(CHLORIDES["NaCl"], Q_(1200.0, "degC"))
    assert j.to("mol/(m**2*s)").dimensionality == Q_(1.0, "mol/(m**2*s)").dimensionality
    # and the explicit reduction of the denominator
    denom = (2.0 * math.pi * Q_(0.058443, "kg/mol") * GAS_CONSTANT * Q_(1473.15, "K")) ** 0.5
    ratio = (Q_(1.0, "Pa") / denom).to("mol/(m**2*s)")
    assert ratio.dimensionality == Q_(1.0, "mol/(m**2*s)").dimensionality


def test_trouton_enthalpy_is_molar_energy() -> None:
    h = trouton_enthalpy(Q_(136.45, "degC"))
    require_dimensionality(h.to("J/mol"), "molar_energy", "dHvap")


def test_gibbs_of_reaction_is_molar_energy() -> None:
    """A complete supplied table must give a molar energy, with the T dS term."""
    table = {
        "A": {"Hf": _v(Q_(-100.0, "kJ/mol")), "S": _v(Q_(50.0, "J/(mol*K)"))},
        "B": {"Hf": _v(Q_(-200.0, "kJ/mol")), "S": _v(Q_(80.0, "J/(mol*K)"))},
    }
    dg = gibbs_of_reaction({"B": 1.0, "A": -1.0}, Q_(1000.0, "K"), thermo=table)
    require_dimensionality(dg.to("J/mol"), "molar_energy", "dG")


# ---------------------------------------------------------------------------
# golden: hand-traceable arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_clausius_clapeyron_at_boiling_point() -> None:
    r"""At T = Tb the exponent vanishes and p_sat = p_ref exactly.

    p_sat(Tb) = p_ref exp[-(dHvap/R)(1/Tb - 1/Tb)] = p_ref exp(0) = 101325 Pa.
    This is the boundary condition the one-point form is anchored on, so it is
    an exact identity rather than a numerical result, and it must hold for every
    tabulated species.
    """
    for name, sp in CHLORIDES.items():
        p, capped = vapour_pressure(sp, sp.transition_T.quantity)
        assert float(p.to("Pa").magnitude) == pytest.approx(101325.0, rel=1e-9), name
        assert capped is False, name


@pytest.mark.golden
def test_golden_ticl4_vapour_pressure_below_boiling() -> None:
    r"""Hand-check p_sat for TiCl4 at 100 degC with its SOURCED dHvap = 36.2 kJ/mol.

    Tb = 136.45 degC = 409.60 K, T = 100 degC = 373.15 K.
      1/T  = 1/373.15 = 2.6798874e-3 K^-1
      1/Tb = 1/409.60  = 2.4414063e-3 K^-1
      1/T - 1/Tb       = 2.3848119e-4 K^-1
      dHvap/R    = 36200 / 8.314462618 = 4353.8593 K
      exponent   = -4353.8593 * 2.3848119e-4 = -1.038314
      p_sat      = 101325 * exp(-1.038314) = 101325 * 0.354051 = 35874.24 Pa

    i.e. 0.354 atm at 100 degC, which is consistent with TiCl4 being a volatile
    liquid well below its boiling point.
    """
    t_b = 136.45 + 273.15
    t = 373.15
    delta = 1.0 / t - 1.0 / t_b
    assert delta == pytest.approx(2.3848119e-4, rel=1e-7)
    h_over_r = 36200.0 / 8.314462618
    assert h_over_r == pytest.approx(4353.8593, rel=1e-7)
    expo = -h_over_r * delta
    assert expo == pytest.approx(-1.038314, rel=1e-6)
    expected = 101325.0 * math.exp(expo)
    assert expected == pytest.approx(35874.24, rel=1e-6)
    p, capped = vapour_pressure(CHLORIDES["TiCl4"], Q_(100.0, "degC"))
    assert float(p.to("Pa").magnitude) == pytest.approx(expected, rel=1e-6)
    assert capped is False
    assert float(p.to("Pa").magnitude) / 101325.0 == pytest.approx(0.354051, rel=1e-6)


@pytest.mark.golden
def test_golden_trouton_arithmetic() -> None:
    r"""Trouton: dHvap = 85 * Tb. For AlCl3 with Tb = 180 degC = 453.15 K,

      dHvap = 85 * 453.15 = 38517.75 J/mol = 38.52 kJ/mol.

    Reported for orientation: the experimental sublimation enthalpy of AlCl3 is
    NOT used here because it could not be sourced in this build, and AlCl3
    vapour is dimeric (Al2Cl6) near its sublimation point, where Trouton is
    known to be unreliable. The value is an estimate, flagged as such.
    """
    expected = 85.0 * 453.15
    assert expected == pytest.approx(38517.75, rel=1e-9)
    h = trouton_enthalpy(Q_(180.0, "degC"))
    assert float(h.to("J/mol").magnitude) == pytest.approx(expected, rel=1e-9)
    assert CHLORIDES["AlCl3"].enthalpy_is_trouton is True
    assert float(CHLORIDES["AlCl3"].enthalpy().to("J/mol").magnitude) == pytest.approx(
        expected, rel=1e-9)


@pytest.mark.golden
def test_golden_gibbs_arithmetic() -> None:
    r"""Hand-check dG = sum(nu dHf) - T sum(nu S) for a two-species reaction.

    Reaction A -> B with dHf(A) = -100 kJ/mol, S(A) = 50 J/(mol K),
                         dHf(B) = -200 kJ/mol, S(B) = 80 J/(mol K), T = 1000 K.
      sum(nu dHf) = (+1)(-200) + (-1)(-100) = -100 kJ/mol
      sum(nu S)   = (+1)(80) + (-1)(50) = +30 J/(mol K)
      dG = -100 kJ/mol - 1000 K * 0.030 kJ/(mol K) = -100 - 30 = -130 kJ/mol
    Spontaneous, and the entropy term makes it MORE spontaneous because the
    reaction increases entropy.
    """
    table = {
        "A": {"Hf": _v(Q_(-100.0, "kJ/mol")), "S": _v(Q_(50.0, "J/(mol*K)"))},
        "B": {"Hf": _v(Q_(-200.0, "kJ/mol")), "S": _v(Q_(80.0, "J/(mol*K)"))},
    }
    dg = gibbs_of_reaction({"B": 1.0, "A": -1.0}, Q_(1000.0, "K"), thermo=table)
    assert float(dg.to("kJ/mol").magnitude) == pytest.approx(-130.0, rel=1e-9)


@pytest.mark.golden
def test_golden_hertz_knudsen_arithmetic() -> None:
    r"""Hand-check J = p/sqrt(2 pi M R T) for NaCl at its boiling point.

    At Tb = 1465 degC = 1738.15 K, p_sat = 101325 Pa by the anchor condition.
      M = 58.443 g/mol = 0.058443 kg/mol
      R T        = 8.314462618 * 1738.15 = 14451.7832 J/mol
      2 pi M     = 6.2831853 * 0.058443 = 0.3672082 kg/mol
      2 pi M R T = 0.3672082 * 14451.7832 = 5306.8133 kg J mol^-2
      sqrt(...)  = 72.8479
      J = 101325 / 72.8479 = 1390.91 mol m^-2 s^-1

    An upper bound (free molecular escape, no boundary layer), and it is large
    because it is evaluated AT the boiling point where p_sat is a full
    atmosphere. At a 1200 degC roast NaCl sits below its boiling point and the
    flux is correspondingly smaller.
    """
    m = 0.058443
    denom_sq = 2.0 * math.pi * m * 8.314462618 * 1738.15
    assert denom_sq == pytest.approx(5306.8133, rel=1e-6)
    denom = math.sqrt(denom_sq)
    assert denom == pytest.approx(72.8479, rel=1e-6)
    expected = 101325.0 / denom
    assert expected == pytest.approx(1390.91, rel=1e-5)
    j = hertz_knudsen_flux(CHLORIDES["NaCl"], Q_(1465.0, "degC"))
    assert float(j.to("mol/(m**2*s)").magnitude) == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# benchmark: literature comparison with the error REPORTED
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_chloride_boiling_points(capsys: pytest.CaptureFixture[str]) -> None:
    """Tabulated transition points against the values stated in the task brief.

    The brief states AlCl3 boils at 180 degC, FeCl3 sublimes near 315 degC and
    TiCl4 boils at 136 degC. The module's values come independently from
    PubChem property records. The error between the two is reported per species.
    """
    brief = {"AlCl3": 180.0, "FeCl3": 315.0, "TiCl4": 136.0}
    for name, ref in brief.items():
        model = float(CHLORIDES[name].transition_T.quantity.to("degC").magnitude)
        err = abs(model - ref) / ref * 100.0
        print(f"\nBENCHMARK {name} transition point: brief {ref:.1f} degC, PubChem "
              f"{model:.2f} degC, error {err:.3f} percent "
              f"(source: {SOURCE_PUBCHEM.url}, accessed {SOURCE_PUBCHEM.accessed}, "
              f"CID {CHLORIDES[name].pubchem_cid})")
        assert err < 1.0, f"{name}: {err} percent from the brief value"


@pytest.mark.benchmark
def test_benchmark_alkali_chlorides_are_not_volatile_at_roast(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """Alkali chloride volatility at a 1200 degC roast, against their boiling points.

    The quantitative claim being checked: NaCl (1465 degC), KCl (sublimes
    1500 degC) and LiCl (1360 degC) all have 1 atm transition points ABOVE a
    1200 degC roast, so a naive "boiling point below roast temperature" screen
    would wrongly call them non-removable, while a vapour-pressure screen gives
    the actual partial pressure driving evaporation. Both figures are reported.
    """
    for name in ("NaCl", "KCl", "LiCl"):
        sp = CHLORIDES[name]
        t_b = float(sp.transition_T.quantity.to("degC").magnitude)
        p, capped = vapour_pressure(sp, Q_(1200.0, "degC"))
        frac = float(p.to("Pa").magnitude) / 101325.0
        print(f"\nBENCHMARK {name} at 1200 degC: 1 atm transition point {t_b:.0f} degC "
              f"(above the roast), computed p_sat {float(p.to('Pa').magnitude):.0f} Pa "
              f"= {frac:.3f} atm, capped={capped}. dHvap from Trouton "
              f"(estimate, tag {TROUTON_ENTROPY.tag.value}), so this is an "
              f"order-of-magnitude figure, not a measurement.")
        assert t_b > 1200.0
        assert 0.0 < frac < 1.0


# ---------------------------------------------------------------------------
# the reductant rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("element", ["Ti", "Al", "B"])
@pytest.mark.parametrize("temperature_C", [800.0, 1000.0, 1200.0, 1400.0])
def test_reductant_required_elements_blocked_without_carbon(
    element: str, temperature_C: float
) -> None:
    """Ti, Al and B must be blocked at EVERY temperature without a reductant.

    Liu et al. (2026, doi 10.3390/min16080836): carbonaceous reductants are
    indispensable for these elements. Temperature cannot substitute, so the gate
    must be temperature independent.
    """
    r = removal_screen(element, Q_(temperature_C, "degC"), reductant_present=False)
    assert r.thermodynamically_blocked is True
    assert r.blocked_reason is not None
    assert "10.3390/min16080836" in r.blocked_reason
    assert "NOT REMOVED" in r.verdict
    assert requires_reductant(element) is True


@pytest.mark.parametrize("element", ["Ti", "Al", "B"])
def test_reductant_unblocks_the_thermodynamic_gate(element: str) -> None:
    r = removal_screen(element, Q_(1200.0, "degC"), reductant_present=True)
    assert r.thermodynamically_blocked is False
    assert r.blocked_reason is None


@pytest.mark.parametrize("element", ["Na", "K", "Li", "Fe"])
def test_alkalis_and_iron_need_no_reductant(element: str) -> None:
    """Alkalis and Fe are removable under HCl alone, per the same source."""
    assert requires_reductant(element) is False
    r = removal_screen(element, Q_(1200.0, "degC"), reductant_present=False)
    assert r.thermodynamically_blocked is False


def test_reductant_rule_is_sourced() -> None:
    assert REDUCTANT_REQUIRED.tag is Tag.SOURCED
    assert REDUCTANT_REQUIRED.source is not None
    assert REDUCTANT_REQUIRED.source.doi == "10.3390/min16080836"
    assert "indispensable" in REDUCTANT_REQUIRED.basis


def test_blocked_verdict_says_volatility_is_irrelevant() -> None:
    """A blocked species must not be reported as removable just because it is volatile."""
    r = removal_screen("Ti", Q_(1200.0, "degC"), reductant_present=False)
    assert r.volatile is True          # TiCl4 would boil far below the roast
    assert r.thermodynamically_blocked is True
    assert "volatility is irrelevant" in r.verdict


# ---------------------------------------------------------------------------
# the three conditions stay separate
# ---------------------------------------------------------------------------

def test_transport_is_never_assessed_here() -> None:
    """The screen must always declare that it did not test transport."""
    for el in ("Ti", "Al", "Na", "Fe"):
        r = removal_screen(el, Q_(1200.0, "degC"), reductant_present=True)
        assert r.transport_assessed_here is False


def test_permissive_verdict_defers_to_diffusion() -> None:
    """Even a full pass must point at the transport question, not claim removal."""
    r = removal_screen("Ti", Q_(1200.0, "degC"), reductant_present=True)
    assert "NOT a removal prediction" in r.verdict
    assert "ae.physics.diffusion" in r.verdict


def test_subboiling_species_reported_as_partial() -> None:
    r = removal_screen("Na", Q_(1200.0, "degC"), reductant_present=True)
    assert r.volatile is False
    assert "PARTIAL AT BEST" in r.verdict
    assert r.T_over_Tb < 1.0


# ---------------------------------------------------------------------------
# missing-data discipline
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("species", ["Al2O3(s)", "TiO2(s)", "B2O3(s)", "AlCl3(s)", "TiCl4(l)"])
def test_gibbs_raises_for_unsourced_species(species: str) -> None:
    """The oxide-to-chloride screen must refuse to run on missing data.

    NIST-JANAF and the NIST WebBook were unreachable from this sandbox, so
    these formation enthalpies are absent. A dG computed from a partial table
    would look authoritative and be arbitrary, so the call raises and names the
    source that would supply the value.
    """
    with pytest.raises(KeyError, match="no thermochemical data|would be supplied|NIST"):
        gibbs_of_reaction({species: -1.0, "CO2(g)": 1.0}, Q_(1200.0, "degC"))
    assert species in THERMO_MISSING
    assert "NIST" in THERMO_MISSING[species]


def test_gibbs_raises_when_entropy_absent() -> None:
    """Hf alone is not enough: the T dS term needs S, which PubChem rarely gives."""
    assert "Hf" in THERMO["FeCl3(s)"]
    assert "S" not in THERMO["FeCl3(s)"]
    with pytest.raises(KeyError, match="has no 'S'"):
        gibbs_of_reaction({"FeCl3(s)": 1.0, "CO2(g)": -1.0}, Q_(1200.0, "degC"))


def test_rejected_pubchem_alumina_value_is_documented() -> None:
    """The inconsistent PubChem Al2O3 value must be recorded as rejected, not used."""
    assert "Al2O3(s)" not in THERMO
    note = THERMO_MISSING["Al2O3(s)"]
    assert "-130.0" in note and "NOT used" in note
    assert "1676" in note


def test_every_thermo_entry_is_sourced() -> None:
    for species, props in THERMO.items():
        for prop, value in props.items():
            assert value.tag is Tag.SOURCED, f"{species}.{prop}"
            assert value.source is not None, f"{species}.{prop}"
            assert value.source.url or value.source.doi, f"{species}.{prop}"


def test_every_chloride_transition_point_is_sourced() -> None:
    for name, sp in CHLORIDES.items():
        assert sp.transition_T.tag is Tag.SOURCED, name
        assert sp.transition_T.source is not None, name
        assert sp.transition_T.source.url == SOURCE_PUBCHEM.url, name
        assert sp.pubchem_cid is not None, name


def test_only_ticl4_has_a_sourced_vaporization_enthalpy() -> None:
    """Honesty check on which dHvap values are real and which are Trouton."""
    sourced = {n for n, sp in CHLORIDES.items() if not sp.enthalpy_is_trouton}
    assert sourced == {"TiCl4"}
    for name, sp in CHLORIDES.items():
        if name != "TiCl4":
            assert sp.enthalpy_is_trouton is True, name


def test_screen_rejects_untabulated_element() -> None:
    with pytest.raises(KeyError, match="no chloride species tabulated"):
        removal_screen("Zr", Q_(1200.0, "degC"), reductant_present=True)


def test_screen_feedstock_covers_assayed_elements_only(quartz: Feedstock) -> None:
    results = screen_feedstock(quartz, Q_(1200.0, "degC"), reductant_present=True)
    assert set(results) == {"Al", "Ti", "Li", "Fe", "Na", "K", "B"}
    assert all(r.transport_assessed_here is False for r in results.values())


# ---------------------------------------------------------------------------
# physical sanity
# ---------------------------------------------------------------------------

def test_vapour_pressure_is_capped_far_above_boiling() -> None:
    """Extrapolating 1000 degC above Tb must be capped and flagged, not reported."""
    p, capped = vapour_pressure(CHLORIDES["TiCl4"], Q_(1200.0, "degC"))
    assert capped is True
    assert float(p.to("Pa").magnitude) == pytest.approx(101325.0, rel=1e-12)


def test_vapour_pressure_rises_with_temperature() -> None:
    """Second-law direction: a positive dHvap means more vapour when hotter."""
    lo, _ = vapour_pressure(CHLORIDES["NaCl"], Q_(1000.0, "degC"))
    hi, _ = vapour_pressure(CHLORIDES["NaCl"], Q_(1300.0, "degC"))
    assert float(hi.to("Pa").magnitude) > float(lo.to("Pa").magnitude)


def test_vapour_pressure_never_negative() -> None:
    for name, sp in CHLORIDES.items():
        for t_c in (25.0, 500.0, 1200.0):
            p, _ = vapour_pressure(sp, Q_(t_c, "degC"))
            assert float(p.to("Pa").magnitude) >= 0.0, (name, t_c)


def test_flux_is_zero_when_bulk_pressure_exceeds_saturation() -> None:
    """No evaporation against a saturated atmosphere: the driving force is gone."""
    j = hertz_knudsen_flux(
        CHLORIDES["NaCl"], Q_(1000.0, "degC"),
        partial_pressure_bulk=Q_(1.0, "atm"))
    assert float(j.magnitude) == 0.0


def test_flux_scales_with_evaporation_coefficient() -> None:
    full = hertz_knudsen_flux(CHLORIDES["NaCl"], Q_(1200.0, "degC"))
    tenth = hertz_knudsen_flux(
        CHLORIDES["NaCl"], Q_(1200.0, "degC"), evaporation_coefficient=0.1)
    assert float(tenth.magnitude) == pytest.approx(0.1 * float(full.magnitude), rel=1e-12)


def test_species_rejects_negative_enthalpy() -> None:
    with pytest.raises(ValidationError, match="positive|second law"):
        ChlorideSpecies(
            formula="XCl", cation="X",
            molar_mass=_v(Q_(100.0, "g/mol"), Tag.DERIVED, basis="test"),
            transition_T=_v(Q_(300.0, "degC")),
            transition_kind=PhaseChange.BOILING,
            enthalpy_vaporization=_v(Q_(-10.0, "kJ/mol"), Tag.ASSUMED))


def test_species_rejects_nonpositive_transition_temperature() -> None:
    with pytest.raises(ValidationError, match="positive"):
        ChlorideSpecies(
            formula="XCl", cation="X",
            molar_mass=_v(Q_(100.0, "g/mol"), Tag.DERIVED, basis="test"),
            transition_T=_v(Q_(0.0, "K")),
            transition_kind=PhaseChange.BOILING)


def test_reference_pressure_is_one_atmosphere() -> None:
    assert float(P_REFERENCE.to("Pa").magnitude) == pytest.approx(101325.0, rel=1e-12)


def test_liu_source_note_records_the_403() -> None:
    """The provenance must record that only the abstract was retrievable."""
    assert SOURCE_LIU_2026.note is not None
    assert "403" in SOURCE_LIU_2026.note
    assert "abstract" in SOURCE_LIU_2026.note.lower()
