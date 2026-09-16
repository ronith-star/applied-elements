"""Tests for ae.physics.reagents: stoichiometry, mass balance, effluent and cost.

Mandatory closures, asserted here rather than hoped for:
  - every tabulated reaction balances for BOTH element counts and charge
  - the overall flowsheet mass balance closes to 1e-9 relative
  - the liquor charge balance closes to 1e-9 relative
  - no price, limit or availability figure is hardcoded: all come from SITE

The headline quantitative result is that HF consumption is dominated by attack
on the silica matrix rather than by the impurity load. For the reference
impurity fixture the ratio is 44.73 at 0.1 percent silica dissolution and 447.3
at 1 percent, on the tabulated 6 HF solution route (SiO2 + 6 HF -> H2SiF6).
The ratio is linear in the dissolved fraction because the impurity term does
not move with it.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from ae.core.feedstock import MOLAR_MASS, Feedstock, ImpurityProfile, OreType
from ae.core.provenance import MissingValueError, Source, Tag, Tier, Value
from ae.core.site import (
    Currency,
    LabourRates,
    PermittingRegime,
    PowerSupply,
    ReagentPrices,
    Site,
)
from ae.core.units import Q_, require_dimensionality
from ae.physics.reagents import (
    ACID_MOLAR_MASS,
    ACID_PROTONS,
    BASE_EQUIVALENTS,
    CATION_CHARGE,
    LEACH_REACTIONS,
    M_SIO2,
    SILICA_HF_STOICH,
    Acid,
    Base,
    acid_demand_per_cation,
    charge_balance,
    check_reaction_balance,
    fluoride_effluent,
    hf_silica_demand,
    impurity_acid_demand,
    impurity_moles,
    neutralization_demand,
    reagent_balance,
    reagent_cost,
)

_SRC = Source(
    citation="Test fixture source, not a real reference",
    tier=Tier.T2, url="https://example.invalid/fixture", accessed=dt.date(2026, 9, 16),
    note="Synthetic fixture for unit tests. Carries no real price or limit.",
)


def _v(q, tag=Tag.SOURCED, **kw):
    if tag in (Tag.SOURCED, Tag.MEASURED):
        return Value(quantity=q, tag=tag, source=_SRC, **kw)
    kw.setdefault("basis", "test fixture")
    return Value(quantity=q, tag=tag, **kw)


@pytest.fixture
def quartz() -> Feedstock:
    """Impurities at the Muller et al. HPQ reference limits, lattice split measured."""
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
        characterized=True,
    )


@pytest.fixture
def site() -> Site:
    """Synthetic site. Prices and limits are FIXTURES, not real market data."""
    return Site(
        site_id="IN-TG-VKB", name="Synthetic test site", country="IN",
        region="Telangana", currency=Currency.INR,
        power=PowerSupply(energy_price=_v(Q_(7.0, "INR/kWh")),
                          rate_basis="published_tariff"),
        labour=LabourRates(fully_loaded_operator=_v(Q_(300.0, "INR/hour"))),
        reagents=ReagentPrices(
            prices={"HF": _v(Q_(180.0, "INR/kg")), "HCl": _v(Q_(12.0, "INR/kg")),
                    "Ca(OH)2": _v(Q_(6.0, "INR/kg")), "NaOH": _v(Q_(45.0, "INR/kg"))},
            locally_available={"HF": False, "HCl": True, "Ca(OH)2": True, "NaOH": True}),
        permitting=PermittingRegime(
            jurisdiction="Synthetic fixture jurisdiction",
            effluent_limits={"F": _v(Q_(2.0, "mg/L"))}),
    )


@pytest.fixture
def site_no_fluoride_limit(site: Site) -> Site:
    return site.model_copy(update={
        "permitting": PermittingRegime(jurisdiction="Fixture with no fluoride limit")})


# ---------------------------------------------------------------------------
# reaction balance: mass AND charge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(LEACH_REACTIONS))
def test_every_reaction_balances(name: str) -> None:
    """Element counts and charge must both close exactly, or the nu factors are wrong."""
    resid = check_reaction_balance(LEACH_REACTIONS[name])
    assert all(abs(v) < 1e-12 for v in resid.values()), f"{name}: {resid}"
    assert "charge" in resid


@pytest.mark.parametrize("name,expected_nu", [
    ("hematite_HCl", 3.0), ("alumina_HCl", 3.0), ("lime_HCl", 2.0),
    ("potash_HCl", 1.0), ("silica_HF", 6.0),
])
def test_reaction_nu_matches_charge_rule(name: str, expected_nu: float) -> None:
    """The tabulated nu must equal the charge rule, or one of the two is wrong."""
    rxn = LEACH_REACTIONS[name]
    assert rxn.acid_per_cation == pytest.approx(expected_nu, abs=1e-12)
    if rxn.cation in CATION_CHARGE:
        assert acid_demand_per_cation(rxn.cation, rxn.acid) == pytest.approx(
            expected_nu, abs=1e-12)


def test_unbalanced_reaction_is_detected() -> None:
    """The checker must catch a deliberately broken reaction, or it proves nothing."""
    broken = LEACH_REACTIONS["hematite_HCl"].model_copy(update={
        "reactants": {"Fe2O3": (1, 0, {"Fe": 2, "O": 3}),
                      "HCl": (5, 0, {"H": 1, "Cl": 1})}})   # 5 HCl instead of 6
    resid = check_reaction_balance(broken)
    assert abs(resid["Cl"]) == pytest.approx(1.0, abs=1e-12)
    assert abs(resid["H"]) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# dimensional analysis
# ---------------------------------------------------------------------------

def test_impurity_moles_are_moles(quartz: Feedstock) -> None:
    moles = impurity_moles(quartz, Q_(1.0, "tonne_metric"))
    for el, n in moles.items():
        assert n.to("mol").dimensionality == Q_(1.0, "mol").dimensionality, el


def test_hf_silica_demand_units() -> None:
    n, m = hf_silica_demand(Q_(1.0, "tonne_metric"), 0.001)
    assert n.to("mol").dimensionality == Q_(1.0, "mol").dimensionality
    require_dimensionality(m, "mass", "m_SiO2")


def test_molar_mass_of_silica_is_correct() -> None:
    """M_SiO2 must equal Si + 2 O from the shared atomic weight table."""
    expected = (MOLAR_MASS["Si"] + 2 * MOLAR_MASS["O"]) / 1000.0
    assert M_SIO2 == pytest.approx(expected, rel=1e-12)
    assert M_SIO2 == pytest.approx(0.060083, rel=1e-5)


# ---------------------------------------------------------------------------
# golden: hand-traceable arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_impurity_moles_arithmetic(quartz: Feedstock) -> None:
    r"""Hand-check n_i = m C_i / M_i on the LEACHABLE inventory.

    Al: 30.0 ppm total with 60 percent lattice, so leachable = 12.0 ppm.
      n_Al = 1000 kg * 12.0e-6 / 0.0269815 kg/mol
           = 0.012 kg / 0.0269815 kg/mol = 0.4447492 mol

    Fe: 3.0 ppm total with 10 percent lattice, so leachable = 2.70 ppm.
      n_Fe = 1000 * 2.70e-6 / 0.055845 = 0.0027 / 0.055845 = 0.0483481 mol
    """
    moles = impurity_moles(quartz, Q_(1000.0, "kg"))
    assert float(moles["Al"].to("mol").magnitude) == pytest.approx(0.4447492, rel=1e-6)
    assert float(moles["Fe"].to("mol").magnitude) == pytest.approx(0.0483481, rel=1e-6)
    assert 0.012 / 0.0269815 == pytest.approx(0.4447492, rel=1e-6)


@pytest.mark.golden
def test_golden_hf_silica_demand_arithmetic() -> None:
    r"""Hand-check n_HF = 6 m_diss / M_SiO2 at 0.1 percent silica dissolution.

    m_diss = 0.001 * 1000 kg = 1.0 kg
    n_SiO2 = 1.0 / 0.060083 = 16.643643 mol
    n_HF   = 6 * 16.643643 = 99.861860 mol per tonne of charge

    The 4 HF route (SiF4 gas) gives 4 * 16.643643 = 66.574573 mol, which is
    exactly two thirds of the solution route, a 50 percent difference in HF
    demand from the product speciation alone.
    """
    n_sio2 = 1.0 / 0.060083
    assert n_sio2 == pytest.approx(16.643643, rel=1e-6)
    n_hf, m_diss = hf_silica_demand(Q_(1000.0, "kg"), 0.001)
    assert float(m_diss.to("kg").magnitude) == pytest.approx(1.0, rel=1e-12)
    assert float(n_hf.to("mol").magnitude) == pytest.approx(6.0 * n_sio2, rel=1e-6)
    assert float(n_hf.to("mol").magnitude) == pytest.approx(99.861860, rel=1e-6)
    n_hf4, _ = hf_silica_demand(Q_(1000.0, "kg"), 0.001, route="SiF4_gas")
    assert float(n_hf4.to("mol").magnitude) == pytest.approx(66.574573, rel=1e-6)
    assert float(n_hf4.magnitude) / float(n_hf.magnitude) == pytest.approx(2.0 / 3.0, rel=1e-12)


@pytest.mark.golden
def test_golden_silica_dominates_hf_demand(quartz: Feedstock, site: Site) -> None:
    r"""The headline result: the matrix consumes far more HF than the impurities.

    Impurity HF demand for the fixture, per tonne, on the leachable inventory:
      Al (30 ppm, 60 pct lattice -> 12.0 ppm) 0.444749 mol * 3 = 1.334248
      Ti (10 ppm, 90 pct lattice -> 1.0 ppm)  0.020891 mol * 4 = 0.083565
      Li (5 ppm, 80 pct -> 1.0 ppm)           0.144092 mol * 1 = 0.144092
      Fe (3 ppm, 10 pct -> 2.7 ppm)           0.048348 mol * 3 = 0.145044
      Na (8 ppm, 30 pct -> 5.6 ppm)           0.243587 mol * 1 = 0.243587
      K  (8 ppm, 30 pct -> 5.6 ppm)           0.143229 mol * 1 = 0.143229
      B  (1 ppm, 50 pct -> 0.5 ppm)           0.046253 mol * 3 = 0.138760
      total = 2.232525 mol HF per tonne

    Silica demand at 0.1 percent dissolution = 99.861860 mol per tonne, so the
    ratio is 99.861860 / 2.232525 = 44.73, i.e. the matrix consumes 45 times
    more HF than every impurity in the ore combined. At 1 percent dissolution
    the ratio is 447.3, since the silica term is linear in the dissolved
    fraction while the impurity term does not move.
    """
    total, per_el = impurity_acid_demand(quartz, Q_(1000.0, "kg"), Acid.HF)
    assert float(per_el["Al"].to("mol").magnitude) == pytest.approx(1.334248, rel=1e-6)
    assert float(total.to("mol").magnitude) == pytest.approx(2.232525, rel=1e-6)
    b = reagent_balance(quartz, site, Q_(1000.0, "kg"), Acid.HF,
                        silica_dissolved_fraction=0.001)
    assert b["silica_acid_demand_mol"] == pytest.approx(99.861860, rel=1e-6)
    assert b["silica_to_impurity_demand_ratio"] == pytest.approx(44.7305, rel=1e-5)
    b10 = reagent_balance(quartz, site, Q_(1000.0, "kg"), Acid.HF,
                          silica_dissolved_fraction=0.01)
    assert b10["silica_to_impurity_demand_ratio"] == pytest.approx(447.305, rel=1e-5)


@pytest.mark.golden
def test_golden_neutralization_arithmetic() -> None:
    r"""Hand-check lime demand and CaF2 sludge for 100 mol of residual HF.

    HF is monoprotic, so 100 mol HF is 100 proton equivalents.
    Ca(OH)2 carries 2 hydroxide equivalents, so
      n_lime = 100 / 2 = 50.0 mol
      m_lime = 50.0 * 74.092 g/mol = 3704.6 g = 3.7046 kg

    Fluoride precipitation, Ca(OH)2 + 2 HF -> CaF2 + 2 H2O:
      n_CaF2 = 100 / 2 = 50.0 mol
      M_CaF2 = 40.078 + 2(18.998) = 78.074 g/mol
      m_CaF2 = 50.0 * 78.074 = 3903.7 g = 3.9037 kg dry sludge
    """
    n = neutralization_demand({Acid.HF: Q_(100.0, "mol")}, Base.LIME)
    assert n["proton_equivalents_mol"] == pytest.approx(100.0, rel=1e-12)
    assert n["base_moles_mol"] == pytest.approx(50.0, rel=1e-12)
    assert BASE_EQUIVALENTS[Base.LIME][1] == pytest.approx(74.092, rel=1e-5)
    assert n["base_mass_kg"] == pytest.approx(3.7046, rel=1e-4)
    assert n["CaF2_moles_mol"] == pytest.approx(50.0, rel=1e-12)
    assert n["CaF2_sludge_dry_kg"] == pytest.approx(3.9037, rel=1e-4)
    assert n["fluoride_removed_by_base"] is True


@pytest.mark.golden
def test_golden_diprotic_acid_doubles_neutralization() -> None:
    r"""H2SO4 carries 2 protons, so 100 mol needs 200 equivalents, i.e. 100 mol lime.

      proton equivalents = 100 mol * 2 = 200
      n_lime = 200 / 2 = 100.0 mol
    Contrast HCl: 100 mol * 1 = 100 equivalents, n_lime = 50.0 mol.
    """
    n_sulph = neutralization_demand({Acid.H2SO4: Q_(100.0, "mol")}, Base.LIME)
    n_hcl = neutralization_demand({Acid.HCl: Q_(100.0, "mol")}, Base.LIME)
    assert n_sulph["proton_equivalents_mol"] == pytest.approx(200.0, rel=1e-12)
    assert n_sulph["base_moles_mol"] == pytest.approx(100.0, rel=1e-12)
    assert n_hcl["base_moles_mol"] == pytest.approx(50.0, rel=1e-12)
    assert ACID_PROTONS[Acid.H2SO4] == 2


@pytest.mark.golden
def test_golden_fluoride_concentration_arithmetic(site: Site) -> None:
    r"""Hand-check the discharge concentration for 50 mol F, 20 mol precipitated.

    Discharged = 50 - 20 = 30 mol F
    Mass = 30 * 18.998 g/mol = 569.94 g = 569940 mg
    Volume = 10 m3 = 10000 L
    Concentration = 569940 / 10000 = 56.994 mg/L

    Against the fixture limit of 2.0 mg/L this is 28.5 times over, so
    compliant must be False. The limit is a FIXTURE, not a real jurisdiction.
    """
    r = fluoride_effluent(site, Q_(50.0, "mol"), Q_(10.0, "m**3"), Q_(20.0, "mol"))
    assert r["fluoride_discharged_mol"] == pytest.approx(30.0, rel=1e-12)
    assert 30.0 * 18.998 == pytest.approx(569.94, rel=1e-6)
    assert r["concentration_mg_per_L"] == pytest.approx(56.994, rel=1e-5)
    assert r["ratio_to_limit"] == pytest.approx(28.497, rel=1e-4)
    assert r["compliant"] is False


@pytest.mark.golden
def test_golden_acid_molar_masses() -> None:
    """Acid molar masses from IUPAC atomic weights, checkable by hand.

    HF    = 1.008 + 18.998 = 20.006
    HCl   = 1.008 + 35.45 = 36.458
    HNO3  = 1.008 + 14.007 + 3(15.999) = 1.008 + 14.007 + 47.997 = 63.012
    H2SO4 = 2(1.008) + 32.06 + 4(15.999) = 2.016 + 32.06 + 63.996 = 98.072
    """
    assert ACID_MOLAR_MASS[Acid.HF] == pytest.approx(20.006, abs=1e-9)
    assert ACID_MOLAR_MASS[Acid.HCl] == pytest.approx(36.458, abs=1e-9)
    assert ACID_MOLAR_MASS[Acid.HNO3] == pytest.approx(63.012, abs=1e-9)
    assert ACID_MOLAR_MASS[Acid.H2SO4] == pytest.approx(98.072, abs=1e-9)


# ---------------------------------------------------------------------------
# benchmark: literature comparison with the error REPORTED
# ---------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_impurity_load_against_xia_2024(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    """Acid demand for the Xia et al. 2024 feed impurity load.

    Xia et al. (2024, doi 10.3390/min14070727) report a feed at 128.86 ug/g
    total trace impurities. The paper reports NO reagent consumption, so this
    is not a validation of any reagent figure. What is checked is the mole
    accounting: distributing 128.86 ppm as Al at the trivalent charge rule gives
    a definite HCl demand per tonne, and the reported figure is that demand
    together with the statement that no published consumption exists to compare
    it against.

    n_Al   = 1000 kg * 128.86e-6 / 0.0269815 = 4.775891 mol per tonne
    n_HCl  = 3 * 4.775891 = 14.327673 mol = 0.52239 kg HCl per tonne
    """
    n_al = 1000.0 * 128.86e-6 / 0.0269815
    n_hcl = 3.0 * n_al
    kg_hcl = n_hcl * 36.458 / 1000.0
    print(f"\nBENCHMARK Xia 2024 feed impurity load 128.86 ug/g treated as Al: "
          f"{n_al:.6f} mol Al per tonne, HCl demand {n_hcl:.6f} mol = {kg_hcl:.5f} kg "
          f"per tonne at the trivalent charge rule. The paper reports NO reagent "
          f"consumption, so NO error against literature can be computed for this "
          f"quantity: the benchmark is the impurity mole accounting only "
          f"(source: doi 10.3390/min14070727).")
    assert n_al == pytest.approx(4.775891, rel=1e-5)
    assert n_hcl == pytest.approx(14.327673, rel=1e-5)
    assert kg_hcl == pytest.approx(0.52239, rel=1e-4)


@pytest.mark.benchmark
def test_benchmark_yang_2020_iron_acid_demand(capsys: pytest.CaptureFixture[str]) -> None:
    """Acid demand for the iron removed in Yang and Li 2020.

    Reported: Fe2O3 from 0.0857 to 0.0223 percent, i.e. 634 ppm Fe2O3 removed.
    On an Fe basis with the factor 111.69/159.687 = 0.699435:
      Fe removed = 634 * 0.699435 = 443.44 ppm
      n_Fe       = 1000 * 443.44e-6 / 0.055845 = 7.94046 mol per tonne
      n_HCl      = 3 * 7.94046 = 23.82137 mol = 0.86847 kg per tonne

    The paper reports no acid consumption figure, so again no literature error
    can be computed: what is reported is the stoichiometric floor implied by
    their own assay change, which is the quantity a flowsheet would budget.
    """
    factor = 111.69 / 159.687
    fe_removed_ppm = 634.0 * factor
    n_fe = 1000.0 * fe_removed_ppm * 1e-6 / 0.055845
    n_hcl = 3.0 * n_fe
    kg = n_hcl * 36.458 / 1000.0
    print(f"\nBENCHMARK Yang and Li 2020 iron removal: 634 ppm Fe2O3 removed = "
          f"{fe_removed_ppm:.2f} ppm Fe, implying {n_fe:.5f} mol Fe and {n_hcl:.5f} mol "
          f"HCl = {kg:.5f} kg per tonne as the stoichiometric floor. The paper reports "
          f"no acid consumption, so no literature error is computable "
          f"(source: doi 10.1515/htmp-2020-0081).")
    assert fe_removed_ppm == pytest.approx(443.44, rel=1e-3)
    assert n_fe == pytest.approx(7.94046, rel=1e-4)
    assert kg == pytest.approx(0.86847, rel=1e-4)


# ---------------------------------------------------------------------------
# mass balance and charge balance closure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("acid,silica_frac", [
    (Acid.HCl, 0.0), (Acid.HNO3, 0.0), (Acid.H2SO4, 0.0),
    (Acid.HF, 0.0), (Acid.HF, 0.001), (Acid.HF, 0.01),
])
def test_mass_balance_closes(
    quartz: Feedstock, site: Site, acid: Acid, silica_frac: float
) -> None:
    """Mass in must equal mass out to 1e-9 relative, for every acid."""
    b = reagent_balance(quartz, site, Q_(1000.0, "kg"), acid,
                        silica_dissolved_fraction=silica_frac,
                        excess_factor=1.5, water_mass=Q_(3000.0, "kg"))
    assert b["mass_balance_relative_residual"] < 1e-9
    assert b["mass_in_kg"] == pytest.approx(b["mass_out_kg"], rel=1e-9)
    assert b["product_mass_kg"] > 0.0
    assert b["liquor_mass_kg"] > 0.0


def test_charge_balance_closes() -> None:
    r"""Electroneutrality for a synthetic liquor, exact by construction.

    1 mol Fe3+ (3 eq) + 1 mol Al3+ (3 eq) + 2 mol Na+ (2 eq) = 8 cation eq
    8 mol Cl- = 8 anion eq. Residual must be identically zero.
    """
    cb = charge_balance(
        {"Fe": Q_(1.0, "mol"), "Al": Q_(1.0, "mol"), "Na": Q_(2.0, "mol")},
        {"Cl": Q_(8.0, "mol")})
    assert cb["cation_equivalents"] == pytest.approx(8.0, abs=1e-12)
    assert cb["anion_equivalents"] == pytest.approx(8.0, abs=1e-12)
    assert abs(cb["relative_residual"]) < 1e-9


def test_charge_balance_with_divalent_anion_and_protons() -> None:
    r"""SO4(2-) counts twice, and free protons count as cations.

    2 mol Al3+ (6 eq) + 1 mol H+ (1 eq) = 7 cation eq
    3 mol SO4(2-) (6 eq) + 1 mol OH- (1 eq) = 7 anion eq
    """
    cb = charge_balance(
        {"Al": Q_(2.0, "mol")}, {"SO4": Q_(3.0, "mol")},
        h_plus=Q_(1.0, "mol"), oh_minus=Q_(1.0, "mol"))
    assert cb["cation_equivalents"] == pytest.approx(7.0, abs=1e-12)
    assert cb["anion_equivalents"] == pytest.approx(7.0, abs=1e-12)
    assert abs(cb["relative_residual"]) < 1e-12


def test_charge_imbalance_is_detected() -> None:
    """An omitted species must show up as a residual, or the check is useless."""
    cb = charge_balance({"Fe": Q_(1.0, "mol")}, {"Cl": Q_(2.0, "mol")})
    assert cb["residual"] == pytest.approx(1.0, abs=1e-12)
    assert abs(cb["relative_residual"]) > 0.3


def test_charge_balance_rejects_unknown_species() -> None:
    with pytest.raises(KeyError, match="no charge for cation"):
        charge_balance({"Xx": Q_(1.0, "mol")}, {"Cl": Q_(1.0, "mol")})
    with pytest.raises(KeyError, match="no charge for anion"):
        charge_balance({"Fe": Q_(1.0, "mol")}, {"Yy": Q_(1.0, "mol")})


# ---------------------------------------------------------------------------
# missing-data and site discipline
# ---------------------------------------------------------------------------

def test_reagent_budget_requires_measured_lattice_split(
    quartz_no_lattice_split: Feedstock, site: Site
) -> None:
    """An uncharacterized ore must not produce a reagent budget.

    Without the lattice split, the leachable inventory is unknown, and assuming
    the whole inventory is leachable overstates acid demand while simultaneously
    overstating the achievable grade.
    """
    with pytest.raises(MissingValueError):
        reagent_balance(quartz_no_lattice_split, site, Q_(1000.0, "kg"), Acid.HCl)


def test_hf_price_blocked_when_not_locally_available(site: Site) -> None:
    """HF availability is a real constraint and must not be assumed away."""
    with pytest.raises(ValueError, match="not locally available"):
        reagent_cost(site, Acid.HF, Q_(100.0, "kg"))


def test_unpriced_reagent_raises(site: Site) -> None:
    stripped = site.model_copy(update={
        "reagents": ReagentPrices(prices={"HCl": _v(Q_(12.0, "INR/kg"))},
                                  locally_available={"HCl": True})})
    with pytest.raises(KeyError, match="no delivered price"):
        reagent_cost(stripped, Acid.HF, Q_(100.0, "kg"))


def test_reagent_cost_uses_site_prices(site: Site) -> None:
    r"""Cost must come from the SITE: 100 kg HCl at 12 INR/kg = 1200 INR,
    plus 50 kg lime at 6 INR/kg = 300 INR, total 1500 INR."""
    c = reagent_cost(site, Acid.HCl, Q_(100.0, "kg"), Base.LIME, Q_(50.0, "kg"))
    assert c["acid_cost"] == pytest.approx(1200.0, rel=1e-12)
    assert c["base_cost"] == pytest.approx(300.0, rel=1e-12)
    assert c["total_cost"] == pytest.approx(1500.0, rel=1e-12)
    assert c["currency"] == "INR"


def test_fluoride_screen_requires_a_limit(site_no_fluoride_limit: Site) -> None:
    """No limit means no compliance answer: the limit belongs to the jurisdiction."""
    with pytest.raises(KeyError, match="no fluoride effluent limit"):
        fluoride_effluent(site_no_fluoride_limit, Q_(10.0, "mol"), Q_(1.0, "m**3"))


def test_fluoride_limit_provenance_is_reported(site: Site) -> None:
    r = fluoride_effluent(site, Q_(1.0, "mol"), Q_(100.0, "m**3"))
    assert "fixture" in str(r["limit_source"]).lower()


# ---------------------------------------------------------------------------
# physical sanity
# ---------------------------------------------------------------------------

def test_caustic_does_not_remove_fluoride() -> None:
    """NaF is soluble: the caustic route must report zero fluoride precipitated."""
    n = neutralization_demand({Acid.HF: Q_(100.0, "mol")}, Base.CAUSTIC)
    assert n["fluoride_removed_by_base"] is False
    assert n["CaF2_moles_mol"] == 0.0
    assert n["CaF2_sludge_dry_kg"] == 0.0
    assert "NaF is soluble" in n["note"]
    # caustic is monoprotic-equivalent, so it needs twice the moles of lime
    assert n["base_moles_mol"] == pytest.approx(100.0, rel=1e-12)


def test_excess_factor_below_one_rejected(quartz: Feedstock, site: Site) -> None:
    """Charging less acid than the reaction consumes is not a flowsheet."""
    with pytest.raises(ValueError, match="at least 1.0"):
        reagent_balance(quartz, site, Q_(1000.0, "kg"), Acid.HCl, excess_factor=0.9)


def test_silica_dissolution_rejected_for_non_hf(quartz: Feedstock, site: Site) -> None:
    """Only HF attacks silica appreciably: claiming otherwise must raise."""
    with pytest.raises(ValueError, match="only HF attacks"):
        reagent_balance(quartz, site, Q_(1000.0, "kg"), Acid.HCl,
                        silica_dissolved_fraction=0.01)


def test_unknown_silica_route_rejected() -> None:
    with pytest.raises(KeyError, match="unknown silica-HF route"):
        hf_silica_demand(Q_(1000.0, "kg"), 0.001, route="magic")
    assert set(SILICA_HF_STOICH) == {"H2SiF6_solution", "SiF4_gas"}


def test_negative_ore_mass_rejected(quartz: Feedstock) -> None:
    with pytest.raises(ValueError, match="positive"):
        impurity_moles(quartz, Q_(-1.0, "kg"))


def test_precipitated_fluoride_cannot_exceed_total(site: Site) -> None:
    with pytest.raises(ValueError, match="exceeds total fluoride"):
        fluoride_effluent(site, Q_(10.0, "mol"), Q_(1.0, "m**3"), Q_(20.0, "mol"))


def test_zero_effluent_volume_rejected(site: Site) -> None:
    with pytest.raises(ValueError, match="positive"):
        fluoride_effluent(site, Q_(10.0, "mol"), Q_(0.0, "m**3"))


def test_untabulated_cation_charge_raises() -> None:
    with pytest.raises(KeyError, match="no formal charge tabulated"):
        acid_demand_per_cation("Xx", Acid.HCl)


def test_all_impurity_moles_non_negative(quartz: Feedstock) -> None:
    for el, n in impurity_moles(quartz, Q_(1.0, "tonne_metric")).items():
        assert float(n.magnitude) >= 0.0, el


def test_excessive_silica_dissolution_gives_negative_product(
    quartz: Feedstock, site: Site
) -> None:
    """Dissolving more than the charge must raise, not report a negative product."""
    with pytest.raises(ValueError, match="fraction|must lie|between"):
        reagent_balance(quartz, site, Q_(1000.0, "kg"), Acid.HF,
                        silica_dissolved_fraction=1.5)
