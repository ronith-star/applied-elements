"""FEEDSTOCK schema and oxide conversion tests."""
import datetime as dt

import pytest

from ae.core.feedstock import (
    MOLAR_MASS,
    OXIDE_STOICH,
    Feedstock,
    ImpurityProfile,
    OreType,
    oxide_to_element,
)
from ae.core.provenance import MISSING, MissingValueError, Source, Tag, Tier, Value
from ae.core.units import Q_

SRC = Source(citation="Test source 2026", tier=Tier.T1, doi="10.1000/test",
             accessed=dt.date(2026, 9, 16))


def sourced(mag, unit="ppm_mass"):
    return Value(quantity=Q_(mag, unit), tag=Tag.SOURCED, source=SRC)


@pytest.mark.golden
def test_oxide_conversion_worked_example():
    """Hand-traceable: 0.22 wt% Al2O3 = 2200 ppm oxide.
    Al fraction = 2(26.9815) / (2(26.9815) + 3(15.999)) = 53.963/101.960 = 0.52926
    2200 x 0.52926 = 1164.4 ppm Al."""
    m_al, m_o = MOLAR_MASS["Al"], MOLAR_MASS["O"]
    assert m_al == pytest.approx(26.9815, abs=1e-4)
    assert m_o == pytest.approx(15.999, abs=1e-3)
    assert 2 * m_al == pytest.approx(53.963, abs=1e-3)
    assert 2 * m_al + 3 * m_o == pytest.approx(101.960, abs=1e-3)
    assert 53.963 / 101.960 == pytest.approx(0.52926, abs=1e-5)
    # 0.22 wt% as ppm.
    assert 0.22 * 1e4 == 2200.0
    el, ppm, factor = oxide_to_element("Al2O3", 2200.0)
    assert el == "Al"
    assert factor == pytest.approx(0.52926, abs=1e-5)
    assert ppm == pytest.approx(1164.4, abs=0.1)


@pytest.mark.golden
def test_fe2o3_conversion_worked_example():
    """Hand-traceable: 0.02 wt% Fe2O3 = 200 ppm oxide, Fe = 139.886 ppm.

    Every step asserted, not just the endpoint:
      2 x M(Fe) = 2 x 55.845          = 111.690 g/mol
      M(Fe2O3) = 111.690 + 3 x 15.999 = 159.687 g/mol
      mass fraction Fe = 111.690/159.687 = 0.6994308
      200 ppm oxide x 0.6994308          = 139.886 ppm Fe
    """
    assert MOLAR_MASS["Fe"] == pytest.approx(55.845, abs=1e-3)
    assert MOLAR_MASS["O"] == pytest.approx(15.999, abs=1e-3)
    assert 0.02 * 1e4 == 200.0, "0.02 wt% = 200 ppm"
    # Molar masses, from the module's own table rather than retyped.
    m_fe, m_o = MOLAR_MASS["Fe"], MOLAR_MASS["O"]
    assert 2 * m_fe == pytest.approx(111.690, abs=1e-3)
    m_oxide = 2 * m_fe + 3 * m_o
    assert m_oxide == pytest.approx(159.687, abs=1e-3)
    assert 111.690 / 159.687 == pytest.approx(0.6994308, abs=1e-7)

    el, ppm, factor = oxide_to_element("Fe2O3", 200.0)
    assert el == "Fe"
    assert factor == pytest.approx(2 * m_fe / m_oxide, rel=1e-12)
    assert factor == pytest.approx(0.69944, abs=1e-5)
    assert ppm == pytest.approx(200.0 * factor, rel=1e-12)
    assert ppm == pytest.approx(139.886, abs=1e-3)


@pytest.mark.parametrize("oxide", sorted(OXIDE_STOICH))
def test_all_oxide_factors_are_physical(oxide):
    el, ppm, f = oxide_to_element(oxide, 1000.0)
    assert 0.0 < f < 1.0, f"{oxide} factor must be a mass fraction below unity"
    assert ppm == pytest.approx(1000.0 * f)
    assert el in MOLAR_MASS


def test_unknown_oxide_raises():
    with pytest.raises(KeyError, match="unknown oxide"):
        oxide_to_element("XO2", 100.0)


def test_sample_id_pattern_enforced():
    ok = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
                   deposit_name="Vikarabad", country="IN")
    assert ok.id_parts == {"mineral": "Q", "country": "IN", "deposit": "VKB", "number": "001"}
    for bad in ["AEQ-IN-VKB-001", "AE-X-IN-VKB-001", "AE-Q-IND-VKB-001", "AE-Q-IN-VKB-1"]:
        with pytest.raises(Exception):
            Feedstock(sample_id=bad, ore_type=OreType.VEIN_QUARTZ,
                      deposit_name="x", country="IN")


def test_unknown_element_rejected():
    with pytest.raises(ValueError, match="unknown element"):
        ImpurityProfile(total={"Xx": sourced(1.0)})


def test_unmeasured_element_raises_with_guidance():
    p = ImpurityProfile(total={"Al": sourced(1164.0), "Fe": sourced(140.0)}, method="XRF")
    assert p.total_ppm("Al") == pytest.approx(1164.0)
    with pytest.raises(KeyError, match="four-oxide royalty assay"):
        p.total_ppm("Ti")


def test_lattice_split_raises_when_unmeasured():
    """The purity ceiling must never be guessed."""
    p = ImpurityProfile(total={"Al": sourced(30.0)}, method="LA_ICP_MS")
    with pytest.raises(MissingValueError, match="MISSING"):
        p.lattice_ppm("Al")


def test_lattice_ppm_when_measured():
    p = ImpurityProfile(
        total={"Al": sourced(30.0)},
        lattice_fraction={"Al": Value(quantity=Q_(0.6, "dimensionless"), tag=Tag.SOURCED,
                                      source=SRC)},
        method="LA_ICP_MS")
    assert p.lattice_ppm("Al") == pytest.approx(18.0)


def test_lattice_fraction_range_validated():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ImpurityProfile(total={"Al": sourced(30.0)},
                        lattice_fraction={"Al": Value(quantity=Q_(1.4, "dimensionless"),
                                                      tag=Tag.SOURCED, source=SRC)})


def test_mole_basis_impurity_rejected():
    """A mole-ratio assay must not enter a mass-ratio field."""
    with pytest.raises(ValueError, match="mole-basis"):
        ImpurityProfile(total={"Al": Value(quantity=Q_(30.0, "umol/mol"), tag=Tag.SOURCED,
                                           source=SRC)})


def test_sum_excludes_OH():
    p = ImpurityProfile(total={"Al": sourced(20.0), "Ti": sourced(5.0), "OH": sourced(100.0)})
    assert p.sum_ppm() == pytest.approx(25.0)


def test_characterization_tiers_replace_the_binary_flag():
    """Four tiers, because a binary flag blocked all early screening.

    The tiers are unmeasured, screened, bulk_quantified and located, matching
    CharacterizationTier. An earlier version of this docstring said "three",
    omitting unmeasured, while the assertions below already covered all four.

    The previous gate accepted characterized=True on any full element suite
    measured by LA-ICP-MS or GDMS, and rejected XRF with the rationale that
    "bulk XRF cannot resolve lattice impurities". That rationale was WRONG: XRF
    measures total element content wherever the atoms sit. Its real limits are
    that it cannot measure Li or B at all and has inadequate ppm detection for
    Ti and the alkalis; and no bulk method, XRF or ICP-MS, carries the spatial
    information a purification ceiling needs.
    """
    els = {e: sourced(1.0) for e in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")}

    def fs(**kw):
        base = dict(sample_id="AE-Q-IN-VKB-002", ore_type=OreType.VEIN_QUARTZ,
                    deposit_name="V", country="IN")
        base.update(kw)
        return Feedstock(**base)

    # No measurement at all.
    assert fs().characterization_tier == "unmeasured"

    # XRF on the full suite is SCREENED, not rejected outright.
    xrf = fs(impurities=ImpurityProfile(total=els, method="XRF"))
    assert xrf.characterization_tier == "screened"
    assert xrf.permits_output("screening_rank")
    assert xrf.permits_output("mass_yield_estimate")
    assert not xrf.permits_output("reagent_demand")
    assert not xrf.permits_output("purification_ceiling")

    # Full suite by a bulk method with adequate limits is BULK_QUANTIFIED.
    bulk = fs(impurities=ImpurityProfile(total=els, method="ICP_MS"))
    assert bulk.characterization_tier == "bulk_quantified"
    assert bulk.permits_output("reagent_demand")
    assert bulk.permits_output("impurity_removal_estimate")
    assert not bulk.permits_output("purification_ceiling"), (
        "bulk totals carry no spatial information, so no ceiling"
    )

    # LA-ICP-MS on the full suite but with the lattice split UNMEASURED is
    # still only bulk_quantified. The method alone is not the criterion.
    unlocated = fs(impurities=ImpurityProfile(total=els, method="LA_ICP_MS"))
    assert unlocated.characterization_tier == "bulk_quantified"

    # With the lattice fraction measured for every element, it is LOCATED.
    located = fs(impurities=ImpurityProfile(
        total=els, method="LA_ICP_MS",
        lattice_fraction={e: sourced(0.3, "dimensionless") for e in els}))
    assert located.characterization_tier == "located"
    assert located.permits_output("purification_ceiling")
    assert located.permits_output("product_grade_claim")


def test_characterized_flag_requires_the_located_tier():
    els = {e: sourced(1.0) for e in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")}
    with pytest.raises(ValueError, match="requires tier 'located'"):
        Feedstock(sample_id="AE-Q-IN-VKB-005", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="V", country="IN", characterized=True,
                  impurities=ImpurityProfile(total=els, method="XRF"))
    ok = Feedstock(sample_id="AE-Q-IN-VKB-006", ore_type=OreType.VEIN_QUARTZ,
                   deposit_name="V", country="IN", characterized=True,
                   impurities=ImpurityProfile(
                       total=els, method="LA_ICP_MS",
                       lattice_fraction={e: sourced(0.3, "dimensionless")
                                         for e in els}))
    assert ok.characterized and ok.characterization_tier == "located"


def test_characterization_gap_names_the_missing_measurement():
    """The gap must be actionable by a campaign planner: which elements, which
    method, not a score."""
    partial = Feedstock(sample_id="AE-Q-IN-VKB-007", ore_type=OreType.VEIN_QUARTZ,
                        deposit_name="V", country="IN",
                        impurities=ImpurityProfile(
                            total={"Al": sourced(1.0), "Fe": sourced(1.0)},
                            method="XRF"))
    gap = partial.characterization_gap()
    assert gap["tier"] == "screened"
    assert gap["next_tier"] == "bulk_quantified"
    assert set(gap["missing_elements"]) == {"Ti", "Li", "Na", "K", "B"}
    joined = " ".join(gap["to_advance"])
    assert "Li" in joined and "B" in joined
    assert "ICP-MS" in joined or "GDMS" in joined

    els = {e: sourced(1.0) for e in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")}
    bulk = Feedstock(sample_id="AE-Q-IN-VKB-008", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="V", country="IN",
                     impurities=ImpurityProfile(total=els, method="ICP_MS"))
    g2 = bulk.characterization_gap()
    assert g2["next_tier"] == "located"
    assert not g2["missing_elements"]
    assert len(g2["lattice_unmeasured"]) == 7
    assert "LA-ICP-MS" in " ".join(g2["to_advance"])


def test_require_output_raises_with_the_gap_in_the_message():
    els = {e: sourced(1.0) for e in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")}
    bulk = Feedstock(sample_id="AE-Q-IN-VKB-009", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="V", country="IN",
                     impurities=ImpurityProfile(total=els, method="ICP_MS"))
    bulk.require_output("reagent_demand")          # permitted, no raise
    with pytest.raises(ValueError, match="LA-ICP-MS"):
        bulk.require_output("purification_ceiling")
    with pytest.raises(ValueError, match="unknown output"):
        bulk.permits_output("free_lunch")


def test_uncharacterized_is_the_default():
    f = Feedstock(sample_id="AE-Q-IN-VKB-005", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="Vikarabad", country="IN")
    assert f.characterized is False
    assert f.sio2_percent is MISSING


def test_tier_docstrings_enumerate_every_tier_in_the_type():
    """A docstring that names fewer tiers than CharacterizationTier defines is
    how "three tiers" survived alongside a four-member Literal. Checked by
    reflection against the type, so the two cannot drift again."""
    import typing

    from ae.core.feedstock import CharacterizationTier
    tiers = set(typing.get_args(CharacterizationTier))
    assert tiers == {"unmeasured", "screened", "bulk_quantified", "located"}

    doc = Feedstock.characterization_tier.__doc__ or ""
    missing = sorted(t for t in tiers if f"``{t}``" not in doc)
    assert not missing, f"tiers defined but not documented: {missing}"

    # And the stated count must match.
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, n in words.items():
        if f"{word} tiers" in doc.lower():
            assert n == len(tiers), (
                f"docstring says '{word} tiers' but the type defines {len(tiers)}"
            )
