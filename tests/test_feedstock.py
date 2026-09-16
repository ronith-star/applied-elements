"""FEEDSTOCK schema and oxide conversion tests."""
import datetime as dt
import pytest
from ae.core.units import Q_
from ae.core.provenance import Tag, Tier, Source, Value, MISSING, MissingValueError
from ae.core.feedstock import (OreType, Feedstock, ImpurityProfile, oxide_to_element,
                               OXIDE_STOICH, MOLAR_MASS, ELEMENTS)

SRC = Source(citation="Test source 2026", tier=Tier.T1, doi="10.1000/test",
             accessed=dt.date(2026, 9, 16))


def sourced(mag, unit="ppm_mass"):
    return Value(quantity=Q_(mag, unit), tag=Tag.SOURCED, source=SRC)


@pytest.mark.golden
def test_oxide_conversion_worked_example():
    """Hand-traceable: 0.22 wt% Al2O3 = 2200 ppm oxide.
    Al fraction = 2(26.9815) / (2(26.9815) + 3(15.999)) = 53.963/101.960 = 0.52926
    2200 x 0.52926 = 1164.4 ppm Al."""
    el, ppm, factor = oxide_to_element("Al2O3", 2200.0)
    assert el == "Al"
    assert factor == pytest.approx(0.52926, abs=1e-5)
    assert ppm == pytest.approx(1164.4, abs=0.1)


@pytest.mark.golden
def test_fe2o3_conversion_worked_example():
    """0.02 wt% Fe2O3 = 200 ppm oxide; Fe fraction = 111.69/159.687 = 0.69944."""
    el, ppm, factor = oxide_to_element("Fe2O3", 200.0)
    assert el == "Fe"
    assert factor == pytest.approx(0.69944, abs=1e-5)
    assert ppm == pytest.approx(139.9, abs=0.1)


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


def test_characterized_requires_full_suite_and_capable_method():
    els = {e: sourced(1.0) for e in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")}
    with pytest.raises(ValueError, match="single-grain capable"):
        Feedstock(sample_id="AE-Q-IN-VKB-002", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="V", country="IN", characterized=True,
                  impurities=ImpurityProfile(total=els, method="XRF"))
    with pytest.raises(ValueError, match="missing"):
        Feedstock(sample_id="AE-Q-IN-VKB-003", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="V", country="IN", characterized=True,
                  impurities=ImpurityProfile(total={"Al": sourced(1.0)}, method="LA_ICP_MS"))
    ok = Feedstock(sample_id="AE-Q-IN-VKB-004", ore_type=OreType.VEIN_QUARTZ,
                   deposit_name="V", country="IN", characterized=True,
                   impurities=ImpurityProfile(total=els, method="LA_ICP_MS"))
    assert ok.characterized


def test_uncharacterized_is_the_default():
    f = Feedstock(sample_id="AE-Q-IN-VKB-005", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="Vikarabad", country="IN")
    assert f.characterized is False
    assert f.sio2_percent is MISSING
