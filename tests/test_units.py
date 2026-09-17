"""Dimensional-analysis harness. Every equation module must add cases here."""
import pytest

from ae.core.units import (
    Q_,
    DimensionalityError,
    as_dimensionless,
    ratio_basis,
    require_dimensionality,
    require_fraction,
    to_ppb_mass,
    to_ppm_mass,
)


def test_mass_ratio_aliases_agree():
    assert to_ppm_mass(Q_(30.0, "ppm_mass")).magnitude == pytest.approx(30.0)
    assert to_ppm_mass(Q_(30.0, "ug/g")).magnitude == pytest.approx(30.0)
    assert to_ppm_mass(Q_(30.0, "mg/kg")).magnitude == pytest.approx(30.0)
    assert to_ppb_mass(Q_(5.0, "ppb_mass")).magnitude == pytest.approx(5.0)
    assert to_ppb_mass(Q_(1.0, "ppm_mass")).magnitude == pytest.approx(1000.0)


def test_mass_fraction_converts_to_ppm():
    # 0.22 wt% Al2O3 -> 2200 ppm by mass
    assert to_ppm_mass(Q_(0.0022, "dimensionless")).magnitude == pytest.approx(2200.0)


def test_pint_alone_cannot_separate_ratio_bases():
    """Documents WHY ratio_basis exists: all three reduce to dimensionless."""
    assert str(Q_(30.0, "ug/g").dimensionality) == "dimensionless"
    assert str(Q_(30.0, "umol/mol").dimensionality) == "dimensionless"
    assert str(Q_(30.0, "mL/L").dimensionality) == "dimensionless"


@pytest.mark.parametrize("unit,expected", [
    ("ug/g", "mass"), ("mg/kg", "mass"), ("ppm_mass", "mass"), ("ppb_mass", "mass"),
    ("umol/mol", "mole"), ("mmol/mol", "mole"),
    ("mL/L", "volume"), ("cm**3/m**3", "volume"),
    ("dimensionless", "bare"),
    ("ug/mol", "mixed"),
])
def test_ratio_basis_classifies(unit, expected):
    assert ratio_basis(Q_(1.0, unit)) == expected


def test_mass_to_mole_conversion_factors():
    """Pins the numbers the ratio_basis docstring cites, so the claim is executable.

    Mass to mole MULTIPLIES by M(SiO2)/M(Al): the mole-basis figure is larger.
    The two mole conventions in use differ from each other by a further 3x.
    """
    M_Al, M_Si, M_O = 26.9815, 28.085, 15.999
    M_SiO2 = M_Si + 2 * M_O
    assert M_SiO2 == pytest.approx(60.083, abs=1e-3)
    assert M_SiO2 / M_Al == pytest.approx(2.227, abs=1e-3)
    w = 30e-6  # 30 ppm Al by mass
    per_formula_unit = (w / M_Al) / ((1 - w) / M_SiO2)
    per_atom = (w / M_Al) / ((1 - w) / M_SiO2 * 3)
    assert per_formula_unit * 1e6 == pytest.approx(66.8, abs=0.1)
    assert per_atom * 1e6 == pytest.approx(22.3, abs=0.1)
    assert per_formula_unit / per_atom == pytest.approx(3.0, abs=1e-6)
    # The mole-basis value is LARGER than the mass-basis value, not smaller.
    assert per_formula_unit * 1e6 > 30.0


def test_mole_ratio_is_not_silently_a_mass_ratio():
    """The defect this unit system exists to prevent: a single reported "30 ppm"
    can mean 30 (mass), 66.8 (per mol SiO2) or 22.3 (per mol atoms)."""
    # The three readings of "30 ppm", recomputed from molar masses.
    M_SIO2, M_AL = 60.083, 26.9815
    assert 30.0 * M_SIO2 / M_AL == pytest.approx(66.8, abs=0.1)
    assert 30.0 * (M_SIO2 / 3.0) / M_AL == pytest.approx(22.3, abs=0.1)
    with pytest.raises(ValueError, match="mole-basis"):
        to_ppm_mass(Q_(30.0, "umol/mol"))
    with pytest.raises(ValueError, match="volume-basis"):
        to_ppm_mass(Q_(30.0, "mL/L"))
    with pytest.raises(ValueError):
        to_ppb_mass(Q_(30.0, "umol/mol"))


def test_currency_cannot_be_added_to_mass():
    with pytest.raises(DimensionalityError):
        _ = Q_(100.0, "USD") + Q_(1.0, "kg")


def test_currencies_do_not_auto_convert():
    """FX is an economic assumption with a date, never a unit conversion."""
    with pytest.raises(DimensionalityError):
        Q_(100.0, "USD").to("INR")


def test_specific_energy_dimensionality():
    e = Q_(12.0, "MWh/tonne")
    require_dimensionality(e, "specific_energy_mass", "furnace_energy")
    assert e.to("J/kg").magnitude == pytest.approx(12.0 * 3.6e9 / 1000.0)


def test_require_dimensionality_rejects_bare_float():
    with pytest.raises(TypeError, match="must be a pint Quantity"):
        require_dimensionality(12.0, "specific_energy_mass", "furnace_energy")


def test_require_dimensionality_rejects_wrong_kind():
    with pytest.raises(DimensionalityError):
        require_dimensionality(Q_(5.0, "kg"), "energy", "enthalpy")


@pytest.mark.parametrize("kind,q", [
    ("mass", Q_(1.0, "tonne")),
    ("mass_flow", Q_(1.0, "tonne/hour")),
    ("temperature", Q_(1000.0, "degC")),
    ("energy", Q_(1.0, "kJ")),
    ("power", Q_(1.0, "MW")),
    ("pressure", Q_(1.0, "bar")),
    ("molar_energy", Q_(1.0, "kJ/mol")),
    ("area", Q_(1.0, "m**2")),
    ("volume", Q_(1.0, "m**3")),
    ("density", Q_(2650.0, "kg/m**3")),
    ("specific_surface_area", Q_(1.0, "m**2/g")),
    ("magnetic_flux_density", Q_(1.5, "tesla")),
    ("time", Q_(4.0, "hour")),
])
def test_registered_dimensionalities_are_correct(kind, q):
    require_dimensionality(q, kind, kind)


def test_fraction_range_enforced():
    assert require_fraction(0.92, "recovery") == pytest.approx(0.92)
    with pytest.raises(ValueError, match="recovery"):
        require_fraction(1.4, "recovery")
    with pytest.raises(ValueError):
        require_fraction(-0.01, "recovery")


def test_as_dimensionless_rejects_units():
    assert as_dimensionless(Q_(0.5, "dimensionless")) == pytest.approx(0.5)
    with pytest.raises(DimensionalityError):
        as_dimensionless(Q_(0.5, "kg"))
