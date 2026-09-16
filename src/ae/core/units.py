"""Project unit registry and dimensional-analysis enforcement.

Every physical quantity in this platform is a :class:`pint.Quantity`. Bare floats
are accepted only where a quantity is genuinely dimensionless (mass fractions,
recovery, efficiencies), and those are still range-checked.

Rationale for a single shared registry: pint quantities created by two different
``UnitRegistry`` instances cannot be combined, and the resulting error message is
opaque. Import ``UREG`` (or the ``Q_`` constructor) from this module everywhere.

Concentration convention
------------------------
Trace impurity levels in quartz are reported in the literature as mass ratios:
"ppm" almost always means micrograms of element per gram of solid (ug/g) and
"ppb" means nanograms per gram (ng/g). pint's built-in ``ppm`` is a bare
dimensionless 1e-6, which silently conflates mass ratio with mole ratio and with
volume ratio.

Note carefully: pint reduces ``ug/g``, ``umol/mol`` and ``mL/L`` ALL to
``dimensionless``, so a dimensionality check cannot separate them. The guard is
therefore :func:`ratio_basis`, which inspects unit composition, and the
``to_ppm_mass``/``to_ppb_mass`` helpers that call it:

    >>> from ae.core.units import Q_, to_ppm_mass
    >>> to_ppm_mass(Q_(30.0, "ug/g")).magnitude
    30.0

``ppm_mass`` and ``ppb_mass`` are defined as aliases of ug/g and ng/g so that the
numeric value a paper reports can be entered verbatim without a conversion step.
"""

from __future__ import annotations

from typing import Final

import pint

__all__ = [
    "UREG",
    "Q_",
    "Quantity",
    "DimensionalityError",
    "ratio_basis",
    "to_ppm_mass",
    "to_ppb_mass",
    "require_dimensionality",
    "require_fraction",
    "as_dimensionless",
]

UREG: Final[pint.UnitRegistry] = pint.UnitRegistry(auto_reduce_dimensions=False)
Quantity = UREG.Quantity
Q_ = UREG.Quantity
DimensionalityError = pint.DimensionalityError

# --- project-specific units -------------------------------------------------
# Mass-ratio concentrations. Defined against kg/kg so that the dimensionality is
# [mass]/[mass] and cannot be mixed with a mole or volume ratio by accident.
UREG.define("ppm_mass = 1e-6 * kg / kg")
UREG.define("ppb_mass = 1e-9 * kg / kg")
UREG.define("ppt_mass = 1e-12 * kg / kg")

#: Declared ratio basis for project aliases whose mass trace pint erases.
_ALIAS_BASIS: Final[dict[str, str]] = {
    "ppm_mass": "mass",
    "ppb_mass": "mass",
    "ppt_mass": "mass",
}

# Currency. pint has no money dimension; these are defined as independent base
# units so that a cost cannot be added to a mass without raising. Exchange rates
# are NOT defined here: converting INR to USD is an economic assumption with a
# date and a source, handled explicitly in ae.econ.fx, never by unit magic.
UREG.define("USD = [currency_usd]")
UREG.define("INR = [currency_inr]")

# Convenience aliases used throughout the process literature.
UREG.define("tonne_metric = 1000 * kg = t_metric")
UREG.define("MWh_per_tonne = megawatt_hour / tonne")

#: Reference units for each quantity kind this platform passes between modules.
#: Checks compare pint dimensionality OBJECTS against these, never strings: pint
#: orders the factors in a dimensionality string by its own rules, so
#: ``str(dimensionality) == "[mass] * [length] ** 2 / [time] ** 2"`` is a brittle
#: test that breaks on a pint upgrade. Comparing against a reference unit is
#: stable and doubles as documentation of the expected unit.
_REF_UNITS: Final[dict[str, str]] = {
    "mass": "kg",
    "mass_flow": "kg/s",
    "length": "m",
    "temperature": "K",
    "energy": "J",
    "specific_energy_mass": "J/kg",
    "power": "W",
    "pressure": "Pa",
    "molar_energy": "J/mol",
    "concentration_mass_ratio": "kg/kg",  # reduces to dimensionless
    "time": "s",
    "area": "m**2",
    "volume": "m**3",
    "density": "kg/m**3",
    "specific_surface_area": "m**2/kg",
    "magnetic_flux_density": "T",
    "molar_mass": "kg/mol",
    "heat_capacity_molar": "J/(mol*K)",
    "thermal_conductivity": "W/(m*K)",
    "heat_transfer_coefficient": "W/(m**2*K)",
    "diffusivity": "m**2/s",
    "rate_first_order": "1/s",
    "cost_per_mass_usd": "USD/kg",
    "cost_per_mass_inr": "INR/kg",
}

#: Dimensionality objects, resolved once at import.
DIMS: Final[dict[str, object]] = {
    kind: UREG.Quantity(1.0, unit).dimensionality for kind, unit in _REF_UNITS.items()
}


def ratio_basis(q: Quantity) -> str:
    """Classify a dimensionless ratio as ``"mass"``, ``"mole"``, ``"volume"``,
    ``"bare"`` or ``"mixed"`` by inspecting its unit composition.

    This exists because **pint cannot tell these apart by dimensionality**: it
    reduces ``ug/g``, ``umol/mol`` and ``mL/L`` all to ``dimensionless``, so a
    dimensional check passes a mole ratio into a mass-ratio slot silently. That
    is not a hypothetical: trace-element data is published in all three bases and
    the numeric values differ by the ratio of molar masses (30 ppm Al by mass is
    about 11 ppm by mole in SiO2, a factor of 2.7).

    ``"bare"`` means a true dimensionless number with no unit trace, i.e. a mass
    fraction entered as 0.0022 rather than as 2200 ug/g. Callers decide whether
    to accept it; :func:`to_ppm_mass` does, because a bare fraction in this
    codebase is a mass fraction by convention.
    """
    if not isinstance(q, Quantity):
        return "bare"
    container = q.units._units
    if not container:
        return "bare"
    bases: set[str] = set()
    for unit_name in container:
        # Project ratio aliases are defined as scaled kg/kg, which pint collapses
        # to a dimensionless scalar, erasing the mass trace. Their basis is
        # therefore declared here rather than inferred.
        if unit_name in _ALIAS_BASIS:
            bases.add(_ALIAS_BASIS[unit_name])
            continue
        dim = UREG.get_base_units(unit_name)[1].dimensionality
        if "[mass]" in dim:
            bases.add("mass")
        elif "[substance]" in dim:
            bases.add("mole")
        elif "[length]" in dim:
            bases.add("volume")
        else:
            bases.add("other")
    if len(bases) == 1:
        return bases.pop()
    return "mixed"


def to_ppm_mass(q: Quantity) -> Quantity:
    """Convert a mass-ratio concentration to ppm by mass (ug/g).

    Accepts the ``ppm_mass``/``ppb_mass`` aliases, any mass-per-mass unit
    (``ug/g``, ``mg/kg``), and a bare dimensionless mass fraction.

    Raises
    ------
    ValueError
        If ``q`` is a mole ratio or a volume ratio. pint would convert these
        without complaint because all three reduce to dimensionless, so this
        check is explicit rather than dimensional. Convert with an explicit
        molar-mass calculation instead, so the stoichiometry is visible.
    """
    basis = ratio_basis(q)
    if basis not in ("mass", "bare"):
        raise ValueError(
            f"expected a mass-ratio concentration, got a {basis}-basis ratio "
            f"({q.units}). Mole and volume ratios reduce to dimensionless in pint "
            f"and would convert silently; convert explicitly via molar mass."
        )
    return q.to("ug/g")


def to_ppb_mass(q: Quantity) -> Quantity:
    """Convert a mass-ratio concentration to ppb by mass (ng/g)."""
    basis = ratio_basis(q)
    if basis not in ("mass", "bare"):
        raise ValueError(
            f"expected a mass-ratio concentration, got a {basis}-basis ratio ({q.units})"
        )
    return q.to("ng/g")


def require_dimensionality(q: Quantity, kind: str, name: str = "value") -> Quantity:
    """Assert that ``q`` has the dimensionality registered under ``kind``.

    Parameters
    ----------
    q
        Quantity to check.
    kind
        Key into :data:`DIMS`, e.g. ``"specific_energy_mass"``.
    name
        Name used in the error message, so a failure names the offending field.

    Returns
    -------
    Quantity
        ``q`` unchanged, to allow use inline in an expression.
    """
    if kind not in DIMS:
        raise KeyError(f"unknown dimensionality kind {kind!r}; known: {sorted(DIMS)}")
    if not isinstance(q, Quantity):
        raise TypeError(
            f"{name} must be a pint Quantity with dimensionality {kind!r}, "
            f"got {type(q).__name__}. Bare numbers are rejected so that a unit "
            f"mistake cannot propagate silently."
        )
    if q.dimensionality != DIMS[kind]:
        raise DimensionalityError(
            q.units, UREG.Unit(_REF_UNITS[kind]),
            extra_msg=f" while checking {name!r} as {kind!r}",
        )
    return q


def require_fraction(x: float, name: str = "fraction", lo: float = 0.0, hi: float = 1.0) -> float:
    """Validate a genuinely dimensionless fraction (recovery, efficiency, yield).

    These are the one class of quantity carried as bare floats, because wrapping
    a recovery of 0.92 in a unit adds noise without adding a check. The range
    check is the substitute for the dimensional check.
    """
    if isinstance(x, Quantity):
        x = as_dimensionless(x)
    if not (lo <= float(x) <= hi):
        raise ValueError(f"{name} must lie in [{lo}, {hi}], got {x}")
    return float(x)


def as_dimensionless(q: Quantity | float) -> float:
    """Reduce a dimensionless quantity to a float, raising if it carries units."""
    if not isinstance(q, Quantity):
        return float(q)
    return float(q.to("dimensionless").magnitude)
