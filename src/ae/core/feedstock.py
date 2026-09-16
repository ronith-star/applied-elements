"""FEEDSTOCK: an ore, described well enough to run every downstream model.

Platform principle: nothing is hardcoded to one deposit. Every model takes a
FEEDSTOCK and a SITE, so a new ore is a new object rather than a new code path.

Impurity convention
-------------------
Impurities are stored as ELEMENTAL mass ratios (ppm by mass), not as oxides.
Royalty and whole-rock assays are usually reported as oxides (Al2O3, Fe2O3),
and converting by eye is a recurring error, so :func:`oxide_to_element` does it
with explicit stoichiometry and tags the result DERIVED with the factor used.

Why the lattice fraction is a first-class field
-----------------------------------------------
Lattice-substituted Al, Ti, Li and B cannot be removed by any acid leach, so the
lattice inventory sets the purity ceiling of the deposit no matter how good the
flowsheet is. A FEEDSTOCK that reports a total impurity level without the lattice
split cannot answer the only question that matters for HPQ, and the schema makes
that explicit: :attr:`ImpurityProfile.lattice_fraction` may be MISSING, and the
purification models raise rather than guess when it is.
"""

from __future__ import annotations

import enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.provenance import MISSING, Tag, Value, _Missing
from ae.core.units import Q_, Quantity, to_ppm_mass

__all__ = [
    "OreType",
    "InclusionType",
    "ELEMENTS",
    "MOLAR_MASS",
    "OXIDE_STOICH",
    "oxide_to_element",
    "ImpurityProfile",
    "InclusionCharacter",
    "PhysicalProperties",
    "Feedstock",
]


class OreType(str, enum.Enum):
    """Ore classes this platform models. Extend, do not overload."""

    VEIN_QUARTZ = "vein_quartz"
    PEGMATITE_QUARTZ = "pegmatite_quartz"
    QUARTZITE = "quartzite"
    ALASKITE = "alaskite"           # the Spruce Pine host rock
    INDUSTRIAL_SAND = "industrial_sand"
    NOVACULITE = "novaculite"
    HYDROTHERMAL_CRYSTAL = "hydrothermal_crystal"
    DOLOMITE = "dolomite"


class InclusionType(str, enum.Enum):
    FLUID = "fluid"
    MINERAL = "mineral"
    MELT = "melt"


#: Elements tracked at ppm/ppb level for silica. From the platform brief plus OH.
ELEMENTS: tuple[str, ...] = (
    "Al", "Ti", "Fe", "Li", "Na", "K", "Ca", "B", "P", "Ge", "Mg", "Mn", "Cr", "Cu",
    "Zr", "U", "Th", "OH",
)

#: g/mol. Source: IUPAC 2021 standard atomic weights.
MOLAR_MASS: dict[str, float] = {
    "Al": 26.9815, "Ti": 47.867, "Fe": 55.845, "Li": 6.94, "Na": 22.9898,
    "K": 39.0983, "Ca": 40.078, "B": 10.81, "P": 30.9738, "Ge": 72.630,
    "Mg": 24.305, "Mn": 54.938, "Cr": 51.996, "Cu": 63.546, "Zr": 91.224,
    "U": 238.029, "Th": 232.038, "Si": 28.085, "O": 15.999, "OH": 17.007,
}

#: Oxide formulas as (element, n_element, n_oxygen) for assay conversion.
OXIDE_STOICH: dict[str, tuple[str, int, int]] = {
    "Al2O3": ("Al", 2, 3), "Fe2O3": ("Fe", 2, 3), "FeO": ("Fe", 1, 1),
    "TiO2": ("Ti", 1, 2), "CaO": ("Ca", 1, 1), "MgO": ("Mg", 1, 1),
    "Na2O": ("Na", 2, 1), "K2O": ("K", 2, 1), "Li2O": ("Li", 2, 1),
    "B2O3": ("B", 2, 3), "P2O5": ("P", 2, 5), "MnO": ("Mn", 1, 1),
    "Cr2O3": ("Cr", 2, 3), "ZrO2": ("Zr", 1, 2), "SiO2": ("Si", 1, 2),
}


def oxide_to_element(oxide: str, oxide_ppm: float) -> tuple[str, float, float]:
    """Convert an oxide mass concentration to its elemental equivalent.

    Parameters
    ----------
    oxide
        Formula as written in the assay, e.g. ``"Al2O3"``.
    oxide_ppm
        Oxide concentration in ppm by mass (equivalently, wt% times 10000).

    Returns
    -------
    (element, element_ppm, factor)
        The factor is the multiplier applied, returned so it can be recorded in
        the DERIVED value's basis string rather than left implicit.

    Examples
    --------
    0.22 wt% Al2O3 is 2200 ppm oxide, and Al is 52.9 percent of Al2O3 by mass:

    >>> el, ppm, f = oxide_to_element("Al2O3", 2200.0)
    >>> el, round(ppm), round(f, 4)
    ('Al', 1164, 0.5292)
    """
    if oxide not in OXIDE_STOICH:
        raise KeyError(f"unknown oxide {oxide!r}; known: {sorted(OXIDE_STOICH)}")
    el, n_el, n_o = OXIDE_STOICH[oxide]
    m_oxide = n_el * MOLAR_MASS[el] + n_o * MOLAR_MASS["O"]
    factor = (n_el * MOLAR_MASS[el]) / m_oxide
    return el, oxide_ppm * factor, factor


class ImpurityProfile(BaseModel):
    """Per-element impurity inventory with the location split that sets the ceiling.

    ``total`` is the bulk concentration. ``lattice_fraction`` is the fraction of
    that element structurally substituted into the quartz lattice, which no
    leach or flotation step can touch. Both are per element.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    total: dict[str, Value] = Field(default_factory=dict)
    lattice_fraction: dict[str, Value | _Missing] = Field(default_factory=dict)
    method: Literal["XRF", "ICP_MS", "ICP_OES", "LA_ICP_MS", "GDMS", "INAA", "mixed"] | None = None
    basis_material: Literal["run_of_mine", "crushed", "beneficiated", "single_grain"] | None = None

    @model_validator(mode="after")
    def _known_elements_and_units(self) -> "ImpurityProfile":
        for name, mapping in (("total", self.total), ("lattice_fraction", self.lattice_fraction)):
            for el in mapping:
                if el not in ELEMENTS:
                    raise ValueError(
                        f"{name}: unknown element {el!r}; tracked elements are {ELEMENTS}"
                    )
        for el, v in self.total.items():
            # Raises if a mole-basis ratio was supplied; see ae.core.units.ratio_basis
            to_ppm_mass(v.quantity)
        for el, v in self.lattice_fraction.items():
            if isinstance(v, Value):
                f = float(v.quantity.to("dimensionless").magnitude)
                if not 0.0 <= f <= 1.0:
                    raise ValueError(f"lattice_fraction[{el}] must be in [0, 1], got {f}")
        return self

    def total_ppm(self, element: str) -> float:
        """Bulk concentration in ppm by mass, raising if never measured."""
        if element not in self.total:
            raise KeyError(
                f"{element} was not measured in this profile (method={self.method}). "
                f"Available: {sorted(self.total)}. A four-oxide royalty assay reports no "
                f"Ti, Li, B, Na, K, U or Th, and those decide HPQ viability."
            )
        return float(to_ppm_mass(self.total[element].quantity).magnitude)

    def lattice_ppm(self, element: str) -> float:
        """Lattice-bound concentration in ppm by mass.

        Raises
        ------
        MissingValueError
            If the lattice split was not measured. This is deliberate: guessing
            the split would fabricate the deposit's purity ceiling, which is the
            single most decision-relevant number in the platform.
        """
        bulk = self.total_ppm(element)
        frac = self.lattice_fraction.get(element, MISSING)
        if isinstance(frac, _Missing) or frac is MISSING:
            return frac._raise() if isinstance(frac, _Missing) else MISSING._raise()
        return bulk * float(frac.quantity.to("dimensionless").magnitude)

    def sum_ppm(self, elements: tuple[str, ...] | None = None) -> float:
        """Total trace sum over measured elements, excluding OH.

        OH is excluded because it is reported on a different basis in the
        literature (as water content or by FTIR absorbance) and adding it to a
        cation sum double-counts hydrogen-compensated Al substitution.
        """
        els = elements or tuple(e for e in self.total if e != "OH")
        return sum(self.total_ppm(e) for e in els)


class InclusionCharacter(BaseModel):
    """Fluid and mineral inclusion population, which drives calcination response."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fluid_density: Value | _Missing = MISSING       # inclusions per mm^3
    fluid_salinity: Value | _Missing = MISSING      # wt% NaCl equivalent
    homogenisation_T: Value | _Missing = MISSING    # microthermometry, degC
    mineral_phases: tuple[str, ...] = ()
    mineral_volume_fraction: Value | _Missing = MISSING
    median_inclusion_size: Value | _Missing = MISSING


class PhysicalProperties(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    density: Value | _Missing = MISSING
    bond_work_index: Value | _Missing = MISSING     # kWh/short ton, Bond convention
    f80: Value | _Missing = MISSING                 # feed 80 percent passing size
    decrepitation_index: Value | _Missing = MISSING  # fraction fines after thermal shock
    grain_size: Value | _Missing = MISSING


class Feedstock(BaseModel):
    """One characterized (or uncharacterized) ore.

    ``sample_id`` encodes mineral, country, deposit and sample number, so data
    from several deposits and sites can share one table without collision:
    ``AE-Q-IN-VKB-001`` is quartz, India, Vikarabad, sample 1.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sample_id: str = Field(pattern=r"^AE-[QD]-[A-Z]{2}-[A-Z0-9]{2,6}-\d{3}$")
    ore_type: OreType
    deposit_name: str
    country: str = Field(min_length=2, max_length=2, description="ISO 3166-1 alpha-2")
    region: str | None = None
    impurities: ImpurityProfile = Field(default_factory=ImpurityProfile)
    inclusions: InclusionCharacter = Field(default_factory=InclusionCharacter)
    physical: PhysicalProperties = Field(default_factory=PhysicalProperties)
    sio2_percent: Value | _Missing = MISSING
    characterized: bool = Field(
        default=False,
        description="True only when the analytical suite required by the AE-Q "
                    "protocol has been run. False means every grade-conditional "
                    "result derived from this feedstock is a scenario.",
    )
    note: str | None = None

    @model_validator(mode="after")
    def _characterized_means_measured(self) -> "Feedstock":
        if self.characterized:
            required = ("Al", "Ti", "Li", "Fe", "Na", "K", "B")
            missing = [e for e in required if e not in self.impurities.total]
            if missing:
                raise ValueError(
                    f"characterized=True requires measured {required}; missing {missing}"
                )
            if self.impurities.method not in ("LA_ICP_MS", "GDMS", "mixed"):
                raise ValueError(
                    "characterized=True requires a single-grain capable method "
                    "(LA_ICP_MS or GDMS): bulk XRF cannot resolve lattice impurities"
                )
        return self

    @property
    def id_parts(self) -> dict[str, str]:
        _, mineral, country, deposit, num = self.sample_id.split("-")
        return {"mineral": mineral, "country": country, "deposit": deposit, "number": num}
