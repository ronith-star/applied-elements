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

from ae.core.provenance import MISSING, Value, _Missing
from ae.core.units import to_ppm_mass

__all__ = [
    "ELEMENTS",
    "MOLAR_MASS",
    "OXIDE_STOICH",
    "Feedstock",
    "ImpurityProfile",
    "InclusionCharacter",
    "InclusionType",
    "OreType",
    "PhysicalProperties",
    "oxide_to_element",
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

#: How well an ore is known. Ordered from least to most informative; see
#: :meth:`Feedstock.characterization_tier` for the criteria and
#: :meth:`Feedstock.permits_output` for what each tier unlocks.
CharacterizationTier = Literal[
    "unmeasured", "screened", "bulk_quantified", "located",
]

#: Elements the AE-Q protocol requires quantified before an ore is considered
#: bulk-quantified. Li and B are on the list precisely because XRF cannot
#: measure them, which is what forces ICP-MS or GDMS rather than a cheaper scan.
REQUIRED_SUITE: tuple[str, ...] = ("Al", "Ti", "Li", "Fe", "Na", "K", "B")

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
    ('Al', 1164, 0.5293)
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
    def _known_elements_and_units(self) -> ImpurityProfile:
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
    def _characterized_means_measured(self) -> Feedstock:
        if self.characterized:
            if self.characterization_tier != "located":
                raise ValueError(
                    f"characterized=True requires tier 'located' (full element suite "
                    f"by a spatially resolved method); this feedstock is at tier "
                    f"'{self.characterization_tier}'. See characterization_gap() for "
                    f"what is missing."
                )
        return self

    @property
    def characterization_tier(self) -> CharacterizationTier:
        """How well this ore is known, on four tiers rather than a binary flag.

        The tiers exist because a binary characterized flag blocked all early
        screening: a candidate ore with an XRF scan is not "characterized", but
        it is not unknown either, and refusing to model it at all is the wrong
        response. Each tier unlocks different outputs (see
        :meth:`permits_output`).

        ``unmeasured``
            No impurity measurement of any kind. Supports no output; every
            grade-conditional result is a scenario.
        ``screened``
            Any measurement at all, typically XRF. Gives major-element
            composition and a Fe indication.
        ``bulk_quantified``
            The full element suite (Al, Ti, Li, Fe, Na, K, B) by a bulk method
            with adequate detection limits, i.e. ICP-MS after digestion or
            GDMS. This gives TOTAL content per element at ppm-to-ppb limits.
        ``located``
            The full suite by a spatially resolved method (LA-ICP-MS), with the
            lattice fraction measured rather than assumed, plus inclusion work.
            Only this tier supports a purification ceiling.

        WHY XRF IS INSUFFICIENT, stated correctly. A previous version of this
        module claimed bulk XRF "cannot resolve lattice impurities", implying it
        misses lattice-bound atoms. That is wrong: XRF measures TOTAL element
        content irrespective of where the atoms sit. Its actual limitations are
        that it cannot measure Li or B at all (both too light for practical
        XRF), its detection limits for Ti and the alkalis are poor at the ppm
        level that matters here, and, like every bulk method, it reports no
        spatial information. The last point applies equally to bulk ICP-MS,
        which does achieve the detection limits: the reason bulk analysis cannot
        set a purification ceiling is that it cannot tell you WHERE the
        impurities are, not that it cannot see them.
        """
        total = self.impurities.total
        method = self.impurities.method
        if not total:
            return "unmeasured"
        required = REQUIRED_SUITE
        have_suite = all(e in total for e in required)
        lattice_measured = bool(self.impurities.lattice_fraction) and all(
            not isinstance(v, _Missing)
            for e in required
            for v in [self.impurities.lattice_fraction.get(e, MISSING)]
        )
        if have_suite and method in ("LA_ICP_MS", "mixed") and lattice_measured:
            return "located"
        if have_suite and method in ("ICP_MS", "GDMS", "LA_ICP_MS", "INAA", "mixed"):
            return "bulk_quantified"
        return "screened"

    def characterization_gap(self) -> dict[str, object]:
        """What is missing to reach the next tier, as a checklist.

        Written to be actionable by a campaign planner: it names the elements
        and the measurement, not a score.
        """
        required = REQUIRED_SUITE
        total = self.impurities.total
        missing_elements = [e for e in required if e not in total]
        lattice_unmeasured = [
            e for e in required
            if isinstance(self.impurities.lattice_fraction.get(e, MISSING), _Missing)
        ]
        tier = self.characterization_tier
        nxt = {"unmeasured": "screened", "screened": "bulk_quantified",
               "bulk_quantified": "located", "located": None}[tier]
        needed: list[str] = []
        if tier in ("unmeasured", "screened"):
            if missing_elements:
                needed.append(
                    f"quantify {', '.join(missing_elements)} by ICP-MS after "
                    f"digestion or GDMS (XRF cannot measure Li or B and has "
                    f"inadequate ppm limits for Ti and the alkalis)"
                )
            if self.impurities.method in (None, "XRF", "ICP_OES"):
                needed.append(
                    f"current method {self.impurities.method!r} does not reach the "
                    f"required detection limits for the full suite"
                )
        elif tier == "bulk_quantified":
            needed.append(
                f"locate impurities by LA-ICP-MS on single grains and measure the "
                f"lattice fraction for {', '.join(lattice_unmeasured)}; bulk totals "
                f"cannot set a purification ceiling because they carry no spatial "
                f"information"
            )
            needed.append("fluid-inclusion microthermometry and CL imaging")
        return {
            "tier": tier,
            "next_tier": nxt,
            "missing_elements": missing_elements,
            "lattice_unmeasured": lattice_unmeasured,
            "to_advance": needed,
            "basis_material": self.impurities.basis_material,
        }

    def permits_output(self, output: str) -> bool:
        """Whether this ore's characterization tier supports a given output.

        The gate the binary flag was trying to be, at the right granularity:
        a screened ore may be ranked and costed on mass yield, but only a
        located ore may have a purification ceiling or a product grade claimed.
        """
        tier = self.characterization_tier
        order = ("unmeasured", "screened", "bulk_quantified", "located")
        need = {
            "screening_rank": "screened",
            "mass_yield_estimate": "screened",
            "reagent_demand": "bulk_quantified",
            "impurity_removal_estimate": "bulk_quantified",
            "purification_ceiling": "located",
            "product_grade_claim": "located",
            "qualification_dossier": "located",
        }
        if output not in need:
            raise ValueError(
                f"unknown output {output!r}; known outputs are {sorted(need)}"
            )
        return order.index(tier) >= order.index(need[output])

    def require_output(self, output: str) -> None:
        """Raise unless the tier supports the output, naming the gap."""
        if not self.permits_output(output):
            gap = self.characterization_gap()
            raise ValueError(
                f"{self.sample_id}: output {output!r} requires a higher "
                f"characterization tier than '{gap['tier']}'. To advance: "
                f"{'; '.join(gap['to_advance']) or 'see characterization_gap()'}"
            )

    @property
    def id_parts(self) -> dict[str, str]:
        _, mineral, country, deposit, num = self.sample_id.split("-")
        return {"mineral": mineral, "country": country, "deposit": deposit, "number": num}
