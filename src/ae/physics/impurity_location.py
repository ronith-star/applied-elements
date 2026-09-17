"""Where each impurity element actually sits, which decides the purity ceiling.

This is the most decision-relevant model in the physical separation track. Two
ores with identical bulk assays have completely different value if one carries
its aluminium as feldspar inclusions and the other carries it substituted into
the quartz lattice. The first can be beneficiated to HPQ; the second cannot be,
by any flowsheet, at any cost.

Four locations, and what can reach each
---------------------------------------
============  =======================================  =============================
Location      Physical form                             Removed by
============  =======================================  =============================
surface       adsorbed films, clay coatings, grinding   washing, desliming,
              media debris, fines adhering to grains    attrition scrubbing, leaching
fluid         aqueous or gaseous cavities, primary      calcination plus quenching
              and secondary, salinity-bearing           (decrepitation), then leaching
mineral       discrete feldspar, mica, rutile,          liberation by grinding, then
              zircon, Fe-oxide grains                   flotation, magnetic sep, leach
lattice       substitutional Al3+, Fe3+, B3+, Ti4+,     NOTHING in the standard
              Ge4+, P5+ for Si4+, plus interstitial     flowsheet. Hot chlorination
              charge compensators Li+, Na+, K+, H+      touches part of it, see below
============  =======================================  =============================

The substitution mechanisms are set out in Lin et al. 2020 Section 2.2
(doi:10.1007/s42461-020-00247-0): substitutional ions Al3+, Fe3+, B3+, Ti4+,
Ge4+, P5+ replacing Si4+, with interstitial Li+, K+, Na+, H+ and Fe2+ acting as
charge compensators, and configurations including Al3+ plus P5+ for 2Si4+,
silanol groups 4H+ for Si4+, and Al3+ plus an alkali in the interstice of the
Si-O tetrahedron. The same source states that alkali charge compensators are
only weakly bound and can be removed relatively readily, while the trivalent and
tetravalent substitutional ions Al3+ and Ti4+ resist removal because their
lattice energies are low relative to the Si-O-Si structure.

That asymmetry has a direct commercial consequence: Na and K in the lattice are
a chlorination problem, while Al and Ti in the lattice are a geology problem.
Hot chlorination requires a carbonaceous reductant for Ti, Al and B removal
(Liu L., Liu H., Li J. et al. 2026, Mechanism Study on Deep Removal of Lattice
Impurities from High-Purity Quartz by Chlorination Roasting, Minerals 16(8):836,
doi:10.3390/min16080836).

Equations
---------
**(E1) Partition closure.** Definitional, not empirical.

.. math::
    f_{s} + f_{f} + f_{m} + f_{l} = 1

- :math:`f_s, f_f, f_m, f_l`: fraction of an element's total inventory in
  surface, fluid inclusion, mineral inclusion and lattice locations. Each
  dimensionless in [0, 1]. Enforced to 1e-9 in
  :class:`ElementPartition`; an open inventory is a rejected input, not a
  warning.

**(E2) Location-resolved concentration.**

.. math::
    c_{j}^{(el)} = c_{\\text{total}}^{(el)} \\, f_{j}^{(el)}

- :math:`c_{\\text{total}}^{(el)}`: bulk concentration of the element, ppm by
  mass (ug/g), range 0.01 to 10000 ppm for the elements this platform tracks.
- :math:`c_j^{(el)}`: concentration residing in location :math:`j`, same units.
- Mass balance over :math:`j` returns :math:`c_{\\text{total}}` exactly, which
  is asserted in code.

**(E3) Achievable floor concentration.** The number the whole platform turns on.

.. math::
    c_{\\text{floor}}^{(el)} = c_{\\text{total}}^{(el)} \\left[ f_{l}^{(el)}
    + \\sum_{j \\ne l} f_{j}^{(el)} \\left( 1 - \\eta_{j} \\right) \\right]

- :math:`\\eta_j`: removal efficiency of the flowsheet for location :math:`j`,
  dimensionless in [0, 1]. A perfect flowsheet has :math:`\\eta_j = 1` for
  surface, fluid and mineral, giving
  :math:`c_{\\text{floor}} = c_{\\text{total}} f_l`.
- With a perfect flowsheet the floor is the lattice inventory, full stop. No
  choice of reagent, residence time or stage count moves it, which is why
  :func:`ae.core.feedstock.ImpurityProfile.lattice_ppm` raising on unmeasured
  data is correct behaviour and must not be worked around.

**(E4) Total trace sum floor and the HPQ gate.**

.. math::
    \\Sigma_{\\text{floor}} = \\sum_{el} c_{\\text{floor}}^{(el)}

Compared element-by-element against the single-grain HPQ reference limits and
against the trace sum limit of 50 ppm. A deposit passes the gate only if every
element clears its own limit; the sum clearing 50 ppm is necessary, not
sufficient.

**(E5) Implied overall removal rate.** For benchmarking against published
campaigns.

.. math::
    R = 1 - \\frac{\\Sigma_{\\text{product}}}{\\Sigma_{\\text{feed}}}

- :math:`R`: dimensionless removal fraction in [0, 1].
- Xia et al. 2024 (doi:10.3390/min14070727) report
  :math:`\\Sigma_{\\text{feed}} = 128.86` ug/g and
  :math:`\\Sigma_{\\text{product}} = 24.23` ug/g for Pakistan vein quartz, an
  81.20 percent removal, with the residue identified as lattice-bound Al, Ti
  and Li. That paper reports only those two endpoints, so no per-stage
  intermediate is modelled or attributed anywhere in this platform.

Reference limits
----------------
HPQ single-grain reference limits used for the gate, from Muller et al. 2012 in
Gotze J. and Mockel R. eds., Quartz: Deposits, Mineralogy and Analytics
(doi:10.1007/978-3-642-22161-3): Al below 30 ppm, Ti below 10, Li below 5, Na
below 8, K below 8, Ca below 5, Fe below 3, P below 2, B below 1, trace sum
below 50 ppm.

LIMITATIONS
-----------
1. **No partition data exists for the Vikarabad deposit.** It is
   uncharacterized in the citable record: no LA-ICP-MS, no cathodoluminescence,
   no fluid inclusion microthermometry, no beneficiation testwork. Every result
   this module produces for it is a SCENARIO conditional on a future
   characterization campaign, never a finding, and
   :func:`partition_fractions` raises rather than defaulting.
2. **Published per-element partition data is thin and this module does not
   invent it.** What the literature reliably gives is (a) which elements
   substitute into the lattice and by what mechanism, qualitatively, and (b)
   before-and-after totals for specific deposits with the residue attributed to
   named lattice elements. What it does not give, for any deposit retrievable
   here, is a numeric four-way split per element. :data:`PARTITION_PRIORS`
   therefore contains priors only for the elements where the mechanism is
   established enough to bound one branch, every one of them tagged ASSUMED
   with its reasoning, and :data:`NO_PARTITION_DATA` names the elements with
   nothing at all. Do not read the priors as measurements.
3. **A prior is not transferable between deposits.** Lattice Al in a
   high-temperature pegmatite quartz and in a low-temperature hydrothermal vein
   quartz differ by orders of magnitude because Al substitution is governed by
   crystallisation temperature and melt or fluid chemistry. Al lattice fraction
   is a property of the deposit, measured grain by grain, not a property of the
   element.
4. **The four locations are an idealisation.** Grain-boundary films, healed
   microfracture trails decorated with sub-micron fluid inclusions, and
   nanoscale exsolved phases sit between the categories. An operational
   definition tied to the measurement method (bulk ICP-MS after a staged leach
   versus LA-ICP-MS on inclusion-free domains) is what makes the split
   reproducible, and this module records that method on the partition object.
5. **Removal efficiencies are inputs, not predictions.** :func:`floor_profile`
   takes :math:`\\eta_j` from the caller. It does not know what a given
   flowsheet achieves, and the default is the perfect-flowsheet case
   (:math:`\\eta = 1`), which is a bound and not an expectation.
6. **An SiO2 weight percentage is not a comparable purity metric across
   papers.** :func:`lattice_ceiling_sio2_percent` converts a cation trace sum
   by difference, which is one of at least two conventions in use. Xia et al.
   2024's 99.998 wt% is exactly what their 24.23 ug/g implies by difference, but
   Qu et al. 2025's figures are not: their 18.72 ug/g (HT) and 114.56 ug/g (PX)
   imply 99.998128 and 99.988544 wt% by difference, ABOVE their published upper
   bounds of 99.970 and 99.985 by 0.0281 and 0.0035 percent relative, one-sided
   in both cases. Their SiO2 must therefore come from a whole-rock
   determination carrying oxide oxygen, water and phases outside the trace
   list. Compare deposits on the trace sum in ug/g, never on a quoted SiO2
   percentage, and treat a supplier's "99.99x percent SiO2" claim as
   uninterpretable until the convention is stated. Measured in
   ``test_benchmark_qu_2025_two_ore_bodies``.
7. **Hot chlorination breaks the four-location model's central claim, partly.**
   Lin et al. 2020 describe chlorination activating structural Al, Fe and Ti to
   the grain surface by reducing activation energy; the conceptual flowsheet
   there puts hot chlorination against lattice impurities and above 99.999
   percent SiO2. Liu et al. 2026 establish that a carbonaceous reductant is
   required for Ti, Al and B. So "lattice is unremovable" is exactly true for
   washing, flotation, magnetic separation and acid leaching, and NOT
   unconditionally true once chlorination roasting is in the flowsheet. This
   module models the lattice as unremovable by the PHYSICAL flowsheet, which is
   its scope, and :func:`floor_profile` will accept a lattice removal
   efficiency only if the caller passes one explicitly and names the mechanism.

References
----------
Every entry below was resolved against the Crossref REST API for the DOI shown,
and the fields here (author list, year, title, journal, volume, issue, pages)
are as Crossref returned them. Resolved 17 September 2026. Entries without a
DOI say what was checked instead and what remains unverified.

Muller, A., Wanvik, J. E. and Ihlen, P. M. (2012) Petrological and Chemical
Characterisation of High-Purity Quartz Deposits with Examples from Norway, in
Gotze, J. and Mockel, R. (eds.) Quartz: Deposits, Mineralogy and Analytics,
Springer Geology, pp. 71-118, doi 10.1007/978-3-642-22161-3_4 (volume DOI
10.1007/978-3-642-22161-3). Source of the HPQ single-grain reference limits used
by the gate. Closed access: the chapter full text was not retrievable from this
sandbox, so the limits are carried as previously extracted values and the
chapter's author list, title, pages and volume are what was verified here.

Lin, M., Liu, Z., Wei, Y., Liu, B., Meng, Y., Qiu, H., Lei, S., Zhang, X. and
Li, Y. (2020) A Critical Review on the Mineralogy and Processing for High-Grade
Quartz, Mining, Metallurgy and Exploration 37(5), 1627-1639,
doi 10.1007/s42461-020-00247-0. Source of the chlorination-activation
description and of the conceptual flowsheet placing hot chlorination against
lattice impurities.

Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand by
Vein Quartz Purification and Characteristics: A Case Study of Pakistan Vein
Quartz, Minerals 14(7), 727, doi 10.3390/min14070727. Source of the 128.86 ug/g
feed and 24.23 ug/g product trace sums and of the residual being lattice-bound
Al, Ti and Li.

Qu, S., Yang, X., Xia, M., Hou, Z. and Wang, Y. (2025) Analyzing trace element
distribution and purification of vein quartz: a case study of Qianqi Furong mine
in Inner Mongolia, European Journal of Mineralogy 37(6), 953-970,
doi 10.5194/ejm-37-953-2025. Source of the 18.72 ug/g and 114.56 ug/g figures
and of the published SiO2 upper bounds against which they are inconsistent by
difference.

Liu, L., Liu, H., Li, J., Peng, T., Wang, W., Wang, F. and Liu, G. (2026)
Mechanism Study on Deep Removal of Lattice Impurities from High-Purity Quartz by
Chlorination Roasting, Minerals 16(8), 836, doi 10.3390/min16080836. Source of
the requirement for a carbonaceous reductant for Ti, Al and B.
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import ELEMENTS, Feedstock, OreType
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.units import Q_, Quantity, require_fraction

if TYPE_CHECKING:
    from pint import Quantity as _PintQuantity

    # ae.core.units.Quantity is a runtime alias for UREG.Quantity, which mypy
    # cannot use as a type annotation. Qty annotates against pint's generic
    # Quantity[Any], which is a BASE class of the registry-bound UREG.Quantity
    # and not the same class (isinstance holds, identity does not), so the
    # annotation approximates the runtime class: it cannot express the registry
    # binding, and mypy will therefore not catch a quantity built on a foreign
    # registry. Runtime behaviour is unchanged, the shared UREG registry remains
    # the only registry in play, and every dimensionality check stays a runtime
    # check through require_dimensionality.
    type Qty = _PintQuantity[Any]
else:
    Qty = Quantity


__all__ = [
    "HPQ_SINGLE_GRAIN_LIMITS_PPM",
    "HPQ_TRACE_SUM_LIMIT_PPM",
    "NO_PARTITION_DATA",
    "PARTITION_PRIORS",
    "SRC_LIN_2020",
    "SRC_LIU_2026",
    "SRC_MULLER_2012",
    "SRC_QU_2025",
    "SRC_XIA_2024",
    "ElementPartition",
    "Location",
    "PartitionModel",
    "floor_concentration",
    "floor_profile",
    "hpq_gate",
    "implied_removal_rate",
    "lattice_ceiling_sio2_percent",
    "location_concentrations",
    "partition_fractions",
    "removable_ppm",
]

def _q(value: Value) -> Qty:
    """Annotation-level accessor for ``Value.quantity``.

    ``ae.core.provenance.Value.quantity`` is annotated with the runtime alias
    ``ae.core.units.Quantity``, which mypy resolves to a variable rather than a
    type. This helper re-expresses the same object as ``Qty`` so callers type
    check without altering a core module or weakening a runtime check.
    """
    return cast(Qty, value.quantity)


_ACCESSED: Final[_dt.date] = _dt.date(2026, 9, 16)

SRC_MULLER_2012: Final[Source] = Source(
    citation=(
        "Muller A., Wanvik J. E., Ihlen P. M. 2012, Petrological and Chemical "
        "Characterisation of High-Purity Quartz Deposits with Examples from Norway, "
        "in Gotze J. and Mockel R. eds., Quartz: Deposits, Mineralogy and Analytics, "
        "Springer Geology"
    ),
    tier=Tier.T1,
    doi="10.1007/978-3-642-22161-3",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Volume DOI verified via CrossRef. Single-grain HPQ reference limits as "
        "supplied in the platform brief: Al<30, Ti<10, Li<5, Na<8, K<8, Ca<5, Fe<3, "
        "P<2, B<1 ppm, trace sum <50 ppm. The chapter text itself was not retrievable "
        "in this environment, so these limits are carried at the brief's authority "
        "against this volume, and the per-element table should be re-verified against "
        "the printed chapter before any commercial commitment."
    ),
)

SRC_LIN_2020: Final[Source] = Source(
    citation=(
        "Lin M., Liu Z., Wei Y. et al. 2020, A Critical Review on the Mineralogy and "
        "Processing for High-Grade Quartz, Mining, Metallurgy and Exploration "
        "37:1627-1639"
    ),
    tier=Tier.T1,
    doi="10.1007/s42461-020-00247-0",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Full text read. Section 2.2 substitution mechanisms and charge compensation; "
        "Section 3 on alkali removal by chlorination versus the resistance of Al3+ and "
        "Ti4+; Table 3 conceptual flowsheet with product grade per stage."
    ),
)

SRC_XIA_2024: Final[Source] = Source(
    citation=(
        "Xia M., Yang X., Hou Z. 2024, Preparation of High-Purity Quartz Sand by Vein "
        "Quartz Purification and Characteristics: A Case Study of Pakistan Vein "
        "Quartz, Minerals 14(7):727"
    ),
    tier=Tier.T1,
    doi="10.3390/min14070727",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Abstract read in full; publisher blocked full text in this environment. "
        "Endpoints only: total impurities 128.86 ug/g feed to 24.23 ug/g product, "
        "81.20 percent removal, SiO2 99.998 wt%, residual identified as lattice Al, "
        "Ti, Li, main feed impurities Al, K, Ca, Na, Ti, Fe, Li. No per-stage data is "
        "published, so none is modelled."
    ),
)

SRC_QU_2025: Final[Source] = Source(
    citation=(
        "Qu S., Yang X., Xia M. et al. 2025, Analyzing trace element distribution and "
        "purification of vein quartz: a case study of Qianqi Furong mine in Inner "
        "Mongolia, European Journal of Mineralogy 37:953-970"
    ),
    tier=Tier.T1,
    doi="10.5194/ejm-37-953-2025",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Abstract read in full; full text not retrievable in this environment. "
        "Endpoints only: HT quartz 322.96 to 18.72 ug/g total trace elements, PX "
        "quartz 4944.73 to 114.56 ug/g, after calcination, crushing and sieving, "
        "magnetic separation, gravity separation, flotation and acid leaching; final "
        "SiO2 99.963-99.970 wt% (HT) and 99.974-99.985 wt% (PX)."
    ),
)

SRC_LIU_2026: Final[Source] = Source(
    citation=(
        "Liu L., Liu H., Li J. et al. 2026, Mechanism Study on Deep Removal of Lattice "
        "Impurities from High-Purity Quartz by Chlorination Roasting, Minerals 16(8):836"
    ),
    tier=Tier.T1,
    doi="10.3390/min16080836",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Metadata verified via CrossRef. Cited for the established requirement of a "
        "carbonaceous reductant for Ti, Al and B removal in chlorination roasting, as "
        "given in the platform brief. Full text not retrieved."
    ),
)


class Location(str, enum.Enum):
    """The four places an impurity atom can be, ordered by ease of removal."""

    SURFACE = "surface"
    FLUID_INCLUSION = "fluid"
    MINERAL_INCLUSION = "mineral"
    LATTICE = "lattice"

    @property
    def removable_by_physical_flowsheet(self) -> bool:
        """True for everything except the lattice.

        The lattice returns False because washing, desliming, attrition,
        flotation, magnetic separation and acid leaching cannot reach a
        substituted ion. Chlorination roasting can reach part of it, which is a
        chemical route outside this module's scope; see LIMITATIONS item 7.
        """
        return self is not Location.LATTICE


#: Single-grain HPQ reference limits, ppm by mass. Source: SRC_MULLER_2012.
HPQ_SINGLE_GRAIN_LIMITS_PPM: Final[dict[str, float]] = {
    "Al": 30.0,
    "Ti": 10.0,
    "Li": 5.0,
    "Na": 8.0,
    "K": 8.0,
    "Ca": 5.0,
    "Fe": 3.0,
    "P": 2.0,
    "B": 1.0,
}

#: Trace sum limit, ppm by mass. Necessary but not sufficient: an ore can clear
#: 50 ppm total and still fail on Al alone.
HPQ_TRACE_SUM_LIMIT_PPM: Final[float] = 50.0

#: Elements with NO published four-way partition data retrievable in this
#: environment, for any quartz deposit. Listed explicitly because an empty entry
#: in a prior table is indistinguishable from a zero.
NO_PARTITION_DATA: Final[tuple[str, ...]] = (
    "Ge",
    "Mg",
    "Mn",
    "Cr",
    "Cu",
    "Zr",
    "U",
    "Th",
    "OH",
    "Ca",
    "P",
)


class ElementPartition(BaseModel):
    """Four-way location split of one element's inventory, with provenance.

    The four fractions must close to 1 within 1e-9. ``method`` records how the
    split was established, because the operational definition of "lattice"
    depends on it: LA-ICP-MS on inclusion-free domains measures something
    different from bulk ICP-MS after a staged leach, and mixing the two in one
    table produces a split that cannot be reproduced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    element: str
    surface: Value
    fluid: Value
    mineral: Value
    lattice: Value
    method: Literal[
        "LA_ICP_MS_inclusion_free_domains",
        "staged_leach_mass_balance",
        "CL_plus_LA_ICP_MS",
        "assumed_from_mechanism",
    ]
    note: str | None = None

    @model_validator(mode="after")
    def _closes_and_is_known(self) -> ElementPartition:
        if self.element not in ELEMENTS:
            raise ValueError(
                f"unknown element {self.element!r}; tracked elements are {ELEMENTS}"
            )
        fracs = {}
        for loc in ("surface", "fluid", "mineral", "lattice"):
            v: Value = getattr(self, loc)
            f = float(_q(v).to("dimensionless").magnitude)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"{self.element} {loc} fraction {f} outside [0, 1]")
            fracs[loc] = f
        total = math.fsum(fracs.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"{self.element} partition sums to {total}, not 1: an element's "
                f"inventory must be fully accounted for across surface, fluid "
                f"inclusion, mineral inclusion and lattice (Eq. E1)"
            )
        return self

    @property
    def fractions(self) -> dict[str, float]:
        """The split as a plain mapping keyed by location value string."""
        return {
            "surface": float(_q(self.surface).to("dimensionless").magnitude),
            "fluid": float(_q(self.fluid).to("dimensionless").magnitude),
            "mineral": float(_q(self.mineral).to("dimensionless").magnitude),
            "lattice": float(_q(self.lattice).to("dimensionless").magnitude),
        }

    @property
    def weakest_tag(self) -> Tag:
        """The weakest provenance tag among the four fractions.

        A partition is only as good as its worst-supported branch, so a caller
        writing a report should quote this rather than the strongest tag.
        """
        order = [Tag.MEASURED, Tag.SOURCED, Tag.COMPUTED, Tag.DERIVED, Tag.ASSUMED]
        tags = [self.surface.tag, self.fluid.tag, self.mineral.tag, self.lattice.tag]
        return max(tags, key=order.index)


class PartitionModel(BaseModel):
    """A set of element partitions attached to one feedstock.

    Held separately from :class:`ae.core.feedstock.Feedstock` because a
    partition is an interpretation of measurements, several interpretations of
    the same assay are possible, and each should be comparable against the
    others rather than overwriting the ore record.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    sample_id: str
    partitions: dict[str, ElementPartition] = Field(default_factory=dict)
    note: str | None = None

    def get(self, element: str) -> ElementPartition:
        if element not in self.partitions:
            raise KeyError(
                f"no partition for {element} on {self.sample_id}; measured elements "
                f"are {sorted(self.partitions)}. Guessing the split would fabricate "
                f"the purity ceiling."
            )
        return self.partitions[element]


def _assumed(frac: float, basis: str) -> Value:
    return Value(
        quantity=Q_(frac, "dimensionless"),
        tag=Tag.ASSUMED,
        basis=basis,
        confidence="low",
    )


#: Partition PRIORS by (ore type, element). Every branch is ASSUMED.
#:
#: Read this table as "what the established substitution mechanism bounds", not
#: as measurements. The literature retrievable here supports the MECHANISM
#: (which elements enter the lattice, and that alkalis are weakly bound while
#: Al3+ and Ti4+ are not) and supports ENDPOINT totals for named deposits with
#: the residue attributed to lattice Al, Ti and Li. It does not supply a numeric
#: four-way split per element for any deposit. The numbers below are therefore
#: engineering estimates with their reasoning stated, provided so a scenario can
#: be run and its sensitivity to the split exposed, and every one of them should
#: be replaced by LA-ICP-MS measurement before any commercial decision.
PARTITION_PRIORS: Final[dict[tuple[OreType, str], ElementPartition]] = {
    (OreType.VEIN_QUARTZ, "Al"): ElementPartition(
        element="Al",
        surface=_assumed(
            0.05,
            (
                "ESTIMATE. Small surface film term for clay and grinding debris; Lin "
                "et al. 2020 Table 3 shows washing and desliming removing clay "
                "minerals before any chemical step, implying a real but minor "
                "surface inventory."
            ),
        ),
        fluid=_assumed(
            0.05,
            "ESTIMATE. Al is not a major fluid inclusion solute compared with Na, K and "
            "Ca; assigned a small share. No published numeric split exists.",
        ),
        mineral=_assumed(
            0.45,
            "ESTIMATE. Xia et al. 2024 (doi:10.3390/min14070727) report K-feldspar as "
            "the mineral impurity in Pakistan vein quartz with Al among the main feed "
            "impurities, and report 81.20 percent overall removal with the residue "
            "identified as lattice Al, Ti and Li: mineral-hosted Al must therefore be a "
            "large share of feed Al in that deposit. 0.45 is a mid estimate, NOT a "
            "measured value.",
        ),
        lattice=_assumed(
            0.45,
            "ESTIMATE, and the single most important number to measure. Complement of "
            "the other three branches. Lin et al. 2020 Section 2.2 establishes Al3+ for "
            "Si4+ substitution with alkali or H charge compensation, and Section 3 that "
            "Al3+ resists removal because of its low lattice energy relative to Si-O-Si. "
            "Lattice Al is deposit-specific and temperature-controlled, so this prior is "
            "NOT transferable: measure by LA-ICP-MS on inclusion-free domains.",
        ),
        method="assumed_from_mechanism",
        note=(
            "Al is the gating element for HPQ (limit 30 ppm). This prior exists to make "
            "a scenario runnable, not to answer the question."
        ),
    ),
    (OreType.VEIN_QUARTZ, "Ti"): ElementPartition(
        element="Ti",
        surface=_assumed(0.02, "ESTIMATE. Ti has no significant adsorbed-film mechanism."),
        fluid=_assumed(
            0.03,
            "ESTIMATE. Ti solubility in aqueous inclusion fluids is very low, so a near-"
            "zero share; assigned 0.03 rather than 0 to avoid asserting exactly zero.",
        ),
        mineral=_assumed(
            0.35,
            "ESTIMATE. Rutile and Ti-bearing oxide inclusions are a recognised mineral "
            "host in quartz. Share unmeasured.",
        ),
        lattice=_assumed(
            0.60,
            "ESTIMATE. Ti4+ substitutes directly for Si4+ with no charge compensation "
            "required, which makes it the most refractory substituent (Lin et al. 2020 "
            "Section 2.2 and 3, low removal rates for Al3+ and Ti4+); Xia et al. 2024 "
            "identify Ti as one of the three elements remaining after full purification. "
            "Weighted toward lattice on that mechanism, but not measured.",
        ),
        method="assumed_from_mechanism",
    ),
    (OreType.VEIN_QUARTZ, "Li"): ElementPartition(
        element="Li",
        surface=_assumed(0.02, "ESTIMATE. Negligible surface mechanism for Li."),
        fluid=_assumed(
            0.10,
            "ESTIMATE. Li is a soluble alkali and can reside in inclusion fluid; share "
            "unmeasured.",
        ),
        mineral=_assumed(
            0.13,
            "ESTIMATE. Mica and Li-bearing phases where present; in Xia et al. 2024 the "
            "only mineral impurity reported is K-feldspar, so the mineral share of Li is "
            "small for that deposit.",
        ),
        lattice=_assumed(
            0.75,
            "ESTIMATE. Li+ sits interstitially as the charge compensator for "
            "substitutional Al3+ (Lin et al. 2020 Section 2.2) and Xia et al. 2024 name "
            "Li as one of the three residual lattice elements after full purification. "
            "Heavily weighted to lattice on mechanism; the specific fraction is an "
            "estimate. Note the tension with Lin et al.'s statement that alkali charge "
            "compensators are weakly bound and readily activated, which applies to "
            "chlorination, not to leaching.",
        ),
        method="assumed_from_mechanism",
    ),
    (OreType.VEIN_QUARTZ, "Fe"): ElementPartition(
        element="Fe",
        surface=_assumed(
            0.25,
            "ESTIMATE. Fe-oxide staining and grinding media contamination are well "
            "recognised surface sources; Lin et al. 2020 Table 3 places magnetic "
            "separation against Fe-bearing impurities early in the flowsheet, implying a "
            "substantial accessible share.",
        ),
        fluid=_assumed(0.05, "ESTIMATE. Minor dissolved Fe in inclusion fluids."),
        mineral=_assumed(
            0.60,
            "ESTIMATE. Discrete hematite and other Fe-oxide inclusions are the dominant "
            "Fe host reported for vein quartz. Consistent with Fe being routinely "
            "reduced below 3 ppm by magnetic separation plus leaching in published "
            "campaigns, which requires most Fe to be non-lattice.",
        ),
        lattice=_assumed(
            0.10,
            "ESTIMATE. Fe3+ does substitute for Si4+ with charge compensation (Lin et al. "
            "2020 Section 2.2), but Fe is not among the residual elements Xia et al. 2024 "
            "report after purification, so the lattice share must be small in that "
            "deposit. Low-weighted on that negative evidence.",
        ),
        method="assumed_from_mechanism",
    ),
    (OreType.VEIN_QUARTZ, "Na"): ElementPartition(
        element="Na",
        surface=_assumed(0.10, "ESTIMATE. Adsorbed salts on grain surfaces."),
        fluid=_assumed(
            0.50,
            "ESTIMATE, and the best-grounded branch in this table. Fluid inclusions in "
            "vein quartz are aqueous and saline, conventionally reported as wt% NaCl "
            "equivalent by microthermometry, so Na is by construction a major fluid "
            "inclusion solute. Xia et al. 2024 list Na among the main feed impurities "
            "and do NOT list it among the residual lattice elements. The 0.50 magnitude "
            "is still an estimate.",
        ),
        mineral=_assumed(0.25, "ESTIMATE. Plagioclase and Na-bearing feldspar where present."),
        lattice=_assumed(
            0.15,
            "ESTIMATE. Na+ acts as an interstitial charge compensator (Lin et al. 2020 "
            "Section 2.2) but is weakly bound and readily activated, and is absent from "
            "the residual set of Xia et al. 2024.",
        ),
        method="assumed_from_mechanism",
    ),
    (OreType.VEIN_QUARTZ, "K"): ElementPartition(
        element="K",
        surface=_assumed(0.08, "ESTIMATE. Adsorbed salts."),
        fluid=_assumed(0.27, "ESTIMATE. K is a subordinate alkali in aqueous inclusion fluid."),
        mineral=_assumed(
            0.55,
            "ESTIMATE, and better grounded than most entries here: Xia et al. 2024 "
            "identify fine-grained K-feldspar as the mineral impurity in Pakistan vein "
            "quartz and list K among the main feed impurities, while K is absent from "
            "the residual lattice set. Mineral-dominated on that basis; magnitude "
            "unmeasured.",
        ),
        lattice=_assumed(
            0.10,
            "ESTIMATE. K+ interstitial charge compensation is possible but sterically "
            "unfavourable relative to Li+ and H+, and K is absent from the post-"
            "purification residue of Xia et al. 2024.",
        ),
        method="assumed_from_mechanism",
    ),
    (OreType.VEIN_QUARTZ, "B"): ElementPartition(
        element="B",
        surface=_assumed(0.05, "ESTIMATE."),
        fluid=_assumed(0.15, "ESTIMATE. Borate is soluble and can reside in inclusion fluid."),
        mineral=_assumed(0.20, "ESTIMATE. Tourmaline where present as an inclusion phase."),
        lattice=_assumed(
            0.60,
            "ESTIMATE. B3+ substitutes for Si4+ with H+ charge compensation (Lin et al. "
            "2020 Section 2.2 records B3+ plus H+ for Si4+), and Liu et al. 2026 "
            "(doi:10.3390/min16080836) treat B alongside Ti and Al as requiring a "
            "carbonaceous reductant in chlorination roasting, which is consistent with a "
            "lattice-dominated inventory. B has the tightest HPQ limit (1 ppm), so this "
            "estimate matters and is unmeasured.",
        ),
        method="assumed_from_mechanism",
    ),
}


def partition_fractions(
    feedstock: Feedstock,
    element: str,
    model: PartitionModel | None = None,
    allow_prior: bool = False,
) -> tuple[dict[str, float], Tag, str]:
    """Four-way partition for one element, measured first, prior only on request.

    Parameters
    ----------
    feedstock
        The ore. Its ``impurities.lattice_fraction`` is consulted first: when
        that element's lattice fraction has been measured, the lattice branch is
        taken from it and the caller must supply the remaining split via
        ``model``, because a measured lattice fraction alone does not resolve
        surface from fluid from mineral.
    element
        Element symbol, e.g. ``"Al"``.
    model
        A :class:`PartitionModel` carrying measured or interpreted splits for
        this sample. Takes precedence over priors.
    allow_prior
        When no measured split exists, ``False`` (default) raises and ``True``
        returns the ASSUMED ore-type prior. The default raises because the
        partition sets the purity ceiling, which is the single most
        decision-relevant number in the platform.

    Returns
    -------
    (fractions, tag, provenance_note)
        ``fractions`` keyed ``"surface"``, ``"fluid"``, ``"mineral"``,
        ``"lattice"``. ``tag`` is the weakest tag behind the split.

    Raises
    ------
    ValueError
        When the split is unmeasured and ``allow_prior`` is False.
    KeyError
        When ``allow_prior`` is True but no prior exists for this ore type and
        element pair, which is the case for every element in
        :data:`NO_PARTITION_DATA` and for every ore type other than vein quartz.
    """
    if element not in ELEMENTS:
        raise ValueError(f"unknown element {element!r}; tracked: {ELEMENTS}")
    if model is not None and element in model.partitions:
        ep = model.get(element)
        return (
            ep.fractions,
            ep.weakest_tag,
            f"partition from PartitionModel for {model.sample_id}, method {ep.method}",
        )
    if not allow_prior:
        lattice_known = element in feedstock.impurities.lattice_fraction and isinstance(
            feedstock.impurities.lattice_fraction.get(element), Value
        )
        detail = (
            "a measured lattice fraction is present on the feedstock but the "
            "surface, fluid and mineral split is not resolved by it; supply a "
            "PartitionModel"
            if lattice_known
            else "no lattice fraction has been measured on this feedstock either"
        )
        raise ValueError(
            f"no measured partition for {element} on {feedstock.sample_id}: {detail}. "
            f"The four-way split sets the achievable purity floor (Eq. E3) and will "
            f"not be guessed. Pass allow_prior=True to substitute the ASSUMED "
            f"ore-type prior, which makes every downstream number a scenario."
        )
    key = (feedstock.ore_type, element)
    if key not in PARTITION_PRIORS:
        available = sorted(e for (o, e) in PARTITION_PRIORS if o is feedstock.ore_type)
        raise KeyError(
            f"no partition prior for {element} in {feedstock.ore_type.value}. "
            f"Priors exist for {available or 'no elements of this ore type'}. "
            f"Elements with no published partition data at all: {NO_PARTITION_DATA}. "
            f"Measure it; do not substitute another element's split."
        )
    ep = PARTITION_PRIORS[key]
    return (
        ep.fractions,
        Tag.ASSUMED,
        (
            f"ASSUMED ore-type prior for {element} in {feedstock.ore_type.value}; "
            f"every branch is an engineering estimate, not a measurement"
        ),
    )


def location_concentrations(
    feedstock: Feedstock,
    element: str,
    model: PartitionModel | None = None,
    allow_prior: bool = False,
) -> dict[str, float]:
    """Concentration of one element in each location, ppm by mass (Eq. E2).

    Mass balance closes exactly: the four returned values sum to the bulk
    concentration, asserted before return.
    """
    bulk = feedstock.impurities.total_ppm(element)
    if bulk < 0.0:
        raise AssertionError(f"negative bulk concentration {bulk} ppm for {element}")
    fracs, _tag, _note = partition_fractions(
        feedstock, element, model=model, allow_prior=allow_prior
    )
    out = {loc: bulk * f for loc, f in fracs.items()}
    if not math.isclose(math.fsum(out.values()), bulk, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(
            f"location concentrations sum to {math.fsum(out.values())} ppm but bulk "
            f"is {bulk} ppm: mass balance on {element} failed"
        )
    return out


def floor_concentration(
    feedstock: Feedstock,
    element: str,
    model: PartitionModel | None = None,
    allow_prior: bool = False,
    efficiencies: Mapping[str, float] | None = None,
    lattice_removal: float = 0.0,
    lattice_removal_mechanism: str | None = None,
) -> float:
    """Achievable floor concentration for one element, ppm by mass (Eq. E3).

    Parameters
    ----------
    feedstock, element, model, allow_prior
        As :func:`partition_fractions`.
    efficiencies
        Removal efficiency per location for ``"surface"``, ``"fluid"`` and
        ``"mineral"``. Defaults to 1.0 each, the perfect-flowsheet bound.
    lattice_removal
        Fraction of the lattice inventory removed. Must be 0 unless a mechanism
        is named, because no step in a physical flowsheet reaches the lattice.
    lattice_removal_mechanism
        Required when ``lattice_removal`` is non-zero, e.g. ``"chlorination
        roasting with carbonaceous reductant, Liu et al. 2026
        doi:10.3390/min16080836"``. Forces the caller to state what physics they
        are invoking.

    Returns
    -------
    float
        Floor concentration in ppm by mass, never below the un-removed lattice
        residue and never above the bulk concentration (both asserted).
    """
    eff = {"surface": 1.0, "fluid": 1.0, "mineral": 1.0}
    if efficiencies:
        unknown = set(efficiencies) - {"surface", "fluid", "mineral"}
        if unknown:
            raise KeyError(
                f"unknown efficiency keys {sorted(unknown)}; lattice removal is set "
                f"by lattice_removal with a named mechanism, not here"
            )
        for k, v in efficiencies.items():
            eff[k] = require_fraction(v, f"efficiency[{k}]")
    lr = require_fraction(lattice_removal, "lattice_removal")
    if lr > 0.0 and not lattice_removal_mechanism:
        raise ValueError(
            "lattice_removal above zero requires lattice_removal_mechanism to be "
            "named. No step in the physical flowsheet (washing, desliming, attrition, "
            "flotation, magnetic separation, acid leaching) removes lattice-substituted "
            "Al, Ti, Li or B. Chlorination roasting reaches part of the lattice and "
            "needs a carbonaceous reductant for Ti, Al and B (Liu et al. 2026, "
            "doi:10.3390/min16080836); name it if that is what you mean."
        )
    conc = location_concentrations(
        feedstock, element, model=model, allow_prior=allow_prior
    )
    floor = (
        conc["surface"] * (1.0 - eff["surface"])
        + conc["fluid"] * (1.0 - eff["fluid"])
        + conc["mineral"] * (1.0 - eff["mineral"])
        + conc["lattice"] * (1.0 - lr)
    )
    bulk = feedstock.impurities.total_ppm(element)
    if floor < -1e-12:
        raise AssertionError(f"negative floor concentration {floor} ppm for {element}")
    if floor > bulk + 1e-9:
        raise AssertionError(
            f"floor {floor} ppm exceeds bulk {bulk} ppm for {element}: purification "
            f"cannot concentrate an impurity"
        )
    return max(floor, 0.0)


def removable_ppm(
    feedstock: Feedstock,
    element: str,
    model: PartitionModel | None = None,
    allow_prior: bool = False,
) -> float:
    """Ppm of an element that a perfect physical flowsheet could remove.

    Equals bulk minus the lattice inventory. Reported separately from the floor
    because it is the quantity a flowsheet designer is buying with each stage.
    """
    conc = location_concentrations(
        feedstock, element, model=model, allow_prior=allow_prior
    )
    return conc["surface"] + conc["fluid"] + conc["mineral"]


def floor_profile(
    feedstock: Feedstock,
    elements: tuple[str, ...] | None = None,
    model: PartitionModel | None = None,
    allow_prior: bool = False,
    efficiencies: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Floor concentration for several elements, ppm by mass (Eq. E3 per element).

    Defaults to the elements that have an HPQ reference limit and are measured
    on this feedstock, so the result can be fed straight to :func:`hpq_gate`.
    """
    if elements is None:
        elements = tuple(
            e for e in HPQ_SINGLE_GRAIN_LIMITS_PPM if e in feedstock.impurities.total
        )
    if not elements:
        raise ValueError(
            f"{feedstock.sample_id} has no measured elements among the HPQ reference "
            f"set {sorted(HPQ_SINGLE_GRAIN_LIMITS_PPM)}; a four-oxide royalty assay "
            f"cannot answer the HPQ question"
        )
    return {
        e: floor_concentration(
            feedstock, e, model=model, allow_prior=allow_prior, efficiencies=efficiencies
        )
        for e in elements
    }


def hpq_gate(
    floors_ppm: Mapping[str, float],
    limits_ppm: Mapping[str, float] | None = None,
    sum_limit_ppm: float = HPQ_TRACE_SUM_LIMIT_PPM,
) -> tuple[bool, dict[str, tuple[float, float, bool]], str]:
    """Test a floor profile against the HPQ single-grain reference limits (Eq. E4).

    Returns
    -------
    (passes, per_element, verdict)
        ``per_element`` maps element to (floor_ppm, limit_ppm, passes).
        ``passes`` is True only if every element with a limit clears it AND the
        trace sum clears ``sum_limit_ppm``. Elements without a limit are
        reported but do not gate.

    Notes
    -----
    A failing element cannot be compensated by a comfortable margin elsewhere.
    Al at 45 ppm against a 30 ppm limit disqualifies the material for crucible
    inner-layer sand regardless of how low Fe is, because the specification is
    per element. The trace sum test is therefore necessary and not sufficient,
    and this function reports both.
    """
    limits = dict(limits_ppm or HPQ_SINGLE_GRAIN_LIMITS_PPM)
    per: dict[str, tuple[float, float, bool]] = {}
    failures: list[str] = []
    for el, floor in floors_ppm.items():
        if floor < 0.0:
            raise ValueError(f"negative floor for {el}: {floor} ppm")
        if el not in limits:
            continue
        ok = floor <= limits[el]
        per[el] = (floor, limits[el], ok)
        if not ok:
            failures.append(f"{el} {floor:.2f} > {limits[el]:.2f} ppm")
    trace_sum = math.fsum(v for k, v in floors_ppm.items() if k != "OH")
    sum_ok = trace_sum <= sum_limit_ppm
    if not sum_ok:
        failures.append(f"trace sum {trace_sum:.2f} > {sum_limit_ppm:.2f} ppm")
    passes = not failures
    if passes:
        verdict = (
            f"clears all {len(per)} gated limits; trace sum {trace_sum:.2f} ppm against "
            f"{sum_limit_ppm:.2f} ppm. Conditional on the partition provenance: a gate "
            f"passed on ASSUMED priors is a scenario, not a qualification."
        )
    else:
        verdict = "FAILS on " + "; ".join(failures)
    return passes, per, verdict


def implied_removal_rate(feed_sum_ppm: float, product_sum_ppm: float) -> float:
    """Overall trace removal fraction (Eq. E5).

    Examples
    --------
    Xia et al. 2024: 128.86 ug/g feed, 24.23 ug/g product.
    1 - 24.23/128.86 = 1 - 0.188034 = 0.811966, i.e. 81.20 percent, matching the
    81.20 percent the paper reports.

    >>> round(implied_removal_rate(128.86, 24.23) * 100, 2)
    81.2
    """
    if feed_sum_ppm <= 0.0:
        raise ValueError(f"feed trace sum must be positive, got {feed_sum_ppm}")
    if product_sum_ppm < 0.0:
        raise ValueError(f"product trace sum cannot be negative, got {product_sum_ppm}")
    if product_sum_ppm > feed_sum_ppm:
        raise ValueError(
            f"product trace sum {product_sum_ppm} exceeds feed {feed_sum_ppm}: "
            f"purification cannot add impurity (check for grinding media "
            f"contamination, which is a real effect but is a different mass balance)"
        )
    return require_fraction(1.0 - product_sum_ppm / feed_sum_ppm, "removal rate")


def lattice_ceiling_sio2_percent(total_trace_floor_ppm: float) -> float:
    """SiO2 purity implied by a residual trace sum, wt percent.

    .. math::
        \\mathrm{SiO_2}\\,[\\%] = 100 - \\frac{\\Sigma_{\\text{floor}}}{10^{4}}

    Examples
    --------
    24.23 ppm residual trace is 24.23e-4 wt percent, so
    100 - 0.002423 = 99.997577 wt percent SiO2, consistent with the 99.998 wt
    percent reported by Xia et al. 2024.

    >>> round(lattice_ceiling_sio2_percent(24.23), 6)
    99.997577

    Notes
    -----
    This converts a cation trace sum to an SiO2 figure by difference, which is
    how the quartz literature quotes purity. It is NOT an independent
    measurement of SiO2 and it ignores OH and any oxygen stoichiometry
    correction, so the last digit is convention rather than data.
    """
    if total_trace_floor_ppm < 0.0:
        raise ValueError("trace sum cannot be negative")
    if total_trace_floor_ppm > 1e6:
        raise ValueError("trace sum above 1e6 ppm is not a trace")
    return 100.0 - total_trace_floor_ppm / 1e4
