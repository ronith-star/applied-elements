r"""Chlorination roasting: metal chloride formation and volatilization from quartz.

Process being modelled. A quartz concentrate is held at 900 to 1300 degC in a
flowing HCl or Cl2 atmosphere, optionally with a carbonaceous reductant (carbon
black, graphite, charcoal) mixed into the charge. Impurity cations react to form
metal chlorides, and those chlorides leave the solid if their vapour pressure at
the roasting temperature is high enough to sustain the required evaporation
flux. Removal therefore requires THREE conditions to hold simultaneously, and
this module tests each separately rather than collapsing them into one
efficiency:

1. THERMODYNAMIC. The chlorination reaction must be spontaneous,
   :math:`\Delta G_r(T) < 0`. For impurities present as simple oxides in an HCl
   atmosphere this is usually satisfied for alkalis and iron. For Ti, Al and B
   it is NOT satisfied without a reductant (see the reductant rule below).
2. VOLATILITY. The chloride must have appreciable vapour pressure at the
   roasting temperature. This is the step the boiling and sublimation points
   govern: TiCl4 boils at 136.4 degC, AlCl3 sublimes near 180 degC, FeCl3
   sublimes near 315 degC, all far below roasting temperature, whereas NaCl
   boils at 1465 degC and KCl sublimes near 1500 degC, so the alkali chlorides
   are only marginally volatile in the roasting window and their removal is
   slower than a "boiling point" comparison suggests.
3. TRANSPORT. The cation must reach a surface where the chlorinating gas is,
   which for a lattice-substituted cation is a solid-state diffusion problem
   handled in :mod:`ae.physics.diffusion` and NOT in this module. A species can
   pass tests 1 and 2 and still not be removed because it cannot get out of the
   lattice. :func:`removal_screen` reports the three conditions separately so
   that a pass on volatility is never mistaken for removal.

THE REDUCTANT RULE
==================
Liu et al. (2026, Minerals 16(8), 836, doi 10.3390/min16080836) state that
carbonaceous reductants are indispensable for enabling spontaneous chlorination
of substitutional impurities such as Ti, Al and B, while alkali metals (Na, K,
Li) can be effectively removed under an HCl atmosphere at moderate temperatures.
This module enforces that finding structurally: :func:`removal_screen` returns
``thermodynamically_blocked`` for Ti, Al and B whenever
``reductant_present=False``, irrespective of temperature or gas composition, and
:data:`REDUCTANT_REQUIRED` records the rule with its source.

The chemistry behind the rule, for orientation (not itself a claim sourced from
this build's accessible literature): direct chlorination of a refractory oxide,

.. math:: \mathrm{MO_x}(s) + x\,\mathrm{Cl_2}(g)
          \rightarrow \mathrm{MCl_{2x}}(g) + \tfrac{x}{2}\,\mathrm{O_2}(g)

liberates oxygen, and the resulting :math:`\Delta G_r` is positive for the
strongly bound oxides of Ti, Al and B. Carbon consumes that oxygen as CO or CO2,

.. math:: \mathrm{MO_x}(s) + x\,\mathrm{Cl_2}(g) + x\,\mathrm{C}(s)
          \rightarrow \mathrm{MCl_{2x}}(g) + x\,\mathrm{CO}(g)

which adds roughly the free energy of CO formation per oxygen atom removed and
turns the reaction spontaneous. :func:`gibbs_of_reaction` computes either form
from a supplied thermochemical table, so the sign can be checked rather than
asserted, but see THERMOCHEMICAL DATA STATUS: this build ships no table capable
of that calculation for Ti, Al or B.

EQUATIONS
=========

Symbols
-------
=======================  ======================  ===========================
Symbol                   Unit                    Valid range
=======================  ======================  ===========================
:math:`p^{sat}`          Pa                      > 0
:math:`p_{ref}`          Pa                      101325 (1 atm reference)
:math:`T_b`              K                       > 0 (normal boiling point)
:math:`\Delta H_{vap}`   J mol^-1                > 0
:math:`T`                K                       > 0
:math:`\Delta G_r`       J mol^-1                any sign
:math:`\Delta H_r`       J mol^-1                any sign
:math:`\Delta S_r`       J mol^-1 K^-1           any sign
:math:`J_{evap}`         mol m^-2 s^-1           >= 0
:math:`M`                kg mol^-1               > 0
:math:`\alpha`           dimensionless           (0, 1] evaporation coefficient
=======================  ======================  ===========================

Clausius-Clapeyron, one-point form
----------------------------------
Derivation. The Clapeyron equation for a condensed-to-vapour transition with an
ideal vapour and negligible condensed-phase molar volume is
:math:`d\ln p/dT = \Delta H_{vap}/(\mathcal{R}T^2)`. Treating
:math:`\Delta H_{vap}` as constant and integrating from the normal boiling (or
sublimation) point, where :math:`p = p_{ref}` by definition:

.. math:: \boxed{p^{sat}(T) = p_{ref}\,\exp\!\left[-\frac{\Delta H_{vap}}
          {\mathcal{R}}\left(\frac{1}{T}-\frac{1}{T_b}\right)\right]}

Dimensional check: :math:`\Delta H_{vap}/\mathcal{R}` has unit K, multiplied by
:math:`1/T` gives a dimensionless exponent. At :math:`T = T_b` the exponent is
zero and :math:`p^{sat} = p_{ref}`, which is the boundary condition.

This is a two-parameter fit anchored on one measured point. It is accurate near
:math:`T_b` and degrades far from it, because :math:`\Delta H_{vap}` falls with
temperature and vanishes at the critical point. At roasting temperature
(1200 degC) extrapolated from a boiling point of 136 degC (TiCl4) the
extrapolation spans a factor of 3.5 in absolute temperature and the computed
pressure is far above the critical pressure, which is physically meaningless.
:func:`vapour_pressure` therefore CAPS the returned pressure at the system total
pressure and sets a ``supercritical_extrapolation`` flag, and
:func:`removal_screen` treats any species whose :math:`T` exceeds :math:`T_b` as
simply "fully volatile" rather than quoting a fictitious pressure. That is the
honest use of the relation: it discriminates volatile from non-volatile, and it
does not produce a meaningful number for a gas hundreds of degrees above its
boiling point.

Trouton's rule (used only where :math:`\Delta H_{vap}` is unsourced)
--------------------------------------------------------------------
.. math:: \Delta S_{vap} \approx 85\ \mathrm{J\,mol^{-1}\,K^{-1}}
          \quad\Rightarrow\quad \Delta H_{vap} \approx 85\,T_b

An empirical regularity, not a law, and it is poor for associated or ionic
liquids. Every value obtained this way is tagged ASSUMED with Trouton named in
the basis string, and :data:`CHLORIDES` marks which entries rely on it.

Gibbs energy of reaction
------------------------
.. math:: \Delta G_r(T) = \sum_i \nu_i \Delta H_{f,i}^{\circ}
          - T \sum_i \nu_i S_i^{\circ}

with :math:`\nu_i` positive for products and negative for reactants. This is the
second-law screen: :math:`\Delta G_r < 0` is necessary for spontaneity at the
stated standard states. Constant-:math:`C_p` corrections are NOT applied,
because applying a Kirchhoff correction with unsourced heat capacities would add
apparent rigour without adding information; the omission biases
:math:`\Delta G_r` by an amount that grows with :math:`T - 298` K and is stated
in LIMITATIONS.

Hertz-Knudsen evaporation flux
------------------------------
The maximum molar flux leaving a surface into a vacuum, from kinetic theory:

.. math:: J_{evap} = \alpha\,\frac{p^{sat} - p_{bulk}}{\sqrt{2\pi M \mathcal{R} T}}

Dimensional check: Pa / sqrt(kg mol^-1 J mol^-1 K^-1 K) = Pa /
sqrt(kg J mol^-2) = Pa / (sqrt(kg^2 m^2 s^-2 mol^-2)) = Pa mol s kg^-1 m^-1 =
mol m^-2 s^-1 after substituting Pa = kg m^-1 s^-2. The unit reduction is
enforced by a test in ``tests/test_chlorination.py`` rather than asserted here.
This gives an UPPER BOUND on the
removal rate: it assumes every molecule that leaves is swept away, so a real
roast with a finite gas sweep and a stagnant boundary layer is slower.

THERMOCHEMICAL DATA STATUS
==========================
NIST-JANAF (janaf.nist.gov) and the NIST Chemistry WebBook (webbook.nist.gov)
are both outside this sandbox's network allowlist and returned proxy errors on
2026-09-16. No request for access was made, per the task instruction to log the
attempt and continue. Standard formation enthalpies and entropies for the
oxide-to-chloride reactions of Al, Ti and B could therefore NOT be sourced.
Consequences, all visible in the code rather than papered over:

- :data:`THERMO` contains only the species whose formation data could be
  retrieved from PubChem (which aggregates HSDB and other compendia, each
  carrying its own reference). Entropies are almost entirely absent from that
  source, so :func:`gibbs_of_reaction` requires the caller to supply a complete
  table and raises when any species is missing.
- There is NO built-in oxide-to-chloride :math:`\Delta G_r` for Al, Ti or B in
  this build. The reductant requirement is enforced from the Liu et al. (2026)
  finding as a RULE, not recomputed from thermodynamics.
- :data:`CHLORIDES` boiling and sublimation points ARE sourced (PubChem), and
  the volatility screen based on them is the part of this module that rests on
  retrieved data.

LIMITATIONS
===========
1. No Gibbs screen for Ti, Al or B from first principles in this build (see
   above). :func:`gibbs_of_reaction` is working machinery with no data to feed
   it for those elements. Supplying a JANAF-derived table makes it live.
2. Vapour pressure far above the boiling point is meaningless. The
   Clausius-Clapeyron extrapolation from 136 degC (TiCl4) to 1200 degC is not a
   physical pressure, and the code caps it and flags it rather than reporting it.
   Do not read a capped pressure as a measurement.
3. Trouton's rule supplies :math:`\Delta H_{vap}` for AlCl3, FeCl3, NaCl, KCl
   and LiCl. AlCl3 vapour is dimeric (Al2Cl6) near its sublimation point, and
   the alkali chlorides are ionic melts, both cases where Trouton is known to be
   unreliable. Those pressures are order-of-magnitude indicators only.
4. Activity coefficients are ignored. Impurities in quartz are dilute, and their
   thermodynamic activity is not unity, so a screen run at unit activity
   overstates the driving force for chloride formation. The direction of the
   error is known (optimistic) but its size is not quantified here.
5. Alkali chloride volatility in the roasting window is marginal, not
   comfortable: NaCl boils at 1465 degC and KCl sublimes near 1500 degC, both
   ABOVE a 1200 degC roast. Removal proceeds by evaporation below the boiling
   point, at a rate this module bounds with Hertz-Knudsen but does not calibrate.
6. No kinetics of the surface reaction, no gas-film resistance, no pore
   diffusion in an agglomerated charge, and no accounting for chloride
   condensation in a cooler downstream zone (which is how AlCl3 fouls equipment
   in practice).
7. Silica chlorination is not modelled. SiCl4 forms from SiO2 under aggressive
   chlorination and represents a yield loss of the product itself, which a
   complete flowsheet model must include and this module does not.
8. The reductant rule is binary. Liu et al. (2026) report that a reductant is
   indispensable for Ti, Al and B; this module does not model reductant dosage,
   carbon type, or the resulting CO/CO2 ratio, so it cannot tell you HOW MUCH
   carbon to add.

References
----------
Liu, L., Liu, H., Li, J., Peng, T., Wang, W. and Wang, F. (2026) Mechanism Study
on Deep Removal of Lattice Impurities from High-Purity Quartz by Chlorination
Roasting, Minerals 16(8), 836, doi 10.3390/min16080836.

PubChem compound records (National Library of Medicine), accessed 2026-09-16 via
the PUG-View REST API: AlCl3 CID 24012, FeCl3 CID 24380, TiCl4 CID 24193, BCl3
CID 25135, SiCl4 CID 24816, NaCl CID 5234, KCl CID 4873, LiCl CID 433294. Each
property in :data:`CHLORIDES` carries the specific compendium string PubChem
attributes it to.

Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand by
Vein Quartz Purification and Characteristics: A Case Study of Pakistan Vein
Quartz, Minerals 14(7), 727, doi 10.3390/min14070727. Chlorination roasting is
one stage of the flowsheet whose endpoint this platform benchmarks against.
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import MISSING, Source, Tag, Tier, Value, _Missing
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = [
    "GAS_CONSTANT",
    "P_REFERENCE",
    "TROUTON_ENTROPY",
    "PhaseChange",
    "ChlorideSpecies",
    "CHLORIDES",
    "REDUCTANT_REQUIRED",
    "THERMO",
    "ScreenResult",
    "SOURCE_LIU_2026",
    "SOURCE_PUBCHEM",
    "trouton_enthalpy",
    "vapour_pressure",
    "gibbs_of_reaction",
    "hertz_knudsen_flux",
    "requires_reductant",
    "removal_screen",
    "screen_feedstock",
]

#: Molar gas constant, CODATA 2018 exact value.
GAS_CONSTANT: Final[Quantity] = Q_(8.314462618, "J/(mol*K)")

#: Standard atmosphere, the pressure at which a normal boiling point is defined.
P_REFERENCE: Final[Quantity] = Q_(101325.0, "Pa")

#: Trouton's rule entropy of vaporization. Empirical regularity, not a law.
TROUTON_ENTROPY: Final[Value] = Value(
    quantity=Q_(85.0, "J/(mol*K)"),
    tag=Tag.ASSUMED,
    basis="Trouton's rule, the empirical observation that the entropy of "
          "vaporization of many liquids at their normal boiling point clusters near "
          "85 J/(mol K). An approximation, not a measurement, and known to be poor for "
          "associated vapours (AlCl3 is dimeric as Al2Cl6) and for ionic melts (NaCl, "
          "KCl, LiCl). Used only where a measured enthalpy of vaporization could not be "
          "sourced, and every value so derived is flagged in CHLORIDES.",
    confidence="low",
)

SOURCE_LIU_2026: Final[Source] = Source(
    citation="Liu, L., Liu, H., Li, J., Peng, T., Wang, W. and Wang, F. (2026) Mechanism "
             "Study on Deep Removal of Lattice Impurities from High-Purity Quartz by "
             "Chlorination Roasting, Minerals 16(8), 836",
    tier=Tier.T1,
    doi="10.3390/min16080836",
    accessed=_dt.date(2026, 9, 16),
    extraction="manual",
    note="Abstract retrieved via CrossRef and Unpaywall metadata. Publisher full text "
         "(mdpi.com) returned HTTP 403 from this sandbox, so only quantities stated in "
         "the abstract are used.",
)

SOURCE_PUBCHEM: Final[Source] = Source(
    citation="PubChem compound records, National Library of Medicine, retrieved via the "
             "PUG-View REST API",
    tier=Tier.T1,
    url="https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/",
    accessed=_dt.date(2026, 9, 16),
    extraction="api",
    note="PubChem aggregates HSDB, USCG CHRIS, EPA and NTP compendia; the specific "
         "compendium string for each property is recorded in the property's basis.",
)


class PhaseChange(str, enum.Enum):
    """Whether the 1 atm transition point is a boiling point or a sublimation point."""

    BOILING = "boiling"
    SUBLIMATION = "sublimation"


class ChlorideSpecies(BaseModel):
    """One metal chloride, with the volatility data needed for the screen."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    formula: str
    cation: str
    molar_mass: Value
    transition_T: Value                       # normal boiling or sublimation point
    transition_kind: PhaseChange
    enthalpy_vaporization: Value | _Missing = MISSING
    enthalpy_is_trouton: bool = False
    pubchem_cid: int | None = None

    @model_validator(mode="after")
    def _physical(self) -> "ChlorideSpecies":
        t_b = float(self.transition_T.quantity.to("K").magnitude)
        if t_b <= 0.0:
            raise ValueError(f"{self.formula}: transition temperature must be positive")
        if isinstance(self.enthalpy_vaporization, Value):
            h = float(self.enthalpy_vaporization.quantity.to("J/mol").magnitude)
            if h <= 0.0:
                raise ValueError(
                    f"{self.formula}: enthalpy of vaporization must be positive; a "
                    f"non-positive value would make the vapour pressure fall with "
                    f"temperature, violating the second law"
                )
        return self

    def enthalpy(self) -> Quantity:
        """Enthalpy of vaporization, falling back to Trouton's rule if unsourced."""
        if isinstance(self.enthalpy_vaporization, Value):
            return self.enthalpy_vaporization.quantity.to("J/mol")
        return trouton_enthalpy(self.transition_T.quantity)


def trouton_enthalpy(transition_T: Quantity) -> Quantity:
    r"""Estimate :math:`\Delta H_{vap} \approx \Delta S_{Trouton} T_b`.

    Returns a quantity in J/mol. An estimate, never a measurement: see
    :data:`TROUTON_ENTROPY` for why it is unreliable for these species.
    """
    require_dimensionality(transition_T, "temperature", "transition_T")
    t_b = transition_T.to("K")
    if float(t_b.magnitude) <= 0.0:
        raise ValueError("transition temperature must be positive")
    return (TROUTON_ENTROPY.quantity * t_b).to("J/mol")


def _sourced_T(value_c: float, compendium: str) -> Value:
    """Build a SOURCED temperature Value attributed to a PubChem compendium string."""
    return Value(
        quantity=Q_(value_c, "degC"),
        tag=Tag.SOURCED,
        source=SOURCE_PUBCHEM,
        basis=f"PubChem property record, attributed there to: {compendium}",
        confidence="high",
    )


def _molar_mass(formula: str, mass: float) -> Value:
    return Value(
        quantity=Q_(mass, "g/mol"),
        tag=Tag.DERIVED,
        basis=f"sum of IUPAC 2021 standard atomic weights for {formula}, as tabulated in "
              f"ae.core.feedstock.MOLAR_MASS",
    )


#: Metal chlorides relevant to quartz purification, with sourced 1 atm transition
#: points. Molar masses are DERIVED from the IUPAC atomic weights already in
#: ae.core.feedstock. Enthalpies of vaporization are SOURCED only for TiCl4;
#: the rest fall back to Trouton and are flagged.
CHLORIDES: Final[dict[str, ChlorideSpecies]] = {
    "TiCl4": ChlorideSpecies(
        formula="TiCl4", cation="Ti",
        molar_mass=_molar_mass("TiCl4", 189.679),
        transition_T=_sourced_T(136.45, "boiling point 136.45 degC at 760 mm Hg"),
        transition_kind=PhaseChange.BOILING,
        enthalpy_vaporization=Value(
            quantity=Q_(36.2, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
            basis="PubChem 'Heat of Vaporization' property: 36.2 kJ/mol at 136.45 degC",
            confidence="high"),
        pubchem_cid=24193),
    "AlCl3": ChlorideSpecies(
        formula="AlCl3", cation="Al",
        molar_mass=_molar_mass("AlCl3", 133.341),
        transition_T=_sourced_T(180.0, "boiling point 180 degC at 760 mm Hg; PubChem also "
                                       "records sublimation at 182.7 degC at 752 mm Hg and "
                                       "'sublimes readily at 178 degC'"),
        transition_kind=PhaseChange.SUBLIMATION,
        enthalpy_is_trouton=True,
        pubchem_cid=24012),
    "FeCl3": ChlorideSpecies(
        formula="FeCl3", cation="Fe",
        molar_mass=_molar_mass("FeCl3", 162.204),
        transition_T=_sourced_T(316.0, "boiling point about 316 degC (decomposes); PubChem "
                                       "also records melting at 304 to 307.6 degC"),
        transition_kind=PhaseChange.SUBLIMATION,
        enthalpy_is_trouton=True,
        pubchem_cid=24380),
    "BCl3": ChlorideSpecies(
        formula="BCl3", cation="B",
        molar_mass=_molar_mass("BCl3", 117.17),
        transition_T=_sourced_T(12.5, "boiling point 12.5 degC at 760 mm Hg"),
        transition_kind=PhaseChange.BOILING,
        enthalpy_is_trouton=True,
        pubchem_cid=25135),
    "SiCl4": ChlorideSpecies(
        formula="SiCl4", cation="Si",
        molar_mass=_molar_mass("SiCl4", 169.897),
        transition_T=_sourced_T(57.57, "boiling point 57.57 degC at 760 mm Hg"),
        transition_kind=PhaseChange.BOILING,
        enthalpy_is_trouton=True,
        pubchem_cid=24816),
    "NaCl": ChlorideSpecies(
        formula="NaCl", cation="Na",
        molar_mass=_molar_mass("NaCl", 58.443),
        transition_T=_sourced_T(1465.0, "boiling point 1465 degC"),
        transition_kind=PhaseChange.BOILING,
        enthalpy_is_trouton=True,
        pubchem_cid=5234),
    "KCl": ChlorideSpecies(
        formula="KCl", cation="K",
        molar_mass=_molar_mass("KCl", 74.551),
        transition_T=_sourced_T(1500.0, "sublimes 1500 degC"),
        transition_kind=PhaseChange.SUBLIMATION,
        enthalpy_is_trouton=True,
        pubchem_cid=4873),
    "LiCl": ChlorideSpecies(
        formula="LiCl", cation="Li",
        molar_mass=_molar_mass("LiCl", 42.394),
        transition_T=_sourced_T(1360.0, "boiling point 1360 degC; PubChem also records "
                                        "1350 degC at 760 mm Hg and 1383 degC"),
        transition_kind=PhaseChange.BOILING,
        enthalpy_is_trouton=True,
        pubchem_cid=433294),
}

#: Cations whose chlorination is not spontaneous without a carbonaceous reductant.
#:
#: This is a SOURCED RULE, not a computed result: the thermochemical data needed
#: to derive it was unreachable in this build (see THERMOCHEMICAL DATA STATUS).
REDUCTANT_REQUIRED: Final[Value] = Value(
    quantity=Q_(1.0, "dimensionless"),
    tag=Tag.SOURCED,
    source=SOURCE_LIU_2026,
    basis="Liu et al. 2026 abstract: 'carbonaceous reductants are indispensable for "
          "enabling spontaneous chlorination of substitutional impurities such as Ti, Al, "
          "and B, while alkali metals including Na, K, and Li can be effectively removed "
          "under HCl atmosphere at moderate temperatures'. Encoded as the rule that Ti, "
          "Al and B are thermodynamically blocked when no reductant is present.",
    confidence="high",
)

#: Cations the rule applies to.
_REDUCTANT_REQUIRED_CATIONS: Final[frozenset[str]] = frozenset({"Ti", "Al", "B"})

#: Thermochemical data that could actually be retrieved. Deliberately sparse:
#: entropies are absent from the accessible source for nearly every species, so
#: gibbs_of_reaction cannot be run from this table alone and says so.
THERMO: Final[dict[str, dict[str, Value]]] = {
    "FeCl3(s)": {
        "Hf": Value(quantity=Q_(-399.5, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                    basis="PubChem 'Other Experimental Properties': standard molar "
                          "enthalpy of formation at 298.15 K, -399.5 kJ/mol (crystal)",
                    confidence="high"),
        "Gf": Value(quantity=Q_(-334.0, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                    basis="PubChem 'Other Experimental Properties': standard molar Gibbs "
                          "energy of formation at 298.15 K, -334.0 kJ/mol (crystal)",
                    confidence="high"),
    },
    "LiCl(s)": {
        "Hf": Value(quantity=Q_(-408.6, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                    basis="PubChem 'Other Experimental Properties': standard molar "
                          "enthalpy of formation at 298.15 K, -408.6 kJ/mol",
                    confidence="high"),
    },
    "NaCl(s)": {
        "Hf": Value(quantity=Q_(-410.9, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                    basis="PubChem property text: enthalpy of formation -410.9 kJ/mol at "
                          "25 degC",
                    confidence="medium"),
    },
    "KCl(s)": {
        "Hf": Value(quantity=Q_(-436.7, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                    basis="PubChem property text: enthalpy of formation -436.7 kJ/mol",
                    confidence="medium"),
        "S": Value(quantity=Q_(82.55, "J/(mol*K)"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                   basis="PubChem property text: entropy S = 82.55 J/mol-K",
                   confidence="medium"),
    },
    "CO2(g)": {
        "Hf": Value(quantity=Q_(-393.51, "kJ/mol"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                    basis="PubChem 'Other Experimental Properties': heat of formation "
                          "-393.51 kJ/mol",
                    confidence="high"),
        "S": Value(quantity=Q_(213.785, "J/(mol*K)"), tag=Tag.SOURCED, source=SOURCE_PUBCHEM,
                   basis="PubChem 'Other Experimental Properties': entropy 213.785 J/K-mol",
                   confidence="high"),
    },
}

#: Species whose formation data is needed for the oxide-to-chloride screen but
#: could not be sourced in this build, with what would supply it.
THERMO_MISSING: Final[dict[str, str]] = {
    "Al2O3(s)": "NIST-JANAF Thermochemical Tables (janaf.nist.gov) or NIST Chemistry "
                "WebBook (webbook.nist.gov), both outside this sandbox's network "
                "allowlist on 2026-09-16. PubChem returned a value labelled 'heat of "
                "formation -130.0 kJ/mol' for CID 9989226 which is inconsistent with the "
                "accepted magnitude for corundum (about -1676 kJ/mol) and was therefore "
                "NOT used.",
    "TiO2(s)": "NIST-JANAF or NIST WebBook (network-blocked). PubChem returned no "
               "formation enthalpy for CID 26042.",
    "B2O3(s)": "NIST-JANAF or NIST WebBook (network-blocked). PubChem returned no "
               "formation enthalpy for CID 518682.",
    "AlCl3(s)": "NIST-JANAF or NIST WebBook (network-blocked). PubChem returned no "
                "formation enthalpy for CID 24012.",
    "TiCl4(l)": "NIST-JANAF or NIST WebBook (network-blocked). PubChem returned no "
                "formation enthalpy for CID 24193.",
    "BCl3(g)": "NIST-JANAF or NIST WebBook (network-blocked).",
    "SiO2(s)": "NIST-JANAF or NIST WebBook (network-blocked). The PubChem string 'Heat of "
               "Formation = -903.2 kJ/mol' appears on an amorphous-silica record and its "
               "phase basis is ambiguous, so it was not used for alpha quartz.",
    "Cl2(g)": "Zero by definition as an element in its standard state, but the entropy "
              "(needed for the T*dS term) requires JANAF.",
    "CO(g)": "NIST-JANAF or NIST WebBook (network-blocked). PubChem gives the heat of "
             "formation only in kcal/mol from an unstated reference state.",
}


class ScreenResult(BaseModel):
    """Per-element outcome of the three-condition chlorination screen."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    element: str
    chloride: str
    thermodynamically_blocked: bool
    blocked_reason: str | None
    volatile: bool
    transition_T_C: float
    T_over_Tb: float
    vapour_pressure_Pa: float | None
    vapour_pressure_is_capped: bool
    enthalpy_is_trouton: bool
    transport_assessed_here: bool = False
    verdict: str


def vapour_pressure(
    species: ChlorideSpecies, temperature: Quantity, total_pressure: Quantity | None = None
) -> tuple[Quantity, bool]:
    r"""Saturation vapour pressure by the one-point Clausius-Clapeyron relation.

    Returns ``(pressure, capped)``. ``capped`` is True when the raw
    extrapolation exceeds the system total pressure, in which case the returned
    value is the total pressure and the species should be treated as fully
    volatile rather than as having that numeric pressure. See LIMITATIONS item 2:
    extrapolating hundreds of degrees above the boiling point does not produce a
    physical pressure.
    """
    require_dimensionality(temperature, "temperature", "temperature")
    p_tot = total_pressure if total_pressure is not None else P_REFERENCE
    require_dimensionality(p_tot, "pressure", "total_pressure")
    t = temperature.to("K")
    if float(t.magnitude) <= 0.0:
        raise ValueError("absolute temperature must be positive")
    t_b = species.transition_T.quantity.to("K")
    dh = species.enthalpy().to("J/mol")
    expo = float(
        (-dh / GAS_CONSTANT * (1.0 / t - 1.0 / t_b)).to("dimensionless").magnitude
    )
    # Guard the exponential: a 1000 K extrapolation can overflow a float.
    expo = min(expo, 700.0)
    p_raw = P_REFERENCE * math.exp(expo)
    assert float(p_raw.magnitude) >= 0.0, "vapour pressure cannot be negative"
    if float(p_raw.to("Pa").magnitude) > float(p_tot.to("Pa").magnitude):
        return p_tot.to("Pa"), True
    return p_raw.to("Pa"), False


def gibbs_of_reaction(
    stoichiometry: dict[str, float], temperature: Quantity,
    thermo: dict[str, dict[str, Value]] | None = None,
) -> Quantity:
    r"""Gibbs energy of reaction from formation enthalpies and absolute entropies.

    :math:`\Delta G_r(T) = \sum \nu_i \Delta H_{f,i} - T\sum \nu_i S_i`, with
    ``stoichiometry`` mapping species keys to signed coefficients (positive for
    products, negative for reactants).

    Raises
    ------
    KeyError
        If any species lacks ``Hf`` or ``S`` in the table. The error names the
        measurement that would supply it. No species is silently skipped and no
        value is estimated, because a Gibbs sign computed from a partial table
        is worse than no answer: it looks authoritative and is arbitrary.
    """
    require_dimensionality(temperature, "temperature", "temperature")
    table = thermo if thermo is not None else THERMO
    h_sum = Q_(0.0, "J/mol")
    s_sum = Q_(0.0, "J/(mol*K)")
    for species, nu in stoichiometry.items():
        if species not in table:
            hint = THERMO_MISSING.get(species, "no accessible source identified")
            raise KeyError(
                f"no thermochemical data for {species!r} in this build. Would be "
                f"supplied by: {hint}"
            )
        entry = table[species]
        for prop in ("Hf", "S"):
            if prop not in entry:
                hint = THERMO_MISSING.get(species, "NIST-JANAF or NIST WebBook")
                raise KeyError(
                    f"{species!r} has no {prop!r} in the supplied table, so dG cannot be "
                    f"computed at {temperature.to('K'):~P}. Would be supplied by: {hint}"
                )
        h_sum = h_sum + nu * entry["Hf"].quantity.to("J/mol")
        s_sum = s_sum + nu * entry["S"].quantity.to("J/(mol*K)")
    dg = h_sum - temperature.to("K") * s_sum
    return dg.to("kJ/mol")


def hertz_knudsen_flux(
    species: ChlorideSpecies, temperature: Quantity,
    partial_pressure_bulk: Quantity | None = None,
    evaporation_coefficient: float = 1.0,
    total_pressure: Quantity | None = None,
) -> Quantity:
    r"""Maximum molar evaporation flux from the Hertz-Knudsen relation.

    :math:`J = \alpha (p^{sat}-p_{bulk}) / \sqrt{2\pi M \mathcal{R} T}`.

    An UPPER BOUND on the removal rate: it assumes free molecular escape with no
    boundary-layer resistance, so a real roast is slower. Returns mol m^-2 s^-1.
    """
    require_dimensionality(temperature, "temperature", "temperature")
    alpha = require_fraction(evaporation_coefficient, "evaporation_coefficient", 1e-6, 1.0)
    p_bulk = partial_pressure_bulk if partial_pressure_bulk is not None else Q_(0.0, "Pa")
    require_dimensionality(p_bulk, "pressure", "partial_pressure_bulk")
    p_sat, _ = vapour_pressure(species, temperature, total_pressure)
    driving = (p_sat - p_bulk).to("Pa")
    if float(driving.magnitude) <= 0.0:
        return Q_(0.0, "mol/(m**2*s)")
    m = species.molar_mass.quantity.to("kg/mol")
    denom = (2.0 * math.pi * m * GAS_CONSTANT * temperature.to("K")) ** 0.5
    flux = alpha * driving / denom
    out = flux.to("mol/(m**2*s)")
    assert float(out.magnitude) >= 0.0, "evaporation flux cannot be negative"
    return out


def requires_reductant(element: str) -> bool:
    """Whether chlorination of ``element`` needs a carbonaceous reductant.

    True for Ti, Al and B, per Liu et al. (2026). See :data:`REDUCTANT_REQUIRED`.
    """
    return element in _REDUCTANT_REQUIRED_CATIONS


def removal_screen(
    element: str,
    temperature: Quantity,
    reductant_present: bool,
    total_pressure: Quantity | None = None,
    volatility_margin: float = 1.0,
) -> ScreenResult:
    """Screen one element against the thermodynamic and volatility conditions.

    Transport (condition 3) is NOT assessed here: a lattice-substituted cation
    must first reach a gas-accessible surface, which is a solid-state diffusion
    question handled by :mod:`ae.physics.diffusion`. ``transport_assessed_here``
    is always False to keep that gap explicit in the returned record.

    ``volatility_margin`` is the ratio of temperature to the chloride's 1 atm
    transition temperature required to call the chloride volatile. The default
    of 1.0 means the roast must at least reach the boiling or sublimation point.
    """
    require_dimensionality(temperature, "temperature", "temperature")
    chloride = next(
        (c for c in CHLORIDES.values() if c.cation == element), None
    )
    if chloride is None:
        raise KeyError(
            f"no chloride species tabulated for {element!r}; tabulated cations: "
            f"{sorted({c.cation for c in CHLORIDES.values()})}"
        )
    t_k = float(temperature.to("K").magnitude)
    t_b_k = float(chloride.transition_T.quantity.to("K").magnitude)
    ratio = t_k / t_b_k

    blocked = requires_reductant(element) and not reductant_present
    reason = None
    if blocked:
        reason = (
            f"{element} is substitutional and its oxide-to-chloride conversion is not "
            f"spontaneous without a carbonaceous reductant (Liu et al. 2026, doi "
            f"10.3390/min16080836). No reductant was specified, so chlorination does not "
            f"reach {element} at any temperature in this screen."
        )

    p_sat, capped = vapour_pressure(chloride, temperature, total_pressure)
    volatile = ratio >= volatility_margin

    if blocked:
        verdict = (
            f"NOT REMOVED: thermodynamically blocked. {chloride.formula} would be "
            f"volatile at this temperature (T/Tb = {ratio:.2f}) but it never forms "
            f"without a reductant, so volatility is irrelevant."
        )
    elif not volatile:
        verdict = (
            f"PARTIAL AT BEST: {chloride.formula} is below its 1 atm transition point "
            f"(T/Tb = {ratio:.2f}, transition at "
            f"{float(chloride.transition_T.quantity.to('degC').magnitude):.0f} degC). "
            f"Removal proceeds only by sub-boiling evaporation, whose rate is bounded by "
            f"hertz_knudsen_flux and is not calibrated here."
        )
    else:
        verdict = (
            f"CHEMISTRY AND VOLATILITY PERMIT REMOVAL: {chloride.formula} forms and is "
            f"above its 1 atm transition point (T/Tb = {ratio:.2f}). This is NOT a "
            f"removal prediction: if the {element} is lattice-substituted it must still "
            f"diffuse to a gas-accessible surface, which ae.physics.diffusion assesses "
            f"separately and which is the binding constraint for Ti."
        )

    return ScreenResult(
        element=element,
        chloride=chloride.formula,
        thermodynamically_blocked=blocked,
        blocked_reason=reason,
        volatile=volatile,
        transition_T_C=float(chloride.transition_T.quantity.to("degC").magnitude),
        T_over_Tb=ratio,
        vapour_pressure_Pa=float(p_sat.to("Pa").magnitude),
        vapour_pressure_is_capped=capped,
        enthalpy_is_trouton=chloride.enthalpy_is_trouton,
        transport_assessed_here=False,
        verdict=verdict,
    )


def screen_feedstock(
    feedstock: Feedstock,
    temperature: Quantity,
    reductant_present: bool,
    elements: tuple[str, ...] | None = None,
    total_pressure: Quantity | None = None,
) -> dict[str, ScreenResult]:
    """Run :func:`removal_screen` for every measured impurity of ``feedstock``.

    Elements default to those actually assayed in the feedstock's impurity
    profile and tabulated in :data:`CHLORIDES`, so an unassayed element produces
    no result rather than a default one.
    """
    tabulated = {c.cation for c in CHLORIDES.values()}
    if elements is None:
        elements = tuple(
            e for e in feedstock.impurities.total if e in tabulated
        )
        if not elements:
            raise KeyError(
                f"feedstock {feedstock.sample_id} has no assayed element with a "
                f"tabulated chloride (assayed: {sorted(feedstock.impurities.total)}, "
                f"tabulated: {sorted(tabulated)}); a chlorination screen on an "
                f"unassayed ore would be a scenario with no input"
            )
    return {
        el: removal_screen(el, temperature, reductant_present, total_pressure)
        for el in elements
    }
