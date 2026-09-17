r"""SiO2 polymorph stability expressed as PROCESS WINDOWS, not as a phase diagram.

Purpose
-------
A phase diagram tells you which polymorph is thermodynamically stable. A kiln tells you
which polymorph you actually get, and the two differ by hundreds of degrees for silica
because the reconstructive transitions (quartz to tridymite to cristobalite) break Si-O
bonds and are kinetically obstructed, while the displacive alpha to beta inversion is
instantaneous and unavoidable. This module reports both, labelled, and never lets a
thermodynamic boundary be read as a process prediction.

EQUATIONS
---------
(1) Landau order parameter for the alpha to beta quartz inversion (tricritical, Holland
    and Powell 1998/2011 formalism):

    .. math::

        Q(T) = \left(\frac{T_c - T}{T_{c,0}}\right)^{1/4} \quad (T < T_c),
        \qquad Q(T) = 0 \quad (T \ge T_c)

    with the pressure dependence of the critical temperature

    .. math::

        T_c = T_{c,0} + \frac{V_D\,(P - P_0)}{S_D}

    Symbols:
      Q          order parameter, dimensionless, range [0, 1] at P_0 for T in [0, T_c].
      T          temperature, K, valid range 298 to 1200 K for quartz in this module.
      T_c        critical temperature at pressure P, K.
      T_{c,0}    critical temperature at the reference pressure P_0, 847 K for quartz.
      P          pressure, Pa, valid range 1e4 to 1e9 Pa (the HP Landau form is calibrated
                 for petrological pressures; this module is used at P_0 in practice).
      P_0        reference pressure, 1e5 Pa.
      V_D        volume of disordering, 1.188e-6 m^3/mol for quartz.
      S_D        entropy of disordering, 4.95 J/(mol K) for quartz.

    Derivation: the HP tricritical Landau expansion writes the excess Gibbs energy of the
    ordered phase relative to the disordered one as
    G_ex = S_D [ T_{c,0}(Q_0^2 - Q_0^6/3) - (T_c Q^2 - T_{c,0} Q^6/3) - T (Q_0^2 - Q^2) ]
    + (P - P_0) V_D Q_0^2. Minimising G_ex with respect to Q at fixed T and P gives the
    Q(T) above (see Holland and Powell 2011, doi 10.1111/j.1525-1314.2010.00923.x, and the
    calibration of the quartz transition in Carpenter et al. 1998,
    doi 10.2138/am-1998-1-201).

(2) Excess (transition-related) molar volume, which is the quantity that cracks grains:

    .. math::

        V_{ex}(T) = V_D\,Q(T)^2, \qquad
        \varepsilon_{V}(T_1 \to T_2) = \frac{V_{ex}(T_1) - V_{ex}(T_2)}{V_0}

    Symbols:
      V_{ex}          excess molar volume attributable to the ordering, m^3/mol.
      V_0             molar volume of the endmember at 298.15 K and P_0, 2.269e-5 m^3/mol.
      \varepsilon_V   volumetric strain, dimensionless, positive on heating.

    At P_0, Q_0 = Q(298.15 K) = ((847 - 298.15)/847)^{1/4} = 0.89721, so the CUMULATIVE
    transition-related expansion from 298.15 K to any T above T_c is
    V_D Q_0^2 / V_0 = 1.188e-6 x 0.80498 / 2.269e-5 = 0.04215, i.e. 4.21 vol%.

(3) Relative rate of a thermally activated reconstructive transformation, Arrhenius ratio:

    .. math::

        \frac{k(T_2)}{k(T_1)} = \exp\!\left[-\frac{E_a}{R}
        \left(\frac{1}{T_2} - \frac{1}{T_1}\right)\right]

    Symbols:
      k       rate constant of the JMAK kinetic law, units 1/s^n, never used absolutely
              here (see LIMITATIONS: no pre-exponential factor could be sourced).
      E_a     apparent activation energy, J/mol. 555 +/- 24 kJ/mol for cristobalite
              formation from sintered silica glass (Breneman and Halloran 2014,
              doi 10.1111/jace.12889), calibrated over 1200 to 1350 degC only.
      R       8.31446261815324 J/(mol K).
      T_1,T_2 temperatures, K, valid range 1473 to 1623 K by calibration of E_a.

    Derivation: dividing k = A exp(-E_a/(R T)) at two temperatures cancels A, which is why
    a ratio can be reported from a source that gives E_a without A.

(4) JMAK (Johnson-Mehl-Avrami-Kolmogorov) transformed fraction, provided for completeness
    and requiring a user-supplied rate constant:

    .. math::

        X(t) = 1 - \exp\left[-(k\,t)^{n}\right]

    Symbols:
      X   transformed volume fraction, dimensionless, range [0, 1].
      t   hold time, s, range 0 to 1e6 s.
      k   rate constant, 1/s. NOT DEFAULTED: see LIMITATIONS.
      n   Avrami exponent, dimensionless. 3.0 +/- 0.6 for cristobalite from sintered
          silica (Breneman and Halloran 2014), consistent with three-dimensional growth from a
          constant nucleus population.

WHAT THE LITERATURE ACTUALLY SUPPORTS, AND WHERE THIS BRIEF WAS WRONG
--------------------------------------------------------------------
The platform brief states the alpha to beta inversion carries "roughly 3.7 vol%
expansion". That number is not the discontinuity at 573 degC. Two distinct quantities are
conflated in the trade literature:

  (a) The CUMULATIVE transition-related expansion of alpha-quartz on heating from room
      temperature to the inversion. The HP2011 Landau calibration gives 4.21 vol% (see
      equation 2 above). Values quoted in the 3.5 to 4.5 vol% band, including the brief's
      3.7 vol%, belong to this quantity.
  (b) The STEP change at the inversion itself, which Ringdalen 2015
      (doi 10.1007/s11837-014-1149-y) reports as "an increase in volume of around 0.4%".

Both are real and they differ by an order of magnitude. This module returns (a) from
:func:`alpha_beta_cumulative_volume_strain` and (b) from :data:`TRANSITIONS`, each
separately labelled, because a thermal-shock model fed the wrong one is wrong by 10x.

The mechanistic claim in the brief, that the alpha to beta inversion is "what cracks
inclusion trails and is why calcination works", is only partly supported. The much larger
strain available on heating is the quartz to cristobalite conversion: Ringdalen 2015
measured up to 37% volume expansion across her sample set and found that specific surface
area rose with cristobalite content, attributing it to "cracking when the volume
increases". The evidence for the 0.4% inversion step being the dominant cracking mechanism
is weak; this module therefore reports strains and does not assert a cracking mechanism.

LIMITATIONS
-----------
1. NO ABSOLUTE KINETICS. :func:`jmak_fraction` requires a rate constant because no source
   reachable from this environment reports a pre-exponential factor for the quartz to
   cristobalite transformation. Breneman and Halloran 2014 report n and E_a but not A. Do not
   substitute a guessed A: Ringdalen 2015 measured 1% to 76% cristobalite after one hour
   at 1500 degC ACROSS DIFFERENT NATURAL QUARTZ SAMPLES, a 76-fold spread at fixed
   temperature and time, which means any single A is a property of one ore and not of
   SiO2. Absolute transformation kinetics for a specific deposit must be measured.
2. E_a EXTRAPOLATION. The 555 kJ/mol value is calibrated on 1200 to 1350 degC data for
   SINTERED SILICA GLASS, not for natural quartz, and Breneman and Halloran report the temperature
   of maximum rate as 1500 to 1600 degC. A rate ratio evaluated above roughly 1500 degC
   therefore extrapolates past a rate maximum and will overstate the rate increase. Above
   1623 K this module raises rather than extrapolate.
3. TRIDYMITE MAY NOT APPEAR AT ALL. The 870 degC boundary is a thermodynamic equilibrium.
   Ringdalen 2015 found "the tridymite phase expected from the phase diagram" absent in
   her heat-treated natural quartz, consistent with the long-standing view that tridymite
   stability requires alkali or other mineralisers. A schedule crossing 870 degC should be
   read as permitting tridymite, not producing it.
4. PRESSURE. Every transition temperature here is for P_0 = 1 bar. The Landau T_c
   pressure term is implemented but not validated in this module; coesite and stishovite
   are out of scope entirely.
5. ORDINAL RISK ONLY. :func:`devitrification_risk` returns an ordinal level built from a
   temperature window, a hold time and a contact material. It is a design screen, not a
   quantitative prediction of cristobalite fraction, and it has not been validated against
   a held-out dataset because no such dataset was reachable.
6. CONTACT-MATERIAL ONSET IS ONE EXPERIMENT. The 1000 degC glass onset and the material
   ranking come from Warden et al. 2024 (doi 10.52825/siliconpv.v2i.1311), a single study
   at one temperature (1500 degC) for one duration (3 h) on one commercial crucible glass.
   The direction of the effect is credible, the magnitudes are not general.

TYPE CHECKING NOTE
------------------
``mypy --strict`` reports ``Quantity? has no attribute "to"`` and ``Variable
"ae.core.units.Quantity" is not valid as a type`` against this module. Those errors
originate in ``ae.core.units``, which declares ``Quantity = UREG.Quantity`` (a variable
binding, not a type alias), so mypy cannot treat it as a type. The same errors appear
against the four core modules themselves (46 of them), and the core modules are not ours
to change. Every annotation here follows the core convention deliberately rather than
diverging from it for a clean checker run.

References
----------
Every entry below was resolved against the Crossref REST API for the DOI shown,
and the fields here (author list, year, title, journal, volume, issue, pages)
are as Crossref returned them. Resolved 17 September 2026. Entries without a
DOI say what was checked instead and what remains unverified.

Holland, T. J. B. and Powell, R. (2011) An improved and extended internally
consistent thermodynamic dataset for phases of petrological interest, involving
a new equation of state for solids, Journal of Metamorphic Geology 29(3),
333-383, doi 10.1111/j.1525-1314.2010.00923.x (dataset ds62). Source of the
Landau formalism used for equations (1) and (2) and of every Landau coefficient
in :data:`QUARTZ_LANDAU`.

Holland, T. J. B. and Powell, R. (1998) An internally consistent thermodynamic
data set for phases of petrological interest, Journal of Metamorphic Geology
16(3), 309-343, doi 10.1111/j.1525-1314.1998.00140.x. The earlier dataset of the
same pair, cited only for the lineage of the Landau treatment. The in-text
phrase "Holland and Powell 1998/2011 formalism" refers to these two papers; no
numerical value in this module comes from the 1998 paper.

Carpenter, M. A., Salje, E. K. H., Graeme-Barber, A., Wruck, B., Dove, M. T. and
Knight, K. S. (1998) Calibration of excess thermodynamic properties and elastic
constant variations associated with the alpha to beta phase transition in
quartz, American Mineralogist 83(1-2), 2-22, doi 10.2138/am-1998-1-201. Cited
for the tricritical character of the inversion, not for numerical input.

Ringdalen, E. (2015) Changes in Quartz During Heating and the Possible Effects
on Si Production, JOM 67(2), 484-492, doi 10.1007/s11837-014-1149-y. Source of
the 0.4 percent step volume change at the inversion, of the up-to-37 percent
expansion on cristobalite conversion, of the 1 to 76 percent cristobalite spread
after one hour at 1500 degC across natural samples, and of the absence of
tridymite in her heat-treated material. The year is the February 2015 print
issue; Crossref's issued date is the 2014-10-10 online-first publication, which
is why the DOI contains -014-.

Breneman, R. C. and Halloran, J. W. (2014) Kinetics of Cristobalite Formation in
Sintered Silica, Journal of the American Ceramic Society 97(7), 2272-2278,
doi 10.1111/jace.12889. Source of the Avrami exponent n = 3.0 +/- 0.6, the
apparent activation energy E_a = 555 +/- 24 kJ/mol, the 1200 to 1350 degC
calibration window, and the 1500 to 1600 degC rate maximum, all taken from the
published abstract. No pre-exponential factor is reported there, which is why
:func:`jmak_fraction` has no default rate constant. THIS ENTRY WAS PREVIOUSLY
WRONG: it named "Zhang, Y., Mavrogenes, J.A. and others" with pages 3378-3384,
written from memory. Neither Zhang nor Mavrogenes is an author, and the page
range was wrong.

Bourova, E. and Richet, P. (1998) Quartz and Cristobalite: high-temperature cell
parameters and volumes of fusion, Geophysical Research Letters 25(13),
2333-2336, doi 10.1029/98gl01581. Source of the cristobalite and quartz
high-temperature volume data used for the liquid-phase transition entry. THIS
ENTRY WAS PREVIOUSLY WRONG: it named "Wenk, H.-R., Brunini, A. and others" with
pages 2261-2264, written from memory. Neither is an author.

Warden, G. K., Gawel, B. A., Juel, M., Erbe, A. and Di Sabatino, M. (2024)
Cristobalite Formation in Fused Quartz Crucibles for Czochralski Silicon
Production in Different Conditions, SiliconPV Conference Proceedings 2,
doi 10.52825/siliconpv.v2i.1311. Source of the contact-material ranking and the
1000 degC glass onset. One study, one temperature, one duration, one commercial
crucible glass: the direction is credible, the magnitudes are not general.
Crossref returns no page range for this article.
"""

from __future__ import annotations

import enum
import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.units import (
    Q_,
    REGISTRY_QUANTITY,
    DimensionalityError,
    Quantity,
    require_dimensionality,
)

__all__ = [
    "QUARTZ_LANDAU",
    "R_GAS",
    "SRC_BOUROVA",
    "SRC_BRENEMAN",
    "SRC_CARPENTER",
    "SRC_HP2011",
    "SRC_RINGDALEN",
    "SRC_WARDEN",
    "TRANSITIONS",
    "ContactMaterial",
    "DevitrificationAssessment",
    "LandauParameters",
    "Polymorph",
    "RiskLevel",
    "ScheduleSegment",
    "ThermalSchedule",
    "Transition",
    "TransitionCharacter",
    "alpha_beta_cumulative_volume_strain",
    "arrhenius_rate_ratio",
    "cristobalite_onset_temperature",
    "crossed_transitions",
    "devitrification_risk",
    "excess_molar_volume",
    "jmak_fraction",
    "landau_critical_temperature",
    "order_parameter",
    "require_molar_volume",
]

#: Molar gas constant, J/(mol K). CODATA 2018 exact value (SI redefinition 2019).
R_GAS: Final[float] = 8.31446261815324

#: Reference unit for molar volume. ``ae.core.units._REF_UNITS`` has no molar-volume kind
#: and the core modules are not ours to extend, so the check lives here and follows the
#: same contract as ``require_dimensionality``: TypeError on a bare float, DimensionalityError
#: on the wrong kind.
_MOLAR_VOLUME_REF: Final[str] = "m**3/mol"


def require_molar_volume(quantity: object, name: str) -> Quantity:
    """Assert that ``quantity`` is a molar volume, returning it unchanged.

    Mirrors :func:`ae.core.units.require_dimensionality` for a kind the core registry does
    not define. Raises TypeError for a bare number and DimensionalityError for the wrong
    dimension, so the guard behaves identically to the core checks in tests.
    """
    if not isinstance(quantity, REGISTRY_QUANTITY):
        raise TypeError(
            f"{name} must be a pint Quantity with molar volume dimensions "
            f"(e.g. Q_(2.269e-5, 'm**3/mol')), got {type(quantity).__name__}"
        )
    ref = Q_(1.0, _MOLAR_VOLUME_REF)
    if quantity.dimensionality != ref.dimensionality:
        raise DimensionalityError(
            quantity.units, ref.units, extra_msg=f" while checking {name!r} as molar volume"
        )
    return quantity

_ACCESSED: Final = __import__("datetime").date(2026, 9, 16)

SRC_HP2011: Final[Source] = Source(
    citation=(
        "Holland, T.J.B. and Powell, R. 2011, An improved and extended internally "
        "consistent thermodynamic dataset for phases of petrological interest, involving a "
        "new equation of state for solids, Journal of Metamorphic Geology 29:333-383 "
        "(dataset ds62)"
    ),
    tier=Tier.T1,
    doi="10.1111/j.1525-1314.2010.00923.x",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Endmember parameters read from the machine-readable ds62 transcription shipped in "
        "burnman 2.1.0 (minerals/HP_2011_ds62.py, autogenerated from tc-ds62.txt), because "
        "webbook.nist.gov and the publisher PDF are both unreachable from this sandbox. "
        "Values are stated there in SI units."
    ),
)

SRC_CARPENTER: Final[Source] = Source(
    citation=(
        "Carpenter, M.A., Salje, E.K.H., Graeme-Barber, A., Wruck, B., Dove, M.T. and "
        "Knight, K.S. 1998, Calibration of excess thermodynamic properties and elastic "
        "constant variations associated with the alpha-beta phase transition in quartz, "
        "American Mineralogist 83:2-22"
    ),
    tier=Tier.T1,
    doi="10.2138/am-1998-1-201",
    accessed=_ACCESSED,
    note="Cited for the tricritical character of the inversion, not for numerical input.",
)

SRC_RINGDALEN: Final[Source] = Source(
    citation=(
        "Ringdalen, E. 2015, Changes in Quartz During Heating and the Possible Effects on "
        "Si Production, JOM 67(2):484-492"
    ),
    tier=Tier.T1,
    doi="10.1007/s11837-014-1149-y",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Sole author Eli Ringdalen, confirmed against Crossref for this DOI. The year is "
        "the FEBRUARY 2015 print issue (JOM 67(2)); Crossref's issued date is the "
        "2014-10-10 online-first publication, which is why the DOI carries -014-. Both "
        "years refer to the same paper and 2015 is used throughout this module."
    ),
)

SRC_WARDEN: Final[Source] = Source(
    citation=(
        "Warden, G.K., Gawel, B.A., Juel, M., Erbe, A. and Di Sabatino, M. 2024, "
        "Cristobalite Formation in Fused Quartz Crucibles for Czochralski Silicon "
        "Production in Different Conditions, SiliconPV Conference Proceedings vol 2"
    ),
    tier=Tier.T2,
    doi="10.52825/siliconpv.v2i.1311",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Peer-reviewed conference proceedings, treated as T2 rather than T1 because it is "
        "a single-condition experiment (1500 degC, 3 h, one commercial crucible glass). "
        "Author list verified against the Crossref and OpenAlex records for this DOI, "
        "which agree exactly; the archived PDF yielded no extractable author text."
    ),
)

SRC_BRENEMAN: Final[Source] = Source(
    citation=(
        "Breneman, R. C. and Halloran, J. W. 2014, Kinetics of Cristobalite Formation in "
        "Sintered Silica, Journal of the American Ceramic Society 97(7):2272-2278"
    ),
    tier=Tier.T1,
    doi="10.1111/jace.12889",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "n = 3.0 +/- 0.6 and E_a = 555 +/- 24 kJ/mol taken from the published abstract, "
        "which states both explicitly; the full text is paywalled from this sandbox and no "
        "pre-exponential factor was obtainable. AUTHORS AND PAGES CORRECTED: this entry "
        "read \"Zhang, Y., Mavrogenes, J.A. and others 2014 ... 97:3378-3384\" until it "
        "was resolved against Crossref. Neither Zhang nor Mavrogenes is an author and the "
        "page range was wrong; both were written from memory and never checked."
    ),
)

SRC_BOUROVA: Final[Source] = Source(
    citation=(
        "Bourova, E. and Richet, P. 1998, Quartz and Cristobalite: high-temperature cell "
        "parameters and volumes of fusion, Geophysical Research Letters 25(13):2333-2336"
    ),
    tier=Tier.T1,
    doi="10.1029/98gl01581",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Melting temperature of quartz adjusted to 1673 K and volumes of fusion of "
        "beta-cristobalite and quartz of -0.1 and 3.85 cm^3/mol taken from the published "
        "abstract; full text paywalled from this sandbox."
    ),
)


class Polymorph(str, enum.Enum):
    """SiO2 phases this platform models at ambient pressure.

    ``ALPHA_QUARTZ`` and ``BETA_QUARTZ`` are the two sides of a single displacive
    inversion and share one thermodynamic endmember plus a Landau correction, which is why
    :mod:`ae.physics.thermal` keys its Cp coefficients on ``QUARTZ`` rather than on the two
    separately.
    """

    ALPHA_QUARTZ = "alpha_quartz"
    BETA_QUARTZ = "beta_quartz"
    QUARTZ = "quartz"
    TRIDYMITE = "tridymite"
    CRISTOBALITE = "cristobalite"
    SILICA_LIQUID = "silica_liquid"
    SILICA_GLASS = "silica_glass"
    AMORPHOUS = "amorphous"


class TransitionCharacter(str, enum.Enum):
    """Mechanism class, which decides whether kinetics may be ignored.

    DISPLACIVE transitions rotate bonds without breaking them: they are effectively
    instantaneous, cannot be suppressed by fast heating, and are fully reversible.
    RECONSTRUCTIVE transitions break Si-O bonds: they are slow, often bypassed entirely,
    and on cooling they usually do not reverse, which is why cristobalite survives to room
    temperature in a fired product.
    """

    DISPLACIVE = "displacive"
    RECONSTRUCTIVE = "reconstructive"
    MELTING = "melting"


class ContactMaterial(str, enum.Enum):
    """Material in contact with silica during a hold, which changes the onset."""

    NONE = "none"
    GRAPHITE = "graphite"
    SILICON_CARBIDE = "silicon_carbide"
    ALUMINA = "alumina"


class RiskLevel(str, enum.Enum):
    """Ordinal devitrification risk. Not a probability, not a fraction."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class LandauParameters(BaseModel):
    """Tricritical Landau parameters for a displacive inversion."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    tc_0: Value
    s_d: Value
    v_d: Value
    v_0: Value
    p_0: Value

    @model_validator(mode="after")
    def _dimensions(self) -> LandauParameters:
        require_dimensionality(self.tc_0.quantity, "temperature", "tc_0")
        require_dimensionality(self.s_d.quantity, "heat_capacity_molar", "s_d")
        require_molar_volume(self.v_d.quantity, "v_d")
        require_molar_volume(self.v_0.quantity, "v_0")
        require_dimensionality(self.p_0.quantity, "pressure", "p_0")
        if self.s_d.magnitude <= 0.0:
            raise ValueError("entropy of disordering must be positive (second law)")
        if self.v_0.magnitude <= 0.0:
            raise ValueError("molar volume must be positive")
        return self


def _hp(quantity: Quantity, field: str) -> Value:
    """Wrap a Holland and Powell ds62 endmember parameter as a SOURCED Value."""
    return Value(
        quantity=quantity,
        tag=Tag.SOURCED,
        source=SRC_HP2011,
        confidence="high",
        basis=f"ds62 endmember parameter {field}",
    )


#: Quartz alpha to beta inversion, Holland and Powell ds62 ``landau_hp`` modifier.
#: T_c0 = 847 K is 573.85 degC, which is the textbook 573 degC inversion.
QUARTZ_LANDAU: Final[LandauParameters] = LandauParameters(
    tc_0=_hp(Q_(847.0, "K"), "Tc_0 (quartz)"),
    s_d=_hp(Q_(4.95, "J/(mol*K)"), "S_D (quartz)"),
    v_d=_hp(Q_(1.188e-6, "m**3/mol"), "V_D (quartz)"),
    v_0=_hp(Q_(2.269e-5, "m**3/mol"), "V_0 (quartz)"),
    p_0=_hp(Q_(1.0e5, "Pa"), "P_0 (reference pressure)"),
)


class Transition(BaseModel):
    """One SiO2 transition, with its character and its strain.

    ``volume_strain`` is the strain of the transition ITSELF (the step), not the cumulative
    expansion of the low-temperature phase on the way to it. See the module docstring for
    why the distinction matters.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    from_phase: Polymorph
    to_phase: Polymorph
    temperature: Value
    character: TransitionCharacter
    reversible_on_cooling: bool
    volume_strain: Value | None = None
    equilibrium_only: bool = Field(
        default=False,
        description="True when the boundary is thermodynamic and routinely not realised "
        "in practice, e.g. tridymite in mineraliser-free natural quartz.",
    )
    note: str | None = None

    @model_validator(mode="after")
    def _dimensions(self) -> Transition:
        require_dimensionality(self.temperature.quantity, "temperature", f"{self.name}.T")
        if self.temperature.quantity.to("K").magnitude <= 0.0:
            raise ValueError(f"{self.name}: absolute temperature must be positive")
        return self

    @property
    def temperature_k(self) -> float:
        """Transition temperature in kelvin."""
        return float(self.temperature.quantity.to("K").magnitude)


def _sourced(q: Quantity, src: Source, basis: str) -> Value:
    return Value(quantity=q, tag=Tag.SOURCED, source=src, basis=basis, confidence="high")


#: The canonical ambient-pressure transition set.
TRANSITIONS: Final[tuple[Transition, ...]] = (
    Transition(
        name="alpha_beta_quartz_inversion",
        from_phase=Polymorph.ALPHA_QUARTZ,
        to_phase=Polymorph.BETA_QUARTZ,
        temperature=_sourced(
            Q_(847.0, "K"),
            SRC_HP2011,
            "ds62 Landau Tc_0 for quartz, 847 K = 573.85 degC, matching the textbook "
            "573 degC inversion",
        ),
        character=TransitionCharacter.DISPLACIVE,
        reversible_on_cooling=True,
        volume_strain=_sourced(
            Q_(0.004, "dimensionless"),
            SRC_RINGDALEN,
            "STEP change at the inversion only: Ringdalen 2015 reports the alpha to beta "
            "transformation at 573 degC gives an increase in volume of around 0.4 percent. "
            "The cumulative 3.5 to 4.5 percent figure quoted in trade literature is a "
            "different quantity, returned by alpha_beta_cumulative_volume_strain().",
        ),
        note=(
            "Instantaneous and unavoidable. Cannot be suppressed by heating rate and "
            "reverses on every cooling cycle, so it loads a grain once per thermal cycle."
        ),
    ),
    Transition(
        name="beta_quartz_to_tridymite",
        from_phase=Polymorph.BETA_QUARTZ,
        to_phase=Polymorph.TRIDYMITE,
        temperature=_sourced(
            Q_(1143.15, "K"),
            SRC_RINGDALEN,
            "870 degC, quoted by Ringdalen 2015 from the SiO2 phase diagram as the "
            "temperature at which tridymite will be formed",
        ),
        character=TransitionCharacter.RECONSTRUCTIVE,
        reversible_on_cooling=False,
        equilibrium_only=True,
        note=(
            "Ringdalen 2015 did not observe the tridymite phase expected from the phase "
            "diagram in heat-treated natural quartz. Treat as permitted, not produced; "
            "tridymite formation is generally held to require alkali mineralisers."
        ),
    ),
    Transition(
        name="tridymite_to_cristobalite",
        from_phase=Polymorph.TRIDYMITE,
        to_phase=Polymorph.CRISTOBALITE,
        temperature=_sourced(
            Q_(1743.15, "K"),
            SRC_WARDEN,
            "1470 degC, stated by Warden et al. 2024 as the temperature above which "
            "cristobalite forms in pure quartz",
        ),
        character=TransitionCharacter.RECONSTRUCTIVE,
        reversible_on_cooling=False,
        note=(
            "Ringdalen 2015 measured cristobalite already at 1400 degC and incomplete "
            "conversion after 6 h at 1600 degC in natural quartz, i.e. the practical "
            "window straddles this equilibrium boundary in both directions."
        ),
    ),
    Transition(
        name="silica_melting",
        from_phase=Polymorph.CRISTOBALITE,
        to_phase=Polymorph.SILICA_LIQUID,
        temperature=_sourced(
            Q_(1996.0, "K"),
            SRC_BOUROVA,
            "Bourova and Richet 1998 report the volume of beta-cristobalite returning to its "
            "750 K value at the melting temperature of 2000 K; 1996 K is the Holland, "
            "Green and Powell ds633 melting point of the quartz liquid endmember used in "
            "ae.physics.thermal. See LIMITATIONS: sources disagree at the 1996 to 2000 K "
            "level and the brief's 1723 K is the quartz (not cristobalite) melting point.",
        ),
        character=TransitionCharacter.MELTING,
        reversible_on_cooling=False,
        note=(
            "Silica does not recrystallise on cooling at industrial rates: the melt "
            "vitrifies. The brief's 'melting near 1723 C' is a unit slip; 1723 K (1450 "
            "degC) is the commonly quoted metastable melting point of QUARTZ, and Bourova and "
            "Richet 1998 adjust it to 1673 K."
        ),
    ),
)


def order_parameter(temperature: Quantity, params: LandauParameters = QUARTZ_LANDAU,
                    pressure: Quantity | None = None) -> float:
    """Landau order parameter Q of the alpha to beta inversion.

    Implements equation (1) of the module docstring.

    Parameters
    ----------
    temperature
        Temperature as a pint Quantity, any temperature unit.
    params
        Landau parameters. Defaults to quartz; pass another set for a different mineral so
        that nothing in this module is hardcoded to one phase.
    pressure
        Pressure. Defaults to ``params.p_0``.

    Returns
    -------
    float
        Order parameter in [0, 1], zero at and above T_c.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> round(order_parameter(Q_(298.15, "K")), 5)
    0.89721
    >>> order_parameter(Q_(1000.0, "K"))
    0.0
    """
    t_k = float(require_dimensionality(temperature, "temperature", "temperature").to("K").magnitude)
    if t_k <= 0.0:
        raise ValueError(f"absolute temperature must be positive, got {t_k} K")
    tc = float(landau_critical_temperature(params, pressure).to("K").magnitude)
    tc0 = float(params.tc_0.quantity.to("K").magnitude)
    if t_k >= tc:
        return 0.0
    q = ((tc - t_k) / tc0) ** 0.25
    assert q >= 0.0, "order parameter cannot be negative"
    return float(q)


def landau_critical_temperature(params: LandauParameters = QUARTZ_LANDAU,
                                pressure: Quantity | None = None) -> Quantity:
    """Critical temperature T_c at ``pressure``, from ``T_c = T_c0 + V_D (P - P_0) / S_D``.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> round(float(landau_critical_temperature().to("K").magnitude), 2)
    847.0
    """
    p = params.p_0.quantity if pressure is None else require_dimensionality(
        pressure, "pressure", "pressure")
    dp = (p - params.p_0.quantity).to("Pa")
    tc = params.tc_0.quantity.to("K") + (params.v_d.quantity * dp / params.s_d.quantity).to("K")
    assert float(tc.magnitude) > 0.0, "critical temperature must be positive"
    return tc


def excess_molar_volume(temperature: Quantity, params: LandauParameters = QUARTZ_LANDAU,
                        pressure: Quantity | None = None) -> Quantity:
    """Transition-related excess molar volume ``V_ex = V_D Q^2``, m^3/mol.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> v = excess_molar_volume(Q_(298.15, "K"))
    >>> round(float(v.to("cm**3/mol").magnitude), 5)
    0.95632
    """
    q = order_parameter(temperature, params, pressure)
    v = params.v_d.quantity * (q * q)
    assert float(v.to("m**3/mol").magnitude) >= 0.0, "excess volume cannot be negative"
    return v.to("m**3/mol")


def alpha_beta_cumulative_volume_strain(
    t_from: Quantity, t_to: Quantity, params: LandauParameters = QUARTZ_LANDAU,
    pressure: Quantity | None = None,
) -> float:
    """Cumulative transition-related volumetric strain between two temperatures.

    This is quantity (a) of the module docstring: the expansion attributable to the
    order-disorder transition accumulated over a heating interval, NOT the step at T_c. It
    is the quantity that matches the 3.5 to 4.5 vol% figures quoted for the alpha to beta
    inversion in the trade literature.

    Parameters
    ----------
    t_from, t_to
        Interval endpoints. Positive strain means expansion going from ``t_from`` to
        ``t_to``.
    params, pressure
        As for :func:`order_parameter`.

    Returns
    -------
    float
        Volumetric strain, dimensionless.

    Examples
    --------
    Heating alpha-quartz from 298.15 K to above T_c releases the whole ordering volume:

    >>> from ae.core.units import Q_
    >>> round(alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(900.0, "K")), 5)
    0.04215
    """
    v_from = excess_molar_volume(t_from, params, pressure)
    v_to = excess_molar_volume(t_to, params, pressure)
    v0 = params.v_0.quantity.to("m**3/mol")
    strain = float(((v_from - v_to) / v0).to("dimensionless").magnitude)
    return strain


def cristobalite_onset_temperature(
    starting_phase: Polymorph,
    contact: ContactMaterial = ContactMaterial.NONE,
) -> Value:
    """Practical cristobalite onset temperature for a starting phase and contact material.

    Crystalline quartz and fused silica have very different onsets, and the onset for glass
    drops further in contact with graphite or alumina. Warden et al. 2024 state that
    cristobalite "forms at above 1470 degC in its pure form and above 1000 degC for quartz
    glass", and measured layer thickness after 3 h at 1500 degC in the order
    graphite > alumina > silicon carbide.

    Parameters
    ----------
    starting_phase
        The silica phase being held. Glass and amorphous silica take the 1000 degC onset.
    contact
        Material in contact with the silica.

    Returns
    -------
    Value
        Onset temperature, SOURCED where Warden et al. state a number and ASSUMED where
        this function has to interpolate, with the basis recorded either way.

    Notes
    -----
    The brief groups graphite, SiC and alumina together as dropping the onset to roughly
    1000 degC. Warden et al. 2024 contradict the SiC part: SiC gave the THINNEST
    cristobalite layer of the three contact materials, i.e. it retarded rather than
    promoted the transformation relative to graphite and alumina. That correction is
    carried here.
    """
    glassy = starting_phase in (
        Polymorph.SILICA_GLASS, Polymorph.AMORPHOUS, Polymorph.SILICA_LIQUID)
    if not glassy:
        return _sourced(
            Q_(1743.15, "K"), SRC_WARDEN,
            "1470 degC onset for crystalline quartz in its pure form (Warden et al. 2024)",
        )
    base = _sourced(
        Q_(1273.15, "K"), SRC_WARDEN,
        "1000 degC onset for quartz glass (Warden et al. 2024)",
    )
    if contact in (ContactMaterial.NONE, ContactMaterial.SILICON_CARBIDE):
        if contact is ContactMaterial.SILICON_CARBIDE:
            return Value(
                quantity=Q_(1273.15, "K"),
                tag=Tag.ASSUMED,
                basis=(
                    "ESTIMATE. Warden et al. 2024 measured the thinnest cristobalite layer "
                    "of their three contact materials under SiC (indicating retardation "
                    "relative to graphite and alumina) but report no onset temperature for "
                    "it. The glass onset of 1000 degC is carried unchanged rather than "
                    "raised, because the retardation was observed in layer thickness after "
                    "3 h at 1500 degC and gives no basis for shifting an onset."
                ),
                confidence="low",
            )
        return base
    return Value(
        quantity=Q_(1273.15, "K"),
        tag=Tag.ASSUMED,
        basis=(
            f"ESTIMATE. Contact with {contact.value} enhances cristobalite formation "
            "(Warden et al. 2024 rank graphite > alumina > SiC by layer thickness after "
            "3 h at 1500 degC), but no source reachable here reports a SHIFTED onset "
            "temperature for a specific contact material. The 1000 degC glass onset is "
            "used as a lower bound and should be treated as such, not as a measured onset "
            "for this contact material."
        ),
        confidence="low",
    )


def arrhenius_rate_ratio(
    t_ref: Quantity,
    t_query: Quantity,
    activation_energy: Value | None = None,
    allow_extrapolation: bool = False,
) -> float:
    """Ratio ``k(t_query) / k(t_ref)`` for a thermally activated transformation.

    Implements equation (3). Only the activation energy is required because the
    pre-exponential factor cancels in the ratio, which is what makes this computable from
    a source that reports E_a without A.

    Parameters
    ----------
    t_ref, t_query
        Reference and query temperatures.
    activation_energy
        Molar activation energy as a Value. Defaults to the Breneman and Halloran 2014 cristobalite
        value of 555 kJ/mol.
    allow_extrapolation
        The default E_a is calibrated on 1200 to 1350 degC data and Breneman and Halloran report the
        rate MAXIMUM at 1500 to 1600 degC, so an Arrhenius extrapolation above 1350 degC
        overstates the rate. Outside 1473 to 1623 K this raises unless set True.

    Returns
    -------
    float
        Strictly positive rate ratio, greater than one when ``t_query`` exceeds ``t_ref``.

    Examples
    --------
    Doubling-scale sensitivity between 1450 and 1500 degC:

    >>> from ae.core.units import Q_
    >>> r = arrhenius_rate_ratio(Q_(1723.15, "K"), Q_(1773.15, "K"), allow_extrapolation=True)
    >>> r > 1.0
    True
    """
    ea = activation_energy if activation_energy is not None else Value(
        quantity=Q_(555.0, "kJ/mol"),
        tag=Tag.SOURCED,
        source=SRC_BRENEMAN,
        basis="apparent activation energy 555 +/- 24 kJ/mol for the kinetic constant of "
              "cristobalite formation from sintered silica, 1200 to 1350 degC",
        confidence="medium",
    )
    require_dimensionality(ea.quantity, "molar_energy", "activation_energy")
    ea_j = float(ea.quantity.to("J/mol").magnitude)
    if ea_j <= 0.0:
        raise ValueError("activation energy must be positive for an Arrhenius rate")
    t1 = float(require_dimensionality(t_ref, "temperature", "t_ref").to("K").magnitude)
    t2 = float(require_dimensionality(t_query, "temperature", "t_query").to("K").magnitude)
    if t1 <= 0.0 or t2 <= 0.0:
        raise ValueError("absolute temperatures must be positive")
    if not allow_extrapolation and activation_energy is None:
        for t in (t1, t2):
            if not 1473.0 <= t <= 1623.0:
                raise ValueError(
                    f"{t} K is outside the 1473 to 1623 K band in which the default "
                    f"E_a (Breneman and Halloran 2014, calibrated 1200 to 1350 degC, rate maximum "
                    f"reported at 1500 to 1600 degC) can be used without extrapolating "
                    f"past a rate maximum; pass allow_extrapolation=True to override and "
                    f"record that the result is an extrapolation"
                )
    ratio = math.exp(-(ea_j / R_GAS) * (1.0 / t2 - 1.0 / t1))
    assert ratio > 0.0, "an Arrhenius rate ratio is strictly positive"
    return ratio


def jmak_fraction(hold_time: Quantity, rate_constant: Value, avrami_n: Value | None = None
                  ) -> float:
    """Transformed fraction from the JMAK law ``X = 1 - exp[-(k t)^n]``.

    Parameters
    ----------
    hold_time
        Isothermal hold time.
    rate_constant
        Rate constant k with dimension 1/time. REQUIRED, deliberately not defaulted: no
        source reachable from this environment reports a pre-exponential factor for the
        quartz to cristobalite transformation, and Ringdalen 2015 measured a 76-fold spread
        in cristobalite fraction across natural quartz samples at fixed temperature and
        time, so k is an ore property that must be measured.
    avrami_n
        Avrami exponent. Defaults to the Breneman and Halloran 2014 value of 3.0.

    Returns
    -------
    float
        Transformed fraction in [0, 1].

    Examples
    --------
    With ``k t = 1`` and ``n = 3``, ``X = 1 - exp(-1) = 0.6321``:

    >>> from ae.core.units import Q_
    >>> from ae.core.provenance import Tag, Value
    >>> k = Value(quantity=Q_(1.0, "1/hour"), tag=Tag.ASSUMED, basis="doctest")
    >>> round(jmak_fraction(Q_(1.0, "hour"), k), 4)
    0.6321
    """
    n_val = avrami_n if avrami_n is not None else Value(
        quantity=Q_(3.0, "dimensionless"),
        tag=Tag.SOURCED,
        source=SRC_BRENEMAN,
        basis="Avrami time exponent 3.0 +/- 0.6 (Breneman and Halloran 2014), consistent with "
              "three-dimensional growth from a constant nucleus population",
        confidence="medium",
    )
    require_dimensionality(rate_constant.quantity, "rate_first_order", "rate_constant")
    k = float(rate_constant.quantity.to("1/s").magnitude)
    t = float(require_dimensionality(hold_time, "time", "hold_time").to("s").magnitude)
    n = float(n_val.quantity.to("dimensionless").magnitude)
    if k < 0.0:
        raise ValueError("a JMAK rate constant cannot be negative")
    if t < 0.0:
        raise ValueError("hold time cannot be negative")
    if n <= 0.0:
        raise ValueError("the Avrami exponent must be positive")
    x = 1.0 - math.exp(-((k * t) ** n))
    assert 0.0 <= x <= 1.0, f"transformed fraction out of range: {x}"
    return x


class ScheduleSegment(BaseModel):
    """One ramp-and-hold leg of a thermal schedule."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    target_temperature: Quantity
    hold_time: Quantity
    label: str | None = None

    @model_validator(mode="after")
    def _dimensions(self) -> ScheduleSegment:
        require_dimensionality(self.target_temperature, "temperature", "target_temperature")
        require_dimensionality(self.hold_time, "time", "hold_time")
        if float(self.target_temperature.to("K").magnitude) <= 0.0:
            raise ValueError("absolute temperature must be positive")
        if float(self.hold_time.to("s").magnitude) < 0.0:
            raise ValueError("hold time cannot be negative")
        return self

    @property
    def target_k(self) -> float:
        return float(self.target_temperature.to("K").magnitude)


class ThermalSchedule(BaseModel):
    """A furnace recipe: where it starts, where it goes, in what and against what.

    Atmosphere is carried because Ringdalen 2015 found cristobalite forms more readily in
    inert than in reducing atmospheres, and contact material is carried because Warden et
    al. 2024 found it to be "the most determining factor" for cristobalite layer thickness.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    start_temperature: Quantity
    segments: tuple[ScheduleSegment, ...]
    starting_phase: Polymorph = Polymorph.ALPHA_QUARTZ
    atmosphere: Literal["air", "inert", "reducing", "chlorinating", "vacuum"] = "air"
    contact: ContactMaterial = ContactMaterial.NONE
    quench: bool = Field(
        default=False,
        description="True when the schedule ends in a water or air quench rather than a "
        "controlled cool, which forces a second pass through the alpha-beta inversion at "
        "a high strain rate.",
    )

    @model_validator(mode="after")
    def _dimensions(self) -> ThermalSchedule:
        require_dimensionality(self.start_temperature, "temperature", "start_temperature")
        if not self.segments:
            raise ValueError("a thermal schedule needs at least one segment")
        return self

    @property
    def peak_temperature_k(self) -> float:
        """Highest temperature reached, kelvin."""
        return max(s.target_k for s in self.segments)

    @property
    def start_k(self) -> float:
        return float(self.start_temperature.to("K").magnitude)

    def total_hold_above(self, temperature: Quantity) -> Quantity:
        """Total hold time in segments whose target is at or above ``temperature``."""
        thr = float(require_dimensionality(temperature, "temperature", "temperature")
                    .to("K").magnitude)
        total = math.fsum([float(s.hold_time.to("s").magnitude) for s in self.segments
                           if s.target_k >= thr])
        return Q_(total, "s")


def crossed_transitions(
    schedule: ThermalSchedule,
    feedstock: Feedstock,
    transitions: tuple[Transition, ...] = TRANSITIONS,
) -> tuple[Transition, ...]:
    """Which transitions a schedule crosses, in ascending temperature order.

    Parameters
    ----------
    schedule
        The furnace recipe.
    feedstock
        The ore. Present because a transition set is a property of the material, and
        because an uncharacterized feedstock makes the answer a scenario rather than a
        prediction (see Notes).
    transitions
        Transition set to test against. Defaults to :data:`TRANSITIONS`.

    Returns
    -------
    tuple of Transition
        Every transition whose temperature lies in the interval actually traversed,
        including the start temperature, lowest first.

    Notes
    -----
    Crossing a boundary is necessary but not sufficient for the product phase to appear.
    Reconstructive transitions need time and often a mineraliser, so read a RECONSTRUCTIVE
    entry as "permitted" and a DISPLACIVE entry as "occurred". For an uncharacterized
    feedstock the mineraliser inventory is unknown, which is the dominant uncertainty on
    the reconstructive entries: ``feedstock.characterized is False`` makes every
    reconstructive result a scenario.

    Examples
    --------
    An 800 degC calcination crosses only the inversion:

    >>> from ae.core.units import Q_
    >>> from ae.core.feedstock import Feedstock, OreType
    >>> f = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
    ...               deposit_name="Vikarabad", country="IN")
    >>> s = ThermalSchedule(start_temperature=Q_(298.15, "K"),
    ...                     segments=(ScheduleSegment(target_temperature=Q_(1073.15, "K"),
    ...                                               hold_time=Q_(2.0, "hour")),))
    >>> [t.name for t in crossed_transitions(s, f)]
    ['alpha_beta_quartz_inversion']
    """
    if not isinstance(feedstock, Feedstock):
        raise TypeError("crossed_transitions requires a Feedstock; the transition set and "
                        "its realisation are material properties, not global constants")
    lo = min(schedule.start_k, min(s.target_k for s in schedule.segments))
    hi = schedule.peak_temperature_k
    hit = [t for t in transitions if lo <= t.temperature_k <= hi]
    return tuple(sorted(hit, key=lambda t: t.temperature_k))


class DevitrificationAssessment(BaseModel):
    """Ordinal devitrification screen for a schedule. Not a predicted fraction."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    level: RiskLevel
    onset_temperature: Value
    margin_above_onset: Quantity
    hold_above_onset: Quantity
    drivers: tuple[str, ...]
    is_scenario: bool
    note: str

    @model_validator(mode="after")
    def _dimensions(self) -> DevitrificationAssessment:
        require_dimensionality(self.margin_above_onset, "temperature", "margin_above_onset")
        require_dimensionality(self.hold_above_onset, "time", "hold_above_onset")
        if float(self.hold_above_onset.to("s").magnitude) < 0.0:
            raise ValueError("hold time cannot be negative")
        return self


def devitrification_risk(schedule: ThermalSchedule, feedstock: Feedstock
                         ) -> DevitrificationAssessment:
    """Flag devitrification risk for a schedule, as an ordinal level with its drivers.

    Risk is built from four factors, each of which is reported in ``drivers`` so the caller
    can see what produced the level: the margin of the peak temperature above the onset for
    the starting phase, the hold time above that onset, the contact material, and the
    atmosphere.

    Parameters
    ----------
    schedule
        The furnace recipe, including contact material and atmosphere.
    feedstock
        The ore. An uncharacterized feedstock sets ``is_scenario``, because the alkali and
        alkaline-earth inventory that mineralises the reconstructive transformation is then
        unknown.

    Returns
    -------
    DevitrificationAssessment

    Notes
    -----
    Cristobalite matters twice over in this business. In a fused quartz crucible it causes
    "significant structural defects in silicon ingots" (Warden et al. 2024). In a silicon
    smelter feed it is not purely harmful: Ringdalen 2015 reports a higher specific surface
    area for cristobalite than for quartz and a correspondingly higher rate for SiO2
    reduction reactions, so the same transformation that ruins a crucible can help a
    submerged arc furnace. This function reports risk, and the caller decides its sign.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> from ae.core.feedstock import Feedstock, OreType
    >>> f = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
    ...               deposit_name="Vikarabad", country="IN")
    >>> s = ThermalSchedule(start_temperature=Q_(298.15, "K"),
    ...                     segments=(ScheduleSegment(target_temperature=Q_(1073.15, "K"),
    ...                                               hold_time=Q_(2.0, "hour")),))
    >>> devitrification_risk(s, f).level is RiskLevel.NONE
    True
    """
    if not isinstance(feedstock, Feedstock):
        raise TypeError("devitrification_risk requires a Feedstock")
    onset = cristobalite_onset_temperature(schedule.starting_phase, schedule.contact)
    onset_k = float(onset.quantity.to("K").magnitude)
    peak = schedule.peak_temperature_k
    margin = peak - onset_k
    hold_s = float(schedule.total_hold_above(Q_(onset_k, "K")).to("s").magnitude)

    drivers: list[str] = []
    if margin <= 0.0:
        return DevitrificationAssessment(
            level=RiskLevel.NONE,
            onset_temperature=onset,
            margin_above_onset=Q_(margin, "K"),
            hold_above_onset=Q_(hold_s, "s"),
            drivers=(f"peak {peak:.1f} K is {-margin:.1f} K below the "
                     f"{schedule.starting_phase.value} onset of {onset_k:.1f} K",),
            is_scenario=not feedstock.characterized,
            note="Peak temperature does not reach the cristobalite onset for this starting "
                 "phase. Reconstructive transformation is not expected.",
        )

    score = 0
    if margin > 0.0:
        score += 1
        drivers.append(f"peak {peak:.1f} K exceeds onset {onset_k:.1f} K by {margin:.1f} K")
    if margin > 200.0:
        score += 1
        drivers.append("margin above onset exceeds 200 K")
    if hold_s >= 3600.0:
        score += 1
        drivers.append(f"hold above onset is {hold_s / 3600.0:.2f} h, at or above the 1 h "
                       "exposure at which Ringdalen 2015 measured 1 to 76 percent "
                       "cristobalite in natural quartz")
    if hold_s >= 6 * 3600.0:
        score += 1
        drivers.append("hold above onset is at or above the 6 h exposure at which Ringdalen "
                       "2015 still found incomplete conversion at 1600 degC")
    if schedule.contact in (ContactMaterial.GRAPHITE, ContactMaterial.ALUMINA):
        score += 1
        drivers.append(f"contact with {schedule.contact.value} enhances cristobalite "
                       "formation (Warden et al. 2024 rank graphite > alumina > SiC by "
                       "layer thickness after 3 h at 1500 degC)")
    if schedule.contact is ContactMaterial.SILICON_CARBIDE:
        drivers.append("contact with silicon carbide gave the THINNEST cristobalite layer "
                       "of the three materials tested by Warden et al. 2024, contradicting "
                       "the platform brief's grouping of SiC with graphite and alumina")
    if schedule.atmosphere == "inert":
        score += 1
        drivers.append("inert atmosphere: Ringdalen 2015 found cristobalite forms more "
                       "readily in inert than in reducing atmospheres")
    if schedule.atmosphere == "reducing":
        drivers.append("reducing atmosphere retards cristobalite formation relative to "
                       "inert (Ringdalen 2015)")

    level = (RiskLevel.LOW if score <= 1 else
             RiskLevel.MODERATE if score == 2 else
             RiskLevel.HIGH if score <= 4 else
             RiskLevel.SEVERE)
    return DevitrificationAssessment(
        level=level,
        onset_temperature=onset,
        margin_above_onset=Q_(margin, "K"),
        hold_above_onset=Q_(hold_s, "s"),
        drivers=tuple(drivers),
        is_scenario=not feedstock.characterized,
        note="Ordinal screen from temperature margin, hold time, contact material and "
             "atmosphere. It is NOT a predicted cristobalite fraction: no pre-exponential "
             "factor for the transformation could be sourced, and the ore-to-ore spread "
             "measured by Ringdalen 2015 was 76-fold at fixed temperature and time.",
    )
