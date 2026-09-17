r"""Energy balances for calcination, quenching and fusion of silica.

Scope: the specific energy of a thermal step, in kWh per tonne of product, decomposed into
the part thermodynamics fixes (sensible plus latent plus transition enthalpy) and the part
the furnace loses. Those two are kept separate throughout, because the first is a property
of SiO2 and the second is a property of a plant, and conflating them is how a process
engineer ends up defending a furnace design with a thermodynamic number.

EQUATIONS
---------
(1) Holland and Powell heat-capacity polynomial (their Cp form, ds62/ds633):

    .. math::

        C_p(T) = a + bT + \frac{c}{T^2} + \frac{d}{\sqrt{T}}

    Symbols:
      C_p   molar isobaric heat capacity, J/(mol K).
      T     temperature, K. Valid 298.15 K to the phase's ``t_max`` (see
            :data:`CP_COEFFICIENTS`; 1800 K for quartz, 2000 K for cristobalite).
      a     J/(mol K); b J/(mol K^2); c J K/mol; d J/(mol K^{1/2}).

    Source of the functional form and of every coefficient: Holland and Powell 2011,
    doi 10.1111/j.1525-1314.2010.00923.x (dataset ds62), with the liquid endmember from
    Holland, Green and Powell 2018, doi 10.1093/petrology/egy048 (dataset ds633).

(2) Analytic integral of (1), the sensible enthalpy:

    .. math::

        \int_{T_1}^{T_2} C_p\,dT = \left[aT + \frac{b T^2}{2} - \frac{c}{T}
        + 2d\sqrt{T}\right]_{T_1}^{T_2}

    Derivation: term-by-term antiderivative of (1). :math:`\int c T^{-2} dT = -c/T` and
    :math:`\int d T^{-1/2} dT = 2 d \sqrt{T}`. Dimensions: [J/(mol K)][K] = [J/mol].

(3) Landau excess enthalpy for the alpha to beta quartz inversion. The inversion has no
    latent heat of the first-order kind; its energy cost is spread over a wide interval
    below T_c as an excess heat capacity that diverges at T_c. From the HP tricritical
    Gibbs excess (see :mod:`ae.physics.phases` equation 1):

    .. math::

        H_{ex}(T) = G_{ex}(T) - T \frac{\partial G_{ex}}{\partial T}
        = G_{ex}(T) + T\,S_D\left[Q_0^2 - Q(T)^2\right]

    .. math::

        C_{p,ex}(T) = \frac{T\,S_D}{2\,T_{c,0}\,Q(T)^2} \quad (T < T_c), \qquad
        C_{p,ex} = 0 \quad (T \ge T_c)

    Symbols:
      H_{ex}     excess molar enthalpy, J/mol.
      C_{p,ex}   excess molar heat capacity, J/(mol K), diverging as T approaches T_c.
      S_D        entropy of disordering, 4.95 J/(mol K) for quartz.
      Q, Q_0     order parameter at T and at 298.15 K.
      T_{c,0}    847 K for quartz.

    Derivation: :math:`C_{p,ex} = -T\,\partial^2 G_{ex}/\partial T^2`, and differentiating
    the HP expression twice with :math:`Q = ((T_c-T)/T_{c,0})^{1/4}` gives the form above
    (this is the ``landau_hp`` excess implemented in the HP formalism). The sign in
    :math:`H_{ex}` follows from the HP first derivative
    :math:`\partial G_{ex}/\partial T = S_D\left[Q(T)^2 - Q_0^2\right]`, so the excess
    ENTROPY is :math:`S_{ex} = -S_D\left[Q(T)^2 - Q_0^2\right]`, which is positive on
    heating because :math:`Q` falls from :math:`Q_0` toward zero as the structure
    disorders. Getting this sign backwards flips :math:`H_{ex}(T_c) - H_{ex}(T_0)` from
    +2646 to -4104 J/mol, i.e. it makes disordering exothermic, which violates the second
    law for a transition to a higher-entropy phase on heating. The total inversion
    enthalpy from 298.15 K to T_c is 2646 J/mol, i.e. 44.0 kJ/kg, computed by this module
    rather than asserted.

(4) Total thermal enthalpy of a phase-change path, per unit mass:

    .. math::

        \Delta h = \frac{1}{M}\left[\left(H^0_{f,2} + \int_{T_0}^{T_2} C_{p,2}dT\right)
        - \left(H^0_{f,1} + \int_{T_0}^{T_1} C_{p,1}dT + H_{ex,1}(T_1)\right)\right]

    Symbols:
      \Delta h   specific enthalpy change, J/kg.
      H^0_f      standard enthalpy of formation at T_0 = 298.15 K, J/mol.
      M          molar mass of SiO2, 0.0600843 kg/mol (ds62).
      subscripts 1 and 2 denote the initial and final phase.

    The difference in H^0_f is what carries the transition enthalpy, so a single expression
    handles heating, transformation and melting without a separate latent-heat term.

(5) Specific energy of a furnace step:

    .. math::

        e = \frac{\Delta h}{\eta}, \qquad
        \eta = \eta_{thermal}\,(1 - f_{loss})

    Symbols:
      e                specific energy input, J/kg, convertible to kWh/tonne.
      \eta             overall efficiency, dimensionless, range (0, 1].
      \eta_{thermal}   energy actually delivered to the charge divided by energy input.
      f_{loss}         additional fractional loss (structure, flue), range [0, 1).

    Reference points for eta, US DOE via Galitsky and Worrell 2008 (OSTI 927883): "only
    about 33-40% of the energy consumed by a continuous furnace goes toward melting the
    glass", with "up to 30%" lost through the structure and "another 30%" through flue gas.

(6) Quench heat rejection and mean cooling rate:

    .. math::

        q = \frac{1}{M}\left[H_1(T_{hot}) - H_1(T_{cold})\right], \qquad
        \dot{T} = \frac{T_{hot} - T_{cold}}{\Delta t}

    Symbols:
      q        specific heat rejected to the quench medium, J/kg, positive.
      \dot{T}  mean cooling rate, K/s. A MEAN, not the surface rate that governs
               thermal-shock fracture (see LIMITATIONS).

LIMITATIONS
-----------
1. NO Cp FOR SILICA GLASS, DISTINCT FROM THE LIQUID. webbook.nist.gov (Shomate
   coefficients) and the Richet et al. 1982 drop-calorimetry paper
   (doi 10.1016/0016-7037(82)90383-0, the standard source for amorphous SiO2 Cp from 1000
   to 1800 K) are both unreachable from this sandbox, and no allowlisted substitute
   publishes glass-specific coefficients. This module therefore uses the ds633 quartz
   LIQUID endmember (Cp = 82.5 J/(mol K), temperature-independent) for molten and glassy
   silica, and flags any such call. Below the glass transition (roughly 1475 K for silica)
   a constant liquid Cp is not the glass Cp, so glass sensible-heat results in that range
   carry an unquantified error.
2. THE LIQUID Cp IS A CONSTANT. ds633 gives b = c = d = 0 for ``qL``, so the model cannot
   reproduce any curvature in the melt heat capacity and cannot be extrapolated to a
   temperature range where that curvature matters.
3. HP2011 DOES NOT REPRODUCE THE TRIDYMITE-CRISTOBALITE BOUNDARY. Equating ds62 Gibbs
   energies in this module gives quartz-tridymite equilibrium at 866 degC (against the
   accepted 870 degC, good) but places tridymite-cristobalite equilibrium above 2600 K,
   against the accepted 1470 degC. The ds62 dataset is calibrated for petrological
   equilibria and its silica polymorph relations at 1 bar are not fit for predicting the
   high-temperature reconstructive boundaries. Take transition TEMPERATURES from
   :mod:`ae.physics.phases`, which sources them, and use this module only for the
   ENTHALPY along a path whose temperatures were set elsewhere.
4. THE ds633 LIQUID ENDMEMBER APPEARS STABLE AT EVERY TEMPERATURE at 1 bar in a naive
   Gibbs comparison against the ds62 solids, which is the opposite failure from a missing
   melt and equally disqualifying. Evaluating G(qL) - G(crst) over 400 to 2600 K gives
   -3735 J/mol at 400 K and -3451 J/mol at 2600 K, and G(qL) - G(q) gives -852 J/mol and
   -6911 J/mol at the same limits: the liquid sits BELOW both solids throughout, so the
   comparison yields no crossing and would imply silica is molten at room temperature.
   The cause is that ds62 and ds633 are not mutually calibrated for a 1 bar melting
   point (ds633 fits the liquid for peridotite-to-granite melting at crustal and mantle
   pressures, where the relevant equilibria are not the 1 bar silica melting curve).
   Melting temperatures must therefore be SUPPLIED, not computed, and this module refuses
   to infer one: see the ``transition_temperature`` requirement on
   :class:`ThermalStep`. The enthalpies along a supplied path remain usable, because
   they depend on H_0 and Cp rather than on the relative Gibbs energies.
5. LOSSES ARE NOT MODELLED FROM GEOMETRY. ``f_loss`` and ``eta_thermal`` are inputs, not
   predictions. There is no wall-conduction, radiation or flue-gas model here; a real
   design needs one, and the published efficiency bands cited above are for soda-lime and
   specialty glass furnaces, not for fused quartz.
6. MEAN COOLING RATE IS NOT A THERMAL-SHOCK CRITERION. Decrepitation depends on the
   SURFACE temperature gradient and on the fluid-inclusion population, neither of which
   this module has. ``feedstock.physical.decrepitation_index`` exists for the measured
   response and raises when unmeasured; that is the correct behaviour and this module does
   not work around it.
7. NO KILN SCALE-UP. Specific energy is per tonne of product with no throughput
   dependence, so it cannot capture the strong capacity dependence of real furnace
   efficiency (Galitsky and Worrell 2008 note all-electric furnaces are typically used
   below 75 ton/day).
8. IMPURITY EFFECTS ON Cp ARE IGNORED. Every coefficient is for pure SiO2. At the ppm
   impurity levels relevant to HPQ this is a very small error on Cp, but the same is NOT
   true of the melting behaviour, where alkali content dominates.

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
333-383, doi 10.1111/j.1525-1314.2010.00923.x. Dataset ds62. Source of the heat
capacity functional form and of every coefficient for the solid quartz
endmembers.

Holland, T. J. B., Green, E. C. R. and Powell, R. (2018) Melting of Peridotites
through to Granites: A Simple Thermodynamic Model in the System KNCFMASHTOCr,
Journal of Petrology 59(5), 881-900, doi 10.1093/petrology/egy048. Dataset
ds633. Source of the quartz LIQUID endmember, including the constant
Cp = 82.5 J/(mol K) used above.

Richet, P., Bottinga, Y., Denielou, L., Petitet, J. P. and Tequi, C. (1982)
Thermodynamic properties of quartz, cristobalite and amorphous SiO2: drop
calorimetry measurements between 1000 and 1800 K and a review from 0 to 2000 K,
Geochimica et Cosmochimica Acta 46(12), 2639-2658,
doi 10.1016/0016-7037(82)90383-0. Independent drop-calorimetry dataset over the
temperature range this module integrates across. Closed access; cited as the
comparison dataset, not as the source of any coefficient used here.

Galitsky, C., Worrell, E., Masanet, E. and Graus, W. (2008) Energy Efficiency
Improvement and Cost Saving Opportunities for the Glass Industry: An ENERGY STAR
Guide for Energy and Plant Managers, Lawrence Berkeley National Laboratory,
report LBNL-57335, doi 10.2172/927883 (OSTI 927883). Source of the statement
that only about 33 to 40 percent of the energy consumed by a continuous furnace
goes toward melting, and of the observation that all-electric furnaces are
typically used below 75 ton/day. Crossref lists Galitsky twice in the author
record for this DOI; the duplicate is in the upstream metadata, not a second
person.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import MISSING, Source, Tag, Tier, Value, _Missing
from ae.core.site import Site
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction
from ae.physics.phases import (
    QUARTZ_LANDAU,
    LandauParameters,
    Polymorph,
    order_parameter,
)

__all__ = [
    "CP_COEFFICIENTS",
    "FORMATION_ENTHALPY",
    "M_SIO2",
    "SRC_HGP2018",
    "SRC_HP2011",
    "SRC_LBNL_GLASS",
    "T_REF",
    "CpCoefficients",
    "EnergyBalance",
    "ThermalStep",
    "calcination_energy",
    "electricity_cost",
    "fusion_energy",
    "integrated_enthalpy",
    "landau_excess_enthalpy",
    "landau_excess_heat_capacity",
    "molar_heat_capacity",
    "phase_enthalpy",
    "quench_heat_rejection",
    "specific_heat_capacity",
    "specific_transition_enthalpy",
]

#: Molar mass of SiO2, kg/mol. Holland and Powell ds62 endmember value.
M_SIO2: Final[float] = 0.0600843

#: Thermodynamic reference temperature for every enthalpy of formation here, K.
T_REF: Final[float] = 298.15

_ACCESSED: Final[_dt.date] = _dt.date(2026, 9, 16)

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
        "Endmember Cp coefficients, H_0 and S_0 read from the machine-readable ds62 "
        "transcription shipped in burnman 2.1.0 (minerals/HP_2011_ds62.py, autogenerated "
        "from tc-ds62.txt, values in SI units). webbook.nist.gov is blocked by the sandbox "
        "network allowlist, so NIST-JANAF Shomate coefficients could not be used; this "
        "substitution is deliberate and recorded rather than worked around."
    ),
)

SRC_HGP2018: Final[Source] = Source(
    citation=(
        "Holland, T.J.B., Green, E.C.R. and Powell, R. 2018, Melting of Peridotites through "
        "to Granites: A Simple Thermodynamic Model in the System KNCFMASHTOCr, Journal of "
        "Petrology 59:881-900 (dataset ds633)"
    ),
    tier=Tier.T1,
    doi="10.1093/petrology/egy048",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Quartz liquid endmember qL (Cp = 82.5 J/(mol K), H_0 = -921080 J/mol, "
        "S_0 = 16.3 J/(mol K)) read from the ds633 transcription in burnman 2.1.0 "
        "(minerals/HGP_2018_ds633.py)."
    ),
)

SRC_LBNL_GLASS: Final[Source] = Source(
    citation=(
        "Galitsky, C. and Worrell, E. 2008, Energy Efficiency Improvement and Cost Saving "
        "Opportunities for the Glass Industry: An ENERGY STAR Guide for Energy and Plant "
        "Managers, Lawrence Berkeley National Laboratory LBNL-57335-Revision, US "
        "Department of Energy"
    ),
    tier=Tier.T1,
    doi="10.2172/927883",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Government laboratory report. Figures used: theoretical 2.2 MMBtu per short ton "
        "to melt glass; 33 to 40 percent of continuous-furnace energy reaching the glass; "
        "specialty-glass electric melter 10.3 (8.9 to 11.6) MMBtu per short ton; "
        "state-of-the-art electric melters 780 to 800 kWh per short ton for soda-lime and "
        "sodium borate glass (attributed there to Hibscher et al. 2005)."
    ),
)


class CpCoefficients(BaseModel):
    """Holland and Powell Cp polynomial coefficients with their validity range.

    The four coefficients are held as provenance Values, not floats, so that the dataset
    they came from travels with them and a future swap to NIST-JANAF Shomate coefficients
    is visible in every downstream result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    phase: Polymorph
    a: Value
    b: Value
    c: Value
    d: Value
    t_min: Quantity
    t_max: Quantity
    landau: LandauParameters | None = None
    is_liquid_proxy_for_glass: bool = Field(
        default=False,
        description="True when these coefficients are the LIQUID endmember standing in for "
        "glass because no glass-specific Cp could be sourced. See LIMITATIONS item 1.",
    )

    @model_validator(mode="after")
    def _dimensions(self) -> CpCoefficients:
        require_dimensionality(self.a.quantity, "heat_capacity_molar", "a")
        require_dimensionality(self.t_min, "temperature", "t_min")
        require_dimensionality(self.t_max, "temperature", "t_max")
        if float(self.t_min.to("K").magnitude) >= float(self.t_max.to("K").magnitude):
            raise ValueError("t_min must be below t_max")
        if float(self.t_min.to("K").magnitude) <= 0.0:
            raise ValueError("absolute temperature must be positive")
        return self

    @property
    def raw(self) -> tuple[float, float, float, float]:
        """Coefficients as SI floats: (a, b, c, d) for J/(mol K) with T in K."""
        return (
            float(self.a.quantity.to("J/(mol*K)").magnitude),
            float(self.b.quantity.to("J/(mol*K**2)").magnitude),
            float(self.c.quantity.to("J*K/mol").magnitude),
            float(self.d.quantity.to("J/(mol*K**0.5)").magnitude),
        )

    def in_range(self, t_k: float) -> bool:
        return (float(self.t_min.to("K").magnitude) <= t_k
                <= float(self.t_max.to("K").magnitude))


def _v(mag: float, unit: str, src: Source, field: str) -> Value:
    return Value(quantity=Q_(mag, unit), tag=Tag.SOURCED, source=src,
                 basis=f"dataset endmember parameter {field}", confidence="high")


def _cp(phase: Polymorph, a: float, b: float, c: float, d: float, t_max: float,
        src: Source, name: str, landau: LandauParameters | None = None,
        liquid_proxy: bool = False) -> CpCoefficients:
    return CpCoefficients(
        phase=phase,
        a=_v(a, "J/(mol*K)", src, f"Cp[0] ({name})"),
        b=_v(b, "J/(mol*K**2)", src, f"Cp[1] ({name})"),
        c=_v(c, "J*K/mol", src, f"Cp[2] ({name})"),
        d=_v(d, "J/(mol*K**0.5)", src, f"Cp[3] ({name})"),
        t_min=Q_(T_REF, "K"),
        t_max=Q_(t_max, "K"),
        landau=landau,
        is_liquid_proxy_for_glass=liquid_proxy,
    )


#: Cp coefficients by polymorph.
#:
#: t_max values are ASSUMED upper bounds on the polynomial, not published validity limits:
#: the ds62/ds633 transcription carries no per-endmember temperature range. The bounds used
#: are the temperature at which each phase ceases to be relevant at 1 bar (quartz melts
#: metastably near 1700 K, cristobalite melts near 2000 K), which is a conservative reading.
#: See :data:`CP_T_MAX_BASIS`.
CP_COEFFICIENTS: Final[dict[Polymorph, CpCoefficients]] = {
    Polymorph.QUARTZ: _cp(
        Polymorph.QUARTZ, 92.9, -6.42e-4, -714900.0, -716.1, 1800.0,
        SRC_HP2011, "q", landau=QUARTZ_LANDAU),
    Polymorph.TRIDYMITE: _cp(
        Polymorph.TRIDYMITE, 74.9, 3.10e-3, -1174000.0, -236.7, 2000.0,
        SRC_HP2011, "trd"),
    Polymorph.CRISTOBALITE: _cp(
        Polymorph.CRISTOBALITE, 72.7, 1.304e-3, -4129000.0, 0.0, 2000.0,
        SRC_HP2011, "crst"),
    Polymorph.SILICA_LIQUID: _cp(
        Polymorph.SILICA_LIQUID, 82.5, 0.0, 0.0, 0.0, 3000.0,
        SRC_HGP2018, "qL"),
    Polymorph.SILICA_GLASS: _cp(
        Polymorph.SILICA_GLASS, 82.5, 0.0, 0.0, 0.0, 3000.0,
        SRC_HGP2018, "qL", liquid_proxy=True),
}

#: Why t_max is what it is, recorded because it is an assumption and not a source value.
CP_T_MAX_BASIS: Final[str] = (
    "ASSUMED. The ds62/ds633 SI transcription carries no per-endmember temperature validity "
    "range, and the publisher PDF is unreachable from this sandbox. t_max is set to the "
    "temperature above which each phase is irrelevant at 1 bar (1800 K quartz, 2000 K "
    "tridymite and cristobalite, 3000 K liquid). Holland and Powell fit their Cp forms to "
    "calorimetric data that for silica extends to roughly 1800 K, so extrapolation beyond "
    "these bounds is refused rather than silently performed."
)

#: Standard enthalpy of formation at 298.15 K and 1 bar, J/mol, by polymorph.
FORMATION_ENTHALPY: Final[dict[Polymorph, Value]] = {
    Polymorph.QUARTZ: _v(-910720.0, "J/mol", SRC_HP2011, "H_0 (q)"),
    Polymorph.TRIDYMITE: _v(-907110.0, "J/mol", SRC_HP2011, "H_0 (trd)"),
    Polymorph.CRISTOBALITE: _v(-904270.0, "J/mol", SRC_HP2011, "H_0 (crst)"),
    Polymorph.SILICA_LIQUID: _v(-921080.0, "J/mol", SRC_HGP2018, "H_0 (qL)"),
    Polymorph.SILICA_GLASS: _v(-921080.0, "J/mol", SRC_HGP2018, "H_0 (qL)"),
}

#: Polymorphs that share the quartz endmember, so callers may pass either side of the
#: displacive inversion without knowing that HP treats them as one phase plus a Landau term.
_QUARTZ_ALIASES: Final[tuple[Polymorph, ...]] = (
    Polymorph.ALPHA_QUARTZ, Polymorph.BETA_QUARTZ, Polymorph.QUARTZ)


def _resolve(phase: Polymorph) -> Polymorph:
    if phase in _QUARTZ_ALIASES:
        return Polymorph.QUARTZ
    if phase is Polymorph.AMORPHOUS:
        return Polymorph.SILICA_GLASS
    return phase


def _coeffs(phase: Polymorph) -> CpCoefficients:
    key = _resolve(phase)
    if key not in CP_COEFFICIENTS:
        raise KeyError(
            f"no Cp coefficients for {phase.value}; available: "
            f"{sorted(p.value for p in CP_COEFFICIENTS)}"
        )
    return CP_COEFFICIENTS[key]


def _check_range(coeffs: CpCoefficients, t_k: float, allow_extrapolation: bool) -> None:
    if not allow_extrapolation and not coeffs.in_range(t_k):
        raise ValueError(
            f"{t_k:.2f} K is outside the {coeffs.t_min.to('K').magnitude:.2f} to "
            f"{coeffs.t_max.to('K').magnitude:.2f} K range assumed for the "
            f"{coeffs.phase.value} Cp polynomial. {CP_T_MAX_BASIS} Pass "
            f"allow_extrapolation=True to override and record that the result is an "
            f"extrapolation."
        )


def molar_heat_capacity(phase: Polymorph, temperature: Quantity,
                        include_landau: bool = True,
                        allow_extrapolation: bool = False) -> Quantity:
    """Molar isobaric heat capacity from equation (1), plus the Landau excess (3).

    Parameters
    ----------
    phase
        Polymorph. ``ALPHA_QUARTZ`` and ``BETA_QUARTZ`` both resolve to the quartz
        endmember, which is correct: HP treat them as one phase with a Landau correction.
    temperature
        Temperature.
    include_landau
        Add the excess Cp of the order-disorder transition where the phase has one. The
        excess diverges at T_c, which is physical: the inversion absorbs heat over a wide
        interval rather than at a single point.
    allow_extrapolation
        Permit evaluation outside the assumed validity range.

    Returns
    -------
    Quantity
        Cp in J/(mol K), strictly positive.

    Examples
    --------
    Alpha-quartz at 298.15 K, base polynomial only:

    >>> from ae.core.units import Q_
    >>> cp = molar_heat_capacity(Polymorph.ALPHA_QUARTZ, Q_(298.15, "K"),
    ...                          include_landau=False)
    >>> round(float(cp.magnitude), 4)
    43.1942
    """
    coeffs = _coeffs(phase)
    t_k = float(require_dimensionality(temperature, "temperature", "temperature")
                .to("K").magnitude)
    if t_k <= 0.0:
        raise ValueError(f"absolute temperature must be positive, got {t_k} K")
    _check_range(coeffs, t_k, allow_extrapolation)
    a, b, c, d = coeffs.raw
    cp = a + b * t_k + c / (t_k * t_k) + d / math.sqrt(t_k)
    if include_landau and coeffs.landau is not None:
        cp += float(landau_excess_heat_capacity(temperature, coeffs.landau)
                    .to("J/(mol*K)").magnitude)
    if cp <= 0.0:
        raise ValueError(
            f"Cp for {phase.value} evaluated to {cp:.3f} J/(mol K) at {t_k:.2f} K, which "
            f"is unphysical and means the polynomial is being used outside its range"
        )
    return Q_(cp, "J/(mol*K)")


def specific_heat_capacity(phase: Polymorph, temperature: Quantity,
                           include_landau: bool = True,
                           allow_extrapolation: bool = False) -> Quantity:
    """Isobaric heat capacity per unit mass, J/(kg K), from ``Cp / M``.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> c = specific_heat_capacity(Polymorph.ALPHA_QUARTZ, Q_(298.15, "K"),
    ...                            include_landau=False)
    >>> round(float(c.to("J/(kg*K)").magnitude), 1)
    718.9
    """
    cp = molar_heat_capacity(phase, temperature, include_landau, allow_extrapolation)
    return (cp / Q_(M_SIO2, "kg/mol")).to("J/(kg*K)")


def landau_excess_heat_capacity(temperature: Quantity,
                                params: LandauParameters = QUARTZ_LANDAU) -> Quantity:
    """Excess Cp of a tricritical order-disorder transition, equation (3).

    Returns zero at and above T_c, and diverges as T approaches T_c from below.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> round(float(landau_excess_heat_capacity(Q_(298.15, "K")).magnitude), 4)
    1.0823
    >>> float(landau_excess_heat_capacity(Q_(1000.0, "K")).magnitude)
    0.0
    """
    q = order_parameter(temperature, params)
    if q <= 1.0e-12:
        return Q_(0.0, "J/(mol*K)")
    t_k = float(temperature.to("K").magnitude)
    tc0 = float(params.tc_0.quantity.to("K").magnitude)
    s_d = float(params.s_d.quantity.to("J/(mol*K)").magnitude)
    cp_ex = t_k * s_d / (2.0 * tc0 * q * q)
    assert cp_ex > 0.0, "excess heat capacity of an ordering transition is positive"
    return Q_(cp_ex, "J/(mol*K)")


def landau_excess_enthalpy(temperature: Quantity,
                           params: LandauParameters = QUARTZ_LANDAU) -> Quantity:
    """Excess molar enthalpy H_ex of the ordering transition, equation (3), J/mol.

    Notes
    -----
    Referenced so that ``H_ex(T) - H_ex(298.15 K)`` is the heat absorbed by the inversion
    between those temperatures. For quartz that difference from 298.15 K to T_c is
    2646 J/mol (44.0 kJ/kg), which is the inversion's whole energy cost.
    """
    t_k = float(require_dimensionality(temperature, "temperature", "temperature")
                .to("K").magnitude)
    tc0 = float(params.tc_0.quantity.to("K").magnitude)
    s_d = float(params.s_d.quantity.to("J/(mol*K)").magnitude)
    q = order_parameter(temperature, params)
    q0 = order_parameter(Q_(T_REF, "K"), params)
    g_ex = (tc0 * s_d * (q0 ** 2 - q0 ** 6 / 3.0)
            - s_d * (tc0 * q ** 2 - tc0 * q ** 6 / 3.0)
            - t_k * s_d * (q0 ** 2 - q ** 2))
    s_ex = -s_d * (q * q - q0 * q0)
    return Q_(g_ex + t_k * s_ex, "J/mol")


def integrated_enthalpy(phase: Polymorph, t_from: Quantity, t_to: Quantity,
                        include_landau: bool = True,
                        allow_extrapolation: bool = False) -> Quantity:
    """Sensible enthalpy of one phase between two temperatures, equation (2), J/mol.

    Parameters
    ----------
    phase
        Polymorph, assumed not to transform over the interval. The caller is responsible
        for splitting a path at its transitions, which is what :func:`phase_enthalpy` and
        :func:`specific_transition_enthalpy` do.
    t_from, t_to
        Interval endpoints. Reversing them negates the result, as an enthalpy difference
        should.

    Returns
    -------
    Quantity
        Enthalpy change in J/mol, positive on heating.

    Examples
    --------
    Alpha to beta quartz from 298.15 K to 1000 K, base polynomial only. Hand check with
    F(T) = 92.9 T - 3.21e-4 T^2 + 714900/T - 1432.2 sqrt(T), term by term:

      F(1000)   = 92900.0000 - 321.0000 + 714.9000 - 45290.1406 = 48003.7594
      F(298.15) = 27698.1350 -  28.5348 + 2397.7863 - 24729.8269 =  5337.5597
      ----------------------------------------------------------------------
      difference                                                 = 42666.1997 J/mol

    (1432.2 sqrt(1000) = 1432.2 x 31.6227766 = 45290.1406 and
    1432.2 sqrt(298.15) = 1432.2 x 17.2670206 = 24729.8269)

    >>> from ae.core.units import Q_
    >>> dh = integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(1000.0, "K"),
    ...                          include_landau=False)
    >>> round(float(dh.magnitude), 1)
    42666.2
    """
    coeffs = _coeffs(phase)
    t1 = float(require_dimensionality(t_from, "temperature", "t_from").to("K").magnitude)
    t2 = float(require_dimensionality(t_to, "temperature", "t_to").to("K").magnitude)
    for t in (t1, t2):
        if t <= 0.0:
            raise ValueError(f"absolute temperature must be positive, got {t} K")
        _check_range(coeffs, t, allow_extrapolation)
    a, b, c, d = coeffs.raw

    def antideriv(t: float) -> float:
        return a * t + 0.5 * b * t * t - c / t + 2.0 * d * math.sqrt(t)

    dh = antideriv(t2) - antideriv(t1)
    if include_landau and coeffs.landau is not None:
        dh += float((landau_excess_enthalpy(Q_(t2, "K"), coeffs.landau)
                     - landau_excess_enthalpy(Q_(t1, "K"), coeffs.landau)).magnitude)
    if t2 > t1 and dh <= 0.0:
        raise ValueError(
            f"heating {phase.value} from {t1:.2f} to {t2:.2f} K returned a non-positive "
            f"enthalpy change ({dh:.3f} J/mol), which violates Cp > 0"
        )
    return Q_(dh, "J/mol")


def phase_enthalpy(phase: Polymorph, temperature: Quantity,
                   allow_extrapolation: bool = False) -> Quantity:
    """Absolute molar enthalpy ``H(T) = H^0_f + int Cp dT + H_ex``, J/mol.

    Absolute in the sense that all polymorphs share the same 298.15 K formation-enthalpy
    reference, so differences between phases are physically meaningful. That is what makes
    equation (4) able to carry the transition enthalpy without a separate latent-heat term.
    """
    h0 = float(FORMATION_ENTHALPY[_resolve(phase)].quantity.to("J/mol").magnitude)
    sens = float(integrated_enthalpy(phase, Q_(T_REF, "K"), temperature,
                                     include_landau=True,
                                     allow_extrapolation=allow_extrapolation).magnitude)
    return Q_(h0 + sens, "J/mol")


def specific_transition_enthalpy(from_phase: Polymorph, to_phase: Polymorph,
                                 temperature: Quantity,
                                 allow_extrapolation: bool = False) -> Quantity:
    """Specific enthalpy of transformation at one temperature, J/kg.

    Examples
    --------
    Quartz to cristobalite at 1743.15 K (1470 degC):

    >>> from ae.core.units import Q_
    >>> dh = specific_transition_enthalpy(Polymorph.QUARTZ, Polymorph.CRISTOBALITE,
    ...                                    Q_(1743.15, "K"), allow_extrapolation=True)
    >>> round(float(dh.to("kJ/kg").magnitude), 1)
    50.9
    """
    h1 = phase_enthalpy(from_phase, temperature, allow_extrapolation)
    h2 = phase_enthalpy(to_phase, temperature, allow_extrapolation)
    return ((h2 - h1) / Q_(M_SIO2, "kg/mol")).to("J/kg")


class ThermalStep(BaseModel):
    """One furnace step: a path through phases and temperatures, with its efficiency.

    ``eta_thermal`` and ``loss_fraction`` are separate because they describe different
    things: the first is how much of the energy input reaches the charge, the second is an
    additional deduction for losses the caller wants itemised. Overall efficiency is
    ``eta_thermal * (1 - loss_fraction)`` and is asserted to lie in (0, 1].
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: str
    from_phase: Polymorph
    to_phase: Polymorph
    t_start: Quantity
    t_end: Quantity
    eta_thermal: Value
    loss_fraction: Value | _Missing = MISSING
    transition_temperature: Quantity | None = Field(
        default=None,
        description="Temperature at which the phase change is taken to occur. Required "
        "when from_phase differs from to_phase: this module refuses to infer a melting or "
        "transformation temperature from the ds62/ds633 datasets, which do not reproduce "
        "the 1 bar silica polymorph boundaries (LIMITATIONS items 3 and 4).",
    )

    @model_validator(mode="after")
    def _dimensions(self) -> ThermalStep:
        require_dimensionality(self.t_start, "temperature", "t_start")
        require_dimensionality(self.t_end, "temperature", "t_end")
        eta = require_fraction(
            float(self.eta_thermal.quantity.to("dimensionless").magnitude),
            "eta_thermal", lo=1e-6, hi=1.0)
        if isinstance(self.loss_fraction, Value):
            require_fraction(
                float(self.loss_fraction.quantity.to("dimensionless").magnitude),
                "loss_fraction", lo=0.0, hi=0.999999)
        if _resolve(self.from_phase) != _resolve(self.to_phase):
            if self.transition_temperature is None:
                raise ValueError(
                    f"step {self.name!r} changes phase from {self.from_phase.value} to "
                    f"{self.to_phase.value} but no transition_temperature was given. "
                    f"Supply it from ae.physics.phases.TRANSITIONS, which sources the "
                    f"temperatures; the thermodynamic datasets used here do not reproduce "
                    f"the 1 bar silica polymorph boundaries."
                )
            require_dimensionality(self.transition_temperature, "temperature",
                                   "transition_temperature")
        assert eta > 0.0
        return self

    @property
    def overall_efficiency(self) -> float:
        """Overall efficiency, dimensionless, in (0, 1]."""
        eta = float(self.eta_thermal.quantity.to("dimensionless").magnitude)
        loss = (float(self.loss_fraction.quantity.to("dimensionless").magnitude)
                if isinstance(self.loss_fraction, Value) else 0.0)
        overall = eta * (1.0 - loss)
        if not 0.0 < overall <= 1.0:
            raise ValueError(
                f"overall efficiency {overall} is outside (0, 1]: an efficiency above unity "
                f"would violate the first law"
            )
        return overall


class EnergyBalance(BaseModel):
    """Result of a thermal-step energy balance, itemised so the balance can be checked.

    ``theoretical`` is what SiO2 demands, ``supplied`` is what the furnace must input, and
    ``losses`` is the difference. ``supplied = theoretical + losses`` is asserted on
    construction, which is the first law closing.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    step_name: str
    theoretical: Quantity
    supplied: Quantity
    losses: Quantity
    overall_efficiency: float
    sensible: Quantity
    transition: Quantity
    is_scenario: bool
    uses_liquid_cp_for_glass: bool = False
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _balance_closes(self) -> EnergyBalance:
        for field in ("theoretical", "supplied", "losses", "sensible", "transition"):
            require_dimensionality(getattr(self, field), "specific_energy_mass", field)
        theo = float(self.theoretical.to("J/kg").magnitude)
        sup = float(self.supplied.to("J/kg").magnitude)
        loss = float(self.losses.to("J/kg").magnitude)
        sens = float(self.sensible.to("J/kg").magnitude)
        trans = float(self.transition.to("J/kg").magnitude)
        if theo < 0.0 or sup < 0.0 or loss < 0.0:
            raise ValueError("specific energies in a heating balance must be non-negative")
        if abs(theo + loss - sup) > 1e-6 * max(1.0, abs(sup)):
            raise ValueError(
                f"energy balance does not close: theoretical {theo:.6g} + losses "
                f"{loss:.6g} != supplied {sup:.6g} J/kg"
            )
        if abs(sens + trans - theo) > 1e-6 * max(1.0, abs(theo)):
            raise ValueError(
                f"itemisation does not close: sensible {sens:.6g} + transition "
                f"{trans:.6g} != theoretical {theo:.6g} J/kg"
            )
        if not 0.0 < self.overall_efficiency <= 1.0:
            raise ValueError("overall efficiency must lie in (0, 1]")
        return self

    @property
    def kwh_per_tonne(self) -> float:
        """Supplied specific energy in kWh per tonne, the industry unit."""
        return float(self.supplied.to("kWh/tonne").magnitude)

    @property
    def theoretical_kwh_per_tonne(self) -> float:
        return float(self.theoretical.to("kWh/tonne").magnitude)


def _balance(step: ThermalStep, feedstock: Feedstock,
             allow_extrapolation: bool) -> EnergyBalance:
    """Shared machinery for the public step functions. Splits a path at its transition."""
    if not isinstance(feedstock, Feedstock):
        raise TypeError(
            "every thermal model in this platform takes a Feedstock, so nothing is "
            "hardcoded to one deposit"
        )
    t1 = float(step.t_start.to("K").magnitude)
    t2 = float(step.t_end.to("K").magnitude)
    if t2 < t1:
        raise ValueError(
            f"step {step.name!r} ends below its start temperature; use "
            f"quench_heat_rejection for cooling"
        )
    notes: list[str] = []
    same_phase = _resolve(step.from_phase) == _resolve(step.to_phase)
    if same_phase:
        sens = float(integrated_enthalpy(step.from_phase, step.t_start, step.t_end,
                                         allow_extrapolation=allow_extrapolation)
                     .magnitude) / M_SIO2
        trans = 0.0
    else:
        assert step.transition_temperature is not None
        t_tr = step.transition_temperature
        t_tr_k = float(t_tr.to("K").magnitude)
        if not t1 <= t_tr_k <= t2:
            raise ValueError(
                f"step {step.name!r}: transition temperature {t_tr_k:.2f} K lies outside "
                f"the traversed interval {t1:.2f} to {t2:.2f} K, so the phase change "
                f"cannot occur on this path"
            )
        leg1 = float(integrated_enthalpy(step.from_phase, step.t_start, t_tr,
                                         allow_extrapolation=allow_extrapolation).magnitude)
        leg2 = float(integrated_enthalpy(step.to_phase, t_tr, step.t_end,
                                         allow_extrapolation=allow_extrapolation).magnitude)
        sens = (leg1 + leg2) / M_SIO2
        trans = float(specific_transition_enthalpy(
            step.from_phase, step.to_phase, t_tr,
            allow_extrapolation=allow_extrapolation).to("J/kg").magnitude)
        notes.append(
            f"path split at the supplied transition temperature {t_tr_k:.2f} K; the "
            f"transition enthalpy is the difference of absolute phase enthalpies there"
        )
    theo = sens + trans
    if theo < 0.0:
        raise ValueError(
            f"step {step.name!r} returned a negative theoretical energy demand "
            f"({theo:.4g} J/kg) for a heating path, which is unphysical"
        )
    eta = step.overall_efficiency
    sup = theo / eta
    proxy = any(_coeffs(p).is_liquid_proxy_for_glass
                for p in (step.from_phase, step.to_phase))
    if proxy:
        notes.append(
            "silica glass Cp is the ds633 LIQUID endmember standing in for glass: no "
            "glass-specific Cp could be sourced from this sandbox (LIMITATIONS item 1)"
        )
    if not feedstock.characterized:
        notes.append(
            f"feedstock {feedstock.sample_id} is not characterized, so this is a SCENARIO "
            f"gated on a characterization campaign, not a finding"
        )
    return EnergyBalance(
        step_name=step.name,
        theoretical=Q_(theo, "J/kg"),
        supplied=Q_(sup, "J/kg"),
        losses=Q_(sup - theo, "J/kg"),
        overall_efficiency=eta,
        sensible=Q_(sens, "J/kg"),
        transition=Q_(trans, "J/kg"),
        is_scenario=not feedstock.characterized,
        uses_liquid_cp_for_glass=proxy,
        notes=tuple(notes),
    )


def calcination_energy(feedstock: Feedstock, step: ThermalStep,
                       allow_extrapolation: bool = False) -> EnergyBalance:
    """Specific energy to calcine a quartz feedstock, kWh per tonne inside the result.

    Calcination here means heating without a reconstructive phase change: the charge
    crosses the alpha to beta inversion, fluid inclusions pressurise and decrepitate, and
    the product is still quartz. The inversion's energy cost is included automatically
    through the Landau excess enthalpy.

    Parameters
    ----------
    feedstock
        The ore. Required by platform rule and used to mark the result a scenario when the
        ore is uncharacterized.
    step
        The thermal step. ``from_phase`` and ``to_phase`` are normally both quartz.
    allow_extrapolation
        Permit Cp evaluation outside the assumed validity range.

    Returns
    -------
    EnergyBalance

    Notes
    -----
    This function deliberately does NOT predict decrepitation or impurity removal. The
    fluid-inclusion population that governs both is a measured feedstock property
    (``feedstock.inclusions``, ``feedstock.physical.decrepitation_index``) and raises when
    unmeasured.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> from ae.core.feedstock import Feedstock, OreType
    >>> from ae.core.provenance import Tag, Value
    >>> f = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
    ...               deposit_name="Vikarabad", country="IN")
    >>> s = ThermalStep(name="calcine", from_phase=Polymorph.ALPHA_QUARTZ,
    ...                 to_phase=Polymorph.BETA_QUARTZ, t_start=Q_(298.15, "K"),
    ...                 t_end=Q_(1173.15, "K"),
    ...                 eta_thermal=Value(quantity=Q_(1.0, "dimensionless"),
    ...                                   tag=Tag.ASSUMED, basis="doctest, no losses"))
    >>> bal = calcination_energy(f, s)
    >>> bal.is_scenario
    True
    """
    return _balance(step, feedstock, allow_extrapolation)


def fusion_energy(feedstock: Feedstock, step: ThermalStep,
                  allow_extrapolation: bool = True) -> EnergyBalance:
    """Specific energy to fuse silica into a melt or glass, kWh per tonne in the result.

    ``allow_extrapolation`` defaults True here because fusion necessarily runs to 1996 K or
    beyond, past the 1800 K bound assumed for the quartz polynomial. That is a real
    extrapolation and is recorded as such rather than hidden.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> from ae.core.feedstock import Feedstock, OreType
    >>> from ae.core.provenance import Tag, Value
    >>> f = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
    ...               deposit_name="Vikarabad", country="IN")
    >>> s = ThermalStep(name="fuse", from_phase=Polymorph.QUARTZ,
    ...                 to_phase=Polymorph.SILICA_LIQUID, t_start=Q_(298.15, "K"),
    ...                 t_end=Q_(1996.0, "K"), transition_temperature=Q_(1996.0, "K"),
    ...                 eta_thermal=Value(quantity=Q_(1.0, "dimensionless"),
    ...                                   tag=Tag.ASSUMED, basis="doctest, no losses"))
    >>> round(fusion_energy(f, s).theoretical_kwh_per_tonne, 1)
    599.7
    """
    return _balance(step, feedstock, allow_extrapolation)


def quench_heat_rejection(feedstock: Feedstock, phase: Polymorph, t_hot: Quantity,
                          t_cold: Quantity, duration: Quantity,
                          allow_extrapolation: bool = False
                          ) -> tuple[Quantity, Quantity]:
    """Heat rejected in a quench and the mean cooling rate, equation (6).

    Parameters
    ----------
    feedstock
        The ore, per platform rule.
    phase
        Phase of the material being quenched.
    t_hot, t_cold
        Start and end temperatures. ``t_cold`` must be below ``t_hot``.
    duration
        Quench duration, used only for the MEAN cooling rate.

    Returns
    -------
    (Quantity, Quantity)
        Specific heat rejected, J/kg (positive), and mean cooling rate, K/s.

    Notes
    -----
    The mean cooling rate is not a thermal-shock criterion. Fracture is driven by the
    surface temperature gradient and by the fluid-inclusion population, and this function
    has neither. See LIMITATIONS item 6.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> from ae.core.feedstock import Feedstock, OreType
    >>> f = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
    ...               deposit_name="Vikarabad", country="IN")
    >>> q, rate = quench_heat_rejection(f, Polymorph.QUARTZ, Q_(1173.15, "K"),
    ...                                 Q_(298.15, "K"), Q_(60.0, "s"))
    >>> float(q.magnitude) > 0.0, round(float(rate.to("K/s").magnitude), 3)
    (True, 14.583)
    """
    if not isinstance(feedstock, Feedstock):
        raise TypeError("quench_heat_rejection requires a Feedstock")
    t_h = float(require_dimensionality(t_hot, "temperature", "t_hot").to("K").magnitude)
    t_c = float(require_dimensionality(t_cold, "temperature", "t_cold").to("K").magnitude)
    dt = float(require_dimensionality(duration, "time", "duration").to("s").magnitude)
    if t_c >= t_h:
        raise ValueError(f"a quench must cool: t_cold {t_c} K is not below t_hot {t_h} K")
    if dt <= 0.0:
        raise ValueError("quench duration must be positive")
    dh = float(integrated_enthalpy(phase, t_cold, t_hot,
                                   allow_extrapolation=allow_extrapolation).magnitude)
    q_spec = dh / M_SIO2
    if q_spec <= 0.0:
        raise ValueError(
            f"quench heat rejection computed as {q_spec:.4g} J/kg, which is non-positive "
            f"and violates the second law for a body cooling to a colder medium"
        )
    return Q_(q_spec, "J/kg"), Q_((t_h - t_c) / dt, "K/s")


def electricity_cost(balance: EnergyBalance, site: Site) -> Quantity:
    """Cost of the supplied energy at a site's electricity price, per tonne of product.

    Parameters
    ----------
    balance
        Result of a step energy balance.
    site
        The plant location. Its ``power.energy_price`` supplies both the rate and the
        currency, so no exchange rate is applied and none is implied.

    Returns
    -------
    Quantity
        Cost per tonne, in the site's own currency.

    Notes
    -----
    Electricity only. Demand charges, outage-driven lost production and fuel-fired
    alternatives are site fields this function does not touch, and a thermal step run on
    natural gas must not be costed with this.
    """
    if not isinstance(site, Site):
        raise TypeError("electricity_cost requires a Site so no tariff is hardcoded")
    price = site.power.energy_price.quantity
    energy = balance.supplied.to("kWh/tonne")
    cost = (energy * price).to_reduced_units()
    unit = str(cost.units)
    if "USD" not in unit and "INR" not in unit:
        raise ValueError(
            f"site {site.site_id} energy price {price.units} did not produce a currency per "
            f"mass cost (got {unit}); the price must be currency per energy"
        )
    if float(cost.magnitude) < 0.0:
        raise ValueError("energy cost cannot be negative")
    return cost
