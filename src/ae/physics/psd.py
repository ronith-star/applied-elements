r"""Particle size distributions: log-normal and Rosin-Rammler, moments, and the
number-area-volume weighting conversions that are the commonest error source in the field.

Why the weighting conversions get their own module section: a laser-diffraction instrument
reports a VOLUME-weighted distribution, a microscope image analysis reports a
NUMBER-weighted one, and a BET surface area is an AREA-weighted property. For a log-normal
with geometric standard deviation 1.5 the number-median and volume-median diameters differ
by 64 percent. Quoting a d50 without its weighting basis is therefore not a rounding issue,
it is a different number, and mixing the two silently propagates into every downstream
surface-area, packing and viscosity estimate.

EQUATIONS
---------
(1) Log-normal number density (Hatch and Choate 1929, doi 10.1016/s0016-0032(29)91451-4):

    .. math::

        f_0(d) = \frac{1}{d\,s\,\sqrt{2\pi}}
        \exp\left[-\frac{(\ln d - \ln d_{gn})^2}{2 s^2}\right]

    Symbols:
      f_0       number density function, 1/m (per unit diameter).
      d         particle diameter, m, valid range 1e-9 to 1e-2 m.
      d_{gn}    geometric (count) median diameter, m. Also the count geometric mean.
      s         ln(sigma_g), dimensionless, valid range 0 (monodisperse) to about 1.2.
                Above s of roughly 1.2 the log-normal form is rarely defensible for a
                comminution product.
      sigma_g   geometric standard deviation, dimensionless, >= 1.

(2) Moments of the log-normal (derived, not cited: substitute u = ln d in
    :math:`\int d^k f_0(d)\,dd` and complete the square):

    .. math::

        M_k \equiv \int_0^\infty d^k f_0(d)\,dd = d_{gn}^k \exp\!\left(\frac{k^2 s^2}{2}\right)

    Symbols: M_k the k-th raw moment about the origin, units m^k; k the order, any real.

(3) Hatch-Choate conversions between weighted median diameters (derived from (2)):

    .. math::

        d_{ga} = d_{gn}\exp(2 s^2), \qquad
        d_{gv} = d_{gn}\exp(3 s^2), \qquad
        d_{32} = d_{gn}\exp\left(\tfrac{5}{2}s^2\right)

    Symbols:
      d_{ga}   area-weighted (surface) median diameter, m.
      d_{gv}   volume-weighted (mass) median diameter, m, which is what a laser
               diffraction d50 reports.
      d_{32}   Sauter mean diameter, m, defined as M_3/M_2 and the only mean that
               preserves the surface-to-volume ratio.

    Derivation of d_{32}: :math:`M_3/M_2 = d_{gn}\exp(9s^2/2)/\exp(4s^2/2)
    = d_{gn}\exp(5s^2/2)`. A log-normal stays log-normal under reweighting with the same
    s and a shifted median, which is the property that makes (3) exact rather than
    approximate.

(4) Quantiles of the log-normal:

    .. math::

        d_p = d_{g}\exp\!\left[z_p\,s\right]

    with :math:`z_p` the standard normal quantile (z_{0.1} = -1.281552,
    z_{0.9} = +1.281552) and :math:`d_g` the median on the chosen weighting basis.

(5) Span, the standard polydispersity index:

    .. math::

        \mathrm{span} = \frac{d_{90} - d_{10}}{d_{50}}

(6) Rosin-Rammler cumulative volume undersize (Rosin and Rammler 1933, as characterised by
    Alderliesten 2013, doi 10.1002/ppsc.201200021):

    .. math::

        Q_3(d) = 1 - \exp\left[-\left(\frac{d}{d'}\right)^{n}\right]

    Symbols:
      Q_3      cumulative VOLUME fraction undersize, dimensionless, range [0, 1].
      d        diameter, m.
      d'       location parameter, m: the size at which Q_3 = 1 - 1/e = 0.6321.
      n        spread parameter, dimensionless. Alderliesten 2013 shows that
               distributions with n < 3 "cannot exist" as physically complete
               distributions and can at most describe the measured range, so this module
               warns below n = 3 (see LIMITATIONS).

(7) Rosin-Rammler moments and derived diameters (derived by substituting
    :math:`x = (d/d')^n`, which turns the integral into a gamma function):

    .. math::

        d_{50} = d'\,(\ln 2)^{1/n}, \qquad
        \mathbb{E}_v[d] = d'\,\Gamma\!\left(1 + \tfrac{1}{n}\right), \qquad
        d_{32} = \frac{d'}{\Gamma\!\left(1 - \tfrac{1}{n}\right)}

    Symbols: :math:`\Gamma` the gamma function; :math:`\mathbb{E}_v[d]` the volume-weighted
    mean diameter, m. The d_{32} expression requires n > 1 for the gamma argument to be
    positive, and diverges as n approaches 1.

    Derivation of d_{32} for Rosin-Rammler: the specific surface per unit volume of solid
    is :math:`\int (6/d)\,dQ_3`, and with the substitution above
    :math:`\int_0^\infty d^{-1} dQ_3 = \Gamma(1 - 1/n)/d'`, so
    :math:`S_v = 6\Gamma(1-1/n)/d'` and :math:`d_{32} \equiv 6/S_v = d'/\Gamma(1-1/n)`.

(8) Specific surface area from a distribution:

    .. math::

        S_m = \frac{6}{\rho\,d_{32}\,\psi}

    Symbols:
      S_m   specific surface area, m^2/kg.
      \rho  true particle density, kg/m^3 (2650 for quartz).
      d_32  Sauter mean diameter, m.
      \psi  sphericity, dimensionless, range (0, 1]. Defaults to 1 (spheres) and must be
            supplied for an angular comminution product.

    Derivation: for a sphere, area/volume = 6/d exactly; integrating over a distribution
    replaces d by its surface-volume mean d_32 by construction, and dividing by density
    converts per-volume to per-mass. Non-spherical particles have more area at the same
    volume, so dividing by psi <= 1 raises S_m.

LIMITATIONS
-----------
1. S_m IS A GEOMETRIC SURFACE AREA, NOT A BET SURFACE AREA. Equation (8) counts only the
   external envelope of a smooth convex particle. Real quartz particles have surface
   roughness, internal cracks and (after calcination) decrepitated inclusion cavities, all
   of which BET nitrogen adsorption sees and equation (8) does not. Ringdalen 2015
   (doi 10.1007/s11837-014-1149-y) measured 0.38 to 0.47 m^2/g on heat-treated natural
   quartz lump, values that a geometric calculation on lump-sized particles underestimates
   by orders of magnitude. Expect equation (8) to UNDERSTATE BET area, by a factor that is
   a property of the ore and its thermal history, and never substitute one for the other.
2. SPHERICITY IS AN INPUT, NOT A PREDICTION. No source reachable here gives a sphericity
   for a crushed vein quartz product, so psi defaults to 1 and any other value must be
   supplied by the caller with its own basis.
3. THE ROSIN-RAMMLER LOCATION PARAMETER IS NOT PHYSICALLY INTERPRETABLE. Alderliesten 2013
   shows that the physical meaning of d' depends on the value of n, which makes it
   unsuitable as a model input even though it remains useful for production control. Use
   d_32 or a moment-ratio mean for physical models, which is why this module exposes both.
4. n < 3 IS NOT A VALID DISTRIBUTION. Per Alderliesten 2013, Rosin-Rammler fits with
   n < 3 describe the measured window only and must not be extrapolated to the fine tail.
   This module refuses to integrate moments for n <= 1 and flags 1 < n < 3.
5. NO COMMINUTION MODEL. Nothing here predicts what PSD a mill produces from a given ore.
   That needs the Bond work index, which is a measured feedstock property
   (``feedstock.physical.bond_work_index``) and is absent for an uncharacterized deposit.
6. SINGLE-POPULATION FORMS ONLY. Both distributions are unimodal by construction. A real
   graded filler blend is deliberately multimodal and must be handled as a mixture of
   these objects, which is what :mod:`ae.physics.packing` does.
7. NO SIZE-DEPENDENT COMPOSITION. Impurity content very often varies with size fraction
   (feldspar and mica report to specific sizes in a crushed quartz), and this module
   carries no composition at all. Pair it with a measured ``ImpurityProfile`` per fraction.

TYPE CHECKING NOTE
------------------
``mypy --strict`` reports ``Quantity? has no attribute "to"`` and ``Variable
"ae.core.units.Quantity" is not valid as a type`` against this module. Those errors
originate in ``ae.core.units``, which declares ``Quantity = UREG.Quantity`` (a variable
binding, not a type alias), so mypy cannot treat it as a type. The same errors appear
against the four core modules themselves (46 of them), and the core modules are not ours
to change. Every annotation here follows the core convention deliberately rather than
diverging from it for a clean checker run.
"""

from __future__ import annotations

import datetime as _dt
import enum
import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = [
    "SRC_ALDERLIESTEN",
    "SRC_HATCH_CHOATE",
    "SRC_RINGDALEN_SSA",
    "Z_10",
    "Z_90",
    "LogNormalPSD",
    "PSDSummary",
    "RosinRammlerPSD",
    "Weighting",
    "convert_lognormal_median",
    "feedstock_sphericity_assumption",
    "specific_surface_area",
]

_ACCESSED: Final[_dt.date] = _dt.date(2026, 9, 16)

#: Standard normal quantiles for the 10th and 90th percentiles, used in equation (4).
#: scipy.stats.norm.ppf(0.1) = -1.2815515655446004.
Z_10: Final[float] = -1.2815515655446004
Z_90: Final[float] = 1.2815515655446004

SRC_HATCH_CHOATE: Final[Source] = Source(
    citation=(
        "Hatch, T. and Choate, S.P. 1929, Statistical description of the size properties of "
        "non-uniform particulate substances, Journal of the Franklin Institute 207:369-387"
    ),
    tier=Tier.T1,
    doi="10.1016/s0016-0032(29)91451-4",
    accessed=_ACCESSED,
    note=(
        "Original source of the log-normal moment conversions. The full text is not "
        "reachable from this sandbox; the conversions implemented here are DERIVED from "
        "equation (2) of this module rather than transcribed, so the derivation stands on "
        "its own and the citation is attributional."
    ),
)

SRC_ALDERLIESTEN: Final[Source] = Source(
    citation=(
        "Alderliesten, M. 2013, Mean Particle Diameters. Part VII. The Rosin-Rammler Size "
        "Distribution: Physical and Mathematical Properties and Relationships to "
        "Moment-Ratio Defined Mean Particle Diameters, Particle and Particle Systems "
        "Characterization 30:244-257"
    ),
    tier=Tier.T1,
    doi="10.1002/ppsc.201200021",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Used for two statements taken from the published abstract (full text paywalled "
        "from this sandbox): the Rosin-Rammler volume density is identical to the Weibull "
        "density, and distributions with spread parameter below three cannot exist."
    ),
)

SRC_RINGDALEN_SSA: Final[Source] = Source(
    citation=(
        "Ringdalen, E. 2015, Changes in Quartz During Heating and the Possible Effects on "
        "Si Production, JOM 67:484-492"
    ),
    tier=Tier.T1,
    doi="10.1007/s11837-014-1149-y",
    accessed=_ACCESSED,
    extraction="manual",
    note="BET specific surface areas of 0.38 to 0.47 m^2/g on heat-treated natural quartz.",
)


class Weighting(str, enum.Enum):
    """Distribution weighting basis. Mixing these is the error this module exists to stop.

    NUMBER  : count-weighted, what image analysis and single-particle counters report.
    AREA    : surface-weighted.
    VOLUME  : volume or mass weighted, what laser diffraction and sieving report.
    """

    NUMBER = "number"
    AREA = "area"
    VOLUME = "volume"


class PSDSummary(BaseModel):
    """Descriptors of a distribution on one stated weighting basis."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    weighting: Weighting
    d10: Quantity
    d50: Quantity
    d90: Quantity
    span: float
    sauter_d32: Quantity
    specific_surface_area: Quantity

    @model_validator(mode="after")
    def _checks(self) -> PSDSummary:
        for field in ("d10", "d50", "d90", "sauter_d32"):
            q = require_dimensionality(getattr(self, field), "length", field)
            if float(q.to("m").magnitude) <= 0.0:
                raise ValueError(f"{field} must be a positive diameter")
        require_dimensionality(self.specific_surface_area, "specific_surface_area",
                               "specific_surface_area")
        d10 = float(self.d10.to("m").magnitude)
        d50 = float(self.d50.to("m").magnitude)
        d90 = float(self.d90.to("m").magnitude)
        if not d10 <= d50 <= d90:
            raise ValueError(
                f"quantiles are not monotonic: d10 {d10:.4g} <= d50 {d50:.4g} <= d90 "
                f"{d90:.4g} m is violated"
            )
        if self.span < 0.0:
            raise ValueError("span cannot be negative")
        if float(self.specific_surface_area.to("m**2/kg").magnitude) <= 0.0:
            raise ValueError("specific surface area must be positive")
        return self


def feedstock_sphericity_assumption(feedstock: Feedstock) -> Value:
    """Sphericity to use for a feedstock, as an explicitly ASSUMED Value.

    No sphericity measurement for a crushed vein quartz product was obtainable from any
    source reachable here, and sphericity is not a field of
    :class:`ae.core.feedstock.PhysicalProperties`, so this returns an honest ASSUMED value
    of unity rather than a fabricated number. Override it wherever a measurement exists.
    """
    if not isinstance(feedstock, Feedstock):
        raise TypeError("feedstock_sphericity_assumption requires a Feedstock")
    return Value(
        quantity=Q_(1.0, "dimensionless"),
        tag=Tag.ASSUMED,
        basis=(
            f"ESTIMATE, sphericity 1.0 (perfect spheres) for {feedstock.sample_id}. No "
            f"sphericity measurement for crushed quartz was obtainable from any source "
            f"reachable from this environment, and the core Feedstock model carries no "
            f"sphericity field. Unity is used because it makes the geometric surface area "
            f"of equation (8) a strict LOWER bound: a real angular comminution product has "
            f"psi below 1 and therefore MORE area per unit mass than this returns."
        ),
        confidence="low",
    )


def specific_surface_area(sauter_d32: Quantity, density: Quantity,
                          sphericity: Value | None = None) -> Quantity:
    """Geometric specific surface area from the Sauter mean diameter, equation (8).

    Parameters
    ----------
    sauter_d32
        Sauter mean diameter (M_3/M_2 of the distribution).
    density
        True particle density.
    sphericity
        Sphericity as a Value in (0, 1]. Defaults to 1, i.e. spheres.

    Returns
    -------
    Quantity
        Specific surface area, m^2/kg.

    Notes
    -----
    Geometric, not BET. See LIMITATIONS item 1.

    Examples
    --------
    A 15.0833 um Sauter mean at quartz density:

    >>> from ae.core.units import Q_
    >>> s = specific_surface_area(Q_(15.083327243425664, "um"), Q_(2650.0, "kg/m**3"))
    >>> round(float(s.to("m**2/kg").magnitude), 4)
    150.1095
    """
    d32 = require_dimensionality(sauter_d32, "length", "sauter_d32")
    rho = require_dimensionality(density, "density", "density")
    d_m = float(d32.to("m").magnitude)
    rho_v = float(rho.to("kg/m**3").magnitude)
    if d_m <= 0.0:
        raise ValueError("Sauter mean diameter must be positive")
    if rho_v <= 0.0:
        raise ValueError("density must be positive")
    psi = 1.0
    if sphericity is not None:
        psi = float(sphericity.quantity.to("dimensionless").magnitude)
        require_fraction(psi, "sphericity", lo=1e-6, hi=1.0)
    s = 6.0 / (rho_v * d_m * psi)
    assert s > 0.0, "specific surface area must be positive"
    return Q_(s, "m**2/kg")


class LogNormalPSD(BaseModel):
    """Log-normal particle size distribution, equations (1) to (5).

    The distribution is stored by its COUNT median ``d_gn`` and ``sigma_g``, and every
    other median is derived, so the weighting basis of a returned diameter is always
    explicit and never ambiguous.

    Parameters
    ----------
    d_gn
        Geometric count median diameter.
    sigma_g
        Geometric standard deviation, >= 1. Unity is monodisperse.
    density
        True particle density, used for surface-area outputs.
    sphericity
        Sphericity Value, default spheres.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> psd = LogNormalPSD(d_gn=Q_(10.0, "um"), sigma_g=1.5, density=Q_(2650.0, "kg/m**3"))
    >>> round(float(psd.median(Weighting.VOLUME).to("um").magnitude), 4)
    16.3756
    >>> round(float(psd.sauter_d32().to("um").magnitude), 4)
    15.0833
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    d_gn: Quantity
    sigma_g: float = Field(ge=1.0)
    density: Quantity
    sphericity: Value | None = None

    @model_validator(mode="after")
    def _checks(self) -> LogNormalPSD:
        require_dimensionality(self.d_gn, "length", "d_gn")
        require_dimensionality(self.density, "density", "density")
        if float(self.d_gn.to("m").magnitude) <= 0.0:
            raise ValueError("d_gn must be a positive diameter")
        if float(self.density.to("kg/m**3").magnitude) <= 0.0:
            raise ValueError("density must be positive")
        if self.sigma_g < 1.0:
            raise ValueError(
                f"geometric standard deviation must be at least 1 (1 is monodisperse), "
                f"got {self.sigma_g}"
            )
        if self.sphericity is not None:
            require_fraction(
                float(self.sphericity.quantity.to("dimensionless").magnitude),
                "sphericity", lo=1e-6, hi=1.0)
        return self

    @property
    def s(self) -> float:
        """``s = ln(sigma_g)``, the log-space standard deviation, dimensionless."""
        return math.log(self.sigma_g)

    def moment(self, k: float) -> Quantity:
        """k-th raw moment of the NUMBER distribution, equation (2), units m^k."""
        d = float(self.d_gn.to("m").magnitude)
        return Q_(d ** k * math.exp(0.5 * k * k * self.s * self.s), f"m**{k}")

    def median(self, weighting: Weighting = Weighting.NUMBER) -> Quantity:
        """Median diameter on a stated weighting basis, equation (3).

        Notes
        -----
        The exponents are 0, 2 and 3 for number, area and volume respectively. This is the
        conversion that laser diffraction versus microscopy comparisons get wrong.
        """
        exponent = {Weighting.NUMBER: 0.0, Weighting.AREA: 2.0, Weighting.VOLUME: 3.0}[
            weighting]
        return (self.d_gn * math.exp(exponent * self.s * self.s)).to(self.d_gn.units)

    def sauter_d32(self) -> Quantity:
        """Sauter mean diameter ``M_3/M_2 = d_gn exp(5 s^2 / 2)``, equation (3)."""
        return (self.d_gn * math.exp(2.5 * self.s * self.s)).to(self.d_gn.units)

    def quantile(self, p: float, weighting: Weighting = Weighting.VOLUME) -> Quantity:
        """Quantile diameter at cumulative fraction ``p`` on a weighting basis, eq (4)."""
        require_fraction(p, "p", lo=1e-9, hi=1.0 - 1e-9)
        z = _norm_ppf(p)
        return (self.median(weighting) * math.exp(z * self.s)).to(self.d_gn.units)

    def span(self, weighting: Weighting = Weighting.VOLUME) -> float:
        """Span ``(d90 - d10)/d50``, equation (5). Independent of weighting basis.

        Notes
        -----
        Because all three quantiles scale with the same median, span depends only on
        ``sigma_g``: ``span = exp(z90 s) - exp(z10 s)``. That is a useful check on a
        reported span, and it is why this method returns the same value on every basis.
        """
        d10 = float(self.quantile(0.1, weighting).to("m").magnitude)
        d50 = float(self.median(weighting).to("m").magnitude)
        d90 = float(self.quantile(0.9, weighting).to("m").magnitude)
        span = (d90 - d10) / d50
        assert span >= 0.0, "span cannot be negative"
        return span

    def specific_surface_area(self) -> Quantity:
        """Geometric specific surface area, equation (8), m^2/kg."""
        return specific_surface_area(self.sauter_d32(), self.density, self.sphericity)

    def summary(self, weighting: Weighting = Weighting.VOLUME) -> PSDSummary:
        """Full descriptor set on one stated basis."""
        return PSDSummary(
            weighting=weighting,
            d10=self.quantile(0.1, weighting),
            d50=self.median(weighting),
            d90=self.quantile(0.9, weighting),
            span=self.span(weighting),
            sauter_d32=self.sauter_d32(),
            specific_surface_area=self.specific_surface_area(),
        )


def convert_lognormal_median(median: Quantity, sigma_g: float, from_weighting: Weighting,
                             to_weighting: Weighting) -> Quantity:
    """Convert a log-normal median between weighting bases, equation (3).

    This is the single function to reach for when an instrument reports a d50 on one basis
    and a model needs it on another.

    Parameters
    ----------
    median
        Median diameter on ``from_weighting``.
    sigma_g
        Geometric standard deviation, shared by all bases for a log-normal.
    from_weighting, to_weighting
        Source and target bases.

    Returns
    -------
    Quantity
        Median on ``to_weighting``.

    Examples
    --------
    A laser-diffraction volume d50 of 16.3756 um with sigma_g 1.5 is a count median of
    10 um, a 64 percent difference:

    >>> from ae.core.units import Q_
    >>> d = convert_lognormal_median(Q_(16.375575970496996, "um"), 1.5,
    ...                              Weighting.VOLUME, Weighting.NUMBER)
    >>> round(float(d.to("um").magnitude), 6)
    10.0
    """
    require_dimensionality(median, "length", "median")
    if sigma_g < 1.0:
        raise ValueError("geometric standard deviation must be at least 1")
    if float(median.to("m").magnitude) <= 0.0:
        raise ValueError("median must be a positive diameter")
    exps = {Weighting.NUMBER: 0.0, Weighting.AREA: 2.0, Weighting.VOLUME: 3.0}
    s = math.log(sigma_g)
    factor = math.exp((exps[to_weighting] - exps[from_weighting]) * s * s)
    return (median * factor).to(median.units)


class RosinRammlerPSD(BaseModel):
    """Rosin-Rammler (Weibull) volume distribution, equations (6) to (8).

    Parameters
    ----------
    d_prime
        Location parameter: the size at which cumulative volume undersize is 0.6321.
        Alderliesten 2013 shows this parameter is not physically interpretable on its own;
        use :meth:`sauter_d32` for physical models.
    n
        Spread parameter. Values below 3 are flagged by :attr:`spread_warning`.
    density
        True particle density.
    sphericity
        Sphericity Value, default spheres.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> rr = RosinRammlerPSD(d_prime=Q_(20.0, "um"), n=3.5, density=Q_(2650.0, "kg/m**3"))
    >>> round(float(rr.median().to("um").magnitude), 4)
    18.0116
    >>> round(float(rr.sauter_d32().to("um").magnitude), 4)
    15.6741
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    d_prime: Quantity
    n: float = Field(gt=0.0)
    density: Quantity
    sphericity: Value | None = None

    @model_validator(mode="after")
    def _checks(self) -> RosinRammlerPSD:
        require_dimensionality(self.d_prime, "length", "d_prime")
        require_dimensionality(self.density, "density", "density")
        if float(self.d_prime.to("m").magnitude) <= 0.0:
            raise ValueError("d_prime must be a positive diameter")
        if float(self.density.to("kg/m**3").magnitude) <= 0.0:
            raise ValueError("density must be positive")
        if self.n <= 0.0:
            raise ValueError("the Rosin-Rammler spread parameter must be positive")
        if self.sphericity is not None:
            require_fraction(
                float(self.sphericity.quantity.to("dimensionless").magnitude),
                "sphericity", lo=1e-6, hi=1.0)
        return self

    @property
    def spread_warning(self) -> str | None:
        """Non-None when ``n`` is in the range Alderliesten 2013 rules out."""
        if self.n < 3.0:
            return (
                f"spread parameter n = {self.n} is below 3. Alderliesten 2013 "
                f"(doi 10.1002/ppsc.201200021) shows Rosin-Rammler distributions with "
                f"n < 3 cannot exist as complete distributions and at most roughly "
                f"describe the measured size range; do not extrapolate to the fine tail "
                f"and do not use d_prime as a model input."
            )
        return None

    def cumulative_undersize(self, diameter: Quantity) -> float:
        """Cumulative VOLUME fraction undersize at a diameter, equation (6)."""
        d = float(require_dimensionality(diameter, "length", "diameter").to("m").magnitude)
        if d < 0.0:
            raise ValueError("diameter cannot be negative")
        dp = float(self.d_prime.to("m").magnitude)
        q = 1.0 - math.exp(-((d / dp) ** self.n))
        assert 0.0 <= q <= 1.0, f"cumulative fraction out of range: {q}"
        return q

    def quantile(self, p: float) -> Quantity:
        """Inverse of equation (6): ``d_p = d' [-ln(1-p)]^{1/n}``, volume basis."""
        require_fraction(p, "p", lo=1e-12, hi=1.0 - 1e-12)
        return (self.d_prime * (-math.log(1.0 - p)) ** (1.0 / self.n)).to(self.d_prime.units)

    def median(self) -> Quantity:
        """Volume median ``d50 = d' (ln 2)^{1/n}``, equation (7)."""
        return (self.d_prime * math.log(2.0) ** (1.0 / self.n)).to(self.d_prime.units)

    def volume_weighted_mean(self) -> Quantity:
        """``E_v[d] = d' Gamma(1 + 1/n)``, equation (7)."""
        return (self.d_prime * math.gamma(1.0 + 1.0 / self.n)).to(self.d_prime.units)

    def sauter_d32(self) -> Quantity:
        """``d32 = d' / Gamma(1 - 1/n)``, equation (7). Requires ``n > 1``."""
        if self.n <= 1.0:
            raise ValueError(
                f"the Sauter mean of a Rosin-Rammler distribution requires n > 1 (the "
                f"gamma argument 1 - 1/n must be positive); n = {self.n} gives a "
                f"divergent surface area, which is the mathematical signature of a fine "
                f"tail with unbounded surface"
            )
        return (self.d_prime / math.gamma(1.0 - 1.0 / self.n)).to(self.d_prime.units)

    def span(self) -> float:
        """Span on the volume basis, equation (5)."""
        d10 = float(self.quantile(0.1).to("m").magnitude)
        d50 = float(self.median().to("m").magnitude)
        d90 = float(self.quantile(0.9).to("m").magnitude)
        return (d90 - d10) / d50

    def specific_surface_area(self) -> Quantity:
        """Geometric specific surface area, equation (8), m^2/kg."""
        return specific_surface_area(self.sauter_d32(), self.density, self.sphericity)

    def summary(self) -> PSDSummary:
        """Full descriptor set, volume basis (the basis the form is defined on)."""
        return PSDSummary(
            weighting=Weighting.VOLUME,
            d10=self.quantile(0.1),
            d50=self.median(),
            d90=self.quantile(0.9),
            span=self.span(),
            sauter_d32=self.sauter_d32(),
            specific_surface_area=self.specific_surface_area(),
        )


def _norm_ppf(p: float) -> float:
    """Standard normal quantile via bisection on the error function.

    Uses ``math.erf`` so the module has no scipy dependency, and converges to better than
    1e-12 in the diameter, which is far below any measurement precision in this domain.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie strictly in (0, 1), got {p}")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
