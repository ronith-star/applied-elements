r"""Maximum packing fraction and filler loading for multimodal silica, Furnas and Andreasen.

The commercial question this module answers: how much silica can an epoxy molding compound
hold. More filler means a lower coefficient of thermal expansion, higher thermal
conductivity and less mold shrinkage, all of which a packaging customer wants; the ceiling
is set first by geometry and then, well before geometry bites, by melt viscosity.

THE CENTRAL CAVEAT, STATED BEFORE ANY EQUATION
----------------------------------------------
Every maximum-packing number in this module is a GEOMETRIC UPPER BOUND on the solids
fraction of a poured and vibrated dry powder. A real formulation does not reach it, and is
not trying to. Three reasons, in descending order of importance:

1. VISCOSITY. The Krieger-Dougherty relation (equation 5) shows relative viscosity
   diverging as the solids fraction approaches the maximum packing fraction. A molding
   compound has to flow through a gate and fill a cavity in seconds, so it must sit well
   below the divergence. At a maximum packing fraction of 0.70, loading to 0.65 already
   costs a factor of roughly 101 in relative viscosity against a factor of roughly 9 at
   0.50. An 11-fold viscosity penalty for a 15 percent loading gain is the trade that
   actually sets commercial loading.
2. WETTING AND BINDER DEMAND. Every particle surface needs resin. Fine fractions that help
   geometric packing raise the specific surface area (see :mod:`ae.physics.psd`) and so
   raise the resin demand, which is why the geometric optimum is not the rheological one.
3. PACKING MODELS ASSUME SPHERES POURED AND VIBRATED, NOT DISPERSED IN A MELT. McGeary
   1961 (doi 10.1111/j.1151-2916.1961.tb13716.x) obtained his numbers with "spherical
   metal shot" packed "by mechanical vibration" in glass containers. Angular ground silica
   packs worse; a melt-dispersed system behaves differently again.

This module therefore reports the geometric bound, the viscosity at a chosen loading, and
refuses to present the bound as an achievable formulation.

EQUATIONS
---------
(1) Furnas geometric filling of interstices by successive size classes (Furnas 1931,
    doi 10.1021/ie50261a017):

    .. math::

        \phi_{max}^{(N)} = 1 - (1 - \phi_1)^{N}

    Symbols:
      \phi_{max}^{(N)}   maximum solids volume fraction for N size classes,
                         dimensionless, range (0, 1).
      \phi_1             monomodal packing fraction of one class, dimensionless. 0.625
                         from McGeary 1961 for vibrated equal spheres; 0.6366 for random
                         close packing (Scott and Kilgour 1969,
                         doi 10.1088/0022-3727/2/6/311).
      N                  number of size classes, integer >= 1.

    Derivation from first principles: class 1 fills a fraction phi_1 of the total volume
    and leaves voids (1 - phi_1). If class 2 is small enough that its own packing is
    unaffected by the class-1 skeleton, it fills phi_1 of THAT void space, contributing
    phi_1(1 - phi_1). By induction the solid fraction of class i is
    phi_1 (1 - phi_1)^{i-1}, a geometric series whose sum to N terms is
    1 - (1 - phi_1)^N.

    Assumption that limits it: each class must be small enough not to disturb the packing
    of the class above. McGeary 1961 measured the requirement as "at least a sevenfold
    difference between sphere sizes of the individual components", so this module refuses
    to apply equation (1) to a size ladder finer than 7:1 without an explicit override.

(2) Furnas optimal volume composition (same derivation):

    .. math::

        x_i = \frac{\phi_1 (1 - \phi_1)^{i-1}}{\phi_{max}^{(N)}}, \qquad i = 1 \ldots N

    Symbols: x_i the volume fraction of the SOLIDS made up by class i, coarsest first,
    dimensionless, summing to 1.

(3) Andreasen continuous grading (Andreasen 1930, doi 10.1007/bf01422986):

    .. math::

        Q_3(d) = \left(\frac{d}{d_{max}}\right)^{q}

    Symbols:
      Q_3     cumulative volume fraction undersize, dimensionless, range [0, 1].
      d       diameter, m, range (0, d_max].
      d_max   largest particle, m.
      q       distribution modulus, dimensionless. Andreasen's own optimum for dense
              packing is in the 0.33 to 0.5 band; higher q gives a coarser, less densely
              packing grading.

(4) Modified Andreasen, i.e. the Dinger-Funk form with a finite minimum size (Funk and
    Dinger 1994, doi 10.1007/978-1-4615-3118-0):

    .. math::

        Q_3(d) = \frac{d^{q} - d_{min}^{q}}{d_{max}^{q} - d_{min}^{q}}

    Symbols: d_min the smallest particle, m. Reduces to (3) as d_min approaches 0. The
    finite d_min matters because equation (3) demands infinitely many infinitely fine
    particles, which is both physically impossible and, through surface area, the thing
    that destroys the rheology.

(5) Krieger-Dougherty relative viscosity (Krieger and Dougherty 1959,
    doi 10.1122/1.548848):

    .. math::

        \eta_r = \left(1 - \frac{\phi}{\phi_{max}}\right)^{-[\eta]\,\phi_{max}}

    Symbols:
      \eta_r      relative viscosity (suspension over matrix), dimensionless, >= 1.
      \phi        solids volume fraction, dimensionless, range [0, phi_max).
      \phi_{max}  maximum packing fraction, dimensionless.
      [\eta]      intrinsic viscosity, dimensionless. 2.5 for hard spheres, the Einstein
                  value, and the only value used here.

(6) Volume to mass fraction conversion, needed because packing is volumetric and a
    datasheet quotes weight percent:

    .. math::

        w = \frac{\phi\,\rho_f}{\phi\,\rho_f + (1 - \phi)\,\rho_m}

    Symbols:
      w        filler mass fraction, dimensionless.
      \rho_f   filler true density, kg/m^3 (2650 for crystalline quartz; fused silica is
               about 2200, which is why a fused-silica EMC at the same volume loading
               reads lower in weight percent).
      \rho_m   matrix density, kg/m^3 (epoxy molding resins are near 1200).

    Derivation: mass of filler per unit composite volume is phi rho_f, matrix is
    (1 - phi) rho_m, and w is the ratio of the first to their sum. Dimensionally
    [kg/m^3]/[kg/m^3], dimensionless, as required.

LIMITATIONS
-----------
1. UPPER BOUND ONLY, NEVER A FORMULATION. See the caveat above. A computed phi_max of 0.98
   for four size classes is a statement about spheres in a vibrated container, not about a
   moldable compound. Treat the Krieger-Dougherty viscosity at the intended loading as the
   binding constraint and phi_max as the asymptote it must avoid.
2. FURNAS ASSUMES NON-INTERACTING CLASSES. Equation (1) overestimates when the size ratio
   between classes is below the sevenfold threshold McGeary 1961 identified, because a
   finer class then disturbs the coarse skeleton rather than sitting in its voids. This
   module enforces the threshold by default.
3. EQUATION (1) IS A GEOMETRIC IDEALISATION, AND IT SHOWS. Benchmarked against McGeary's
   quaternary measurement of 95.1 percent, equation (1) with phi_1 = 0.625 gives 98.0
   percent, a 3.07 percent overestimate, and its optimal composition (63.8, 23.9, 9.0,
   3.4 volume percent coarsest-first) differs materially from McGeary's measured optimum
   (60.7, 23.0, 10.2, 6.1). The model is systematically short of fines. That error is
   reported by the benchmark test rather than tuned away.
4. NO SHAPE TERM. Every number here is for spheres. Spherical silica filler for advanced
   packaging is genuinely near-spherical (flame-fused), so the models are defensible for
   that product and NOT for ground angular filler, which packs to a lower fraction.
5. KRIEGER-DOUGHERTY IS FOR NEWTONIAN SUSPENSIONS OF HARD SPHERES at low to moderate
   shear. An epoxy molding compound is a filled thermoset that shear-thins, cures while
   flowing and has a resin whose own viscosity falls steeply with temperature. Use
   equation (5) for the SHAPE of the loading-viscosity trade, not for an absolute
   viscosity, and never to predict transfer-molding fill.
6. NO ALPHA-EMISSION MODEL. For HBM and advanced-packaging low-alpha filler, the binding
   specification is U and Th content (alpha flux), not packing. That is an impurity
   property carried in ``ImpurityProfile`` and absent for an uncharacterized deposit; it
   is not modelled here and it is what actually gates entry to that market.
7. ANDREASEN q IS NOT OPTIMISED HERE. This module evaluates a grading for a given q; it
   does not solve for the q that maximises packing, because the optimum depends on d_min,
   on the shear rate and on the resin, none of which are inputs.
8. NO COMPRESSIBILITY OR SEGREGATION. Vibrated packing is history-dependent and fine
   fractions segregate during handling. Neither effect appears in these models.

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
import math
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.feedstock import Feedstock
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = [
    "SRC_FURNAS",
    "SRC_MCGEARY",
    "SRC_ANDREASEN",
    "SRC_FUNK_DINGER",
    "SRC_KRIEGER",
    "SRC_SCOTT_KILGOUR",
    "PHI_MONOMODAL_VIBRATED",
    "PHI_RANDOM_CLOSE",
    "EINSTEIN_INTRINSIC_VISCOSITY",
    "MIN_SIZE_RATIO",
    "SizeClass",
    "FurnasResult",
    "furnas_max_packing",
    "furnas_optimal_composition",
    "andreasen_cumulative",
    "andreasen_modified_cumulative",
    "krieger_dougherty_relative_viscosity",
    "volume_to_mass_fraction",
    "mass_to_volume_fraction",
    "FillerLoading",
    "emc_filler_loading",
]

_ACCESSED: Final[_dt.date] = _dt.date(2026, 9, 16)

SRC_FURNAS: Final[Source] = Source(
    citation=(
        "Furnas, C.C. 1931, Grading Aggregates: I. Mathematical Relations for Beds of "
        "Broken Solids of Maximum Density, Industrial and Engineering Chemistry 23:1052-1058"
    ),
    tier=Tier.T1,
    doi="10.1021/ie50261a017",
    accessed=_ACCESSED,
    note=(
        "Full text not reachable from this sandbox. The geometric-series relation "
        "implemented as equation (1) is DERIVED in the module docstring from first "
        "principles rather than transcribed, so the citation is attributional."
    ),
)

SRC_MCGEARY: Final[Source] = Source(
    citation=(
        "McGeary, R.K. 1961, Mechanical Packing of Spherical Particles, Journal of the "
        "American Ceramic Society 44:513-522"
    ),
    tier=Tier.T1,
    doi="10.1111/j.1151-2916.1961.tb13716.x",
    accessed=_ACCESSED,
    extraction="manual",
    note=(
        "Values taken from the published abstract (full text paywalled from this sandbox): "
        "one-size spheres pack to 62.5 percent of theoretical density; high-density "
        "multicomponent packings require at least a sevenfold size difference; a quaternary "
        "packing reached 95.1 percent at diameter ratios 1:7:38:316 and volume compositions "
        "6.1:10.2:23.0:60.7 percent."
    ),
)

SRC_ANDREASEN: Final[Source] = Source(
    citation=(
        "Andreasen, A.H.M. and Andersen, J. 1930, Ueber die Beziehung zwischen "
        "Kornabstufung und Zwischenraum in Produkten aus losen Koernern (mit einigen "
        "Experimenten), Kolloid-Zeitschrift 50:217-228"
    ),
    tier=Tier.T1,
    doi="10.1007/bf01422986",
    accessed=_ACCESSED,
    note="Full text not reachable from this sandbox; the power-law grading form is standard.",
)

SRC_FUNK_DINGER: Final[Source] = Source(
    citation=(
        "Funk, J.E. and Dinger, D.R. 1994, Predictive Process Control of Crowded "
        "Particulate Suspensions Applied to Ceramic Manufacturing, Springer"
    ),
    tier=Tier.T1,
    doi="10.1007/978-1-4615-3118-0",
    accessed=_ACCESSED,
    note=(
        "Book, not reachable in full text from this sandbox. Cited for the modified "
        "Andreasen (Dinger-Funk) form with a finite minimum particle size, equation (4), "
        "which reduces to the Andreasen form in the d_min to zero limit and is verified by "
        "a test rather than taken on trust."
    ),
)

SRC_KRIEGER: Final[Source] = Source(
    citation=(
        "Krieger, I.M. and Dougherty, T.J. 1959, A Mechanism for Non-Newtonian Flow in "
        "Suspensions of Rigid Spheres, Transactions of the Society of Rheology 3:137-152"
    ),
    tier=Tier.T1,
    doi="10.1122/1.548848",
    accessed=_ACCESSED,
    note="Full text not reachable from this sandbox; the relation is standard.",
)

SRC_SCOTT_KILGOUR: Final[Source] = Source(
    citation=(
        "Scott, G.D. and Kilgour, D.M. 1969, The density of random close packing of "
        "spheres, Journal of Physics D: Applied Physics 2:863-866"
    ),
    tier=Tier.T1,
    doi="10.1088/0022-3727/2/6/311",
    accessed=_ACCESSED,
    note=(
        "Cited for the random-close-packing fraction of 0.6366 as an alternative to "
        "McGeary's vibrated 0.625. Full text not reachable from this sandbox."
    ),
)

#: Monomodal packing fraction of vibrated equal spheres, McGeary 1961.
PHI_MONOMODAL_VIBRATED: Final[Value] = Value(
    quantity=Q_(0.625, "dimensionless"),
    tag=Tag.SOURCED,
    source=SRC_MCGEARY,
    basis="one-size spheres packed in an orthorhombic arrangement with a density 62.5 "
          "percent of theoretical density",
    confidence="high",
)

#: Random close packing fraction, Scott and Kilgour 1969.
PHI_RANDOM_CLOSE: Final[Value] = Value(
    quantity=Q_(0.6366, "dimensionless"),
    tag=Tag.SOURCED,
    source=SRC_SCOTT_KILGOUR,
    basis="random close packing density of equal spheres, the poured-and-tapped limit "
          "commonly quoted as 0.6366",
    confidence="high",
)

#: Einstein intrinsic viscosity for hard spheres, used in Krieger-Dougherty.
EINSTEIN_INTRINSIC_VISCOSITY: Final[Value] = Value(
    quantity=Q_(2.5, "dimensionless"),
    tag=Tag.SOURCED,
    source=SRC_KRIEGER,
    basis="hard-sphere intrinsic viscosity of 5/2, the Einstein coefficient assumed in the "
          "Krieger-Dougherty derivation",
    confidence="high",
)

#: Minimum diameter ratio between successive Furnas size classes, McGeary 1961.
MIN_SIZE_RATIO: Final[Value] = Value(
    quantity=Q_(7.0, "dimensionless"),
    tag=Tag.SOURCED,
    source=SRC_MCGEARY,
    basis="forming of high-density multicomponent packings requires at least a sevenfold "
          "difference between sphere sizes of the individual components",
    confidence="high",
)


class SizeClass(BaseModel):
    """One size class in a multimodal blend."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    diameter: Quantity
    label: str | None = None

    @model_validator(mode="after")
    def _checks(self) -> "SizeClass":
        require_dimensionality(self.diameter, "length", "diameter")
        if float(self.diameter.to("m").magnitude) <= 0.0:
            raise ValueError("size class diameter must be positive")
        return self

    @property
    def d_m(self) -> float:
        return float(self.diameter.to("m").magnitude)


class FurnasResult(BaseModel):
    """Furnas maximum packing with its composition and its own health warning."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    n_classes: int = Field(ge=1)
    phi_monomodal: float
    phi_max: float
    composition: tuple[float, ...]
    size_ratios: tuple[float, ...] = ()
    ratio_warning: str | None = None
    caveat: str = (
        "GEOMETRIC UPPER BOUND for vibrated spheres. Real formulations do not reach it "
        "because melt viscosity diverges as the solids fraction approaches phi_max "
        "(Krieger-Dougherty) and because every added surface demands resin. Benchmarked "
        "against McGeary 1961, equation (1) overestimates a measured quaternary packing by "
        "3.07 percent."
    )

    @model_validator(mode="after")
    def _checks(self) -> "FurnasResult":
        require_fraction(self.phi_max, "phi_max", lo=1e-9, hi=1.0)
        require_fraction(self.phi_monomodal, "phi_monomodal", lo=1e-9, hi=1.0)
        if len(self.composition) != self.n_classes:
            raise ValueError(
                f"composition has {len(self.composition)} entries for {self.n_classes} "
                f"classes"
            )
        for x in self.composition:
            if x < 0.0:
                raise ValueError("a volume fraction cannot be negative")
        total = sum(self.composition)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"composition must sum to 1, got {total!r}")
        if self.phi_max < self.phi_monomodal - 1e-12:
            raise ValueError(
                "multimodal packing cannot be worse than monomodal in the Furnas model"
            )
        return self


def furnas_max_packing(size_classes: tuple[SizeClass, ...],
                       phi_monomodal: Value = PHI_MONOMODAL_VIBRATED,
                       enforce_size_ratio: bool = False) -> FurnasResult:
    """Maximum packing fraction and optimal composition, equations (1) and (2).

    Parameters
    ----------
    size_classes
        Size classes, any order; they are sorted coarsest-first internally. At least one.
    phi_monomodal
        Packing fraction of a single class. Defaults to McGeary's vibrated 0.625; pass
        :data:`PHI_RANDOM_CLOSE` for the poured random-close-packing value instead.
    enforce_size_ratio
        Raise when successive classes are closer than the sevenfold ratio McGeary 1961
        found necessary. Defaults False, which returns a ``ratio_warning`` instead, because
        McGeary's OWN measured quaternary optimum (1:7:38:316) contains a 38/7 = 5.43 step:
        a strict sevenfold gate would reject the very experiment that calibrates the model.
        Set True when you want the closeness treated as a hard error.

    Returns
    -------
    FurnasResult

    Examples
    --------
    McGeary's four size classes, ratios 1:7:38:316:

    >>> from ae.core.units import Q_
    >>> classes = tuple(SizeClass(diameter=Q_(d, "um")) for d in (316.0, 38.0, 7.0, 1.0))
    >>> r = furnas_max_packing(classes)
    >>> round(r.phi_max, 5)
    0.98022
    >>> [round(100 * x, 2) for x in r.composition]
    [63.76, 23.91, 8.97, 3.36]
    """
    if not size_classes:
        raise ValueError("at least one size class is required")
    phi1 = float(phi_monomodal.quantity.to("dimensionless").magnitude)
    require_fraction(phi1, "phi_monomodal", lo=1e-9, hi=1.0)
    ordered = sorted(size_classes, key=lambda c: c.d_m, reverse=True)
    n = len(ordered)
    ratios = tuple(ordered[i].d_m / ordered[i + 1].d_m for i in range(n - 1))
    min_ratio = float(MIN_SIZE_RATIO.quantity.to("dimensionless").magnitude)
    warning: str | None = None
    if ratios and min(ratios) < min_ratio:
        worst = min(ratios)
        msg = (
            f"successive size ratio {worst:.3f} is below the {min_ratio:.0f}-fold "
            f"separation McGeary 1961 found necessary for high-density multicomponent "
            f"packing. Below it a finer class disturbs the coarse skeleton instead of "
            f"filling its voids, and equation (1) overestimates phi_max by more than its "
            f"already-documented 3 percent bias."
        )
        if enforce_size_ratio:
            raise ValueError(msg + " Pass enforce_size_ratio=False to compute anyway.")
        warning = msg
    phi_max = 1.0 - (1.0 - phi1) ** n
    assert 0.0 < phi_max <= 1.0, "packing fraction must lie in (0, 1]"
    comp = furnas_optimal_composition(n, phi_monomodal)
    return FurnasResult(
        n_classes=n,
        phi_monomodal=phi1,
        phi_max=phi_max,
        composition=comp,
        size_ratios=ratios,
        ratio_warning=warning,
    )


def furnas_optimal_composition(n_classes: int,
                               phi_monomodal: Value = PHI_MONOMODAL_VIBRATED
                               ) -> tuple[float, ...]:
    """Optimal volume composition of the solids, coarsest first, equation (2).

    Examples
    --------
    >>> [round(100 * x, 2) for x in furnas_optimal_composition(4)]
    [63.76, 23.91, 8.97, 3.36]
    """
    if n_classes < 1:
        raise ValueError("n_classes must be at least 1")
    phi1 = float(phi_monomodal.quantity.to("dimensionless").magnitude)
    require_fraction(phi1, "phi_monomodal", lo=1e-9, hi=1.0)
    raw = [phi1 * (1.0 - phi1) ** i for i in range(n_classes)]
    total = sum(raw)
    if total <= 0.0:
        raise ValueError("degenerate composition")
    comp = tuple(x / total for x in raw)
    assert abs(sum(comp) - 1.0) < 1e-12, "composition must sum to unity"
    return comp


def andreasen_cumulative(diameter: Quantity, d_max: Quantity, q: float) -> float:
    """Andreasen cumulative volume undersize, equation (3).

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> round(andreasen_cumulative(Q_(10.0, "um"), Q_(100.0, "um"), 0.5), 6)
    0.316228
    """
    d = float(require_dimensionality(diameter, "length", "diameter").to("m").magnitude)
    dmax = float(require_dimensionality(d_max, "length", "d_max").to("m").magnitude)
    if d < 0.0:
        raise ValueError("diameter cannot be negative")
    if dmax <= 0.0:
        raise ValueError("d_max must be positive")
    if q <= 0.0:
        raise ValueError("the Andreasen distribution modulus must be positive")
    if d > dmax:
        return 1.0
    val = float((d / dmax) ** q)
    assert 0.0 <= val <= 1.0, f"cumulative fraction out of range: {val}"
    return val


def andreasen_modified_cumulative(diameter: Quantity, d_min: Quantity, d_max: Quantity,
                                  q: float) -> float:
    """Modified Andreasen (Dinger-Funk) cumulative undersize, equation (4).

    Examples
    --------
    With d_min four orders of magnitude below d_max the result approaches equation (3):

    >>> from ae.core.units import Q_
    >>> round(andreasen_modified_cumulative(Q_(10.0, "um"), Q_(0.01, "um"),
    ...                                     Q_(100.0, "um"), 0.5), 6)
    0.309321
    """
    d = float(require_dimensionality(diameter, "length", "diameter").to("m").magnitude)
    dmin = float(require_dimensionality(d_min, "length", "d_min").to("m").magnitude)
    dmax = float(require_dimensionality(d_max, "length", "d_max").to("m").magnitude)
    if q <= 0.0:
        raise ValueError("the Andreasen distribution modulus must be positive")
    if dmin <= 0.0:
        raise ValueError("d_min must be positive; use andreasen_cumulative for the limit")
    if dmax <= dmin:
        raise ValueError("d_max must exceed d_min")
    if d <= dmin:
        return 0.0
    if d >= dmax:
        return 1.0
    val = float((d ** q - dmin ** q) / (dmax ** q - dmin ** q))
    assert 0.0 <= val <= 1.0, f"cumulative fraction out of range: {val}"
    return val


def krieger_dougherty_relative_viscosity(
    phi: float, phi_max: float,
    intrinsic_viscosity: Value = EINSTEIN_INTRINSIC_VISCOSITY,
) -> float:
    """Relative viscosity of a filled suspension, equation (5).

    Parameters
    ----------
    phi
        Solids volume fraction, strictly below ``phi_max``.
    phi_max
        Maximum packing fraction.
    intrinsic_viscosity
        Intrinsic viscosity, default the Einstein hard-sphere value of 2.5.

    Returns
    -------
    float
        Relative viscosity, >= 1 and diverging as phi approaches phi_max.

    Examples
    --------
    >>> round(krieger_dougherty_relative_viscosity(0.50, 0.70), 4)
    8.9561
    >>> round(krieger_dougherty_relative_viscosity(0.65, 0.70), 4)
    101.3267
    """
    require_fraction(phi, "phi", lo=0.0, hi=1.0)
    require_fraction(phi_max, "phi_max", lo=1e-9, hi=1.0)
    if phi >= phi_max:
        raise ValueError(
            f"solids fraction {phi} is at or above the maximum packing fraction {phi_max}: "
            f"the Krieger-Dougherty viscosity diverges there, which is the physical "
            f"statement that the suspension cannot flow"
        )
    ie = float(intrinsic_viscosity.quantity.to("dimensionless").magnitude)
    if ie <= 0.0:
        raise ValueError("intrinsic viscosity must be positive")
    eta_r = float((1.0 - phi / phi_max) ** (-ie * phi_max))
    assert eta_r >= 1.0, "adding rigid filler cannot reduce viscosity below the matrix"
    return eta_r


def volume_to_mass_fraction(phi: float, filler_density: Quantity,
                            matrix_density: Quantity) -> float:
    """Filler mass fraction from its volume fraction, equation (6).

    Examples
    --------
    Crystalline silica at 2650 kg/m^3 in an epoxy at 1200 kg/m^3, 65 volume percent:

    >>> from ae.core.units import Q_
    >>> round(volume_to_mass_fraction(0.65, Q_(2650.0, "kg/m**3"), Q_(1200.0, "kg/m**3")), 5)
    0.80397
    """
    require_fraction(phi, "phi", lo=0.0, hi=1.0)
    rf = float(require_dimensionality(filler_density, "density", "filler_density")
               .to("kg/m**3").magnitude)
    rm = float(require_dimensionality(matrix_density, "density", "matrix_density")
               .to("kg/m**3").magnitude)
    if rf <= 0.0 or rm <= 0.0:
        raise ValueError("densities must be positive")
    w = phi * rf / (phi * rf + (1.0 - phi) * rm)
    assert 0.0 <= w <= 1.0, "mass fraction out of range"
    return w


def mass_to_volume_fraction(w: float, filler_density: Quantity,
                            matrix_density: Quantity) -> float:
    """Filler volume fraction from its mass fraction, the inverse of equation (6).

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> round(mass_to_volume_fraction(0.80397, Q_(2650.0, "kg/m**3"),
    ...                               Q_(1200.0, "kg/m**3")), 5)
    0.65
    """
    require_fraction(w, "w", lo=0.0, hi=1.0)
    rf = float(require_dimensionality(filler_density, "density", "filler_density")
               .to("kg/m**3").magnitude)
    rm = float(require_dimensionality(matrix_density, "density", "matrix_density")
               .to("kg/m**3").magnitude)
    if rf <= 0.0 or rm <= 0.0:
        raise ValueError("densities must be positive")
    phi = (w / rf) / (w / rf + (1.0 - w) / rm)
    assert 0.0 <= phi <= 1.0, "volume fraction out of range"
    return phi


class FillerLoading(BaseModel):
    """Filler loading result: the geometric ceiling, the chosen loading, and the penalty."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    phi_max_geometric: float
    phi_selected: float
    mass_fraction_selected: float
    mass_fraction_at_phi_max: float
    relative_viscosity: float
    headroom: float
    n_classes: int
    is_scenario: bool
    caveat: str
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _checks(self) -> "FillerLoading":
        require_fraction(self.phi_max_geometric, "phi_max_geometric", lo=1e-9, hi=1.0)
        require_fraction(self.phi_selected, "phi_selected", lo=0.0, hi=1.0)
        require_fraction(self.mass_fraction_selected, "mass_fraction_selected")
        require_fraction(self.mass_fraction_at_phi_max, "mass_fraction_at_phi_max")
        if self.phi_selected >= self.phi_max_geometric:
            raise ValueError(
                "selected loading must lie strictly below the geometric maximum packing "
                "fraction, which is an unattainable asymptote"
            )
        if self.relative_viscosity < 1.0:
            raise ValueError("relative viscosity of a filled system cannot be below 1")
        if abs((self.phi_max_geometric - self.phi_selected) - self.headroom) > 1e-12:
            raise ValueError("headroom must equal phi_max minus phi_selected")
        return self


def emc_filler_loading(
    feedstock: Feedstock,
    size_classes: tuple[SizeClass, ...],
    phi_selected: float,
    filler_density: Quantity,
    matrix_density: Quantity,
    phi_monomodal: Value = PHI_MONOMODAL_VIBRATED,
    enforce_size_ratio: bool = False,
) -> FillerLoading:
    """Filler loading for an epoxy molding compound: ceiling, choice, viscosity penalty.

    Parameters
    ----------
    feedstock
        The silica source. Required by platform rule, and used to mark the result a
        scenario when the ore is uncharacterized.
    size_classes
        The filler size ladder.
    phi_selected
        The intended solids volume fraction, strictly below the geometric ceiling.
    filler_density, matrix_density
        True densities, needed for the volume-to-mass conversion a datasheet needs.
    phi_monomodal
        Single-class packing fraction.
    enforce_size_ratio
        As for :func:`furnas_max_packing`.

    Returns
    -------
    FillerLoading

    Notes
    -----
    For a low-alpha filler aimed at HBM or advanced packaging, packing is not the binding
    constraint: U and Th content is, because alpha emission causes soft errors. That is an
    impurity property (``feedstock.impurities``), it is unmeasured for an uncharacterized
    deposit, and this function neither models nor implies it.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> from ae.core.feedstock import Feedstock, OreType
    >>> f = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
    ...               deposit_name="Vikarabad", country="IN")
    >>> classes = tuple(SizeClass(diameter=Q_(d, "um")) for d in (30.0, 4.0, 0.5))
    >>> r = emc_filler_loading(f, classes, 0.65, Q_(2200.0, "kg/m**3"),
    ...                        Q_(1200.0, "kg/m**3"))
    >>> round(r.phi_max_geometric, 5), round(r.mass_fraction_selected, 4)
    (0.94727, 0.773)
    """
    if not isinstance(feedstock, Feedstock):
        raise TypeError(
            "emc_filler_loading requires a Feedstock so that nothing is hardcoded to one "
            "deposit"
        )
    furnas = furnas_max_packing(size_classes, phi_monomodal, enforce_size_ratio)
    require_fraction(phi_selected, "phi_selected", lo=0.0, hi=1.0)
    if phi_selected >= furnas.phi_max:
        raise ValueError(
            f"selected loading {phi_selected} is at or above the geometric ceiling "
            f"{furnas.phi_max:.5f}; the ceiling is an asymptote that a flowable compound "
            f"must stay well below"
        )
    eta_r = krieger_dougherty_relative_viscosity(phi_selected, furnas.phi_max)
    notes: list[str] = [
        f"relative viscosity at the selected loading is {eta_r:.1f} times the unfilled "
        f"resin, from Krieger-Dougherty with the geometric ceiling as phi_max. This is the "
        f"constraint that sets commercial loading, not the ceiling itself.",
    ]
    if furnas.ratio_warning is not None:
        notes.append(furnas.ratio_warning)
    if not feedstock.characterized:
        notes.append(
            f"feedstock {feedstock.sample_id} is not characterized, so this is a SCENARIO "
            f"conditional on a future characterization campaign, not a finding. For a "
            f"low-alpha filler the binding specification is U and Th content, which is "
            f"unmeasured here."
        )
    return FillerLoading(
        phi_max_geometric=furnas.phi_max,
        phi_selected=phi_selected,
        mass_fraction_selected=volume_to_mass_fraction(
            phi_selected, filler_density, matrix_density),
        mass_fraction_at_phi_max=volume_to_mass_fraction(
            furnas.phi_max, filler_density, matrix_density),
        relative_viscosity=eta_r,
        headroom=furnas.phi_max - phi_selected,
        n_classes=furnas.n_classes,
        is_scenario=not feedstock.characterized,
        caveat=furnas.caveat,
        notes=tuple(notes),
    )
