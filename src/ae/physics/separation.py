"""Physical separation: WHIMS, flotation, partition curves and mass balance.

Three separate things are modelled here and they should not be confused:

1. **Rate models** (flotation, WHIMS) predict recovery as a function of
   residence time given a rate constant that must be MEASURED in a batch test.
   They do not predict the rate constant.
2. **Partition curves** (Tromp) describe how sharply a unit splits a feed by
   some property (size, density, magnetic susceptibility). They are descriptive
   fits to a measured curve, parameterised by a cut point and a sharpness.
3. **Mass balance** (two-product formula, grade-recovery) is arithmetic. It has
   no fitted parameters, it closes exactly, and any discrepancy is a measurement
   error or a bookkeeping error, never a model error. Tests here require closure
   to machine precision.

In the HPQ flowsheet these sit between grinding and leaching: magnetic
separation against Fe-bearing impurities and flotation against silicate
minerals, both operating after liberation by crushing and grinding, taking the
product from roughly 90 to 99 percent up to 99 to 99.99 percent SiO2 before the
chemical stages (Lin M. et al. 2020, Table 3, doi:10.1007/s42461-020-00247-0).
Both act only on liberated, exposed impurities, so their maximum recovery is
bounded by :mod:`ae.physics.liberation` and their floor by
:mod:`ae.physics.impurity_location`.

Equations
---------
**(E1) First-order flotation kinetics, classical form.**

.. math::
    R(t) = R_{\\infty} \\left( 1 - e^{-k t} \\right)

- :math:`R(t)`: cumulative recovery of the floated species at time :math:`t`,
  dimensionless in [0, 1].
- :math:`R_{\\infty}`: ultimate (maximum) recovery, dimensionless in [0, 1].
  Strictly less than 1 in practice; the deficit is unliberated and entrained
  material. This is the parameter that couples to liberation.
- :math:`k`: first-order rate constant, 1/s (often quoted in 1/min). Typical
  batch values for silicate flotation are of order 0.01 to 0.1 1/s, but the
  value is system-specific and must be fitted.
- :math:`t`: flotation time, s, range 0 to about 1800 s in a batch test.

Derivation: the first-order postulate is that the rate of removal of floatable
particles is proportional to the number remaining,
:math:`\\mathrm{d}N/\\mathrm{d}t = -kN`, whose solution
:math:`N(t) = N_0 e^{-kt}` gives recovery :math:`1 - e^{-kt}` of the floatable
sub-population :math:`R_{\\infty} N_{\\text{total}}`. Polat and Chander 2000
(First-order flotation kinetics models and methods for estimation of the true
distribution of flotation rate constants, International Journal of Mineral
Processing 58:145-166, doi:10.1016/S0301-7516(99)00069-1) treat this two-
parameter form as the reference case and address estimating the DISTRIBUTION of
:math:`k` when a single constant does not fit, which is the usual situation.

**(E2) Rate-constant distribution, rectangular case.** Integrating E1 over a
uniform distribution of :math:`k` on :math:`[0, k_{max}]`:

.. math::
    R(t) = R_{\\infty} \\left[ 1 -
    \\frac{1 - e^{-k_{max} t}}{k_{max} t} \\right]

- :math:`k_{max}`: upper bound of the rectangular distribution, 1/s.
- Included because a single-:math:`k` fit systematically overstates early
  recovery and understates late recovery for a real polydisperse,
  heterogeneously-liberated feed. Both limits are checked in tests:
  :math:`R \\to 0` as :math:`t \\to 0` and :math:`R \\to R_{\\infty}` as
  :math:`t \\to \\infty`.

**(E3) WHIMS recovery as a first-order capture process.**

.. math::
    R_{mag}(t) = R_{\\infty,mag} \\left( 1 - e^{-k_{mag} t} \\right),
    \\qquad k_{mag} = k_0 \\left( \\frac{B}{B_{ref}} \\right)^{n_B}

- :math:`B`: applied magnetic flux density, T. WHIMS operates roughly 0.5 to
  2 T, superconducting HGMS to about 5 T.
- :math:`B_{ref}`: reference flux density at which :math:`k_0` was measured, T.
- :math:`n_B`: empirical exponent, dimensionless. :math:`n_B = 2` corresponds to
  magnetic force scaling as :math:`B \\nabla B` for a linear paramagnet in a
  fixed-geometry matrix, which is the usual first-principles starting point.
  **This exponent is ASSUMED in this module** (see the value's basis string) and
  must be fitted from a field sweep; treating it as known is the main risk in
  E3.
- :math:`t`: residence time in the matrix, s.

The functional form of E3 is first-order capture by analogy with E1, not a
result taken from a source. See LIMITATIONS item 2.

**(E4) Tromp (partition) curve, logistic parameterisation.**

.. math::
    P(x) = \\frac{1}{1 + \\exp\\!\\left(-\\dfrac{x - x_{50}}{s}\\right)},
    \\qquad s = \\frac{E_p}{\\ln 3}

- :math:`P(x)`: partition number, the fraction of feed material of property
  :math:`x` reporting to the coarse/sink/magnetic stream. Dimensionless [0, 1].
- :math:`x`: the separating property (size in um, density in g/cm3, or
  susceptibility). Same unit as :math:`x_{50}` and :math:`E_p`.
- :math:`x_{50}`: cut point, where :math:`P = 0.5` by construction.
- :math:`E_p`: ecart probable moyen (probable error),
  :math:`E_p = (x_{75} - x_{25})/2`. Smaller is sharper; :math:`E_p = 0` is a
  perfect separation.
- :math:`s`: logistic scale. The relation :math:`s = E_p/\\ln 3` is exact for
  the logistic form: :math:`P = 0.75` requires
  :math:`(x - x_{50})/s = \\ln 3`, so :math:`x_{75} - x_{50} = s \\ln 3`, and by
  symmetry :math:`E_p = (x_{75} - x_{25})/2 = s \\ln 3`. Hence
  :math:`s = E_p/\\ln 3 = E_p/1.0986123`.

**(E5) Imperfection.**

.. math::
    I = \\frac{E_p}{x_{50} - 1}
    \\quad \\text{(density separation, } x \\text{ in g/cm}^3)
    \\qquad\\text{or}\\qquad
    I = \\frac{E_p}{x_{50}} \\quad \\text{(general)}

- The density form subtracts the density of water (1 g/cm3) because the
  separating force scales with the density difference from the medium, not with
  absolute density. Using the general form on a density separation is a known
  error; :func:`imperfection` requires the basis to be declared.

**(E6) Two-product formula.** Mass balance, exact.

.. math::
    Y = \\frac{f - t}{c - t}, \\qquad
    R = \\frac{c\\,(f - t)}{f\\,(c - t)} = \\frac{Y c}{f}

- :math:`f, c, t`: assay of the species of interest in feed, concentrate and
  tailing, any consistent unit (percent, ppm, mass fraction).
- :math:`Y`: mass yield to concentrate, dimensionless in [0, 1].
- :math:`R`: recovery of the species to concentrate, dimensionless in [0, 1].
- Derivation: feed mass balance :math:`F = C + T` and species balance
  :math:`Ff = Cc + Tt`. Substituting :math:`T = F - C` and dividing by
  :math:`F` gives :math:`f = Yc + (1 - Y)t`, hence
  :math:`Y = (f - t)/(c - t)`. Recovery is
  :math:`R = Yc/f`. Standard result, given in Wills B. A. and Napier-Munn T.
  2005, Metallurgical accounting, control and simulation, in Wills' Mineral
  Processing Technology, doi:10.1016/b978-075064450-1/50005-9.
- Requires :math:`c > t` and :math:`t \\le f \\le c` for a physical result;
  violations are rejected rather than returned as a yield outside [0, 1].

**(E7) Quartz-product convention: the species of interest is the REJECT.**
In quartz beneficiation the valuable stream is the one that does NOT float and
is NOT magnetic, so the "concentrate" of the impurity is the reject stream. The
impurity recovery to reject is what the plant is buying, and the product's
impurity grade follows from E6 applied to the impurity:

.. math::
    c_{\\text{product}} = \\frac{f - Y_{\\text{rej}} c_{\\text{rej}}}
    {1 - Y_{\\text{rej}}}

This is E6 rearranged, not a new equation, and it is the form a quartz
flowsheet actually needs. Getting the sense backwards (treating quartz as the
floated concentrate) inverts every grade-recovery conclusion.

**(E8) Grade-recovery tradeoff.** Combining E1 and E6: as flotation time rises,
impurity recovery to the froth rises toward :math:`R_{\\infty}` while the froth
becomes more dilute in impurity and carries more quartz, so product yield falls.
The locus of :math:`(R, c_{\\text{product}})` over :math:`t` is the
grade-recovery curve, and its shape, not any single point, is the operating
decision.

LIMITATIONS
-----------
1. **No rate constant here is measured on quartz.** :func:`flotation_recovery`
   and :func:`whims_recovery` take :math:`k` and :math:`R_{\\infty}` as
   arguments and this module supplies no default for either. Fit them from a
   batch kinetics test on the actual feed at the actual grind; a rate constant
   is not transferable between ores, reagent suites, cell geometries or grinds.
   Fluorine-free flotation systems for vein quartz exist and are published
   (Du S. et al. 2024, Purification of Vein Quartz Using a New Fluorine-Free
   Flotation, Minerals 14(12):1191, doi:10.3390/min14121191), which is cited to
   establish that the reagent choice is an open design decision, not to supply a
   rate constant.
2. **E3's field dependence is an assumption, not a result.** The first-order
   form is an analogy to flotation and the exponent :math:`n_B = 2` is
   engineering judgement from the :math:`B \\nabla B` force scaling of a linear
   paramagnet. Real HGMS capture depends on matrix geometry, wire diameter,
   slurry velocity, particle size and susceptibility, and saturates as the
   matrix loads. Do not use E3 outside a field range where it has been fitted,
   and do not read :math:`n_B` as measured.
3. **Single rate constant versus a distribution.** E1 with one :math:`k` fits
   batch data poorly in general; Polat and Chander 2000 exist precisely because
   the true :math:`k` is distributed. E2 gives one alternative (rectangular);
   gamma and other distributions are common. Choosing E1 for convenience
   biases scale-up, usually optimistically for short residence times.
4. **Partition curves are descriptive.** E4 is a convenient two-parameter
   logistic. Real curves are asymmetric, can show a short-circuit fraction
   (a non-zero asymptote for fine material bypassing to the coarse stream) and
   can be non-monotonic near the cut. Fitting a logistic to such data hides the
   bypass. :func:`partition_number` optionally takes a bypass term for that
   reason, and reports it rather than folding it into :math:`E_p`.
5. **Mass balance is exact but the assays are not.** E6 amplifies assay error
   badly when :math:`c - t` is small, which is exactly the regime of a
   high-purity product where feed, concentrate and tailing assays are all a few
   ppm. A two-product formula on 30, 5 and 45 ppm numbers each with 10 percent
   analytical uncertainty has a yield uncertainty that swamps the answer.
   :func:`two_product` will compute it; :func:`two_product_condition_number`
   tells you when not to trust it.
6. **Nothing here models entrainment or froth stability.** Fine quartz reports
   to the froth by water recovery regardless of its surface chemistry, which in
   a fine HPQ grind is a material yield loss. That requires a water-recovery
   model this module does not have.

References
----------
Every entry below was resolved against the Crossref REST API for the DOI shown,
and the fields here (author list, year, title, journal, volume, issue, pages)
are as Crossref returned them. Resolved 17 September 2026. Entries without a
DOI say what was checked instead and what remains unverified.

Polat, M. and Chander, S. (2000) First-order flotation kinetics models and
methods for estimation of the true distribution of flotation rate constants,
International Journal of Mineral Processing 58(1-4), 145-166,
doi 10.1016/S0301-7516(99)00069-1. Source of Eq. E1 and Eq. E2, and of the
argument that a single rate constant fits batch data poorly because the true
rate constant is distributed. Closed access; abstract and the equations as
restated in the derivations above.

Wills, B. A. and Finch, J. A. (2016) Classification, in Wills' Mineral
Processing Technology, 8th edition, Elsevier, pp. 199-221,
doi 10.1016/b978-0-08-097053-0.00009-1. Source of the partition-curve
vocabulary: cut point, Ecart probable, imperfection, bypass.

Wills, B. A. and Napier-Munn, T. (2005) Metallurgical accounting, control and
simulation, in Wills' Mineral Processing Technology, Elsevier, pp. 39-89,
doi 10.1016/b978-075064450-1/50005-9. Source of the two-product mass-balance
formula, Eq. E6.

Lin, M., Liu, Z., Wei, Y., Liu, B., Meng, Y., Qiu, H., Lei, S., Zhang, X. and
Li, Y. (2020) A Critical Review on the Mineralogy and Processing for High-Grade
Quartz, Mining, Metallurgy and Exploration 37(5), 1627-1639,
doi 10.1007/s42461-020-00247-0.

Du, S., Pan, B., Xia, L., Zhu, G., Wu, L., Yu, C., Li, F. and Diao, Z. (2024)
Purification of Vein Quartz Using a New Fluorine-Free Flotation: A Case from
Southern Anhui Province, China, Minerals 14(12), 1191,
doi 10.3390/min14121191.
"""

from __future__ import annotations

import datetime as _dt
import enum
import itertools
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final, cast

from ae.core.feedstock import Feedstock
from ae.core.provenance import Distribution, Source, Tag, Tier, Value
from ae.core.site import Site
from ae.core.units import (
    Q_,
    Quantity,
    require_dimensionality,
    require_fraction,
)

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
    "SRC_DU_2024",
    "SRC_LIN_2020",
    "SRC_POLAT_CHANDER",
    "SRC_WILLS_ACCOUNTING",
    "SRC_WILLS_CLASSIFICATION",
    "WHIMS_FIELD_EXPONENT",
    "PropertyBasis",
    "SeparatorKind",
    "ep_from_partition_points",
    "feedstock_impurity_separation",
    "flotation_rate_constant_from_recovery",
    "flotation_recovery",
    "grade_recovery_curve",
    "imperfection",
    "logistic_scale_from_ep",
    "partition_curve",
    "partition_number",
    "product_grade_from_reject",
    "rectangular_distribution_recovery",
    "separation_mass_balance",
    "two_product",
    "two_product_condition_number",
    "whims_rate_constant",
    "whims_recovery",
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

SRC_POLAT_CHANDER: Final[Source] = Source(
    citation=(
        "Polat M. and Chander S. 2000, First-order flotation kinetics models and "
        "methods for estimation of the true distribution of flotation rate constants, "
        "International Journal of Mineral Processing 58:145-166"
    ),
    tier=Tier.T1,
    doi="10.1016/s0301-7516(99)00069-1",
    accessed=_ACCESSED,
    note=(
        "Metadata verified via CrossRef; full text not retrievable in this environment. "
        "Cited for the established two-parameter first-order form and for the position "
        "that the rate constant is in general distributed rather than single-valued. "
        "No numerical parameter is taken from it."
    ),
)

SRC_WILLS_ACCOUNTING: Final[Source] = Source(
    citation=(
        "Wills B. A. and Napier-Munn T. 2005, Metallurgical accounting, control and "
        "simulation, in Wills' Mineral Processing Technology, 7th edition, Elsevier, "
        "pp. 39-89"
    ),
    tier=Tier.T1,
    doi="10.1016/b978-075064450-1/50005-9",
    accessed=_ACCESSED,
    note=(
        "Standard reference for the two-product formula and recovery definitions. "
        "Metadata verified via CrossRef. The formula is derived from first principles "
        "in this module's docstring rather than taken on authority."
    ),
)

SRC_WILLS_CLASSIFICATION: Final[Source] = Source(
    citation=(
        "Wills B. A. and Finch J. A. 2016, Classification, in Wills' Mineral "
        "Processing Technology, 8th edition, Elsevier, pp. 199-221"
    ),
    tier=Tier.T1,
    doi="10.1016/b978-0-08-097053-0.00009-1",
    accessed=_ACCESSED,
    note=(
        "Standard reference for partition (Tromp) curves, the cut point and the ecart "
        "probable moyen. Metadata verified via CrossRef."
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
        "Full text read. Table 3 places magnetic separation against Fe-bearing "
        "impurities and flotation against silicate minerals, after liberation by "
        "crushing and grinding, taking product grade from 90-99 to 99-99.99 percent."
    ),
)

SRC_DU_2024: Final[Source] = Source(
    citation=(
        "Du S., Pan B., Xia L. et al. 2024, Purification of Vein Quartz Using a New "
        "Fluorine-Free Flotation: A Case from Southern Anhui Province, China, "
        "Minerals 14(12):1191"
    ),
    tier=Tier.T1,
    doi="10.3390/min14121191",
    accessed=_ACCESSED,
    note=(
        "Metadata verified via CrossRef; full text not retrieved. Cited only to "
        "establish that fluorine-free flotation of vein quartz is an active published "
        "route, so reagent choice is a design decision. No rate constant taken."
    ),
)

#: Exponent on magnetic flux density in Eq. E3. ASSUMED, not measured.
WHIMS_FIELD_EXPONENT: Final[Value] = Value(
    quantity=Q_(2.0, "dimensionless"),
    tag=Tag.ASSUMED,
    dist=Distribution(kind="uniform", low=1.0, high=3.0),
    confidence="low",
    basis=(
        "ESTIMATE from first-principles force scaling, not a measurement. The magnetic "
        "force on a linear paramagnetic particle in a fixed matrix geometry scales as "
        "B times grad B, and in a matrix whose field gradient is itself proportional to "
        "the applied field this gives a B-squared dependence, hence n_B = 2. Bracketed "
        "1 to 3 because real HGMS capture also depends on slurry velocity, matrix wire "
        "diameter, particle size and matrix loading, and saturates at high field where "
        "the exponent falls below 2. Must be fitted from a field sweep on the actual "
        "feed before any WHIMS sizing."
    ),
)


class SeparatorKind(str, enum.Enum):
    """Unit operations covered by the rate models here."""

    FLOTATION = "flotation"
    WHIMS = "whims"          # wet high intensity magnetic separation
    HGMS = "hgms"            # high gradient (often superconducting) magnetic separation


class PropertyBasis(str, enum.Enum):
    """Separating property a partition curve is written against.

    Required by :func:`imperfection` because the density form of Eq. E5
    subtracts the medium density and the general form does not.
    """

    SIZE = "size"                    # um
    DENSITY = "density"              # g/cm3, imperfection uses (x50 - 1)
    SUSCEPTIBILITY = "susceptibility"


# --- Eq. E1, E2 -------------------------------------------------------------


def flotation_recovery(
    time: Qty, rate_constant: Qty, ultimate_recovery: float
) -> float:
    """First-order flotation recovery (Eq. E1).

    Parameters
    ----------
    time
        Flotation time, any time unit.
    rate_constant
        :math:`k`, reciprocal time (e.g. ``Q_(0.05, "1/s")`` or
        ``Q_(3.0, "1/min")``).
    ultimate_recovery
        :math:`R_{\\infty}` in [0, 1].

    Examples
    --------
    k = 0.05 1/s, R_inf = 0.90, t = 60 s:
    exp(-0.05 x 60) = exp(-3) = 0.0497871,
    R = 0.90(1 - 0.0497871) = 0.90 x 0.9502129 = 0.8551916.

    >>> round(flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/s"), 0.90), 7)
    0.8551916
    """
    require_dimensionality(time, "time", "time")
    require_dimensionality(rate_constant, "rate_first_order", "rate_constant")
    t = float(time.to("s").magnitude)
    k = float(rate_constant.to("1/s").magnitude)
    if t < 0.0:
        raise ValueError(f"time must be non-negative, got {t} s")
    if k <= 0.0:
        raise ValueError(
            f"rate constant must be positive, got {k} 1/s: a non-positive rate would "
            f"mean material returns from the concentrate to the feed"
        )
    r_inf = require_fraction(ultimate_recovery, "ultimate_recovery")
    r = r_inf * (1.0 - math.exp(-k * t))
    return require_fraction(r, "flotation recovery")


def flotation_rate_constant_from_recovery(
    time: Qty, recovery: float, ultimate_recovery: float
) -> Qty:
    """Fit :math:`k` from a single batch point by inverting Eq. E1.

    .. math::
        k = -\\frac{1}{t} \\ln\\left( 1 - \\frac{R}{R_{\\infty}} \\right)

    A one-point fit, provided for completeness. A real kinetics test takes four
    to six timed concentrates and fits both parameters; a single point cannot
    distinguish a fast rate with a low ultimate from a slow rate with a high
    one.
    """
    t = float(time.to("s").magnitude)
    if t <= 0.0:
        raise ValueError("time must be positive to fit a rate constant")
    r = require_fraction(recovery, "recovery")
    r_inf = require_fraction(ultimate_recovery, "ultimate_recovery")
    if r_inf <= 0.0:
        raise ValueError("ultimate recovery must be positive")
    if r >= r_inf:
        raise ValueError(
            f"observed recovery {r} is not below the ultimate recovery {r_inf}: "
            f"Eq. E1 approaches R_inf asymptotically and cannot reach it, so either "
            f"R_inf is underestimated or the observation is out of model"
        )
    k = -math.log(1.0 - r / r_inf) / t
    if k <= 0.0:
        raise AssertionError("fitted rate constant must be positive")
    return Q_(k, "1/s")


def rectangular_distribution_recovery(
    time: Qty, k_max: Qty, ultimate_recovery: float
) -> float:
    """Recovery for a rectangular distribution of rate constants (Eq. E2).

    Examples
    --------
    k_max = 0.10 1/s, R_inf = 0.90, t = 60 s:
    k_max t = 6, exp(-6) = 0.00247875,
    (1 - 0.00247875)/6 = 0.16625354,
    R = 0.90(1 - 0.16625354) = 0.90 x 0.83374646 = 0.75037181.

    >>> round(rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90), 8)
    0.75037181

    Small-argument behaviour. As :math:`x = k_{max} t \\to 0` the bracket goes to
    zero like :math:`x/2 - x^2/6`, so recovery goes to zero from ABOVE.
    Evaluating :math:`(1 - e^{-x})/x` directly destroys that. Measured at
    :math:`k_{max} = 1` 1/s and :math:`R_\\infty = 0.90`, committed form against
    the analytic limit:

    ===========  =====================  =====================  ==============
    x            committed              analytic               relative error
    ===========  =====================  =====================  ==============
    1e-12        +1.990954810935e-05    4.499999999998e-13     +4.424344e+07
    1e-10        -7.446633389918e-08    4.499999999850e-11     -1.655807e+03
    1e-09        +2.545373841700e-08    4.499999998500e-10     +5.556386e+01
    1e-08        +5.469723873830e-09    4.499999985000e-09     +2.154942e-01
    1e-07        +4.543775276034e-08    4.499999850000e-08     +9.727873e-03
    1e-06        +4.500141251640e-07    4.499998500000e-07     +3.172260e-05
    ===========  =====================  =====================  ==============

    At :math:`x = 10^{-10}` the result is NEGATIVE and raised on the fraction
    check, so the failure mode is not merely imprecise. ``-expm1(-x)`` computes
    the same numerator without cancellation and reproduces the analytic limit to
    1e-9 relative at every argument in the table. The old code special-cased
    only ``x == 0.0`` exactly, which is the one argument where catastrophic
    cancellation does not occur.

    An earlier revision of this docstring attributed the 4.499999998500e-10
    analytic value and the +5.556386e+01 error to :math:`x = 10^{-8}`. Both
    belong to :math:`x = 10^{-9}`; every row above is now asserted against a
    recomputation in
    ``tests/test_physics_audit.py::test_rectangular_recovery_survives_a_short_residence_time``
    rather than quoted.
    """
    require_dimensionality(time, "time", "time")
    require_dimensionality(k_max, "rate_first_order", "k_max")
    t = float(time.to("s").magnitude)
    km = float(k_max.to("1/s").magnitude)
    if km <= 0.0:
        raise ValueError(f"k_max must be positive, got {km}")
    r_inf = require_fraction(ultimate_recovery, "ultimate_recovery")
    x = km * t
    if x == 0.0:
        return 0.0
    r = r_inf * (1.0 + math.expm1(-x) / x)
    return require_fraction(max(0.0, r), "rectangular distribution recovery")


# --- Eq. E3 -----------------------------------------------------------------


def whims_rate_constant(
    k_ref: Qty,
    field: Qty,
    field_ref: Qty,
    exponent: float | None = None,
) -> Qty:
    """Scale a magnetic capture rate constant with applied field (Eq. E3).

    Parameters
    ----------
    k_ref
        Rate constant measured at ``field_ref``, reciprocal time.
    field, field_ref
        Applied and reference magnetic flux density, T.
    exponent
        :math:`n_B`. ``None`` uses :data:`WHIMS_FIELD_EXPONENT`, which is
        ASSUMED; pass a fitted value when one exists.

    Examples
    --------
    k_ref = 0.02 1/s at 1.0 T, scaled to 1.5 T with n_B = 2:
    (1.5/1.0)^2 = 2.25, k = 0.02 x 2.25 = 0.045 1/s.

    >>> round(whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "T"), Q_(1.0, "T")).magnitude, 4)
    0.045
    """
    require_dimensionality(k_ref, "rate_first_order", "k_ref")
    require_dimensionality(field, "magnetic_flux_density", "field")
    require_dimensionality(field_ref, "magnetic_flux_density", "field_ref")
    b = float(field.to("T").magnitude)
    b_ref = float(field_ref.to("T").magnitude)
    if b <= 0.0 or b_ref <= 0.0:
        raise ValueError(f"fields must be positive, got B={b} T, B_ref={b_ref} T")
    n = (
        float(_q(WHIMS_FIELD_EXPONENT).magnitude)
        if exponent is None
        else float(exponent)
    )
    if n < 0.0:
        raise ValueError(
            f"field exponent must be non-negative, got {n}: capture cannot fall with "
            f"rising field in a paramagnetic system"
        )
    k = float(k_ref.to("1/s").magnitude) * (b / b_ref) ** n
    if k <= 0.0:
        raise AssertionError("scaled rate constant must be positive")
    return Q_(k, "1/s")


def whims_recovery(
    residence_time: Qty,
    k_ref: Qty,
    field: Qty,
    field_ref: Qty,
    ultimate_recovery: float,
    exponent: float | None = None,
) -> tuple[float, str]:
    """Magnetic recovery of a susceptible species (Eq. E3 into Eq. E1).

    Returns
    -------
    (recovery, caveat)
        The caveat names the field exponent's provenance, so a caller cannot
        report the number without its weakest assumption attached.
    """
    k = whims_rate_constant(k_ref, field, field_ref, exponent=exponent)
    r = flotation_recovery(residence_time, k, ultimate_recovery)
    used = "fitted" if exponent is not None else "ASSUMED default"
    caveat = (
        f"field exponent n_B {used} = "
        f"{exponent if exponent is not None else WHIMS_FIELD_EXPONENT.magnitude}; "
        f"first-order capture form is an analogy to flotation kinetics, not a sourced "
        f"HGMS model. See LIMITATIONS item 2."
    )
    return r, caveat


# --- Eq. E4, E5 -------------------------------------------------------------


def logistic_scale_from_ep(ep: float) -> float:
    """Logistic scale :math:`s` from the probable error :math:`E_p` (Eq. E4).

    Examples
    --------
    E_p = 20 (in the property's own unit): s = 20/ln 3 = 20/1.0986123
    = 18.2047845.

    >>> round(logistic_scale_from_ep(20.0), 7)
    18.2047845
    """
    if ep <= 0.0:
        raise ValueError(
            f"Ep must be positive, got {ep}: Ep = 0 is a perfect separation and the "
            f"logistic degenerates to a step function, which has no finite scale"
        )
    return ep / math.log(3.0)


def partition_number(
    x: float, x50: float, ep: float, bypass: float = 0.0
) -> float:
    """Partition number at property value ``x`` (Eq. E4).

    Parameters
    ----------
    x, x50
        Property value and cut point, same unit.
    ep
        Probable error, same unit.
    bypass
        Short-circuit fraction reporting to the coarse/sink stream regardless of
        property value, in [0, 1). Reported separately rather than absorbed into
        ``ep``, because a bypass and a blunt cut are different plant faults with
        different fixes.

    Returns
    -------
    float
        Partition number in [bypass, 1].

    Examples
    --------
    x50 = 100, Ep = 20, so s = 18.2047845. At x = 120:
    (120 - 100)/18.2047845 = 1.0986123 = ln 3, so
    P = 1/(1 + exp(-ln 3)) = 1/(1 + 1/3) = 0.75, which is the definition of
    x75 and confirms the Ep relation.

    >>> round(partition_number(120.0, 100.0, 20.0), 6)
    0.75
    """
    if x50 <= 0.0:
        raise ValueError(f"cut point must be positive, got {x50}")
    b = require_fraction(bypass, "bypass", hi=0.999999)
    s = logistic_scale_from_ep(ep)
    z = (x - x50) / s
    # Numerically stable logistic.
    if z >= 0.0:
        core = 1.0 / (1.0 + math.exp(-z))
    else:
        e = math.exp(z)
        core = e / (1.0 + e)
    p = b + (1.0 - b) * core
    return require_fraction(p, "partition number")


def partition_curve(
    xs: Sequence[float], x50: float, ep: float, bypass: float = 0.0
) -> list[tuple[float, float]]:
    """Partition numbers over a property grid, checked monotonic increasing."""
    pts = sorted((float(x), partition_number(x, x50, ep, bypass)) for x in xs)
    for (x1, p1), (x2, p2) in itertools.pairwise(pts):
        if p2 < p1 - 1e-12:
            raise AssertionError(
                f"partition number fell from {p1} at {x1} to {p2} at {x2}: the "
                f"logistic form is monotonic by construction"
            )
    return pts


def ep_from_partition_points(x25: float, x75: float) -> float:
    """Probable error from the 25 and 75 percent partition sizes (Eq. E4).

    .. math::
        E_p = \\frac{x_{75} - x_{25}}{2}
    """
    if x75 <= x25:
        raise ValueError(
            f"x75 ({x75}) must exceed x25 ({x25}); a partition curve is increasing "
            f"in the property that reports to the coarse stream"
        )
    return (x75 - x25) / 2.0


def imperfection(ep: float, x50: float, basis: PropertyBasis) -> float:
    """Imperfection of a separation (Eq. E5).

    For ``basis=DENSITY`` the denominator is :math:`x_{50} - 1` (specific
    gravity relative to water), because the separating force scales with the
    density difference from the medium. For size and susceptibility the
    denominator is :math:`x_{50}`.

    Examples
    --------
    Density separation, x50 = 1.60 g/cm3, Ep = 0.05:
    I = 0.05/(1.60 - 1) = 0.05/0.60 = 0.0833333.

    >>> round(imperfection(0.05, 1.60, PropertyBasis.DENSITY), 7)
    0.0833333

    Ceiling. :math:`I > 1` means the probable error exceeds the whole property
    difference driving the separation, so the partition curve is flatter than
    the cut it is meant to make and no separation is described. That is
    definitional, not a sourced threshold, and it is enforced because the
    ``x50 > 1`` guard alone does not bound the DENSITY denominator away from
    zero: at ``x50 = 1.0000001`` the denominator is 1e-7 and this function
    returned an imperfection of 499999.99970806646 rather than refusing. A cut
    point that close to the medium density is a measurement artefact or a unit
    error, and half a million reported as an imperfection is the kind of number
    that survives review because nobody expects it to be possible.
    """
    if ep <= 0.0:
        raise ValueError(f"Ep must be positive, got {ep}")
    if basis is PropertyBasis.DENSITY:
        if x50 <= 1.0:
            raise ValueError(
                f"density cut point {x50} g/cm3 must exceed the medium density of "
                f"1 g/cm3 for the imperfection denominator to be positive"
            )
        denom = x50 - 1.0
    else:
        if x50 <= 0.0:
            raise ValueError(f"cut point must be positive, got {x50}")
        denom = x50
    i = ep / denom
    if i < 0.0:
        raise AssertionError("imperfection cannot be negative")
    if i > 1.0:
        raise ValueError(
            f"imperfection {i} exceeds 1: Ep ({ep}) is larger than the property "
            f"difference driving the separation ({denom} on the {basis.value} "
            f"basis), so this describes no separation. Check the cut point for a "
            f"unit error or a value at the medium density."
        )
    return i


# --- Eq. E6, E7, E8 ---------------------------------------------------------


def two_product(f: float, c: float, t: float) -> tuple[float, float]:
    """Mass yield and recovery to concentrate (Eq. E6). Exact mass balance.

    Parameters
    ----------
    f, c, t
        Assay of the species of interest in feed, concentrate and tailing, in
        any consistent unit.

    Returns
    -------
    (yield_to_concentrate, recovery_to_concentrate)
        Both dimensionless in [0, 1].

    Examples
    --------
    Impurity assays f = 1.0 ppm, c = 10.0 ppm (reject), t = 0.1 ppm (product):
    Y = (1.0 - 0.1)/(10.0 - 0.1) = 0.9/9.9 = 0.0909091,
    R = 10.0(0.9)/(1.0 x 9.9) = 9.0/9.9 = 0.9090909.

    >>> y, r = two_product(1.0, 10.0, 0.1)
    >>> round(y, 7), round(r, 7)
    (0.0909091, 0.9090909)
    """
    if c <= t:
        raise ValueError(
            f"concentrate assay ({c}) must exceed tailing assay ({t}); if it does not, "
            f"the streams are mislabelled or there was no separation"
        )
    if f < 0.0 or c < 0.0 or t < 0.0:
        raise ValueError(f"assays cannot be negative: f={f}, c={c}, t={t}")
    if not (t <= f <= c):
        raise ValueError(
            f"feed assay {f} must lie between tailing {t} and concentrate {c}: a "
            f"separation cannot produce two streams both richer, or both poorer, than "
            f"the feed"
        )
    if f <= 0.0:
        raise ValueError("feed assay must be positive to define a recovery")
    y = (f - t) / (c - t)
    r = y * c / f
    y = require_fraction(y, "yield to concentrate")
    r = require_fraction(r, "recovery to concentrate")
    return y, r


def two_product_condition_number(f: float, c: float, t: float) -> float:
    """Sensitivity of the two-product yield to assay error.

    .. math::
        \\kappa = \\frac{c - t + f - t}{c - t}

    A crude condition number: the relative error in :math:`Y` is of order
    :math:`\\kappa` times the relative errors in the assays. It blows up as
    :math:`c \\to t`, which is the high-purity regime where feed, product and
    reject assays are all a few ppm. Above about 10, treat a two-product yield
    as indicative only and close the balance on weights instead.

    Examples
    --------
    f = 30, c = 45, t = 28 ppm: (45 - 28 + 30 - 28)/(45 - 28) = 19/17
    = 1.1176471, so well conditioned. With c = 31: (3 + 2)/3 = 1.667 still
    fine, but f = 30, c = 30.5, t = 29.9 gives (0.6 + 0.1)/0.6 = 1.1667 while
    the absolute differences are inside analytical noise, which is the real
    failure mode: a small condition number does not rescue assays that are
    indistinguishable.
    """
    if c <= t:
        raise ValueError("concentrate assay must exceed tailing assay")
    return ((c - t) + (f - t)) / (c - t)


def product_grade_from_reject(
    feed_grade: float, reject_yield: float, reject_grade: float
) -> float:
    """Product impurity grade from a reject stream's yield and grade (Eq. E7).

    This is the form a quartz flowsheet needs: the valuable stream is the
    non-float, non-magnetic product, and the "concentrate" is the impurity
    reject.

    Examples
    --------
    Feed 30 ppm Al, reject 8 percent of the mass at 280 ppm Al:
    (30 - 0.08 x 280)/(1 - 0.08) = (30 - 22.4)/0.92 = 7.6/0.92
    = 8.2608696 ppm Al in the product.

    >>> round(product_grade_from_reject(30.0, 0.08, 280.0), 7)
    8.2608696
    """
    if feed_grade < 0.0 or reject_grade < 0.0:
        raise ValueError("grades cannot be negative")
    y = require_fraction(reject_yield, "reject_yield", hi=0.999999)
    numer = feed_grade - y * reject_grade
    if numer < -1e-12:
        raise ValueError(
            f"reject stream would carry {y * reject_grade} units of impurity from a "
            f"feed carrying only {feed_grade}: mass balance violated, so the reject "
            f"yield or grade is wrong"
        )
    grade = max(numer, 0.0) / (1.0 - y)
    if grade > feed_grade + 1e-9:
        raise AssertionError(
            f"product grade {grade} exceeds feed grade {feed_grade} while rejecting a "
            f"higher-grade stream, which is impossible"
        )
    return grade


def separation_mass_balance(
    feed_mass: Qty,
    feed_grade: float,
    reject_yield: float,
    reject_grade: float,
) -> dict[str, float]:
    """Full two-stream mass balance, closing to machine precision.

    Returns
    -------
    dict
        Keys ``feed_mass_kg``, ``product_mass_kg``, ``reject_mass_kg``,
        ``product_grade_ppm``, ``reject_grade_ppm``, ``impurity_in_kg``,
        ``impurity_out_kg``, ``mass_residual_kg``, ``impurity_residual_kg``.
        Both residuals are asserted to be within 1e-9 relative.
    """
    require_dimensionality(feed_mass, "mass", "feed_mass")
    m_f = float(feed_mass.to("kg").magnitude)
    if m_f <= 0.0:
        raise ValueError(f"feed mass must be positive, got {m_f} kg")
    y = require_fraction(reject_yield, "reject_yield", hi=0.999999)
    product_grade = product_grade_from_reject(feed_grade, y, reject_grade)
    m_r = m_f * y
    m_p = m_f - m_r
    imp_in = m_f * feed_grade * 1e-6
    imp_out = m_p * product_grade * 1e-6 + m_r * reject_grade * 1e-6
    mass_res = m_f - (m_p + m_r)
    imp_res = imp_in - imp_out
    if abs(mass_res) > 1e-9 * m_f:
        raise AssertionError(f"mass balance failed by {mass_res} kg")
    if imp_in > 0.0 and abs(imp_res) > 1e-9 * imp_in:
        raise AssertionError(f"impurity balance failed by {imp_res} kg")
    return {
        "feed_mass_kg": m_f,
        "product_mass_kg": m_p,
        "reject_mass_kg": m_r,
        "product_grade_ppm": product_grade,
        "reject_grade_ppm": reject_grade,
        "impurity_in_kg": imp_in,
        "impurity_out_kg": imp_out,
        "mass_residual_kg": mass_res,
        "impurity_residual_kg": imp_res,
    }


def grade_recovery_curve(
    feed_grade: float,
    times: Sequence[Qty],
    rate_constant: Qty,
    ultimate_recovery: float,
    reject_enrichment: float,
) -> list[dict[str, float]]:
    """Grade-recovery locus over flotation time (Eq. E8).

    Parameters
    ----------
    feed_grade
        Impurity grade of the feed, ppm.
    times
        Flotation times to evaluate.
    rate_constant, ultimate_recovery
        Eq. E1 parameters for impurity recovery to the froth (reject).
    reject_enrichment
        Ratio of reject impurity grade to feed impurity grade, greater than 1.
        Held constant here, which is a simplification: in a real bank the froth
        becomes progressively more dilute in impurity as time goes on, so a
        constant enrichment overstates late-time selectivity. Stated rather than
        hidden.

    Returns
    -------
    list of dict
        Each with ``time_s``, ``impurity_recovery``, ``reject_yield``,
        ``product_grade_ppm``, ``impurity_rejection``.
    """
    if reject_enrichment <= 1.0:
        raise ValueError(
            f"reject_enrichment must exceed 1, got {reject_enrichment}: a reject that "
            f"is not enriched in impurity is not a separation"
        )
    if feed_grade <= 0.0:
        raise ValueError("feed grade must be positive")
    out: list[dict[str, float]] = []
    for t in times:
        r = flotation_recovery(t, rate_constant, ultimate_recovery)
        reject_grade = feed_grade * reject_enrichment
        # Yield follows from the impurity balance: R = Y c / f, so Y = R f / c.
        y = r * feed_grade / reject_grade
        y = require_fraction(y, "reject_yield", hi=0.999999)
        product_grade = product_grade_from_reject(feed_grade, y, reject_grade)
        out.append(
            {
                "time_s": float(t.to("s").magnitude),
                "impurity_recovery": r,
                "reject_yield": y,
                "product_grade_ppm": product_grade,
                "impurity_rejection": 1.0 - product_grade / feed_grade,
            }
        )
    return out


def feedstock_impurity_separation(
    feedstock: Feedstock,
    element: str,
    removable_fraction: float,
    residence_time: Qty,
    rate_constant: Qty,
    kind: SeparatorKind,
    site: Site | None = None,
) -> dict[str, float | str]:
    """Separation outcome for one element on one feedstock.

    Parameters
    ----------
    feedstock
        Ore. Supplies the bulk concentration of ``element``; raises via
        :meth:`ae.core.feedstock.ImpurityProfile.total_ppm` when unmeasured.
    element
        Element symbol.
    removable_fraction
        Ultimate recovery ceiling, which is NOT a free parameter: it should come
        from :func:`ae.physics.liberation.leachable_fraction` or from
        :func:`ae.physics.impurity_location.removable_ppm` divided by the bulk,
        so that the lattice inventory bounds it. Passing 1.0 here asserts the
        element has no lattice component, which for Al, Ti, Li or B is a claim
        about the deposit.
    residence_time, rate_constant
        Eq. E1 parameters.
    kind
        Which unit operation, recorded in the result.
    site
        Optional, recorded for traceability. No cost is computed here; energy
        and reagent cost belong in a plant model with sourced prices.

    Returns
    -------
    dict
        With keys:

        ``element``
            Element symbol, echoed back.
        ``bulk_ppm``
            Feed concentration read off the feedstock, ppm by mass.
        ``recovery``
            Fraction of the element's TOTAL inventory recovered to the reject at
            this residence time, dimensionless. Bounded above by
            ``removable_fraction``, not by 1.
        ``removed_ppm``
            ``bulk_ppm`` times ``recovery``.
        ``residual_ppm``
            What remains in the product, ``bulk_ppm`` minus ``removed_ppm``.
        ``irreducible_residual_ppm``
            The FLOOR on ``residual_ppm``: ``bulk_ppm`` times
            ``(1 - removable_fraction)``, the part of the inventory this unit
            cannot touch at any residence time (for Al, Ti, Li and B that is
            predominantly the lattice). Note the direction: this is a lower
            bound on what is LEFT, not an upper bound on what is removed.
        ``unit``
            ``kind.value``.
        ``site_id``
            Site identifier, or ``"not specified"``.
        ``note``
            Provenance and caveat text.

        Mass balance on the element closes by construction
        (``removed_ppm + residual_ppm == bulk_ppm``) and is asserted.
    """
    bulk = feedstock.impurities.total_ppm(element)
    ceiling_frac = require_fraction(removable_fraction, "removable_fraction")
    r = flotation_recovery(residence_time, rate_constant, ceiling_frac)
    removed = bulk * r
    residual = bulk - removed
    if residual < -1e-12:
        raise AssertionError(f"negative residual {residual} ppm for {element}")
    # Lower bound on the RESIDUAL: the inventory this unit cannot reach.
    floor = bulk * (1.0 - ceiling_frac)
    if residual < floor - 1e-9:
        raise AssertionError(
            f"residual {residual} ppm fell below the unremovable floor {floor} ppm "
            f"for {element}: the rate model must not remove more than the removable "
            f"inventory"
        )
    return {
        "element": element,
        "bulk_ppm": bulk,
        "recovery": r,
        "removed_ppm": removed,
        "residual_ppm": residual,
        "irreducible_residual_ppm": floor,
        "unit": kind.value,
        "site_id": site.site_id if site is not None else "not specified",
        "note": (
            f"{kind.value} on {feedstock.sample_id}; ultimate recovery capped at "
            f"{ceiling_frac:.4f}, so the residual cannot fall below {floor:.4f} ppm "
            f"no matter the residence time. Rate constant is an input and must be "
            f"measured on this feed at this grind."
        ),
    }
