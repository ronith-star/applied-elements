"""Liberation: how finely must quartz be ground before inclusions are exposed.

An acid leach, a flotation stage and a magnetic separator can only act on an
impurity they can reach. A mineral or fluid inclusion wholly enclosed by quartz
is invisible to all three no matter how aggressive the reagent, and the only
lever that exposes it is size reduction. This module puts a number on that lever
from stereology, with the geometry derived rather than asserted.

The industrial statement of the same fact, from the HPQ processing literature:
mineral impurities in quartz can be separated efficiently by flotation, magnetic
separation and conventional leaching only when they have been exposed or
liberated by crushing and grinding, and at a given size the liberation degree of
fine-grained gangue is severely limited, which is what drives low separation
efficiency (Lin M. et al. 2020, A Critical Review on the Mineralogy and
Processing for High-Grade Quartz, Mining, Metallurgy and Exploration
37:1627-1639, doi:10.1007/s42461-020-00247-0, Section 3).

Equations
---------
**(E1) Random-fracture exposure probability, cubic lattice derivation.**
Derived here from first principles, not taken from a source.

Consider a particle of characteristic size :math:`d_p` containing an inclusion of
characteristic size :math:`d_{inc}`, with the inclusion centre positioned
uniformly at random inside the particle and both idealised as coaxial cubes. The
inclusion is fully enclosed (not exposed) only if its centre lies in the
concentric interior cube of edge :math:`d_p - d_{inc}`. In one dimension the
centre must lie in an interior interval of length :math:`d_p - d_{inc}` out of
:math:`d_p`; the three axes are independent under uniform positioning, so

.. math::
    P(\\text{enclosed}) = \\left( 1 - \\frac{d_{inc}}{d_p} \\right)^{3},
    \\qquad
    E = P(\\text{exposed}) = 1 - \\left( 1 - \\frac{d_{inc}}{d_p} \\right)^{3}

for :math:`d_{inc} \\le d_p`, and :math:`E = 1` for :math:`d_{inc} > d_p`
(an inclusion larger than the particle cannot be enclosed by it).

- :math:`E`: exposure fraction, dimensionless, range [0, 1]. Interpreted as the
  fraction of inclusions of that size class intersecting a particle surface.
- :math:`d_p`: particle size, micrometres, range 1 to 10000 um. Below about
  1 um the continuum geometry and the assumption that an inclusion is small
  compared with a particle both fail.
- :math:`d_{inc}`: inclusion characteristic size, micrometres, range 0.1 to
  1000 um. Fluid inclusions in vein quartz are commonly sub-micron to tens of
  micrometres; mineral inclusions span microns to millimetres.
- :math:`r = d_{inc}/d_p`: size ratio, dimensionless, range [0, 1] in the
  non-trivial branch.

Expanding for small :math:`r` gives :math:`E \\approx 3r - 3r^2 + r^3`, so
exposure rises initially at three times the size ratio: three orthogonal faces
each contribute one chance of intersection. That factor of 3 is the geometric
content of the model and is the reason grinding buys exposure faster than a
naive one-dimensional argument suggests.

**(E2) Liberation size for a target exposure.** Inverting E1.

.. math::
    d_p^{*} = \\frac{d_{inc}}{1 - (1 - E^{*})^{1/3}}

- :math:`d_p^{*}`: required particle size, micrometres.
- :math:`E^{*}`: target exposure fraction, dimensionless, in (0, 1).
  :math:`E^{*} = 1` requires :math:`d_p^{*} = d_{inc}`, i.e. grinding to the
  inclusion size, which is the asymptote and usually uneconomic.

Worked consequence: for :math:`E^{*} = 0.5`, the denominator is
:math:`1 - 0.5^{1/3} = 1 - 0.793701 = 0.206299`, so
:math:`d_p^{*} = 4.847 \\, d_{inc}`. Half of all inclusions are exposed once
particles are ground to roughly five times the inclusion size. For
:math:`E^{*} = 0.9`, :math:`1 - 0.1^{1/3} = 1 - 0.464159 = 0.535841` and
:math:`d_p^{*} = 1.866 \\, d_{inc}`: the last 40 percent of exposure costs a
2.6-fold further size reduction, and by Bond's law (see
:mod:`ae.physics.comminution`) energy scales as :math:`d_p^{-1/2}`, so that
final push costs about 60 percent more grinding energy per tonne.

**(E3) Spherical variant.** Same argument with concentric spheres, for
comparison rather than for use.

.. math::
    E_{sph} = 1 - \\left( 1 - \\frac{d_{inc}}{d_p} \\right)^{3}

The cubic and spherical derivations give the SAME expression, because in both
cases the enclosed-centre region is a concentric body linearly shrunk by
:math:`d_{inc}` in each dimension and volume scales as the cube of the linear
scale. This coincidence is worth stating: it means E1 is robust to particle
shape under uniform inclusion positioning, and correspondingly that agreement
between the two is NOT evidence the model is right.

**(E4) Polydisperse inclusion population.** Exposure over a size distribution.

.. math::
    \\bar{E}(d_p) = \\sum_i w_i \\, E(d_p, d_{inc,i}),
    \\qquad \\sum_i w_i = 1

- :math:`w_i`: number or volume weight of inclusion size class :math:`i`,
  dimensionless. Whether the weights are number-based or volume-based changes
  the answer substantially, because the mass of impurity sits in the large
  inclusions while the count sits in the small ones. This module requires the
  basis to be stated (:class:`InclusionWeighting`) and does not guess.

**(E5) Leachable fraction of an element.** Couples liberation to chemistry.

.. math::
    f_{\\text{leachable}}(d_p) = f_{\\text{surface}}
    + f_{\\text{fluid}} E_{\\text{fluid}}(d_p)
    + f_{\\text{mineral}} E_{\\text{mineral}}(d_p)

with :math:`f_{\\text{surface}} + f_{\\text{fluid}} +
f_{\\text{mineral}} + f_{\\text{lattice}} = 1`. The lattice term never appears:
it is not leachable at any particle size, which is the whole point.
Partitions come from :mod:`ae.physics.impurity_location` and must be measured;
this module never supplies them.

LIMITATIONS
-----------
1. **This is a geometric bound, not a calibrated liberation model.** A real
   liberation curve is measured by image analysis (QEMSCAN, MLA or SEM point
   counting on sized fractions) and fitted; the classical treatment is King
   1979 (A model for the quantitative estimation of mineral liberation by
   grinding, International Journal of Mineral Processing 6:207-220,
   doi:10.1016/0301-7516(79)90037-1). No image-analysis calibration exists for
   any Applied Elements sample, and none was retrievable for quartz-hosted
   inclusions generally in this environment. Read E1 as the answer to "what
   does random positioning alone imply", and expect the real curve to differ.
2. **Fracture is not random with respect to inclusions.** Real breakage is
   preferential: cracks nucleate at inclusions and at grain boundaries because
   these are stress concentrators and weak planes, so true exposure at a given
   :math:`d_p` is generally HIGHER than E1 predicts. E1 is therefore best read
   as a conservative (pessimistic) bound on exposure, hence an optimistic bound
   on the grinding energy needed. The size of the gap is exactly what image
   analysis would measure and this module does not know it.
3. **Thermal decrepitation bypasses the geometry entirely.** Fluid inclusions
   are conventionally opened by calcination and quenching rather than by
   grinding: Lin et al. 2020 report calcination at 600 to 700 degC being
   effective for fluid inclusion removal via the alpha to beta quartz
   transition near 573 degC, with other work supporting 900 degC as more
   effective. When a flowsheet includes calcination plus water quenching, the
   fluid inclusion branch of E5 is governed by decrepitation, not by
   :math:`d_p`, and using E1 for it understates removal. :func:`exposure` will
   not silently model that; a decrepitation model belongs in the thermal track.
4. **One characteristic size per inclusion class.** Inclusions are not cubes or
   spheres; secondary fluid inclusions in vein quartz develop as planar trails
   along microfractures (Xia M. et al. 2024, Minerals 14(7):727,
   doi:10.3390/min14070727), and a planar trail is exposed far more readily than
   an equiaxed body of the same volume. E1 has no aspect-ratio parameter, so it
   underestimates exposure for trail-hosted populations specifically.
5. **Exposure is not recovery.** An exposed inclusion is accessible to a
   reagent; whether it is actually removed depends on leach kinetics,
   diffusion into a partly-closed cavity, reagent selectivity and residence
   time. E5 returns the fraction that CAN be removed by a perfect downstream
   step, an upper bound on what a real one achieves.
6. **No particle size distribution.** :math:`d_p` here is a single size, in
   practice a P80 or a sieve-fraction midpoint. Within one sieve fraction the
   fines are more liberated than the coarse, and a real circuit produces a
   spread. Passing a P80 into E1 gives the exposure of a hypothetical
   monodisperse feed at that size.

References
----------
Every entry below was resolved against the Crossref REST API for the DOI shown,
and the fields here (author list, year, title, journal, volume, issue, pages)
are as Crossref returned them. Resolved 17 September 2026. Entries without a
DOI say what was checked instead and what remains unverified.

King, R. P. (1979) A model for the quantitative estimation of mineral
liberation by grinding, International Journal of Mineral Processing 6(3),
207-220, doi 10.1016/0301-7516(79)90037-1. Cited as the standard stereological
liberation model. This module does NOT implement it: E1 here is the geometric
exposure probability, which is a weaker and different construct. Closed access,
abstract only from this sandbox.

Lin, M., Liu, Z., Wei, Y., Liu, B., Meng, Y., Qiu, H., Lei, S., Zhang, X. and
Li, Y. (2020) A Critical Review on the Mineralogy and Processing for High-Grade
Quartz, Mining, Metallurgy and Exploration 37(5), 1627-1639,
doi 10.1007/s42461-020-00247-0.

Xia, M., Yang, X. and Hou, Z. (2024) Preparation of High-Purity Quartz Sand by
Vein Quartz Purification and Characteristics: A Case Study of Pakistan Vein
Quartz, Minerals 14(7), 727, doi 10.3390/min14070727. Feed-to-product endpoints
only; NO per-stage intermediate is reported there or modelled here.
"""

from __future__ import annotations

import datetime as _dt
import enum
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

from ae.core.feedstock import Feedstock, InclusionType
from ae.core.provenance import Source, Tier, Value, _Missing
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

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
    "SRC_KING_1979",
    "SRC_LIN_2020",
    "SRC_XIA_2024",
    "InclusionWeighting",
    "enclosed_fraction",
    "energy_to_exposure_ratio",
    "exposure",
    "exposure_curve",
    "feedstock_exposure",
    "leachable_fraction",
    "liberation_size",
    "polydisperse_exposure",
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
        "Full text read. Section 3 on liberation by crushing and grinding; Table 3 "
        "conceptual flow diagram with product grade at each stage; calcination "
        "temperatures for fluid inclusion removal."
    ),
)

SRC_KING_1979: Final[Source] = Source(
    citation=(
        "King R. P. 1979, A model for the quantitative estimation of mineral "
        "liberation by grinding, International Journal of Mineral Processing 6:207-220"
    ),
    tier=Tier.T1,
    doi="10.1016/0301-7516(79)90037-1",
    accessed=_ACCESSED,
    note=(
        "Metadata verified via CrossRef; full text not retrievable in this environment "
        "(closed access, no TDM link). Cited to identify the standard calibrated "
        "liberation modelling approach that this module does NOT implement, never as "
        "support for a numerical result here."
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
        "Abstract read in full; publisher blocked full-text retrieval in this "
        "environment. Cited for the reported endpoints (128.86 to 24.23 ug/g, 81.20 "
        "percent removal, residue identified as lattice Al, Ti, Li, SiO2 99.998 wt%) "
        "and for the inclusion description (large primary inclusions plus secondary "
        "fluid inclusions along microfractures). The paper reports NO per-stage "
        "breakdown, so no intermediate is attributed to it anywhere in this platform."
    ),
)


class InclusionWeighting(str, enum.Enum):
    """Basis on which an inclusion size distribution is weighted.

    Required, not defaulted. Number weighting emphasises the abundant fine
    inclusions and gives a pessimistic exposure at coarse sizes; volume (or
    equivalently mass) weighting emphasises the few large inclusions that
    actually carry the impurity mass and gives a more optimistic figure. The two
    can differ by a large factor for a broad distribution, and which one is
    correct depends on the question: mass removal needs volume weighting, count
    of residual defect sites needs number weighting.
    """

    NUMBER = "number"
    VOLUME = "volume"


def enclosed_fraction(particle_size: Qty, inclusion_size: Qty) -> float:
    """Probability an inclusion is fully enclosed by the particle (Eq. E1).

    Parameters
    ----------
    particle_size
        :math:`d_p`, any length unit.
    inclusion_size
        :math:`d_{inc}`, any length unit.

    Returns
    -------
    float
        Enclosed probability in [0, 1].
    """
    require_dimensionality(particle_size, "length", "particle_size")
    require_dimensionality(inclusion_size, "length", "inclusion_size")
    dp = float(particle_size.to("um").magnitude)
    di = float(inclusion_size.to("um").magnitude)
    if dp <= 0.0:
        raise ValueError(f"particle size must be positive, got {dp} um")
    if di <= 0.0:
        raise ValueError(f"inclusion size must be positive, got {di} um")
    if di >= dp:
        return 0.0
    enc = (1.0 - di / dp) ** 3
    return require_fraction(enc, "enclosed_fraction")


def exposure(particle_size: Qty, inclusion_size: Qty) -> float:
    """Exposure fraction E of an inclusion size class at a particle size (Eq. E1).

    Examples
    --------
    A 10 um inclusion in a 100 um particle: r = 0.1, so
    E = 1 - 0.9^3 = 1 - 0.729 = 0.271. About 27 percent of such inclusions
    intersect a particle surface.

    >>> round(exposure(Q_(100.0, "um"), Q_(10.0, "um")), 3)
    0.271

    Grinding the same material to 30 um: r = 1/3, E = 1 - (2/3)^3
    = 1 - 0.296296 = 0.703704.

    >>> round(exposure(Q_(30.0, "um"), Q_(10.0, "um")), 4)
    0.7037
    """
    e = 1.0 - enclosed_fraction(particle_size, inclusion_size)
    return require_fraction(e, "exposure")


def liberation_size(inclusion_size: Qty, target_exposure: float) -> Qty:
    """Particle size required to reach a target exposure fraction (Eq. E2).

    Parameters
    ----------
    inclusion_size
        :math:`d_{inc}`.
    target_exposure
        :math:`E^{*}` in (0, 1]. At exactly 1.0 the required size equals the
        inclusion size, which is the geometric asymptote of the model.

    Examples
    --------
    Half-exposure of a 20 um inclusion population:
    1 - 0.5^(1/3) = 1 - 0.7937005 = 0.2062995, and
    20 / 0.2062995 = 96.95 um.

    >>> round(liberation_size(Q_(20.0, "um"), 0.5).to("um").magnitude, 2)
    96.95
    """
    require_dimensionality(inclusion_size, "length", "inclusion_size")
    e = require_fraction(target_exposure, "target_exposure")
    if e <= 0.0:
        raise ValueError(
            "target_exposure must be positive; zero exposure is achieved at any "
            "particle size and the inverse is unbounded"
        )
    denom = 1.0 - (1.0 - e) ** (1.0 / 3.0)
    di = float(inclusion_size.to("um").magnitude)
    dp = di / denom
    if dp < di:
        raise AssertionError(
            f"liberation size {dp} um below the inclusion size {di} um is unphysical"
        )
    return Q_(dp, "um")


def polydisperse_exposure(
    particle_size: Qty,
    inclusion_sizes: Sequence[Qty],
    weights: Sequence[float],
    weighting: InclusionWeighting,
) -> float:
    """Weighted mean exposure over an inclusion size distribution (Eq. E4).

    Parameters
    ----------
    particle_size
        :math:`d_p`.
    inclusion_sizes
        Characteristic size of each class.
    weights
        Class weights. Must be non-negative and sum to 1 within 1e-9, so a
        distribution that does not close fails loudly rather than being
        renormalised behind the caller's back.
    weighting
        Declared basis of ``weights``. Recorded and range-checked; it does not
        change the arithmetic, because renormalising a number basis into a
        volume basis requires the inclusion volumes, which the caller has and
        this function does not.
    """
    if len(inclusion_sizes) != len(weights):
        raise ValueError(
            f"{len(inclusion_sizes)} size classes but {len(weights)} weights"
        )
    if not inclusion_sizes:
        raise ValueError("at least one inclusion size class is required")
    if any(w < 0.0 for w in weights):
        raise ValueError(f"weights must be non-negative, got {list(weights)}")
    total = math.fsum(weights)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"weights must sum to 1 on the declared {weighting.value} basis, "
            f"got {total}. Mass balance on the inclusion population must close."
        )
    e = math.fsum(
        w * exposure(particle_size, d)
        for w, d in zip(weights, inclusion_sizes, strict=True)
    )
    return require_fraction(e, "polydisperse_exposure")


def exposure_curve(
    inclusion_size: Qty,
    particle_sizes: Sequence[Qty],
) -> list[tuple[float, float]]:
    """Exposure as a function of particle size, for plotting a liberation curve.

    Returns
    -------
    list of (particle_size_um, exposure)
        Sorted by ascending particle size. Monotonically decreasing in exposure,
        which is asserted before return.
    """
    pts = sorted(
        (float(p.to("um").magnitude), exposure(p, inclusion_size))
        for p in particle_sizes
    )
    for (d1, e1), (d2, e2) in itertools.pairwise(pts):
        if e2 > e1 + 1e-12:
            raise AssertionError(
                f"exposure rose from {e1} at {d1} um to {e2} at {d2} um: grinding "
                f"coarser cannot expose more inclusions"
            )
    return pts


def leachable_fraction(
    partition: Mapping[str, float],
    particle_size: Qty,
    fluid_inclusion_size: Qty,
    mineral_inclusion_size: Qty,
) -> float:
    """Fraction of an element accessible to a downstream removal step (Eq. E5).

    Parameters
    ----------
    partition
        Mapping with keys ``"surface"``, ``"fluid"``, ``"mineral"`` and
        ``"lattice"``, each a fraction of the element's total inventory. Must
        sum to 1 within 1e-9. Obtain it from
        :func:`ae.physics.impurity_location.partition_fractions`, which raises
        when the split is unmeasured; do NOT hand-write it.
    particle_size
        :math:`d_p` at the point the removal step acts.
    fluid_inclusion_size, mineral_inclusion_size
        Characteristic sizes of the two inclusion populations.

    Returns
    -------
    float
        Upper bound on the removable fraction at this particle size. Surface
        films are taken as fully accessible; the lattice term is excluded by
        construction and can never be reached.
    """
    required = {"surface", "fluid", "mineral", "lattice"}
    missing = required - set(partition)
    if missing:
        raise KeyError(
            f"partition is missing {sorted(missing)}; all four locations must be "
            f"present so the inventory closes"
        )
    extra = set(partition) - required
    if extra:
        raise KeyError(f"unknown partition keys {sorted(extra)}")
    for k, v in partition.items():
        require_fraction(v, f"partition[{k}]")
    total = math.fsum(partition[k] for k in required)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"partition fractions sum to {total}, not 1: the element's inventory "
            f"must close over surface, fluid, mineral and lattice"
        )
    e_fluid = exposure(particle_size, fluid_inclusion_size)
    e_mineral = exposure(particle_size, mineral_inclusion_size)
    f = (
        partition["surface"]
        + partition["fluid"] * e_fluid
        + partition["mineral"] * e_mineral
    )
    ceiling = 1.0 - partition["lattice"]
    if f > ceiling + 1e-12:
        raise AssertionError(
            f"leachable fraction {f} exceeds the non-lattice inventory {ceiling}: "
            f"a leach cannot remove lattice-substituted material"
        )
    return require_fraction(f, "leachable_fraction")


def feedstock_exposure(
    feedstock: Feedstock,
    particle_size: Qty,
    inclusion_type: InclusionType = InclusionType.FLUID,
) -> tuple[float, Value]:
    """Exposure for a feedstock's own measured inclusion population.

    Reads ``feedstock.inclusions.median_inclusion_size``. Raises when it has not
    been measured, because substituting a typical inclusion size would fabricate
    the liberation size, and the liberation size sets the grinding energy and
    hence a large part of the operating cost.

    Returns
    -------
    (exposure, inclusion_size_value)
        The Value is returned so the caller can propagate its provenance.

    Raises
    ------
    ValueError
        If the inclusion size is MISSING.
    """
    d = feedstock.inclusions.median_inclusion_size
    # ValueError, not TypeError, despite the isinstance check: _Missing is the
    # core module's sentinel for "not measured", so the fault is an absent value,
    # not a wrongly typed argument. This matches ImpurityProfile.lattice_ppm.
    if isinstance(d, _Missing):
        raise ValueError(  # noqa: TRY004
            f"{feedstock.sample_id} has no measured median inclusion size "
            f"(inclusions.median_inclusion_size is MISSING). Exposure and liberation "
            f"size cannot be computed. Measure it by transmitted-light petrography "
            f"or SEM on polished sections; an assumed inclusion size would fabricate "
            f"the grind target and therefore the grinding energy."
        )
    value: Value = d
    require_dimensionality(_q(value), "length", "median_inclusion_size")
    if inclusion_type is InclusionType.MELT:
        raise TypeError(
            "melt inclusions are not modelled: their response to grinding and to "
            "calcination differs from fluid and mineral inclusions and no calibration "
            "was available"
        )
    return exposure(particle_size, _q(value)), value


def energy_to_exposure_ratio(
    coarse_size: Qty, fine_size: Qty, inclusion_size: Qty
) -> tuple[float, float, float]:
    """Exposure gain versus Bond energy cost between two grind sizes.

    Returns
    -------
    (exposure_gain, bond_energy_ratio, exposure_per_energy)
        ``bond_energy_ratio`` is the ratio of Bond specific energies to reach
        the two sizes from a common feed, which for a coarse feed is
        approximately :math:`\\sqrt{d_{coarse}/d_{fine}}` because Bond's law
        gives :math:`W \\propto P_{80}^{-1/2}`. It is work-index independent, so
        this comparison needs no ore property, which is why it is honest for an
        uncharacterized deposit.

    Notes
    -----
    This is the decision-relevant ratio for choosing a grind target: exposure
    has diminishing returns (Eq. E1 saturates) while energy has increasing cost
    (:math:`P_{80}^{-1/2}` diverges), so an optimum exists and it does not sit
    at the finest achievable size.
    """
    dc = float(coarse_size.to("um").magnitude)
    df = float(fine_size.to("um").magnitude)
    if df >= dc:
        raise ValueError(
            f"fine_size ({df} um) must be smaller than coarse_size ({dc} um)"
        )
    e_c = exposure(coarse_size, inclusion_size)
    e_f = exposure(fine_size, inclusion_size)
    gain = e_f - e_c
    if gain < -1e-12:
        raise AssertionError("finer grinding reduced exposure, which is unphysical")
    energy_ratio = math.sqrt(dc / df)
    return gain, energy_ratio, gain / energy_ratio
