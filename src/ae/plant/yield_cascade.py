r"""Yield cascade and statistical process control translated to off-spec fraction.

Two distinct things are conflated in casual plant talk and separated here.

**Mass yield cascade.** Fraction of feed mass surviving to product:

.. math::
   Y = \prod_{i=1}^{n} y_i

with :math:`y_i` the mass yield of stage :math:`i` (dimensionless, 0 to 1). The
reciprocal :math:`1/\prod_{j>i} y_j` is what stage :math:`i` must process per
tonne shipped, which is the basis :mod:`ae.plant.capacity` requires.

**Specification yield.** Fraction of LOTS meeting a purity specification. This
is not a mass loss at all: a lot that assays 34 ppm Al against a 30 ppm limit is
a full lot of saleable mass that cannot be sold into that grade. It follows from
the process capability, not from the flowsheet.

For a one-sided upper specification limit :math:`\mathrm{USL}` on an impurity,
with lot assays distributed :math:`N(\mu, \sigma^2)`:

.. math::
   f_{\mathrm{off}} = \Pr[X > \mathrm{USL}]
                    = 1 - \Phi\!\left(\frac{\mathrm{USL} - \mu}{\sigma}\right)
                    = 1 - \Phi\!\left(3 C_{pk}\right)

where the one-sided capability index is

.. math::
   C_{pk} = \frac{\mathrm{USL} - \mu}{3\sigma}

:math:`\mu` is the process mean (ppm, positive), :math:`\sigma` the lot-to-lot
standard deviation (ppm, positive), and :math:`\Phi` the standard normal CDF.
Inverting gives the process mean a target capability demands:

.. math::
   \mu = \mathrm{USL} - 3 C_{pk} \sigma

which is the operationally useful direction: a customer asks for a capability,
and this says how far below the limit the process must actually run.

Worked consequence, at USL = 30 ppm Al and :math:`C_{pk} = 1.33`:
:math:`\sigma = 3` ppm requires :math:`\mu = 30 - 3(1.33)(3) = 18.0` ppm, and
:math:`\sigma = 5` ppm requires :math:`\mu = 10.0` ppm. Halving variability is
worth more than a large shift in mean, which is why lot-to-lot control, not
best-ever assay, is the qualification-relevant quantity.

Sources
-------
Capability indices and the normal-tail relation are standard SPC: Montgomery,
*Introduction to Statistical Quality Control*, 7th ed., Wiley 2012, chapter 6.
The multi-lot qualification requirement that makes lot-to-lot sigma the binding
quantity is the JEDEC change-control regime (JESD46), not a statistical result.

LIMITATIONS
-----------
* Normality is assumed. Impurity assays are bounded below by zero and are
  frequently right-skewed, so the normal tail UNDERSTATES off-spec fraction in
  the tail that matters. :func:`off_spec_fraction` therefore also accepts a
  lognormal model, and :func:`off_spec_fraction_empirical` takes assays directly
  and makes no distributional assumption at all. Prefer the empirical form once
  any real lot data exists.
* Independence between lots is assumed. Real plants drift, so consecutive lots
  are correlated and the effective sample size is smaller than the lot count.
  That makes a capability estimated from few lots optimistic.
* A capability index computed from fewer than roughly 25 to 30 lots has a wide
  confidence interval. :func:`capability` reports that interval rather than a
  bare point estimate, and refuses fewer than 2 lots.
* Multi-element specifications are NOT independent: the same fluid-inclusion
  population carries Na, K and Li together. :func:`joint_off_spec_fraction`
  offers both the independent bound and a fully-correlated bound, and the truth
  lies between them. Do not use the independent product as the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy import stats

from ae.core.units import require_fraction

__all__ = [
    "cascade_yield",
    "stage_throughput_factors",
    "capability",
    "Capability",
    "off_spec_fraction",
    "off_spec_fraction_empirical",
    "required_process_mean",
    "joint_off_spec_fraction",
]


def cascade_yield(stage_yields: Sequence[float]) -> float:
    """Product of stage mass yields.

    Examples
    --------
    >>> round(cascade_yield([0.98, 0.90, 0.95]), 6)
    0.8379
    """
    if not stage_yields:
        raise ValueError("cascade_yield requires at least one stage")
    total = 1.0
    for i, y in enumerate(stage_yields):
        require_fraction(y, f"stage_yields[{i}]", lo=0.0, hi=1.0)
        total *= y
    return total


def stage_throughput_factors(stage_yields: Sequence[float]) -> list[float]:
    """Tonnes each stage must process per tonne of FINAL product.

    Element :math:`i` is :math:`1/\\prod_{j \\ge i} y_j`, i.e. including the
    stage's own yield, because a stage must feed enough to cover its own loss as
    well as everything downstream.

    Examples
    --------
    Three stages at 0.98, 0.90, 0.95. The last stage handles 1/0.95 = 1.0526 t
    per t shipped, the middle 1/(0.90 x 0.95) = 1.1696, the first
    1/(0.98 x 0.90 x 0.95) = 1.1935.

    >>> [round(x, 4) for x in stage_throughput_factors([0.98, 0.90, 0.95])]
    [1.1935, 1.1696, 1.0526]
    """
    if not stage_yields:
        raise ValueError("stage_throughput_factors requires at least one stage")
    for i, y in enumerate(stage_yields):
        require_fraction(y, f"stage_yields[{i}]", lo=0.0, hi=1.0)
        if y == 0.0:
            raise ValueError(f"stage {i} has zero yield: throughput factor is undefined")
    n = len(stage_yields)
    return [1.0 / float(np.prod(stage_yields[i:])) for i in range(n)]


@dataclass(frozen=True)
class Capability:
    """One-sided process capability with a confidence interval on Cpk."""

    cpk: float
    mean: float
    sigma: float
    n_lots: int
    ci_low: float
    ci_high: float
    usl: float

    @property
    def is_estimated_from_few_lots(self) -> bool:
        """True below 25 lots, where the interval is wide enough to mislead."""
        return self.n_lots < 25

    @property
    def off_spec_fraction(self) -> float:
        return float(stats.norm.sf(3.0 * self.cpk))


def capability(assays: Iterable[float], usl: float, confidence: float = 0.95) -> Capability:
    """One-sided Cpk from lot assays, with a confidence interval.

    The interval uses the standard large-sample variance of Cpk (Montgomery
    2012, section 8.7):

    .. math::
       \\mathrm{Var}(\\hat{C}_{pk}) \\approx
         \\frac{1}{9n}\\left(1 + \\frac{9 \\hat{C}_{pk}^2}{2}\\right)

    Reported because a bare point estimate from a handful of lots is the single
    most over-read number in a qualification package.
    """
    x = np.asarray(list(assays), dtype=float)
    if x.size < 2:
        raise ValueError(
            f"capability needs at least 2 lots to estimate a standard deviation, got {x.size}"
        )
    if np.any(x < 0):
        raise ValueError("impurity assays cannot be negative")
    if usl <= 0:
        raise ValueError(f"upper specification limit must be positive, got {usl}")
    mu = float(x.mean())
    sd = float(x.std(ddof=1))
    if sd == 0.0:
        raise ValueError(
            "all lots assay identically, so sigma is zero and Cpk is infinite. "
            "This is a measurement-resolution artefact, not a capable process."
        )
    cpk = (usl - mu) / (3.0 * sd)
    n = int(x.size)
    var = (1.0 / (9.0 * n)) * (1.0 + 9.0 * cpk**2 / 2.0)
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    half = z * float(np.sqrt(var))
    return Capability(cpk=cpk, mean=mu, sigma=sd, n_lots=n,
                      ci_low=cpk - half, ci_high=cpk + half, usl=usl)


def off_spec_fraction(
    mean: float,
    sigma: float,
    usl: float,
    model: Literal["normal", "lognormal"] = "normal",
) -> float:
    """Fraction of lots exceeding a one-sided upper limit.

    Parameters
    ----------
    mean, sigma
        Process mean and lot-to-lot standard deviation, in the same units as
        ``usl`` (typically ppm). For ``model="lognormal"`` these are the mean and
        standard deviation of the UNDERLYING quantity, not of its logarithm, and
        are converted internally.
    usl
        Upper specification limit, positive.
    model
        ``"lognormal"`` is the better default for an impurity bounded below by
        zero; the normal model understates the upper tail.

    Examples
    --------
    >>> round(off_spec_fraction(18.0, 3.0, 30.0), 6)
    3.2e-05
    >>> round(off_spec_fraction(18.0, 3.0, 30.0, model="lognormal"), 6)
    0.000765
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if mean < 0:
        raise ValueError(f"process mean cannot be negative, got {mean}")
    if usl <= 0:
        raise ValueError(f"usl must be positive, got {usl}")
    if model == "normal":
        return float(stats.norm.sf(usl, loc=mean, scale=sigma))
    # Method of moments: match the mean and variance of the lognormal.
    cv2 = (sigma / mean) ** 2
    s2 = np.log1p(cv2)
    mu_log = np.log(mean) - 0.5 * s2
    return float(stats.lognorm.sf(usl, s=np.sqrt(s2), scale=np.exp(mu_log)))


def off_spec_fraction_empirical(assays: Iterable[float], usl: float) -> tuple[float, int, int]:
    """Observed off-spec fraction with no distributional assumption.

    Returns ``(fraction, n_off, n_total)``. Preferred over the parametric forms
    as soon as real lot data exists, because it cannot be wrong about the shape
    of the tail.
    """
    x = np.asarray(list(assays), dtype=float)
    if x.size == 0:
        raise ValueError("no assays supplied")
    n_off = int(np.sum(x > usl))
    return n_off / x.size, n_off, int(x.size)


def required_process_mean(usl: float, sigma: float, cpk_target: float) -> float:
    """Process mean a target capability demands: ``usl - 3 * cpk * sigma``.

    Examples
    --------
    At a 30 ppm Al limit and Cpk 1.33, a 3 ppm sigma needs an 18.0 ppm mean and
    a 5 ppm sigma needs a 10.0 ppm mean.

    >>> round(required_process_mean(30.0, 3.0, 1.33), 2)
    18.03
    >>> round(required_process_mean(30.0, 5.0, 1.33), 2)
    10.05
    """
    if sigma <= 0 or usl <= 0 or cpk_target <= 0:
        raise ValueError("usl, sigma and cpk_target must all be positive")
    mu = usl - 3.0 * cpk_target * sigma
    if mu <= 0:
        raise ValueError(
            f"a Cpk of {cpk_target} at sigma {sigma} against a limit of {usl} requires a "
            f"process mean of {mu:.3g}, which is not physically attainable: the "
            f"variability alone consumes the whole specification. Reduce sigma."
        )
    return mu


def joint_off_spec_fraction(
    per_element: dict[str, float],
    correlation: Literal["independent", "perfect", "both"] = "both",
) -> dict[str, float]:
    """Combine per-element off-spec fractions into a lot-level fraction.

    A lot fails if ANY specified element exceeds its limit. The two bounds are

    .. math::
       f_{\\mathrm{indep}} = 1 - \\prod_e (1 - f_e), \\qquad
       f_{\\mathrm{perfect}} = \\max_e f_e

    Independence is almost certainly wrong: the same fluid-inclusion population
    carries Na, K and Li, and the same feldspar inclusions carry Al and K, so
    element excursions co-occur. Perfect correlation is the optimistic bound and
    independence the pessimistic one, and the truth sits between. Returning both
    is deliberate, so that no caller can quietly adopt the independent product
    as the answer.
    """
    if not per_element:
        raise ValueError("no per-element fractions supplied")
    for el, f in per_element.items():
        require_fraction(f, f"per_element[{el}]")
    indep = 1.0 - float(np.prod([1.0 - f for f in per_element.values()]))
    perfect = max(per_element.values())
    if correlation == "independent":
        return {"independent": indep}
    if correlation == "perfect":
        return {"perfect": perfect}
    return {"independent": indep, "perfect": perfect, "spread": indep - perfect}
