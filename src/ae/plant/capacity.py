r"""Capacity, OEE and bottleneck identification.

Governing equations
-------------------
**Overall Equipment Effectiveness** (Nakajima's decomposition, the form used by
SEMI E79 for semiconductor equipment):

.. math::
   \mathrm{OEE} = A \cdot P \cdot Q

with

.. math::
   A = \frac{t_{\mathrm{run}}}{t_{\mathrm{planned}}}, \qquad
   P = \frac{r_{\mathrm{actual}}}{r_{\mathrm{nameplate}}}, \qquad
   Q = \frac{n_{\mathrm{good}}}{n_{\mathrm{total}}}

where :math:`A` is availability (dimensionless, 0 to 1), :math:`P` performance
efficiency (0 to 1), :math:`Q` quality yield (0 to 1), :math:`t_{\mathrm{run}}`
operating time (h), :math:`t_{\mathrm{planned}}` planned production time (h),
:math:`r` throughput rate (t/h) and :math:`n` unit counts.

**Effective throughput** of a unit:

.. math::
   \dot m_{\mathrm{eff}} = r_{\mathrm{nameplate}} \cdot t_{\mathrm{planned}}
                           \cdot \mathrm{OEE}

**Bottleneck.** With unit :math:`i` requiring :math:`\tau_i` hours of its own
capacity per tonne of FINAL product, the line rate is set by

.. math::
   \dot m_{\mathrm{line}} = \min_i \frac{C_i}{\tau_i}

where :math:`C_i` is unit :math:`i`'s effective capacity. The per-tonne-of-final
-product basis is the part that is easy to get wrong: a unit early in the
flowsheet must process more tonnes than the plant ships, by the reciprocal of
the cumulative yield downstream of it, so comparing nameplate rates directly
misidentifies the bottleneck whenever stage yields differ.

Sources
-------
OEE decomposition: Nakajima, *Introduction to TPM*, Productivity Press 1988.
The semiconductor-industry formalization with explicit state definitions is SEMI
E79 (Specification for Definition and Measurement of Equipment Productivity).
Neither is reproduced here; only the algebra is, which is standard.

LIMITATIONS
-----------
* OEE is a steady-state average. It cannot represent a plant whose availability
  is correlated with throughput (a furnace pushed harder failing more often), and
  it says nothing about queueing. Batch interaction and surge behaviour need the
  discrete-event model in :mod:`ae.plant.scheduling`.
* Quality yield :math:`Q` here is the fraction of output meeting spec at that
  unit. It is NOT the same as the compounded plant yield, and multiplying stage
  OEEs to get a plant OEE is a category error: see :mod:`ae.plant.yield_cascade`.
* No learning curve. Availability and performance are held constant over the
  asset life, which understates early-life losses and overstates them later.
  A first-plant ramp is modelled explicitly in the econ layer instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = ["OEE", "UnitCapacity", "LineCapacity", "assess_line"]

#: Hours in a calendar year, used when planned time is quoted as a utilisation.
HOURS_PER_YEAR = 8760.0


@dataclass(frozen=True)
class OEE:
    """Availability, performance and quality, and their product.

    Each factor is a bare fraction with an enforced range, following the
    convention in :mod:`ae.core.units` that genuinely dimensionless ratios are
    range-checked rather than unit-wrapped.
    """

    availability: float
    performance: float
    quality: float

    def __post_init__(self) -> None:
        require_fraction(self.availability, "availability")
        require_fraction(self.performance, "performance")
        require_fraction(self.quality, "quality")

    @property
    def value(self) -> float:
        return self.availability * self.performance * self.quality

    @property
    def loss_breakdown(self) -> dict[str, float]:
        """Fraction of nameplate capacity lost to each cause.

        The three losses are reported as exclusive contributions in cascade
        order (availability first, then performance on what remains, then
        quality), so they sum to exactly ``1 - OEE``. Reporting them as
        ``1 - A``, ``1 - P``, ``1 - Q`` instead would double-count, because each
        factor acts on the output of the previous one.
        """
        a_loss = 1.0 - self.availability
        p_loss = self.availability * (1.0 - self.performance)
        q_loss = self.availability * self.performance * (1.0 - self.quality)
        return {"availability": a_loss, "performance": p_loss, "quality": q_loss}

    @classmethod
    def from_times(
        cls,
        planned_hours: float,
        downtime_hours: float,
        nameplate_rate: Quantity,
        actual_output: Quantity,
        good_output: Quantity,
    ) -> "OEE":
        """Build OEE from measured times and tonnages.

        Parameters
        ----------
        planned_hours
            Scheduled production time, excluding planned shutdowns.
        downtime_hours
            Unplanned stoppage within the planned window.
        nameplate_rate
            Design rate, dimensionality [mass]/[time].
        actual_output
            Total mass produced in the window, good and bad.
        good_output
            Mass meeting specification.
        """
        require_dimensionality(nameplate_rate, "mass_flow", "nameplate_rate")
        require_dimensionality(actual_output, "mass", "actual_output")
        require_dimensionality(good_output, "mass", "good_output")
        if planned_hours <= 0:
            raise ValueError("planned_hours must be positive")
        if downtime_hours < 0 or downtime_hours > planned_hours:
            raise ValueError(
                f"downtime_hours ({downtime_hours}) must lie in [0, planned_hours]"
            )
        run_hours = planned_hours - downtime_hours
        a = run_hours / planned_hours
        cap = (nameplate_rate * Q_(run_hours, "hour")).to("tonne")
        act = actual_output.to("tonne")
        good = good_output.to("tonne")
        if act.magnitude > cap.magnitude * (1 + 1e-9):
            raise ValueError(
                f"actual output {act:~P} exceeds the nameplate capacity of the run time "
                f"({cap:~P}); either the rate is understated or the times are wrong"
            )
        if good.magnitude > act.magnitude * (1 + 1e-9):
            raise ValueError(f"good output {good:~P} exceeds total output {act:~P}")
        p = (act / cap).magnitude if cap.magnitude > 0 else 0.0
        q = (good / act).magnitude if act.magnitude > 0 else 0.0
        return cls(availability=a, performance=p, quality=q)


@dataclass(frozen=True)
class UnitCapacity:
    """One unit's nameplate rate, schedule and effectiveness.

    Parameters
    ----------
    name
        Unit label, matching the flowsheet unit where applicable.
    nameplate_rate
        Design throughput, dimensionality [mass]/[time].
    planned_hours
        Planned production hours per year.
    oee
        Effectiveness factors.
    tonnes_per_tonne_product
        Tonnes this unit must process per tonne of FINAL product, i.e. the
        reciprocal of the cumulative yield downstream. Defaults to 1.0, which is
        correct only for the last unit in the line.
    """

    name: str
    nameplate_rate: Quantity
    planned_hours: float
    oee: OEE
    tonnes_per_tonne_product: float = 1.0

    def __post_init__(self) -> None:
        require_dimensionality(self.nameplate_rate, "mass_flow", f"{self.name} nameplate_rate")
        if self.planned_hours <= 0 or self.planned_hours > HOURS_PER_YEAR:
            raise ValueError(
                f"{self.name}: planned_hours must lie in (0, {HOURS_PER_YEAR}], "
                f"got {self.planned_hours}"
            )
        if self.tonnes_per_tonne_product < 1.0:
            raise ValueError(
                f"{self.name}: tonnes_per_tonne_product is {self.tonnes_per_tonne_product}, "
                f"below 1.0. A unit cannot process less than it ships downstream unless "
                f"mass is created; check the yield cascade."
            )

    @property
    def effective_capacity(self) -> Quantity:
        """Own-throughput capacity per year, after OEE."""
        return (self.nameplate_rate * Q_(self.planned_hours, "hour") * self.oee.value).to("tonne")

    @property
    def product_capacity(self) -> Quantity:
        """Capacity expressed as tonnes of FINAL product per year."""
        return self.effective_capacity / self.tonnes_per_tonne_product


@dataclass(frozen=True)
class LineCapacity:
    """Result of a line assessment."""

    line_rate: Quantity
    bottleneck: str
    utilisation: dict[str, float]
    product_capacity: dict[str, Quantity]
    slack: dict[str, Quantity]

    @property
    def bottleneck_margin(self) -> float:
        """Fractional gap between the bottleneck and the next tightest unit.

        A small margin means the bottleneck is not robust: a modest improvement
        moves it elsewhere, so debottlenecking one unit buys little. Reported so
        that a capital plan is not built on a bottleneck that shifts.

        Exactly 0.0 means two or more units bind simultaneously, in which case
        the ``bottleneck`` name is arbitrary (first in declaration order) and
        both must be expanded together to gain any capacity. Infinite for a
        single-unit line.
        """
        caps = sorted(q.to("tonne").magnitude for q in self.product_capacity.values())
        if len(caps) < 2 or caps[0] <= 0:
            return float("inf")
        return (caps[1] - caps[0]) / caps[0]


def assess_line(units: Iterable[UnitCapacity]) -> LineCapacity:
    """Find the bottleneck and the line rate in tonnes of final product per year.

    Examples
    --------
    Two units. The FIRST (the mill) sits upstream of yield loss and so must
    process 1.25 t for every 1 t the plant ships; the second (the leach) is the
    last stage and processes 1.0 t/t. On nameplate rate alone the mill looks
    larger, 10 t/h against 9, but on a final-product basis it is the constraint:

      mill : 10 x 7000 x 0.84645 / 1.25 = 47401.20 t/yr of product
      leach:  9 x 7000 x 0.84645 / 1.00 = 53326.35 t/yr of product

    >>> from ae.core.units import Q_
    >>> o = OEE(0.90, 0.95, 0.99)
    >>> a = UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, o, 1.25)
    >>> b = UnitCapacity("leach", Q_(9.0, "tonne/hour"), 7000.0, o, 1.0)
    >>> r = assess_line([a, b])
    >>> r.bottleneck
    'mill'
    >>> round(r.line_rate.to("tonne").magnitude, 2)
    47401.2
    >>> round(r.bottleneck_margin, 4)
    0.125

    A 10 t/h mill at 1.25 t/t and an 8 t/h leach at 1.0 t/t would instead give
    IDENTICAL product capacities. That is a tie, not a finding: the name
    returned is then merely first-in-declaration-order, and
    ``bottleneck_margin`` is 0.0 to say so.
    """
    units = list(units)
    if not units:
        raise ValueError("assess_line requires at least one unit")
    names = [u.name for u in units]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate unit names: {names}")
    prod_cap = {u.name: u.product_capacity for u in units}
    # On an exact tie min() returns the first in declaration order, which is
    # deterministic but arbitrary. The tie is not hidden: bottleneck_margin is
    # then 0.0, which is the signal that naming a single bottleneck is
    # meaningless because two units bind simultaneously.
    bottleneck = min(prod_cap, key=lambda k: prod_cap[k].to("tonne").magnitude)
    line = prod_cap[bottleneck]
    line_t = line.to("tonne").magnitude
    util = {
        u.name: (line_t * u.tonnes_per_tonne_product)
        / u.effective_capacity.to("tonne").magnitude
        for u in units
    }
    slack = {u.name: (prod_cap[u.name] - line).to("tonne") for u in units}
    return LineCapacity(
        line_rate=line,
        bottleneck=bottleneck,
        utilisation=util,
        product_capacity=prod_cap,
        slack=slack,
    )
