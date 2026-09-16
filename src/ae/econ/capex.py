r"""Capital cost from an equipment list, with scaling, installation and location.

Built bottom-up from named equipment rather than top-down from a capacity
ratio, because a top-down estimate cannot say what would have to change to move
the number, and cannot be audited by a reader who knows the equipment.

Governing relations
-------------------
**Six-tenths rule** for scaling a known equipment cost to a new size:

.. math::
   C_2 = C_1 \left(\frac{S_2}{S_1}\right)^{n}

where :math:`C` is purchased cost (currency), :math:`S` is a size attribute
(capacity, area, volume) and :math:`n` is the scaling exponent (dimensionless,
typically 0.4 to 0.9, with 0.6 the classical default). The exponent is the whole
content of the relation, so it is a required field with its own provenance
rather than a hidden 0.6.

**Lang / Guthrie factored estimate** for installed plant cost:

.. math::
   \mathrm{TIC} = \sum_j C_{p,j} f_{\mathrm{install},j}, \qquad
   \mathrm{TPC} = \mathrm{TIC}\,(1 + f_{\mathrm{indirect}})

with :math:`C_p` purchased equipment cost, :math:`f_{\mathrm{install}}` the
installation factor for that equipment class (piping, electrical, instruments,
structural, erection), and :math:`f_{\mathrm{indirect}}` covering engineering,
construction management, spares and commissioning.

**Escalation and location**:

.. math::
   C_{\mathrm{target}} = C_{\mathrm{base}}
     \cdot \frac{I_{\mathrm{target}}}{I_{\mathrm{base}}}
     \cdot f_{\mathrm{loc}}

where :math:`I` is a cost index (CEPCI or equivalent) and
:math:`f_{\mathrm{loc}}` a location factor. Two separate corrections that are
frequently collapsed into one: escalation moves a cost through TIME at a fixed
location, the location factor moves it through SPACE at a fixed time. Applying
one when you mean the other is a silent error of tens of percent.

Currency is a third, independent axis. A location factor must NOT absorb an
exchange-rate move, which is why :class:`ae.core.site.Site` holds
``construction_cost_index`` separately from FX.

Sources
-------
The six-tenths rule and the factored-estimate method are standard: Peters,
Timmerhaus and West, *Plant Design and Economics for Chemical Engineers*, 5th
ed., McGraw-Hill 2003, chapters 6 and 12; Towler and Sinnott, *Chemical
Engineering Design*, 3rd ed., Butterworth-Heinemann 2021, chapter 6. Neither
the specific Lang factors nor any CEPCI value is hardcoded here: those are
inputs with provenance, because they are published series that change annually
and reproducing a remembered value is exactly the failure mode this platform
forbids.

LIMITATIONS
-----------
* A factored estimate is AACE Class 4 to 5, nominal accuracy minus 30 to plus
  50 percent. :meth:`CapexEstimate.accuracy_band` returns that band explicitly,
  and no result from this module should be quoted without it. A bottom-up
  estimate built from vendor quotes is a different exercise.
* Scaling exponents outside roughly 0.4 to 0.9 usually mean the equipment class
  has changed (a single vessel becoming a train of vessels), where the relation
  does not hold. Values outside that range are accepted but flagged.
* Extrapolating a size ratio beyond about 10x is unsupported by the underlying
  correlations; :func:`scale_cost` warns above 10x and raises above 100x rather
  than returning a confident number.
* No contingency is added automatically. Contingency is a decision, not a
  calculation, so it is an explicit input and is reported as its own line.
* Greenfield only. Brownfield tie-ins, demolition and site remediation are not
  represented and are frequently the difference between estimate and outcome.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from ae.core.provenance import Tag, Value
from ae.core.site import Site
from ae.core.units import Q_, Quantity

__all__ = [
    "Equipment",
    "CapexEstimate",
    "scale_cost",
    "escalate_cost",
    "estimate_capex",
    "EquipmentClass",
    "SCALING_RANGE",
]

#: Range within which the six-tenths family of correlations is supported.
SCALING_RANGE = (0.4, 0.9)

#: Size-ratio thresholds for extrapolation control.
_WARN_RATIO = 10.0
_FAIL_RATIO = 100.0

EquipmentClass = str


@dataclass(frozen=True)
class Equipment:
    """One equipment item with its purchased cost and installation factor.

    Parameters
    ----------
    name
        Item label, e.g. "attrition scrubber".
    purchased_cost
        Free-on-board equipment cost, as a provenance-tagged Value.
    installation_factor
        Multiplier from purchased to installed cost for this item, covering
        piping, electrical, instrumentation, structural and erection. Typically
        1.4 for a simple tank through 3.5 for instrumented process equipment.
    size, size_basis
        The size attribute the cost corresponds to, used when scaling. Optional,
        but scaling without it is refused.
    reference_index
        Cost index value at the date the purchased cost applies to, so the item
        can be escalated. Optional.
    """

    name: str
    purchased_cost: Value
    installation_factor: float
    size: Quantity | None = None
    size_basis: str | None = None
    reference_index: float | None = None
    equipment_class: EquipmentClass = "process"

    def __post_init__(self) -> None:
        if self.installation_factor < 1.0:
            raise ValueError(
                f"{self.name}: installation_factor is {self.installation_factor}, below 1.0. "
                f"Installed cost cannot be less than purchased cost; a factor below 1 "
                f"usually means a percentage was passed instead of a multiplier."
            )
        if self.installation_factor > 6.0:
            raise ValueError(
                f"{self.name}: installation_factor {self.installation_factor} exceeds 6.0, "
                f"outside any published Lang-factor range. Check whether indirects are "
                f"being double-counted here and again at the plant level."
            )
        if self.purchased_cost.quantity.magnitude < 0:
            raise ValueError(f"{self.name}: purchased cost cannot be negative")
        if self.reference_index is not None and self.reference_index <= 0:
            raise ValueError(f"{self.name}: reference_index must be positive")

    @property
    def installed_cost(self) -> Quantity:
        return self.purchased_cost.quantity * self.installation_factor


def scale_cost(
    known_cost: Quantity,
    known_size: Quantity,
    target_size: Quantity,
    exponent: float,
    name: str = "equipment",
) -> Quantity:
    """Scale a cost by the six-tenths rule, with extrapolation control.

    Examples
    --------
    A 2.0 MUSD unit at 5,000 t/yr scaled to 20,000 t/yr at n = 0.6:
    ratio 4.0, factor 4.0**0.6 = 2.2974, so 4.5948 MUSD.

    >>> from ae.core.units import Q_
    >>> c = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
    ...                Q_(20000.0, "tonne/year"), 0.6)
    >>> round(c.to("USD").magnitude, 0)
    4594793.0
    """
    if known_cost.magnitude <= 0:
        raise ValueError(f"{name}: known cost must be positive")
    if known_size.magnitude <= 0 or target_size.magnitude <= 0:
        raise ValueError(f"{name}: sizes must be positive")
    # Sizes must be commensurable, or the ratio is meaningless.
    ratio_q = (target_size / known_size).to("dimensionless")
    ratio = float(ratio_q.magnitude)
    if not SCALING_RANGE[0] <= exponent <= SCALING_RANGE[1]:
        warnings.warn(
            f"{name}: scaling exponent {exponent} lies outside the supported range "
            f"{SCALING_RANGE}. Outside it, the equipment class has usually changed "
            f"(a single vessel becoming a train), where the correlation does not hold.",
            UserWarning, stacklevel=2,
        )
    span = max(ratio, 1.0 / ratio)
    if span > _FAIL_RATIO:
        raise ValueError(
            f"{name}: size ratio {ratio:.4g} extrapolates the correlation more than "
            f"{_FAIL_RATIO:g}x, which the underlying data does not support. Use a "
            f"quote or a different reference item rather than a confident number."
        )
    if span > _WARN_RATIO:
        warnings.warn(
            f"{name}: size ratio {ratio:.4g} exceeds {_WARN_RATIO:g}x; the six-tenths "
            f"family of correlations is not supported this far from the reference.",
            UserWarning, stacklevel=2,
        )
    return known_cost * (ratio ** exponent)


def escalate_cost(
    cost: Quantity, base_index: float, target_index: float, name: str = "cost"
) -> Quantity:
    """Move a cost through TIME using a cost index ratio.

    This is not a location factor and not an exchange rate. Escalation moves a
    cost through time at a fixed location; a location factor moves it through
    space at a fixed time; FX changes the unit of account. Collapsing any two of
    the three is a silent error of tens of percent.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> round(escalate_cost(Q_(1.0e6, "USD"), 708.0, 800.0).to("USD").magnitude, 2)
    1129943.5
    """
    if base_index <= 0 or target_index <= 0:
        raise ValueError(f"{name}: cost indices must be positive")
    return cost * (target_index / base_index)


@dataclass
class CapexEstimate:
    """A factored capital estimate with its components and accuracy band."""

    equipment: list[Equipment]
    total_purchased: Quantity
    total_installed: Quantity
    indirect_cost: Quantity
    contingency: Quantity
    total_project_cost: Quantity
    location_factor: float
    index_ratio: float
    currency: str
    lines: list[tuple[str, float]] = field(default_factory=list)

    #: AACE Class 4-5 factored-estimate accuracy, as (low, high) multipliers.
    accuracy: tuple[float, float] = (0.70, 1.50)

    def accuracy_band(self) -> tuple[Quantity, Quantity]:
        """Total project cost range implied by the estimate class.

        A factored estimate is AACE Class 4 to 5, nominal minus 30 to plus 50
        percent. Returned so no caller can quote the point estimate alone.
        """
        lo, hi = self.accuracy
        return self.total_project_cost * lo, self.total_project_cost * hi

    @property
    def capex_per_annual_tonne(self) -> Quantity | None:
        return None if self._capacity is None else (
            self.total_project_cost / self._capacity)

    _capacity: Quantity | None = None

    @property
    def assumed_share(self) -> float:
        """Fraction of purchased equipment cost carried by ASSUMED values.

        Both numerator and denominator are taken on the RAW (un-escalated,
        un-located) basis. An earlier version divided raw ASSUMED costs by
        ``total_purchased``, which carries the index ratio and location factor,
        so at a location factor of 0.55 an all-assumed equipment list reported a
        share of 1/0.55 = 1.82. A provenance guard that can read 182 percent is
        worse than no guard, since it invites the reader to dismiss it. The
        scaling factors are common to every item and therefore cancel, so the
        raw basis is the correct one.
        """
        raw_total = float(sum(e.purchased_cost.quantity.magnitude
                              for e in self.equipment))
        if raw_total <= 0:
            return 0.0
        assumed = sum(e.purchased_cost.quantity.magnitude for e in self.equipment
                      if e.purchased_cost.tag == Tag.ASSUMED)
        return float(assumed) / raw_total

    def reconciles(self, tol: float = 1e-6) -> bool:
        """The components must sum to the total."""
        total = (self.total_installed + self.indirect_cost + self.contingency)
        return abs(float((total - self.total_project_cost).magnitude)) < tol

    def to_records(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for e in self.equipment:
            rows.append({
                "item": e.name,
                "class": e.equipment_class,
                "purchased": float(e.purchased_cost.quantity.magnitude),
                "install_factor": e.installation_factor,
                "installed": float(e.installed_cost.magnitude),
                "tag": e.purchased_cost.tag.value,
                "source": (e.purchased_cost.source.citation
                           if e.purchased_cost.source else None),
            })
        for label, amount in self.lines:
            rows.append({"item": label, "class": "aggregate", "purchased": None,
                         "install_factor": None, "installed": amount,
                         "tag": None, "source": None})
        return rows


def estimate_capex(
    equipment: list[Equipment],
    site: Site,
    indirect_factor: float,
    contingency_fraction: float,
    base_index: float | None = None,
    target_index: float | None = None,
    capacity: Quantity | None = None,
) -> CapexEstimate:
    """Factored capital estimate for an equipment list at a specific site.

    Parameters
    ----------
    equipment
        Items with purchased costs and installation factors.
    site
        Supplies ``construction_cost_index`` as the location factor and the
        currency. The location factor is held separately from FX on the Site
        precisely so a currency move cannot be mistaken for a construction
        saving. Its default of 1.0 means "reference location"; the applied value
        is reported as ``CapexEstimate.location_factor`` so an unadjusted
        estimate is visible rather than implicit.
    indirect_factor
        Engineering, construction management, spares and commissioning, as a
        fraction of total installed cost. Typically 0.20 to 0.40.
    contingency_fraction
        Explicit contingency on installed plus indirect cost. Contingency is a
        decision, not a calculation, so there is no default.
    base_index, target_index
        Cost-index values for escalation through time. Both or neither.
    capacity
        Annual output, enabling ``capex_per_annual_tonne``.
    """
    if not equipment:
        raise ValueError("an equipment list is required; a capex estimate needs items")
    names = [e.name for e in equipment]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate equipment names: {names}")
    if not 0.0 <= indirect_factor <= 1.0:
        raise ValueError(
            f"indirect_factor {indirect_factor} outside [0, 1]; typical values are "
            f"0.20 to 0.40 of total installed cost"
        )
    if not 0.0 <= contingency_fraction <= 1.0:
        raise ValueError(f"contingency_fraction {contingency_fraction} outside [0, 1]")
    if (base_index is None) != (target_index is None):
        raise ValueError(
            "escalation needs BOTH base_index and target_index, or neither: a one-sided "
            "index is not an escalation"
        )

    cur = site.currency.value
    for e in equipment:
        if cur not in str(e.purchased_cost.quantity.units):
            raise ValueError(
                f"equipment {e.name!r} is priced in "
                f"{e.purchased_cost.quantity.units}, but site {site.site_id} operates "
                f"in {cur}. Convert explicitly through a dated ExchangeRate."
            )

    # Site.construction_cost_index is a validated positive float defaulting to
    # 1.0, so it cannot be absent or non-positive here. A default of 1.0 means
    # "reference location", which is a real assumption rather than a neutral
    # one: it is surfaced on the estimate as location_factor so a reader can see
    # whether a location correction was actually applied.
    loc_f = float(site.construction_cost_index)

    idx_ratio = 1.0 if base_index is None else float(target_index) / float(base_index)
    if base_index is not None and (base_index <= 0 or target_index <= 0):
        raise ValueError("cost indices must be positive")

    zero = Q_(0.0, cur)
    purchased = sum((e.purchased_cost.quantity for e in equipment), zero)
    installed_raw = sum((e.installed_cost for e in equipment), zero)
    # Escalation (time) and location (space) are applied as separate, named
    # factors, never merged into one number.
    installed = installed_raw * idx_ratio * loc_f
    indirect = installed * indirect_factor
    contingency = (installed + indirect) * contingency_fraction
    total = installed + indirect + contingency

    est = CapexEstimate(
        equipment=equipment,
        total_purchased=purchased * idx_ratio * loc_f,
        total_installed=installed,
        indirect_cost=indirect,
        contingency=contingency,
        total_project_cost=total,
        location_factor=loc_f,
        index_ratio=idx_ratio,
        currency=cur,
        lines=[("total installed", float(installed.magnitude)),
               ("indirects", float(indirect.magnitude)),
               ("contingency", float(contingency.magnitude)),
               ("total project cost", float(total.magnitude))],
        _capacity=capacity,
    )
    if not est.reconciles():
        raise AssertionError("capex components do not sum to the total project cost")
    return est
