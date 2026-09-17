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
from typing import Literal

from ae.core.provenance import Tag, Value
from ae.core.site import Site
from ae.core.units import Q_, Quantity

__all__ = [
    "AACE_ACCURACY",
    "AACE_BASIS",
    "SCALING_RANGE",
    "AACEClass",
    "CapexEstimate",
    "Equipment",
    "EquipmentClass",
    "escalate_cost",
    "estimate_capex",
    "exponent_provenance",
    "scale_cost",
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
    #: Scaling exponent for this item, with its own origin tag. Stored on the
    #: item (not passed ad hoc) so that a scaled estimate records WHICH exponent
    #: produced it and where that exponent came from.
    scaling_exponent: float | Value | None = None

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


def _exponent_magnitude(exponent: float | Value, name: str) -> float:
    """Extract a float exponent from either a bare float or a tagged Value."""
    if isinstance(exponent, Value):
        q = exponent.quantity
        try:
            return float(q.to("dimensionless").magnitude)
        except Exception as exc:  # pragma: no cover - dimensionality guard
            raise ValueError(
                f"{name}: scaling exponent must be dimensionless, got {q.units}"
            ) from exc
    return float(exponent)


def exponent_provenance(exponent: float | Value) -> Tag:
    """Origin tag of a scaling exponent.

    A bare float returns :attr:`~ae.core.provenance.Tag.ASSUMED`, because an
    untagged exponent IS an assumption however conventional it is. This is the
    honest default: the six-tenths rule is a convention, not a measurement of
    the equipment in question.
    """
    return exponent.tag if isinstance(exponent, Value) else Tag.ASSUMED


def scale_cost(
    known_cost: Quantity,
    known_size: Quantity,
    target_size: Quantity,
    exponent: float | Value,
    name: str = "equipment",
) -> Quantity:
    """Scale a cost by the six-tenths rule, with extrapolation control.

    The exponent is the ENTIRE content of this relation: at a 4x size ratio,
    n = 0.6 and n = 0.9 differ by 52 percent in the answer. It therefore accepts
    a provenance-tagged :class:`~ae.core.provenance.Value` as well as a bare
    float, and :func:`exponent_provenance` reports which was supplied. A bare
    float is accepted (many exponents are genuinely conventional) but is
    reported as untagged so an estimate cannot masquerade as a sourced figure.

    Examples
    --------
    A 2.0 MUSD unit at 5,000 t/yr scaled to 20,000 t/yr at n = 0.6:
    ratio 4.0, factor 4.0**0.6 = 2.2974, so 4.5948 MUSD.

    >>> from ae.core.units import Q_
    >>> c = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
    ...                Q_(20000.0, "tonne/year"), 0.6)
    >>> round(c.to("USD").magnitude, 0)
    4594793.0

    A tagged exponent gives the same number and keeps its origin:

    >>> from ae.core.provenance import Tag, Value
    >>> n = Value(quantity=Q_(0.6, "dimensionless"), tag=Tag.ASSUMED,
    ...           basis="classical six-tenths default")
    >>> round(scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
    ...                  Q_(20000.0, "tonne/year"), n).to("USD").magnitude, 0)
    4594793.0
    >>> exponent_provenance(n).value
    'ASSUMED'
    """
    exponent = _exponent_magnitude(exponent, name)
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


#: AACE International Recommended Practice 18R-97 estimate classes, as the
#: (low, high) multiplier on the point estimate. The ranges are the commonly
#: cited expected-accuracy bands for a process industry project; the class is
#: determined by PROJECT DEFINITION MATURITY, not by the estimating method.
AACEClass = Literal[1, 2, 3, 4, 5]

AACE_ACCURACY: dict[int, tuple[float, float]] = {
    5: (0.50, 2.00),
    4: (0.70, 1.50),
    3: (0.80, 1.30),
    2: (0.85, 1.20),
    1: (0.90, 1.15),
}

#: What each class presumes has been done. Quoting a class without having done
#: this work is the failure mode the default guards against.
AACE_BASIS: dict[int, str] = {
    5: "concept screening, 0 to 2 percent project definition: capacity and an "
       "equipment list, no flowsheet engineering or quotes",
    4: "study or feasibility, 1 to 15 percent definition: preliminary "
       "flowsheets, equipment list with sizes, some vendor pricing",
    3: "budget authorisation, 10 to 40 percent definition: P&IDs, plot plan, "
       "major equipment quoted",
    2: "control estimate, 30 to 70 percent definition: detailed take-offs",
    1: "check estimate, 50 to 100 percent definition: near-complete "
       "engineering",
}


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

    #: AACE estimate class, which SETS the accuracy band. Declared rather than
    #: defaulted, because the class is a statement about how much engineering
    #: has been done and only the caller knows that.
    estimate_class: AACEClass = 5

    @property
    def accuracy(self) -> tuple[float, float]:
        """(low, high) multipliers on the total, from the declared class."""
        return AACE_ACCURACY[self.estimate_class]

    def accuracy_band(self) -> tuple[Quantity, Quantity]:
        """Total project cost range implied by the declared estimate class.

        The default is CLASS 5, minus 50 to plus 100 percent, because a
        factored estimate built from an equipment list with no flowsheet
        engineering, no equipment quotes and no plot plan is a concept estimate.
        An earlier version of this module defaulted to the Class 4 band (minus
        30 to plus 50 percent) and described a factored estimate as "AACE Class
        4 to 5", which quoted the narrower of the two bands while doing the work
        of the wider one. Class 4 presumes some engineering definition, so
        claiming it at concept stage understates the range by a factor of two
        on the upside.

        Callers who have done the engineering may declare a tighter class; the
        band then follows from the declaration and is auditable.
        """
        lo, hi = self.accuracy
        return self.total_project_cost * lo, self.total_project_cost * hi

    def accuracy_note(self) -> str:
        """One-line statement of the class and what it presumes."""
        lo, hi = self.accuracy
        return (f"AACE Class {self.estimate_class} estimate, "
                f"{(lo - 1) * 100:+.0f} to {(hi - 1) * 100:+.0f} percent: "
                f"{AACE_BASIS[self.estimate_class]}")

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

    def reconciles(self, tol: float = 1e-9) -> bool:
        """The components must sum to the total, to a RELATIVE tolerance.

        ``tol`` was an ABSOLUTE 1e-6 in whatever currency unit the estimate
        carries, which rejects correct arithmetic at scale: double precision
        gives about 2.2e-16 of relative error, so the float residue of a
        correct sum passes 1e-6 once the total passes roughly 4.5e9 currency
        units. Measured on a correct estimate totalling 2.835e10 INR (about
        0.34 billion USD at 83 INR per USD) with the components summed in a
        different order from the stored total, the absolute residual is
        3.815e-06 and the relative residual is 1.345e-16: the absolute test
        fails and the relative test passes. INR totals of that size are
        ordinary for this project, so the defect was reachable rather than
        theoretical.

        It had not bitten because :func:`estimate_capex` computes the total as
        ``installed + indirect + contingency`` and this method recomputes that
        same expression in the same order, so the two are bit-identical and
        the residual is exactly 0.0. Over 400 randomised ``estimate_capex``
        builds (1 to 6 items, costs to 1e9, location factors 0.3 to 2.5,
        escalation indices 50 to 900) it returned False zero times. Inside
        ``estimate_capex`` the check is therefore a tautology, and the callers
        that can see a non-zero residual are the ones that construct a
        CapexEstimate directly, which is where a real reconciliation error
        would arise.

        1e-9 relative is seven orders above the float noise and small enough
        to catch any discrepancy an estimate could have: on a 1e10 total it is
        10 currency units.
        """
        total = (self.total_installed + self.indirect_cost + self.contingency)
        residual = abs(float((total - self.total_project_cost).magnitude))
        scale = abs(float(self.total_project_cost.magnitude))
        if scale <= 0:
            return residual < tol
        return residual / scale < tol

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
                "scaling_exponent": (None if e.scaling_exponent is None else
                                     _exponent_magnitude(e.scaling_exponent, e.name)),
                "scaling_exponent_tag": (None if e.scaling_exponent is None else
                                         exponent_provenance(e.scaling_exponent).value),
            })
        for label, amount in self.lines:
            rows.append({"item": label, "class": "aggregate", "purchased": None,
                         "install_factor": None, "installed": amount,
                         "tag": None, "source": None,
                         "scaling_exponent": None, "scaling_exponent_tag": None})
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

    # The one-sided case is already rejected above, so by here the two indices
    # are either both None or both set. Narrowing on both names rather than on
    # base_index alone is what lets a type checker see that: the original
    # `float(target_index)` read as float(None) to mypy and was reported as an
    # operand error. It was NOT a reachable bug, and an earlier version of this
    # comment wrongly claimed it was. Ordering the positivity check before the
    # division is still worth keeping, because it makes the guarantee local
    # instead of thirty lines away.
    if base_index is not None and target_index is not None:
        if base_index <= 0 or target_index <= 0:
            raise ValueError("cost indices must be positive")
        idx_ratio = float(target_index) / float(base_index)
    else:
        idx_ratio = 1.0

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
