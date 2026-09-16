r"""Process streams and flowsheet mass balance with recycle solved as a linear system.

A flowsheet is a directed graph of unit operations connected by streams. Each
stream carries a solids mass flow and a per-element impurity composition. Units
split and transform those flows. With recycle loops present the flows cannot be
evaluated by a single forward pass, so the balance is solved simultaneously.

Governing equations
-------------------
**Overall solids balance at unit** :math:`u`, with feeds :math:`F_{i}` and
products :math:`P_{j}`:

.. math::
   \sum_{i \in \mathrm{in}(u)} \dot m_i = \sum_{j \in \mathrm{out}(u)} \dot m_j

where :math:`\dot m` is solids mass flow (kg/s, range 0 to 1e4).

**Element balance** for element :math:`e`, with :math:`c_{i,e}` the mass ratio of
:math:`e` in stream :math:`i` (kg/kg, range 0 to 1):

.. math::
   \sum_{i \in \mathrm{in}(u)} \dot m_i c_{i,e}
   = \sum_{j \in \mathrm{out}(u)} \dot m_j c_{j,e} + \dot r_{u,e}

where :math:`\dot r_{u,e}` is the rate of removal of :math:`e` to a reject or
effluent stream (kg/s). Removal is not annihilation: every unit's rejected mass
is tracked in an explicit reject stream, so the plant-wide balance closes.

**Recycle solution, as actually implemented.** For a fixed split specification
the unit operations are linear in flow, so the recycle problem admits a direct
matrix solution :math:`(\mathbf{I} - \mathbf{A})\mathbf{x} = \mathbf{b}` with
:math:`\mathbf{A}` the routing matrix of split fractions. **This module does not
use that formulation.** :meth:`Flowsheet.solve` dispatches on topology:

* **Acyclic flowsheet** (no link routes a stream to an earlier unit): ONE forward
  pass in declaration order, no iteration and no damping, reported as
  ``method="single_pass"`` with ``iterations=1``. The pass is exact, because with
  no recycle every unit's feed is fully determined before it is evaluated.
* **Flowsheet with recycle**: DAMPED SUCCESSIVE SUBSTITUTION on the recycled
  streams, reported as ``method="successive_substitution"`` with the iteration
  count taken.

The recycle iteration is

.. math::
   \mathbf{x}^{(k+1)} = (1 - \alpha)\,\mathbf{x}^{(k)}
                        + \alpha\, \mathbf{G}\!\left(\mathbf{x}^{(k)}\right)

where :math:`\mathbf{x}` collects the recycled stream flows, :math:`\mathbf{G}`
is one forward pass through the flowsheet and :math:`\alpha` is the damping
factor (``damping``, default 0.5, range 0 to 1). Iteration stops when the
largest change in a recycled stream's mass flow falls below ``tol``.

Successive substitution was chosen because it extends unchanged to the nonlinear
case (a unit whose selectivity depends on its own feed grade, which is true of
flotation and of leaching at low acid-to-solids ratio), where no constant
:math:`\mathbf{A}` exists. The cost is that convergence is not guaranteed: the
loop gain plays the role the spectral radius of :math:`\mathbf{A}` would, and a
gain at or above unity describes a loop that gains mass. That case is detected
empirically, by the iteration-to-iteration change RISING for more than five
iterations, and raises rather than being iterated to a nonsense answer. A
non-converged run within ``max_iter`` is reported as ``converged=False``, never
dressed as a solution.

A direct linear solve would be faster and would give an exact convergence
criterion for the linear case. It is not implemented here; see LIMITATIONS.

Sources
-------
The linear recycle formulation is textbook: Himmelblau and Riggs, *Basic
Principles and Calculations in Chemical Engineering*, 8th ed., Prentice Hall
2012, chapter on recycle, bypass and purge. The two-product formula used by
:mod:`ae.physics.separation` is the mineral-processing special case of the same
element balance (Wills and Finch, *Wills' Mineral Processing Technology*, 8th
ed., Butterworth-Heinemann 2015, chapter 3).

LIMITATIONS
-----------
* Steady state only. Batch scheduling, surge capacity and startup transients are
  handled by the discrete-event model in :mod:`ae.plant.scheduling`, not here.
* No direct linear solve. Every case goes through damped successive substitution,
  including the linear ones where a matrix solve would be faster and would supply
  an exact convergence criterion from the spectral radius of the routing matrix.
  Consequence: convergence is diagnosed empirically, so a loop with gain very
  close to unity may exhaust ``max_iter`` and report ``converged=False`` on a
  problem a direct solve would settle exactly. Raising ``max_iter`` is not the
  fix; check the routing.
* Nonlinear units (``grade_dependent`` set) are handled by the same iteration,
  which is why it was chosen, but for those no convergence guarantee exists at
  all, and a converged result is a fixed point rather than a proven unique
  solution.
* Liquid phase is not balanced here. Acid and water balances live in
  :mod:`ae.physics.reagents`, because their stoichiometry couples to impurity
  load rather than to solids flow.
* No energy balance. See :mod:`ae.physics.thermal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ae.core.units import Q_, Quantity, require_dimensionality

__all__ = [
    "Stream",
    "UnitOp",
    "Flowsheet",
    "BalanceResult",
    "MassBalanceError",
]

#: Relative closure tolerance on the solids balance, as a fraction of feed flow.
#: Used by :meth:`Flowsheet._result`. Set at 1e-6 rather than machine epsilon
#: because the recycle iteration terminates on a flow-change tolerance, so a
#: converged recycle solution carries a residual of that order; the single-pass
#: linear case closes near 1e-12 and is checked against the same bound.
CLOSURE_TOL = 1e-6


class MassBalanceError(RuntimeError):
    """Raised when a balance cannot be closed or a specification is unphysical."""


@dataclass(frozen=True)
class Stream:
    """A solids stream with per-element composition.

    Parameters
    ----------
    name
        Unique label.
    mass_flow
        Solids mass flow, a Quantity with dimensionality [mass]/[time].
    composition
        Element symbol to mass ratio (kg/kg). Absent elements are treated as
        not-tracked, NOT as zero: a composition that omits Ti cannot be used to
        claim Ti is absent, and :meth:`element_flow` raises for an untracked
        element rather than returning zero.
    """

    name: str
    mass_flow: Quantity
    composition: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_dimensionality(self.mass_flow, "mass_flow", f"stream {self.name!r} mass_flow")
        if self.mass_flow.magnitude < 0:
            raise MassBalanceError(f"stream {self.name!r} has negative flow")
        for el, c in self.composition.items():
            if not 0.0 <= c <= 1.0:
                raise MassBalanceError(
                    f"stream {self.name!r} composition[{el}] = {c} is not a mass ratio in [0, 1]"
                )
        total = sum(self.composition.values())
        if total > 1.0 + 1e-12:
            raise MassBalanceError(
                f"stream {self.name!r} impurity mass ratios sum to {total:.6g} > 1"
            )

    @property
    def kg_s(self) -> float:
        return float(self.mass_flow.to("kg/s").magnitude)

    def element_flow(self, element: str) -> Quantity:
        """Mass flow of one element, raising if that element is not tracked."""
        if element not in self.composition:
            raise KeyError(
                f"{element} is not tracked in stream {self.name!r} "
                f"(tracked: {sorted(self.composition)}). An absent element is "
                f"untracked, not zero."
            )
        return self.mass_flow * self.composition[element]

    def ppm(self, element: str) -> float:
        return self.composition[element] * 1e6 if element in self.composition else float("nan")


@dataclass
class UnitOp:
    """One unit operation: splits its feed into a product and a reject.

    Parameters
    ----------
    name
        Unique label.
    mass_yield
        Fraction of feed solids reporting to the product stream (0 to 1).
    element_removal
        Per element, the fraction of that element in the feed that reports to the
        REJECT stream. A removal of 0.9 for Fe means 90 percent of the iron
        leaves with the reject. Elements absent from this mapping are assumed to
        follow the solids split with no preferential rejection, which is the
        physically correct default for an element that the unit does not act on.
    grade_dependent
        Callable ``(element, feed_ppm) -> removal_fraction`` used instead of the
        fixed ``element_removal`` when the unit's selectivity depends on feed
        grade. Its presence makes the flowsheet nonlinear.
    """

    name: str
    mass_yield: float
    element_removal: dict[str, float] = field(default_factory=dict)
    grade_dependent: Callable[[str, float], float] | None = None
    kind: Literal["physical", "chemical", "thermal", "classification"] = "physical"

    def __post_init__(self) -> None:
        if not 0.0 < self.mass_yield <= 1.0:
            raise MassBalanceError(
                f"unit {self.name!r} mass_yield must lie in (0, 1], got {self.mass_yield}"
            )
        for el, r in self.element_removal.items():
            if not 0.0 <= r <= 1.0:
                raise MassBalanceError(
                    f"unit {self.name!r} element_removal[{el}] = {r} is not a fraction"
                )

    def apply(self, feed: Stream) -> tuple[Stream, Stream]:
        """Split ``feed`` into (product, reject), closing both balances exactly.

        The element balance is enforced by construction: the reject composition
        is computed as the residual of the feed element flow minus the product
        element flow, so no element can be created or destroyed by rounding.
        """
        m_in = feed.kg_s
        m_prod = m_in * self.mass_yield
        m_rej = m_in - m_prod

        prod_comp: dict[str, float] = {}
        rej_comp: dict[str, float] = {}
        for el, c_in in feed.composition.items():
            el_in = m_in * c_in  # kg/s of element
            if self.grade_dependent is not None:
                rem = float(self.grade_dependent(el, c_in * 1e6))
                if not 0.0 <= rem <= 1.0:
                    raise MassBalanceError(
                        f"unit {self.name!r} grade_dependent returned {rem} for {el}"
                    )
            elif el in self.element_removal:
                rem = self.element_removal[el]
            else:
                # Not acted on: element follows the solids split, so its
                # concentration is unchanged in both product and reject.
                rem = 1.0 - self.mass_yield
            el_rej = el_in * rem
            el_prod = el_in - el_rej
            prod_comp[el] = (el_prod / m_prod) if m_prod > 0 else 0.0
            if m_rej > 0:
                rej_comp[el] = el_rej / m_rej
            elif el_rej > 1e-15:
                raise MassBalanceError(
                    f"unit {self.name!r} rejects {el} but has zero reject flow"
                )
        product = Stream(f"{self.name}.product", Q_(m_prod, "kg/s"), prod_comp)
        reject = Stream(f"{self.name}.reject", Q_(m_rej, "kg/s"), rej_comp)
        return product, reject


@dataclass
class BalanceResult:
    """Solved flowsheet state."""

    streams: dict[str, Stream]
    unit_feeds: dict[str, Stream]
    products: dict[str, Stream]
    rejects: dict[str, Stream]
    overall_yield: float
    closure_error: float
    converged: bool
    iterations: int
    method: Literal["single_pass", "successive_substitution"]

    def element_closure(self, element: str) -> float:
        """Relative element balance error across the whole flowsheet."""
        fed = sum(s.kg_s * s.composition.get(element, 0.0) for s in self.unit_feeds.values())
        out = sum(s.kg_s * s.composition.get(element, 0.0)
                  for s in list(self.products.values()) + list(self.rejects.values()))
        if fed == 0.0:
            return 0.0
        return abs(out - fed) / fed


class Flowsheet:
    """A connected set of unit operations with optional recycle.

    Connections are declared as ``(from_unit, stream_kind, to_unit)`` where
    ``stream_kind`` is ``"product"`` or ``"reject"``. A reject routed back to an
    upstream unit is a recycle; the solver detects it and handles it rather than
    looping forever.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.units: dict[str, UnitOp] = {}
        self.order: list[str] = []
        self.links: list[tuple[str, str, str]] = []

    def add(self, unit: UnitOp) -> "Flowsheet":
        if unit.name in self.units:
            raise ValueError(f"unit {unit.name!r} already present")
        self.units[unit.name] = unit
        self.order.append(unit.name)
        return self

    def connect(self, source: str, kind: Literal["product", "reject"], target: str) -> "Flowsheet":
        for u in (source, target):
            if u not in self.units:
                raise KeyError(f"unknown unit {u!r}")
        if kind not in ("product", "reject"):
            raise ValueError(f"stream kind must be 'product' or 'reject', got {kind!r}")
        self.links.append((source, kind, target))
        return self

    @property
    def has_recycle(self) -> bool:
        """True when any link routes a stream to an earlier unit in the order."""
        idx = {u: i for i, u in enumerate(self.order)}
        return any(idx[t] <= idx[s] for s, _, t in self.links)

    @property
    def is_nonlinear(self) -> bool:
        return any(u.grade_dependent is not None for u in self.units.values())

    def solve(
        self,
        feed: Stream,
        first_unit: str | None = None,
        max_iter: int = 200,
        damping: float = 0.5,
        tol: float = 1e-10,
    ) -> BalanceResult:
        """Solve the steady-state balance.

        With no recycle this is a single forward pass in declaration order
        (``method="single_pass"``). With recycle it is damped successive
        substitution on the recycled streams (``method="successive_substitution"``),
        which converges when the loop gain is below unity and reports failure
        otherwise rather than returning an unconverged answer. Note that
        ``"single_pass"`` denotes a forward sweep, NOT a linear matrix solve:
        no routing matrix is assembled anywhere in this module.
        """
        start = first_unit or self.order[0]
        if start not in self.units:
            raise KeyError(f"unknown first unit {start!r}")
        recycle_targets = self._recycle_map()
        if not recycle_targets:
            state = self._forward(feed, start, {})
            return self._result(feed, state, converged=True, iterations=1, method="single_pass")

        # Damped successive substitution on the recycled streams.
        recycled: dict[str, Stream] = {}
        prev = None
        for it in range(1, max_iter + 1):
            state = self._forward(feed, start, recycled)
            new_recycled = {
                tgt: state["rejects" if kind == "reject" else "products"][src]
                for src, kind, tgt in self.links
                if tgt in recycle_targets
            }
            if damping < 1.0 and recycled:
                blended: dict[str, Stream] = {}
                for tgt, s_new in new_recycled.items():
                    s_old = recycled.get(tgt)
                    if s_old is None:
                        blended[tgt] = s_new
                        continue
                    m = (1 - damping) * s_old.kg_s + damping * s_new.kg_s
                    comp = {
                        el: (1 - damping) * s_old.composition.get(el, 0.0)
                        + damping * s_new.composition.get(el, 0.0)
                        for el in set(s_old.composition) | set(s_new.composition)
                    }
                    blended[tgt] = Stream(s_new.name, Q_(m, "kg/s"), comp)
                new_recycled = blended
            delta = (
                max(abs(new_recycled[t].kg_s - recycled[t].kg_s) for t in new_recycled)
                if recycled and new_recycled
                else float("inf")
            )
            recycled = new_recycled
            if delta < tol:
                return self._result(feed, state, converged=True, iterations=it,
                                    method="successive_substitution")
            if prev is not None and delta > prev * 1.01 and it > 5:
                raise MassBalanceError(
                    f"recycle loop in flowsheet {self.name!r} is diverging (delta {delta:.4g} "
                    f"rising): loop gain is at or above unity, which means the loop gains "
                    f"mass. Check mass_yield and routing rather than raising max_iter."
                )
            prev = delta
        return self._result(feed, state, converged=False, iterations=max_iter,
                            method="successive_substitution")

    # --- internals ----------------------------------------------------------
    def _recycle_map(self) -> set[str]:
        idx = {u: i for i, u in enumerate(self.order)}
        return {t for s, _, t in self.links if idx[t] <= idx[s]}

    def _forward(
        self, feed: Stream, start: str, recycled: dict[str, Stream]
    ) -> dict[str, dict[str, Stream]]:
        feeds: dict[str, Stream] = {}
        products: dict[str, Stream] = {}
        rejects: dict[str, Stream] = {}
        forward_in: dict[str, list[Stream]] = {start: [feed]}
        for tgt, s in recycled.items():
            forward_in.setdefault(tgt, []).append(s)
        for uname in self.order:
            incoming = forward_in.get(uname, [])
            if not incoming:
                continue
            u_feed = _mix(incoming, f"{uname}.feed")
            feeds[uname] = u_feed
            prod, rej = self.units[uname].apply(u_feed)
            products[uname] = prod
            rejects[uname] = rej
            for src, kind, tgt in self.links:
                if src != uname:
                    continue
                if tgt in recycled:
                    continue  # handled as a recycle input next iteration
                forward_in.setdefault(tgt, []).append(prod if kind == "product" else rej)
        return {"feeds": feeds, "products": products, "rejects": rejects}

    def _result(
        self,
        feed: Stream,
        state: dict[str, dict[str, Stream]],
        converged: bool,
        iterations: int,
        method: Literal["single_pass", "successive_substitution"],
    ) -> BalanceResult:
        products, rejects = state["products"], state["rejects"]
        # A stream is terminal when it is not routed onward. Routing is keyed by
        # (unit, kind): an earlier version keyed it by unit alone, which treated
        # a unit's product as consumed whenever its REJECT was recycled, and
        # silently reported an overall yield of zero for a closed grinding loop.
        routed = {(s, k) for s, k, _ in self.links}
        final = [products[u] for u in self.order
                 if u in products and (u, "product") not in routed]
        out_mass = 0.0
        for u in self.order:
            if u in products and (u, "product") not in routed:
                out_mass += products[u].kg_s
            if u in rejects and (u, "reject") not in routed:
                out_mass += rejects[u].kg_s
        closure = abs(out_mass - feed.kg_s) / feed.kg_s if feed.kg_s else 0.0
        if converged and closure > CLOSURE_TOL:
            raise MassBalanceError(
                f"flowsheet {self.name!r} solids balance does not close: in {feed.kg_s:.6g} "
                f"kg/s, out {out_mass:.6g} kg/s, relative error {closure:.3g}. Every "
                f"rejected stream must be terminal or routed, never discarded."
            )
        prod_mass = sum(s.kg_s for s in final)
        return BalanceResult(
            streams={**{f"{k}.feed": v for k, v in state["feeds"].items()},
                     **{f"{k}.product": v for k, v in products.items()},
                     **{f"{k}.reject": v for k, v in rejects.items()}},
            unit_feeds=state["feeds"],
            products=products,
            rejects=rejects,
            overall_yield=prod_mass / feed.kg_s if feed.kg_s else 0.0,
            closure_error=closure,
            converged=converged,
            iterations=iterations,
            method=method,
        )


def _mix(streams: list[Stream], name: str) -> Stream:
    """Combine streams, mass-averaging composition over tracked elements."""
    if len(streams) == 1:
        return Stream(name, streams[0].mass_flow, dict(streams[0].composition))
    m_tot = sum(s.kg_s for s in streams)
    if m_tot <= 0:
        return Stream(name, Q_(0.0, "kg/s"), {})
    els = set().union(*(set(s.composition) for s in streams))
    comp = {
        el: sum(s.kg_s * s.composition.get(el, 0.0) for s in streams) / m_tot
        for el in els
    }
    return Stream(name, Q_(m_tot, "kg/s"), comp)
