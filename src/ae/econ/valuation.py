r"""Discounted cash flow, IRR and levelized cost, with construction and ramp.

This module exists in the form it does because of a specific, expensive error
found in the predecessor analysis: capital was booked at time zero and revenue
from year one, while the qualification roadmap put first qualified revenue 18 to
54 months out. Correcting the timing moved one route's breakeven price from
3,722 to 5,144 USD per tonne, a 38 percent change from timing alone, with no
change to any cost or price input. Timing is therefore not optional here. A
:class:`Project` must state its construction period and its ramp, and revenue
before first qualified sale is structurally impossible to express.

Governing relations
-------------------
Net present value, with cash flows at end of period:

.. math::
   \mathrm{NPV} = \sum_{t=0}^{N} \frac{F_t}{(1+r)^{t}}

where :math:`F_t` is the net cash flow in period :math:`t` (currency) and
:math:`r` the discount rate per period (dimensionless, typically 0.08 to 0.25
for an industrial project; higher with a country risk premium).

Internal rate of return is the :math:`r` solving :math:`\mathrm{NPV}(r) = 0`.
It need not exist or be unique: a cash flow stream with more than one sign
change can have multiple roots, and one that never turns positive has none.
:func:`irr` returns ``None`` in both cases rather than a misleading number, and
:func:`modified_irr` is offered because MIRR is unique by construction.

Levelized cost of product (LCOP), the price at which NPV is exactly zero:

.. math::
   \mathrm{LCOP} = \frac{\sum_t (C_t + I_t)/(1+r)^t}{\sum_t Q_t/(1+r)^t}

with :math:`C_t` operating cost, :math:`I_t` capital spend and :math:`Q_t`
product volume in period :math:`t` (tonne). Note both numerator and denominator
are discounted. Discounting only the costs is a common error that understates
LCOP whenever output is back-loaded, which it always is under a ramp.

Symbols
-------
:math:`r` discount rate (1/period), :math:`t` period index, :math:`N` project
life (periods), :math:`F_t` net cash flow (currency), :math:`Q_t` output
(tonne/period), :math:`\tau` tax rate (dimensionless 0 to 1).

LIMITATIONS
-----------
* Annual periods only. A project whose working capital swings within a year is
  not represented, and monthly resolution would change the answer for a
  short-cycle business.
* Straight-line depreciation over a stated life. Accelerated schedules (MACRS)
  change after-tax NPV materially and are modelled only through the incentives
  layer, not here.
* No debt. All-equity, so the discount rate carries the whole cost of capital
  and there is no tax shield on interest. A levered case needs a WACC and a
  debt schedule, and quoting this NPV as if levered overstates returns.
* Terminal value is zero unless explicitly supplied. A salvage assumption is a
  decision and is not defaulted.
* Prices are real and constant unless a price path is supplied. A real discount
  rate must therefore be used with real prices; mixing a nominal rate with real
  prices is a standard error worth tens of percent over a 20 year life.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CashFlowResult",
    "Project",
    "breakeven_price",
    "irr",
    "levelized_cost",
    "modified_irr",
    "npv",
    "payback_period",
]


def npv(rate: float, flows: np.ndarray | list[float]) -> float:
    """Net present value with flows at ``t = 0, 1, 2, ...``.

    Examples
    --------
    A 100 outflow followed by two inflows of 60 at 10 percent:
    minus 100 + 60/1.1 + 60/1.21 = 4.1322.

    >>> round(npv(0.10, [-100.0, 60.0, 60.0]), 4)
    4.1322
    """
    f = np.asarray(flows, dtype=float)
    if rate <= -1.0:
        raise ValueError(f"discount rate {rate} is at or below -100 percent")
    t = np.arange(f.size)
    return float(np.sum(f / (1.0 + rate) ** t))


def _sign_changes(f: np.ndarray) -> int:
    nz = f[f != 0.0]
    return int(np.sum(np.diff(np.sign(nz)) != 0)) if nz.size > 1 else 0


def irr(flows: np.ndarray | list[float], bracket: tuple[float, float] | None = None,
        tol: float = 1e-9) -> float | None:
    """Internal rate of return, or ``None`` when it is not well defined.

    Returns ``None`` when the stream has no real rate above -100 percent, and
    when it has more than one, because reporting one of several roots as "the"
    IRR is misleading. Use :func:`modified_irr`, which is unique by
    construction.

    Why this is solved exactly rather than scanned. NPV as a function of
    :math:`x = 1/(1+r)` is a polynomial in the cash flows, so its roots are
    obtained exactly from the companion matrix (``numpy.roots``) and each real
    root with :math:`x > 0` is a rate above -100 percent. The earlier
    implementation scanned a grid over a bracket whose upper bound was 10.0 (a
    1000 percent return), and that produced errors in BOTH directions which
    were indistinguishable from correct answers.

    Root above the bound, reported as no root at all: ``[-1.0, 20.0]`` has the
    single exact root 20/1 - 1 = 19.0 and returned ``None``, which the
    docstring reserves for streams that never turn positive.

    Second root above the bound, invisible: flows built from roots at
    :math:`r = 0.20` and :math:`r = 50.0` returned 0.19999999999995 as "the"
    IRR on a stream that has two, the exact failure the multiplicity check
    exists to catch. A first attempted repair widened the bound to 1e4 and
    cross-examined the scan against the sign-change count. That repair was
    UNSOUND and is recorded here rather than quietly replaced: equality of the
    NPV sign at the bracket edge carries no information about roots beyond it,
    and a root pair at 0.20 and 2.0e4 still returned 0.19999999999996645. A
    grid scan can only ever lower-bound the root count, so no bracket-based
    multiplicity test can be correct. The exact solve has no bracket.

    ``bracket`` is retained for call compatibility and ignored; a value passed
    for it is refused rather than silently disregarded.

    Examples
    --------
    >>> round(irr([-100.0, 60.0, 60.0]), 6)
    0.130662
    >>> irr([-100.0, -50.0]) is None
    True
    >>> round(irr([-1.0, 20.0]), 6)
    19.0
    >>> irr([-100.0, 230.0, -132.0]) is None
    True
    """
    if bracket is not None:
        raise ValueError(
            "irr no longer scans a bracket: the roots are solved exactly, and "
            "a bracket-based multiplicity test cannot be correct because a "
            "grid scan only lower-bounds the root count. Remove the argument."
        )
    f = np.asarray(flows, dtype=float)
    if f.size < 2:
        raise ValueError("an IRR needs at least two periods")
    if _sign_changes(f) == 0:
        return None

    # NPV(r) = sum_t f_t x^t with x = 1/(1+r). numpy.roots wants descending
    # powers, and leading zeros (trailing zero cash flows) must be dropped or
    # the companion matrix is singular.
    coef = f[::-1]
    nonzero = np.flatnonzero(coef != 0.0)
    if nonzero.size == 0:
        return None
    coef = coef[nonzero[0]:]
    if coef.size < 2:
        return None

    x = np.roots(coef)
    scale = max(1.0, float(np.max(np.abs(x)))) if x.size else 1.0
    real_x = x[np.abs(x.imag) <= 1e-8 * scale].real
    positive_x = real_x[real_x > 1e-14]
    rates = sorted(float(1.0 / xi - 1.0) for xi in positive_x)

    distinct: list[float] = []
    for r in rates:
        if not distinct or abs(r - distinct[-1]) > 1e-7 * max(1.0, abs(r)):
            distinct.append(r)
    if len(distinct) != 1:
        return None
    r0 = distinct[0]
    # Polish once by Newton so the returned rate satisfies npv(r) = 0 to tol
    # even when the companion-matrix eigenvalue is slightly off.
    for _ in range(50):
        y = npv(r0, f)
        if abs(y) < tol:
            break
        h = 1e-8 * max(1.0, abs(1.0 + r0))
        dy = (npv(r0 + h, f) - npv(r0 - h, f)) / (2.0 * h)
        if dy == 0.0:
            break
        step = y / dy
        if not np.isfinite(step):
            break
        r0 -= step
        if r0 <= -1.0:
            return None
    return float(r0)


def modified_irr(flows: np.ndarray | list[float], finance_rate: float,
                 reinvest_rate: float) -> float | None:
    """Modified IRR: unique by construction, unlike IRR.

    Negative flows are discounted to time zero at ``finance_rate`` and positive
    flows compounded to the final period at ``reinvest_rate``. This removes both
    the multiple-root problem and the implicit assumption that interim cash is
    reinvested at the IRR itself, which is usually indefensible.
    """
    f = np.asarray(flows, dtype=float)
    n = f.size - 1
    if n < 1:
        raise ValueError("MIRR needs at least two periods")
    t = np.arange(f.size)
    neg = np.where(f < 0, f, 0.0)
    pos = np.where(f > 0, f, 0.0)
    pv_neg = float(np.sum(neg / (1.0 + finance_rate) ** t))
    fv_pos = float(np.sum(pos * (1.0 + reinvest_rate) ** (n - t)))
    if pv_neg == 0.0 or fv_pos <= 0.0:
        return None
    return float((fv_pos / -pv_neg) ** (1.0 / n) - 1.0)


def payback_period(flows: np.ndarray | list[float], rate: float = 0.0) -> float | None:
    """Periods until cumulative discounted cash flow first turns non-negative.

    Interpolated within the crossing period. ``None`` if it never pays back,
    which is a real outcome and must not be reported as a large number.
    """
    f = np.asarray(flows, dtype=float)
    t = np.arange(f.size)
    disc = f / (1.0 + rate) ** t
    cum = np.cumsum(disc)
    idx = np.nonzero(cum >= 0.0)[0]
    if idx.size == 0:
        return None
    k = int(idx[0])
    if k == 0:
        return 0.0
    prev = cum[k - 1]
    step = cum[k] - prev
    return float(k - 1 + (-prev / step)) if step != 0 else float(k)


@dataclass
class Project:
    """A project's timing, capital, cost and output profile.

    Parameters
    ----------
    capex_schedule
        Capital spend per period starting at t=0. A single-element list books
        everything at time zero, which is what the predecessor error did; a
        realistic build spreads it over the construction period.
    construction_periods
        Periods before ANY output. Output and revenue are zero throughout, and
        this is enforced rather than trusted.
    ramp_fractions
        Fraction of nameplate achieved in each period after construction, e.g.
        ``[0.3, 0.7, 1.0]``. Must be non-decreasing and end at or below 1.0.
    nameplate_tonnes
        Annual output at full rate.
    price
        Product price per tonne, real.
    cash_cost_per_tonne
        Variable and fixed cash cost per tonne of product at full rate.
    fixed_cost_per_period
        Costs incurred regardless of output, including during construction if
        ``fixed_cost_in_construction`` is set. A plant under construction with
        no fixed cost is unrealistic and understates the funding requirement.
    life_periods
        Total operating periods after construction.
    tax_rate, depreciation_periods
        Straight-line depreciation over the stated life; tax applies to
        operating profit after depreciation, with losses carried forward.
    salvage
        Terminal value, zero unless supplied.
    """

    capex_schedule: list[float]
    construction_periods: int
    ramp_fractions: list[float]
    nameplate_tonnes: float
    price: float
    cash_cost_per_tonne: float
    life_periods: int
    fixed_cost_per_period: float = 0.0
    fixed_cost_in_construction: bool = False
    tax_rate: float = 0.0
    depreciation_periods: int | None = None
    salvage: float = 0.0
    working_capital_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.capex_schedule:
            raise ValueError("capex_schedule must have at least one period")
        if any(c < 0 for c in self.capex_schedule):
            raise ValueError("capex cannot be negative")
        if self.construction_periods < 0:
            raise ValueError("construction_periods cannot be negative")
        if self.construction_periods == 0:
            raise ValueError(
                "construction_periods is 0, which books revenue in the same period as "
                "the capital that creates the plant. This is the specific error this "
                "module exists to prevent: in the predecessor analysis it moved a "
                "breakeven price by 38 percent. State a real build period."
            )
        if not self.ramp_fractions:
            raise ValueError("ramp_fractions must have at least one period")
        if any(not 0.0 <= f <= 1.0 for f in self.ramp_fractions):
            raise ValueError("ramp fractions must lie in [0, 1]")
        if any(b < a for a, b in zip(self.ramp_fractions, self.ramp_fractions[1:])):
            raise ValueError(
                f"ramp_fractions {self.ramp_fractions} decreases. A ramp is monotonic; "
                f"a decline belongs in a price or volume path, not the ramp."
            )
        if self.life_periods < len(self.ramp_fractions):
            raise ValueError(
                f"life_periods {self.life_periods} is shorter than the ramp "
                f"({len(self.ramp_fractions)} periods), so the plant never reaches rate"
            )
        if self.nameplate_tonnes <= 0:
            raise ValueError("nameplate_tonnes must be positive")
        if not 0.0 <= self.tax_rate < 1.0:
            raise ValueError(f"tax_rate {self.tax_rate} outside [0, 1)")
        if self.depreciation_periods is not None and self.depreciation_periods < 1:
            raise ValueError("depreciation_periods must be at least 1")

    @property
    def total_capex(self) -> float:
        return float(sum(self.capex_schedule))

    def output_profile(self) -> np.ndarray:
        """Product tonnes per period, zero through construction."""
        n = self.construction_periods + self.life_periods
        q = np.zeros(n)
        for i in range(self.life_periods):
            frac = (self.ramp_fractions[i] if i < len(self.ramp_fractions)
                    else self.ramp_fractions[-1])
            q[self.construction_periods + i] = frac * self.nameplate_tonnes
        return q

    def _depreciation_schedule(self, capex: np.ndarray, n: int) -> np.ndarray:
        """Straight-line allowance per tranche, never truncated, never early.

        Two defects in the earlier single-line version are fixed here.

        Truncation: it wrote ``total_capex / dep_periods`` into the slice
        ``[construction : construction + dep_periods]``, which numpy silently
        clips at the array end. With 1000 of capex, 20 depreciation periods and
        a 5 period life, only 250 of the 1000 was ever booked and 750 of
        allowance was discarded with no error, so tax was charged on income a
        tax authority would have sheltered. The schedule is now laid down
        tranche by tranche and any allowance that would fall past the final
        period is recognised in that period, so the total always equals the
        capital actually spent.

        Allowance before the spend: it started every period's allowance at
        ``construction_periods`` regardless of when the money left. A 100 spend
        at t=0 with a 900 expansion at t=4, written over 6 periods from t=2,
        reached 333.33 of cumulative allowance by t=3 against 100.00 spent.
        Each tranche now begins in the period it is spent, except tranches
        inside the construction window which begin at first operation, since
        an asset not yet in service is not yet depreciable.
        """
        dep_periods = self.depreciation_periods or self.life_periods
        dep = np.zeros(n)
        last = n - 1
        for i, c in enumerate(capex):
            if c <= 0.0:
                continue
            start = max(i, self.construction_periods)
            if start > last:
                dep[last] += c
                continue
            per = c / dep_periods
            for k in range(dep_periods):
                t = start + k
                if t > last:
                    # Remaining allowance is recognised on disposal rather than
                    # lost: the alternative discards capital relief silently.
                    dep[last] += per
                else:
                    dep[t] += per
        return dep

    def cash_flows(self) -> CashFlowResult:
        """Build the after-tax cash flow stream with explicit timing."""
        q = self.output_profile()
        n = q.size
        capex = np.zeros(n)
        for i, c in enumerate(self.capex_schedule):
            if i >= n:
                raise ValueError("capex_schedule extends beyond the project life")
            capex[i] = c

        revenue = q * self.price
        variable = q * self.cash_cost_per_tonne
        fixed = np.zeros(n)
        start = 0 if self.fixed_cost_in_construction else self.construction_periods
        fixed[start:] = self.fixed_cost_per_period

        # Working capital builds with output and is released at the end.
        wc_level = self.working_capital_fraction * revenue
        wc_flow = -np.diff(np.concatenate([[0.0], wc_level]))
        wc_flow[-1] += wc_level[-1]

        ebitda = revenue - variable - fixed
        dep = self._depreciation_schedule(capex, n)

        # Terminal disposal is a taxable event, not a cash windfall. Book value
        # is capex spent less allowance claimed, and the gain over book value is
        # taxed at the same rate as operating income. Adding salvage to the net
        # flow AFTER tax, as an earlier version did, let a fully depreciated
        # asset be sold tax free: with 1000 of capex fully written down, a 400
        # salvage is a 400 gain and owes 120 at 30 percent.
        book_value = float(np.sum(capex) - np.sum(dep))
        disposal_gain = self.salvage - book_value

        taxable = ebitda - dep
        taxable[-1] += disposal_gain
        tax = np.zeros(n)
        carry = 0.0
        for i in range(n):
            base = taxable[i] + carry
            if base > 0:
                tax[i] = base * self.tax_rate
                carry = 0.0
            else:
                carry = base
        net = ebitda - tax - capex + wc_flow
        net[-1] += self.salvage
        return CashFlowResult(
            periods=np.arange(n), output=q, revenue=revenue, ebitda=ebitda,
            depreciation=dep, tax=tax, capex=capex, working_capital=wc_flow,
            net_cash_flow=net, project=self,
        )


@dataclass
class CashFlowResult:
    """A period-by-period cash flow stream and the metrics derived from it."""

    periods: np.ndarray
    output: np.ndarray
    revenue: np.ndarray
    ebitda: np.ndarray
    depreciation: np.ndarray
    tax: np.ndarray
    capex: np.ndarray
    working_capital: np.ndarray
    net_cash_flow: np.ndarray
    project: Project = field(repr=False)

    def npv(self, rate: float) -> float:
        return npv(rate, self.net_cash_flow)

    def irr(self) -> float | None:
        return irr(self.net_cash_flow)

    def payback(self, rate: float = 0.0) -> float | None:
        return payback_period(self.net_cash_flow, rate)

    def first_revenue_period(self) -> int | None:
        nz = np.nonzero(self.revenue > 0)[0]
        return int(nz[0]) if nz.size else None

    def peak_funding_requirement(self, rate: float = 0.0) -> float:
        """Most negative cumulative cash position, i.e. the capital to raise."""
        t = np.arange(self.net_cash_flow.size)
        cum = np.cumsum(self.net_cash_flow / (1.0 + rate) ** t)
        return float(min(0.0, cum.min()))

    def to_records(self) -> list[dict[str, float]]:
        return [
            {"period": int(t), "output_tonnes": float(self.output[i]),
             "revenue": float(self.revenue[i]), "ebitda": float(self.ebitda[i]),
             "depreciation": float(self.depreciation[i]), "tax": float(self.tax[i]),
             "capex": float(self.capex[i]),
             "working_capital": float(self.working_capital[i]),
             "net_cash_flow": float(self.net_cash_flow[i])}
            for i, t in enumerate(self.periods)
        ]


def levelized_cost(project: Project, rate: float) -> float:
    """Levelized cost per tonne: discounted costs over DISCOUNTED output.

    Discounting the numerator but not the denominator understates LCOP whenever
    output is back-loaded, which it always is under a ramp. That error is
    checked against in the test suite rather than merely avoided here.

    Examples
    --------
    With no ramp and no tax, LCOP reduces to cash cost plus levelized capital.
    """
    cf = project.cash_flows()
    t = np.arange(cf.output.size)
    disc = (1.0 + rate) ** t
    costs = cf.capex + (cf.revenue - cf.ebitda)  # capex + variable + fixed
    pv_cost = float(np.sum(costs / disc))
    pv_out = float(np.sum(cf.output / disc))
    if pv_out <= 0:
        raise ValueError("project has no discounted output; LCOP is undefined")
    return pv_cost / pv_out


def breakeven_price(project: Project, rate: float, tol: float = 1e-8) -> float:
    """Price at which NPV is exactly zero, found by bisection.

    Solved rather than approximated, because with tax and loss carry-forward the
    NPV is only piecewise linear in price, so the closed-form LCOP and the true
    breakeven price differ once tax binds.
    """
    lo, hi = 0.0, max(10.0 * project.cash_cost_per_tonne, 1.0)
    import dataclasses as _dc

    def f(p: float) -> float:
        return _dc.replace(project, price=p).cash_flows().npv(rate)

    grow = 0
    while f(hi) < 0.0:
        hi *= 2.0
        grow += 1
        if grow > 60:
            raise ValueError(
                "no price makes this project NPV-positive within 2^60 of the cost "
                "basis; the structure, not the price, is the problem"
            )
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if abs(hi - lo) < tol:
            break
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
