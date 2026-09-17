r"""Cash cost per tonne of product, built from the physics rather than asserted.

Every line item traces to a physical quantity computed elsewhere in the platform
multiplied by a SITE price, so a cost cannot drift away from the process that
generates it. Energy comes from :mod:`ae.physics.thermal` and the comminution
work index, reagents from :mod:`ae.physics.reagents` stoichiometry against the
impurity load, and the yield divisor from :mod:`ae.plant.yield_cascade`.

Cost basis
----------
.. math::
   C_{\mathrm{cash}} = \frac{1}{Y}\left(
     \sum_k q_k p_k + \frac{L + M + O}{\dot m_{\mathrm{prod}}}\right)
     - c_{\mathrm{credit}}

where :math:`Y` is the cascade mass yield (dimensionless, 0 to 1), :math:`q_k`
the specific consumption of input :math:`k` per tonne of FEED (units per tonne),
:math:`p_k` its unit price at this site (currency per unit), :math:`L`, :math:`M`
and :math:`O` annual labour, maintenance and overhead (currency per year),
:math:`\dot m_{\mathrm{prod}}` annual product output (tonne per year), and
:math:`c_{\mathrm{credit}}` by-product credit per tonne of product.

The :math:`1/Y` is where cost models most often go wrong. Consumables are
consumed per tonne of feed processed, not per tonne shipped, so at a 70 percent
yield every feed-basis cost is 1.43 times larger on a product basis. Mixing the
two bases understates cost by exactly the yield loss.

**Full cost** adds capital recovery at a fixed charge rate:

.. math::
   C_{\mathrm{full}} = C_{\mathrm{cash}}
     + \frac{f\,\mathrm{CAPEX}}{\dot m_{\mathrm{prod}}}

with :math:`f` the fixed charge rate (1/yr, typically 0.08 to 0.15). This is a
levelized approximation, not a substitute for the discounted cash flow in
:mod:`ae.econ.valuation`; it exists because it is the form the industry quotes
and because it is auditable by hand.

LIMITATIONS
-----------
* Steady state at design capacity. A ramp period, where fixed costs are spread
  over less output, is modelled in the valuation layer, not here. Quoting a
  design-capacity cash cost for year one overstates margin materially.
* The fixed charge rate collapses financing structure, tax and asset life into
  one number. It is convenient and wrong in detail; use it for comparison
  between sites, not for a financing decision.
* By-product credits are the most manipulated line in any cost model. Each
  credit must carry its own provenance, and :meth:`CostBuild.credit_share`
  reports what fraction of the cost reduction they supply so a reader can see
  how load-bearing they are.
* No working capital, no inventory carry, no logistics beyond an explicit
  freight line. Freight to a customer is a real cost and is not defaulted to
  zero silently: it must be supplied or explicitly waived.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ae.core.provenance import Tag, Value
from ae.core.site import Site
from ae.core.units import Q_, Quantity, require_dimensionality, require_fraction

__all__ = ["CostBuild", "CostLine", "InputDemand", "cash_cost"]


def _require_currency(q: Quantity, site: Site, what: str) -> Quantity:
    """Check a quantity is denominated in this site's currency.

    There is no "currency" entry in :data:`ae.core.units.DIMS`, deliberately:
    USD and INR are independent base units precisely so they cannot be summed,
    so a single dimensionality check would not distinguish them. The check is
    therefore on the unit itself, and a mismatch names both currencies rather
    than converting silently.
    """
    cur = site.currency.value
    if cur not in str(q.units):
        raise ValueError(
            f"{what} is denominated in {q.units}, but site {site.site_id} operates in "
            f"{cur}. Convert explicitly through an ExchangeRate carrying a date and a "
            f"source; currencies do not auto-convert."
        )
    return q


def _resolve_price(site: Site, name: str) -> Value:
    """Find an input price on a Site, searching reagents then power.

    Raises with the available keys listed, because a silently missing price
    becomes a silently missing cost line, which is the failure mode that makes a
    cost model look competitive.
    """
    if name in site.reagents.prices:
        return _price_checked(site, name, site.reagents.prices[name])
    if name in ("electricity", "power", "energy"):
        return _price_checked(site, name, site.power.energy_price)
    raise ValueError(
        f"site {site.site_id!r} has no price for input {name!r}. Priced reagents: "
        f"{sorted(site.reagents.prices)}; plus 'electricity' from site.power. "
        f"Add the price to the site or drop the demand: a missing price must not "
        f"become a missing cost line."
    )


def _price_checked(site: Site, name: str, v: Value) -> Value:
    _require_currency(v.quantity, site, f"price of {name!r}")
    if v.quantity.magnitude < 0:
        raise ValueError(f"price of {name!r} is negative: {v.quantity:~P}")
    return v


@dataclass(frozen=True)
class InputDemand:
    """Specific consumption of one input, per tonne of FEED.

    Parameters
    ----------
    name
        Input label, matching a price key on the Site.
    per_tonne_feed
        Quantity consumed per tonne of feed. Dimensionality is free (kWh, kg,
        m^3) but must match the price's denominator, which is checked.
    basis
        ``"feed"`` or ``"product"``. Defaults to feed, and the distinction is
        enforced rather than assumed: an input quoted per tonne of product must
        NOT be divided by the yield again.
    """

    name: str
    per_tonne_feed: Quantity
    basis: str = "feed"

    def __post_init__(self) -> None:
        if self.basis not in ("feed", "product"):
            raise ValueError(f"{self.name}: basis must be 'feed' or 'product'")
        if self.per_tonne_feed.magnitude < 0:
            raise ValueError(f"{self.name}: consumption cannot be negative")


@dataclass(frozen=True)
class CostLine:
    """One resolved cost line, per tonne of product."""

    name: str
    amount: Quantity
    tag: Tag
    note: str | None = None

    @property
    def magnitude(self) -> float:
        return float(self.amount.magnitude)


@dataclass
class CostBuild:
    """A complete cash and full cost build with per-line provenance."""

    lines: list[CostLine]
    cash_cost: Quantity
    full_cost: Quantity | None
    yield_used: float
    product_tonnes_per_year: float
    credits: list[CostLine] = field(default_factory=list)

    @property
    def gross_cost(self) -> Quantity:
        """Cash cost before by-product credits, i.e. the sum of the cost lines."""
        zero = Q_(0.0, str(self.cash_cost.units))
        return sum((ln.amount for ln in self.lines), zero)

    @property
    def credit_share(self) -> float:
        """Fraction of gross cost offset by by-product credits.

        Reported because by-product credits are the most manipulated line in any
        cost model: a project whose viability rests on a 30 percent credit is a
        different proposition from one whose credits are incidental.
        """
        gross = float(self.gross_cost.magnitude)
        if gross <= 0:
            return 0.0
        return float(sum(c.magnitude for c in self.credits)) / gross

    @property
    def assumed_share(self) -> float:
        """Fraction of gross cost carried by ASSUMED (unsourced) lines."""
        gross = float(self.gross_cost.magnitude)
        if gross <= 0:
            return 0.0
        assumed = sum(ln.magnitude for ln in self.lines if ln.tag == Tag.ASSUMED)
        return assumed / gross

    def to_records(self) -> list[dict[str, object]]:
        return [
            {"line": ln.name, "usd_per_tonne": ln.magnitude, "tag": ln.tag.value,
             "note": ln.note, "kind": "credit" if ln in self.credits else "cost"}
            for ln in self.lines + self.credits
        ]

    def breakdown_sums_to_cash_cost(self, tol: float = 1e-9) -> bool:
        """Self-check: the lines must reconstruct the headline number."""
        total = sum(ln.magnitude for ln in self.lines) - sum(c.magnitude for c in self.credits)
        return abs(total - float(self.cash_cost.magnitude)) < tol


def cash_cost(
    site: Site,
    demands: list[InputDemand],
    cascade_yield: float,
    product_tonnes_per_year: float,
    annual_labour: Value | None = None,
    annual_maintenance: Value | None = None,
    annual_overhead: Value | None = None,
    credits: list[tuple[str, Value]] | None = None,
    capex: Value | None = None,
    fixed_charge_rate: float | None = None,
    freight: Value | None = None,
    freight_waived: bool = False,
) -> CostBuild:
    """Build a cash cost per tonne of product from physical demands and prices.

    Parameters
    ----------
    site
        Supplies input prices, currency and labour rates.
    demands
        Specific consumptions, per tonne of feed unless flagged otherwise.
    cascade_yield
        Mass yield from :mod:`ae.plant.yield_cascade`. Feed-basis demands are
        divided by this to reach a product basis.
    product_tonnes_per_year
        Annual product output, used to spread annual fixed costs.
    freight, freight_waived
        Freight must be supplied or explicitly waived. Silently defaulting it to
        zero is how an export project acquires a domestic cost structure.

    Raises
    ------
    ValueError
        If freight is neither supplied nor waived, if a price is missing for a
        demand, or if capex is given without a fixed charge rate.
    """
    require_fraction(cascade_yield, "cascade_yield", lo=1e-6, hi=1.0)
    if product_tonnes_per_year <= 0:
        raise ValueError("product_tonnes_per_year must be positive")
    if freight is None and not freight_waived:
        raise ValueError(
            "freight is neither supplied nor waived. Freight to the customer is a "
            "real cost, and defaulting it to zero gives an export project a "
            "domestic cost structure. Pass freight=Value(...) or freight_waived=True "
            "with a reason in the note."
        )
    if capex is not None and fixed_charge_rate is None:
        raise ValueError("capex supplied without a fixed_charge_rate: full cost is undefined")

    cur = site.currency.value
    lines: list[CostLine] = []

    for d in demands:
        price = _resolve_price(site, d.name)
        cost_per_feed_tonne = (d.per_tonne_feed * price.quantity).to(f"{cur}/tonne")
        per_product = (cost_per_feed_tonne / cascade_yield if d.basis == "feed"
                       else cost_per_feed_tonne)
        lines.append(CostLine(
            name=d.name, amount=per_product, tag=price.tag,
            note=(f"{d.per_tonne_feed:~P} per tonne {d.basis} at {price.quantity:~P}"
                  + (f", divided by yield {cascade_yield:.4f}" if d.basis == "feed" else "")),
        ))

    # Annual fixed costs are spread over the annual product RATE, not a bare
    # tonnage. Dividing a currency-per-year by tonnes leaves a stray 1/time
    # dimension, which pint rejects: the divisor must carry the same time basis
    # as the numerator. The failure is the check working, not a nuisance.
    annual_rate = Q_(product_tonnes_per_year, "tonne/year")
    for label, v in (("labour", annual_labour), ("maintenance", annual_maintenance),
                     ("overhead", annual_overhead)):
        if v is None:
            continue
        _require_currency(v.quantity, site, f"annual {label}")
        require_dimensionality(v.quantity / annual_rate, "cost_per_mass_usd"
                               if cur == "USD" else "cost_per_mass_inr",
                               f"annual {label} spread over output")
        lines.append(CostLine(
            name=label,
            amount=(v.quantity / annual_rate).to(f"{cur}/tonne"),
            tag=v.tag,
            note=f"{v.quantity:~P} over {product_tonnes_per_year:,.0f} t/yr",
        ))

    if freight is not None:
        lines.append(CostLine(name="freight", amount=freight.quantity.to(f"{cur}/tonne"),
                              tag=freight.tag, note="delivered basis"))
    else:
        lines.append(CostLine(name="freight", amount=Q_(0.0, f"{cur}/tonne"),
                              tag=Tag.ASSUMED,
                              note="EXPLICITLY WAIVED: ex-works basis, buyer collects"))

    credit_lines: list[CostLine] = []
    for name, v in (credits or []):
        credit_lines.append(CostLine(name=name, amount=v.quantity.to(f"{cur}/tonne"),
                                     tag=v.tag, note="by-product credit"))

    gross = sum((ln.amount for ln in lines), Q_(0.0, f"{cur}/tonne"))
    credit_total = sum((c.amount for c in credit_lines), Q_(0.0, f"{cur}/tonne"))
    cash = gross - credit_total

    full: Quantity | None = None
    if capex is not None and fixed_charge_rate is not None:
        if not 0.0 < fixed_charge_rate < 1.0:
            raise ValueError(
                f"fixed_charge_rate {fixed_charge_rate} is outside (0, 1); typical "
                f"values are 0.08 to 0.15 per year"
            )
        # capex (currency) x charge rate (1/yr) / (tonne/yr) -> currency/tonne
        recovery = (capex.quantity * Q_(fixed_charge_rate, "1/year")
                    / annual_rate).to(f"{cur}/tonne")
        full = cash + recovery
        lines.append(CostLine(name="capital recovery", amount=recovery, tag=capex.tag,
                              note=f"{capex.quantity:~P} at {fixed_charge_rate:.3f}/yr "
                                   f"fixed charge rate (levelized, not a DCF)"))

    build = CostBuild(lines=[ln for ln in lines if ln.name != "capital recovery"],
                      cash_cost=cash, full_cost=full, yield_used=cascade_yield,
                      product_tonnes_per_year=product_tonnes_per_year,
                      credits=credit_lines)
    if not build.breakdown_sums_to_cash_cost():
        raise AssertionError(
            "internal inconsistency: cost lines do not sum to the reported cash cost"
        )
    return build
