"""Cash cost build: yield basis, currency discipline, credit transparency."""
import datetime as dt

import pytest

from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.site import Currency, LabourRates, PowerSupply, ReagentPrices, Site
from ae.core.units import Q_
from ae.econ.unit_economics import InputDemand, cash_cost

SRC = Source(citation="EIA Electric Power Monthly Table 5.6.B", tier=Tier.T1,
             url="https://www.eia.gov/electricity/monthly/", accessed=dt.date(2026, 9, 16))


def sv(mag, unit, tag=Tag.SOURCED):
    return Value(quantity=Q_(mag, unit), tag=tag, source=SRC if tag == Tag.SOURCED else None,
                 basis=None if tag == Tag.SOURCED else "test estimate")


def us_site(**kw):
    base = dict(
        site_id="US-NM-ABQ", name="New Mexico", country="US", region="NM",
        currency=Currency.USD,
        power=PowerSupply(energy_price=sv(0.0541, "USD/kWh"), rate_basis="state_average"),
        labour=LabourRates(fully_loaded_operator=sv(36.92, "USD/hour")),
        reagents=ReagentPrices(prices={"HF": sv(1.80, "USD/kg"), "HCl": sv(0.22, "USD/kg")}),
    )
    base.update(kw)
    return Site(**base)


@pytest.mark.golden
def test_cash_cost_worked_example():
    """Hand-traceable, per tonne of PRODUCT at a 0.70 cascade yield.

      electricity 250 kWh/t feed x 0.0541 USD/kWh = 13.525 USD/t feed
                                     / 0.70       = 19.3214 USD/t product
      HF          12 kg/t feed x 1.80 USD/kg      = 21.600 USD/t feed
                                     / 0.70       = 30.8571 USD/t product
      labour      3,000,000 USD/yr / 5,000 t/yr   = 600.0000 USD/t product
      freight                                     = 95.0000 USD/t product
      gross                                       = 745.1786 USD/t
      credit                                      = -20.0000
      cash cost                                   = 725.1786 USD/t
    """
    # Each line of the per-tonne chain, at the 0.70 cascade yield.
    assert 250.0 * 0.0541 == pytest.approx(13.525, abs=1e-9)
    assert 13.525 / 0.70 == pytest.approx(19.3214, abs=1e-4)
    assert 12.0 * 1.80 == pytest.approx(21.600, abs=1e-9)
    assert 21.600 / 0.70 == pytest.approx(30.8571, abs=1e-4)
    assert 3_000_000.0 / 5_000.0 == pytest.approx(600.0, abs=1e-9)
    gross_ = 19.3214 + 30.8571 + 600.0 + 95.0
    assert gross_ == pytest.approx(745.1786, abs=1e-3)
    assert gross_ - 20.0 == pytest.approx(725.1786, abs=1e-3)
    b = cash_cost(
        site=us_site(),
        demands=[InputDemand("electricity", Q_(250.0, "kWh/tonne")),
                 InputDemand("HF", Q_(12.0, "kg/tonne"))],
        cascade_yield=0.70,
        product_tonnes_per_year=5000.0,
        annual_labour=sv(3_000_000.0, "USD/year"),
        credits=[("silica fines", sv(20.0, "USD/tonne", Tag.ASSUMED))],
        freight=sv(95.0, "USD/tonne"),
    )
    got = {ln.name: ln.magnitude for ln in b.lines}
    assert got["electricity"] == pytest.approx(19.3214, abs=1e-4)
    assert got["HF"] == pytest.approx(30.8571, abs=1e-4)
    assert got["labour"] == pytest.approx(600.0, abs=1e-9)
    assert got["freight"] == pytest.approx(95.0, abs=1e-9)
    assert b.gross_cost.magnitude == pytest.approx(745.1786, abs=1e-3)
    assert b.cash_cost.magnitude == pytest.approx(725.1786, abs=1e-3)
    assert b.breakdown_sums_to_cash_cost()


def test_feed_basis_divides_by_yield_and_product_basis_does_not():
    """The most common error in a cost model. Consumables are consumed per tonne
    of FEED; at 70 percent yield a feed-basis cost is 1/0.7 = 1.4286x larger on a
    product basis. Applying the divisor twice, or not at all, is the defect."""
    kw = dict(site=us_site(), cascade_yield=0.70, product_tonnes_per_year=5000.0,
              freight_waived=True)
    feed = cash_cost(demands=[InputDemand("HF", Q_(12.0, "kg/tonne"), basis="feed")], **kw)
    prod = cash_cost(demands=[InputDemand("HF", Q_(12.0, "kg/tonne"), basis="product")], **kw)
    f = {ln.name: ln.magnitude for ln in feed.lines}["HF"]
    p = {ln.name: ln.magnitude for ln in prod.lines}["HF"]
    assert p == pytest.approx(21.60, abs=1e-9)
    assert f == pytest.approx(21.60 / 0.70, abs=1e-9)
    assert f / p == pytest.approx(1.0 / 0.70, rel=1e-12)


def test_lower_yield_raises_cost_monotonically():
    kw = dict(site=us_site(), demands=[InputDemand("HF", Q_(12.0, "kg/tonne"))],
              product_tonnes_per_year=5000.0, freight_waived=True)
    costs = [cash_cost(cascade_yield=y, **kw).cash_cost.magnitude
             for y in (0.95, 0.80, 0.60, 0.40)]
    assert costs == sorted(costs), "cost must rise as yield falls"
    assert costs[-1] / costs[0] == pytest.approx(0.95 / 0.40, rel=1e-9)


def test_freight_must_be_supplied_or_explicitly_waived():
    """Silently defaulting freight to zero gives an export project a domestic
    cost structure."""
    kw = dict(site=us_site(), demands=[InputDemand("HF", Q_(1.0, "kg/tonne"))],
              cascade_yield=0.9, product_tonnes_per_year=100.0)
    with pytest.raises(ValueError, match="neither supplied nor waived"):
        cash_cost(**kw)
    waived = cash_cost(freight_waived=True, **kw)
    line = {ln.name: ln for ln in waived.lines}["freight"]
    assert line.magnitude == 0.0
    assert "EXPLICITLY WAIVED" in (line.note or "")
    assert line.tag == Tag.ASSUMED, "a waived cost is an assumption, not a fact"


def test_missing_price_raises_with_available_keys():
    with pytest.raises(ValueError, match="no price for input 'H2SO4'"):
        cash_cost(site=us_site(), demands=[InputDemand("H2SO4", Q_(5.0, "kg/tonne"))],
                  cascade_yield=0.9, product_tonnes_per_year=100.0, freight_waived=True)


def test_wrong_currency_raises_rather_than_converting():
    inr_labour = Value(quantity=Q_(47300.0, "INR/year"), tag=Tag.ASSUMED, basis="test")
    with pytest.raises(ValueError, match="do not auto-convert"):
        cash_cost(site=us_site(), demands=[], cascade_yield=0.9,
                  product_tonnes_per_year=100.0, annual_labour=inr_labour,
                  freight_waived=True)


def test_credit_share_is_reported():
    """A project whose viability rests on a 30 percent by-product credit is a
    different proposition from one whose credits are incidental."""
    b = cash_cost(site=us_site(), demands=[InputDemand("HF", Q_(12.0, "kg/tonne"))],
                  cascade_yield=0.70, product_tonnes_per_year=5000.0,
                  credits=[("fines", sv(10.0, "USD/tonne", Tag.ASSUMED))],
                  freight_waived=True)
    assert b.gross_cost.magnitude == pytest.approx(30.8571, abs=1e-3)
    assert b.credit_share == pytest.approx(10.0 / 30.8571, rel=1e-3)
    assert b.cash_cost.magnitude == pytest.approx(20.8571, abs=1e-3)


def test_assumed_share_is_reported():
    b = cash_cost(site=us_site(),
                  demands=[InputDemand("HF", Q_(12.0, "kg/tonne"))],
                  cascade_yield=0.70, product_tonnes_per_year=5000.0,
                  annual_maintenance=sv(500_000.0, "USD/year", Tag.ASSUMED),
                  freight_waived=True)
    # maintenance (100 USD/t, ASSUMED) + waived freight (0, ASSUMED) over gross
    assert b.assumed_share == pytest.approx(100.0 / b.gross_cost.magnitude, rel=1e-6)
    assert 0.0 < b.assumed_share < 1.0


@pytest.mark.golden
def test_full_cost_adds_levelized_capital_recovery():
    """69,290,000 USD capex at an 8 percent fixed charge rate over 5,000 t/yr
    adds 69,290,000 x 0.08 / 5,000 = 1,108.64 USD/t."""
    b = cash_cost(site=us_site(), demands=[InputDemand("HF", Q_(12.0, "kg/tonne"))],
                  cascade_yield=0.70, product_tonnes_per_year=5000.0,
                  capex=sv(69_290_000.0, "USD", Tag.ASSUMED), fixed_charge_rate=0.08,
                  freight_waived=True)
    assert b.full_cost is not None
    delta = b.full_cost.magnitude - b.cash_cost.magnitude
    assert delta == pytest.approx(1108.64, abs=0.01)


def test_capex_without_charge_rate_and_bad_rate_rejected():
    kw = dict(site=us_site(), demands=[], cascade_yield=0.9,
              product_tonnes_per_year=100.0, freight_waived=True)
    with pytest.raises(ValueError, match="without a fixed_charge_rate"):
        cash_cost(capex=sv(1e6, "USD", Tag.ASSUMED), **kw)
    with pytest.raises(ValueError, match="outside"):
        cash_cost(capex=sv(1e6, "USD", Tag.ASSUMED), fixed_charge_rate=1.5, **kw)


def test_input_and_yield_validation():
    with pytest.raises(ValueError, match="basis must be"):
        InputDemand("x", Q_(1.0, "kg/tonne"), basis="tonnes")
    with pytest.raises(ValueError, match="cannot be negative"):
        InputDemand("x", Q_(-1.0, "kg/tonne"))
    with pytest.raises(ValueError):
        cash_cost(site=us_site(), demands=[], cascade_yield=0.0,
                  product_tonnes_per_year=100.0, freight_waived=True)
    with pytest.raises(ValueError, match="product_tonnes_per_year"):
        cash_cost(site=us_site(), demands=[], cascade_yield=0.9,
                  product_tonnes_per_year=0.0, freight_waived=True)


def test_annual_costs_require_a_rate_basis_not_a_bare_tonnage():
    """Regression. An annual cost divided by a bare tonnage leaves a stray
    1/time dimension; the divisor must carry the same time basis as the
    numerator. pint caught this when the module was first written."""
    b = cash_cost(site=us_site(), demands=[], cascade_yield=0.9,
                  product_tonnes_per_year=5000.0,
                  annual_labour=sv(3_000_000.0, "USD/year"), freight_waived=True)
    lab = {ln.name: ln for ln in b.lines}["labour"]
    assert str(lab.amount.units) in ("USD / metric_ton", "USD / tonne")
    assert lab.magnitude == pytest.approx(600.0, abs=1e-9)
    # A per-month figure must give the same annual answer once converted.
    b2 = cash_cost(site=us_site(), demands=[], cascade_yield=0.9,
                   product_tonnes_per_year=5000.0,
                   annual_labour=sv(250_000.0, "USD/month"), freight_waived=True)
    lab2 = {ln.name: ln for ln in b2.lines}["labour"]
    assert lab2.magnitude == pytest.approx(600.0, rel=1e-9)


def test_every_line_carries_a_traceable_note():
    b = cash_cost(site=us_site(),
                  demands=[InputDemand("electricity", Q_(250.0, "kWh/tonne"))],
                  cascade_yield=0.70, product_tonnes_per_year=5000.0, freight_waived=True)
    for ln in b.lines:
        assert ln.note, f"line {ln.name} has no provenance note"
    elec = {ln.name: ln for ln in b.lines}["electricity"]
    assert "divided by yield" in elec.note
    assert elec.tag == Tag.SOURCED
    recs = b.to_records()
    assert all(set(r) == {"line", "usd_per_tonne", "tag", "note", "kind"} for r in recs)
