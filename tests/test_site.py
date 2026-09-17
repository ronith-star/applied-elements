"""SITE schema tests: currency discipline, uncertain policy, contract-rate flags."""
import datetime as dt

import pytest

from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.site import (
    Currency,
    ExchangeRate,
    Incentive,
    LabourRates,
    LogisticsLink,
    PowerSupply,
    ReagentPrices,
    Site,
    TradeMeasure,
)
from ae.core.units import Q_, DimensionalityError

SRC = Source(citation="EIA Electric Power Monthly Table 5.6.B", tier=Tier.T1,
             url="https://www.eia.gov/electricity/monthly/", accessed=dt.date(2026, 9, 16))
D = dt.date(2026, 9, 16)


def sv(mag, unit):
    return Value(quantity=Q_(mag, unit), tag=Tag.SOURCED, source=SRC)


def make_site(**kw):
    base = dict(
        site_id="US-NM-ABQ", name="New Mexico", country="US", region="NM",
        currency=Currency.USD,
        power=PowerSupply(energy_price=sv(0.0541, "USD/kWh"), rate_basis="state_average"),
        labour=LabourRates(fully_loaded_operator=sv(36.92, "USD/hour")),
    )
    base.update(kw)
    return Site(**base)


def test_site_id_pattern():
    assert make_site().site_id == "US-NM-ABQ"
    for bad in ["USA-NM-ABQ", "US-NM", "us-nm-abq"]:
        with pytest.raises(Exception):
            make_site(site_id=bad)


def test_fx_requires_source_or_explicit_note():
    with pytest.raises(ValueError, match="NOT SOURCED"):
        ExchangeRate(base=Currency.USD, quote=Currency.INR, rate=88.0, as_of=D)
    ok = ExchangeRate(base=Currency.USD, quote=Currency.INR, rate=88.0, as_of=D,
                      note="indicative rate, NOT SOURCED to a fixing")
    assert ok.rate == 88.0


def test_fx_conversion_is_explicit_and_dated():
    fx = ExchangeRate(base=Currency.USD, quote=Currency.INR, rate=88.0, as_of=D,
                      note="indicative, NOT SOURCED")
    out = fx.convert(Q_(100.0, "USD/tonne"))
    assert out.magnitude == pytest.approx(8800.0)
    assert "INR" in str(out.units)
    with pytest.raises(ValueError, match="not denominated"):
        fx.convert(Q_(100.0, "INR/tonne"))


def test_currencies_still_do_not_auto_convert():
    with pytest.raises(DimensionalityError):
        Q_(1.0, "USD").to("INR")


def test_contract_power_rate_must_be_flagged():
    with pytest.raises(ValueError, match="cannot be independently verified"):
        PowerSupply(energy_price=sv(0.0444, "USD/kWh"), rate_basis="bilateral_contract")
    ok = PowerSupply(energy_price=sv(0.0444, "USD/kWh"), rate_basis="bilateral_contract",
                     note="bilateral utility contract at an existing smelter, not a "
                          "published state average")
    assert ok.note


def test_discretionary_grant_cannot_be_certain():
    with pytest.raises(ValueError, match="never certain"):
        Incentive(name="DOE grant", kind="grant", value=sv(10e6, "USD"),
                  award_probability=1.0, statute="DOE funding opportunity")
    ok = Incentive(name="45X", kind="tax_credit", value=sv(10e6, "USD"),
                   award_probability=1.0, statute="26 USC 45X")
    assert ok.award_probability == 1.0


def test_expected_incentive_is_probability_weighted():
    s = make_site(incentives=(
        Incentive(name="A", kind="grant", value=sv(10e6, "USD"), award_probability=0.3,
                  statute="prog A"),
        Incentive(name="B", kind="tax_credit", value=sv(4e6, "USD"), award_probability=1.0,
                  statute="26 USC 45X"),
    ))
    assert s.expected_incentive_value().magnitude == pytest.approx(0.3 * 10e6 + 4e6)


def test_incentive_currency_mismatch_raises():
    s = make_site(incentives=(Incentive(name="PLI", kind="grant", value=sv(1e8, "INR"),
                                        award_probability=0.5, statute="PLI scheme"),))
    with pytest.raises(ValueError, match="convert explicitly"):
        s.expected_incentive_value()


def test_tariff_lookup_by_origin_and_hs():
    s = make_site(trade_measures=(
        TradeMeasure(name="Section 301", ad_valorem_rate=0.10, applies_to_origin="IN",
                     hs_codes=("2506", "2505"), instrument="91 FR 47318",
                     effective=dt.date(2026, 7, 28)),))
    assert s.tariff_on("IN", "250610") == pytest.approx(0.10)
    assert s.tariff_on("IN", "700231") == pytest.approx(0.0)
    assert s.tariff_on("NO", "250610") == pytest.approx(0.0)


def test_reagent_not_locally_available_raises():
    r = ReagentPrices(prices={"HF": sv(1200.0, "USD/tonne")},
                      locally_available={"HF": False})
    with pytest.raises(ValueError, match="not locally available"):
        r.price("HF")
    with pytest.raises(KeyError, match="no delivered price"):
        r.price("HCl")


def test_productivity_held_separate_from_wage():
    """A low wage must not be silently credited with equal productivity."""
    l = LabourRates(fully_loaded_operator=sv(2.0, "USD/hour"), productivity_factor=0.6)
    assert l.productivity_factor == 0.6


def test_construction_index_separate_from_fx():
    s = make_site(construction_cost_index=0.55)
    assert s.construction_cost_index == 0.55
    assert s.country_risk_premium == 0.0


def test_observed_freight_basis_recorded():
    link = LogisticsLink(origin="IN-Mundra", destination="US-Charleston",
                         cost_per_tonne=sv(95.0, "USD/tonne"), mode="ocean",
                         basis="observed")
    assert link.basis == "observed"
