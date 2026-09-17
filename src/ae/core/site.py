"""SITE: a place a plant could be built, described well enough to cost it.

Every economic and operational model takes a SITE, so comparing Telangana with
New Mexico is a matter of swapping objects rather than editing code.

Currency handling
-----------------
Costs carry their own currency as a unit (USD or INR), and those units do not
auto-convert (see :mod:`ae.core.units`). Conversion goes through
:class:`ExchangeRate`, which carries a date and a source, because an FX rate is
a dated economic assumption and a model that silently converts at an
undocumented rate cannot be audited. Purchasing power and construction cost
adjustments are held as separate explicit factors, never folded into the FX rate.

Tariffs and incentives
----------------------
Both are modelled as uncertain: an incentive carries an award probability and a
timing distribution rather than being booked as certain revenue, and a tariff
carries the instrument that imposed it so its current status can be re-verified.
Policies change, and a model that treats a credit as guaranteed overstates every
downstream return.
"""

from __future__ import annotations

import datetime as _dt
import enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae.core.provenance import MISSING, Source, Value, _Missing
from ae.core.units import Q_, Quantity

__all__ = [
    "Currency",
    "ExchangeRate",
    "Incentive",
    "LabourRates",
    "LogisticsLink",
    "PermittingRegime",
    "PowerSupply",
    "ReagentPrices",
    "Site",
    "TradeMeasure",
]


class Currency(str, enum.Enum):
    USD = "USD"
    INR = "INR"


class ExchangeRate(BaseModel):
    """A dated FX assumption with a source and a scenario range.

    Deliberately NOT a pint conversion: unit systems have no notion of the date
    or the source of a rate, and an undated rate in a twenty-year NPV is a hidden
    assumption worth several percent of the answer.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    base: Currency
    quote: Currency
    rate: float = Field(gt=0, description="units of quote per one unit of base")
    as_of: _dt.date
    source: Source | None = None
    low: float | None = Field(default=None, gt=0, description="scenario low")
    high: float | None = Field(default=None, gt=0, description="scenario high")
    note: str | None = None

    @model_validator(mode="after")
    def _sane(self) -> ExchangeRate:
        if self.base == self.quote:
            raise ValueError("base and quote currencies must differ")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("low exceeds high")
        if self.source is None and self.note is None:
            raise ValueError(
                "an exchange rate must carry a source or an explicit note marking it "
                "NOT SOURCED, so an undocumented rate cannot silently set the answer"
            )
        return self

    def convert(self, q: Quantity) -> Quantity:
        """Convert a cost from base currency to quote currency."""
        unit = str(q.units)
        if self.base.value not in unit:
            raise ValueError(f"{unit} is not denominated in {self.base.value}")
        return Q_(q.magnitude * self.rate, unit.replace(self.base.value, self.quote.value))


class PowerSupply(BaseModel):
    """Electricity, the dominant operating cost for reduction and fusion routes."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    energy_price: Value                       # e.g. USD/kWh or INR/kWh
    demand_charge: Value | _Missing = MISSING  # per kVA or kW per month
    rate_basis: Literal["published_tariff", "state_average", "bilateral_contract", "ppa"]
    carbon_intensity: Value | _Missing = MISSING   # kg CO2 per kWh
    reliability_hours_lost: Value | _Missing = MISSING  # hours per year of outage
    voltage_class: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _contract_rates_are_flagged(self) -> PowerSupply:
        if self.rate_basis in ("bilateral_contract", "ppa") and not self.note:
            raise ValueError(
                "a bilateral contract or PPA rate is not a published average and must "
                "carry a note saying so, since it cannot be independently verified"
            )
        return self


class LabourRates(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fully_loaded_operator: Value              # currency per hour or per month
    fully_loaded_engineer: Value | _Missing = MISSING
    productivity_factor: float = Field(
        default=1.0, gt=0,
        description="Output per worker-hour relative to the reference site. Held "
                    "separate from wage rate so a low wage is not silently "
                    "credited with equal productivity.")
    loading_multiplier: float | None = Field(
        default=None, gt=1.0,
        description="Ratio of fully loaded to bare wage, if built up rather than sourced")


class ReagentPrices(BaseModel):
    """Delivered reagent prices. Availability is separate from price: HF is not
    freely available everywhere, and a low quoted price with no local supplier is
    not a usable input."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    prices: dict[str, Value] = Field(default_factory=dict)
    locally_available: dict[str, bool] = Field(default_factory=dict)

    def price(self, reagent: str) -> Quantity:
        if reagent not in self.prices:
            raise KeyError(
                f"no delivered price for {reagent!r} at this site; available: "
                f"{sorted(self.prices)}"
            )
        if self.locally_available.get(reagent) is False:
            raise ValueError(
                f"{reagent} is priced but marked not locally available at this site; "
                f"import lead time and permitting must be modelled before use"
            )
        return self.prices[reagent].quantity


class LogisticsLink(BaseModel):
    """One freight lane. Observed lane costs are preferred over quoted rates."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    origin: str
    destination: str
    cost_per_tonne: Value
    mode: Literal["ocean", "rail", "road", "multimodal"]
    transit_days: Value | _Missing = MISSING
    basis: Literal["observed", "quoted", "estimated"] = "estimated"


class TradeMeasure(BaseModel):
    """A tariff or trade restriction, with the instrument that imposed it."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    ad_valorem_rate: float = Field(ge=0.0, le=2.0)
    applies_to_origin: str = Field(min_length=2, max_length=2)
    hs_codes: tuple[str, ...] = ()
    instrument: str = Field(
        min_length=5,
        description="Citation of the legal instrument, e.g. a Federal Register number, "
                    "so current status can be re-verified before use")
    effective: _dt.date | None = None
    verified_on: _dt.date | None = None


class Incentive(BaseModel):
    """A credit, grant or loan modelled as uncertain, never as certain revenue."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    kind: Literal["tax_credit", "grant", "loan_guarantee", "rebate", "accelerated_depreciation"]
    value: Value
    award_probability: float = Field(
        ge=0.0, le=1.0,
        description="Probability of actually receiving it. 1.0 is allowed only for a "
                    "statutory entitlement with no discretionary award step.")
    expected_delay_months: Value | _Missing = MISSING
    statute: str = Field(min_length=3, description="Authorising statute or program citation")
    verified_on: _dt.date | None = None
    conditions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _certainty_must_be_justified(self) -> Incentive:
        if self.award_probability == 1.0 and self.kind in ("grant", "loan_guarantee"):
            raise ValueError(
                f"{self.kind} cannot have award_probability 1.0: a discretionary award "
                f"is never certain, and booking it as certain overstates returns"
            )
        return self


class PermittingRegime(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    jurisdiction: str
    expected_months: Value | _Missing = MISSING
    key_permits: tuple[str, ...] = ()
    effluent_limits: dict[str, Value] = Field(default_factory=dict)
    occupational_limits: dict[str, Value] = Field(default_factory=dict)
    note: str | None = None


class Site(BaseModel):
    """One candidate plant location.

    ``site_id`` follows ``COUNTRY-REGION-TAG``, e.g. ``IN-TG-VKB`` or ``US-NM-ABQ``.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    site_id: str = Field(pattern=r"^[A-Z]{2}-[A-Z0-9]{2,3}-[A-Z0-9]{2,6}$")
    name: str
    country: str = Field(min_length=2, max_length=2)
    region: str
    currency: Currency
    power: PowerSupply
    labour: LabourRates
    reagents: ReagentPrices = Field(default_factory=ReagentPrices)
    logistics: tuple[LogisticsLink, ...] = ()
    trade_measures: tuple[TradeMeasure, ...] = ()
    incentives: tuple[Incentive, ...] = ()
    permitting: PermittingRegime | None = None
    construction_cost_index: float = Field(
        default=1.0, gt=0,
        description="Multiplier on installed capital cost relative to the reference "
                    "site. Held separate from FX so a currency move is not mistaken "
                    "for a change in construction cost.")
    water_cost: Value | _Missing = MISSING
    natural_gas_price: Value | _Missing = MISSING
    tax_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    country_risk_premium: float = Field(
        default=0.0, ge=0.0, le=0.30,
        description="Added to the discount rate. Stated separately so the hurdle "
                    "rate's composition is visible.")
    note: str | None = None

    def tariff_on(self, origin: str, hs_code: str | None = None) -> float:
        """Total ad valorem rate applying to goods of ``origin`` arriving here."""
        total = 0.0
        for m in self.trade_measures:
            if m.applies_to_origin != origin:
                continue
            if hs_code and m.hs_codes and not any(hs_code.startswith(c) for c in m.hs_codes):
                continue
            total += m.ad_valorem_rate
        return total

    def expected_incentive_value(self, currency: Currency | None = None) -> Quantity:
        """Probability-weighted incentive total, in this site's currency.

        Weighting by award probability is the point: the unweighted sum is the
        number a promoter quotes, not the number a model should carry.
        """
        cur = (currency or self.currency).value
        total = 0.0
        for inc in self.incentives:
            q = inc.value.quantity
            if cur not in str(q.units):
                raise ValueError(
                    f"incentive {inc.name!r} is denominated in {q.units}, not {cur}; "
                    f"convert explicitly through an ExchangeRate with a date"
                )
            total += q.magnitude * inc.award_probability
        return Q_(total, cur)
