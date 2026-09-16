"""Factored capex: scaling, escalation vs location, reconciliation, accuracy band."""
import datetime as dt
import pytest
from ae.core.units import Q_, DimensionalityError
from ae.core.provenance import Tag, Tier, Source, Value
from ae.core.site import Currency, PowerSupply, LabourRates, Site
from ae.econ.capex import (
    Equipment, scale_cost, escalate_cost, estimate_capex, SCALING_RANGE,
)

SRC = Source(citation="Vendor quotation, redacted", tier=Tier.T2,
             url="https://example.invalid/quote", accessed=dt.date(2026, 9, 16))


def sv(mag, unit, tag=Tag.SOURCED):
    return Value(quantity=Q_(mag, unit), tag=tag, source=SRC if tag == Tag.SOURCED else None,
                 basis=None if tag == Tag.SOURCED else "class-5 factored estimate")


def site(cci=1.0, cur=Currency.USD, **kw):
    u = cur.value
    base = dict(site_id="US-NM-ABQ", name="New Mexico", country="US", region="NM",
                currency=cur, construction_cost_index=cci,
                power=PowerSupply(energy_price=sv(0.0541, f"{u}/kWh"),
                                  rate_basis="state_average"),
                labour=LabourRates(fully_loaded_operator=sv(36.92, f"{u}/hour")))
    base.update(kw)
    return Site(**base)


def kit():
    return [
        Equipment("attrition scrubber", sv(1_200_000.0, "USD"), 2.4,
                  size=Q_(5000.0, "tonne/year"), size_basis="feed"),
        Equipment("WHIMS", sv(850_000.0, "USD"), 2.1),
        Equipment("leach train", sv(2_400_000.0, "USD"), 3.2),
        Equipment("rotary kiln", sv(3_100_000.0, "USD", Tag.ASSUMED), 2.8),
    ]


@pytest.mark.golden
def test_scale_cost_worked_example():
    """2.0 MUSD at 5,000 t/yr to 20,000 t/yr at n = 0.6.
       ratio = 4.0, 4.0**0.6 = 2.29740, cost = 4,594,793 USD.
    Note the whole content of the rule is the exponent: at n = 0.9 the same
    ratio gives 4.0**0.9 = 3.48220 and 6,964,405 USD, 52 percent higher."""
    c = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
                   Q_(20000.0, "tonne/year"), 0.6)
    assert c.to("USD").magnitude == pytest.approx(4_594_793.0, abs=1.0)
    c9 = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
                    Q_(20000.0, "tonne/year"), 0.9)
    assert c9.to("USD").magnitude == pytest.approx(6_964_405.0, abs=1.0)
    assert c9 / c == pytest.approx(4.0 ** 0.3, rel=1e-9)


def test_scaling_is_sublinear_so_bigger_is_cheaper_per_tonne():
    small = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
                       Q_(5000.0, "tonne/year"), 0.6)
    big = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
                     Q_(20000.0, "tonne/year"), 0.6)
    assert big > small
    # Per annual tonne, the larger plant is cheaper: that is the economy of scale.
    assert (big.magnitude / 20000.0) < (small.magnitude / 5000.0)


def test_exponent_outside_supported_range_warns():
    assert SCALING_RANGE == (0.4, 0.9)
    with pytest.warns(UserWarning, match="outside the supported range"):
        scale_cost(Q_(1e6, "USD"), Q_(1.0, "tonne/year"), Q_(2.0, "tonne/year"), 1.2)


def test_extrapolation_warns_then_raises():
    with pytest.warns(UserWarning, match="exceeds 10x"):
        scale_cost(Q_(1e6, "USD"), Q_(1.0, "tonne/year"), Q_(20.0, "tonne/year"), 0.6)
    with pytest.raises(ValueError, match="does not support"):
        scale_cost(Q_(1e6, "USD"), Q_(1.0, "tonne/year"), Q_(500.0, "tonne/year"), 0.6)
    # Also in the shrinking direction.
    with pytest.raises(ValueError, match="does not support"):
        scale_cost(Q_(1e6, "USD"), Q_(500.0, "tonne/year"), Q_(1.0, "tonne/year"), 0.6)


def test_incommensurable_sizes_rejected():
    with pytest.raises(DimensionalityError):
        scale_cost(Q_(1e6, "USD"), Q_(5000.0, "tonne/year"), Q_(20.0, "m**2"), 0.6)


@pytest.mark.golden
def test_escalation_worked_example():
    """A CEPCI-style move from 708 to 800: 1.0 MUSD x 800/708 = 1,129,943.50."""
    assert escalate_cost(Q_(1.0e6, "USD"), 708.0, 800.0).to("USD").magnitude == \
        pytest.approx(1_129_943.50, abs=0.01)
    # Deflation works too, and round-trips.
    back = escalate_cost(escalate_cost(Q_(1.0e6, "USD"), 708.0, 800.0), 800.0, 708.0)
    assert back.to("USD").magnitude == pytest.approx(1.0e6, rel=1e-12)


def test_escalation_and_location_are_separate_axes():
    """Escalation moves a cost through TIME at fixed location; the location
    factor moves it through SPACE at fixed time. Collapsing them is a silent
    error of tens of percent, so the estimate reports both separately."""
    eq = kit()
    a = estimate_capex(eq, site(cci=1.0), 0.30, 0.15, base_index=708.0, target_index=800.0)
    b = estimate_capex(eq, site(cci=800.0 / 708.0), 0.30, 0.15)
    # Same total by construction, but the composition differs and is visible.
    assert a.total_project_cost.magnitude == pytest.approx(b.total_project_cost.magnitude,
                                                           rel=1e-9)
    assert (a.index_ratio, a.location_factor) == (pytest.approx(800 / 708), 1.0)
    assert (b.index_ratio, b.location_factor) == (1.0, pytest.approx(800 / 708))


def test_one_sided_index_is_not_an_escalation():
    with pytest.raises(ValueError, match="BOTH base_index and target_index"):
        estimate_capex(kit(), site(), 0.30, 0.15, base_index=708.0)


@pytest.mark.golden
def test_factored_estimate_worked_example():
    """Purchased 7.55 MUSD across four items. Installed:
         scrubber 1.20 x 2.4 = 2.880
         WHIMS    0.85 x 2.1 = 1.785
         leach    2.40 x 3.2 = 7.680
         kiln     3.10 x 2.8 = 8.680
         total installed     = 21.025 MUSD
       indirects at 30 percent          =  6.3075
       contingency at 15 percent of both=  4.0999 (0.15 x 27.3325)
       total project cost               = 31.4324 MUSD
    """
    est = estimate_capex(kit(), site(), indirect_factor=0.30, contingency_fraction=0.15,
                         capacity=Q_(5000.0, "tonne/year"))
    assert est.total_purchased.to("USD").magnitude == pytest.approx(7_550_000.0, abs=1.0)
    assert est.total_installed.to("USD").magnitude == pytest.approx(21_025_000.0, abs=1.0)
    assert est.indirect_cost.to("USD").magnitude == pytest.approx(6_307_500.0, abs=1.0)
    assert est.contingency.to("USD").magnitude == pytest.approx(4_099_875.0, abs=1.0)
    assert est.total_project_cost.to("USD").magnitude == pytest.approx(31_432_375.0, abs=1.0)
    assert est.reconciles()


def test_accuracy_band_is_reported_not_optional():
    """A factored estimate is AACE Class 4-5, minus 30 to plus 50 percent. The
    point estimate alone is not a usable answer."""
    est = estimate_capex(kit(), site(), 0.30, 0.15)
    lo, hi = est.accuracy_band()
    t = est.total_project_cost.magnitude
    assert lo.magnitude == pytest.approx(0.70 * t)
    assert hi.magnitude == pytest.approx(1.50 * t)
    # The band spans more than 2x, which is the honest statement of precision.
    assert hi.magnitude / lo.magnitude > 2.0


def test_capex_per_annual_tonne():
    est = estimate_capex(kit(), site(), 0.30, 0.15, capacity=Q_(5000.0, "tonne/year"))
    per_t = est.capex_per_annual_tonne
    assert per_t is not None
    assert per_t.to("USD*year/tonne").magnitude == pytest.approx(31_432_375.0 / 5000.0,
                                                                 rel=1e-9)
    assert estimate_capex(kit(), site(), 0.30, 0.15).capex_per_annual_tonne is None


def test_assumed_share_flags_unsourced_equipment():
    """The kiln is the single largest item and is ASSUMED, which a reader should
    not have to discover by reading the equipment list."""
    est = estimate_capex(kit(), site(), 0.30, 0.15)
    assert est.assumed_share == pytest.approx(3_100_000.0 / 7_550_000.0, rel=1e-9)
    assert est.assumed_share > 0.40


def test_installation_factor_guards():
    with pytest.raises(ValueError, match="below 1.0"):
        Equipment("x", sv(1e6, "USD"), 0.35)
    with pytest.raises(ValueError, match="exceeds 6.0"):
        Equipment("x", sv(1e6, "USD"), 8.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        Equipment("x", sv(-1.0, "USD"), 2.0)


def test_currency_mismatch_rejected():
    inr = [Equipment("mill", sv(1e8, "INR"), 2.0)]
    with pytest.raises(ValueError, match="dated ExchangeRate"):
        estimate_capex(inr, site(), 0.30, 0.15)


def test_estimate_input_guards():
    with pytest.raises(ValueError, match="equipment list is required"):
        estimate_capex([], site(), 0.30, 0.15)
    dup = [Equipment("m", sv(1e6, "USD"), 2.0), Equipment("m", sv(2e6, "USD"), 2.0)]
    with pytest.raises(ValueError, match="duplicate equipment"):
        estimate_capex(dup, site(), 0.30, 0.15)
    with pytest.raises(ValueError, match="indirect_factor"):
        estimate_capex(kit(), site(), 1.5, 0.15)
    with pytest.raises(ValueError, match="contingency_fraction"):
        estimate_capex(kit(), site(), 0.3, -0.1)


def test_no_automatic_contingency():
    """Contingency is a decision, not a calculation: zero must be expressible
    and must be visible as zero rather than defaulted silently."""
    est = estimate_capex(kit(), site(), 0.30, 0.0)
    assert est.contingency.magnitude == 0.0
    assert est.reconciles()
    assert ("contingency", 0.0) in est.lines


def test_records_carry_provenance_per_item():
    est = estimate_capex(kit(), site(), 0.30, 0.15)
    rows = {r["item"]: r for r in est.to_records()}
    assert rows["rotary kiln"]["tag"] == "ASSUMED"
    assert rows["WHIMS"]["tag"] == "SOURCED"
    assert rows["WHIMS"]["source"] == "Vendor quotation, redacted"
    assert rows["leach train"]["installed"] == pytest.approx(7_680_000.0)
    assert rows["total project cost"]["class"] == "aggregate"
