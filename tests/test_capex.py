"""Factored capex: scaling, escalation vs location, reconciliation, accuracy band."""
import dataclasses as dc
import datetime as dt

import pytest

from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.site import Currency, LabourRates, PowerSupply, Site
from ae.core.units import Q_, DimensionalityError
from ae.econ.capex import (
    SCALING_RANGE,
    Equipment,
    escalate_cost,
    estimate_capex,
    exponent_provenance,
    scale_cost,
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
    # Every factor in the docstring chain, asserted.
    ratio = 20000.0 / 5000.0
    assert ratio == 4.0
    assert 4.0 ** 0.6 == pytest.approx(2.29740, abs=1e-5)
    assert 4.0 ** 0.9 == pytest.approx(3.48220, abs=1e-5)
    assert 2.0e6 * 4.0 ** 0.6 == pytest.approx(4_594_793, abs=1.0)
    assert 2.0e6 * 4.0 ** 0.9 == pytest.approx(6_964_405, abs=1.0)
    assert (4.0 ** 0.9 / 4.0 ** 0.6 - 1.0) * 100 == pytest.approx(52, abs=0.5)
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
    # Every line of the factored chain, asserted so no intermediate can drift.
    purchased = [1.20, 0.85, 2.40, 3.10]
    factors = [2.4, 2.1, 3.2, 2.8]
    assert sum(purchased) == pytest.approx(7.55, abs=1e-9)
    lines_ = [p * f for p, f in zip(purchased, factors)]
    assert lines_[0] == pytest.approx(2.880, abs=1e-9)
    assert lines_[1] == pytest.approx(1.785, abs=1e-9)
    assert lines_[2] == pytest.approx(7.680, abs=1e-9)
    assert lines_[3] == pytest.approx(8.680, abs=1e-9)
    installed_ = sum(lines_)
    assert installed_ == pytest.approx(21.025, abs=1e-9)
    indirects_ = 0.30 * installed_
    assert indirects_ == pytest.approx(6.3075, abs=1e-9)
    base_ = installed_ + indirects_
    assert base_ == pytest.approx(27.3325, abs=1e-9)
    conting_ = 0.15 * base_
    assert conting_ == pytest.approx(4.0999, abs=1e-4)
    assert base_ + conting_ == pytest.approx(31.4324, abs=1e-4)
    est = estimate_capex(kit(), site(), indirect_factor=0.30, contingency_fraction=0.15,
                         capacity=Q_(5000.0, "tonne/year"))
    assert est.total_purchased.to("USD").magnitude == pytest.approx(7_550_000.0, abs=1.0)
    assert est.total_installed.to("USD").magnitude == pytest.approx(21_025_000.0, abs=1.0)
    assert est.indirect_cost.to("USD").magnitude == pytest.approx(6_307_500.0, abs=1.0)
    assert est.contingency.to("USD").magnitude == pytest.approx(4_099_875.0, abs=1.0)
    assert est.total_project_cost.to("USD").magnitude == pytest.approx(31_432_375.0, abs=1.0)
    assert est.reconciles()


def test_accuracy_band_defaults_to_class_5_not_class_4():
    """A factored estimate from an equipment list with no flowsheet engineering
    is AACE CLASS 5, minus 50 to plus 100 percent (multipliers 0.50 and 2.00,
    since 2.00 - 1 = +100 percent).

    The previous default was the Class 4 band (minus 30 to plus 50) while the
    docstring described the estimate as "Class 4 to 5", which quoted the
    narrower band for work at the wider class. Class 4 presumes preliminary
    flowsheets and some vendor pricing; at concept stage neither exists, and
    claiming it understates the upside by a factor of two.
    """
    est = estimate_capex(kit(), site(), 0.30, 0.15)
    assert est.estimate_class == 5
    # The band as PERCENTAGES, which is how the docstring states it.
    lo_m, hi_m = est.accuracy
    assert (lo_m - 1) * 100 == pytest.approx(-50.0, abs=1e-9)
    assert (hi_m - 1) * 100 == pytest.approx(100.0, abs=1e-9)
    lo, hi = est.accuracy_band()
    t = est.total_project_cost.magnitude
    assert lo.magnitude == pytest.approx(0.50 * t)
    assert hi.magnitude == pytest.approx(2.00 * t)
    # A 4x span is the honest statement of concept-stage precision.
    assert hi.magnitude / lo.magnitude == pytest.approx(4.0)
    # Class 4 would have claimed roughly half that span.
    c4 = dc.replace(est, estimate_class=4)
    lo4, hi4 = c4.accuracy_band()
    assert (hi4 / lo4).magnitude == pytest.approx(1.50 / 0.70)
    assert hi.magnitude > hi4.magnitude


def test_declaring_a_tighter_class_requires_declaring_it():
    """The class is a statement about engineering maturity, so it must be set
    explicitly and the band follows from it."""
    est = estimate_capex(kit(), site(), 0.30, 0.15)
    spans = {}
    for cls in (5, 4, 3, 2, 1):
        e = dc.replace(est, estimate_class=cls)
        lo, hi = e.accuracy_band()
        spans[cls] = hi.magnitude / lo.magnitude
    # Bands must tighten monotonically as definition maturity rises.
    assert spans[5] > spans[4] > spans[3] > spans[2] > spans[1]
    assert spans[1] == pytest.approx(1.15 / 0.90)


def test_accuracy_note_states_what_the_class_presumes():
    est = estimate_capex(kit(), site(), 0.30, 0.15)
    note = est.accuracy_note()
    assert "Class 5" in note
    assert "-50" in note and "+100" in note
    assert "no flowsheet engineering" in note
    c3 = dc.replace(est, estimate_class=3).accuracy_note()
    assert "P&IDs" in c3 and "Class 3" in c3


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


def test_assumed_share_is_invariant_to_location_and_escalation():
    """Regression. assumed_share previously divided RAW assumed costs by an
    escalated and location-factored total, so at a 0.55 location factor an
    all-assumed list reported 182 percent. The factors are common to every item
    and must cancel."""
    assert round(1.0 / 0.55 * 100) == 182
    # The erroneous result reproduced from its mechanism: raw over factored is
    # 1/0.55 = 181.8 percent, which is the "182 percent" the docstring records.
    assert 1.0 / 0.55 * 100 == pytest.approx(181.8, abs=0.1)
    shares = []
    for cci in (0.55, 1.0, 1.85):
        est = estimate_capex(kit(), site(cci=cci), 0.30, 0.15)
        shares.append(est.assumed_share)
        assert 0.0 <= est.assumed_share <= 1.0, f"cci {cci}: {est.assumed_share}"
    assert shares[0] == pytest.approx(shares[1]) == pytest.approx(shares[2])
    # With escalation applied as well.
    est = estimate_capex(kit(), site(cci=0.55), 0.30, 0.15,
                         base_index=708.0, target_index=800.0)
    assert est.assumed_share == pytest.approx(3_100_000.0 / 7_550_000.0, rel=1e-9)
    # An all-ASSUMED list is exactly 1.0, never above it.
    allassumed = [Equipment("a", sv(1e6, "USD", Tag.ASSUMED), 2.0),
                  Equipment("b", sv(2e6, "USD", Tag.ASSUMED), 2.0)]
    assert estimate_capex(allassumed, site(cci=0.55), 0.30, 0.15).assumed_share == \
        pytest.approx(1.0)


def test_scaling_exponent_carries_provenance():
    """The exponent is the entire content of the six-tenths relation, so an
    untagged one must not masquerade as a sourced figure. A bare float reports
    ASSUMED, which is the honest default: the rule is a convention, not a
    measurement of this equipment."""
    assert exponent_provenance(0.6) == Tag.ASSUMED
    tagged = Value(quantity=Q_(0.72, "dimensionless"), tag=Tag.SOURCED, source=SRC)
    assert exponent_provenance(tagged) == Tag.SOURCED
    # A tagged exponent gives an identical number to the bare float.
    a = scale_cost(Q_(2e6, "USD"), Q_(5e3, "tonne/year"), Q_(2e4, "tonne/year"), 0.72)
    b = scale_cost(Q_(2e6, "USD"), Q_(5e3, "tonne/year"), Q_(2e4, "tonne/year"), tagged)
    assert a.magnitude == pytest.approx(b.magnitude, rel=1e-12)
    # Range validation applies to the tagged form too.
    bad = Value(quantity=Q_(1.4, "dimensionless"), tag=Tag.ASSUMED, basis="test")
    with pytest.warns(UserWarning, match="outside the supported range"):
        scale_cost(Q_(1e6, "USD"), Q_(1.0, "tonne/year"), Q_(2.0, "tonne/year"), bad)


def test_equipment_records_which_exponent_produced_the_cost():
    """A scaled estimate must record the exponent AND its origin, or a reader
    cannot tell a vendor curve from a textbook default."""
    eq = [Equipment("kiln", sv(3.1e6, "USD", Tag.ASSUMED), 2.8,
                    scaling_exponent=Value(quantity=Q_(0.65, "dimensionless"),
                                           tag=Tag.SOURCED, source=SRC)),
          Equipment("WHIMS", sv(850_000.0, "USD"), 2.1, scaling_exponent=0.6),
          Equipment("tank", sv(100_000.0, "USD"), 1.4)]
    rows = {r["item"]: r for r in estimate_capex(eq, site(), 0.30, 0.15).to_records()}
    assert rows["kiln"]["scaling_exponent"] == pytest.approx(0.65)
    assert rows["kiln"]["scaling_exponent_tag"] == "SOURCED"
    assert rows["WHIMS"]["scaling_exponent"] == pytest.approx(0.6)
    assert rows["WHIMS"]["scaling_exponent_tag"] == "ASSUMED", "bare float is an assumption"
    assert rows["tank"]["scaling_exponent"] is None
    assert rows["total project cost"]["scaling_exponent"] is None


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
