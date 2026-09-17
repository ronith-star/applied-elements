"""DCF, IRR, LCOP: timing discipline and the failure modes of each metric."""
import numpy as np
import pytest

from ae.econ.valuation import (
    Project,
    breakeven_price,
    irr,
    levelized_cost,
    modified_irr,
    npv,
    payback_period,
)


def proj(**kw):
    base = dict(capex_schedule=[20e6, 20e6], construction_periods=2,
                ramp_fractions=[0.3, 0.7, 1.0], nameplate_tonnes=5000.0,
                price=3500.0, cash_cost_per_tonne=2000.0, life_periods=15,
                fixed_cost_per_period=3e6)
    base.update(kw)
    return Project(**base)


@pytest.mark.golden
def test_npv_worked_example():
    """Hand-traceable at 10 percent on cash flows -100, 60, 60.

    Each discount factor and term asserted, not just the total:
      (1.10)^1 = 1.10   ->  60/1.10  = 54.5455
      (1.10)^2 = 1.21   ->  60/1.21  = 49.5868
      -100 + 54.5455 + 49.5868       =  4.1322
    """
    assert 1.10 ** 1 == pytest.approx(1.1, abs=1e-12)
    assert 1.10 ** 2 == pytest.approx(1.21, abs=1e-12)
    t1, t2 = 60.0 / 1.1, 60.0 / 1.21
    assert t1 == pytest.approx(54.5455, abs=1e-4)
    assert t2 == pytest.approx(49.5868, abs=1e-4)
    assert -100.0 + t1 + t2 == pytest.approx(4.1322, abs=1e-4)
    # And the module must reproduce the same sum.
    assert npv(0.10, [-100.0, 60.0, 60.0]) == pytest.approx(-100.0 + t1 + t2,
                                                            rel=1e-12)
    assert npv(0.10, [-100.0, 60.0, 60.0]) == pytest.approx(4.1322, abs=1e-4)
    assert npv(0.0, [-100.0, 60.0, 60.0]) == pytest.approx(20.0)
    # At the IRR, NPV is zero by definition.
    r = irr([-100.0, 60.0, 60.0])
    assert npv(r, [-100.0, 60.0, 60.0]) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.golden
def test_irr_worked_example():
    """-100, 60, 60 solves to a rate of 0.130662 (13.0662 percent).

    Verified two ways: the solver's root, and that discounting at the quoted
    rate nearly zeroes the NPV. The residual at 6 significant figures is
    5.02e-05, which is the rounding of the printed rate rather than solver
    error; the solver's own root zeroes it to 1e-9.
    """
    assert 0.130662 * 100 == pytest.approx(13.0662, abs=1e-9)
    r = irr([-100.0, 60.0, 60.0])
    assert r == pytest.approx(0.130662, abs=1e-6)
    quoted = 0.130662
    residual = -100.0 + 60.0 / (1 + quoted) + 60.0 / (1 + quoted) ** 2
    assert residual == pytest.approx(5.02e-05, abs=1e-7), \
        "the residual is the rounding of the quoted rate, not solver error"
    assert abs(residual) < 1e-4
    assert npv(r, [-100.0, 60.0, 60.0]) == pytest.approx(0.0, abs=1e-9)


def test_irr_returns_none_rather_than_a_misleading_number():
    """Three real failure modes of IRR, each returning None."""
    # Never turns positive: no root exists.
    assert irr([-100.0, -50.0, -20.0]) is None
    # No sign change at all (all positive): not an investment.
    assert irr([100.0, 50.0]) is None
    # Multiple sign changes AND multiple real roots: reporting one is misleading.
    # -100, 230, -132 has exact roots at 10 and 20 percent (the polynomial
    # -100x^2 + 230x - 132 factors at x = 1.1 and x = 1.2).
    multi = [-100.0, 230.0, -132.0]
    assert irr(multi) is None
    # MIRR is unique by construction and does return a value there.
    assert modified_irr(multi, 0.10, 0.10) == pytest.approx(0.10, abs=1e-9)
    # Two more, verified against numpy.roots: 25 percent and 400 percent; and
    # 0, 100, 200 percent.
    assert irr([-4000.0, 25000.0, -25000.0]) is None
    assert irr([-1000.0, 6000.0, -11000.0, 6000.0]) is None


def test_multiple_sign_changes_alone_do_not_suppress_a_unique_irr():
    """A stream can change sign twice and still have ONE real root, in which
    case the IRR is well defined and must be returned. -100, 460, -630, 288
    changes sign three times but its other two roots are complex (verified
    against numpy.roots), leaving a single real rate of 160.745 percent.
    Suppressing that would discard a valid answer."""
    # The unique real root, verified independently before the module is asked.
    import numpy as _np
    roots_ = _np.roots([288.0, -630.0, 460.0, -100.0])
    real_ = [r.real for r in roots_ if abs(r.imag) < 1e-9]
    assert len(real_) == 1, "exactly one real root"
    assert (1.0 / real_[0] - 1.0) * 100 == pytest.approx(160.745, abs=1e-2)
    r = irr([-100.0, 460.0, -630.0, 288.0])
    assert r == pytest.approx(1.607452, abs=1e-5)
    assert npv(r, [-100.0, 460.0, -630.0, 288.0]) == pytest.approx(0.0, abs=1e-8)


def test_mirr_avoids_the_reinvestment_assumption():
    flows = [-100.0, 60.0, 60.0]
    at_own_rate = modified_irr(flows, 0.130662, 0.130662)
    assert at_own_rate == pytest.approx(irr(flows), rel=1e-4)
    # Reinvesting at a lower rate than the IRR lowers the realised return.
    assert modified_irr(flows, 0.10, 0.04) < irr(flows)


def test_construction_period_zero_is_refused():
    """The predecessor error: capital at t=0 and revenue from year 1."""
    with pytest.raises(ValueError, match="38 percent"):
        proj(construction_periods=0)


@pytest.mark.golden
def test_timing_shift_reproduces_the_predecessor_error_magnitude():
    """Directional reproduction of the documented failure. Moving first revenue
    later, with NO change to any cost or price, must raise breakeven materially.
    In the predecessor analysis the equivalent correction moved a breakeven price
    from 3,722 to 5,144 USD/t, a 38.2 percent rise."""
    # The predecessor figures: 3,722 -> 5,144 USD/t is a 38.2 percent rise.
    assert (5144.0 / 3722.0 - 1.0) * 100 == pytest.approx(38.2, abs=0.05)
    assert 5144.0 - 3722.0 == pytest.approx(1422.0, abs=1e-9)
    fast = proj(construction_periods=1, ramp_fractions=[1.0])
    slow = proj(construction_periods=3, ramp_fractions=[0.3, 0.7, 1.0])
    b_fast = breakeven_price(fast, 0.12)
    b_slow = breakeven_price(slow, 0.12)
    assert b_slow > b_fast
    rise = (b_slow - b_fast) / b_fast
    print(f"\nbreakeven {b_fast:,.0f} -> {b_slow:,.0f} USD/t, a {rise * 100:.1f} percent "
          f"rise from timing alone")
    assert rise > 0.10, "timing must move breakeven by more than a rounding amount"


def test_no_revenue_during_construction():
    cf = proj().cash_flows()
    assert cf.revenue[0] == 0.0 and cf.revenue[1] == 0.0
    assert cf.output[0] == 0.0 and cf.output[1] == 0.0
    assert cf.first_revenue_period() == 2


def test_output_profile_follows_the_ramp_then_holds():
    cf = proj().cash_flows()
    q = cf.output
    assert q[2] == pytest.approx(0.3 * 5000.0)
    assert q[3] == pytest.approx(0.7 * 5000.0)
    assert q[4] == pytest.approx(5000.0)
    assert q[-1] == pytest.approx(5000.0), "holds at nameplate after the ramp"
    assert q.size == 2 + 15


def test_ramp_must_be_monotonic():
    with pytest.raises(ValueError, match="decreases"):
        proj(ramp_fractions=[0.8, 0.5, 1.0])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        proj(ramp_fractions=[0.5, 1.4])
    with pytest.raises(ValueError, match="never reaches rate"):
        proj(ramp_fractions=[0.2, 0.5, 0.8, 1.0], life_periods=3)


@pytest.mark.golden
def test_lcop_discounts_both_numerator_and_denominator():
    """Discounting costs but not output understates LCOP whenever output is
    back-loaded, which it always is under a ramp. Verified by computing the
    wrong way and showing the sign of the error."""
    p = proj(tax_rate=0.0)
    correct = levelized_cost(p, 0.12)
    cf = p.cash_flows()
    t = np.arange(cf.output.size)
    costs = cf.capex + (cf.revenue - cf.ebitda)
    wrong = float(np.sum(costs / (1.12 ** t))) / float(np.sum(cf.output))
    assert wrong < correct, "the undiscounted-denominator error understates LCOP"
    print(f"\nLCOP correct {correct:,.2f}, undiscounted-output error {wrong:,.2f} "
          f"({(wrong / correct - 1) * 100:.1f} percent low)")
    # At a zero discount rate the two agree, which is the degenerate check.
    assert levelized_cost(proj(tax_rate=0.0), 0.0) == pytest.approx(
        float(np.sum(costs)) / float(np.sum(cf.output)), rel=1e-9)


def test_breakeven_price_zeroes_the_npv():
    p = proj()
    b = breakeven_price(p, 0.12)
    import dataclasses as dc
    assert dc.replace(p, price=b).cash_flows().npv(0.12) == pytest.approx(0.0, abs=1.0)
    # And it sits above cash cost, since capital must be recovered.
    assert b > p.cash_cost_per_tonne


def test_breakeven_equals_lcop_without_tax_and_diverges_with_it():
    """Without tax, breakeven price and LCOP coincide. With tax they must not,
    because loss carry-forward makes NPV only piecewise linear in price."""
    no_tax = proj(tax_rate=0.0)
    assert breakeven_price(no_tax, 0.12) == pytest.approx(
        levelized_cost(no_tax, 0.12), rel=1e-4)
    taxed = proj(tax_rate=0.25)
    assert breakeven_price(taxed, 0.12) > levelized_cost(taxed, 0.12)


def test_tax_reduces_npv_and_loss_carryforward_is_applied():
    a = proj(tax_rate=0.0, price=4500.0)
    b = proj(tax_rate=0.30, price=4500.0)
    assert b.cash_flows().npv(0.12) < a.cash_flows().npv(0.12)
    # In ramp years with depreciation, taxable income can be negative: no tax.
    cf = b.cash_flows()
    assert cf.tax[2] == 0.0, "no tax in the first ramp year under depreciation"
    assert (cf.tax >= 0).all(), "tax is never negative; losses carry forward"


def test_depreciation_shelters_early_income_only():
    p = proj(tax_rate=0.30, depreciation_periods=5, price=5000.0)
    cf = p.cash_flows()
    dep = cf.depreciation
    assert dep[0] == 0.0, "no depreciation during construction"
    assert dep[2] == pytest.approx(40e6 / 5)
    assert dep[2 + 5] == 0.0, "depreciation ends after its life"
    assert float(dep.sum()) == pytest.approx(p.total_capex, rel=1e-9)


def test_fixed_cost_during_construction_changes_funding_need():
    """A plant under construction with zero fixed cost understates the raise."""
    a = proj(fixed_cost_in_construction=False)
    b = proj(fixed_cost_in_construction=True)
    assert b.cash_flows().peak_funding_requirement() < \
        a.cash_flows().peak_funding_requirement()
    assert a.cash_flows().ebitda[0] == 0.0
    assert b.cash_flows().ebitda[0] == pytest.approx(-3e6)


def test_peak_funding_exceeds_capex_when_opex_precedes_revenue():
    p = proj(fixed_cost_in_construction=True)
    need = -p.cash_flows().peak_funding_requirement()
    assert need > p.total_capex, "capex alone understates the money required"


def test_working_capital_builds_and_releases():
    p = proj(working_capital_fraction=0.15)
    cf = p.cash_flows()
    assert cf.working_capital[2] < 0, "builds as output starts"
    assert cf.working_capital[-1] > 0, "released at the end"
    assert float(cf.working_capital.sum()) == pytest.approx(0.0, abs=1e-6)


def test_salvage_is_zero_unless_supplied():
    assert proj().cash_flows().net_cash_flow[-1] < \
        proj(salvage=10e6).cash_flows().net_cash_flow[-1]
    a, b = proj().cash_flows(), proj(salvage=10e6).cash_flows()
    assert b.net_cash_flow[-1] - a.net_cash_flow[-1] == pytest.approx(10e6)


def test_payback_returns_none_when_it_never_pays_back():
    assert payback_period([-100.0, 10.0, 10.0]) is None
    # Interpolated within the crossing period: -100, 60, 60 crosses at 1.667.
    assert payback_period([-100.0, 60.0, 60.0]) == pytest.approx(1.6667, abs=1e-4)
    # Discounting delays payback.
    assert payback_period([-100.0, 60.0, 60.0], 0.10) > \
        payback_period([-100.0, 60.0, 60.0])


def test_capex_spread_beats_capex_at_time_zero():
    """Spreading capital over the build is worth real money, and a model that
    books it all at t=0 understates NPV."""
    lump = proj(capex_schedule=[40e6])
    spread = proj(capex_schedule=[20e6, 20e6])
    assert spread.cash_flows().npv(0.12) > lump.cash_flows().npv(0.12)
    assert lump.total_capex == spread.total_capex


def test_records_and_validation():
    recs = proj().cash_flows().to_records()
    assert len(recs) == 17
    assert set(recs[0]) == {"period", "output_tonnes", "revenue", "ebitda",
                            "depreciation", "tax", "capex", "working_capital",
                            "net_cash_flow"}
    with pytest.raises(ValueError, match="at least one period"):
        proj(capex_schedule=[])
    with pytest.raises(ValueError, match="cannot be negative"):
        proj(capex_schedule=[-1.0])
    with pytest.raises(ValueError, match="tax_rate"):
        proj(tax_rate=1.0)
    with pytest.raises(ValueError, match="nameplate_tonnes"):
        proj(nameplate_tonnes=0.0)
    with pytest.raises(ValueError, match="at or below -100 percent"):
        npv(-1.0, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Adversarial audit: depreciation, salvage tax and IRR bracket.
# Added by the econ/plant/ml audit track. Each test below was written to FAIL
# against the module as committed, and the measured failure is quoted in the
# docstring.
# ---------------------------------------------------------------------------


def test_depreciation_base_is_not_silently_truncated():
    """A depreciation life longer than the operating life loses capital allowance.

    The defect as committed: ``annual_dep = total_capex / dep_periods`` is
    written into ``dep[construction:construction + dep_periods]``, but that
    slice is clipped by the array length, so only ``life_periods`` of the
    schedule survive. With 1000 of capex over 20 depreciation periods and a
    5 period life, 50 per period is booked 5 times, 250 in total, and 750 of
    allowance is discarded with no error. Tax is then charged on 750 of income
    that a tax authority would have sheltered.

    Measured before the fix: dep summed to 250.0 against total_capex 1000.0.
    """
    p = Project(capex_schedule=[1000.0], construction_periods=1,
                ramp_fractions=[1.0], nameplate_tonnes=100.0, price=50.0,
                cash_cost_per_tonne=10.0, life_periods=5, tax_rate=0.30,
                depreciation_periods=20)
    cf = p.cash_flows()
    assert p.total_capex == pytest.approx(1000.0, abs=1e-9)
    # Whatever the chosen convention, allowance may not silently vanish.
    assert cf.depreciation.sum() == pytest.approx(1000.0, abs=1e-6), (
        f"depreciation sums to {cf.depreciation.sum()} against capex "
        f"{p.total_capex}: allowance was truncated by the array length"
    )
    # The convention adopted is the stated 20 period rate, 1000/20 = 50.0 per
    # period, with the unclaimed balance recognised on disposal in the final
    # period as a balancing allowance. Four periods run at rate and the fifth
    # carries 1000 - 4 x 50 = 800. An earlier draft of this test asserted
    # 1000/5 = 200 per period, which is a DIFFERENT convention (re-spreading
    # the base over the shorter life) and would have overstated the shelter in
    # every early period. That draft was wrong and is recorded here rather
    # than quietly replaced.
    assert 1000.0 / 20.0 == pytest.approx(50.0, abs=1e-9)
    assert cf.depreciation[1] == pytest.approx(50.0, abs=1e-6)
    assert cf.depreciation[4] == pytest.approx(50.0, abs=1e-6)
    assert 1000.0 - 4.0 * 50.0 == pytest.approx(800.0, abs=1e-9)
    assert cf.depreciation[-1] == pytest.approx(800.0, abs=1e-6)


def test_depreciation_cannot_precede_the_capital_spend():
    """Allowance taken before the asset is paid for is not a tax position.

    The defect as committed: depreciation starts at ``construction_periods``
    and runs for ``depreciation_periods`` regardless of WHEN capex is spent. A
    100 spend at t=0 and a 900 expansion at t=4, depreciated over 6 periods
    from t=2, books 166.67 per period from t=2, so cumulative allowance
    reaches 333.33 by t=3 against 100.00 actually spent.

    Measured before the fix: periods 2 and 3 had depreciation-to-date above
    capex-to-date.
    """
    p = Project(capex_schedule=[100.0, 0.0, 0.0, 0.0, 900.0],
                construction_periods=2, ramp_fractions=[1.0],
                nameplate_tonnes=100.0, price=100.0, cash_cost_per_tonne=10.0,
                life_periods=6, tax_rate=0.30, depreciation_periods=6)
    cf = p.cash_flows()
    assert 1000.0 / 6.0 == pytest.approx(166.6667, abs=1e-4)
    spent = np.cumsum(cf.capex)
    taken = np.cumsum(cf.depreciation)
    assert spent[3] == pytest.approx(100.0, abs=1e-9)
    bad = np.flatnonzero(taken > spent + 1e-9)
    assert bad.size == 0, (
        f"depreciation-to-date exceeds capex-to-date in periods {bad.tolist()}: "
        f"taken {taken[bad].tolist()} against spent {spent[bad].tolist()}"
    )


def test_salvage_is_taxed_against_its_book_value():
    """Selling a fully depreciated asset for 400 is a 400 taxable gain.

    The defect as committed: ``net[-1] += self.salvage`` adds the terminal
    value after tax has been computed, so a salvage receipt is untaxed however
    much allowance has already been claimed. With capex 1000 fully depreciated
    over the life, book value is 0, so the whole 400 is a gain and 0.30 x 400 =
    120 of tax is owed.

    Measured before the fix: tax in the final period was exactly
    0.30 x (ebitda - depreciation), with 0.0 attributable to the salvage.
    """
    kw = dict(capex_schedule=[1000.0], construction_periods=1,
              ramp_fractions=[1.0], nameplate_tonnes=100.0, price=100.0,
              cash_cost_per_tonne=10.0, life_periods=5, tax_rate=0.30,
              depreciation_periods=5)
    no_salvage = Project(salvage=0.0, **kw).cash_flows()
    with_salvage = Project(salvage=400.0, **kw).cash_flows()
    assert with_salvage.depreciation.sum() == pytest.approx(1000.0, abs=1e-6)
    assert 0.30 * 400.0 == pytest.approx(120.0, abs=1e-9)
    extra_tax = with_salvage.tax[-1] - no_salvage.tax[-1]
    assert extra_tax == pytest.approx(120.0, abs=1e-6), (
        f"salvage of 400 against zero book value added {extra_tax} of tax, "
        f"not 120.0: the terminal receipt bypasses the tax calculation"
    )
    # Net receipt is the after-tax 280, not the gross 400.
    assert 400.0 - 120.0 == pytest.approx(280.0, abs=1e-9)
    assert (with_salvage.net_cash_flow[-1]
            - no_salvage.net_cash_flow[-1]) == pytest.approx(280.0, abs=1e-6)


def test_irr_above_the_default_bracket_is_not_reported_as_undefined():
    """A 1900 percent return is a real IRR, and None reads as "no IRR exists".

    The defect as committed: the scan grid ends at the bracket's upper bound of
    10.0, so a stream whose only root lies above 1000 percent finds no sign
    change and ``irr`` returns None through the ``if not uniq`` branch, which
    the docstring reserves for streams that never turn positive. Flows
    -1.0, 20.0 have the single exact root 20/1 - 1 = 19.0.

    Measured before the fix: irr([-1.0, 20.0]) returned None with one sign
    change in the stream, so the no-root branch was reached on a stream that
    has exactly one root.
    """
    assert 20.0 / 1.0 - 1.0 == pytest.approx(19.0, abs=1e-12)
    got = irr([-1.0, 20.0])
    assert got is not None, (
        "irr([-1.0, 20.0]) returned None, but the stream has one sign change "
        "and the single root r = 19.0"
    )
    assert got == pytest.approx(19.0, rel=1e-6)
    assert npv(got, [-1.0, 20.0]) == pytest.approx(0.0, abs=1e-9)


def test_irr_reports_none_when_a_second_root_lies_outside_the_bracket():
    """Two roots is two roots, whether or not both fall inside the scan range.

    Flows built from roots at r = 0.20 and r = 50.0 have two sign changes. The
    scan grid stops at 10.0 and therefore sees only the first, so a single
    value is returned as "the" IRR on a stream that has two, which is the
    misleading answer the function's own docstring says it exists to avoid.

    Measured before the fix: irr returned 0.19999999999995005 on a stream with
    npv(0.2) = -1.1e-13 and npv(50.0) = -1.3e-15, both roots.
    """
    x1, x2 = 1.0 / 1.2, 1.0 / 51.0
    flows = [1000.0 * x1 * x2, -1000.0 * (x1 + x2), 1000.0]
    assert npv(0.2, flows) == pytest.approx(0.0, abs=1e-9)
    assert npv(50.0, flows) == pytest.approx(0.0, abs=1e-9)
    assert irr(flows) is None, (
        f"irr returned {irr(flows)} on a stream with roots at 0.2 and 50.0; "
        f"reporting one of two roots is the documented failure mode"
    )
    # MIRR is unique by construction and is the answer to offer instead.
    assert modified_irr(flows, 0.08, 0.08) is not None


def test_irr_multiplicity_is_detected_however_far_away_the_second_root_is():
    """The test that killed the first repair. A grid scan cannot do this.

    A first attempt at the bracket fix widened the upper bound to 1e4 and added
    a check comparing the NPV sign at the bracket edge. It passed the roots at
    0.20 and 50.0 case and looked correct. It was not: with the second root
    moved to 2.0e4 the same function returned 0.19999999999996645, because the
    sign of NPV at a finite bound carries no information about roots beyond it.
    A scan lower-bounds the root count and can never establish uniqueness, so
    the roots are now solved exactly from the companion matrix.

    Each root pair below is verified with numpy.roots before the module is
    asked, so the test does not depend on the implementation it audits.
    """
    for second in (50.0, 2.0e4, 1.0e5):
        x1, x2 = 1.0 / 1.2, 1.0 / (1.0 + second)
        flows = [1000.0 * x1 * x2, -1000.0 * (x1 + x2), 1000.0]
        indep = np.roots(np.asarray(flows, dtype=float)[::-1])
        real_rates = sorted(1.0 / r.real - 1.0 for r in indep
                            if abs(r.imag) < 1e-8 and r.real > 0.0)
        assert len(real_rates) == 2, f"fixture must have two real rates, got {real_rates}"
        assert real_rates[0] == pytest.approx(0.2, rel=1e-6)
        assert real_rates[1] == pytest.approx(second, rel=1e-6)
        assert irr(flows) is None, (
            f"irr returned {irr(flows)} on a stream with roots at 0.2 and "
            f"{second}; multiplicity must not depend on a scan window"
        )


def test_irr_refuses_a_bracket_rather_than_ignoring_it():
    """A silently ignored argument is worse than a removed one: a caller who
    passes bracket=(0.0, 0.5) believes the search was constrained."""
    with pytest.raises(ValueError, match="no longer scans a bracket"):
        irr([-100.0, 60.0, 60.0], bracket=(0.0, 0.5))
