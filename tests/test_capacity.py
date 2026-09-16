"""OEE decomposition, capacity and bottleneck identification."""
import pytest
from ae.core.units import Q_, DimensionalityError
from ae.plant.capacity import OEE, UnitCapacity, LineCapacity, assess_line, HOURS_PER_YEAR


@pytest.mark.golden
def test_oee_worked_example():
    """Hand-traceable. A = 0.90, P = 0.95, Q = 0.99.
       OEE = 0.90 x 0.95 x 0.99 = 0.84645
       Losses, in cascade order:
         availability 1 - 0.90                  = 0.10000
         performance  0.90 x (1 - 0.95)         = 0.04500
         quality      0.90 x 0.95 x (1 - 0.99)  = 0.00855
         sum of losses                          = 0.15355 = 1 - 0.84645
    """
    A_, P_, Q_o = 0.90, 0.95, 0.99
    assert A_ * P_ * Q_o == pytest.approx(0.84645, abs=1e-9)
    la_, lp_, lq_ = 1 - A_, A_ * (1 - P_), A_ * P_ * (1 - Q_o)
    assert la_ == pytest.approx(0.10000, abs=1e-9)
    assert lp_ == pytest.approx(0.04500, abs=1e-9)
    assert lq_ == pytest.approx(0.00855, abs=1e-9)
    assert la_ + lp_ + lq_ == pytest.approx(0.15355, abs=1e-9)
    assert la_ + lp_ + lq_ == pytest.approx(1 - 0.84645, abs=1e-9)
    A, P, Q = 0.90, 0.95, 0.99
    assert A * P * Q == pytest.approx(0.84645, abs=1e-9)
    l_a, l_p, l_q = 1 - A, A * (1 - P), A * P * (1 - Q)
    assert l_a == pytest.approx(0.10000, abs=1e-9)
    assert l_p == pytest.approx(0.04500, abs=1e-9)
    assert l_q == pytest.approx(0.00855, abs=1e-9)
    assert l_a + l_p + l_q == pytest.approx(0.15355, abs=1e-9)
    assert l_a + l_p + l_q == pytest.approx(1 - 0.84645, abs=1e-9)
    o = OEE(0.90, 0.95, 0.99)
    assert o.value == pytest.approx(0.84645, abs=1e-9)
    lb = o.loss_breakdown
    assert lb["availability"] == pytest.approx(0.10000, abs=1e-9)
    assert lb["performance"] == pytest.approx(0.04500, abs=1e-9)
    assert lb["quality"] == pytest.approx(0.00855, abs=1e-9)
    assert sum(lb.values()) == pytest.approx(1 - o.value, abs=1e-12)


def test_losses_do_not_double_count():
    """Naive (1-A)+(1-P)+(1-Q) overstates the loss; the cascade form must not."""
    o = OEE(0.80, 0.80, 0.80)
    naive = (1 - 0.80) * 3
    assert sum(o.loss_breakdown.values()) == pytest.approx(1 - o.value)
    assert sum(o.loss_breakdown.values()) < naive


@pytest.mark.parametrize("a,p,q", [(1.1, 0.9, 0.9), (0.9, -0.1, 0.9), (0.9, 0.9, 2.0)])
def test_oee_factors_range_checked(a, p, q):
    with pytest.raises(ValueError):
        OEE(a, p, q)


@pytest.mark.golden
def test_oee_from_times_worked_example():
    """Planned 7000 h, downtime 700 h, so run time 6300 h and A = 0.90.
       Nameplate 10 t/h over 6300 h = 63000 t capacity.
       Actual 56700 t -> P = 56700/63000 = 0.90
       Good 56133 t   -> Q = 56133/56700 = 0.99
       OEE = 0.90 x 0.90 x 0.99 = 0.8019
    """
    run_h_ = 7000.0 - 700.0
    assert run_h_ == 6300.0
    assert run_h_ / 7000.0 == pytest.approx(0.90, abs=1e-12)
    cap_ = 10.0 * run_h_
    assert cap_ == 63000.0
    assert 56700.0 / cap_ == pytest.approx(0.90, abs=1e-12)
    assert 56133.0 / 56700.0 == pytest.approx(0.99, abs=1e-12)
    assert 0.90 * 0.90 * 0.99 == pytest.approx(0.8019, abs=1e-9)
    run_h = 7000.0 - 700.0
    assert run_h == 6300.0
    assert run_h / 7000.0 == pytest.approx(0.90, abs=1e-12)
    cap = 10.0 * run_h
    assert cap == 63000.0
    assert 56700.0 / cap == pytest.approx(0.90, abs=1e-12)
    assert 56133.0 / 56700.0 == pytest.approx(0.99, abs=1e-12)
    o = OEE.from_times(planned_hours=7000.0, downtime_hours=700.0,
                       nameplate_rate=Q_(10.0, "tonne/hour"),
                       actual_output=Q_(56700.0, "tonne"),
                       good_output=Q_(56133.0, "tonne"))
    assert o.availability == pytest.approx(0.90)
    assert o.performance == pytest.approx(0.90)
    assert o.quality == pytest.approx(0.99)
    assert o.value == pytest.approx(0.8019, abs=1e-9)


def test_from_times_rejects_impossible_output():
    with pytest.raises(ValueError, match="exceeds the nameplate capacity"):
        OEE.from_times(7000.0, 700.0, Q_(10.0, "tonne/hour"),
                       Q_(70000.0, "tonne"), Q_(1.0, "tonne"))
    with pytest.raises(ValueError, match="good output"):
        OEE.from_times(7000.0, 700.0, Q_(10.0, "tonne/hour"),
                       Q_(1000.0, "tonne"), Q_(2000.0, "tonne"))
    with pytest.raises(ValueError, match="downtime_hours"):
        OEE.from_times(7000.0, 8000.0, Q_(10.0, "tonne/hour"),
                       Q_(10.0, "tonne"), Q_(10.0, "tonne"))


def test_from_times_requires_correct_dimensionality():
    with pytest.raises(DimensionalityError):
        OEE.from_times(7000.0, 0.0, Q_(10.0, "tonne"),  # rate given as a mass
                       Q_(10.0, "tonne"), Q_(10.0, "tonne"))


@pytest.mark.golden
def test_effective_capacity_worked_example():
    """10 t/h x 7000 h x 0.84645 = 59251.5 t/yr of own throughput.
       At 1.25 t feed per t product that is 47401.2 t/yr of product."""
    oee_ = 0.90 * 0.95 * 0.99
    assert oee_ == pytest.approx(0.84645, abs=1e-9)
    own_ = 10.0 * 7000.0 * oee_
    assert own_ == pytest.approx(59251.5, abs=0.1)
    assert own_ / 1.25 == pytest.approx(47401.2, abs=0.1)
    u = UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, OEE(0.90, 0.95, 0.99), 1.25)
    assert u.effective_capacity.to("tonne").magnitude == pytest.approx(59251.5, abs=0.1)
    assert u.product_capacity.to("tonne").magnitude == pytest.approx(47401.2, abs=0.1)


def test_unit_validates_schedule_and_yield_basis():
    o = OEE(0.9, 0.9, 0.9)
    with pytest.raises(ValueError, match="planned_hours"):
        UnitCapacity("u", Q_(1.0, "tonne/hour"), 0.0, o)
    with pytest.raises(ValueError, match="planned_hours"):
        UnitCapacity("u", Q_(1.0, "tonne/hour"), HOURS_PER_YEAR + 1, o)
    with pytest.raises(ValueError, match="below 1.0"):
        UnitCapacity("u", Q_(1.0, "tonne/hour"), 7000.0, o, 0.8)


@pytest.mark.golden
def test_bottleneck_uses_product_basis_not_nameplate():
    """The defect this basis exists to prevent: the unit with the HIGHER
    nameplate rate is the bottleneck once upstream yield loss is accounted for.

    mill : 10 t/h x 7000 h x 0.84645 = 59251.50 t/yr own throughput
           at 1.25 t feed per t product -> 47401.20 t/yr of product
    leach:  9 t/h x 7000 h x 0.84645 = 53326.35 t/yr own throughput
           at 1.00 -> 53326.35 t/yr of product
    Line rate = min(47401.20, 53326.35) = 47401.20, set by the mill, even though
    the mill's nameplate rate (10 t/h) exceeds the leach's (9 t/h).
    """
    oee_ = 0.90 * 0.95 * 0.99
    assert oee_ == pytest.approx(0.84645, abs=1e-9)
    mill_own_ = 10.0 * 7000.0 * oee_
    leach_own_ = 9.0 * 7000.0 * oee_
    assert mill_own_ == pytest.approx(59251.50, abs=0.1)
    assert leach_own_ == pytest.approx(53326.35, abs=0.1)
    mill_prod_, leach_prod_ = mill_own_ / 1.25, leach_own_ / 1.00
    assert mill_prod_ == pytest.approx(47401.20, abs=0.1)
    assert leach_prod_ == pytest.approx(53326.35, abs=0.1)
    # The inversion: higher nameplate, lower product capacity.
    assert 10.0 > 9.0 and mill_prod_ < leach_prod_
    assert min(mill_prod_, leach_prod_) == pytest.approx(47401.20, abs=0.1)
    oee = 0.90 * 0.95 * 0.99
    assert oee == pytest.approx(0.84645, abs=1e-9)
    mill_own = 10.0 * 7000.0 * oee
    leach_own = 9.0 * 7000.0 * oee
    assert mill_own == pytest.approx(59251.50, abs=1e-1)
    assert leach_own == pytest.approx(53326.35, abs=1e-1)
    mill_prod, leach_prod = mill_own / 1.25, leach_own / 1.00
    assert mill_prod == pytest.approx(47401.20, abs=1e-1)
    assert leach_prod == pytest.approx(53326.35, abs=1e-1)
    # The inversion: higher nameplate, lower product capacity.
    assert 10.0 > 9.0 and mill_prod < leach_prod
    assert min(mill_prod, leach_prod) == pytest.approx(47401.20, abs=1e-1)
    o = OEE(0.90, 0.95, 0.99)
    mill = UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, o, 1.25)
    leach = UnitCapacity("leach", Q_(9.0, "tonne/hour"), 7000.0, o, 1.0)
    r = assess_line([mill, leach])
    assert mill.nameplate_rate > leach.nameplate_rate
    assert r.bottleneck == "mill"
    assert r.line_rate.to("tonne").magnitude == pytest.approx(47401.20, abs=0.05)
    assert leach.product_capacity.to("tonne").magnitude == pytest.approx(53326.35, abs=0.05)


def test_utilisation_is_one_at_bottleneck_and_below_elsewhere():
    o = OEE(0.90, 0.95, 0.99)
    r = assess_line([
        UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, o, 1.25),
        UnitCapacity("leach", Q_(9.0, "tonne/hour"), 7000.0, o, 1.0),
    ])
    assert r.utilisation[r.bottleneck] == pytest.approx(1.0, abs=1e-9)
    others = [v for k, v in r.utilisation.items() if k != r.bottleneck]
    assert all(v < 1.0 for v in others)
    assert r.slack[r.bottleneck].magnitude == pytest.approx(0.0, abs=1e-9)
    assert r.slack["leach"].magnitude > 0.0


def test_exact_tie_reports_zero_margin_rather_than_a_false_single_bottleneck():
    """10 t/h at 1.25 t/t and 8 t/h at 1.0 t/t give IDENTICAL product capacity.
    Naming one of them the bottleneck is arbitrary, and the zero margin says so."""
    o = OEE(0.90, 0.95, 0.99)
    r = assess_line([
        UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, o, 1.25),
        UnitCapacity("leach", Q_(8.0, "tonne/hour"), 7000.0, o, 1.0),
    ])
    caps = [q.to("tonne").magnitude for q in r.product_capacity.values()]
    assert caps[0] == pytest.approx(caps[1], rel=1e-12)
    assert r.bottleneck_margin == pytest.approx(0.0, abs=1e-12)
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in r.utilisation.values())


def test_bottleneck_margin_flags_a_fragile_constraint():
    o = OEE(0.9, 0.95, 0.99)
    tight = assess_line([
        UnitCapacity("a", Q_(10.00, "tonne/hour"), 7000.0, o),
        UnitCapacity("b", Q_(10.01, "tonne/hour"), 7000.0, o),
    ])
    assert tight.bottleneck_margin < 0.01     # debottlenecking 'a' buys almost nothing
    loose = assess_line([
        UnitCapacity("a", Q_(10.0, "tonne/hour"), 7000.0, o),
        UnitCapacity("b", Q_(20.0, "tonne/hour"), 7000.0, o),
    ])
    assert loose.bottleneck_margin == pytest.approx(1.0, rel=1e-6)


def test_single_unit_margin_is_infinite():
    r = assess_line([UnitCapacity("only", Q_(5.0, "tonne/hour"), 7000.0, OEE(1, 1, 1))])
    assert r.bottleneck == "only"
    assert r.bottleneck_margin == float("inf")


def test_empty_and_duplicate_lines_rejected():
    o = OEE(0.9, 0.9, 0.9)
    with pytest.raises(ValueError, match="at least one unit"):
        assess_line([])
    with pytest.raises(ValueError, match="duplicate unit names"):
        assess_line([UnitCapacity("x", Q_(1.0, "tonne/hour"), 100.0, o),
                     UnitCapacity("x", Q_(2.0, "tonne/hour"), 100.0, o)])
