"""End-to-end: ore and site uncertainty propagated through cost to NPV.

This is the integration the platform exists to deliver: an assay range and a
site's price ranges in, a P10/P50/P90 NPV and a ranked list of what to measure
out. It exercises yield_cascade, unit_economics, capex and valuation together
under Monte Carlo and Sobol.
"""
import numpy as np
import pytest
from ae.econ.uncertainty import Uncertain, monte_carlo, sobol_analysis, tornado
from ae.econ.valuation import Project, breakeven_price
from ae.plant.yield_cascade import cascade_yield, off_spec_fraction


def plant_npv(
    mass_yield_leach: float,
    mass_yield_flot: float,
    al_mean_ppm: float,
    al_sigma_ppm: float,
    price: float,
    power_price: float,
    reagent_price: float,
    capex_musd: float,
    discount_rate: float,
) -> dict[str, float]:
    """Ore-to-NPV chain. Deterministic given its inputs, as Sobol requires.

    FEED capacity is fixed by the mine and plant, so PRODUCT = feed x overall
    yield. This is the correct topology for a mine-constrained operation and it
    is what makes yield transmit to value: a lot that fails the 30 ppm Al limit
    is product that cannot be sold, not merely a higher cost per tonne.

    An earlier version of this function held PRODUCT tonnage fixed and let yield
    affect only the cost divisor. The Sobol analysis returned total indices of
    0.0000 for all four yield and purity inputs, which is impossible for a
    purification plant and is how the error was found: lost lots had disappeared
    from the model instead of reducing sales.

    Spec yield is the probability a lot passes the 30 ppm Al limit, on the
    LOGNORMAL model because impurity distributions are right-skewed and the
    normal understates exactly the tail that fails lots.
    """
    USL_AL = 30.0
    FEED_TONNES = 7000.0

    mass_y = cascade_yield([mass_yield_flot, mass_yield_leach])
    spec_y = 1.0 - off_spec_fraction(al_mean_ppm, al_sigma_ppm, USL_AL,
                                     model="lognormal")
    overall = mass_y * spec_y
    product_tonnes = FEED_TONNES * overall

    # Consumables are charged on FEED, so their per-product cost carries 1/yield.
    power_kwh_per_t_feed = 250.0
    reagent_kg_per_t_feed = 12.0
    feed_cost_per_t_feed = (power_kwh_per_t_feed * power_price
                            + reagent_kg_per_t_feed * reagent_price)
    other_variable_per_t_product = 180.0
    cash = feed_cost_per_t_feed / overall + other_variable_per_t_product

    p = Project(
        capex_schedule=[capex_musd * 1e6 / 2] * 2,
        construction_periods=2,
        ramp_fractions=[0.35, 0.75, 1.0],
        nameplate_tonnes=product_tonnes,
        price=price,
        cash_cost_per_tonne=cash,
        life_periods=15,
        fixed_cost_per_period=3.0e6,
        tax_rate=0.21,
    )
    cf = p.cash_flows()
    return {"npv_musd": cf.npv(discount_rate) / 1e6,
            "cash_cost": cash,
            "overall_yield": overall,
            "product_tonnes": product_tonnes,
            "breakeven": breakeven_price(p, discount_rate)}


def chain_inputs():
    """Ranges are scenario bands, not measurements: the Vikarabad deposit is
    uncharacterized, so the Al mean spans a wide band deliberately."""
    return [
        Uncertain("mass_yield_leach", 0.80, 0.97, "triangular", mode=0.92),
        Uncertain("mass_yield_flot", 0.70, 0.93, "triangular", mode=0.85),
        Uncertain("al_mean_ppm", 8.0, 26.0, "triangular", mode=18.0),
        Uncertain("al_sigma_ppm", 1.5, 6.0, "triangular", mode=3.0),
        Uncertain("price", 2800.0, 5200.0, "triangular", mode=3500.0),
        Uncertain("power_price", 0.045, 0.095, "uniform"),
        Uncertain("reagent_price", 1.20, 2.60, "triangular", mode=1.80),
        Uncertain("capex_musd", 25.0, 60.0, "triangular", mode=38.0),
        Uncertain("discount_rate", 0.10, 0.20, "triangular", mode=0.14),
    ]


@pytest.mark.benchmark
def test_full_chain_monte_carlo_10k():
    """The platform's headline requirement: 10,000+ draws with P10/P50/P90."""
    mc = monte_carlo(plant_npv, chain_inputs(), n_draws=10000, seed=42)
    assert mc.n_draws == 10000 and mc.n_failed == 0
    s = mc.summary("npv_musd")
    p_loss = mc.probability_below("npv_musd", 0.0)
    print(f"\nNPV (MUSD) over 10,000 draws:")
    print(f"  P10 {s['P10']:+.1f}   P50 {s['P50']:+.1f}   P90 {s['P90']:+.1f}")
    print(f"  mean {s['mean']:+.1f}  sd {s['sd']:.1f}  P90-P10 {s['P90_minus_P10']:.1f}")
    print(f"  P(NPV < 0) = {p_loss:.3f}")
    ys = mc.summary("overall_yield")
    cc = mc.summary("cash_cost")
    be = mc.summary("breakeven")
    print(f"  overall yield P10 {ys['P10']:.3f} P50 {ys['P50']:.3f} P90 {ys['P90']:.3f}")
    print(f"  cash cost USD/t P10 {cc['P10']:,.0f} P50 {cc['P50']:,.0f} "
          f"P90 {cc['P90']:,.0f}")
    print(f"  breakeven USD/t P10 {be['P10']:,.0f} P50 {be['P50']:,.0f} "
          f"P90 {be['P90']:,.0f}")
    # Ordering and physical sanity.
    assert s["P10"] < s["P50"] < s["P90"]
    assert 0.0 < ys["P10"] < ys["P90"] < 1.0
    assert cc["P10"] > 0.0
    # The spread must be wide: the ore is uncharacterized and the bands say so.
    assert s["P90_minus_P10"] > 10.0
    # P10 percentile standard error, so the quoted figure carries its precision.
    se = mc.percentile_standard_error("npv_musd", 10)
    print(f"  P10 standard error {se:.2f} MUSD")
    assert se < abs(s["P90_minus_P10"]) / 5


@pytest.mark.benchmark
def test_full_chain_sobol_ranks_what_to_measure():
    """Sobol on the real chain. The ranking is the decision-relevant output:
    it says which measurement buys the most variance reduction."""
    ins = chain_inputs()
    r = sobol_analysis(plant_npv, ins, n_base=512, seed=43, output="npv_musd")
    print(f"\nSobol on NPV, N=512, {r.n_evaluations} evaluations")
    for name, st in r.ranking():
        print(f"  {name:20s} S1 {r.first_order[name]:+.4f}  ST {st:+.4f}  "
              f"interaction {r.interaction_share[name]:+.4f}")
    d = r.diagnostics()
    print(f"  sum S1 {d['sum_first_order']:.4f}, sum ST {d['sum_total_order']:.4f}, "
          f"additive fraction {r.additive_fraction:.3f}")
    assert d["sum_first_le_one"], d
    assert d["sum_total_ge_one"], d
    # Price and capex must both matter materially in any sane parameterisation.
    assert r.total_order["price"] > 0.05
    assert r.total_order["capex_musd"] > 0.02
    # REGRESSION: yield and purity must transmit to value. Holding product
    # tonnage fixed made all four of these exactly 0.0000, which is impossible
    # for a purification plant and was a topology error, not a finding.
    yield_st = sum(r.total_order[k] for k in
                   ("mass_yield_leach", "mass_yield_flot", "al_mean_ppm",
                    "al_sigma_ppm"))
    print(f"  combined yield/purity total index: {yield_st:.4f}")
    assert yield_st > 0.05, (
        f"yield and purity carry {yield_st:.4f} of NPV variance; near zero means "
        f"the chain does not transmit yield to volume"
    )
    # Every index bounded in [0, 1] to within sampling error.
    for k in r.names:
        assert -0.1 < r.first_order[k] < 1.1
        assert -0.05 < r.total_order[k] < 1.1


@pytest.mark.benchmark
def test_tornado_and_sobol_can_disagree_on_the_chain():
    """Reported rather than asserted either way: where the chain is
    multiplicative (yield divides cost), interaction is real and a tornado's
    ordering need not match the variance ranking. The memo leads with Sobol."""
    ins = chain_inputs()
    nominal = {"mass_yield_leach": 0.92, "mass_yield_flot": 0.85,
               "al_mean_ppm": 18.0, "al_sigma_ppm": 3.0, "price": 3500.0,
               "power_price": 0.07, "reagent_price": 1.80,
               "capex_musd": 38.0, "discount_rate": 0.14}
    t = tornado(plant_npv, ins, nominal, output="npv_musd")
    r = sobol_analysis(plant_npv, ins, n_base=512, seed=45, output="npv_musd")
    t_order = list(t)
    s_order = [k for k, _ in r.ranking()]
    print(f"\ntornado order: {t_order[:4]}")
    print(f"sobol order:   {s_order[:4]}")
    print(f"additive fraction {r.additive_fraction:.3f} "
          f"(1.0 would mean no interaction)")
    assert set(t_order) == set(s_order)
    # Both must agree that price is in the top three, a weak sanity floor.
    assert "price" in t_order[:3] and "price" in s_order[:3]


def test_chain_is_deterministic_given_inputs():
    """Sobol requires determinism; a stochastic model inflates every total
    index. Verified by exact repetition."""
    kw = {"mass_yield_leach": 0.9, "mass_yield_flot": 0.85, "al_mean_ppm": 18.0,
          "al_sigma_ppm": 3.0, "price": 3500.0, "power_price": 0.07,
          "reagent_price": 1.8, "capex_musd": 38.0, "discount_rate": 0.14}
    a, b = plant_npv(**kw), plant_npv(**kw)
    assert a == b


def test_spec_yield_dominates_when_variability_is_high():
    """Mechanism check: at a 30 ppm limit, raising lot-to-lot sigma from 1.5 to
    6.0 ppm must cut overall yield materially even with mass yields fixed."""
    # The specification limit the mechanism is stated against.
    USL_AL_PPM = 30.0
    assert USL_AL_PPM == 30.0
    base = {"mass_yield_leach": 0.92, "mass_yield_flot": 0.85, "al_mean_ppm": 22.0,
            "price": 3500.0, "power_price": 0.07, "reagent_price": 1.8,
            "capex_musd": 38.0, "discount_rate": 0.14}
    tight = plant_npv(al_sigma_ppm=1.5, **base)
    loose = plant_npv(al_sigma_ppm=6.0, **base)
    assert loose["overall_yield"] < tight["overall_yield"]
    assert loose["cash_cost"] > tight["cash_cost"]
    assert loose["npv_musd"] < tight["npv_musd"]
    assert loose["product_tonnes"] < tight["product_tonnes"], (
        "failing lots must reduce saleable tonnes, not merely raise unit cost"
    )
    print(f"\nsigma 1.5 -> yield {tight['overall_yield']:.3f}, "
          f"cost {tight['cash_cost']:,.0f}, NPV {tight['npv_musd']:+.1f}")
    print(f"sigma 6.0 -> yield {loose['overall_yield']:.3f}, "
          f"cost {loose['cash_cost']:,.0f}, NPV {loose['npv_musd']:+.1f}")
