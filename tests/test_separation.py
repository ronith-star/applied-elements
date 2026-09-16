"""Separation tests: rate models, partition curves, and exact mass balance.

Mass balance closure is tested to MACHINE PRECISION (exact zero residual where
the arithmetic permits it), not to a tolerance, because the two-product formula
has no fitted parameters and any residual would be a bookkeeping error.
"""
import datetime as dt
import math

import pytest

from ae.core.feedstock import Feedstock, ImpurityProfile, OreType
from ae.core.provenance import Source, Tag, Tier, Value
from ae.core.site import Currency, LabourRates, PowerSupply, Site
from ae.core.units import Q_, DimensionalityError
from ae.physics.separation import (
    SRC_DU_2024,
    SRC_LIN_2020,
    SRC_POLAT_CHANDER,
    SRC_WILLS_ACCOUNTING,
    SRC_WILLS_CLASSIFICATION,
    WHIMS_FIELD_EXPONENT,
    PropertyBasis,
    SeparatorKind,
    ep_from_partition_points,
    feedstock_impurity_separation,
    flotation_rate_constant_from_recovery,
    flotation_recovery,
    grade_recovery_curve,
    imperfection,
    logistic_scale_from_ep,
    partition_curve,
    partition_number,
    product_grade_from_reject,
    rectangular_distribution_recovery,
    separation_mass_balance,
    two_product,
    two_product_condition_number,
    whims_rate_constant,
    whims_recovery,
)

SRC = Source(citation="Test source 2026", tier=Tier.T1, doi="10.1000/test",
             accessed=dt.date(2026, 9, 16))


def _feedstock(totals: dict[str, float],
               sample_id: str = "AE-Q-IN-VKB-040") -> Feedstock:
    prof = ImpurityProfile(
        total={k: Value(quantity=Q_(x, "ppm_mass"), tag=Tag.MEASURED, source=SRC)
               for k, x in totals.items()},
        method="LA_ICP_MS",
        basis_material="single_grain",
    )
    return Feedstock(sample_id=sample_id, ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="Test", country="IN", impurities=prof)


def _site() -> Site:
    return Site(
        site_id="IN-TG-VKB", name="Test site", country="IN", region="Telangana",
        currency=Currency.USD,
        power=PowerSupply(
            energy_price=Value(quantity=Q_(0.08, "USD/kWh"), tag=Tag.ASSUMED,
                               basis="test fixture"),
            rate_basis="state_average"),
        labour=LabourRates(
            fully_loaded_operator=Value(quantity=Q_(5.0, "USD/hour"),
                                        tag=Tag.ASSUMED, basis="test fixture")),
    )


# --- (c) dimensional analysis -----------------------------------------------


def test_flotation_requires_a_time_and_a_rate():
    with pytest.raises(DimensionalityError):
        flotation_recovery(Q_(60.0, "m"), Q_(0.05, "1/s"), 0.9)
    with pytest.raises(DimensionalityError):
        flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/m"), 0.9)


def test_flotation_rejects_bare_floats():
    with pytest.raises(TypeError, match="Bare numbers are rejected"):
        flotation_recovery(60.0, Q_(0.05, "1/s"), 0.9)  # type: ignore[arg-type]


def test_rate_constant_units_are_interchangeable():
    """kt is dimensionless, so 1/s and 1/min must agree when kt matches."""
    a = flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/s"), 0.9)
    b = flotation_recovery(Q_(1.0, "min"), Q_(3.0, "1/min"), 0.9)
    assert a == pytest.approx(b, rel=1e-12)


def test_whims_requires_a_magnetic_flux_density():
    with pytest.raises(DimensionalityError):
        whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "A/m"), Q_(1.0, "T"))


def test_whims_rate_constant_returns_reciprocal_time():
    k = whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "T"), Q_(1.0, "T"))
    assert k.dimensionality == Q_(1.0, "1/s").dimensionality


def test_mass_balance_requires_a_mass():
    with pytest.raises(DimensionalityError):
        separation_mass_balance(Q_(1000.0, "m"), 30.0, 0.08, 280.0)


def test_partition_number_is_dimensionless_and_scale_free():
    """A logistic in (x - x50)/s is invariant under a common unit rescaling."""
    a = partition_number(120.0, 100.0, 20.0)
    b = partition_number(0.120, 0.100, 0.020)
    assert a == pytest.approx(b, rel=1e-12)


# --- (d) golden worked examples ---------------------------------------------


@pytest.mark.golden
def test_flotation_worked_example():
    """k = 0.05 1/s, R_inf = 0.90, t = 60 s.

    kt = 3.0, exp(-3) = 0.04978707.
    1 - 0.04978707 = 0.95021293.
    R = 0.90 x 0.95021293 = 0.85519164.
    """
    r = flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/s"), 0.90)
    assert r == pytest.approx(0.85519164, abs=1e-8)


@pytest.mark.golden
def test_flotation_limits():
    """R(0) = 0 exactly, and R -> R_inf as t -> infinity.

    At t = 0: 1 - exp(0) = 0, so R = 0.
    At kt = 20: exp(-20) = 2.061e-9, so R = 0.9(1 - 2.061e-9) = 0.899999998.
    """
    assert flotation_recovery(Q_(0.0, "s"), Q_(0.05, "1/s"), 0.9) == 0.0
    late = flotation_recovery(Q_(400.0, "s"), Q_(0.05, "1/s"), 0.9)
    assert late == pytest.approx(0.9, abs=1e-8)
    assert late < 0.9, "the exponential approaches but never reaches R_inf"


@pytest.mark.golden
def test_rate_constant_inversion_worked_example():
    """R = 0.85519164 at t = 60 s with R_inf = 0.90 must give k = 0.05 1/s.

    R/R_inf = 0.95021293, 1 - 0.95021293 = 0.04978707,
    ln(0.04978707) = -3.0, k = 3.0/60 = 0.05 1/s.
    """
    k = flotation_rate_constant_from_recovery(Q_(60.0, "s"), 0.85519164, 0.90)
    assert k.magnitude == pytest.approx(0.05, rel=1e-7)


@pytest.mark.golden
def test_rectangular_distribution_worked_example():
    """k_max = 0.10 1/s, R_inf = 0.90, t = 60 s.

    k_max t = 6.0, exp(-6) = 0.00247875.
    (1 - 0.00247875)/6 = 0.99752125/6 = 0.16625354.
    R = 0.90(1 - 0.16625354) = 0.90 x 0.83374646 = 0.75037181.
    """
    r = rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)
    assert r == pytest.approx(0.75037181, abs=1e-8)


@pytest.mark.golden
def test_rectangular_is_slower_than_single_k_at_the_mean():
    """A distributed rate always recovers less than a single rate at its mean.

    A rectangular distribution on [0, 0.10] has mean 0.05. At t = 60 s the
    single-k model gives 0.85519164 and the distributed model 0.75037181, so
    the single-k fit OVERSTATES recovery by 0.10481983 in absolute terms, i.e.
    by 13.97 percent relative. That is the scale-up risk of fitting one rate
    constant, and it is why the choice is called out in LIMITATIONS item 3.
    """
    single = flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/s"), 0.90)
    dist = rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)
    assert single - dist == pytest.approx(0.10481983, abs=1e-8)
    assert (single - dist) / single == pytest.approx(0.1225681, abs=1e-6)
    assert dist < single


@pytest.mark.golden
def test_rectangular_limits():
    """R(0) = 0, and the t -> infinity limit is R_inf because (1-e^-x)/x -> 0.

    The approach is ALGEBRAIC (order 1/x), not exponential, so it is far slower
    than Eq. E1: at k_max t = 1e5 the deficit is 1/1e5 of R_inf, giving
    0.9(1 - 1e-5) = 0.899991, and only at k_max t = 1e8 does it reach
    0.89999999. This is a real behavioural difference between the two models,
    not a numerical artefact: a distributed rate constant leaves a slow tail.
    """
    assert rectangular_distribution_recovery(Q_(0.0, "s"), Q_(0.1, "1/s"), 0.9) == 0.0
    mid = rectangular_distribution_recovery(Q_(1e6, "s"), Q_(0.1, "1/s"), 0.9)
    assert mid == pytest.approx(0.899991, abs=1e-6)
    late = rectangular_distribution_recovery(Q_(1e9, "s"), Q_(0.1, "1/s"), 0.9)
    assert late == pytest.approx(0.9, abs=1e-7)
    assert late < 0.9


@pytest.mark.golden
def test_logistic_scale_worked_example():
    """Ep = 20: s = 20/ln 3 = 20/1.09861229 = 18.20478453."""
    assert logistic_scale_from_ep(20.0) == pytest.approx(18.20478453, abs=1e-8)
    assert logistic_scale_from_ep(20.0) * math.log(3.0) == pytest.approx(20.0,
                                                                         rel=1e-12)


@pytest.mark.golden
def test_partition_curve_hits_75_percent_at_x50_plus_ep():
    """The Ep relation, checked at the definition point.

    x50 = 100, Ep = 20, so s = 18.20478453. At x = 120:
    z = 20/18.20478453 = 1.09861229 = ln 3.
    P = 1/(1 + exp(-ln 3)) = 1/(1 + 1/3) = 3/4 = 0.75 exactly.
    By symmetry at x = 80: P = 1/(1 + 3) = 0.25 exactly.
    """
    assert partition_number(120.0, 100.0, 20.0) == pytest.approx(0.75, abs=1e-12)
    assert partition_number(80.0, 100.0, 20.0) == pytest.approx(0.25, abs=1e-12)
    assert partition_number(100.0, 100.0, 20.0) == pytest.approx(0.5, abs=1e-15)


@pytest.mark.golden
def test_ep_round_trips_from_the_quartile_points():
    """Ep = (x75 - x25)/2 = (120 - 80)/2 = 20, recovering the input Ep."""
    assert ep_from_partition_points(80.0, 120.0) == pytest.approx(20.0, abs=1e-12)


@pytest.mark.golden
def test_imperfection_worked_example_density():
    """Density cut at 1.60 g/cm3 with Ep = 0.05: I = 0.05/(1.60 - 1) = 0.0833333."""
    assert imperfection(0.05, 1.60, PropertyBasis.DENSITY) == pytest.approx(
        1.0 / 12.0, abs=1e-12
    )


@pytest.mark.golden
def test_imperfection_worked_example_size():
    """Size cut at 100 um with Ep = 20 um: I = 20/100 = 0.20.

    Note the contrast with the density form: using the density formula here
    would give 20/99 = 0.2020, a 1 percent error, and using the size formula on
    a 1.60 g/cm3 density cut would give 0.05/1.60 = 0.03125 instead of
    0.0833333, a 62 percent error. Declaring the basis is not bureaucracy.
    """
    assert imperfection(20.0, 100.0, PropertyBasis.SIZE) == pytest.approx(0.20,
                                                                          abs=1e-12)
    assert 0.05 / 1.60 == pytest.approx(0.03125, abs=1e-12)
    assert abs(0.03125 - 1.0 / 12.0) / (1.0 / 12.0) == pytest.approx(0.625, abs=1e-3)


@pytest.mark.golden
def test_whims_field_scaling_worked_example():
    """k_ref = 0.02 1/s at 1.0 T, scaled to 1.5 T with n_B = 2.

    (1.5/1.0)^2 = 2.25, k = 0.02 x 2.25 = 0.045 1/s.
    At 0.5 T: (0.5)^2 = 0.25, k = 0.005 1/s.
    """
    assert whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "T"),
                               Q_(1.0, "T")).magnitude == pytest.approx(0.045,
                                                                        abs=1e-12)
    assert whims_rate_constant(Q_(0.02, "1/s"), Q_(0.5, "T"),
                               Q_(1.0, "T")).magnitude == pytest.approx(0.005,
                                                                        abs=1e-12)


@pytest.mark.golden
def test_two_product_worked_example():
    """Impurity f = 1.0, c (reject) = 10.0, t (product) = 0.1 ppm.

    Y = (1.0 - 0.1)/(10.0 - 0.1) = 0.9/9.9 = 0.09090909.
    R = Y c / f = 0.09090909 x 10.0/1.0 = 0.90909091.
    Check: 9.09 percent of the mass carries 90.9 percent of the impurity.
    """
    y, r = two_product(1.0, 10.0, 0.1)
    assert y == pytest.approx(1.0 / 11.0, abs=1e-12)
    assert r == pytest.approx(10.0 / 11.0, abs=1e-12)


@pytest.mark.golden
def test_product_grade_worked_example():
    """Feed 30 ppm Al, reject 8 percent of the mass at 280 ppm Al.

    Impurity to reject = 0.08 x 280 = 22.4 ppm-equivalent of the feed basis.
    Remaining = 30 - 22.4 = 7.6, spread over 1 - 0.08 = 0.92 of the mass.
    Product grade = 7.6/0.92 = 8.26086957 ppm Al.
    Check: 0.08(280) + 0.92(8.26086957) = 22.4 + 7.6 = 30.0 ppm, closing.
    """
    g = product_grade_from_reject(30.0, 0.08, 280.0)
    assert g == pytest.approx(8.26086957, abs=1e-8)
    assert 0.08 * 280.0 + 0.92 * g == pytest.approx(30.0, abs=1e-12)


@pytest.mark.golden
def test_two_product_and_product_grade_are_mutually_consistent():
    """The two formulations are the same mass balance and must agree exactly."""
    g = product_grade_from_reject(30.0, 0.08, 280.0)
    y, r = two_product(30.0, 280.0, g)
    assert y == pytest.approx(0.08, rel=1e-12)
    assert r == pytest.approx(0.08 * 280.0 / 30.0, rel=1e-12)
    assert r == pytest.approx(0.7466667, abs=1e-7)


@pytest.mark.golden
def test_condition_number_worked_example():
    """f = 30, c = 45, t = 28: kappa = (45 - 28 + 30 - 28)/(45 - 28) = 19/17.

    19/17 = 1.11764706, so well conditioned. Contrast the high-purity case
    f = 5.0, c = 5.5, t = 4.9: kappa = (0.6 + 0.1)/0.6 = 1.16666667, still
    small, yet the 0.6 ppm spread between concentrate and tailing is inside
    typical ICP-MS uncertainty at those levels, which is the real failure.
    """
    assert two_product_condition_number(30.0, 45.0, 28.0) == pytest.approx(
        19.0 / 17.0, abs=1e-12
    )
    assert two_product_condition_number(5.0, 5.5, 4.9) == pytest.approx(
        7.0 / 6.0, abs=1e-12
    )


# --- mass balance to MACHINE PRECISION --------------------------------------


def test_mass_balance_closes_exactly():
    """1000 kg feed at 30 ppm, 8 percent reject at 280 ppm.

    Product 920 kg at 8.26086957 ppm. Impurity in = 1000 x 30e-6 = 0.03 kg.
    Impurity out = 920 x 8.26086957e-6 + 80 x 280e-6 = 0.0076 + 0.0224 = 0.03 kg.
    Both residuals are exactly zero in IEEE double arithmetic here.
    """
    bal = separation_mass_balance(Q_(1000.0, "kg"), 30.0, 0.08, 280.0)
    assert bal["product_mass_kg"] == 920.0
    assert bal["reject_mass_kg"] == 80.0
    assert bal["mass_residual_kg"] == 0.0
    assert bal["impurity_residual_kg"] == 0.0
    assert bal["impurity_in_kg"] == bal["impurity_out_kg"]


@pytest.mark.parametrize(
    "feed_kg,feed_ppm,reject_yield,reject_ppm",
    [
        (1000.0, 30.0, 0.08, 280.0),
        (1.0, 128.86, 0.05, 2000.0),
        (5e5, 4944.73, 0.35, 12000.0),
        (12345.678, 0.5, 0.001, 400.0),
        (1e-3, 1e4, 0.5, 1.5e4),
        (777.0, 1.0, 0.9999, 1.0000999),
    ],
)
def test_mass_balance_closes_across_five_orders_of_magnitude(
    feed_kg, feed_ppm, reject_yield, reject_ppm
):
    bal = separation_mass_balance(Q_(feed_kg, "kg"), feed_ppm, reject_yield,
                                  reject_ppm)
    assert abs(bal["mass_residual_kg"]) <= 1e-12 * feed_kg
    assert abs(bal["impurity_residual_kg"]) <= 1e-12 * bal["impurity_in_kg"]
    assert bal["product_mass_kg"] + bal["reject_mass_kg"] == pytest.approx(
        feed_kg, rel=1e-15
    )


def test_mass_balance_in_tonnes_gives_the_same_grades():
    a = separation_mass_balance(Q_(1000.0, "kg"), 30.0, 0.08, 280.0)
    b = separation_mass_balance(Q_(1.0, "tonne"), 30.0, 0.08, 280.0)
    assert a["product_grade_ppm"] == pytest.approx(b["product_grade_ppm"], rel=1e-15)
    assert b["mass_residual_kg"] == 0.0


def test_impurity_cannot_exceed_the_feed_inventory():
    """A reject stream richer than the feed allows would violate mass balance."""
    with pytest.raises(ValueError, match="mass balance violated"):
        separation_mass_balance(Q_(1000.0, "kg"), 30.0, 0.5, 100.0)


def test_grade_recovery_curve_closes_at_every_point():
    rows = grade_recovery_curve(
        30.0, [Q_(t, "s") for t in (10.0, 30.0, 60.0, 120.0, 300.0)],
        Q_(0.02, "1/s"), 0.85, 12.0,
    )
    assert len(rows) == 5
    for row in rows:
        y = row["reject_yield"]
        reject_grade = 30.0 * 12.0
        closed = y * reject_grade + (1.0 - y) * row["product_grade_ppm"]
        assert closed == pytest.approx(30.0, rel=1e-12)


def test_grade_recovery_is_a_tradeoff_not_a_free_lunch():
    """Rising impurity recovery costs product yield: both must move together."""
    rows = grade_recovery_curve(
        30.0, [Q_(t, "s") for t in (10.0, 60.0, 300.0)], Q_(0.02, "1/s"), 0.85, 12.0
    )
    recoveries = [r["impurity_recovery"] for r in rows]
    yields = [r["reject_yield"] for r in rows]
    grades = [r["product_grade_ppm"] for r in rows]
    assert recoveries == sorted(recoveries)
    assert yields == sorted(yields)
    assert grades == sorted(grades, reverse=True)


# --- (e) benchmarks with error reported -------------------------------------


@pytest.mark.benchmark
def test_benchmark_partition_definition_identity(capsys):
    """Benchmark: the logistic Ep relation against its analytic definition.

    The reference value is exact rather than experimental: for a logistic
    partition curve, P(x50 + Ep) = 0.75 identically, because the Ep relation
    s = Ep/ln 3 was derived from that condition. Any implementation error in
    the logistic or in the scale conversion shows up here at full double
    precision. The convention is that of Wills and Finch 2016, Classification
    (doi:10.1016/b978-0-08-097053-0.00009-1).
    """
    errs = []
    for x50, ep in [(100.0, 20.0), (1.6, 0.05), (45.0, 9.0), (1000.0, 250.0)]:
        p_hi = partition_number(x50 + ep, x50, ep)
        p_lo = partition_number(x50 - ep, x50, ep)
        errs.append((x50, ep, abs(p_hi - 0.75), abs(p_lo - 0.25)))
    with capsys.disabled():
        print("\n[benchmark] logistic partition curve vs the Ep definition "
              "(doi:10.1016/b978-0-08-097053-0.00009-1):")
        for x50, ep, e_hi, e_lo in errs:
            print(f"    x50={x50:>7.2f}, Ep={ep:>6.2f}: |P(x75) - 0.75| = "
                  f"{e_hi:.3e}, |P(x25) - 0.25| = {e_lo:.3e}")
    for _, _, e_hi, e_lo in errs:
        assert e_hi < 1e-15
        assert e_lo < 1e-15


@pytest.mark.benchmark
def test_benchmark_mass_balance_residuals(capsys):
    """Benchmark: two-product mass balance residuals at machine precision.

    The reference is exact conservation, so the reported error is the numerical
    residual of the implementation rather than a discrepancy with a
    measurement. Reported across five decades of feed mass so that any
    catastrophic cancellation would be visible. Convention follows Wills and
    Napier-Munn 2005 (doi:10.1016/b978-075064450-1/50005-9).
    """
    cases = [(1e-3, 1e4, 0.5, 1.5e4), (1.0, 128.86, 0.05, 2000.0),
             (1000.0, 30.0, 0.08, 280.0), (5e5, 4944.73, 0.35, 12000.0),
             (1e7, 0.5, 0.001, 400.0)]
    rows = []
    for m, f, y, c in cases:
        bal = separation_mass_balance(Q_(m, "kg"), f, y, c)
        rel_mass = abs(bal["mass_residual_kg"]) / m
        rel_imp = (abs(bal["impurity_residual_kg"]) / bal["impurity_in_kg"]
                   if bal["impurity_in_kg"] > 0 else 0.0)
        rows.append((m, f, rel_mass, rel_imp))
    with capsys.disabled():
        print("\n[benchmark] two-product mass balance residuals "
              "(doi:10.1016/b978-075064450-1/50005-9):")
        for m, f, rel_mass, rel_imp in rows:
            print(f"    feed {m:>12.3e} kg at {f:>10.2f} ppm: relative mass "
                  f"residual {rel_mass:.3e}, relative impurity residual "
                  f"{rel_imp:.3e}")
        print("    reference is exact conservation, so these are implementation "
              "residuals, not model errors")
    for _, _, rel_mass, rel_imp in rows:
        assert rel_mass < 1e-15
        assert rel_imp < 1e-15


@pytest.mark.benchmark
def test_benchmark_first_order_kinetics_against_analytic_solution(capsys):
    """Benchmark: Eq. E1 against the analytic solution of dN/dt = -kN.

    The first-order postulate has a closed-form solution, so the benchmark is
    against numerical integration of the ODE rather than against experimental
    data: no batch flotation kinetics dataset for quartz was retrievable in
    this environment (see LIMITATIONS item 1). An explicit RK4 integration of
    dN/dt = -kN over 60 s with 6000 steps is compared with the closed form, and
    the error is reported. This validates the implementation, not the physics.
    The two-parameter form is that treated as the reference case by Polat and
    Chander 2000 (doi:10.1016/S0301-7516(99)00069-1).
    """
    k, r_inf, t_end, n = 0.05, 0.90, 60.0, 6000
    h = t_end / n
    y = 1.0
    for _ in range(n):
        k1 = -k * y
        k2 = -k * (y + 0.5 * h * k1)
        k3 = -k * (y + 0.5 * h * k2)
        k4 = -k * (y + h * k3)
        y += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    numeric = r_inf * (1.0 - y)
    closed = flotation_recovery(Q_(t_end, "s"), Q_(k, "1/s"), r_inf)
    err = abs(numeric - closed) / closed * 100.0
    with capsys.disabled():
        print(f"\n[benchmark] Eq. E1 vs RK4 integration of dN/dt = -kN "
              f"(k = {k} 1/s, t = {t_end} s, {n} steps): closed form "
              f"{closed:.12f}, numeric {numeric:.12f}, error {err:.3e} percent. "
              f"NO experimental quartz flotation kinetics dataset was retrievable, "
              f"so this is a self-consistency benchmark only "
              f"(doi:10.1016/S0301-7516(99)00069-1 for the model form).")
    assert err < 1e-9


@pytest.mark.benchmark
def test_benchmark_flowsheet_grade_steps_against_lin_2020(capsys):
    """Benchmark: can a two-product balance reach Lin et al. 2020's stage grades?

    Lin et al. 2020 Table 3 (doi:10.1007/s42461-020-00247-0) gives a conceptual
    flowsheet with product grade by stage: mining and washing then crushing and
    desliming to 90 to 99 percent SiO2, magnetic separation plus flotation to 99
    to 99.99 percent, chemical leaching to 99.99 percent, hot chlorination above
    99.999 percent. Those are BANDS in a conceptual diagram, not measured plant
    data, and no yields are given, so this benchmark computes the reject yield
    the two-product formula requires to move between the band edges at a stated
    reject grade, and reports it for plausibility rather than asserting
    agreement.

    The finding worth recording: moving from 99 to 99.99 percent SiO2 means
    taking impurity from 10000 to 100 ppm, a 99 percent impurity rejection, and
    if the reject stream assays 50 percent impurity (a very dirty reject) that
    demands a reject yield of 1.98 percent. The mass balance is comfortable.
    What is not comfortable, and what Table 3 does not show, is that the same
    arithmetic from 99.99 to 99.999 percent requires rejecting material at a
    grade no physical separator produces, which is why that step is
    chlorination chemistry and not a separator.
    """
    rows = []
    for name, from_pct, to_pct, reject_pct in [
        ("99 to 99.99 SiO2", 99.0, 99.99, 50.0),
        ("99.9 to 99.99 SiO2", 99.9, 99.99, 50.0),
        ("99.99 to 99.999 SiO2", 99.99, 99.999, 50.0),
    ]:
        f = (100.0 - from_pct) * 1e4      # ppm impurity
        t = (100.0 - to_pct) * 1e4
        c = reject_pct * 1e4
        y, r = two_product(f, c, t)
        kappa = two_product_condition_number(f, c, t)
        rows.append((name, f, t, y, r, kappa))
    with capsys.disabled():
        print("\n[benchmark] reject yield required for Lin et al. 2020 Table 3 "
              "grade steps (doi:10.1007/s42461-020-00247-0), assuming a 50 wt "
              "percent impurity reject:")
        for name, f, t, y, r, kappa in rows:
            print(f"    {name:>22}: impurity {f:>9.1f} -> {t:>8.1f} ppm, reject "
                  f"yield {y * 100:6.3f} percent, impurity recovery to reject "
                  f"{r * 100:6.2f} percent, condition number {kappa:.4f}")
        print("    Table 3 gives grade BANDS in a conceptual diagram and no "
              "yields, so these are the yields the mass balance requires, not "
              "published values.")
    names = [r[0] for r in rows]
    yields = [r[3] for r in rows]
    assert names[0] == "99 to 99.99 SiO2"
    assert yields[0] == pytest.approx(0.0198, abs=1e-4)
    # Each finer step needs a smaller reject yield, because there is less
    # impurity left to reject, yet gets harder because the reject grade demanded
    # of a real separator rises. Mass balance alone does not see the difficulty.
    assert yields == sorted(yields, reverse=True)


@pytest.mark.benchmark
def test_benchmark_whims_field_exponent_sensitivity(capsys):
    """Benchmark: sensitivity of WHIMS recovery to the ASSUMED field exponent.

    There is no measured exponent to benchmark against, so this reports the
    SPREAD the assumption introduces, which is the honest form of the
    statement. Scaling k from 1.0 T to 1.5 T with n_B across its assumed
    uniform bracket of 1 to 3 gives rate constants of 0.03, 0.045 and 0.0675
    1/s, a factor of 2.25 across the bracket, and at a 60 s residence time
    recoveries of 0.7512 (n=1), 0.8395 (n=2) and 0.8843 (n=3) against
    R_inf = 0.90. The 2.25-fold rate spread compresses to a 17.7 percent
    recovery spread, because the exponential is already well into saturation at
    60 s: the assumption matters much less at long residence time than at short.
    At a 10 s residence time the same bracket gives 0.2333, 0.3261 and 0.4418, a
    89.4 percent spread. The exponent must be measured before any WHIMS sizing
    that runs at short residence time.
    """
    r_inf, t = 0.90, Q_(60.0, "s")
    out = []
    for n in (1.0, 2.0, 3.0):
        k = whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "T"), Q_(1.0, "T"),
                                exponent=n)
        rec, _caveat = whims_recovery(t, Q_(0.02, "1/s"), Q_(1.5, "T"),
                                      Q_(1.0, "T"), r_inf, exponent=n)
        out.append((n, k.magnitude, rec))
    spread = (out[-1][2] - out[0][2]) / out[0][2] * 100.0
    with capsys.disabled():
        print(f"\n[benchmark] WHIMS field exponent sensitivity (n_B ASSUMED = "
              f"{WHIMS_FIELD_EXPONENT.quantity.magnitude}, bracket 1 to 3):")
        for n, k, rec in out:
            print(f"    n_B = {n:.1f}: k = {k:.5f} 1/s, R(60 s) = {rec:.4f}")
        print(f"    recovery spread across the bracket: {spread:.1f} percent "
              f"relative. No measured exponent exists; this is the cost of the "
              f"assumption, not a validated result.")
    assert out[0][1] == pytest.approx(0.030, abs=1e-12)
    assert out[1][1] == pytest.approx(0.045, abs=1e-12)
    assert out[2][1] == pytest.approx(0.0675, abs=1e-12)
    assert out[0][2] == pytest.approx(0.7512, abs=1e-4)
    assert out[1][2] == pytest.approx(0.8395, abs=1e-4)
    assert out[2][2] == pytest.approx(0.8843, abs=1e-4)
    assert spread == pytest.approx(17.7, abs=0.1)
    # Short residence time is where the assumption bites.
    short = [
        whims_recovery(Q_(10.0, "s"), Q_(0.02, "1/s"), Q_(1.5, "T"), Q_(1.0, "T"),
                       r_inf, exponent=n)[0]
        for n in (1.0, 2.0, 3.0)
    ]
    short_spread = (short[-1] - short[0]) / short[0] * 100.0
    with capsys.disabled():
        print(f"    at 10 s residence: R = {short[0]:.4f}, {short[1]:.4f}, "
              f"{short[2]:.4f}, spread {short_spread:.1f} percent relative")
    assert short_spread > 80.0
    assert WHIMS_FIELD_EXPONENT.tag is Tag.ASSUMED


# --- physical sanity: no recovery above 1, no negatives ---------------------


@pytest.mark.parametrize("r_inf", [-0.1, 1.1, 2.0])
def test_ultimate_recovery_must_be_a_fraction(r_inf):
    with pytest.raises(ValueError):
        flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/s"), r_inf)


def test_negative_time_raises():
    with pytest.raises(ValueError, match="non-negative"):
        flotation_recovery(Q_(-1.0, "s"), Q_(0.05, "1/s"), 0.9)


@pytest.mark.parametrize("k", [0.0, -0.05])
def test_non_positive_rate_constant_raises(k):
    with pytest.raises(ValueError, match="must be positive"):
        flotation_recovery(Q_(60.0, "s"), Q_(k, "1/s"), 0.9)


def test_recovery_never_exceeds_ultimate_recovery():
    for t in (0.0, 1.0, 10.0, 100.0, 1e4, 1e8):
        assert flotation_recovery(Q_(t, "s"), Q_(0.05, "1/s"), 0.9) <= 0.9


def test_recovery_is_monotonic_in_time():
    prev = -1.0
    for t in (0.0, 5.0, 15.0, 45.0, 120.0):
        r = flotation_recovery(Q_(t, "s"), Q_(0.05, "1/s"), 0.9)
        assert r > prev
        prev = r


def test_rate_fit_refuses_recovery_at_or_above_the_ultimate():
    with pytest.raises(ValueError, match="cannot reach it"):
        flotation_rate_constant_from_recovery(Q_(60.0, "s"), 0.9, 0.9)


def test_negative_field_exponent_raises():
    with pytest.raises(ValueError, match="cannot fall with rising field"):
        whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "T"), Q_(1.0, "T"),
                            exponent=-1.0)


def test_zero_field_raises():
    with pytest.raises(ValueError, match="fields must be positive"):
        whims_rate_constant(Q_(0.02, "1/s"), Q_(0.0, "T"), Q_(1.0, "T"))


def test_zero_ep_raises_because_the_logistic_degenerates():
    with pytest.raises(ValueError, match="perfect separation"):
        logistic_scale_from_ep(0.0)
    with pytest.raises(ValueError, match="Ep must be positive"):
        partition_number(120.0, 100.0, -5.0)


def test_partition_number_stays_in_bounds_over_a_wide_range():
    for x in (1e-6, 1.0, 50.0, 100.0, 1e3, 1e6, 1e12):
        p = partition_number(x, 100.0, 20.0)
        assert 0.0 <= p <= 1.0


def test_partition_curve_is_monotonic():
    pts = partition_curve([50.0, 200.0, 100.0, 80.0, 120.0], 100.0, 20.0)
    assert [p[0] for p in pts] == [50.0, 80.0, 100.0, 120.0, 200.0]
    assert all(pts[i][1] <= pts[i + 1][1] for i in range(len(pts) - 1))


def test_bypass_raises_the_floor_of_the_partition_curve():
    """A 15 percent bypass means nothing partitions below 0.15.

    The logistic tail is slow: at x = 1e-6 with x50 = 100 and s = 18.2048 the
    core logistic is still 0.004098, so P = 0.15 + 0.85(0.004098) = 0.153484.
    Only far into the tail (x = -1000, z = -60.4) does P reach the 0.15 floor.
    """
    assert partition_number(1e-6, 100.0, 20.0, bypass=0.15) == pytest.approx(
        0.153484, abs=1e-6
    )
    p_fine = partition_number(-1000.0, 100.0, 20.0, bypass=0.15)
    assert p_fine == pytest.approx(0.15, abs=1e-12)
    assert partition_number(100.0, 100.0, 20.0, bypass=0.15) == pytest.approx(
        0.15 + 0.85 * 0.5, abs=1e-12
    )


def test_density_imperfection_rejects_a_cut_below_water():
    with pytest.raises(ValueError, match="must exceed the medium density"):
        imperfection(0.05, 0.9, PropertyBasis.DENSITY)


def test_ep_from_points_requires_ordered_inputs():
    with pytest.raises(ValueError, match="must exceed"):
        ep_from_partition_points(120.0, 80.0)


def test_two_product_rejects_unordered_assays():
    with pytest.raises(ValueError, match="must exceed tailing"):
        two_product(1.0, 0.1, 10.0)


def test_two_product_rejects_a_feed_outside_the_stream_assays():
    with pytest.raises(ValueError, match="must lie between"):
        two_product(20.0, 10.0, 1.0)
    with pytest.raises(ValueError, match="must lie between"):
        two_product(0.5, 10.0, 1.0)


def test_two_product_rejects_negative_assays():
    with pytest.raises(ValueError):
        two_product(-1.0, 10.0, 0.1)


def test_product_grade_never_exceeds_feed_grade():
    """Over every FEASIBLE (yield, enrichment) pair the product is cleaner.

    Feasibility is the mass-balance condition y x enrichment <= 1: a reject
    taking 20 percent of the mass at 20 times the feed grade would carry four
    times the impurity that exists. Such pairs are rejected, not clipped, and
    that is tested separately.
    """
    tested = 0
    for y in (0.001, 0.05, 0.2, 0.5):
        for enrich in (1.5, 5.0, 20.0):
            if y * enrich > 1.0:
                continue
            g = product_grade_from_reject(30.0, y, 30.0 * enrich)
            assert g <= 30.0
            tested += 1
    assert tested == 9


@pytest.mark.parametrize("y,enrich", [(0.2, 20.0), (0.5, 5.0), (0.5, 20.0)])
def test_infeasible_yield_enrichment_pairs_are_rejected(y, enrich):
    """y x enrichment > 1 means the reject carries more impurity than exists."""
    with pytest.raises(ValueError, match="mass balance violated"):
        product_grade_from_reject(30.0, y, 30.0 * enrich)


def test_product_grade_rejects_impossible_reject_inventory():
    with pytest.raises(ValueError, match="mass balance violated"):
        product_grade_from_reject(30.0, 0.5, 100.0)


def test_reject_yield_of_one_is_refused():
    with pytest.raises(ValueError):
        product_grade_from_reject(30.0, 1.0, 280.0)


def test_grade_recovery_requires_an_enriched_reject():
    with pytest.raises(ValueError, match="must exceed 1"):
        grade_recovery_curve(30.0, [Q_(60.0, "s")], Q_(0.02, "1/s"), 0.9, 1.0)


# --- FEEDSTOCK and SITE plumbing --------------------------------------------


def test_feedstock_separation_is_capped_by_the_removable_fraction():
    """The lattice floor must survive an arbitrarily long residence time.

    Al 100 ppm with 55 percent removable: even at 1e6 s the residual cannot
    fall below 45 ppm.
    """
    f = _feedstock({"Al": 100.0})
    out = feedstock_impurity_separation(f, "Al", 0.55, Q_(1e6, "s"),
                                        Q_(0.05, "1/s"), SeparatorKind.FLOTATION)
    assert out["irreducible_residual_ppm"] == pytest.approx(45.0, abs=1e-9)
    assert out["residual_ppm"] >= out["irreducible_residual_ppm"] - 1e-9
    assert out["residual_ppm"] == pytest.approx(45.0, abs=1e-6)


def test_feedstock_separation_worked_numbers():
    """Al 100 ppm, removable 0.55, k = 0.05 1/s, t = 60 s.

    R = 0.55(1 - exp(-3)) = 0.55 x 0.95021293 = 0.52261711.
    removed = 100 x 0.52261711 = 52.261711 ppm.
    residual = 100 - 52.261711 = 47.738289 ppm, above the 45.0 ppm floor.
    """
    f = _feedstock({"Al": 100.0})
    out = feedstock_impurity_separation(f, "Al", 0.55, Q_(60.0, "s"),
                                        Q_(0.05, "1/s"), SeparatorKind.WHIMS)
    assert out["recovery"] == pytest.approx(0.52261711, abs=1e-8)
    assert out["removed_ppm"] == pytest.approx(52.261711, abs=1e-6)
    assert out["residual_ppm"] == pytest.approx(47.738289, abs=1e-6)
    assert out["unit"] == "whims"


def test_feedstock_separation_raises_on_an_unmeasured_element():
    f = _feedstock({"Al": 100.0})
    with pytest.raises(KeyError, match="was not measured"):
        feedstock_impurity_separation(f, "Ti", 0.5, Q_(60.0, "s"), Q_(0.05, "1/s"),
                                      SeparatorKind.FLOTATION)


def test_site_is_recorded_but_optional():
    f = _feedstock({"Al": 100.0})
    a = feedstock_impurity_separation(f, "Al", 0.5, Q_(60.0, "s"), Q_(0.05, "1/s"),
                                      SeparatorKind.FLOTATION)
    b = feedstock_impurity_separation(f, "Al", 0.5, Q_(60.0, "s"), Q_(0.05, "1/s"),
                                      SeparatorKind.FLOTATION, site=_site())
    assert a["site_id"] == "not specified"
    assert b["site_id"] == "IN-TG-VKB"
    assert a["residual_ppm"] == b["residual_ppm"]


def test_two_feedstocks_give_two_different_outcomes():
    """Ore-agnosticism: the answer tracks the feedstock, not a module constant."""
    dirty = _feedstock({"Al": 400.0}, sample_id="AE-Q-IN-VKB-041")
    clean = _feedstock({"Al": 25.0}, sample_id="AE-Q-NO-DRG-002")
    a = feedstock_impurity_separation(dirty, "Al", 0.5, Q_(60.0, "s"),
                                      Q_(0.05, "1/s"), SeparatorKind.FLOTATION)
    b = feedstock_impurity_separation(clean, "Al", 0.5, Q_(60.0, "s"),
                                      Q_(0.05, "1/s"), SeparatorKind.FLOTATION)
    assert a["recovery"] == pytest.approx(b["recovery"], rel=1e-12)
    assert a["residual_ppm"] / b["residual_ppm"] == pytest.approx(16.0, rel=1e-12)


@pytest.mark.parametrize("kind", list(SeparatorKind))
def test_all_separator_kinds_are_usable(kind):
    f = _feedstock({"Fe": 12.0})
    out = feedstock_impurity_separation(f, "Fe", 0.9, Q_(60.0, "s"), Q_(0.05, "1/s"),
                                        kind)
    assert out["unit"] == kind.value
    assert "must be measured" in str(out["note"])


# --- provenance discipline --------------------------------------------------


def test_whims_exponent_is_assumed_with_an_honest_basis():
    assert WHIMS_FIELD_EXPONENT.tag is Tag.ASSUMED
    assert WHIMS_FIELD_EXPONENT.source is None
    assert WHIMS_FIELD_EXPONENT.basis and "ESTIMATE" in WHIMS_FIELD_EXPONENT.basis
    assert WHIMS_FIELD_EXPONENT.confidence == "low"
    assert WHIMS_FIELD_EXPONENT.dist is not None
    assert WHIMS_FIELD_EXPONENT.dist.kind == "uniform"


def test_whims_recovery_caveat_names_the_assumption():
    _, caveat = whims_recovery(Q_(60.0, "s"), Q_(0.02, "1/s"), Q_(1.5, "T"),
                               Q_(1.0, "T"), 0.9)
    assert "ASSUMED default" in caveat
    _, caveat_fit = whims_recovery(Q_(60.0, "s"), Q_(0.02, "1/s"), Q_(1.5, "T"),
                                   Q_(1.0, "T"), 0.9, exponent=1.7)
    assert "fitted" in caveat_fit


def test_no_rate_constant_is_supplied_by_the_module():
    """Every rate model takes k as an argument; none is a module constant."""
    import ae.physics.separation as sep

    for name in dir(sep):
        obj = getattr(sep, name)
        if isinstance(obj, Value) and name != "WHIMS_FIELD_EXPONENT":
            pytest.fail(f"unexpected module-level Value {name}: rate constants must "
                        f"be measured per feed, not held as constants")


def test_sources_are_real_dated_and_tiered():
    for src in (SRC_POLAT_CHANDER, SRC_WILLS_ACCOUNTING, SRC_WILLS_CLASSIFICATION,
                SRC_LIN_2020, SRC_DU_2024):
        assert src.tier is Tier.T1
        assert src.doi
        assert src.accessed == dt.date(2026, 9, 16)
        assert src.note and len(src.note) > 40


def test_du_2024_is_cited_only_for_reagent_openness():
    assert "No rate constant taken" in (SRC_DU_2024.note or "")


def test_polat_chander_is_flagged_as_metadata_only():
    assert "No numerical parameter is taken from it" in (SRC_POLAT_CHANDER.note or "")
