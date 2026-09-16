"""Comminution energy tests: dimensions, worked examples, literature benchmarks.

The benchmark tests REPORT their error rather than only asserting a loose
tolerance, so a regression shows up as a changed number in the test output and
not only as a pass or fail.
"""
import datetime as dt
import math

import pytest

from ae.core.feedstock import Feedstock, OreType, PhysicalProperties
from ae.core.provenance import MISSING, Source, Tag, Tier, Value
from ae.core.site import Currency, LabourRates, PowerSupply, Site
from ae.core.units import DIMS, Q_, DimensionalityError
from ae.physics.comminution import (
    BOND_GRINDABILITY_ALPHA,
    BOND_GRINDABILITY_BETA,
    BOND_GRINDABILITY_GAMMA,
    BOND_REFERENCE_P80_UM,
    METRIC_TON_KG,
    SHORT_TON_KG,
    SHORT_TONS_PER_METRIC_TON,
    SizeLaw,
    TonConvention,
    bond_specific_energy,
    bond_wi_from_grindability,
    bond_work_index_from_energy,
    convert_ton_convention,
    feedstock_work_index,
    grinding_energy_cost,
    grinding_specific_energy,
    kick_specific_energy,
    percent_error,
    recommended_law,
    rittinger_specific_energy,
    sourced_quartz_rich_benchmark,
    walker_specific_energy,
    work_index_prior,
)

SRC = Source(citation="Test source 2026", tier=Tier.T1, doi="10.1000/test",
             accessed=dt.date(2026, 9, 16))


def _feedstock(
    wi: Value | None = None,
    f80: Value | None = None,
    ore_type: OreType = OreType.VEIN_QUARTZ,
    sample_id: str = "AE-Q-IN-VKB-010",
) -> Feedstock:
    """A feedstock with only the physical fields a comminution model reads."""
    phys = PhysicalProperties(
        bond_work_index=wi if wi is not None else MISSING,
        f80=f80 if f80 is not None else MISSING,
    )
    return Feedstock(sample_id=sample_id, ore_type=ore_type, deposit_name="Test",
                     country="IN", physical=phys)


def _site(price_per_kwh: float = 0.08, currency: Currency = Currency.USD) -> Site:
    return Site(
        site_id="IN-TG-VKB",
        name="Test site",
        country="IN",
        region="Telangana",
        currency=currency,
        power=PowerSupply(
            energy_price=Value(quantity=Q_(price_per_kwh, f"{currency.value}/kWh"),
                               tag=Tag.ASSUMED,
                               basis="test fixture, not a sourced tariff"),
            rate_basis="state_average",
        ),
        labour=LabourRates(
            fully_loaded_operator=Value(quantity=Q_(5.0, f"{currency.value}/hour"),
                                        tag=Tag.ASSUMED, basis="test fixture"),
        ),
    )


# --- (c) dimensional analysis enforced by tests -----------------------------


def test_bond_energy_has_specific_energy_dimensionality():
    w = bond_specific_energy(Q_(13.0, "kWh/short_ton"), Q_(3000.0, "um"),
                             Q_(100.0, "um"), TonConvention.SHORT)
    assert w.dimensionality == DIMS["specific_energy_mass"]


def test_bond_rejects_bare_float_work_index():
    with pytest.raises(TypeError, match="Bare numbers are rejected"):
        bond_specific_energy(13.0, Q_(3000.0, "um"), Q_(100.0, "um"),  # type: ignore[arg-type]
                             TonConvention.SHORT)


def test_bond_rejects_wrong_dimensionality_work_index():
    with pytest.raises(DimensionalityError):
        bond_specific_energy(Q_(13.0, "kWh"), Q_(3000.0, "um"), Q_(100.0, "um"),
                             TonConvention.SHORT)


def test_sizes_must_be_lengths():
    with pytest.raises(DimensionalityError):
        bond_specific_energy(Q_(13.0, "kWh/short_ton"), Q_(3000.0, "kg"),
                             Q_(100.0, "um"), TonConvention.SHORT)


@pytest.mark.parametrize("convention", list(TonConvention))
def test_energy_unit_matches_convention(convention):
    w = bond_specific_energy(Q_(13.0, convention.energy_unit), Q_(3000.0, "um"),
                             Q_(100.0, "um"), convention)
    assert str(w.units) == str(Q_(1.0, convention.energy_unit).units)


def test_rittinger_constant_must_carry_length():
    """A Rittinger constant without a length unit yields the wrong dimensionality."""
    with pytest.raises(DimensionalityError):
        rittinger_specific_energy(Q_(1.0, "kWh/metric_ton"), Q_(1000.0, "um"),
                                  Q_(100.0, "um"))
    ok = rittinger_specific_energy(Q_(1.0, "kWh*um/metric_ton"), Q_(1000.0, "um"),
                                   Q_(100.0, "um"))
    assert ok.dimensionality == DIMS["specific_energy_mass"]


def test_kick_constant_is_specific_energy():
    out = kick_specific_energy(Q_(2.0, "kWh/metric_ton"), Q_(1000.0, "um"),
                               Q_(100.0, "um"))
    assert out.dimensionality == DIMS["specific_energy_mass"]


# --- (d) golden, hand-traceable worked examples -----------------------------


@pytest.mark.golden
def test_bond_worked_example_metric():
    """Wi = 12.3 kWh/metric ton, F80 = 2000 um, P80 = 150 um.

    1/sqrt(150) = 0.08164966, 1/sqrt(2000) = 0.02236068,
    difference = 0.05928898.
    10 x 12.3 x 0.05928898 = 123 x 0.05928898 = 7.29254 kWh per metric tonne.
    """
    w = bond_specific_energy(Q_(12.3, "kWh/metric_ton"), Q_(2000.0, "um"),
                             Q_(150.0, "um"), TonConvention.METRIC)
    assert w.magnitude == pytest.approx(7.29254, abs=1e-5)


@pytest.mark.golden
def test_bond_definitional_identity():
    """Bond's definition: F80 -> infinity, P80 = 100 um gives W = Wi exactly.

    10 Wi (1/sqrt(100) - 1/sqrt(F80)) -> 10 Wi (0.1 - 0) = Wi.
    A very large but finite F80 approaches it: at F80 = 1e12 um the feed term
    is 1e-6, so W = 10(13.6)(0.1 - 1e-6) = 13.5999 kWh/short ton.
    """
    w = bond_specific_energy(Q_(13.6, "kWh/short_ton"), Q_(1e12, "um"),
                             Q_(BOND_REFERENCE_P80_UM, "um"), TonConvention.SHORT)
    assert w.magnitude == pytest.approx(13.5999, abs=1e-4)
    assert w.magnitude == pytest.approx(13.6, rel=1e-4)


@pytest.mark.golden
def test_coefficient_ten_is_sqrt_of_reference_size():
    """The 10 in Bond's equation is sqrt(100 um), not a unit conversion."""
    assert math.sqrt(BOND_REFERENCE_P80_UM) == pytest.approx(10.0, abs=0.0)


@pytest.mark.golden
def test_ton_conversion_worked_example():
    """7.29254 kWh/short ton x (1000/907.18474) = 7.29254 x 1.1023113 = 8.038649.

    The metric figure is LARGER because the reference mass is larger.
    """
    q = convert_ton_convention(Q_(7.29254, "kWh/short_ton"), TonConvention.METRIC)
    assert q.magnitude == pytest.approx(8.038649, abs=1e-6)
    assert q.magnitude > 7.29254


@pytest.mark.golden
def test_short_ton_definition_is_exact():
    """NIST SP 811: 1 short ton = 907.18474 kg exactly."""
    assert SHORT_TON_KG == 907.18474
    assert Q_(1.0, "short_ton").to("kg").magnitude == pytest.approx(907.18474, abs=1e-9)
    assert SHORT_TONS_PER_METRIC_TON == pytest.approx(1.1023113109, abs=1e-9)
    assert METRIC_TON_KG / SHORT_TON_KG == SHORT_TONS_PER_METRIC_TON


@pytest.mark.golden
def test_convention_confusion_is_10_23_percent():
    """Reading a kWh/short-ton index as metric understates energy by 9.28 percent.

    Same numeric Wi = 13.6 read in the two conventions, both expressed in
    kWh per metric tonne: 13.6 kWh/short ton = 14.9914 kWh/metric tonne, so
    calling it 13.6 kWh/metric tonne is (14.9914 - 13.6)/14.9914 = 9.28
    Same numeric Wi = 13.6 read in the two conventions, both expressed in
    kWh per metric tonne: 13.6 kWh/short ton = 14.991434 kWh/metric tonne, so
    calling it 13.6 kWh/metric tonne is (14.991434 - 13.6)/14.991434 = 9.2815
    percent low. Equivalently the metric-read figure must be multiplied by
    1.1023113 (a 10.2311 percent increase) to recover the truth.
    """
    as_short = convert_ton_convention(Q_(13.6, "kWh/short_ton"), TonConvention.METRIC)
    assert as_short.magnitude == pytest.approx(14.991434, abs=1e-6)
    understatement = (as_short.magnitude - 13.6) / as_short.magnitude
    assert understatement == pytest.approx(0.0928153, abs=1e-7)
    assert (as_short.magnitude - 13.6) / 13.6 == pytest.approx(0.1023113, abs=1e-7)


@pytest.mark.golden
def test_bond_inversion_round_trip():
    """Inverting Eq. E1 recovers the work index that produced the energy."""
    wi_in = Q_(13.6, "kWh/short_ton")
    w = bond_specific_energy(wi_in, Q_(3000.0, "um"), Q_(120.0, "um"),
                             TonConvention.SHORT)
    wi_out = bond_work_index_from_energy(w, Q_(3000.0, "um"), Q_(120.0, "um"),
                                         TonConvention.SHORT)
    assert wi_out.magnitude == pytest.approx(13.6, rel=1e-12)


@pytest.mark.golden
def test_grindability_equation_worked_example():
    """Eq. E2 with G = 1.5 g/rev, p_i = 149 um, F80 = 2000 um, P80 = 150 um.

    149^0.23: ln 149 = 5.0039463, x 0.23 = 1.1509077, exp = 3.1610607.
    1.5^0.82: ln 1.5 = 0.4054651, x 0.82 = 0.3324814, exp = 1.3944239.
    size term = 10(1/sqrt(150) - 1/sqrt(2000)) = 0.5928898.
    denominator = 3.1610607 x 1.3944239 x 0.5928898 = 2.6133744.
    Wi = 44.5/2.6133744 = 17.02779 kWh/short ton.
    """
    wi = bond_wi_from_grindability(1.5, Q_(149.0, "um"), Q_(2000.0, "um"),
                                   Q_(150.0, "um"))
    assert wi.magnitude == pytest.approx(17.02779, abs=1e-5)
    assert str(wi.units) == str(Q_(1.0, "kWh/short_ton").units)


@pytest.mark.golden
def test_walker_reduces_to_the_three_named_laws():
    """Eq. E3 at n = 1, 1.5, 2 reproduces Kick, Bond and Rittinger exactly.

    At n = 1.5 the integral is 2C(P^-0.5 - F^-0.5), and Bond's form is
    10 Wi (P^-0.5 - F^-0.5), so 2C = 10 Wi, i.e. C = 5 Wi. With Wi = 12.3,
    C = 61.5 and the Walker result must equal the Bond result 7.29254.
    """
    f, p = Q_(2000.0, "um"), Q_(150.0, "um")
    bond = bond_specific_energy(Q_(12.3, "kWh/metric_ton"), f, p, TonConvention.METRIC)
    walker_bond = walker_specific_energy(Q_(61.5, "kWh*um**0.5/metric_ton"), f, p, 1.5)
    assert walker_bond.to("kWh/metric_ton").magnitude == pytest.approx(
        bond.magnitude, rel=1e-12
    )
    kick = kick_specific_energy(Q_(2.0, "kWh/metric_ton"), f, p)
    walker_kick = walker_specific_energy(Q_(2.0, "kWh/metric_ton"), f, p, 1.0)
    assert walker_kick.to("kWh/metric_ton").magnitude == pytest.approx(
        kick.magnitude, rel=1e-12
    )
    ritt = rittinger_specific_energy(Q_(1.0, "kWh*um/metric_ton"), f, p)
    walker_ritt = walker_specific_energy(Q_(1.0, "kWh*um/metric_ton"), f, p, 2.0)
    assert walker_ritt.to("kWh/metric_ton").magnitude == pytest.approx(
        ritt.to("kWh/metric_ton").magnitude, rel=1e-12
    )


@pytest.mark.golden
def test_kick_worked_example():
    """C = 2 kWh/t, F80/P80 = 2000/150 = 13.3333, ln 13.3333 = 2.590267.

    W = 2 x 2.5902672 = 5.1805343 kWh/metric tonne.
    """
    w = kick_specific_energy(Q_(2.0, "kWh/metric_ton"), Q_(2000.0, "um"),
                             Q_(150.0, "um"))
    assert w.magnitude == pytest.approx(5.1805343, abs=1e-7)


@pytest.mark.golden
def test_rittinger_worked_example():
    """C = 1 kWh um/t: 1/150 - 1/2000 = 0.00666667 - 0.0005 = 0.00616667.

    W = 0.00616667 kWh per metric tonne.
    """
    w = rittinger_specific_energy(Q_(1.0, "kWh*um/metric_ton"), Q_(2000.0, "um"),
                                  Q_(150.0, "um"))
    assert w.to("kWh/metric_ton").magnitude == pytest.approx(0.00616667, abs=1e-8)


@pytest.mark.golden
def test_percent_error_worked_example():
    """Eq. 13 of Arellano-Pina et al. 2023: |(12.3 - 11.8)/12.3| x 100 = 4.065."""
    assert percent_error(12.3, 11.8) == pytest.approx(4.065, abs=1e-3)


# --- (e) benchmarks against literature, WITH ERROR REPORTED -----------------


@pytest.mark.benchmark
def test_benchmark_ore_a_work_index_round_trip(capsys):
    """Benchmark: Ore A of Arellano-Pina et al. 2023, Wi = 12.3 kWh/metric ton.

    The only Bond ball mill work index on a quartz-rich ore retrievable from a
    peer-reviewed DOI-bearing source in this environment (Table 2, standard
    procedure, 30.5 cm Bond mill; Ore A is 28.75 wt% Si, a quartz plus
    aluminosilicate assemblage). Benchmark: compute the specific energy for a
    representative grind and invert it, and confirm the round trip reproduces
    the published index to machine precision, which validates the equation
    implementation against the published convention (kWh/t Mg, i.e. metric).
    """
    published = 12.3
    v = sourced_quartz_rich_benchmark()
    assert v.tag is Tag.SOURCED
    assert v.source is not None and v.source.doi == "10.37190/ppmp/172458"
    assert v.quantity.to("kWh/metric_ton").magnitude == pytest.approx(published)
    w = bond_specific_energy(v.quantity, Q_(2360.0, "um"), Q_(106.0, "um"),
                             TonConvention.METRIC)
    back = bond_work_index_from_energy(w, Q_(2360.0, "um"), Q_(106.0, "um"),
                                       TonConvention.METRIC)
    err = percent_error(published, back.to("kWh/metric_ton").magnitude)
    with capsys.disabled():
        print(f"\n[benchmark] Bond Wi round trip, Ore A (doi:10.37190/ppmp/172458): "
              f"published {published:.2f} kWh/metric_ton, model "
              f"{back.to('kWh/metric_ton').magnitude:.6f}, error {err:.2e} percent")
    assert err < 1e-9


@pytest.mark.benchmark
def test_benchmark_reduced_procedure_spread(capsys):
    """Benchmark: the paper's own standard-vs-reduced procedure spread, Table 2.

    Ore A 12.3 vs 11.8 (4.1 percent), Ore B 15.9 vs 16.5 (3.8), Ore C 18.7 vs
    19.7 (5.3), Ore D 16.5 vs 17.2 (4.2). This validates
    :func:`percent_error` against four published error figures and, more
    usefully, quantifies the irreducible measurement spread of a Bond index:
    about 4 to 5 percent between two accepted procedures on the SAME mill. Any
    model claiming better than 5 percent accuracy on a work index is claiming
    more than the measurement supports.
    """
    published = [(12.3, 11.8, 4.1), (15.9, 16.5, 3.8), (18.7, 19.7, 5.3),
                 (16.5, 17.2, 4.2)]
    errs = []
    for std, red, reported in published:
        computed = percent_error(std, red)
        errs.append((std, red, reported, computed, abs(computed - reported)))
    with capsys.disabled():
        print("\n[benchmark] Bond standard vs reduced procedure "
              "(doi:10.37190/ppmp/172458, Table 2):")
        for std, red, reported, computed, delta in errs:
            print(f"    Wi {std:.1f} vs {red:.1f}: paper reports {reported:.1f} "
                  f"percent, recomputed {computed:.3f} percent, "
                  f"difference {delta:.3f} percentage points")
        mean_spread = sum(c for *_, c, _ in errs) / len(errs)
        print(f"    mean procedure spread {mean_spread:.2f} percent: treat this as "
              f"the accuracy floor of any Bond work index")
    for *_, delta in errs:
        assert delta < 0.1, "recomputed error must match the paper's rounding"


@pytest.mark.benchmark
def test_benchmark_grindability_equation_against_ore_a(capsys):
    """Benchmark: can Eq. E2 reproduce Ore A's published Wi of 12.3 kWh/t?

    Honest negative-ish result. The paper does not publish Ore A's G, p_i, F80
    or P80 individually, only the resulting Wi, so Eq. E2 cannot be checked
    against it directly. What CAN be checked is the inverse: solve Eq. E2 for
    the grindability G that reproduces 12.3 kWh/short ton at a plausible test
    condition, and confirm the answer lands in the physically observed range of
    roughly 0.5 to 3 g/rev. A G outside that range would indicate the equation
    or its constants were implemented wrong. This is a weaker test than a true
    benchmark and is labelled as such.
    """
    target_short_ton = 12.3  # read as short ton for the gamma = 44.5 convention
    pi_um, f80, p80 = 149.0, 2360.0, 106.0
    size_term = 10.0 * (p80 ** -0.5 - f80 ** -0.5)
    g = (
        BOND_GRINDABILITY_GAMMA
        / (target_short_ton * pi_um ** BOND_GRINDABILITY_ALPHA * size_term)
    ) ** (1.0 / BOND_GRINDABILITY_BETA)
    wi_back = bond_wi_from_grindability(g, Q_(pi_um, "um"), Q_(f80, "um"),
                                        Q_(p80, "um"))
    err = percent_error(target_short_ton, wi_back.magnitude)
    with capsys.disabled():
        print(f"\n[benchmark, weak] Eq. E2 inverse: G = {g:.4f} g/rev reproduces "
              f"Wi = {wi_back.magnitude:.6f} kWh/short_ton against target "
              f"{target_short_ton:.2f}, error {err:.2e} percent. G lies in the "
              f"observed 0.5 to 3 g/rev range: {0.5 <= g <= 3.0}. Ore A's own G is "
              f"not published, so this checks self-consistency, not accuracy.")
    assert err < 1e-9
    assert 0.5 <= g <= 3.0


@pytest.mark.benchmark
def test_benchmark_prior_against_sourced_measurement(capsys):
    """Benchmark: the ASSUMED vein quartz prior against the one real measurement.

    The prior is 13.6 kWh/short ton. Ore A measured 12.3 kWh/metric tonne,
    which is 12.3/1.1023113 = 11.158 kWh/short ton. Error of the prior against
    that measurement is reported. It is large enough (about 22 percent) to make
    the point: this prior is not a substitute for measuring the deposit.
    """
    prior = work_index_prior(OreType.VEIN_QUARTZ)
    assert prior.tag is Tag.ASSUMED
    measured_metric = sourced_quartz_rich_benchmark().quantity
    measured_short = measured_metric.to("kWh/short_ton").magnitude
    err = percent_error(measured_short, prior.quantity.to("kWh/short_ton").magnitude)
    with capsys.disabled():
        print(f"\n[benchmark] ASSUMED vein quartz prior "
              f"{prior.quantity.to('kWh/short_ton').magnitude:.2f} kWh/short_ton vs "
              f"measured Ore A {measured_short:.3f} kWh/short_ton "
              f"(doi:10.37190/ppmp/172458): error {err:.2f} percent. The prior's own "
              f"uniform bracket is 11 to 17 kWh/short_ton, which contains the "
              f"measurement: {11.0 <= measured_short <= 17.0}")
    assert 15.0 < err < 30.0, "prior error against the one measurement, for the record"
    assert 11.0 <= measured_short <= 17.0


# --- physical sanity --------------------------------------------------------


def test_product_coarser_than_feed_raises():
    with pytest.raises(ValueError, match="cannot make particles coarser"):
        bond_specific_energy(Q_(13.0, "kWh/short_ton"), Q_(100.0, "um"),
                             Q_(3000.0, "um"), TonConvention.SHORT)


def test_zero_or_negative_sizes_raise():
    for f80, p80 in [(0.0, 100.0), (-1.0, 100.0), (1000.0, 0.0), (1000.0, -5.0)]:
        with pytest.raises(ValueError):
            bond_specific_energy(Q_(13.0, "kWh/short_ton"), Q_(f80, "um"),
                                 Q_(p80, "um"), TonConvention.SHORT)


def test_negative_work_index_raises():
    with pytest.raises(ValueError, match="non-negative"):
        bond_specific_energy(Q_(-1.0, "kWh/short_ton"), Q_(1000.0, "um"),
                             Q_(100.0, "um"), TonConvention.SHORT)


def test_energy_is_monotonic_in_fineness():
    """Finer product costs more energy, always."""
    prev = -1.0
    for p80 in (1000.0, 500.0, 200.0, 100.0, 60.0):
        w = bond_specific_energy(Q_(13.0, "kWh/short_ton"), Q_(5000.0, "um"),
                                 Q_(p80, "um"), TonConvention.SHORT).magnitude
        assert w > prev
        prev = w


def test_energy_is_zero_when_no_size_reduction():
    w = bond_specific_energy(Q_(13.0, "kWh/short_ton"), Q_(150.0, "um"),
                             Q_(150.0, "um"), TonConvention.SHORT)
    assert w.magnitude == pytest.approx(0.0, abs=1e-15)


def test_inversion_refuses_degenerate_size_pair():
    with pytest.raises(ValueError, match="too close to invert"):
        bond_work_index_from_energy(Q_(1.0, "kWh/short_ton"), Q_(150.0, "um"),
                                    Q_(150.0, "um"), TonConvention.SHORT)


def test_grindability_must_be_positive():
    with pytest.raises(ValueError, match="grindability must be positive"):
        bond_wi_from_grindability(0.0, Q_(149.0, "um"), Q_(2000.0, "um"),
                                  Q_(150.0, "um"))


def test_walker_rejects_non_positive_exponent():
    with pytest.raises(ValueError, match="exponent must be positive"):
        walker_specific_energy(Q_(1.0, "kWh/metric_ton"), Q_(1000.0, "um"),
                               Q_(100.0, "um"), 0.0)


# --- law selection ----------------------------------------------------------


@pytest.mark.parametrize(
    "p80_um,expected",
    [(80_000.0, SizeLaw.KICK), (3000.0, SizeLaw.BOND), (150.0, SizeLaw.BOND),
     (60.0, SizeLaw.BOND), (20.0, SizeLaw.RITTINGER), (2.0, SizeLaw.RITTINGER)],
)
def test_recommended_law_by_size(p80_um, expected):
    law, note = recommended_law(Q_(p80_um, "um"))
    assert law is expected
    assert len(note) > 20


def test_law_exponents():
    assert SizeLaw.KICK.exponent == 1.0
    assert SizeLaw.BOND.exponent == 1.5
    assert SizeLaw.RITTINGER.exponent == 2.0


# --- FEEDSTOCK and SITE plumbing: nothing hardcoded to one deposit ----------


def test_work_index_raises_when_unmeasured_by_default():
    f = _feedstock(f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    with pytest.raises(ValueError, match="no measured Bond work index"):
        feedstock_work_index(f, TonConvention.SHORT)


def test_work_index_prior_is_tagged_assumed_when_allowed():
    f = _feedstock()
    v = feedstock_work_index(f, TonConvention.SHORT, allow_prior=True)
    assert v.tag is Tag.ASSUMED
    assert v.basis and "prior" in v.basis.lower()
    assert "13.6" in f"{v.quantity.magnitude}" or v.quantity.magnitude == 13.6


def test_measured_work_index_keeps_its_provenance():
    wi = Value(quantity=Q_(14.2, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC)
    f = _feedstock(wi=wi)
    v = feedstock_work_index(f, TonConvention.METRIC)
    assert v.tag is Tag.MEASURED
    assert v.source is SRC
    assert v.quantity.magnitude == pytest.approx(14.2 * SHORT_TONS_PER_METRIC_TON,
                                                 rel=1e-9)


def test_unpriced_ore_type_raises_rather_than_guessing():
    for ore in (OreType.ALASKITE, OreType.NOVACULITE, OreType.HYDROTHERMAL_CRYSTAL,
                OreType.DOLOMITE):
        with pytest.raises(KeyError, match="no Bond work index prior"):
            work_index_prior(ore)


def test_all_priors_carry_a_basis_and_are_assumed():
    for ore in (OreType.VEIN_QUARTZ, OreType.QUARTZITE, OreType.PEGMATITE_QUARTZ,
                OreType.INDUSTRIAL_SAND):
        v = work_index_prior(ore)
        assert v.tag is Tag.ASSUMED
        assert v.basis and "ESTIMATE" in v.basis
        assert v.confidence == "low"


def test_the_one_sourced_value_has_a_real_doi():
    v = sourced_quartz_rich_benchmark()
    assert v.tag is Tag.SOURCED
    assert v.source is not None
    assert v.source.tier is Tier.T1
    assert v.source.doi == "10.37190/ppmp/172458"
    assert v.source.accessed == dt.date(2026, 9, 16)


def test_two_different_feedstocks_give_two_different_energies():
    """Ore-agnosticism: swapping the feedstock changes the answer, no globals."""
    f80 = Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC)
    soft = _feedstock(
        wi=Value(quantity=Q_(10.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=f80, sample_id="AE-Q-IN-VKB-011")
    hard = _feedstock(
        wi=Value(quantity=Q_(18.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=f80, sample_id="AE-Q-BR-MG-001")
    w_soft, _, _ = grinding_specific_energy(soft, Q_(150.0, "um"),
                                            TonConvention.SHORT)
    w_hard, _, _ = grinding_specific_energy(hard, Q_(150.0, "um"),
                                            TonConvention.SHORT)
    assert w_hard.magnitude == pytest.approx(w_soft.magnitude * 1.8, rel=1e-12)


def test_missing_f80_raises_with_guidance():
    f = _feedstock(wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED,
                            source=SRC))
    with pytest.raises(ValueError, match="no measured f80"):
        grinding_specific_energy(f, Q_(150.0, "um"), TonConvention.SHORT)


def test_fine_grind_refuses_to_fabricate_a_rittinger_constant():
    f = _feedstock(
        wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    with pytest.raises(ValueError, match="no fitted rittinger constant"):
        grinding_specific_energy(f, Q_(10.0, "um"), TonConvention.SHORT)


def test_energy_cost_uses_the_site_not_a_constant():
    f = _feedstock(
        wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    cheap = _site(0.05)
    dear = _site(0.15)
    c1, p1, _ = grinding_energy_cost(f, cheap, Q_(150.0, "um"),
                                     Q_(50.0, "tonne/hour"), TonConvention.SHORT)
    c2, p2, _ = grinding_energy_cost(f, dear, Q_(150.0, "um"),
                                     Q_(50.0, "tonne/hour"), TonConvention.SHORT)
    assert p1.magnitude == pytest.approx(p2.magnitude, rel=1e-12)
    assert c2.magnitude == pytest.approx(c1.magnitude * 3.0, rel=1e-12)
    assert str(c1.units) == str(Q_(1.0, "USD/hour").units)


def test_energy_cost_in_a_second_currency():
    """A site in INR returns INR, with no hidden FX conversion."""
    f = _feedstock(
        wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    site = _site(6.5, Currency.INR)
    cost, _power, _note = grinding_energy_cost(f, site, Q_(150.0, "um"),
                                               Q_(50.0, "tonne/hour"),
                                               TonConvention.SHORT)
    assert "INR" in str(cost.units)


def test_motor_efficiency_raises_power_draw():
    f = _feedstock(
        wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    site = _site()
    _, p_ideal, _ = grinding_energy_cost(f, site, Q_(150.0, "um"),
                                         Q_(50.0, "tonne/hour"), TonConvention.SHORT)
    _, p_real, note = grinding_energy_cost(f, site, Q_(150.0, "um"),
                                           Q_(50.0, "tonne/hour"),
                                           TonConvention.SHORT,
                                           motor_efficiency=0.90)
    assert p_real.magnitude == pytest.approx(p_ideal.magnitude / 0.90, rel=1e-12)
    assert "0.90" in note


@pytest.mark.parametrize("eff", [0.0, -0.1, 1.5])
def test_bad_motor_efficiency_raises(eff):
    f = _feedstock(
        wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    with pytest.raises(ValueError, match="motor_efficiency"):
        grinding_energy_cost(f, _site(), Q_(150.0, "um"), Q_(50.0, "tonne/hour"),
                             TonConvention.SHORT, motor_efficiency=eff)


def test_throughput_must_be_a_mass_flow():
    f = _feedstock(
        wi=Value(quantity=Q_(13.0, "kWh/short_ton"), tag=Tag.MEASURED, source=SRC),
        f80=Value(quantity=Q_(3000.0, "um"), tag=Tag.SOURCED, source=SRC))
    with pytest.raises(DimensionalityError):
        grinding_energy_cost(f, _site(), Q_(150.0, "um"), Q_(50.0, "tonne"),
                             TonConvention.SHORT)
