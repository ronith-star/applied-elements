"""Liberation stereology tests: geometry, limits, worked examples, benchmarks."""
import datetime as dt
import itertools
import math

import pytest

from ae.core.feedstock import (
    Feedstock,
    InclusionCharacter,
    InclusionType,
    OreType,
)
from ae.core.provenance import MISSING, Source, Tag, Tier, Value
from ae.core.units import Q_, DimensionalityError
from ae.physics.liberation import (
    SRC_KING_1979,
    SRC_LIN_2020,
    SRC_XIA_2024,
    InclusionWeighting,
    enclosed_fraction,
    energy_to_exposure_ratio,
    exposure,
    exposure_curve,
    feedstock_exposure,
    leachable_fraction,
    liberation_size,
    polydisperse_exposure,
)

SRC = Source(citation="Test source 2026", tier=Tier.T1, doi="10.1000/test",
             accessed=dt.date(2026, 9, 16))


def _feedstock(inclusion_size_um: float | None = None,
               sample_id: str = "AE-Q-IN-VKB-020") -> Feedstock:
    inc = InclusionCharacter(
        median_inclusion_size=(
            Value(quantity=Q_(inclusion_size_um, "um"), tag=Tag.MEASURED, source=SRC)
            if inclusion_size_um is not None
            else MISSING
        )
    )
    return Feedstock(sample_id=sample_id, ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="Test", country="IN", inclusions=inc)


# --- (c) dimensional analysis -----------------------------------------------


def test_sizes_must_be_lengths():
    with pytest.raises(DimensionalityError):
        exposure(Q_(100.0, "kg"), Q_(10.0, "um"))
    with pytest.raises(DimensionalityError):
        exposure(Q_(100.0, "um"), Q_(10.0, "s"))


def test_exposure_is_dimensionless_and_unit_agnostic():
    """The model depends only on the size RATIO, so units must cancel."""
    a = exposure(Q_(100.0, "um"), Q_(10.0, "um"))
    b = exposure(Q_(0.1, "mm"), Q_(10.0, "um"))
    c = exposure(Q_(1e-4, "m"), Q_(1e-5, "m"))
    assert isinstance(a, float)
    assert a == pytest.approx(b, rel=1e-12)
    assert a == pytest.approx(c, rel=1e-12)


def test_liberation_size_returns_a_length():
    d = liberation_size(Q_(20.0, "um"), 0.5)
    assert d.to("mm").magnitude == pytest.approx(0.09695, abs=1e-5)


# --- (d) golden worked examples ---------------------------------------------


@pytest.mark.golden
def test_exposure_worked_example_ten_percent_ratio():
    """d_inc = 10 um in d_p = 100 um: r = 0.1.

    Enclosed = (1 - 0.1)^3 = 0.9^3 = 0.729.
    E = 1 - 0.729 = 0.271.
    """
    assert enclosed_fraction(Q_(100.0, "um"), Q_(10.0, "um")) == pytest.approx(
        0.729, abs=1e-12
    )
    assert exposure(Q_(100.0, "um"), Q_(10.0, "um")) == pytest.approx(0.271, abs=1e-12)


@pytest.mark.golden
def test_exposure_worked_example_one_third_ratio():
    """d_inc = 10 um in d_p = 30 um: r = 1/3.

    Enclosed = (2/3)^3 = 8/27 = 0.2962963.
    E = 1 - 8/27 = 19/27 = 0.7037037.
    """
    e = exposure(Q_(30.0, "um"), Q_(10.0, "um"))
    assert e == pytest.approx(19.0 / 27.0, abs=1e-12)
    assert e == pytest.approx(0.7037037, abs=1e-7)


@pytest.mark.golden
def test_small_ratio_expansion_gives_factor_three():
    """For small r, E -> 3r: three orthogonal faces each offer one intersection.

    At r = 0.001: E = 1 - 0.999^3 = 1 - 0.997002999 = 0.002997001, and
    3r = 0.003, so the ratio E/(3r) = 0.999000333, within 0.1 percent of 1.
    """
    r = 0.001
    e = exposure(Q_(1000.0, "um"), Q_(1.0, "um"))
    assert e == pytest.approx(0.002997001, abs=1e-9)
    assert e / (3.0 * r) == pytest.approx(0.999000333, abs=1e-9)


@pytest.mark.golden
def test_liberation_size_half_exposure():
    """E* = 0.5: 0.5^(1/3) = 0.7937005, 1 - 0.7937005 = 0.2062995.

    d_p* = 20 / 0.2062995 = 96.94644 um, i.e. 4.847322 times the inclusion size.
    """
    d = liberation_size(Q_(20.0, "um"), 0.5)
    assert d.magnitude == pytest.approx(96.94644, abs=1e-5)
    assert d.magnitude / 20.0 == pytest.approx(4.847322, abs=1e-6)


@pytest.mark.golden
def test_liberation_size_ninety_percent_exposure():
    """E* = 0.9: 0.1^(1/3) = 0.4641589, 1 - 0.4641589 = 0.5358411.

    d_p* = 20 / 0.5358411 = 37.32450 um, i.e. 1.866225 times the inclusion size.
    Going from 50 to 90 percent exposure needs a 96.94644/37.32450 = 2.597394
    fold finer grind, which by Bond's law costs sqrt(2.597394) = 1.611643,
    about 61 percent more grinding energy per tonne.
    """
    d90 = liberation_size(Q_(20.0, "um"), 0.9)
    d50 = liberation_size(Q_(20.0, "um"), 0.5)
    assert d90.magnitude == pytest.approx(37.32450, abs=1e-5)
    ratio = d50.magnitude / d90.magnitude
    assert ratio == pytest.approx(2.597394, abs=1e-6)
    assert math.sqrt(ratio) == pytest.approx(1.611643, abs=1e-6)


@pytest.mark.golden
def test_liberation_size_round_trips_through_exposure():
    """liberation_size and exposure are exact inverses."""
    for target in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        d = liberation_size(Q_(7.5, "um"), target)
        assert exposure(d, Q_(7.5, "um")) == pytest.approx(target, rel=1e-12)


@pytest.mark.golden
def test_full_exposure_requires_grinding_to_the_inclusion_size():
    """E* = 1 gives d_p* = d_inc exactly: the geometric asymptote."""
    d = liberation_size(Q_(12.0, "um"), 1.0)
    assert d.magnitude == pytest.approx(12.0, rel=1e-12)


@pytest.mark.golden
def test_polydisperse_worked_example():
    """Three classes at d_p = 100 um: 2, 10 and 40 um, weights 0.5, 0.3, 0.2.

    E(2) = 1 - 0.98^3 = 1 - 0.941192 = 0.058808.
    E(10) = 1 - 0.9^3 = 0.271.
    E(40) = 1 - 0.6^3 = 1 - 0.216 = 0.784.
    Mean = 0.5(0.058808) + 0.3(0.271) + 0.2(0.784)
         = 0.029404 + 0.0813 + 0.1568 = 0.267504.
    """
    e = polydisperse_exposure(
        Q_(100.0, "um"),
        [Q_(2.0, "um"), Q_(10.0, "um"), Q_(40.0, "um")],
        [0.5, 0.3, 0.2],
        InclusionWeighting.VOLUME,
    )
    assert e == pytest.approx(0.267504, abs=1e-9)


@pytest.mark.golden
def test_leachable_fraction_worked_example():
    """Partition surface 0.05, fluid 0.20, mineral 0.30, lattice 0.45.

    At d_p = 100 um with d_fluid = 5 um and d_mineral = 20 um:
    E_fluid = 1 - 0.95^3 = 1 - 0.857375 = 0.142625.
    E_mineral = 1 - 0.8^3 = 1 - 0.512 = 0.488.
    f = 0.05 + 0.20(0.142625) + 0.30(0.488)
      = 0.05 + 0.028525 + 0.1464 = 0.224925.
    The ceiling is 1 - 0.45 = 0.55, so the result is well inside it: at this
    grind only 22.5 percent of the element is reachable although 55 percent is
    non-lattice.
    """
    f = leachable_fraction(
        {"surface": 0.05, "fluid": 0.20, "mineral": 0.30, "lattice": 0.45},
        Q_(100.0, "um"), Q_(5.0, "um"), Q_(20.0, "um"),
    )
    assert f == pytest.approx(0.224925, abs=1e-9)
    assert f < 1.0 - 0.45


@pytest.mark.golden
def test_energy_to_exposure_worked_example():
    """From 200 to 50 um with 10 um inclusions.

    E(200) = 1 - 0.95^3 = 0.142625.
    E(50) = 1 - 0.8^3 = 0.488.
    gain = 0.488 - 0.142625 = 0.345375.
    Bond energy ratio = sqrt(200/50) = sqrt(4) = 2.
    exposure per unit energy = 0.345375/2 = 0.1726875.
    """
    gain, ratio, per_energy = energy_to_exposure_ratio(
        Q_(200.0, "um"), Q_(50.0, "um"), Q_(10.0, "um")
    )
    assert gain == pytest.approx(0.345375, abs=1e-9)
    assert ratio == pytest.approx(2.0, abs=1e-12)
    assert per_energy == pytest.approx(0.1726875, abs=1e-9)


# --- (e) benchmarks, with error reported ------------------------------------


@pytest.mark.benchmark
def test_benchmark_grind_implied_by_xia_removal(capsys):
    """Benchmark against Xia et al. 2024 (doi:10.3390/min14070727).

    The paper reports 81.20 percent removal of total trace impurities
    (128.86 to 24.23 ug/g) with the residue identified as lattice-bound Al, Ti
    and Li. It does NOT report a liberation curve or a per-stage breakdown, so
    the geometric model cannot be validated against its intermediates and this
    benchmark does not pretend to.

    What can be tested is a NECESSARY CONDITION. If 81.20 percent of the trace
    inventory was removed and the residue is lattice, then at minimum 81.20
    percent of the inventory had to be exposed. Inverting Eq. E2 for
    E* = 0.8120 gives the required particle-to-inclusion size ratio:
    1 - (1 - 0.8120)^(1/3) = 1 - 0.188^(1/3) = 1 - 0.5728654 = 0.4271346, so
    d_p* = 2.3412 d_inc (the test's own f-string prints 2.3414, computed from
    the unrounded removal fraction 0.81196648 rather than the rounded 0.8120). The paper's flowsheet includes calcination and water
    quenching, which opens fluid inclusions thermally rather than geometrically,
    so the real flowsheet is NOT bound by this ratio. The test therefore reports
    the ratio as the grinding-only requirement and flags the discrepancy, rather
    than claiming agreement.
    """
    removal = 1.0 - 24.23 / 128.86
    d_inc = Q_(10.0, "um")
    d_required = liberation_size(d_inc, removal)
    ratio = d_required.magnitude / 10.0
    achieved = exposure(d_required, d_inc)
    err = abs(achieved - removal) / removal * 100.0
    with capsys.disabled():
        print(f"\n[benchmark] Xia et al. 2024 (doi:10.3390/min14070727): reported "
              f"removal {removal * 100:.2f} percent (128.86 to 24.23 ug/g). "
              f"Grinding-only exposure of that fraction requires "
              f"d_p = {ratio:.4f} x d_inc; model exposure at that size is "
              f"{achieved * 100:.4f} percent, inversion error {err:.2e} percent. "
              f"CAVEAT: the paper's flowsheet includes calcination plus water "
              f"quenching, which opens fluid inclusions thermally, so the actual "
              f"plant is not bound by this geometric ratio. This validates the "
              f"inversion, not the physics of that campaign.")
    assert err < 1e-9
    assert removal == pytest.approx(0.8120, abs=5e-5)
    assert 2.0 < ratio < 3.0


@pytest.mark.benchmark
def test_benchmark_qu_2025_removal_requires_finer_grind(capsys):
    """Benchmark: Qu et al. 2025 (doi:10.5194/ejm-37-953-2025), two ore bodies.

    HT quartz 322.96 to 18.72 ug/g (94.20 percent removal), PX quartz 4944.73
    to 114.56 ug/g (97.68 percent removal), both by a flowsheet including
    calcination, crushing, sieving, magnetic separation, gravity separation,
    flotation and acid leaching. Higher removal demands more exposure and hence
    a finer geometric grind: the required particle-to-inclusion ratios are
    computed and reported. The monotonic ordering (higher removal implies finer
    required grind) is the physical content being checked.
    """
    cases = [("HT", 322.96, 18.72), ("PX", 4944.73, 114.56),
             ("Xia vein quartz", 128.86, 24.23)]
    rows = []
    for name, feed, prod in cases:
        removal = 1.0 - prod / feed
        ratio = liberation_size(Q_(10.0, "um"), removal).magnitude / 10.0
        rows.append((name, removal, ratio))
    with capsys.disabled():
        print("\n[benchmark] required grinding-only size ratio vs published removal:")
        for name, removal, ratio in rows:
            print(f"    {name:>16}: removal {removal * 100:6.2f} percent -> "
                  f"d_p/d_inc = {ratio:.4f}")
        print("    sources: doi:10.5194/ejm-37-953-2025 (HT, PX), "
              "doi:10.3390/min14070727 (Xia). All three flowsheets include "
              "calcination, so these ratios are upper bounds on the grind needed, "
              "not predictions of it.")
    by_removal = sorted(rows, key=lambda r: r[1])
    ratios = [r[2] for r in by_removal]
    assert ratios == sorted(ratios, reverse=True), (
        "higher removal must require a finer (smaller ratio) grind"
    )


# --- physical sanity and limits ---------------------------------------------


def test_exposure_bounds():
    assert exposure(Q_(1000.0, "um"), Q_(0.5, "um")) < 0.01
    assert exposure(Q_(10.0, "um"), Q_(10.0, "um")) == pytest.approx(1.0)
    assert exposure(Q_(5.0, "um"), Q_(10.0, "um")) == pytest.approx(1.0)


def test_exposure_is_monotonic_decreasing_in_particle_size():
    prev = 2.0
    for dp in (10.0, 20.0, 50.0, 100.0, 500.0, 2000.0):
        e = exposure(Q_(dp, "um"), Q_(10.0, "um"))
        assert e < prev
        prev = e


def test_exposure_curve_checks_monotonicity():
    pts = exposure_curve(Q_(10.0, "um"),
                         [Q_(x, "um") for x in (500.0, 50.0, 200.0, 20.0, 100.0)])
    assert [p[0] for p in pts] == [20.0, 50.0, 100.0, 200.0, 500.0]
    assert all(a[1] > b[1] for a, b in itertools.pairwise(pts))


@pytest.mark.parametrize("dp,di", [(0.0, 10.0), (-5.0, 10.0), (100.0, 0.0),
                                   (100.0, -1.0)])
def test_non_positive_sizes_raise(dp, di):
    with pytest.raises(ValueError):
        exposure(Q_(dp, "um"), Q_(di, "um"))


def test_zero_target_exposure_raises():
    with pytest.raises(ValueError, match="must be positive"):
        liberation_size(Q_(10.0, "um"), 0.0)


def test_target_exposure_above_one_raises():
    with pytest.raises(ValueError, match=r"\[0.0, 1.0\]"):
        liberation_size(Q_(10.0, "um"), 1.2)


def test_polydisperse_weights_must_close():
    with pytest.raises(ValueError, match="must sum to 1"):
        polydisperse_exposure(Q_(100.0, "um"), [Q_(5.0, "um"), Q_(20.0, "um")],
                              [0.5, 0.4], InclusionWeighting.NUMBER)


def test_polydisperse_weights_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        polydisperse_exposure(Q_(100.0, "um"), [Q_(5.0, "um"), Q_(20.0, "um")],
                              [1.2, -0.2], InclusionWeighting.NUMBER)


def test_polydisperse_length_mismatch_raises():
    with pytest.raises(ValueError, match="size classes but"):
        polydisperse_exposure(Q_(100.0, "um"), [Q_(5.0, "um")], [0.5, 0.5],
                              InclusionWeighting.NUMBER)


def test_polydisperse_single_class_equals_monodisperse():
    a = polydisperse_exposure(Q_(100.0, "um"), [Q_(10.0, "um")], [1.0],
                              InclusionWeighting.VOLUME)
    assert a == pytest.approx(exposure(Q_(100.0, "um"), Q_(10.0, "um")), rel=1e-12)


def test_number_and_volume_weighting_both_accepted_and_recorded():
    """The basis does not change the arithmetic, but must be declared."""
    args = (Q_(100.0, "um"), [Q_(5.0, "um"), Q_(40.0, "um")], [0.7, 0.3])
    n = polydisperse_exposure(*args, InclusionWeighting.NUMBER)
    v = polydisperse_exposure(*args, InclusionWeighting.VOLUME)
    assert n == pytest.approx(v, rel=1e-12)
    assert set(InclusionWeighting) == {InclusionWeighting.NUMBER,
                                       InclusionWeighting.VOLUME}


def test_leachable_partition_must_close():
    with pytest.raises(ValueError, match="sum to"):
        leachable_fraction({"surface": 0.1, "fluid": 0.1, "mineral": 0.1,
                            "lattice": 0.1}, Q_(100.0, "um"), Q_(5.0, "um"),
                           Q_(20.0, "um"))


def test_leachable_partition_requires_all_four_locations():
    with pytest.raises(KeyError, match="missing"):
        leachable_fraction({"surface": 0.5, "lattice": 0.5}, Q_(100.0, "um"),
                           Q_(5.0, "um"), Q_(20.0, "um"))


def test_leachable_partition_rejects_unknown_keys():
    with pytest.raises(KeyError, match="unknown partition keys"):
        leachable_fraction({"surface": 0.25, "fluid": 0.25, "mineral": 0.25,
                            "lattice": 0.2, "grain_boundary": 0.05},
                           Q_(100.0, "um"), Q_(5.0, "um"), Q_(20.0, "um"))


def test_leachable_fraction_never_exceeds_the_non_lattice_inventory():
    """Even at infinite fineness, the lattice bounds the leachable fraction."""
    part = {"surface": 0.1, "fluid": 0.2, "mineral": 0.3, "lattice": 0.4}
    f_fine = leachable_fraction(part, Q_(1.0, "um"), Q_(5.0, "um"), Q_(20.0, "um"))
    assert f_fine == pytest.approx(0.6, rel=1e-12)
    assert f_fine <= 1.0 - part["lattice"] + 1e-12


def test_all_lattice_means_nothing_is_leachable():
    f = leachable_fraction({"surface": 0.0, "fluid": 0.0, "mineral": 0.0,
                            "lattice": 1.0}, Q_(1.0, "um"), Q_(5.0, "um"),
                           Q_(20.0, "um"))
    assert f == pytest.approx(0.0, abs=1e-15)


def test_energy_ratio_requires_a_finer_second_size():
    with pytest.raises(ValueError, match="must be smaller"):
        energy_to_exposure_ratio(Q_(50.0, "um"), Q_(200.0, "um"), Q_(10.0, "um"))


def test_exposure_per_energy_has_an_interior_optimum():
    """Exposure per unit grinding energy peaks at a finite grind, then falls.

    This is the decision-relevant behaviour and it is NOT simple monotonic
    decline, which is worth stating because the naive expectation is wrong.
    Exposure per unit Bond energy, measured from a 200 um baseline with 10 um
    inclusions, runs 0.090775 (to 100 um), 0.172687 (50 um), 0.226760 (25 um),
    0.212344 (12.5 um), 0.196425 (10.5 um): it RISES while exposure is still
    climbing steeply, peaks near d_p of roughly 2.5 d_inc, and only then falls
    as exposure saturates at 1 while Bond energy keeps diverging as
    P80^(-1/2). The optimum grind is interior, which is exactly why a grind
    target is a decision rather than "as fine as affordable".
    """
    fines = [100.0, 50.0, 25.0, 12.5, 10.5]
    per_energy = [
        energy_to_exposure_ratio(Q_(200.0, "um"), Q_(f, "um"), Q_(10.0, "um"))[2]
        for f in fines
    ]
    assert per_energy[0] == pytest.approx(0.090775, abs=1e-6)
    assert per_energy[2] == pytest.approx(0.226760, abs=1e-6)
    assert per_energy[-1] == pytest.approx(0.196425, abs=1e-6)
    peak = max(range(len(per_energy)), key=lambda i: per_energy[i])
    assert 0 < peak < len(per_energy) - 1, "the optimum must be interior"
    assert per_energy[peak] > per_energy[-1]


def test_marginal_exposure_per_halving_eventually_falls():
    """Successive halvings cost the same Bond energy ratio but buy less exposure.

    Each halving costs a factor sqrt(2) = 1.414214 in Bond energy. The exposure
    bought per halving for 10 um inclusions runs 0.069484 (400 to 200 um),
    0.128375, 0.217000, 0.296000, then 0.208000 (25 to 12.5 um): the last step
    buys less than the one before it because exposure is saturating.
    """
    sizes = [400.0, 200.0, 100.0, 50.0, 25.0, 12.5]
    gains = [
        energy_to_exposure_ratio(Q_(a, "um"), Q_(b, "um"), Q_(10.0, "um"))[0]
        for a, b in itertools.pairwise(sizes)
    ]
    assert gains[0] == pytest.approx(0.069484, abs=1e-6)
    assert gains[3] == pytest.approx(0.296000, abs=1e-6)
    assert gains[-1] == pytest.approx(0.208000, abs=1e-6)
    assert gains[-1] < gains[-2], "exposure gain per halving must eventually fall"


# --- FEEDSTOCK plumbing -----------------------------------------------------


def test_feedstock_exposure_raises_when_inclusion_size_unmeasured():
    f = _feedstock(None)
    with pytest.raises(ValueError, match="no measured median inclusion size"):
        feedstock_exposure(f, Q_(100.0, "um"))


def test_feedstock_exposure_uses_the_measured_size_and_returns_provenance():
    f = _feedstock(10.0)
    e, v = feedstock_exposure(f, Q_(100.0, "um"))
    assert e == pytest.approx(0.271, abs=1e-12)
    assert v.tag is Tag.MEASURED
    assert v.source is SRC


def test_two_feedstocks_with_different_inclusions_give_different_exposure():
    coarse = _feedstock(40.0, sample_id="AE-Q-IN-VKB-021")
    fine = _feedstock(2.0, sample_id="AE-Q-NO-DRG-001")
    e_coarse, _ = feedstock_exposure(coarse, Q_(100.0, "um"))
    e_fine, _ = feedstock_exposure(fine, Q_(100.0, "um"))
    assert e_coarse == pytest.approx(0.784, abs=1e-12)
    assert e_fine == pytest.approx(0.058808, abs=1e-9)
    assert e_coarse > e_fine


def test_melt_inclusions_are_refused_rather_than_modelled():
    f = _feedstock(10.0)
    with pytest.raises(TypeError, match="melt inclusions are not modelled"):
        feedstock_exposure(f, Q_(100.0, "um"), InclusionType.MELT)


@pytest.mark.parametrize("kind", [InclusionType.FLUID, InclusionType.MINERAL])
def test_fluid_and_mineral_inclusions_both_supported(kind):
    f = _feedstock(10.0)
    e, _ = feedstock_exposure(f, Q_(100.0, "um"), kind)
    assert e == pytest.approx(0.271, abs=1e-12)


# --- provenance discipline --------------------------------------------------


def test_sources_are_real_and_dated():
    for src in (SRC_LIN_2020, SRC_KING_1979, SRC_XIA_2024):
        assert src.tier is Tier.T1
        assert src.doi
        assert src.accessed == dt.date(2026, 9, 16)
        assert src.note


def test_king_source_is_flagged_as_not_implemented():
    """The calibrated liberation model is CITED as absent, not claimed."""
    assert "does NOT implement" in (SRC_KING_1979.note or "")


def test_xia_source_disclaims_per_stage_data():
    assert "NO per-stage" in (SRC_XIA_2024.note or "")
