"""Impurity location tests: partition closure, the purity ceiling, HPQ gating.

The central behavioural requirement tested here is that the model RAISES rather
than guesses when the four-way split is unmeasured, because that split sets the
achievable purity floor and the Vikarabad deposit is uncharacterized.
"""
import datetime as dt
import math

import pytest

from ae.core.feedstock import ELEMENTS, Feedstock, ImpurityProfile, OreType
from ae.core.provenance import MissingValueError, Source, Tag, Tier, Value
from ae.core.units import Q_
from ae.physics.impurity_location import (
    HPQ_SINGLE_GRAIN_LIMITS_PPM,
    HPQ_TRACE_SUM_LIMIT_PPM,
    NO_PARTITION_DATA,
    PARTITION_PRIORS,
    SRC_LIN_2020,
    SRC_LIU_2026,
    SRC_MULLER_2012,
    SRC_QU_2025,
    SRC_XIA_2024,
    ElementPartition,
    Location,
    PartitionModel,
    floor_concentration,
    floor_profile,
    hpq_gate,
    implied_removal_rate,
    lattice_ceiling_sio2_percent,
    location_concentrations,
    partition_fractions,
    removable_ppm,
)

SRC = Source(citation="Test source 2026", tier=Tier.T1, doi="10.1000/test",
             accessed=dt.date(2026, 9, 16))


def _v(x: float, unit: str = "ppm_mass") -> Value:
    return Value(quantity=Q_(x, unit), tag=Tag.SOURCED, source=SRC)


def _frac(x: float) -> Value:
    return Value(quantity=Q_(x, "dimensionless"), tag=Tag.MEASURED, source=SRC)


def _feedstock(totals: dict[str, float],
               lattice: dict[str, float] | None = None,
               ore_type: OreType = OreType.VEIN_QUARTZ,
               sample_id: str = "AE-Q-IN-VKB-030") -> Feedstock:
    prof = ImpurityProfile(
        total={k: _v(x) for k, x in totals.items()},
        lattice_fraction={k: _frac(x) for k, x in (lattice or {}).items()},
        method="LA_ICP_MS",
        basis_material="single_grain",
    )
    return Feedstock(sample_id=sample_id, ore_type=ore_type, deposit_name="Test",
                     country="IN", impurities=prof)


def _partition(element: str, s: float, f: float, m: float, l: float,
               method: str = "LA_ICP_MS_inclusion_free_domains") -> ElementPartition:
    return ElementPartition(
        element=element,
        surface=_frac(s), fluid=_frac(f), mineral=_frac(m), lattice=_frac(l),
        method=method,  # type: ignore[arg-type]
    )


# --- (c) dimensional and schema enforcement ---------------------------------


def test_partition_must_close_to_one():
    with pytest.raises(ValueError, match="must be fully accounted for"):
        _partition("Al", 0.1, 0.1, 0.1, 0.1)


def test_partition_closes_within_tolerance():
    p = _partition("Al", 0.25, 0.25, 0.25, 0.25)
    assert math.fsum(p.fractions.values()) == pytest.approx(1.0, abs=1e-12)


def test_partition_rejects_fraction_outside_unit_interval():
    with pytest.raises(ValueError, match="outside"):
        ElementPartition(element="Al", surface=_frac(1.5), fluid=_frac(-0.2),
                         mineral=_frac(0.0), lattice=_frac(-0.3),
                         method="LA_ICP_MS_inclusion_free_domains")


def test_partition_rejects_unknown_element():
    with pytest.raises(ValueError, match="unknown element"):
        _partition("Xx", 0.25, 0.25, 0.25, 0.25)


def test_partition_fractions_are_dimensionless():
    p = _partition("Ti", 0.1, 0.1, 0.2, 0.6)
    for v in (p.surface, p.fluid, p.mineral, p.lattice):
        assert str(v.quantity.units) == "dimensionless"


def test_weakest_tag_is_reported():
    p = ElementPartition(
        element="Al",
        surface=Value(quantity=Q_(0.1, "dimensionless"), tag=Tag.MEASURED, source=SRC),
        fluid=Value(quantity=Q_(0.1, "dimensionless"), tag=Tag.SOURCED, source=SRC),
        mineral=Value(quantity=Q_(0.3, "dimensionless"), tag=Tag.ASSUMED,
                      basis="test estimate"),
        lattice=Value(quantity=Q_(0.5, "dimensionless"), tag=Tag.MEASURED, source=SRC),
        method="CL_plus_LA_ICP_MS",
    )
    assert p.weakest_tag is Tag.ASSUMED


def test_location_removability_flags():
    assert Location.SURFACE.removable_by_physical_flowsheet
    assert Location.FLUID_INCLUSION.removable_by_physical_flowsheet
    assert Location.MINERAL_INCLUSION.removable_by_physical_flowsheet
    assert not Location.LATTICE.removable_by_physical_flowsheet


# --- (d) golden worked examples ---------------------------------------------


@pytest.mark.golden
def test_location_concentrations_worked_example():
    """Al at 120 ppm bulk, split 0.05/0.05/0.45/0.45.

    surface = 120 x 0.05 = 6.0 ppm
    fluid   = 120 x 0.05 = 6.0 ppm
    mineral = 120 x 0.45 = 54.0 ppm
    lattice = 120 x 0.45 = 54.0 ppm
    Sum = 6 + 6 + 54 + 54 = 120.0 ppm, closing exactly on the bulk assay.
    """
    f = _feedstock({"Al": 120.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.05, 0.05, 0.45, 0.45)})
    conc = location_concentrations(f, "Al", model=model)
    assert conc["surface"] == pytest.approx(6.0, abs=1e-12)
    assert conc["fluid"] == pytest.approx(6.0, abs=1e-12)
    assert conc["mineral"] == pytest.approx(54.0, abs=1e-12)
    assert conc["lattice"] == pytest.approx(54.0, abs=1e-12)
    assert math.fsum(conc.values()) == pytest.approx(120.0, abs=1e-12)


@pytest.mark.golden
def test_perfect_flowsheet_floor_is_the_lattice_inventory():
    """With eta = 1 on surface, fluid and mineral, the floor IS the lattice.

    Al 120 ppm with lattice fraction 0.45: floor = 120 x 0.45 = 54.0 ppm.
    The HPQ limit is 30 ppm, so this ore cannot make HPQ sand no matter the
    flowsheet, which is the entire point of the model.
    """
    f = _feedstock({"Al": 120.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.05, 0.05, 0.45, 0.45)})
    assert 120.0 * 0.45 == pytest.approx(54.0, abs=1e-12)
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Al"] == pytest.approx(30.0, abs=1e-12)
    floor = floor_concentration(f, "Al", model=model)
    assert floor == pytest.approx(54.0, abs=1e-12)
    assert floor > HPQ_SINGLE_GRAIN_LIMITS_PPM["Al"]
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Al"] == pytest.approx(30.0, abs=1e-12)


@pytest.mark.golden
def test_imperfect_flowsheet_floor_worked_example():
    """Al 120 ppm, split 0.05/0.05/0.45/0.45, eta_s = 0.9, eta_f = 0.5, eta_m = 0.8.

    surface residue = 6.0 x (1 - 0.9) = 0.6 ppm
    fluid residue   = 6.0 x (1 - 0.5) = 3.0 ppm
    mineral residue = 54.0 x (1 - 0.8) = 10.8 ppm
    lattice residue = 54.0 x (1 - 0) = 54.0 ppm
    floor = 0.6 + 3.0 + 10.8 + 54.0 = 68.4 ppm
    """
    f = _feedstock({"Al": 120.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.05, 0.05, 0.45, 0.45)})
    assert 120.0 * 0.05 == pytest.approx(6.0, abs=1e-12)
    assert 120.0 * 0.45 == pytest.approx(54.0, abs=1e-12)
    assert 6.0 * (1.0 - 0.9) == pytest.approx(0.6, abs=1e-12)
    assert 6.0 * (1.0 - 0.5) == pytest.approx(3.0, abs=1e-12)
    assert 54.0 * (1.0 - 0.8) == pytest.approx(10.8, abs=1e-12)
    assert 54.0 * (1.0 - 0.0) == pytest.approx(54.0, abs=1e-12)
    assert 0.6 + 3.0 + 10.8 + 54.0 == pytest.approx(68.4, abs=1e-12)
    floor = floor_concentration(f, "Al", model=model,
                                efficiencies={"surface": 0.9, "fluid": 0.5,
                                              "mineral": 0.8})
    assert floor == pytest.approx(68.4, abs=1e-12)
    # Every residue line of the worked example, derived from the split and the
    # efficiencies rather than restated.
    surface_ppm = 120.0 * 0.05
    fluid_ppm = 120.0 * 0.05
    mineral_ppm = 120.0 * 0.45
    assert surface_ppm == pytest.approx(6.0, abs=1e-12)
    assert mineral_ppm == pytest.approx(54.0, abs=1e-12)
    assert surface_ppm * (1.0 - 0.9) == pytest.approx(0.6, abs=1e-12)
    assert fluid_ppm * (1.0 - 0.5) == pytest.approx(3.0, abs=1e-12)
    assert mineral_ppm * (1.0 - 0.8) == pytest.approx(10.8, abs=1e-12)
    # The four residues must sum to the floor the model returns.
    assert (surface_ppm * (1.0 - 0.9) + fluid_ppm * (1.0 - 0.5)
            + mineral_ppm * (1.0 - 0.8) + mineral_ppm) == pytest.approx(
        floor, abs=1e-12)


@pytest.mark.golden
def test_removable_ppm_worked_example():
    """Al 120 ppm with lattice 0.45: removable = 120 x (1 - 0.45) = 66.0 ppm."""
    f = _feedstock({"Al": 120.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.05, 0.05, 0.45, 0.45)})
    assert removable_ppm(f, "Al", model=model) == pytest.approx(66.0, abs=1e-12)


@pytest.mark.golden
def test_implied_removal_rate_worked_example():
    """Xia et al. 2024: 1 - 24.23/128.86.

    24.23/128.86 = 0.1880335, 1 - 0.1880335 = 0.8119665, i.e. 81.1966 percent,
    which the paper rounds to 81.20 percent.
    """
    assert 24.23 / 128.86 == pytest.approx(0.1880335, abs=1e-7)
    assert 1.0 - 0.1880335 == pytest.approx(0.8119665, abs=1e-7)
    r = implied_removal_rate(128.86, 24.23)
    assert 100.0 * r == pytest.approx(81.1966, abs=1e-4)
    assert r == pytest.approx(0.8119665, abs=1e-7)
    assert round(r * 100, 2) == 81.2
    ratio = 24.23 / 128.86
    assert ratio == pytest.approx(0.1880335, abs=1e-7)
    percent = r * 100
    assert percent == pytest.approx(81.1966, abs=1e-4)


@pytest.mark.golden
def test_sio2_from_trace_sum_worked_example():
    """24.23 ppm residual trace = 24.23e-4 wt percent.

    100 - 0.002423 = 99.997577 wt percent SiO2, which rounds to the 99.998 wt
    percent the paper reports.
    """
    # ppm to wt percent is a division by 1e4 by definition (1 percent = 1e4
    # ppm). The percent figure is DERIVED from the ppm figure, not restated:
    # an assertion of 24.23e-4 against 0.002423 compares one literal with
    # itself in different notation and verifies nothing.
    trace_wt_percent = 24.23 / 1e4
    assert trace_wt_percent == pytest.approx(0.002423, abs=1e-12)
    assert 100.0 - trace_wt_percent == pytest.approx(99.997577, abs=1e-9)
    s = lattice_ceiling_sio2_percent(24.23)
    assert s == pytest.approx(99.997577, abs=1e-6)
    assert round(s, 3) == 99.998
    trace_ppm = 24.23
    trace_percent = trace_ppm / 1e4
    assert trace_percent == pytest.approx(24.23e-4, abs=1e-9)
    assert trace_percent == pytest.approx(0.002423, abs=1e-9)
    baseline = 100.0
    assert baseline - trace_percent == pytest.approx(s, abs=1e-6)


@pytest.mark.golden
def test_hpq_gate_worked_example_pass():
    """Floors Al 12, Ti 4, Li 2, Fe 1 ppm against limits 30, 10, 5, 3.

    Every element clears. Trace sum = 12 + 4 + 2 + 1 = 19 ppm against the
    50 ppm limit, so the gate passes.
    """
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Ti"] == pytest.approx(10.0, abs=1e-12)
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Li"] == pytest.approx(5.0, abs=1e-12)
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Fe"] == pytest.approx(3.0, abs=1e-12)
    assert HPQ_TRACE_SUM_LIMIT_PPM == pytest.approx(50.0, abs=1e-12)
    assert 12.0 + 4.0 + 2.0 + 1.0 == pytest.approx(19.0, abs=1e-12)
    passes, per, verdict = hpq_gate({"Al": 12.0, "Ti": 4.0, "Li": 2.0, "Fe": 1.0})
    assert passes
    assert per["Al"] == (12.0, 30.0, True)
    assert "19.00" in verdict
    assert per["Ti"] == (4.0, 10.0, True)
    assert per["Li"] == (2.0, 5.0, True)
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Ti"] == pytest.approx(10.0, abs=1e-9)
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM["Li"] == pytest.approx(5.0, abs=1e-9)
    assert HPQ_TRACE_SUM_LIMIT_PPM == pytest.approx(50.0, abs=1e-9)


@pytest.mark.golden
def test_hpq_gate_worked_example_one_element_fails():
    """A single element out of specification disqualifies the material.

    Floors Al 45, Ti 1, Li 0.5, Fe 0.2 ppm: trace sum is 46.7 ppm, comfortably
    inside the 50 ppm limit, and three of four elements are far inside their
    limits, but Al at 45 ppm exceeds its 30 ppm limit, so the gate FAILS. The
    specification is per element and a low sum does not compensate.
    """
    passes, per, verdict = hpq_gate({"Al": 45.0, "Ti": 1.0, "Li": 0.5, "Fe": 0.2})
    assert not passes
    assert per["Al"][2] is False
    assert per["Ti"][2] is True
    assert "Al 45.00 > 30.00" in verdict
    total = 45.0 + 1.0 + 0.5 + 0.2
    assert total == pytest.approx(46.7, abs=1e-12)
    assert HPQ_TRACE_SUM_LIMIT_PPM == pytest.approx(50.0, abs=1e-12)
    assert total < HPQ_TRACE_SUM_LIMIT_PPM
    assert total == pytest.approx(46.7, abs=1e-9)
    assert HPQ_TRACE_SUM_LIMIT_PPM == pytest.approx(50.0, abs=1e-9)


@pytest.mark.golden
def test_trace_sum_alone_can_fail_while_every_element_passes():
    """Nine elements each inside its limit can still breach the 50 ppm sum.

    Al 29, Ti 9, Li 4, Na 7, K 7, Ca 4, Fe 2, P 1.5, B 0.9: every one is below
    its limit, but the sum is 64.4 ppm, above 50, so the gate fails on the sum.
    This is why both tests are applied.
    """
    floors = {"Al": 29.0, "Ti": 9.0, "Li": 4.0, "Na": 7.0, "K": 7.0, "Ca": 4.0,
              "Fe": 2.0, "P": 1.5, "B": 0.9}
    for el, v in floors.items():
        assert v < HPQ_SINGLE_GRAIN_LIMITS_PPM[el]
    passes, per, verdict = hpq_gate(floors)
    assert not passes
    assert all(ok for _, _, ok in per.values())
    assert "trace sum 64.40 > 50.00" in verdict


# --- (e) benchmarks with error reported -------------------------------------


@pytest.mark.benchmark
def test_benchmark_xia_2024_endpoints(capsys):
    """Benchmark: Xia et al. 2024 (doi:10.3390/min14070727) endpoint arithmetic.

    Published: 128.86 ug/g feed, 24.23 ug/g product, 81.20 percent removal,
    SiO2 99.998 wt percent, residue identified as lattice Al, Ti and Li. Eq. E5
    and the SiO2 conversion are checked against the two independent published
    figures (the removal percentage and the SiO2 percentage), and the errors are
    reported. Only endpoints are used: the paper publishes no per-stage data.
    """
    published_removal_pct = 81.20
    published_sio2 = 99.998
    r = implied_removal_rate(128.86, 24.23) * 100.0
    s = lattice_ceiling_sio2_percent(24.23)
    err_r = abs(r - published_removal_pct) / published_removal_pct * 100.0
    err_s = abs(s - published_sio2) / published_sio2 * 100.0
    with capsys.disabled():
        print(f"\n[benchmark] Xia et al. 2024 (doi:10.3390/min14070727): "
              f"removal published {published_removal_pct:.2f} percent, model "
              f"{r:.4f} percent, error {err_r:.4f} percent relative. "
              f"SiO2 published {published_sio2:.3f} wt percent, model "
              f"{s:.6f} wt percent, error {err_s:.2e} percent relative "
              f"(the SiO2 agreement is a rounding check on a by-difference "
              f"convention, not an independent measurement).")
    assert err_r < 0.01, "Eq. E5 must reproduce the published removal percentage"
    # Measured error 4.23e-4 percent relative: 99.997577 is what the paper's own
    # 24.23 ug/g implies, and 99.998 is that figure rounded to three decimals.
    assert err_s < 1e-3


@pytest.mark.benchmark
def test_benchmark_qu_2025_two_ore_bodies(capsys):
    """Benchmark: Qu et al. 2025 (doi:10.5194/ejm-37-953-2025) HT and PX quartz.

    Published totals: HT 322.96 to 18.72 ug/g, PX 4944.73 to 114.56 ug/g, with
    final SiO2 99.963 to 99.970 wt percent (HT) and 99.974 to 99.985 wt percent
    (PX). Eq. E5 gives the removal rates and the SiO2 conversion is checked
    against the published RANGES, with the deviation from the nearest range
    bound reported.

    The PX result is the substantive finding: 97.68 percent removal, the highest
    of the three campaigns modelled in this platform, still leaves 114.56 ug/g,
    which is more than twice the 50 ppm HPQ trace sum limit. High removal
    efficiency on a dirty feed does not produce HPQ; the feed grade sets the
    outcome.
    """
    cases = [("HT", 322.96, 18.72, 99.963, 99.970),
             ("PX", 4944.73, 114.56, 99.974, 99.985)]
    rows = []
    for name, feed, prod, lo, hi in cases:
        r = implied_removal_rate(feed, prod) * 100.0
        s = lattice_ceiling_sio2_percent(prod)
        if s < lo:
            dev = (lo - s) / lo * 100.0
        elif s > hi:
            dev = (s - hi) / hi * 100.0
        else:
            dev = 0.0
        rows.append((name, feed, prod, r, s, lo, hi, dev))
    with capsys.disabled():
        print("\n[benchmark] Qu et al. 2025 (doi:10.5194/ejm-37-953-2025):")
        for name, feed, prod, r, s, lo, hi, dev in rows:
            print(f"    {name}: {feed:.2f} -> {prod:.2f} ug/g, removal {r:.2f} "
                  f"percent; SiO2 by difference {s:.6f} wt percent against "
                  f"published {lo:.3f} to {hi:.3f}, deviation from range "
                  f"{dev:.4f} percent relative")
        print(f"    PX residual {114.56:.2f} ug/g is "
              f"{114.56 / HPQ_TRACE_SUM_LIMIT_PPM:.2f}x the 50 ppm HPQ trace sum "
              f"limit despite 97.68 percent removal: feed grade, not removal "
              f"efficiency, decides HPQ viability")
        print(f"    NOTE: by-difference SiO2 exceeds the published upper bound in "
              f"both cases (HT +{rows[0][7]:.4f}, PX +{rows[1][7]:.4f} percent "
              f"relative), so Qu et al.'s SiO2 is not 100 minus the cation trace "
              f"sum. Trace sum in ug/g is the comparable metric, not SiO2 percent.")
    ht = rows[0]
    px = rows[1]
    assert ht[3] == pytest.approx(94.2036, abs=1e-3)
    assert px[3] == pytest.approx(97.6832, abs=1e-3)
    # SUBSTANTIVE NEGATIVE RESULT, not a tolerance to be loosened. Converting the
    # published trace sums by difference gives 99.998128 wt% (HT) and 99.988544
    # (PX), both ABOVE the published upper bounds of 99.970 and 99.985, by 0.0281
    # and 0.0035 percent relative. The deviations are one-sided, so the paper's
    # SiO2 figures cannot be 100 minus the cation trace sum: they must come from
    # a whole-rock determination that also carries oxide oxygen, water and phases
    # outside the trace list. Xia et al. 2024 agrees with the by-difference
    # convention and Qu et al. 2025 does not, which means the convention is not
    # comparable across papers and an SiO2 percentage should never be used as the
    # primary purity metric in this platform. Use the trace sum in ug/g.
    assert ht[7] > 0.0 and px[7] > 0.0, "both deviate above the published range"
    assert ht[7] < 0.05 and px[7] < 0.05
    assert ht[4] > ht[6] and px[4] > px[6], (
        "by-difference SiO2 exceeds the published upper bound in both cases"
    )
    assert px[2] > HPQ_TRACE_SUM_LIMIT_PPM


@pytest.mark.benchmark
def test_benchmark_prior_floors_against_xia_residual(capsys):
    """Benchmark: do the ASSUMED vein quartz priors reproduce Xia's residual split?

    Xia et al. 2024 report that after full purification only Al, Ti and Li
    remain, totalling 24.23 ug/g. The priors in this module assign lattice
    fractions of 0.45 (Al), 0.60 (Ti) and 0.75 (Li). Applying them to a
    hypothetical feed whose Al, Ti and Li sum to the same feed proportion tests
    whether the priors are even self-consistent with the one published
    endpoint pair, and reports the discrepancy.

    The result is a MISS and is reported as such: the priors imply a residual
    of 39.58 ppm from a 128.86 ppm feed (69.29 percent removal) against the
    24.23 ppm (81.20 percent) actually achieved, a 14.67 percent relative error
    on the removal rate and a 1.63-fold overstatement of the residual, so the
    priors overstate lattice content for that deposit. That is the
    expected outcome for engineering estimates applied to a different deposit,
    and it is the quantitative reason the priors must not be used for a
    decision.
    """
    feed_total = 128.86
    # Feed composition: Xia lists Al, K, Ca, Na, Ti, Fe, Li as the main
    # impurities, without per-element values. Distribute the published total
    # across the seven named elements in equal shares, which is an explicit
    # test construction and NOT a claim about the paper.
    els = ("Al", "K", "Ca", "Na", "Ti", "Fe", "Li")
    share = feed_total / len(els)
    residual = 0.0
    used = []
    for el in els:
        key = (OreType.VEIN_QUARTZ, el)
        if key in PARTITION_PRIORS:
            lat = PARTITION_PRIORS[key].fractions["lattice"]
            used.append(el)
        else:
            lat = 0.0  # Ca has no prior at all; treated as fully removable here
        residual += share * lat
    model_removal = (1.0 - residual / feed_total) * 100.0
    published_removal = 81.1966
    err = abs(model_removal - published_removal) / published_removal * 100.0
    with capsys.disabled():
        print(f"\n[benchmark] ASSUMED vein quartz priors vs Xia et al. 2024 "
              f"(doi:10.3390/min14070727): priors imply residual {residual:.2f} "
              f"ug/g from a {feed_total:.2f} ug/g feed, i.e. {model_removal:.2f} "
              f"percent removal, against the published {published_removal:.2f} "
              f"percent. ERROR {err:.2f} percent relative. Elements with priors: "
              f"{used}; Ca has no prior (in NO_PARTITION_DATA) and was treated as "
              f"fully removable. The priors OVERSTATE lattice content for this "
              f"deposit; they are engineering estimates and must be replaced by "
              f"LA-ICP-MS measurement.")
    assert residual > 24.23, "priors are more pessimistic than Xia's real result"
    # the priors quoted in the docstring, read back from the module
    assert PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Al")].fractions[
        "lattice"] == pytest.approx(0.45, abs=1e-12)
    assert PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Ti")].fractions[
        "lattice"] == pytest.approx(0.60, abs=1e-12)
    assert PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Li")].fractions[
        "lattice"] == pytest.approx(0.75, abs=1e-12)
    # and the four reported figures
    assert residual == pytest.approx(39.58, abs=5e-3)
    assert model_removal == pytest.approx(69.29, abs=5e-3)
    # Recomputed from the paper's endpoints, not restated: 128.86 to 24.23
    # ug/g is a removal of (128.86 - 24.23) / 128.86 = 81.19664752... percent.
    # The variable holds the figure at the four-decimal precision the paper
    # states it, so the tolerance is half a unit in its last reported place,
    # not floating-point epsilon: the check is that the paper's rounding is
    # consistent with its own endpoints, which is exactly what a hand-check
    # can verify.
    assert published_removal == pytest.approx(
        (128.86 - 24.23) / 128.86 * 100.0, abs=5e-5)
    assert round(published_removal, 2) == pytest.approx(81.20, abs=1e-9)
    assert err == pytest.approx(14.67, abs=5e-3)
    assert residual / 24.23 == pytest.approx(1.63, abs=5e-3)
    assert 10.0 < err < 40.0
    # The prior lattice fractions the docstring names, read from the prior table.
    assert PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Al")].fractions["lattice"] == (
        pytest.approx(0.45, abs=1e-12))
    assert PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Ti")].fractions["lattice"] == (
        pytest.approx(0.60, abs=1e-12))
    assert PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Li")].fractions["lattice"] == (
        pytest.approx(0.75, abs=1e-12))
    # The miss the benchmark reports, every figure derived from the computed
    # residual rather than quoted.
    assert residual == pytest.approx(39.58, abs=0.005)
    # model_removal is already a percentage, so it is compared directly.
    assert model_removal == pytest.approx(69.29, abs=0.005)
    xia_removal_pct = (1.0 - 24.23 / feed_total) * 100.0
    assert xia_removal_pct == pytest.approx(81.20, abs=0.005)
    rel_err_pct = abs(model_removal - xia_removal_pct) / xia_removal_pct
    assert rel_err_pct * 100.0 == pytest.approx(14.67, abs=0.01)
    assert residual / 24.23 == pytest.approx(1.63, abs=0.005)
    assert residual > 24.23


# --- the central behaviour: raise, do not guess -----------------------------


def test_partition_raises_when_unmeasured():
    f = _feedstock({"Al": 120.0})
    with pytest.raises(ValueError, match="will not be guessed"):
        partition_fractions(f, "Al")


def test_partition_raise_message_distinguishes_a_known_lattice_fraction():
    """A measured lattice fraction still does not resolve the other three."""
    f = _feedstock({"Al": 120.0}, lattice={"Al": 0.45})
    with pytest.raises(ValueError, match="not resolved by it"):
        partition_fractions(f, "Al")
    f2 = _feedstock({"Al": 120.0})
    with pytest.raises(ValueError, match="no lattice fraction has been measured"):
        partition_fractions(f2, "Al")


def test_prior_is_available_only_on_explicit_request_and_tagged_assumed():
    f = _feedstock({"Al": 120.0})
    fracs, tag, note = partition_fractions(f, "Al", allow_prior=True)
    assert tag is Tag.ASSUMED
    assert "ASSUMED ore-type prior" in note
    assert math.fsum(fracs.values()) == pytest.approx(1.0, abs=1e-12)


def test_no_prior_for_elements_without_published_data():
    f = _feedstock({e: 1.0 for e in ("U", "Th", "Zr", "Ge")})
    for el in ("U", "Th", "Zr", "Ge"):
        with pytest.raises(KeyError, match="no partition prior"):
            partition_fractions(f, el, allow_prior=True)
        assert el in NO_PARTITION_DATA


def test_no_prior_for_other_ore_types():
    """Priors exist for vein quartz only; other ore types raise."""
    for ore in (OreType.QUARTZITE, OreType.PEGMATITE_QUARTZ, OreType.ALASKITE,
                OreType.INDUSTRIAL_SAND):
        f = _feedstock({"Al": 100.0}, ore_type=ore, sample_id="AE-Q-IN-VKB-031")
        with pytest.raises(KeyError, match="no partition prior"):
            partition_fractions(f, "Al", allow_prior=True)


def test_unmeasured_element_raises_from_the_feedstock():
    f = _feedstock({"Al": 120.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Ti": _partition("Ti", 0.02, 0.03, 0.35, 0.60)})
    with pytest.raises(KeyError, match="was not measured"):
        location_concentrations(f, "Ti", model=model)


def test_partition_model_get_raises_with_guidance():
    m = PartitionModel(sample_id="AE-Q-IN-VKB-030",
                       partitions={"Al": _partition("Al", 0.1, 0.1, 0.3, 0.5)})
    with pytest.raises(KeyError, match="fabricate the purity ceiling"):
        m.get("Ti")


def test_unknown_element_rejected_before_anything_else():
    f = _feedstock({"Al": 100.0})
    with pytest.raises(ValueError, match="unknown element"):
        partition_fractions(f, "Zz")


def test_measured_model_takes_precedence_over_prior():
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.0, 0.0, 0.9, 0.1)})
    fracs, tag, note = partition_fractions(f, "Al", model=model, allow_prior=True)
    assert fracs["lattice"] == pytest.approx(0.1)
    assert tag is Tag.MEASURED
    assert "PartitionModel" in note


# --- lattice removal requires a named mechanism -----------------------------


def test_lattice_removal_requires_a_named_mechanism():
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.1, 0.1, 0.3, 0.5)})
    with pytest.raises(ValueError, match="requires lattice_removal_mechanism"):
        floor_concentration(f, "Al", model=model, lattice_removal=0.5)


def test_lattice_removal_allowed_when_mechanism_is_named():
    """Chlorination roasting reaches part of the lattice; it must be declared.

    Al 100 ppm, lattice 0.5, so 50 ppm lattice. A 40 percent lattice removal
    leaves 50 x 0.6 = 30.0 ppm, and the other three locations are fully
    removed, so the floor is 30.0 ppm.
    """
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.1, 0.1, 0.3, 0.5)})
    assert 100.0 * 0.5 == pytest.approx(50.0, abs=1e-12)
    assert 1.0 - 0.4 == pytest.approx(0.6, abs=1e-12)
    assert 50.0 * 0.6 == pytest.approx(30.0, abs=1e-12)
    assert 100.0 * 0.4 == pytest.approx(40.0, abs=1e-12)
    floor = floor_concentration(
        f, "Al", model=model, lattice_removal=0.4,
        lattice_removal_mechanism=(
            "chlorination roasting with carbonaceous reductant, Liu et al. 2026 "
            "doi:10.3390/min16080836"
        ),
    )
    assert floor == pytest.approx(30.0, abs=1e-12)
    al_ppm = 100.0
    lattice_fraction = 0.5
    lattice_ppm = al_ppm * lattice_fraction
    assert lattice_ppm == pytest.approx(50.0, abs=1e-9)
    lattice_removal_fraction = 0.4
    lattice_removal_pct = lattice_removal_fraction * 100.0
    assert lattice_removal_pct == pytest.approx(40.0, abs=1e-9)
    remaining_fraction = 1.0 - lattice_removal_fraction
    assert remaining_fraction == pytest.approx(0.6, abs=1e-9)
    final_check = lattice_ppm * remaining_fraction
    assert final_check == pytest.approx(30.0, abs=1e-12)


def test_lattice_efficiency_cannot_be_smuggled_through_efficiencies():
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.1, 0.1, 0.3, 0.5)})
    with pytest.raises(KeyError, match="unknown efficiency keys"):
        floor_concentration(f, "Al", model=model, efficiencies={"lattice": 1.0})


@pytest.mark.parametrize("eff", [-0.1, 1.1])
def test_efficiencies_must_be_fractions(eff):
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.1, 0.1, 0.3, 0.5)})
    with pytest.raises(ValueError):
        floor_concentration(f, "Al", model=model, efficiencies={"surface": eff})


# --- physical sanity --------------------------------------------------------


def test_floor_never_exceeds_bulk():
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.25, 0.25, 0.25, 0.25)})
    floor = floor_concentration(f, "Al", model=model,
                                efficiencies={"surface": 0.0, "fluid": 0.0,
                                              "mineral": 0.0})
    assert floor == pytest.approx(100.0, abs=1e-12)


def test_floor_is_monotonic_in_efficiency():
    f = _feedstock({"Al": 100.0})
    model = PartitionModel(sample_id=f.sample_id,
                           partitions={"Al": _partition("Al", 0.2, 0.2, 0.2, 0.4)})
    prev = 1e9
    for eff in (0.0, 0.25, 0.5, 0.75, 1.0):
        floor = floor_concentration(f, "Al", model=model,
                                    efficiencies={"surface": eff, "fluid": eff,
                                                  "mineral": eff})
        assert floor < prev
        prev = floor
    assert prev == pytest.approx(40.0, abs=1e-12)


def test_removal_rate_rejects_impossible_inputs():
    with pytest.raises(ValueError, match="must be positive"):
        implied_removal_rate(0.0, 10.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        implied_removal_rate(100.0, -1.0)
    with pytest.raises(ValueError, match="cannot add impurity"):
        implied_removal_rate(24.23, 128.86)


def test_sio2_conversion_rejects_negatives():
    with pytest.raises(ValueError, match="cannot be negative"):
        lattice_ceiling_sio2_percent(-1.0)


def test_sio2_conversion_is_monotonic_decreasing():
    assert (lattice_ceiling_sio2_percent(10.0)
            > lattice_ceiling_sio2_percent(50.0)
            > lattice_ceiling_sio2_percent(500.0))


def test_hpq_gate_rejects_negative_floor():
    with pytest.raises(ValueError, match="negative floor"):
        hpq_gate({"Al": -1.0})


def test_floor_profile_defaults_to_measured_hpq_elements():
    f = _feedstock({"Al": 20.0, "Ti": 5.0, "Li": 2.0, "Fe": 1.0, "Na": 3.0,
                    "K": 3.0, "B": 0.5})
    model = PartitionModel(
        sample_id=f.sample_id,
        partitions={el: _partition(el, 0.1, 0.1, 0.3, 0.5)
                    for el in ("Al", "Ti", "Li", "Fe", "Na", "K", "B")},
    )
    prof = floor_profile(f, model=model)
    assert set(prof) == {"Al", "Ti", "Li", "Fe", "Na", "K", "B"}
    assert prof["Al"] == pytest.approx(10.0, abs=1e-12)


def test_floor_profile_raises_on_a_four_oxide_assay():
    """An assay with no HPQ-relevant elements cannot answer the HPQ question."""
    f = _feedstock({"Mg": 10.0, "Mn": 5.0})
    with pytest.raises(ValueError, match="four-oxide royalty assay"):
        floor_profile(f, model=PartitionModel(sample_id=f.sample_id))


def test_gate_ignores_elements_without_a_limit_but_counts_them_in_the_sum():
    passes, per, verdict = hpq_gate({"Al": 5.0, "Mg": 10.0})
    assert "Mg" not in per
    assert passes
    assert "15.00" in verdict


def test_oh_is_excluded_from_the_trace_sum():
    """OH is reported on a different basis and must not enter a cation sum."""
    _, _, with_oh = hpq_gate({"Al": 5.0, "OH": 1000.0})
    assert "5.00" in with_oh


# --- reference data and provenance discipline -------------------------------


def test_hpq_limits_match_the_brief():
    assert HPQ_SINGLE_GRAIN_LIMITS_PPM == {
        "Al": 30.0, "Ti": 10.0, "Li": 5.0, "Na": 8.0, "K": 8.0, "Ca": 5.0,
        "Fe": 3.0, "P": 2.0, "B": 1.0,
    }
    assert HPQ_TRACE_SUM_LIMIT_PPM == 50.0


def test_the_sum_limit_exceeds_the_sum_of_the_element_limits():
    """A sanity check on the reference set: 72 ppm of element limits vs 50 sum.

    The nine element limits sum to 30 + 10 + 5 + 8 + 8 + 5 + 3 + 2 + 1 = 72 ppm,
    which is ABOVE the 50 ppm trace sum limit. The two criteria are therefore
    genuinely independent and neither implies the other, which is why the gate
    applies both.
    """
    assert math.fsum([30.0, 10.0, 5.0, 8.0, 8.0, 5.0, 3.0, 2.0,
                      1.0]) == pytest.approx(72.0)
    assert sorted(HPQ_SINGLE_GRAIN_LIMITS_PPM.values(), reverse=True) == [
        30.0, 10.0, 8.0, 8.0, 5.0, 5.0, 3.0, 2.0, 1.0]
    assert HPQ_TRACE_SUM_LIMIT_PPM == pytest.approx(50.0, abs=1e-12)
    assert math.fsum(HPQ_SINGLE_GRAIN_LIMITS_PPM.values()) == pytest.approx(72.0)
    assert math.fsum(HPQ_SINGLE_GRAIN_LIMITS_PPM.values()) > HPQ_TRACE_SUM_LIMIT_PPM
    sorted_limits = sorted(HPQ_SINGLE_GRAIN_LIMITS_PPM.values())
    assert sorted_limits == pytest.approx([1.0, 2.0, 3.0, 5.0, 5.0, 8.0, 8.0, 10.0, 30.0])
    assert HPQ_TRACE_SUM_LIMIT_PPM == pytest.approx(50.0)


def test_all_priors_are_assumed_with_a_stated_basis():
    for ep in PARTITION_PRIORS.values():
        assert ep.method == "assumed_from_mechanism"
        assert ep.weakest_tag is Tag.ASSUMED
        for loc in ("surface", "fluid", "mineral", "lattice"):
            v: Value = getattr(ep, loc)
            assert v.tag is Tag.ASSUMED
            assert v.basis and "ESTIMATE" in v.basis
            assert v.confidence == "low"


def test_every_prior_closes():
    for ep in PARTITION_PRIORS.values():
        assert math.fsum(ep.fractions.values()) == pytest.approx(1.0, abs=1e-9)


def test_priors_cover_only_vein_quartz():
    assert {ore for ore, _ in PARTITION_PRIORS} == {OreType.VEIN_QUARTZ}


def test_priors_cover_the_elements_xia_names_as_residual():
    """Al, Ti and Li are the elements the published residue consists of."""
    priored = {el for _, el in PARTITION_PRIORS}
    assert {"Al", "Ti", "Li"} <= priored


def test_no_partition_data_elements_are_real_and_have_no_priors():
    priored = {el for _, el in PARTITION_PRIORS}
    for el in NO_PARTITION_DATA:
        assert el in ELEMENTS
        assert el not in priored


def test_sources_are_real_dated_and_tiered():
    for src in (SRC_MULLER_2012, SRC_LIN_2020, SRC_XIA_2024, SRC_QU_2025,
                SRC_LIU_2026):
        assert src.tier is Tier.T1
        assert src.doi
        assert src.accessed == dt.date(2026, 9, 16)
        assert src.note and len(src.note) > 40


def test_muller_source_discloses_that_the_chapter_was_not_retrieved():
    assert "re-verified" in (SRC_MULLER_2012.note or "")


def test_lattice_priors_are_highest_for_ti_li_and_b():
    """The mechanism ordering: Ti4+, B3+ and interstitial Li are lattice-heavy.

    Ti4+ substitutes for Si4+ with no charge compensation needed, B3+ with H+,
    and Li+ is itself the charge compensator for lattice Al3+. Fe, by contrast,
    is dominated by discrete oxide inclusions. The priors encode that ordering,
    and the test pins it so a later edit cannot silently invert the mechanism.
    """
    lat = {el: PARTITION_PRIORS[(OreType.VEIN_QUARTZ, el)].fractions["lattice"]
           for _, el in PARTITION_PRIORS}
    assert lat["Li"] > lat["Al"]
    assert lat["Ti"] > lat["Al"]
    assert lat["B"] > lat["Al"]
    assert lat["Fe"] < lat["Al"]
    assert lat["Fe"] < lat["K"] + 0.01


def test_alkali_priors_are_fluid_or_mineral_dominated():
    """Na is fluid-inclusion dominated, K is mineral (feldspar) dominated."""
    na = PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "Na")].fractions
    k = PARTITION_PRIORS[(OreType.VEIN_QUARTZ, "K")].fractions
    assert na["fluid"] == max(na.values())
    assert k["mineral"] == max(k.values())


def test_feedstock_lattice_ppm_still_raises_when_unmeasured():
    """The core module's guard must not be worked around by this module."""
    f = _feedstock({"Al": 100.0})
    with pytest.raises(MissingValueError):
        f.impurities.lattice_ppm("Al")
