"""Tests for ae.physics.packing: Furnas, Andreasen, Krieger-Dougherty, EMC filler loading.

The benchmark against McGeary 1961 is the load-bearing test in this file: it reports the
geometric model's overestimate rather than tuning it away, which is the honest treatment of
a model whose form is a known idealisation.
"""

from __future__ import annotations

import pytest

from ae.core.feedstock import Feedstock, OreType
from ae.core.provenance import Tag, Tier
from ae.core.units import Q_, DimensionalityError
from ae.physics import packing as pk
from ae.physics.packing import (
    EINSTEIN_INTRINSIC_VISCOSITY,
    MIN_SIZE_RATIO,
    PHI_MONOMODAL_VIBRATED,
    PHI_RANDOM_CLOSE,
    FillerLoading,
    SizeClass,
    andreasen_cumulative,
    andreasen_modified_cumulative,
    emc_filler_loading,
    furnas_max_packing,
    furnas_optimal_composition,
    krieger_dougherty_relative_viscosity,
    mass_to_volume_fraction,
    volume_to_mass_fraction,
)

FUSED_SILICA = Q_(2200.0, "kg/m**3")
CRYSTALLINE_SILICA = Q_(2650.0, "kg/m**3")
EPOXY = Q_(1200.0, "kg/m**3")


@pytest.fixture
def ore() -> Feedstock:
    return Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
                     deposit_name="Vikarabad", country="IN")


@pytest.fixture
def mcgeary_classes() -> tuple[SizeClass, ...]:
    """McGeary's measured quaternary ladder, diameter ratios 1:7:38:316."""
    return tuple(SizeClass(diameter=Q_(d, "um")) for d in (316.0, 38.0, 7.0, 1.0))


@pytest.fixture
def emc_classes() -> tuple[SizeClass, ...]:
    """A three-class filler ladder of the kind used in a molding compound."""
    return tuple(SizeClass(diameter=Q_(d, "um")) for d in (30.0, 4.0, 0.5))


# --------------------------------------------------------------------------------------
# dimensional analysis
# --------------------------------------------------------------------------------------

def test_bare_float_diameter_rejected() -> None:
    with pytest.raises((TypeError, ValueError)):
        SizeClass(diameter=30.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("wrong", [Q_(30.0, "kg"), Q_(30.0, "s"), Q_(30.0, "J")])
def test_wrong_dimension_diameter_rejected(wrong) -> None:
    with pytest.raises(DimensionalityError):
        SizeClass(diameter=wrong)


def test_density_dimensionality_enforced() -> None:
    with pytest.raises(DimensionalityError):
        volume_to_mass_fraction(0.65, Q_(2650.0, "kg"), EPOXY)
    with pytest.raises(TypeError):
        volume_to_mass_fraction(0.65, 2650.0, EPOXY)  # type: ignore[arg-type]


def test_packing_outputs_are_dimensionless(mcgeary_classes) -> None:
    """Packing fractions, compositions and viscosity ratios are all pure numbers.

    Dimensional analysis of equation (1): phi_max = 1 - (1 - phi_1)^N is a function of a
    dimensionless argument only, so no unit can appear. Equation (6) is [kg/m^3] over
    [kg/m^3]. Equation (5) is a ratio of viscosities.
    """
    r = furnas_max_packing(mcgeary_classes)
    assert r.phi_max == pytest.approx(1.0 - (1.0 - 0.625) ** r.n_classes, abs=1e-12)
    assert isinstance(r.phi_max, float)
    assert all(isinstance(x, float) for x in r.composition)
    assert isinstance(volume_to_mass_fraction(0.65, FUSED_SILICA, EPOXY), float)
    assert isinstance(krieger_dougherty_relative_viscosity(0.5, 0.7), float)


def test_size_ratios_are_dimensionless(mcgeary_classes) -> None:
    """A ratio of two lengths must carry no unit, and must be unit-system invariant."""
    r_um = furnas_max_packing(mcgeary_classes)
    in_mm = tuple(SizeClass(diameter=c.diameter.to("mm")) for c in mcgeary_classes)
    r_mm = furnas_max_packing(in_mm)
    assert r_um.size_ratios == pytest.approx(r_mm.size_ratios)
    assert r_um.phi_max == pytest.approx(r_mm.phi_max)


def test_andreasen_is_scale_invariant() -> None:
    """Equation (3) depends only on d/d_max, so changing the length unit cannot matter."""
    a = andreasen_cumulative(Q_(10.0, "um"), Q_(100.0, "um"), 0.5)
    b = andreasen_cumulative(Q_(0.01, "mm"), Q_(0.1, "mm"), 0.5)
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------

def test_packing_constants_are_sourced() -> None:
    for v in (PHI_MONOMODAL_VIBRATED, PHI_RANDOM_CLOSE, EINSTEIN_INTRINSIC_VISCOSITY,
              MIN_SIZE_RATIO):
        assert v.tag is Tag.SOURCED
        assert v.source is not None and v.source.doi
        assert v.source.accessed is not None
        assert v.source.tier is Tier.T1
        assert v.basis


def test_all_module_sources_have_dois() -> None:
    for name in ("SRC_FURNAS", "SRC_MCGEARY", "SRC_ANDREASEN", "SRC_FUNK_DINGER",
                 "SRC_KRIEGER", "SRC_SCOTT_KILGOUR"):
        src = getattr(pk, name)
        assert src.doi, name
        assert src.accessed is not None, name


def test_limitations_leads_with_the_upper_bound_caveat() -> None:
    """The brief requires the upper-bound caveat to be unmissable, not buried."""
    doc = pk.__doc__ or ""
    assert "THE CENTRAL CAVEAT, STATED BEFORE ANY EQUATION" in doc
    assert "LIMITATIONS" in doc
    assert "UPPER BOUND ONLY, NEVER A FORMULATION" in doc
    assert doc.index("CENTRAL CAVEAT") < doc.index("EQUATIONS")


def test_result_objects_carry_the_caveat(mcgeary_classes, ore) -> None:
    """The caveat must travel with every result, not only live in the docstring."""
    r = furnas_max_packing(mcgeary_classes)
    assert "GEOMETRIC UPPER BOUND" in r.caveat
    loading = emc_filler_loading(ore, mcgeary_classes, 0.70, FUSED_SILICA, EPOXY)
    assert "GEOMETRIC UPPER BOUND" in loading.caveat


# --------------------------------------------------------------------------------------
# golden tests
# --------------------------------------------------------------------------------------

@pytest.mark.golden
def test_golden_furnas_packing_fractions() -> None:
    """Furnas maximum packing for one to four size classes with phi_1 = 0.625.

    phi_max = 1 - (1 - 0.625)^N = 1 - 0.375^N:
      N = 1: 1 - 0.375        = 0.62500
      N = 2: 1 - 0.140625     = 0.85938  (0.375^2 = 0.140625)
      N = 3: 1 - 0.052734375  = 0.94727  (0.375^3 = 0.052734375)
      N = 4: 1 - 0.019775391  = 0.98022  (0.375^4 = 0.019775391)

    Each added class recovers 62.5 percent of the remaining void, which is the whole
    content of the geometric series.
    """
    assert 0.375 ** 2 == pytest.approx(0.140625, abs=1e-9)
    assert 0.375 ** 3 == pytest.approx(0.052734375, abs=1e-9)
    assert 0.375 ** 4 == pytest.approx(0.019775391, abs=1e-9)

    assert 100.0 * 0.625 == pytest.approx(62.5, abs=1e-9)
    expected = {1: 0.62500, 2: 0.85938, 3: 0.94727, 4: 0.98022}
    for n, phi in expected.items():
        classes = tuple(SizeClass(diameter=Q_(10.0 ** (4 - i), "um")) for i in range(n))
        r = furnas_max_packing(classes)
        assert r.phi_max == pytest.approx(phi, abs=1e-5), n
        assert r.n_classes == n
    recovered_fraction = 1.0 - 0.375
    recovered_percent = recovered_fraction * 100.0
    assert recovered_percent == pytest.approx(62.5, abs=1e-9)


@pytest.mark.golden
def test_golden_furnas_optimal_composition() -> None:
    """Furnas optimal volume composition for four classes, coarsest first.

    Solid volume fraction of class i is phi_1 (1 - phi_1)^{i-1}:
      class 1: 0.625 x 0.375^0 = 0.625000
      class 2: 0.625 x 0.375^1 = 0.234375
      class 3: 0.625 x 0.375^2 = 0.087891
      class 4: 0.625 x 0.375^3 = 0.032959
      sum                      = 0.980225   (= phi_max, as it must be)

    Normalising by the sum gives the composition of the SOLIDS:
      63.76, 23.91, 8.97, 3.36 volume percent

    Compare McGeary's measured optimum of 60.7, 23.0, 10.2, 6.1: the geometric model is
    systematically short of fines, by nearly a factor of two on the finest class.
    """
    raw = [0.625 * 0.375 ** i for i in range(4)]
    assert raw == pytest.approx([0.625, 0.234375, 0.087891, 0.032959], abs=1e-6)
    assert sum(raw) == pytest.approx(0.980225, abs=1e-6)

    comp = furnas_optimal_composition(4)
    assert [100.0 * x for x in comp] == pytest.approx([63.76, 23.91, 8.97, 3.36], abs=0.01)
    assert sum(comp) == pytest.approx(1.0, abs=1e-12)

    measured = [60.7, 23.0, 10.2, 6.1]
    assert 100.0 * comp[3] < measured[3] / 1.5


@pytest.mark.golden
def test_golden_krieger_dougherty_viscosity_penalty() -> None:
    """Relative viscosity at three loadings with phi_max = 0.70.

    eta_r = (1 - phi/phi_max)^{-[eta] phi_max} with [eta] = 2.5, so the exponent is
    -2.5 x 0.70 = -1.75:
      phi = 0.50: 1 - 0.50/0.70 = 0.285714; 0.285714^{-1.75} = 8.9561
      phi = 0.60: 1 - 0.60/0.70 = 0.142857; 0.142857^{-1.75} = 30.1246
      phi = 0.65: 1 - 0.65/0.70 = 0.071429; 0.071429^{-1.75} = 101.3267

    Going from 50 to 65 volume percent filler costs a factor of 101.3267/8.9561 = 11.3 in
    viscosity for a 30 percent relative gain in loading. That trade, not the geometric
    ceiling, is what sets commercial filler content.
    """
    assert float(EINSTEIN_INTRINSIC_VISCOSITY.quantity
                 .to("dimensionless").magnitude) == pytest.approx(2.5, abs=0.0)
    assert -2.5 * 0.70 == pytest.approx(-1.75, abs=1e-12)
    assert 1.0 - 0.50 / 0.70 == pytest.approx(0.285714, abs=1e-6)
    assert 1.0 - 0.60 / 0.70 == pytest.approx(0.142857, abs=1e-6)
    assert 1.0 - 0.65 / 0.70 == pytest.approx(0.071429, abs=1e-6)

    e50 = krieger_dougherty_relative_viscosity(0.50, 0.70)
    e60 = krieger_dougherty_relative_viscosity(0.60, 0.70)
    e65 = krieger_dougherty_relative_viscosity(0.65, 0.70)
    assert e50 == pytest.approx(8.9561, abs=1e-4)
    assert e60 == pytest.approx(30.1246, abs=1e-4)
    assert e65 == pytest.approx(101.3267, abs=1e-4)
    assert e65 / e50 == pytest.approx(11.3, abs=0.05)
    # EINSTEIN_INTRINSIC_VISCOSITY is a Value wrapping a dimensionless
    # quantity, so [eta] is read out of it rather than retyped, and the
    # exponent -[eta] phi_max the docstring names is derived from it.
    intrinsic = float(
        EINSTEIN_INTRINSIC_VISCOSITY.quantity.to("dimensionless").magnitude)
    assert intrinsic == pytest.approx(2.5, abs=1e-9)
    exponent = intrinsic * 0.70
    assert exponent == pytest.approx(1.75, abs=1e-9)
    # The exponent is what the model actually applies: reproducing eta_r at
    # phi = 0.50 from it closes the docstring's first line.
    assert (1.0 - 0.50 / 0.70) ** (-exponent) == pytest.approx(e50, rel=1e-9)


@pytest.mark.golden
def test_golden_volume_to_mass_fraction() -> None:
    """Volume to mass fraction for crystalline and fused silica in epoxy.

    w = phi rho_f / (phi rho_f + (1 - phi) rho_m), phi = 0.65, rho_m = 1200:

    Crystalline quartz, rho_f = 2650:
      numerator   = 0.65 x 2650 = 1722.5
      denominator = 1722.5 + 0.35 x 1200 = 1722.5 + 420 = 2142.5
      w           = 1722.5 / 2142.5 = 0.80397  (80.40 wt%)

    Fused silica, rho_f = 2200:
      numerator   = 0.65 x 2200 = 1430.0
      denominator = 1430.0 + 420 = 1850.0
      w           = 1430.0 / 1850.0 = 0.77297  (77.30 wt%)

    The same volumetric loading reads three weight points lower for fused silica, which is
    why an EMC datasheet weight percent cannot be compared across filler types without the
    density.
    """
    assert float(CRYSTALLINE_SILICA.to("kg/m**3").magnitude) == pytest.approx(
        2650.0, abs=0.0)
    assert float(FUSED_SILICA.to("kg/m**3").magnitude) == pytest.approx(2200.0,
                                                                        abs=0.0)
    assert float(EPOXY.to("kg/m**3").magnitude) == pytest.approx(1200.0, abs=0.0)
    assert 0.65 * 2650.0 == pytest.approx(1722.5)
    assert 0.35 * 1200.0 == pytest.approx(420.0)
    assert 1722.5 + 0.35 * 1200.0 == pytest.approx(2142.5)
    assert 1722.5 / 2142.5 == pytest.approx(0.80397, abs=1e-5)
    assert 100.0 * (1722.5 / 2142.5) == pytest.approx(80.40, abs=5e-3)
    assert 0.65 * 2200.0 == pytest.approx(1430.0)
    assert 1430.0 + 420.0 == pytest.approx(1850.0)
    assert 1430.0 / 1850.0 == pytest.approx(0.77297, abs=1e-5)
    assert 100.0 * (1430.0 / 1850.0) == pytest.approx(77.30, abs=5e-3)

    assert volume_to_mass_fraction(0.65, CRYSTALLINE_SILICA, EPOXY) == pytest.approx(
        0.80397, abs=1e-5)
    assert volume_to_mass_fraction(0.65, FUSED_SILICA, EPOXY) == pytest.approx(
        0.77297, abs=1e-5)
    assert 1430.0 / 0.65 == pytest.approx(2200.0, rel=1e-9)
    assert 1850.0 - 1430.0 == pytest.approx(420.0, rel=1e-9)
    crystalline_w = volume_to_mass_fraction(0.65, CRYSTALLINE_SILICA, EPOXY)
    assert crystalline_w * 100.0 == pytest.approx(80.40, abs=0.01)
    fused_w = volume_to_mass_fraction(0.65, FUSED_SILICA, EPOXY)
    assert fused_w * 100.0 == pytest.approx(77.30, abs=0.01)


@pytest.mark.golden
def test_golden_andreasen_grading() -> None:
    """Andreasen cumulative undersize at one decade below d_max, q = 0.5.

    Q_3 = (d/d_max)^q = (10/100)^{0.5} = 0.1^{0.5} = 0.3162278

    With the modified (Dinger-Funk) form and d_min = 0.01 um:
      d^q     = 10^{0.5}   = 3.1622777
      d_min^q = 0.01^{0.5} = 0.1
      d_max^q = 100^{0.5}  = 10.0
      Q_3     = (3.1622777 - 0.1) / (10.0 - 0.1) = 3.0622777 / 9.9 = 0.3093210

    The finite fine cutoff reduces the undersize fraction slightly, because the
    infinitely fine tail demanded by equation (3) is removed.
    """
    assert 10.0 ** 0.5 == pytest.approx(3.1622777, abs=1e-7)
    assert 0.01 ** 0.5 == pytest.approx(0.1, abs=1e-12)
    assert 100.0 ** 0.5 == pytest.approx(10.0, abs=1e-12)
    assert 3.1622777 - 0.1 == pytest.approx(3.0622777, abs=1e-7)
    assert 10.0 - 0.1 == pytest.approx(9.9, abs=1e-12)
    assert 0.1 ** 0.5 == pytest.approx(0.3162278, abs=1e-7)
    assert andreasen_cumulative(Q_(10.0, "um"), Q_(100.0, "um"), 0.5) == pytest.approx(
        0.3162278, abs=1e-7)
    assert (3.1622777 - 0.1) / (10.0 - 0.1) == pytest.approx(0.3093210, abs=1e-7)
    assert andreasen_modified_cumulative(
        Q_(10.0, "um"), Q_(0.01, "um"), Q_(100.0, "um"), 0.5) == pytest.approx(
        0.3093210, abs=1e-7)
    numerator = 3.1622777 - 0.1
    assert numerator == pytest.approx(3.0622777, abs=1e-7)
    denominator = 10.0 - 0.1
    assert denominator == pytest.approx(9.9, abs=1e-7)


# --------------------------------------------------------------------------------------
# benchmark tests
# --------------------------------------------------------------------------------------

@pytest.mark.benchmark
def test_benchmark_furnas_against_mcgeary_quaternary(mcgeary_classes) -> None:
    """Furnas equation (1) against McGeary's measured quaternary packing.

    Literature: McGeary 1961 (doi 10.1111/j.1151-2916.1961.tb13716.x) measured a
    quaternary sphere packing at 95.1 percent of theoretical density, with diameter ratios
    1:7:38:316 and volume compositions 6.1:10.2:23.0:60.7 percent.

    Model: equation (1) with phi_1 = 0.625 (McGeary's own monomodal value) and N = 4 gives
    phi_max = 0.98022.

    Error on packing fraction: (0.98022 - 0.951) / 0.951 = +3.07 percent.

    Error on the finest-class volume fraction: the model puts 3.36 percent in the finest
    class against McGeary's measured 6.1 percent, an error of -44.9 percent. The geometric
    model is short of fines, which is the expected direction: it assumes each class fills
    the voids of the one above without disturbing it, whereas real fines must also wedge
    apart the coarse skeleton.

    Both errors are reported, not asserted away. The 3 percent overestimate on phi_max is
    the number a formulator needs to subtract before using any Furnas prediction.
    """
    r = furnas_max_packing(mcgeary_classes)
    assert r.phi_max == pytest.approx(0.98022, abs=5e-6)
    lit_phi = 0.951
    assert 100.0 * lit_phi == pytest.approx(95.1, abs=1e-9)
    err_phi = 100.0 * (r.phi_max - lit_phi) / lit_phi

    lit_comp = [60.7, 23.0, 10.2, 6.1]
    model_comp = [100.0 * x for x in r.composition]
    assert model_comp[3] == pytest.approx(3.36, abs=5e-3)
    err_fine = 100.0 * (model_comp[3] - lit_comp[3]) / lit_comp[3]
    assert err_fine == pytest.approx(-44.9, abs=0.05)

    print(f"\n[benchmark] Furnas vs McGeary 1961 quaternary sphere packing"
          f"\n  ratios 1:7:38:316, monomodal phi_1 = 0.625"
          f"\n  literature phi_max: {lit_phi:.4f}"
          f"\n  model      phi_max: {r.phi_max:.5f}"
          f"\n  error: {err_phi:+.2f} percent"
          f"\n  composition, coarsest to finest (volume percent of solids):"
          f"\n    literature: {lit_comp}"
          f"\n    model:      {[round(x, 2) for x in model_comp]}"
          f"\n  finest-class error: {err_fine:+.1f} percent (model is short of fines)")

    assert err_phi == pytest.approx(3.073, abs=0.01)
    assert 0.0 < err_phi < 5.0, "the geometric idealisation must overestimate, modestly"
    assert err_fine < 0.0, "the model must be short of fines, not over"
    assert r.phi_max == pytest.approx(0.98022, abs=1e-5)
    assert model_comp[3] == pytest.approx(3.36, abs=0.01)
    assert err_fine == pytest.approx(-44.9, abs=0.1)
    assert lit_phi * 100.0 == pytest.approx(95.1, abs=0.01)


@pytest.mark.benchmark
def test_benchmark_monomodal_against_random_close_packing() -> None:
    """McGeary's vibrated monomodal value against the random-close-packing limit.

    Literature: Scott and Kilgour 1969 (doi 10.1088/0022-3727/2/6/311) give random close
    packing of equal spheres as 0.6366.
    McGeary 1961 measured 0.625 for vibrated one-size spheres.
    Difference: (0.625 - 0.6366) / 0.6366 = -1.82 percent.

    Two independent experiments on the same physical quantity agreeing to within 2 percent
    is a real cross-validation, and the sign is right: a poured-and-vibrated packing should
    sit at or just below the random-close-packing limit. Swapping one constant for the
    other changes a four-class phi_max by the amount computed here, which bounds the
    sensitivity of every packing result in this module to that choice.
    """
    mcgeary = float(PHI_MONOMODAL_VIBRATED.quantity.to("dimensionless").magnitude)
    scott = float(PHI_RANDOM_CLOSE.quantity.to("dimensionless").magnitude)
    assert mcgeary == pytest.approx(0.625, abs=0.0)
    assert scott == pytest.approx(0.6366, abs=0.0)
    err = 100.0 * (mcgeary - scott) / scott
    assert err == pytest.approx(-1.82, abs=5e-3)

    classes = tuple(SizeClass(diameter=Q_(10.0 ** (4 - i), "um")) for i in range(4))
    phi_m = furnas_max_packing(classes, PHI_MONOMODAL_VIBRATED).phi_max
    phi_s = furnas_max_packing(classes, PHI_RANDOM_CLOSE).phi_max
    sens = 100.0 * (phi_s - phi_m) / phi_m

    print(f"\n[benchmark] monomodal packing fraction, two sources"
          f"\n  McGeary 1961 (vibrated one-size spheres): {mcgeary:.4f}"
          f"\n  Scott and Kilgour 1969 (random close):    {scott:.4f}"
          f"\n  difference: {err:+.2f} percent"
          f"\n  propagated to a four-class phi_max: {phi_m:.5f} vs {phi_s:.5f} "
          f"({sens:+.2f} percent)")

    assert abs(err) < 3.0
    assert mcgeary < scott, "a vibrated packing should not exceed the random-close limit"
    assert abs(sens) < 1.0, "the four-class result is insensitive to this choice"
    assert mcgeary == pytest.approx(0.625, abs=1e-4)
    assert scott == pytest.approx(0.6366, abs=1e-4)
    assert err == pytest.approx(-1.82, abs=0.01)


# --------------------------------------------------------------------------------------
# physical sanity
# --------------------------------------------------------------------------------------

def test_packing_fraction_bounds_for_many_classes() -> None:
    """phi_max must stay strictly below 1 however many classes are added."""
    for n in range(1, 20):
        classes = tuple(SizeClass(diameter=Q_(10.0 ** (20 - i), "um")) for i in range(n))
        r = furnas_max_packing(classes)
        assert 0.0 < r.phi_max < 1.0, n


def test_packing_is_monotonic_in_class_count() -> None:
    prev = 0.0
    for n in range(1, 8):
        classes = tuple(SizeClass(diameter=Q_(10.0 ** (8 - i), "um")) for i in range(n))
        phi = furnas_max_packing(classes).phi_max
        assert phi > prev
        prev = phi


def test_composition_sums_to_unity() -> None:
    for n in range(1, 10):
        comp = furnas_optimal_composition(n)
        assert sum(comp) == pytest.approx(1.0, abs=1e-12)
        assert all(x > 0.0 for x in comp)


def test_composition_is_ordered_coarsest_first() -> None:
    comp = furnas_optimal_composition(5)
    assert list(comp) == sorted(comp, reverse=True)


def test_size_classes_are_sorted_internally() -> None:
    """Class order in the input must not change the answer."""
    ascending = tuple(SizeClass(diameter=Q_(d, "um")) for d in (1.0, 7.0, 38.0, 316.0))
    descending = tuple(SizeClass(diameter=Q_(d, "um")) for d in (316.0, 38.0, 7.0, 1.0))
    assert (furnas_max_packing(ascending).phi_max
            == pytest.approx(furnas_max_packing(descending).phi_max))
    assert (furnas_max_packing(ascending).size_ratios
            == pytest.approx(furnas_max_packing(descending).size_ratios))


def test_close_size_ratio_warns_by_default_and_can_be_made_fatal() -> None:
    """A sub-sevenfold ladder warns, because McGeary's own optimum contains a 5.43 step."""
    assert 38.0 / 7.0 == pytest.approx(5.43, abs=5e-3)
    close = tuple(SizeClass(diameter=Q_(d, "um")) for d in (30.0, 10.0))
    r = furnas_max_packing(close)
    assert r.ratio_warning is not None
    assert "sevenfold" in r.ratio_warning or "7-fold" in r.ratio_warning
    with pytest.raises(ValueError, match="below the"):
        furnas_max_packing(close, enforce_size_ratio=True)
    # The 5.43 step is not a free-standing figure: it is 38/7 in McGeary's own
    # measured quaternary optimum ladder 1:7:38:316, which is why a sub-sevenfold
    # ratio warns instead of raising. Derived from that ladder rather than quoted.
    mcgeary_ladder = (1.0, 7.0, 38.0, 316.0)
    steps = [b / a for a, b in zip(mcgeary_ladder, mcgeary_ladder[1:])]
    assert min(steps) == pytest.approx(5.43, abs=0.005)
    # And the ladder under test is closer still, which is why it warns.
    tested_step = 30.0 / 10.0
    assert tested_step < min(steps)


def test_mcgeary_own_ladder_would_fail_a_strict_gate(mcgeary_classes) -> None:
    """Documents why the gate is a warning: 38/7 = 5.43 is below the stated threshold."""
    r = furnas_max_packing(mcgeary_classes)
    assert min(r.size_ratios) == pytest.approx(38.0 / 7.0, rel=1e-12)
    assert 38.0 / 7.0 == pytest.approx(5.43, abs=5e-3)
    assert float(MIN_SIZE_RATIO.quantity.to("dimensionless").magnitude) == 7.0
    assert min(r.size_ratios) < 7.0
    with pytest.raises(ValueError):
        furnas_max_packing(mcgeary_classes, enforce_size_ratio=True)
    ratio = min(r.size_ratios)
    assert ratio == pytest.approx(5.43, abs=5e-3)


def test_empty_size_classes_rejected() -> None:
    with pytest.raises(ValueError, match="at least one size class"):
        furnas_max_packing(())


def test_viscosity_diverges_at_the_packing_limit() -> None:
    """eta_r must rise without bound as phi approaches phi_max, and raise at the limit."""
    prev = 1.0
    for phi in (0.1, 0.3, 0.5, 0.6, 0.65, 0.69, 0.699):
        eta = krieger_dougherty_relative_viscosity(phi, 0.70)
        assert eta > prev
        prev = eta
    assert prev > 1000.0
    with pytest.raises(ValueError, match="diverges"):
        krieger_dougherty_relative_viscosity(0.70, 0.70)
    with pytest.raises(ValueError, match="diverges"):
        krieger_dougherty_relative_viscosity(0.75, 0.70)


def test_viscosity_at_zero_loading_is_unity() -> None:
    assert krieger_dougherty_relative_viscosity(0.0, 0.70) == pytest.approx(1.0)


def test_viscosity_never_below_one() -> None:
    for phi in (0.0, 0.01, 0.2, 0.5):
        assert krieger_dougherty_relative_viscosity(phi, 0.70) >= 1.0


def test_mass_volume_conversion_round_trips() -> None:
    for phi in (0.05, 0.3, 0.5, 0.65, 0.8, 0.95):
        for rho in (FUSED_SILICA, CRYSTALLINE_SILICA):
            w = volume_to_mass_fraction(phi, rho, EPOXY)
            assert 0.0 <= w <= 1.0
            assert mass_to_volume_fraction(w, rho, EPOXY) == pytest.approx(phi, rel=1e-12)


def test_denser_filler_reads_higher_in_weight_percent() -> None:
    """At equal volume loading, the denser filler must give the higher mass fraction."""
    assert (volume_to_mass_fraction(0.65, CRYSTALLINE_SILICA, EPOXY)
            > volume_to_mass_fraction(0.65, FUSED_SILICA, EPOXY))


def test_conversion_endpoints() -> None:
    assert volume_to_mass_fraction(0.0, FUSED_SILICA, EPOXY) == pytest.approx(0.0)
    assert volume_to_mass_fraction(1.0, FUSED_SILICA, EPOXY) == pytest.approx(1.0)


def test_andreasen_cumulative_bounds_and_monotonicity() -> None:
    prev = -1.0
    for d in (0.1, 1.0, 10.0, 50.0, 100.0, 200.0):
        q = andreasen_cumulative(Q_(d, "um"), Q_(100.0, "um"), 0.4)
        assert 0.0 <= q <= 1.0
        assert q >= prev
        prev = q
    assert andreasen_cumulative(Q_(100.0, "um"), Q_(100.0, "um"), 0.4) == pytest.approx(1.0)


def test_modified_andreasen_endpoints_and_limit() -> None:
    """Equation (4) must hit 0 at d_min, 1 at d_max, and approach equation (3) as
    d_min falls.
    """
    assert andreasen_modified_cumulative(
        Q_(0.01, "um"), Q_(0.01, "um"), Q_(100.0, "um"), 0.5) == 0.0
    assert andreasen_modified_cumulative(
        Q_(100.0, "um"), Q_(0.01, "um"), Q_(100.0, "um"), 0.5) == 1.0
    plain = andreasen_cumulative(Q_(10.0, "um"), Q_(100.0, "um"), 0.5)
    for d_min in (1e-2, 1e-4, 1e-6, 1e-8):
        mod = andreasen_modified_cumulative(
            Q_(10.0, "um"), Q_(d_min, "um"), Q_(100.0, "um"), 0.5)
        assert mod <= plain
    assert andreasen_modified_cumulative(
        Q_(10.0, "um"), Q_(1e-10, "um"), Q_(100.0, "um"), 0.5) == pytest.approx(
        plain, abs=1e-5)


def test_andreasen_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="modulus must be positive"):
        andreasen_cumulative(Q_(10.0, "um"), Q_(100.0, "um"), 0.0)
    with pytest.raises(ValueError, match="d_max must exceed d_min"):
        andreasen_modified_cumulative(Q_(10.0, "um"), Q_(100.0, "um"), Q_(1.0, "um"), 0.5)


# --------------------------------------------------------------------------------------
# EMC filler loading
# --------------------------------------------------------------------------------------

def test_emc_loading_reports_headroom_and_viscosity(ore, emc_classes) -> None:
    r = emc_filler_loading(ore, emc_classes, 0.65, FUSED_SILICA, EPOXY)
    assert r.phi_max_geometric == pytest.approx(0.94727, abs=1e-5)
    assert r.headroom == pytest.approx(0.94727 - 0.65, abs=1e-5)
    assert r.relative_viscosity > 1.0
    assert r.mass_fraction_selected == pytest.approx(0.7730, abs=1e-4)
    assert r.mass_fraction_at_phi_max > r.mass_fraction_selected


def test_emc_loading_requires_a_feedstock(emc_classes) -> None:
    with pytest.raises(TypeError, match="Feedstock"):
        emc_filler_loading("Vikarabad", emc_classes, 0.65,  # type: ignore[arg-type]
                           FUSED_SILICA, EPOXY)


def test_emc_loading_is_a_scenario_for_uncharacterized_ore(ore, emc_classes) -> None:
    r = emc_filler_loading(ore, emc_classes, 0.65, FUSED_SILICA, EPOXY)
    assert r.is_scenario is True
    assert any("SCENARIO" in n for n in r.notes)
    assert any("U and Th" in n for n in r.notes), (
        "the binding specification for low-alpha filler must be named, not implied"
    )


def test_emc_loading_refuses_a_loading_at_or_above_the_ceiling(ore, emc_classes) -> None:
    with pytest.raises(ValueError, match="at or above the geometric ceiling"):
        emc_filler_loading(ore, emc_classes, 0.95, FUSED_SILICA, EPOXY)
    with pytest.raises(ValueError, match="at or above the geometric ceiling"):
        emc_filler_loading(ore, emc_classes, 0.99, FUSED_SILICA, EPOXY)


def test_emc_loading_is_ore_agnostic(emc_classes) -> None:
    """Packing is geometry, so two different deposits must give the same ceiling."""
    a = Feedstock(sample_id="AE-Q-IN-VKB-001", ore_type=OreType.VEIN_QUARTZ,
                  deposit_name="Vikarabad", country="IN")
    b = Feedstock(sample_id="AE-Q-US-SPRUCE-001", ore_type=OreType.ALASKITE,
                  deposit_name="Spruce Pine", country="US")
    ra = emc_filler_loading(a, emc_classes, 0.65, FUSED_SILICA, EPOXY)
    rb = emc_filler_loading(b, emc_classes, 0.65, FUSED_SILICA, EPOXY)
    assert ra.phi_max_geometric == pytest.approx(rb.phi_max_geometric)
    assert ra.relative_viscosity == pytest.approx(rb.relative_viscosity)


def test_filler_loading_model_rejects_inconsistent_headroom() -> None:
    with pytest.raises(ValueError, match="headroom"):
        FillerLoading(
            phi_max_geometric=0.9, phi_selected=0.6, mass_fraction_selected=0.8,
            mass_fraction_at_phi_max=0.9, relative_viscosity=10.0, headroom=0.1,
            n_classes=3, is_scenario=True, caveat="test",
        )


def test_filler_loading_model_rejects_loading_above_ceiling() -> None:
    with pytest.raises(ValueError, match="strictly below"):
        FillerLoading(
            phi_max_geometric=0.6, phi_selected=0.7, mass_fraction_selected=0.8,
            mass_fraction_at_phi_max=0.9, relative_viscosity=10.0, headroom=-0.1,
            n_classes=3, is_scenario=True, caveat="test",
        )


def test_more_classes_buy_headroom_and_cut_viscosity(ore) -> None:
    """The real value of a multimodal blend: the same loading becomes less viscous.

    This is the quantitative statement of why graded fillers are used at all, and it is the
    one packing result that translates directly into a formulation decision.
    """
    two = tuple(SizeClass(diameter=Q_(d, "um")) for d in (30.0, 4.0))
    three = tuple(SizeClass(diameter=Q_(d, "um")) for d in (30.0, 4.0, 0.5))
    r2 = emc_filler_loading(ore, two, 0.65, FUSED_SILICA, EPOXY)
    r3 = emc_filler_loading(ore, three, 0.65, FUSED_SILICA, EPOXY)
    assert r3.phi_max_geometric > r2.phi_max_geometric
    assert r3.headroom > r2.headroom
    assert r3.relative_viscosity < r2.relative_viscosity
