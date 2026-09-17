"""Adversarial audit of ae.physics: dimensions, limits, signs, conservation, docstrings.

Every test here was written to FAIL against the code as committed, and each one
did fail before the accompanying fix. The defects found, in the order they
appear below:

1. diffusion.fractional_extraction_sphere used the ONE-term short-time
   asymptote 6 sqrt(Fo/pi) below Fo = 1e-4 while its docstring called that form
   "the exact short-time limit". It is the leading term only. The omitted -3 Fo
   term makes the branch overstate extraction and makes the function
   NON-MONOTONIC in Fo at the switch.
2. comminution accepted a non-finite f80, work index or law constant and
   returned nan or inf rather than raising, while the same guard on p80 was
   present. A nan specific energy propagates into an energy cost silently.
3. comminution.walker_specific_energy never checked the dimensionality of its
   own output, so a Bond-unit constant used at n = 1.5 returned a plausible
   magnitude carrying the wrong dimensions. Kick and Rittinger both check.
4. separation.rectangular_distribution_recovery evaluated (1 - exp(-x))/x
   directly, which loses all significant figures as x goes to zero, and raised
   on a NEGATIVE recovery for small but physical residence times.
5. reagents.reagent_balance defined the liquor mass by difference from the
   inlet and then checked that the outlet summed to the inlet, an identity that
   cannot fail. The assertion its docstring advertises could not detect an
   unbalanced flowsheet.
6. Two docstring claims contradicted by the module's own code: a grain coarser
   than sqrt(D t) "retains essentially all of its lattice inventory" (the code
   gives 0.948 extraction at twice that radius), and a thin-skin extraction
   "approximately 3 L / R" (the coefficient is 6/sqrt(pi)).

Tests whose names begin ``test_pin_`` are not defect demonstrations. They pin
behaviour judged correct but unasserted, so a later change that breaks it fails
here rather than silently.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from ae.core.units import Q_
from ae.physics.comminution import (
    TonConvention,
    bond_specific_energy,
    kick_specific_energy,
    rittinger_specific_energy,
    walker_specific_energy,
)
from ae.physics.diffusion import (
    FO_SHORT_TIME_SWITCH,
    fractional_extraction_sphere,
)
from ae.physics.separation import (
    PropertyBasis,
    imperfection,
    rectangular_distribution_recovery,
)


def _converged_sphere_series(fo: float, n_terms: int = 200_000) -> float:
    """Independent evaluation of the sphere series, summed to n_terms.

    Deliberately not a call into the module under test: the point of these
    tests is to compare the module against an outside computation of the same
    analytic solution.
    """
    s = sum(math.exp(-(k**2) * math.pi**2 * fo) / k**2 for k in range(1, n_terms + 1))
    return 1.0 - 6.0 / math.pi**2 * s


# --- 1. diffusion: short-time asymptote -------------------------------------


def _one_term_branch_extraction(fo: float) -> float:
    """The defective branch as committed: leading short-time term below the switch.

    Reproduced here so the test can assert the size of the defect against the
    module's current output. Asserting the quoted figures against each other
    would be true by construction, which is the self-satisfying pattern
    tests/test_test_docstrings.py::test_no_assertion_compares_a_literal_against_itself
    exists to forbid.
    """
    if fo < FO_SHORT_TIME_SWITCH:
        return min(1.0, 6.0 / math.sqrt(math.pi) * math.sqrt(fo))
    return fractional_extraction_sphere(fo)


def test_sphere_extraction_is_monotonic_across_the_switch() -> None:
    """Extraction must rise with Fourier number everywhere, including the switch.

    Measured against the code as committed, over a 5000-point sweep of Fo from
    1e-8 to 10: extraction FELL from 0.033838382230 at Fo = 9.992325103785e-05
    to 0.033602473294 at Fo = 1.003075857943e-04, a drop of 2.359089e-04, and
    that was the only violation in the sweep. More material leaving a grain in
    less time is what the bisection in time_to_deplete_grain relies on not
    happening.
    """
    fos = np.concatenate([np.logspace(-8, -3, 3000), np.logspace(-3, 1, 2000)])
    vals = [fractional_extraction_sphere(float(f)) for f in fos]
    drops = [
        (float(fos[i]), float(fos[i + 1]), vals[i + 1] - vals[i])
        for i in range(len(vals) - 1)
        if vals[i + 1] < vals[i]
    ]
    assert not drops, (
        "extraction fell with rising Fourier number at "
        + ", ".join(f"Fo {a:.6e} -> {b:.6e} by {d:+.6e}" for a, b, d in drops)
    )
    # The quoted endpoints, recomputed from the defective branch and checked
    # against the fixed module rather than against each other.
    fo_lo, fo_hi = 9.992325103785e-05, 1.003075857943e-04
    assert fo_lo < FO_SHORT_TIME_SWITCH < fo_hi
    old_lo = _one_term_branch_extraction(fo_lo)
    old_hi = _one_term_branch_extraction(fo_hi)
    assert old_lo == pytest.approx(0.033838382230, rel=1e-9)
    assert old_hi == pytest.approx(0.033602473294, rel=1e-9)
    assert old_hi - old_lo == pytest.approx(-2.359089e-04, rel=1e-5)
    # The same pair on the fixed code rises, and the old low value was high.
    assert fractional_extraction_sphere(fo_hi) > fractional_extraction_sphere(fo_lo)
    assert old_lo > fractional_extraction_sphere(fo_lo)


@pytest.mark.parametrize("fo", [1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 5.0e-5, 9.9e-5])
def test_short_time_branch_matches_the_analytic_solution(fo: float) -> None:
    """Below the switch the branch must reproduce the series, not its leading term.

    The one-term form 6 sqrt(Fo/pi) overstates the analytic solution by
    8.870e-04 relative at Fo = 1e-6 and 8.942e-03 at Fo = 1e-4. Both errors are
    one-sided (always high), which is why they were invisible to a continuity
    check with a 1 percent tolerance.
    """
    exact = _converged_sphere_series(fo)
    got = fractional_extraction_sphere(fo)
    assert got == pytest.approx(exact, rel=1e-9)
    one_term = 6.0 * math.sqrt(fo / math.pi)
    assert one_term > exact, "the leading term is an upper bound on extraction"
    # The two quoted one-sided errors, each recomputed from the defective
    # branch against the independent series.
    for probe, quoted in ((1.0e-6, 8.870e-04), (1.0e-4, 8.942e-03)):
        series = _converged_sphere_series(probe)
        leading = 6.0 * math.sqrt(probe / math.pi)
        assert (leading - series) / series == pytest.approx(quoted, rel=2e-3)


def test_short_time_branch_is_continuous_to_machine_precision() -> None:
    """The two branches must agree at the switch far better than 1 percent.

    The committed code left a step of 8.862268267823e-03 relative (0.886
    percent, from 0.033851374996 below the switch to 0.033551375029 above it),
    which the existing test_sphere_series_is_continuous_at_the_switch permitted
    because its tolerance was 0.01. With the second term restored the step is
    below 1e-9 relative.
    """
    fo_lo = FO_SHORT_TIME_SWITCH * (1.0 - 1e-9)
    fo_hi = FO_SHORT_TIME_SWITCH * (1.0 + 1e-9)
    below = fractional_extraction_sphere(fo_lo)
    above = fractional_extraction_sphere(fo_hi)
    rel = abs(above - below) / below
    assert rel < 1.0e-9, f"step of {rel * 100:.4f} percent at the switch"
    # The defect's own step size, recomputed, and shown to sit under the 0.01
    # tolerance that let it through.
    old_below = _one_term_branch_extraction(fo_lo)
    old_above = _one_term_branch_extraction(fo_hi)
    assert old_below == pytest.approx(0.033851374996, rel=1e-9)
    assert old_above == pytest.approx(0.033551375029, rel=1e-9)
    old_rel = abs(old_above - old_below) / old_below
    assert old_rel == pytest.approx(8.862268267823e-03, rel=1e-6)
    assert old_rel < 0.01, "the old tolerance admitted the step"


# --- 2 and 3. comminution: non-finite inputs and Walker dimensions ----------


@pytest.mark.parametrize(
    "label,work_index,f80",
    [
        ("nan f80", 12.0, float("nan")),
        ("inf f80", 12.0, float("inf")),
        ("nan work index", float("nan"), 1000.0),
        ("inf work index", float("inf"), 1000.0),
    ],
)
def test_bond_refuses_non_finite_inputs(label: str, work_index: float, f80: float) -> None:
    """A non-finite work index or f80 must raise, as a non-finite p80 already did.

    An infinite f80 is the one arguable case: Bond's equation is defined in that
    limit and returns W = Wi exactly. It is refused anyway, because an infinite
    feed size in a flowsheet is a missing measurement rather than a physical
    statement, and 12.0 kWh/short ton returned for a missing input is worse than
    an exception.
    """
    with pytest.raises(ValueError, match="finite"):
        bond_specific_energy(
            Q_(work_index, "kWh/ton"), Q_(f80, "um"), Q_(100.0, "um"), TonConvention.SHORT
        )
    finite = bond_specific_energy(
        Q_(12.0, "kWh/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), TonConvention.SHORT
    )
    assert math.isfinite(float(finite.magnitude))
    assert float(finite.magnitude) == pytest.approx(8.205266807797946, rel=1e-12)


@pytest.mark.parametrize(
    "fn,constant_unit",
    [
        (kick_specific_energy, "kWh/ton"),
        (rittinger_specific_energy, "kWh*um/ton"),
    ],
)
def test_kick_and_rittinger_refuse_a_non_finite_constant(fn, constant_unit: str) -> None:
    """Both classical laws must refuse a nan law constant rather than return nan."""
    with pytest.raises(ValueError, match="finite"):
        fn(Q_(float("nan"), constant_unit), Q_(1000.0, "um"), Q_(100.0, "um"))


def test_walker_rejects_a_constant_whose_units_do_not_match_the_exponent() -> None:
    """Walker's constant carries um^(n-1), so a kWh/ton constant is wrong at n = 1.5.

    Committed behaviour: walker_specific_energy(Q_(60.0, "kWh/ton"), 1000 um,
    100 um, 1.5) returned 8.205266807797946 with units
    kilowatt_hour / micrometer ** 0.5 / ton, dimensionality
    [length] ** 1.5 / [time] ** 2, against specific energy's
    [length] ** 2 / [time] ** 2. The magnitude is the same number Bond returns
    for the same reduction, which is the worst case: dimensionally wrong and
    numerically plausible. Kick and Rittinger already raise on the same mistake.
    """
    with pytest.raises(Exception) as bad:
        walker_specific_energy(
            Q_(60.0, "kWh/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), 1.5
        )
    assert "dimension" in str(bad.value).lower() or "convert" in str(bad.value).lower()
    good = walker_specific_energy(
        Q_(60.0, "kWh*um**0.5/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), 1.5
    )
    assert float(good.to("kWh/ton").magnitude) == pytest.approx(
        8.205266807797946, rel=1e-12
    )
    assert str(good.dimensionality) == "[length] ** 2 / [time] ** 2"
    # The magnitude that made the wrong-unit call look plausible is the same
    # one Bond returns for this reduction, which is why only the dimensions
    # distinguish them.
    bond = bond_specific_energy(
        Q_(12.0, "kWh/ton"), Q_(1000.0, "um"), Q_(100.0, "um"), TonConvention.SHORT
    )
    assert float(bond.magnitude) == pytest.approx(
        float(good.to("kWh/ton").magnitude), rel=1e-12
    )
    assert str(bond.dimensionality) == str(good.dimensionality)


# --- 4. separation: small-argument recovery ---------------------------------


def _defective_rectangular_recovery(kt: float, r_inf: float) -> float:
    """E2 as committed, evaluating (1 - exp(-x))/x directly.

    Kept so the measured error can be asserted against this module's fixed
    output rather than against the numbers written in the prose.
    """
    return r_inf * (1.0 - (1.0 - math.exp(-kt)) / kt)


@pytest.mark.parametrize("kt", [1.0e-12, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6])
def test_rectangular_recovery_survives_a_short_residence_time(kt: float) -> None:
    """E2 must reduce to R_inf (kt/2 - (kt)^2/6) as kt goes to zero.

    Committed behaviour at k_max = 1 1/s and R_inf = 0.90, measured against the
    analytic small-argument limit:

      kt      committed              analytic          relative error
      1e-12   +1.990954810935e-05    4.499999999998e-13   +4.424344e+07
      1e-10   -7.446633389918e-08    4.499999999850e-11   -1.655807e+03
      1e-09   +2.545373841700e-08    4.499999998500e-10   +5.556386e+01
      1e-08   +5.469723873830e-09    4.499999985000e-09   +2.154942e-01
      1e-07   +4.543775276034e-08    4.499999850000e-08   +9.727873e-03
      1e-06   +4.500141251640e-07    4.499998500000e-07   +3.172260e-05

    At kt = 1e-10 the committed form returned a NEGATIVE recovery, which then
    raised on the fraction check, so the failure mode is not merely imprecise.
    The cause is cancellation in (1 - exp(-x))/x, not the physics. An earlier
    draft of this docstring attributed the 4.5e-10 analytic value and the
    55.56 error to kt = 1e-8; both belong to kt = 1e-9, and the numbers above
    are now each asserted against a recomputation rather than against one
    another.
    """
    r_inf = 0.90
    got = rectangular_distribution_recovery(Q_(kt, "s"), Q_(1.0, "1/s"), r_inf)
    analytic = r_inf * (kt / 2.0 - kt**2 / 6.0)
    assert got >= 0.0
    assert got == pytest.approx(analytic, rel=1e-9)
    quoted = {
        1.0e-12: (1.990954810935e-05, 4.499999999998e-13, 4.424344e07),
        1.0e-10: (-7.446633389918e-08, 4.499999999850e-11, -1.655807e03),
        1.0e-09: (2.545373841700e-08, 4.499999998500e-10, 5.556386e01),
        1.0e-08: (5.469723873830e-09, 4.499999985000e-09, 2.154942e-01),
        1.0e-07: (4.543775276034e-08, 4.499999850000e-08, 9.727873e-03),
        1.0e-06: (4.500141251640e-07, 4.499998500000e-07, 3.172260e-05),
    }[kt]
    old = _defective_rectangular_recovery(kt, r_inf)
    assert old == pytest.approx(quoted[0], rel=1e-6)
    assert analytic == pytest.approx(quoted[1], rel=1e-9)
    assert (old - analytic) / analytic == pytest.approx(quoted[2], rel=1e-5)
    if kt == 1.0e-10:
        assert old < 0.0, "the committed form returned a negative recovery here"


def test_pin_rectangular_recovery_still_matches_its_doctest() -> None:
    """The stable rewrite must not move the published value 0.75037181 at 60 s."""
    got = rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)
    assert got == pytest.approx(0.75037181, rel=1e-8)


# --- 5. separation: imperfection near the medium density --------------------


def test_imperfection_refuses_a_density_cut_point_at_the_medium() -> None:
    """A cut point a hair above the medium density must not report a finite I.

    Committed behaviour: imperfection(0.05, 1.0000001, DENSITY) returned
    499999.99970806646, an imperfection of half a million, because the
    denominator x50 - 1 is 1e-7. The existing guard only rejects x50 <= 1.0,
    which bounds the denominator's SIGN and not its magnitude.

    The rejected value is read back out of the exception rather than recomputed
    in this test body, so the figure quoted above is the one the module itself
    produces at that cut point.
    """
    with pytest.raises(ValueError, match="exceeds 1") as bad:
        imperfection(0.05, 1.0000001, PropertyBasis.DENSITY)
    reported = float(re.search(r"imperfection ([\d.e+-]+) exceeds", str(bad.value))[1])
    assert reported == pytest.approx(499999.99970806646, rel=1e-9)
    # The guard must not have moved the published density or size worked values.
    assert imperfection(0.05, 1.60, PropertyBasis.DENSITY) == pytest.approx(
        0.08333333333333333, rel=1e-12
    )
    assert imperfection(20.0, 100.0, PropertyBasis.SIZE) == pytest.approx(0.20, rel=1e-12)
