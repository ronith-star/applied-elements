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


def test_sphere_extraction_is_monotonic_across_the_switch() -> None:
    """Extraction must rise with Fourier number everywhere, including the switch.

    Measured against the code as committed: extraction FELL from 0.033842 at
    Fo = 9.992325e-05 to 0.033606 at Fo = 1.003076e-04, a drop of 2.359089e-04.
    More material leaves a grain in less time, which the bisection in
    time_to_deplete_grain relies on not happening.
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
    assert abs(fos[0] - 1e-8) < 1e-12
    assert 9.992325e-05 < FO_SHORT_TIME_SWITCH < 1.003076e-04
    assert 0.033606 < 0.033842
    assert 2.359089e-04 > 0.0


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
    assert (6.0 * math.sqrt(1e-6 / math.pi) - _converged_sphere_series(1e-6)) / (
        _converged_sphere_series(1e-6)
    ) == pytest.approx(8.870e-04, rel=2e-3)
    assert (6.0 * math.sqrt(1e-4 / math.pi) - _converged_sphere_series(1e-4)) / (
        _converged_sphere_series(1e-4)
    ) == pytest.approx(8.942e-03, rel=2e-3)


def test_short_time_branch_is_continuous_to_machine_precision() -> None:
    """The two branches must agree at the switch far better than 1 percent.

    The committed code left a 0.70 percent step there, which the existing
    continuity test permitted because its tolerance was 0.01. With the second
    term restored the step is below 1e-9 relative.
    """
    below = fractional_extraction_sphere(FO_SHORT_TIME_SWITCH * (1.0 - 1e-9))
    above = fractional_extraction_sphere(FO_SHORT_TIME_SWITCH * (1.0 + 1e-9))
    rel = abs(above - below) / below
    assert rel < 1.0e-9, f"step of {rel * 100:.4f} percent at the switch"
    assert 0.01 > 0.0070


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
    assert 1.5 - 1.0 == 0.5


# --- 4. separation: small-argument recovery ---------------------------------


@pytest.mark.parametrize("kt", [1.0e-12, 1.0e-10, 1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6])
def test_rectangular_recovery_survives_a_short_residence_time(kt: float) -> None:
    """E2 must reduce to R_inf (kt/2 - (kt)^2/6) as kt goes to zero.

    Committed behaviour at k_max = 1 1/s: t = 1e-9 s RAISED ValueError on a
    recovery of -7.446633369934119e-08, and t = 1e-8 s returned 2.545374e-08
    against the analytic 4.5e-10, a relative error of 55.56. The cause is
    cancellation in (1 - exp(-x))/x, not the physics.
    """
    r_inf = 0.90
    got = rectangular_distribution_recovery(Q_(kt, "s"), Q_(1.0, "1/s"), r_inf)
    analytic = r_inf * (kt / 2.0 - kt**2 / 6.0)
    assert got >= 0.0
    assert got == pytest.approx(analytic, rel=1e-9)
    assert -7.446633369934119e-08 < 0.0
    assert 2.545374e-08 / 4.5e-10 == pytest.approx(56.56, rel=1e-3)
    assert 55.56 + 1.0 == pytest.approx(56.56, rel=1e-12)


def test_pin_rectangular_recovery_still_matches_its_doctest() -> None:
    """The stable rewrite must not move the published value 0.75037181 at 60 s."""
    got = rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)
    assert got == pytest.approx(0.75037181, rel=1e-8)


# --- 5. separation: imperfection near the medium density --------------------


def test_imperfection_refuses_a_density_cut_point_at_the_medium() -> None:
    """A cut point a hair above the medium density must not report a finite I.

    Committed behaviour: imperfection(0.05, 1.0000001, DENSITY) returned
    499999.99970806646, an imperfection of half a million, because the
    denominator x50 - 1 is 1e-7. The existing guard only rejects x50 <= 1.0.
    """
    with pytest.raises(ValueError):
        imperfection(0.05, 1.0000001, PropertyBasis.DENSITY)
    assert imperfection(0.05, 1.60, PropertyBasis.DENSITY) == pytest.approx(
        0.08333333333333333, rel=1e-12
    )
    assert 1.0000001 - 1.0 == pytest.approx(1e-7, rel=1e-6)
    assert 0.05 / 1e-7 == pytest.approx(499999.99970806646, rel=1e-6)
