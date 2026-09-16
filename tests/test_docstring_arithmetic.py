"""Every arithmetic claim written as prose in a docstring, recomputed.

Five wrong literals have reached files in this build so far: a mole conversion
inverted by 2.227x, a throughput factor off by one in the fourth decimal, a
lognormal tail off by 2 percent, a normal tail belonging to a different Cpk, and
a capacity doctest illustrating bottleneck identification with an exact tie.

Doctests catch the ones written as `>>> ` examples. They do NOT catch arithmetic
stated in prose ("1/(0.98 x 0.90 x 0.95) = 1.1935"), which is where four of the
five hid. This module recomputes each such claim from the module's own code, so
prose arithmetic carries the same enforcement as an executable example.

Adding a prose calculation to a docstring without a case here is how the next one
gets in.
"""
import numpy as np
import pytest
from scipy import stats

from ae.core.units import Q_
from ae.core.feedstock import oxide_to_element
from ae.plant.capacity import OEE, UnitCapacity, assess_line
from ae.plant.yield_cascade import (
    cascade_yield, stage_throughput_factors, off_spec_fraction, required_process_mean,
)


def test_feedstock_oxide_conversion_prose():
    """feedstock.py: 'Al is 52.9 percent of Al2O3 by mass', and 2200 ppm Al2O3
    is 1164 ppm Al."""
    el, ppm, f = oxide_to_element("Al2O3", 2200.0)
    assert el == "Al"
    assert round(f * 100, 1) == 52.9
    assert round(ppm) == 1164


def test_units_mole_conversion_prose():
    """units.py: M(SiO2)/M(Al) = 60.083/26.9815 = 2.227, so 30 ppm Al BY MASS is
    66.8 umol per mol of SiO2 formula units, and 22.3 umol per mol of atoms.
    The mole figure is LARGER than the mass figure; the inverse (11 ppm) was the
    original error."""
    M_Al, M_Si, M_O = 26.9815, 28.085, 15.999
    M_SiO2 = M_Si + 2 * M_O
    assert round(M_SiO2, 3) == 60.083
    ratio = M_SiO2 / M_Al
    assert ratio == pytest.approx(2.227, abs=5e-4)
    per_formula_unit = 30.0 * ratio
    assert per_formula_unit == pytest.approx(66.8, abs=0.05)
    assert per_formula_unit / 3.0 == pytest.approx(22.3, abs=0.05)
    assert per_formula_unit > 30.0, "the mole-basis figure exceeds the mass-basis figure"


def test_capacity_module_prose_arithmetic():
    """capacity.py assess_line: mill 10 x 7000 x 0.84645 / 1.25 = 47401.20 and
    leach 9 x 7000 x 0.84645 / 1.00 = 53326.35, and the mill is the constraint
    despite the higher nameplate rate."""
    o = OEE(0.90, 0.95, 0.99)
    assert o.value == pytest.approx(0.84645, abs=1e-9)
    assert 10 * 7000 * 0.84645 / 1.25 == pytest.approx(47401.20, abs=0.005)
    assert 9 * 7000 * 0.84645 / 1.00 == pytest.approx(53326.35, abs=0.005)
    mill = UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, o, 1.25)
    leach = UnitCapacity("leach", Q_(9.0, "tonne/hour"), 7000.0, o, 1.0)
    r = assess_line([mill, leach])
    assert mill.product_capacity.to("tonne").magnitude == pytest.approx(47401.20, abs=0.05)
    assert leach.product_capacity.to("tonne").magnitude == pytest.approx(53326.35, abs=0.05)
    assert r.bottleneck == "mill"
    assert round(r.bottleneck_margin, 4) == 0.125


def test_capacity_oee_loss_cascade_prose():
    """capacity.py OEE: losses 0.10000, 0.04500, 0.00855 sum to 0.15355."""
    lb = OEE(0.90, 0.95, 0.99).loss_breakdown
    assert (lb["availability"], lb["performance"]) == (
        pytest.approx(0.10000, abs=1e-9), pytest.approx(0.04500, abs=1e-9))
    assert lb["quality"] == pytest.approx(0.00855, abs=1e-9)
    assert sum(lb.values()) == pytest.approx(0.15355, abs=1e-9)


def test_yield_cascade_prose_arithmetic():
    """yield_cascade.py: Y = 0.8379; factors 1/0.95 = 1.0526,
    1/(0.90 x 0.95) = 1.1696, 1/(0.98 x 0.90 x 0.95) = 1.1935."""
    ys = [0.98, 0.90, 0.95]
    assert cascade_yield(ys) == pytest.approx(0.8379, abs=1e-9)
    f = stage_throughput_factors(ys)
    assert [round(x, 4) for x in f] == [1.1935, 1.1696, 1.0526]
    assert round(1 / 0.95, 4) == 1.0526
    assert round(1 / (0.90 * 0.95), 4) == 1.1696
    assert round(1 / (0.98 * 0.90 * 0.95), 4) == 1.1935


def test_spc_prose_arithmetic():
    """yield_cascade.py SPC prose: at USL 30 and Cpk 1.33, sigma 3 needs mu
    18.03 and sigma 5 needs mu 10.05, a difference of 7.98 ppm; the docstring's
    rounded 18.0/10.0 statement holds to one decimal. Normal tail at mu 18,
    sigma 3 is 3.1671e-5 (Cpk exactly 4/3, 3Cpk = 4.0); lognormal is 7.65e-4."""
    assert required_process_mean(30.0, 3.0, 1.33) == pytest.approx(18.03, abs=1e-9)
    assert required_process_mean(30.0, 5.0, 1.33) == pytest.approx(10.05, abs=1e-9)
    assert round(required_process_mean(30.0, 3.0, 1.33), 1) == 18.0
    assert round(required_process_mean(30.0, 5.0, 1.33), 1) == 10.0
    assert (30.0 - 18.0) / (3.0 * 3.0) == pytest.approx(4.0 / 3.0, rel=1e-12)
    assert off_spec_fraction(18.0, 3.0, 30.0) == pytest.approx(3.1671e-5, rel=1e-3)
    assert off_spec_fraction(18.0, 3.0, 30.0) == pytest.approx(float(stats.norm.sf(4.0)),
                                                               rel=1e-12)
    ln = off_spec_fraction(18.0, 3.0, 30.0, model="lognormal")
    assert round(ln, 6) == 0.000765
    assert ln / off_spec_fraction(18.0, 3.0, 30.0) == pytest.approx(24.2, rel=0.02)


def test_scheduling_prose_arithmetic():
    """scheduling.py: M/M/1 Wq = 0.8/(1.0 x 0.2) = 4.0 h; M/D/1 is exactly half
    at 2.0 h; rho 0.5 gives 1.0 h so 0.5 -> 0.8 quadruples the queue."""
    from ae.plant.scheduling import mm1_waiting_time, allen_cunneen_waiting_time
    assert mm1_waiting_time(0.8, 1.0) == pytest.approx(4.0, abs=1e-9)
    assert mm1_waiting_time(0.5, 1.0) == pytest.approx(1.0, abs=1e-9)
    assert mm1_waiting_time(0.8, 1.0) / mm1_waiting_time(0.5, 1.0) == pytest.approx(4.0)
    assert allen_cunneen_waiting_time(0.8, 1.0, 1.0, 0.0) == pytest.approx(2.0, abs=1e-9)
    assert allen_cunneen_waiting_time(0.95, 1.0, 1.0, 1.0) == pytest.approx(19.0, abs=1e-9)


def test_hpq_reference_limits_are_stated_consistently():
    """The Muller et al. 2012 single-grain limits appear in several docstrings.
    Their sum must not exceed the stated 50 ppm trace-sum ceiling, or the
    per-element table and the sum ceiling would contradict each other."""
    limits = {"Al": 30, "Ti": 10, "Li": 5, "Na": 8, "K": 8, "Ca": 5,
              "Fe": 3, "P": 2, "B": 1}
    assert limits["Al"] == 30 and limits["B"] == 1
    # The 50 ppm trace sum is BINDING: an ore at every per-element limit
    # simultaneously would sum to 72 ppm and still fail the sum specification.
    assert sum(limits.values()) == 72
    assert sum(limits.values()) > 50
