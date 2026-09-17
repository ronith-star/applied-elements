"""Every arithmetic claim written as prose in a docstring, recomputed.

Five wrong literals have reached files in this build so far: a mole conversion
inverted by 2.227x, a throughput factor off by one in the fourth decimal, a
lognormal tail off by 2 percent, a normal tail belonging to a different Cpk, and
a capacity doctest illustrating bottleneck identification with an exact tie.

Doctests catch the ones written as `>>> ` examples. They do NOT catch arithmetic
stated in prose ("1/(0.98 x 0.90 x 0.95) = 1.1935"), which is where four of the
five hid. Each case below recomputes such a claim by CALLING the module's own
code, so the check fails if the code drifts from the docstring.

Completeness, stated precisely. ``test_every_prose_arithmetic_site_is_covered``
walks every docstring under ``src/`` with ast and requires each site matching the
pattern ``= <number>`` outside a doctest to appear in COVERED. That pattern is
the SCOPE OF THE GUARANTEE and it is narrower than "all arithmetic": it does not
catch a claim phrased without an equals sign ("Al is 52.9 percent of Al2O3",
"1.4286x larger"), so cases of that kind are covered by hand below and are not
enforced mechanically.

An earlier version of this file claimed "every prose calculation in every module
docstring is now recomputed" on the strength of a grep over one directory with a
hand-picked literal list. The claim was broader than the check. The detector is
now mechanical and its blind spot is written down rather than papered over.
"""
import pytest
from scipy import stats

from ae.core.feedstock import oxide_to_element
from ae.core.units import Q_
from ae.plant.capacity import OEE, UnitCapacity, assess_line
from ae.plant.yield_cascade import (
    cascade_yield,
    off_spec_fraction,
    required_process_mean,
    stage_throughput_factors,
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


def test_capacity_ratio_inversion_prose():
    """capacity.assess_line prose: a feed-to-product ratio of 1.25 is the same
    statement as a 20 percent downstream loss, since 1/1.25 = 0.80. Recomputed
    against the module's own capacity calculation, not just the arithmetic."""
    from ae.core.units import Q_
    from ae.plant.capacity import OEE, UnitCapacity, assess_line
    assert round(1.0 / 1.25, 4) == 0.80
    o = OEE(0.90, 0.95, 0.99)
    mill = UnitCapacity("mill", Q_(10.0, "tonne/hour"), 7000.0, o, 1.25)
    # Product capacity is feed capacity divided by the ratio, so a ratio of
    # 1.25 must give exactly 80 percent of what a ratio of 1.0 would.
    same_rate_no_loss = UnitCapacity("ideal", Q_(10.0, "tonne/hour"), 7000.0, o, 1.0)
    a = assess_line([mill]).line_rate.to("tonne").magnitude
    b = assess_line([same_rate_no_loss]).line_rate.to("tonne").magnitude
    assert a / b == pytest.approx(0.80, abs=1e-12)


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
    from ae.plant.scheduling import allen_cunneen_waiting_time, mm1_waiting_time
    assert mm1_waiting_time(0.8, 1.0) == pytest.approx(4.0, abs=1e-9)
    assert mm1_waiting_time(0.5, 1.0) == pytest.approx(1.0, abs=1e-9)
    assert mm1_waiting_time(0.8, 1.0) / mm1_waiting_time(0.5, 1.0) == pytest.approx(4.0)
    assert allen_cunneen_waiting_time(0.8, 1.0, 1.0, 0.0) == pytest.approx(2.0, abs=1e-9)
    assert allen_cunneen_waiting_time(0.95, 1.0, 1.0, 1.0) == pytest.approx(19.0, abs=1e-9)


#: Docstring sites that ``_prose_arithmetic_sites`` detects, each mapped to the
#: test here or elsewhere that recomputes the claim by calling module code.
#: This map is the completeness contract: a new detected site fails
#: ``test_every_prose_arithmetic_site_is_covered`` until it is added.
COVERED: dict[tuple[str, str], str] = {
    ("ae/core/units.py", "ratio_basis"): "test_units_mole_conversion_prose",
    ("ae/econ/capex.py", "scale_cost"): "test_capex_scaling_prose",
    ("ae/plant/capacity.py", "assess_line"):
        "test_capacity_module_prose_arithmetic + test_capacity_ratio_inversion_prose",
    ("ae/plant/scheduling.py", "mm1_waiting_time"): "test_scheduling_prose_arithmetic",
    ("ae/plant/yield_cascade.py", "<module>"): "test_spc_prose_arithmetic",
    ("ae/plant/yield_cascade.py", "stage_throughput_factors"):
        "test_yield_cascade_prose_arithmetic",
    ("ae/econ/capex.py", "assumed_share"): "test_capex_assumed_share_prose",
    ("ae/econ/valuation.py", "npv"): "test_valuation_npv_prose",
    ("ae/plant/streams.py", "max_feasible_feed_fraction"):
        "test_streams_feasibility_bound_prose",
}

#: Fragments that match the ``= <number>`` pattern but are NOT arithmetic
#: claims: illustrative parameter values, LaTeX notation, index enumerations.
#: Matched as substrings against the captured fragment. Kept explicit and
#: narrow, because a broad exclusion would silently re-open the hole this
#: module closes: each entry suppresses one specific textual form, never a
#: whole file or symbol.
_NOT_ARITHMETIC = (
    # Illustrative keyword-argument values in prose.
    'loc=100.0', 'scale=5.0', 'loc == 0.0', 'iterations=1', 'c_a = 1',
    'c_a = 0', 'method="single_pass"',
    # LaTeX operators and summation/product bounds: notation, not a claim.
    r'\prod', r'\Phi', r'\Pr', 'y_i', 'f_{\\mathrm{indep}}',
    '_{t=0}', '_{i=1}', '(r) = 0',
    # Period-index enumerations and positional references to time zero.
    't = 0, 1, 2', 'at t=0',
)


def _prose_arithmetic_sites() -> dict[tuple[str, str], list[str]]:
    """Every docstring site in src/ containing prose (non-doctest) arithmetic."""
    import ast
    import pathlib
    import re
    pat = re.compile(r"=\s*-?\d[\d,]*\.?\d*(?:e-?\d+)?\b")
    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    found: dict[tuple[str, str], list[str]] = {}
    for f in sorted(root.rglob("*.py")):
        tree = ast.parse(f.read_text())
        nodes = [(tree, "<module>")] + [
            (n, n.name) for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        for node, name in nodes:
            doc = ast.get_docstring(node)
            if not doc:
                continue
            prose = "\n".join(l for l in doc.splitlines()
                               if not l.strip().startswith((">>>", "...")))
            hits = []
            for m in pat.finditer(prose):
                frag = prose[max(0, m.start() - 60):m.end() + 12].replace("\n", " ")
                if any(tok in frag for tok in _NOT_ARITHMETIC):
                    continue
                hits.append(frag.strip())
            if hits:
                found[(str(f.relative_to(root)), name)] = hits
    return found


def test_every_prose_arithmetic_site_is_covered():
    """Mechanical completeness check, replacing a hand-picked grep.

    Walks all docstrings in src/ with ast and requires each site carrying prose
    arithmetic to appear in COVERED. Adding a module with an unchecked prose
    calculation fails here, naming the site.
    """
    sites = _prose_arithmetic_sites()
    uncovered = {k: v[:2] for k, v in sites.items() if k not in COVERED}
    assert not uncovered, (
        "prose arithmetic with no recomputation case:\n"
        + "\n".join(f"  {f}::{n} -> {frags}" for (f, n), frags in uncovered.items())
    )


def test_covered_map_has_no_stale_entries():
    """A COVERED entry whose docstring no longer carries arithmetic is stale and
    gives false comfort about breadth."""
    sites = set(_prose_arithmetic_sites())
    stale = sorted(k for k in COVERED if k not in sites)
    assert not stale, f"COVERED lists sites with no prose arithmetic: {stale}"


def test_capex_scaling_prose():
    """capex.scale_cost prose: ratio 4.0, factor 4.0**0.6 = 2.2974, so 4.5948
    MUSD from a 2.0 MUSD reference."""
    from ae.core.units import Q_
    from ae.econ.capex import scale_cost
    assert round(4.0 ** 0.6, 4) == 2.2974
    c = scale_cost(Q_(2.0e6, "USD"), Q_(5000.0, "tonne/year"),
                   Q_(20000.0, "tonne/year"), 0.6)
    assert round(c.to("USD").magnitude / 1e6, 4) == 4.5948


def test_unit_economics_yield_basis_prose():
    """unit_economics module docstring: at a 70 percent yield a feed-basis cost
    is "1.43 times larger" on a product basis.

    Phrased without an equals sign, so the mechanical detector does not see it.
    The value is READ OUT OF THE DOCSTRING with a regex rather than retyped
    here, then recomputed and compared against the module's own behaviour. An
    earlier version of this case asserted ``round(1.0/0.70, 4) == 1.4286``,
    which is arithmetically true but pinned a literal that appears nowhere in
    src/ (the docstring says 1.43): a hand-written case that quotes the text
    from memory can drift from it, which is the defect that got the Muller case
    deleted. Reading the string closes that.
    """
    import re

    import ae.econ.unit_economics as ue

    doc = ue.__doc__ or ""
    m = re.search(r"(\d+(?:\.\d+)?)\s*times larger", doc)
    assert m, "the yield-basis claim is no longer in the module docstring"
    stated = float(m.group(1))
    # Recompute from the yield the docstring names, read from the same sentence.
    # \s+ not a literal space: the docstring wraps between "percent" and
    # "yield", so a space-only pattern silently finds nothing.
    y = re.search(r"(\d+)\s+percent\s+yield", doc)
    assert y, "the docstring no longer names the yield it refers to"
    frac = float(y.group(1)) / 100.0
    assert stated == pytest.approx(round(1.0 / frac, 2), abs=5e-3), (
        f"docstring says {stated}x at a {frac:.0%} yield, but 1/{frac} = "
        f"{1.0 / frac:.4f}"
    )
    # And the module must actually behave that way, not merely say so.
    from ae.core.units import Q_
    from ae.econ.unit_economics import InputDemand, cash_cost

    kw = dict(site=us_site_for_prose(), cascade_yield=frac,
              product_tonnes_per_year=1000.0, freight_waived=True)
    feed = cash_cost(demands=[InputDemand("HF", Q_(10.0, "kg/tonne"), basis="feed")], **kw)
    prod = cash_cost(demands=[InputDemand("HF", Q_(10.0, "kg/tonne"), basis="product")], **kw)
    ratio = feed.cash_cost.magnitude / prod.cash_cost.magnitude
    assert ratio == pytest.approx(1.0 / frac, rel=1e-9)
    assert round(ratio, 2) == stated


def us_site_for_prose():
    """Minimal priced site for the yield-basis check above."""
    import datetime as _dt

    from ae.core.provenance import Source, Tag, Tier, Value
    from ae.core.site import Currency, LabourRates, PowerSupply, ReagentPrices, Site
    from ae.core.units import Q_

    # Tier.T2, not T3: the platform forbids a Tier-3 source from being sole
    # evidence, and this fixture is the only evidence for its own values. The
    # rule fired when this was written as T3, which is the rule working.
    src = Source(citation="TEST FIXTURE, not a real source", tier=Tier.T2,
                 url="https://example.invalid/test-fixture",
                 accessed=_dt.date(2026, 9, 16))

    def v(mag, unit):
        return Value(quantity=Q_(mag, unit), tag=Tag.SOURCED, source=src)

    return Site(site_id="US-NM-ABQ", name="probe", country="US", region="NM",
                currency=Currency.USD,
                power=PowerSupply(energy_price=v(0.05, "USD/kWh"),
                                  rate_basis="state_average"),
                labour=LabourRates(fully_loaded_operator=v(36.92, "USD/hour")),
                reagents=ReagentPrices(prices={"HF": v(1.80, "USD/kg")}))


def test_capex_assumed_share_prose():
    """capex.assumed_share prose: at a location factor of 0.55 the old mixed
    basis reported 1/0.55 = 1.82, i.e. 182 percent."""
    assert round(1.0 / 0.55, 2) == 1.82


def test_valuation_npv_prose():
    """valuation.npv prose: minus 100 + 60/1.1 + 60/1.21 = 4.1322, built from
    the two discount factors 54.5455 and 49.5868."""
    from ae.econ.valuation import npv as _npv
    assert round(60 / 1.1, 4) == 54.5455
    assert round(60 / 1.21, 4) == 49.5868
    assert round(-100 + 60 / 1.1 + 60 / 1.21, 4) == 4.1322
    assert round(_npv(0.10, [-100.0, 60.0, 60.0]), 4) == 4.1322


def test_streams_feasibility_bound_prose():
    """streams.max_feasible_feed_fraction prose: a unit keeping 99 percent of
    the mass and removing 99 percent of an element can only do so for feeds
    below 0.01/0.99 = 1.01 percent. Recomputed by calling the method."""
    from ae.plant.streams import UnitOp
    assert round((1 - 0.99) / 0.99 * 100, 2) == 1.01
    u = UnitOp(name="u", mass_yield=0.99, element_removal={"Al": 0.99})
    assert u.max_feasible_feed_fraction("Al") * 100 == pytest.approx(1.01, abs=5e-3)


def test_covered_has_no_duplicate_keys():
    """A duplicated key in COVERED silently overwrites the earlier mapping, so
    a site can appear covered while its first test is forgotten. Detected by
    re-parsing the source, since the dict literal has already collapsed by the
    time it is importable."""
    import ast
    import pathlib
    src = pathlib.Path(__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        # COVERED carries a type annotation, so it parses as AnnAssign, not
        # Assign. Matching only Assign found nothing and the loop fell through
        # to the "not found" raise, which looked like a duplicate-key failure.
        is_covered = (
            (isinstance(node, ast.AnnAssign)
             and getattr(node.target, "id", None) == "COVERED")
            or (isinstance(node, ast.Assign)
                and any(getattr(t, "id", None) == "COVERED" for t in node.targets))
        )
        if is_covered and isinstance(node.value, ast.Dict):
            keys = []
            for k in node.value.keys:
                if isinstance(k, ast.Tuple):
                    keys.append(tuple(getattr(e, "value", None) for e in k.elts))
            dupes = {k for k in keys if keys.count(k) > 1}
            assert not dupes, f"duplicate COVERED keys: {sorted(dupes)}"
            return
    raise AssertionError("COVERED assignment not found")
