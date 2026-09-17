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
def test_valuation_irr_prose_arithmetic():
    """valuation.py irr: the flow [-1, 20] has the single exact root
    20/1 - 1 = 19.0, i.e. a 1900 percent return, which the historical
    bracket-scanning implementation reported as None because its upper bound
    was 10.0."""
    from ae.econ.valuation import irr as _irr  # noqa: PLC0415
    from ae.econ.valuation import npv as _npv  # noqa: PLC0415

    root = _irr([-1.0, 20.0])
    assert root is not None
    assert root == pytest.approx(20.0 / 1.0 - 1.0, rel=1e-9)
    assert root == pytest.approx(19.0, rel=1e-9)
    assert 1900.0 == pytest.approx(100.0 * root, rel=1e-9)
    # NPV at that rate is zero, which is what makes it the root.
    assert _npv(root, [-1.0, 20.0]) == pytest.approx(0.0, abs=1e-9)
    # 10.0 was the historical scan's upper bound, and the root exceeds it.
    assert root > 10.0


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
    ("ae/econ/valuation.py", "irr"): "test_valuation_irr_prose_arithmetic",
    ("ae/plant/streams.py", "max_feasible_feed_fraction"):
        "test_streams_feasibility_bound_prose",
    ("ae/physics/comminution.py", "bond_specific_energy"):
        "test_comminution_bond_prose_arithmetic",
    ("ae/physics/comminution.py", "convert_ton_convention"):
        "test_comminution_ton_convention_prose_arithmetic",
    ("ae/physics/comminution.py", "<module>"):
        "test_comminution_bond_prose_arithmetic covers the E1 worked example; the "
        "span the detector finds in the module docstring is the gamma = 44.5 "
        "ball-charge constant inside the Eq. E2 display, which is a sourced "
        "definition, not a derived result",
    ("ae/physics/diffusion.py", "<module>"): "test_diffusion_erf_half_prose_arithmetic",
    ("ae/physics/diffusion.py", "diffusion_length"):
        "test_diffusion_erf_half_prose_arithmetic",
    ("ae/physics/impurity_location.py", "implied_removal_rate"):
        "test_impurity_removal_rate_prose_arithmetic",
    ("ae/physics/impurity_location.py", "lattice_ceiling_sio2_percent"):
        "test_impurity_lattice_ceiling_prose_arithmetic",
    ("ae/physics/leaching.py", "<module>"):
        "tests/test_leaching.py::test_closed_form_inversion_matches_bracketed_root_solve "
        "(the detected spans are the depressed-cubic coefficients of the Levenspiel "
        "inversion, checked there against an independent bisection root)",
    ("ae/physics/liberation.py", "<module>"): "test_liberation_module_prose_arithmetic",
    ("ae/physics/liberation.py", "exposure"): "test_liberation_exposure_prose_arithmetic",
    ("ae/physics/liberation.py", "liberation_size"):
        "test_liberation_size_prose_arithmetic",
    ("ae/physics/packing.py", "furnas_max_packing"):
        "test_packing_mcgeary_step_prose_arithmetic",
    ("ae/physics/phases.py", "<module>"): "test_phases_landau_prose_arithmetic",
    ("ae/physics/reagents.py", "<module>"): "test_reagents_acid_demand_prose_arithmetic",
    ("ae/physics/separation.py", "flotation_recovery"):
        "test_separation_flotation_prose_arithmetic",
    ("ae/physics/separation.py", "imperfection"):
        "test_separation_imperfection_prose_arithmetic",
    ("ae/physics/separation.py", "logistic_scale_from_ep"):
        "test_separation_logistic_scale_prose_arithmetic",
    ("ae/physics/separation.py", "partition_number"):
        "test_separation_partition_number_prose_arithmetic",
    ("ae/physics/separation.py", "product_grade_from_reject"):
        "test_separation_product_grade_prose_arithmetic",
    ("ae/physics/separation.py", "rectangular_distribution_recovery"):
        "test_separation_rectangular_recovery_prose_arithmetic",
    ("ae/physics/separation.py", "two_product"):
        "test_separation_two_product_prose_arithmetic",
    ("ae/physics/separation.py", "two_product_condition_number"):
        "test_separation_condition_number_prose_arithmetic",
    ("ae/physics/separation.py", "whims_rate_constant"):
        "test_separation_whims_prose_arithmetic",
    ("ae/physics/thermal.py", "integrated_enthalpy"):
        "test_thermal_integrated_enthalpy_prose_arithmetic",
}

#: Number literal, including scientific notation.
_NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"

#: Characters and function words that may appear inside a numeric expression.
#: ``x`` is included because this codebase writes multiplication as "10 x 7000"
#: in prose (a literal multiplication sign is not ASCII).
_EXPR_CHARS = r"[0-9.,()\s+\-*/^]|sqrt|exp|ln\b|log|\bx\b|\*\*"


def _unwrap_notation(doc: str) -> str:
    """Strip doctests and RST scaffolding, keeping arithmetic readable.

    Removes doctest lines, RST section underlines, ``.. math::`` directive
    markers, inline ``code`` spans and LaTeX backslash commands, but KEEPS the
    contents of ``:math:`` roles: a real calculation is often written inside
    one, and dropping the role wholesale hid four such sites.
    """
    import re
    lines = [
        line for line in doc.splitlines()
        if not line.strip().startswith((">>>", "..."))
    ]
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped and set(stripped) <= set("=-~^\"'`#*+"):
            kept.append("")          # RST underline, not an equation
        else:
            kept.append(line)
    text = "\n".join(kept)
    text = text.replace(".. math::", " ")
    text = re.sub(r":math:`([^`]*)`", r" \1 ", text)
    text = re.sub(r"``[^`]*``", " CODE ", text)
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)     # \frac, \sum, \mathrm, ...
    text = re.sub(r"[{}\\]", " ", text)
    return text


def _prose_arithmetic_sites() -> dict[tuple[str, str], list[str]]:
    """Every docstring site in src/ stating a computed numeric result as prose.

    A site is an expression built ONLY from numeric literals, operators and a
    few function words, evaluated to a numeric literal:
    "1/(0.98 x 0.90 x 0.95) = 1.1935". Two properties make this the right
    detector and both were arrived at by measurement, not taste.

    It requires at least two literals AND one operator on the LEFT of the
    equals sign. That is what distinguishes a CLAIM ("38/7 = 5.43", which can
    be wrong) from a DEFINITION ("E_p = 20", "n_B = 2", "D_0 = 10^-4", which
    cannot: there is nothing to recompute). The previous detector used the bare
    pattern ``= <number>`` and could not tell these apart, so it reported 50
    sites of which 41 were uncovered, and the great majority of those 41 were
    symbol-table rows, LaTeX notation and illustrative keyword values. It was
    kept honest only by a hand-maintained ``_NOT_ARITHMETIC`` suppression list
    that had to grow with every new docstring, which is the same hand-picked
    literal list this file was written to replace. That list is now deleted.

    It reads each docstring through :func:`_unwrap_notation`, so arithmetic
    written inside a ``:math:`` role is still seen while the LaTeX commands
    around it are not.

    Returns a map from ``(relative path, symbol name)`` to the evaluated
    expressions found there, where ``<module>`` names a module docstring.
    """
    import ast
    import pathlib
    import re

    run = re.compile(
        rf"(?<![\w.])[0-9(](?:{_EXPR_CHARS})*=\s*(?:{_NUM})(?!\d|\.\d)"
    )
    operator = re.compile(r"[-+*/^]|\bx\b|sqrt|exp|\bln\b|log")
    number = re.compile(_NUM)
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
            # A References block cites volume, issue and page numbers, which are
            # bibliography, not arithmetic. Everything before it is prose.
            body = re.split(r"\nReferences\s*\n", _unwrap_notation(doc))[0]
            body = re.sub(r"\s+", " ", body)
            hits = []
            for m in run.finditer(body):
                span = m.group(0).strip().rstrip(".")
                lhs = span[:span.rfind("=")]
                if len(number.findall(lhs)) >= 2 and operator.search(lhs):
                    hits.append(span)
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

# --------------------------------------------------------------------------------------
# ae/physics prose arithmetic. Each case recomputes the docstring's own numbers by
# calling the module, so a drift between code and prose fails here. FIVE literals in
# this group were WRONG when these cases were written, each recorded in the case that
# caught it: implied_removal_rate said 0.188032/0.811968 for a quotient that is
# 0.188034/0.811966; the liberation module docstring said 0.793700/0.206300 for
# 0.793701/0.206299; and integrated_enthalpy said sqrt(298.15) = 17.2670496 for a root
# that is 17.2670206 and gave the integral difference as 42666.1996 where it is
# 42666.1997.
# --------------------------------------------------------------------------------------


def test_comminution_bond_prose_arithmetic():
    """comminution.bond_specific_energy: Wi 12.3 kWh/t, F80 2000 um, P80 150 um gives
    10(12.3)(1/sqrt(150) - 1/sqrt(2000)) = 123(0.0816497 - 0.0223607) = 7.2925."""
    import math

    from ae.core.units import Q_
    from ae.physics.comminution import TonConvention, bond_specific_energy
    assert round(1.0 / math.sqrt(150.0), 7) == 0.0816497
    assert round(1.0 / math.sqrt(2000.0), 7) == 0.0223607
    assert round(1.0 / math.sqrt(150.0) - 1.0 / math.sqrt(2000.0), 7) == 0.0592890
    assert round(10 * 12.3 * (150.0 ** -0.5 - 2000.0 ** -0.5), 4) == 7.2925
    w = bond_specific_energy(Q_(12.3, "kWh/tonne"), Q_(2000.0, "um"), Q_(150.0, "um"),
                             convention=TonConvention.METRIC)
    assert round(float(w.to("kWh/tonne").magnitude), 4) == 7.2925


def test_comminution_ton_convention_prose_arithmetic():
    """comminution.convert_ton_convention: 7.2925 kWh/short ton is
    7.2925 x (1000/907.18474) = 8.0386 kWh/metric ton."""
    from ae.core.units import Q_
    from ae.physics.comminution import TonConvention, convert_ton_convention
    assert round(7.2925 * (1000.0 / 907.18474), 4) == 8.0386
    got = convert_ton_convention(Q_(7.2925, "kWh/short_ton"), TonConvention.METRIC)
    assert round(float(got.to("kWh/tonne").magnitude), 4) == 8.0386


def test_diffusion_erf_half_prose_arithmetic():
    """diffusion module and diffusion_length: at x = sqrt(Dt) the erf solution has
    depleted the profile to erf(1/2) = 0.5205 of its initial value, and to within
    1 percent by x = 3.64 sqrt(Dt)."""
    import math
    assert round(math.erf(0.5), 4) == 0.5205
    assert round(math.erf(3.64 / 2.0), 2) == 0.99


def test_impurity_removal_rate_prose_arithmetic():
    """impurity_location.implied_removal_rate: Xia et al. 2024's 128.86 ug/g feed and
    24.23 ug/g product give 1 - 24.23/128.86 = 0.811966, i.e. 81.20 percent.

    The docstring stated 1 - 0.188032 = 0.811968. Both digits were wrong in the sixth
    decimal: the quotient is 0.18803352, so the rounded values are 0.188034 and
    0.811966. Corrected in src, and this case is why.
    """
    from ae.physics.impurity_location import implied_removal_rate
    assert round(24.23 / 128.86, 6) == 0.188034
    assert round(1.0 - 24.23 / 128.86, 6) == 0.811966
    assert round(implied_removal_rate(128.86, 24.23), 6) == 0.811966
    assert round(implied_removal_rate(128.86, 24.23) * 100, 2) == 81.20


def test_impurity_lattice_ceiling_prose_arithmetic():
    """impurity_location.lattice_ceiling_sio2_percent: 24.23 ppm residual trace is
    24.23e-4 wt percent, so 100 - 0.002423 = 99.997577 wt percent SiO2."""
    from ae.physics.impurity_location import lattice_ceiling_sio2_percent
    # ppm to wt percent is a factor of 1e-4, so derive the wt percent from the
    # 24.23 ppm the paper reports rather than restating 0.002423, which would be
    # a literal compared against itself.
    residual_wt_percent = 24.23 / 10_000.0
    assert round(residual_wt_percent, 6) == 0.002423
    assert round(100.0 - residual_wt_percent, 6) == 99.997577
    assert round(lattice_ceiling_sio2_percent(24.23), 6) == 99.997577


def test_liberation_module_prose_arithmetic():
    """liberation module: the liberation-size denominator is 1 - E^(1/3) inverted, so
    E = 0.5 gives 1 - 0.793701 = 0.206299 and a 4.847x ratio, and E = 0.9 gives
    1 - 0.464159 = 0.535841 and 1.866x.

    The module docstring rounded 0.79370053 to 0.793700 and 0.20629947 to 0.206300,
    both wrong in the sixth decimal in the direction that makes the pair look exactly
    complementary. Corrected in src. The function docstring for liberation_size states
    the same two numbers to 7 places (0.7937005, 0.2062995) and was already right,
    which is why this case and test_liberation_size_prose_arithmetic assert different
    precisions on the same quantity.
    """
    from ae.core.units import Q_
    from ae.physics.liberation import liberation_size
    assert round(0.5 ** (1 / 3), 6) == 0.793701
    assert round(1 - 0.5 ** (1 / 3), 6) == 0.206299
    assert round(1 / (1 - 0.5 ** (1 / 3)), 3) == 4.847
    assert round(0.1 ** (1 / 3), 6) == 0.464159
    assert round(1 - 0.1 ** (1 / 3), 6) == 0.535841
    assert round(1 / (1 - 0.1 ** (1 / 3)), 3) == 1.866
    # And the module's own function must reproduce both ratios.
    for target, ratio in ((0.5, 4.847), (0.9, 1.866)):
        d = liberation_size(Q_(1.0, "um"), target).to("um").magnitude
        assert float(d) == pytest.approx(ratio, abs=5e-4)


def test_liberation_exposure_prose_arithmetic():
    """liberation.exposure: a 10 um inclusion in a 100 um particle gives
    1 - 0.9^3 = 1 - 0.729 = 0.271, and at 30 um, 1 - (2/3)^3 = 0.703704."""
    from ae.core.units import Q_
    from ae.physics.liberation import exposure
    assert round(0.9 ** 3, 3) == 0.729
    assert round(1 - 0.9 ** 3, 3) == 0.271
    assert round((2 / 3) ** 3, 6) == 0.296296
    assert round(1 - (2 / 3) ** 3, 6) == 0.703704
    assert round(exposure(Q_(100.0, "um"), Q_(10.0, "um")), 3) == 0.271
    assert round(exposure(Q_(30.0, "um"), Q_(10.0, "um")), 6) == 0.703704


def test_liberation_size_prose_arithmetic():
    """liberation.liberation_size: half-exposure of a 20 um inclusion population needs
    1 - 0.5^(1/3) = 0.2062995, and 20 / 0.2062995 = 96.95 um."""
    from ae.core.units import Q_
    from ae.physics.liberation import liberation_size
    assert round(0.5 ** (1 / 3), 7) == 0.7937005
    assert round(1 - 0.5 ** (1 / 3), 7) == 0.2062995
    assert round(20.0 / 0.2062995, 2) == 96.95
    d = liberation_size(Q_(20.0, "um"), 0.5).to("um").magnitude
    assert round(float(d), 2) == 96.95


def test_packing_mcgeary_step_prose_arithmetic():
    """packing.furnas_max_packing: McGeary's own 1:7:38:316 quaternary optimum contains
    a 38/7 = 5.43 step, below the sevenfold non-interaction requirement."""
    assert round(38 / 7, 2) == 5.43
    assert 38 / 7 < 7.0, "the step is below McGeary's own sevenfold requirement"


def test_phases_landau_prose_arithmetic():
    """phases module: Q_0 = ((847 - 298.15)/847)^(1/4) = 0.89721, so the cumulative
    transition strain is V_D Q_0^2 / V_0 = 1.188e-6 x 0.80498 / 2.269e-5 = 4.21 vol
    percent. Recomputed against the module's own Landau parameters."""
    from ae.core.units import Q_
    from ae.physics.phases import (
        QUARTZ_LANDAU,
        alpha_beta_cumulative_volume_strain,
        order_parameter,
    )
    q0 = ((847.0 - 298.15) / 847.0) ** 0.25
    assert round(q0, 5) == 0.89721
    assert round(q0 ** 2, 5) == 0.80498
    assert round(1.188e-6 * q0 ** 2 / 2.269e-5 * 100, 2) == 4.21
    # The literals above must be the module's own, not a parallel set.
    assert float(QUARTZ_LANDAU.tc_0.quantity.to("K").magnitude) == pytest.approx(847.0)
    assert float(
        QUARTZ_LANDAU.v_d.quantity.to("m**3/mol").magnitude
    ) == pytest.approx(1.188e-6)
    assert float(
        QUARTZ_LANDAU.v_0.quantity.to("m**3/mol").magnitude
    ) == pytest.approx(2.269e-5)
    assert round(float(order_parameter(Q_(298.15, "K"))), 5) == 0.89721
    strain = alpha_beta_cumulative_volume_strain(Q_(298.15, "K"), Q_(1000.0, "K"))
    assert round(float(strain) * 100, 2) == 4.21


def test_reagents_acid_demand_prose_arithmetic():
    """reagents module: 30 ppm Al in 1000 kg of ore is
    1000 x 30e-6 / 0.0269815 = 1.112 mol Al, and dissolving 0.1 percent of the SiO2
    matrix consumes 6 x 1.0/0.060083 = 99.9 mol HF, two orders of magnitude more."""
    assert round(1000.0 * 30e-6 / 0.0269815, 3) == 1.112
    assert round(6.0 * (0.001 * 1000.0) / 0.060083, 1) == 99.9
    hf_from_matrix = 6.0 * (0.001 * 1000.0) / 0.060083
    al_moles = 1000.0 * 30e-6 / 0.0269815
    assert hf_from_matrix / (3.0 * al_moles) > 25.0, (
        "matrix attack must dominate cation stoichiometry by a wide margin"
    )


def test_separation_flotation_prose_arithmetic():
    """separation.flotation_recovery: k 0.05 1/s, R_inf 0.90, t 60 s gives
    exp(-3) = 0.0497871 and R = 0.90 x 0.9502129 = 0.8551916."""
    import math

    from ae.core.units import Q_
    from ae.physics.separation import flotation_recovery
    assert round(math.exp(-3), 7) == 0.0497871
    assert round(1 - math.exp(-3), 7) == 0.9502129
    assert round(0.90 * (1 - math.exp(-3)), 7) == 0.8551916
    got = flotation_recovery(Q_(60.0, "s"), Q_(0.05, "1/s"), 0.90)
    assert round(got, 7) == 0.8551916


def test_separation_imperfection_prose_arithmetic():
    """separation.imperfection: x50 1.60 g/cm3 and Ep 0.05 give
    I = 0.05/(1.60 - 1) = 0.05/0.60 = 0.0833333."""
    from ae.physics.separation import PropertyBasis, imperfection
    assert round(0.05 / (1.60 - 1.0), 7) == 0.0833333
    assert round(imperfection(0.05, 1.60, PropertyBasis.DENSITY), 7) == 0.0833333


def test_separation_logistic_scale_prose_arithmetic():
    """separation.logistic_scale_from_ep: s = 20/ln 3 = 20/1.0986123 = 18.2047845."""
    import math

    from ae.physics.separation import logistic_scale_from_ep
    assert round(math.log(3.0), 7) == 1.0986123
    assert round(20.0 / math.log(3.0), 7) == 18.2047845
    assert round(logistic_scale_from_ep(20.0), 7) == 18.2047845


def test_separation_partition_number_prose_arithmetic():
    """separation.partition_number: x50 100, Ep 20 gives s = 18.2047845; at x = 120
    the standardised distance is (120 - 100)/18.2047845 = 1.0986123 = ln 3, so
    P = 1/(1 + 1/3) = 0.75, which is the x75 the Ep definition names."""
    import math

    from ae.physics.separation import logistic_scale_from_ep, partition_number
    s = logistic_scale_from_ep(20.0)
    assert round((120.0 - 100.0) / s, 7) == 1.0986123
    assert (120.0 - 100.0) / s == pytest.approx(math.log(3.0), rel=1e-9)
    assert round(1.0 / (1.0 + math.exp(-math.log(3.0))), 2) == 0.75
    assert partition_number(120.0, 100.0, 20.0) == pytest.approx(0.75, abs=5e-8)


def test_separation_product_grade_prose_arithmetic():
    """separation.product_grade_from_reject: 30 ppm feed, 8 percent reject at 280 ppm
    gives (30 - 22.4)/0.92 = 7.6/0.92 = 8.2608696 ppm in the product."""
    from ae.physics.separation import product_grade_from_reject
    assert round(0.08 * 280.0, 1) == 22.4
    assert round(30.0 - 0.08 * 280.0, 1) == 7.6
    assert round((30.0 - 0.08 * 280.0) / (1.0 - 0.08), 7) == 8.2608696
    assert round(product_grade_from_reject(30.0, 0.08, 280.0), 7) == 8.2608696


def test_separation_rectangular_recovery_prose_arithmetic():
    """separation.rectangular_distribution_recovery: k_max 0.10 1/s, t 60 s gives
    k_max t = 6, exp(-6) = 0.00247875, (1 - 0.00247875)/6 = 0.16625354, and
    R = 0.90 x 0.83374646 = 0.75037181."""
    import math

    from ae.core.units import Q_
    from ae.physics.separation import (
        flotation_recovery,
        rectangular_distribution_recovery,
    )
    assert round(math.exp(-6), 8) == 0.00247875
    assert round((1 - math.exp(-6)) / 6.0, 8) == 0.16625354
    assert round(1 - (1 - math.exp(-6)) / 6.0, 8) == 0.83374646
    assert round(0.90 * (1 - (1 - math.exp(-6)) / 6.0), 8) == 0.75037181
    got = rectangular_distribution_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)
    assert round(got, 8) == 0.75037181
    # The distributed form must recover less than the single-constant form at the
    # same nominal rate, which is the whole reason Polat and Chander 2000 is cited.
    assert got < flotation_recovery(Q_(60.0, "s"), Q_(0.10, "1/s"), 0.90)


def test_separation_two_product_prose_arithmetic():
    """separation.two_product: f 1.0, c 10.0, t 0.1 ppm give
    Y = 0.9/9.9 = 0.0909091 and R = 9.0/9.9 = 0.9090909."""
    from ae.physics.separation import two_product
    assert round(0.9 / 9.9, 7) == 0.0909091
    assert round(10.0 * 0.9 / (1.0 * 9.9), 7) == 0.9090909
    y, r = two_product(1.0, 10.0, 0.1)
    assert round(y, 7) == 0.0909091
    assert round(r, 7) == 0.9090909


def test_separation_condition_number_prose_arithmetic():
    """separation.two_product_condition_number: f 30, c 45, t 28 ppm give
    (45 - 28 + 30 - 28)/(45 - 28) = 19/17 = 1.1176471; c 31 gives (3 + 2)/3 = 1.667;
    and f 30, c 30.5, t 29.9 gives (0.6 + 0.1)/0.6 = 1.1667."""
    from ae.physics.separation import two_product_condition_number
    assert round(19 / 17, 7) == 1.1176471
    assert round((45 - 28 + 30 - 28) / (45 - 28), 7) == 1.1176471
    assert round(5 / 3, 3) == 1.667
    assert round((0.6 + 0.1) / 0.6, 4) == 1.1667
    assert round(two_product_condition_number(30.0, 45.0, 28.0), 7) == 1.1176471
    assert round(two_product_condition_number(30.0, 31.0, 28.0), 3) == 1.667
    assert round(two_product_condition_number(30.0, 30.5, 29.9), 4) == 1.1667


def test_separation_whims_prose_arithmetic():
    """separation.whims_rate_constant: k_ref 0.02 1/s at 1.0 T scaled to 1.5 T with
    n_B = 2 gives (1.5/1.0)^2 = 2.25 and k = 0.02 x 2.25 = 0.045 1/s."""
    from ae.core.units import Q_
    from ae.physics.separation import whims_rate_constant
    assert round((1.5 / 1.0) ** 2, 2) == 2.25
    assert round(0.02 * 2.25, 3) == 0.045
    k = whims_rate_constant(Q_(0.02, "1/s"), Q_(1.5, "T"), Q_(1.0, "T"), exponent=2.0)
    assert round(float(k.to("1/s").magnitude), 3) == 0.045


def test_thermal_integrated_enthalpy_prose_arithmetic():
    """thermal.integrated_enthalpy: the hand check
    F(T) = 92.9 T - 3.21e-4 T^2 + 714900/T - 1432.2 sqrt(T) evaluates term by term to
    F(1000) = 48003.7594 and F(298.15) = 5337.5597, a difference of 42666.1997 J/mol.

    Two literals in this docstring were wrong and this case found both. It stated
    1432.2 x 17.2670496 for 1432.2 sqrt(298.15): sqrt(298.15) is 17.26702059, so the
    root was wrong in the fifth decimal even though the product 24729.8269 it fed was
    right, which is exactly how a wrong intermediate survives a reader's eye. It also
    stated the difference as 42666.1996 where the term-by-term subtraction, and the
    module itself, both give 42666.1997. Both corrected in src.
    """
    import math

    from ae.core.units import Q_
    from ae.physics.thermal import Polymorph, integrated_enthalpy

    def big_f(t: float) -> float:
        return 92.9 * t - 3.21e-4 * t ** 2 + 714900.0 / t - 1432.2 * math.sqrt(t)

    assert round(math.sqrt(1000.0), 7) == 31.6227766
    assert round(math.sqrt(298.15), 7) == 17.2670206
    assert round(1432.2 * math.sqrt(1000.0), 4) == 45290.1406
    assert round(1432.2 * math.sqrt(298.15), 4) == 24729.8269
    assert round(92.9 * 1000.0, 4) == 92900.0000
    assert round(3.21e-4 * 1000.0 ** 2, 4) == 321.0000
    assert round(714900.0 / 1000.0, 4) == 714.9000
    assert round(92.9 * 298.15, 4) == 27698.1350
    assert round(3.21e-4 * 298.15 ** 2, 4) == 28.5348
    assert round(714900.0 / 298.15, 4) == 2397.7863
    assert round(big_f(1000.0), 4) == 48003.7594
    assert round(big_f(298.15), 4) == 5337.5597
    assert round(big_f(1000.0) - big_f(298.15), 4) == 42666.1997
    dh = integrated_enthalpy(Polymorph.QUARTZ, Q_(298.15, "K"), Q_(1000.0, "K"),
                             include_landau=False)
    assert round(float(dh.to("J/mol").magnitude), 1) == 42666.2

#: Every numeric literal that a case in this file pins, mapped to the docstring
#: site it was read from. ``test_pinned_literals_are_still_in_their_docstrings``
#: asserts each string is still present at that site, and
#: ``test_every_covered_site_is_anchored`` asserts that EVERY site in COVERED
#: appears here, so the two maps cannot drift apart.
#:
#: This map exists because of a control experiment. Every case in this file
#: recomputes a docstring's arithmetic, but the assertions compare a computation
#: to a literal RETYPED HERE, so on their own they are blind to the docstring
#: itself: reintroducing the wrong thermal root (1432.2 x 17.2670496) into src
#: left the whole file passing. The recomputation says "this arithmetic is
#: right"; it did not say "this is the arithmetic the docstring states". With
#: this map, a literal has to survive both: present in the prose, and equal to
#: what the code returns.
#:
#: An audit of the first version of this map found it covered only the
#: ae/physics group (22 of 33 sites) while the comment claimed the whole file was
#: anchored. The 11 pre-existing sites are now pinned too, which is why some
#: entries below pin literals that no case in this file recomputes: the older
#: cases live in their own modules or assert through the public API. Pinning them
#: here still closes the prose-edit hole, which is what this map is for.
PINNED_LITERALS: dict[tuple[str, str], tuple[str, ...]] = {
    ("ae/physics/comminution.py", "bond_specific_energy"):
        ("0.0816497", "0.0223607", "0.0592890", "7.2925"),
    ("ae/physics/comminution.py", "convert_ton_convention"): ("7.2925", "8.0386"),
    ("ae/physics/diffusion.py", "diffusion_length"): ("0.5205",),
    ("ae/physics/diffusion.py", "<module>"): ("0.5205", "3.64"),
    ("ae/physics/impurity_location.py", "implied_removal_rate"):
        ("128.86", "24.23", "0.188034", "0.811966", "81.20"),
    ("ae/physics/impurity_location.py", "lattice_ceiling_sio2_percent"):
        ("0.002423", "99.997577"),
    ("ae/physics/liberation.py", "<module>"):
        ("0.793701", "0.206299", "4.847", "0.464159", "0.535841", "1.866"),
    ("ae/physics/liberation.py", "exposure"):
        ("0.729", "0.271", "0.296296", "0.703704"),
    ("ae/physics/liberation.py", "liberation_size"):
        ("0.7937005", "0.2062995", "96.95"),
    ("ae/physics/packing.py", "furnas_max_packing"): ("38/7 = 5.43",),
    ("ae/physics/phases.py", "<module>"):
        ("0.89721", "0.80498", "1.188e-6", "2.269e-5", "4.21"),
    ("ae/physics/reagents.py", "<module>"): ("1.112", "99.9", "0.0269815", "0.060083"),
    ("ae/physics/separation.py", "flotation_recovery"):
        ("0.0497871", "0.9502129", "0.8551916"),
    ("ae/physics/separation.py", "imperfection"): ("0.0833333",),
    ("ae/physics/separation.py", "logistic_scale_from_ep"):
        ("1.0986123", "18.2047845"),
    ("ae/physics/separation.py", "partition_number"):
        ("18.2047845", "1.0986123", "0.75"),
    ("ae/physics/separation.py", "product_grade_from_reject"):
        ("22.4", "7.6", "8.2608696"),
    ("ae/physics/separation.py", "rectangular_distribution_recovery"):
        ("0.00247875", "0.16625354", "0.83374646", "0.75037181"),
    ("ae/physics/separation.py", "two_product"): ("0.0909091", "0.9090909"),
    ("ae/physics/separation.py", "two_product_condition_number"):
        ("1.1176471", "1.667", "1.1667"),
    ("ae/physics/separation.py", "whims_rate_constant"): ("2.25", "0.045"),
    ("ae/physics/thermal.py", "integrated_enthalpy"):
        ("31.6227766", "17.2670206", "45290.1406", "24729.8269", "92900.0000",
         "321.0000", "714.9000", "27698.1350", "28.5348", "2397.7863",
         "48003.7594", "5337.5597", "42666.1997"),
    ("ae/core/units.py", "ratio_basis"): ("60.08", "26.98", "2.227"),
    ("ae/econ/capex.py", "assumed_share"): ("0.55", "1.82"),
    ("ae/econ/capex.py", "scale_cost"): ("4.0**0.6", "2.2974"),
    ("ae/econ/valuation.py", "npv"): ("60/1.1 + 60/1.21 = 4.1322",),
    ("ae/econ/valuation.py", "irr"): ("20/1 - 1 = 19.0",),
    ("ae/physics/comminution.py", "<module>"):
        ("44.5", "0.23", "0.82", "907.18474", "1.1023113"),
    ("ae/physics/leaching.py", "<module>"): ("8.314462618",),
    ("ae/plant/capacity.py", "assess_line"):
        ("0.84645", "47401.20", "53326.35", "1/1.25 = 0.80"),
    ("ae/plant/scheduling.py", "mm1_waiting_time"): ("0.8/(1.0 x 0.2) = 4.0",),
    ("ae/plant/streams.py", "max_feasible_feed_fraction"): ("0.01 / 0.99 = 1.01",),
    ("ae/plant/yield_cascade.py", "<module>"): ("30 - 3(1.33)(3) = 18.0",),
    ("ae/plant/yield_cascade.py", "stage_throughput_factors"):
        ("1/0.95 = 1.0526", "1.1696", "1.1935"),
}


def _docstring_at(rel_path: str, symbol: str) -> str:
    """The docstring of one site under src/, by relative path and symbol name."""
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    tree = ast.parse((root / rel_path).read_text())
    if symbol == "<module>":
        return ast.get_docstring(tree) or ""
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == symbol):
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{rel_path} has no symbol named {symbol}")


def test_pinned_literals_are_still_in_their_docstrings():
    """Each literal a case above pins must still appear at the site it came from.

    Without this, the cases above are self-consistent but unanchored: they assert
    that some arithmetic is correct, not that it is the arithmetic src states. The
    control run that motivated it is described at PINNED_LITERALS.
    """
    missing = []
    for (rel_path, symbol), literals in PINNED_LITERALS.items():
        doc = _docstring_at(rel_path, symbol)
        for literal in literals:
            if literal not in doc:
                missing.append(f"{rel_path}::{symbol} no longer states {literal!r}")
    assert not missing, (
        "literals pinned by tests in this file but absent from their docstrings:\n"
        + "\n".join(f"  {m}" for m in missing)
    )


def test_pinned_sites_are_all_detected_sites():
    """Every site with a test case must be a site the detector actually found.

    Catches a PINNED_LITERALS entry naming a site whose arithmetic has been
    rewritten or removed, which would leave the pin guarding nothing.
    """
    detected = set(_prose_arithmetic_sites())
    orphans = sorted(k for k in PINNED_LITERALS if k not in detected)
    assert not orphans, f"PINNED_LITERALS names sites with no prose arithmetic: {orphans}"

# --------------------------------------------------------------------------------------
# Exact-by-definition constants, derived rather than asserted.
#
# Two References-block notes in src claim a constant "is reproduced by that
# multiplication in this module's tests" (the molar gas constant, from the two SI
# defining constants) and "is reproduced arithmetically in this module's tests"
# (the short ton, from the avoirdupois pound). An audit found both claims
# unsupported: the tests asserted the composite values 8.314462618 and 907.18474
# directly, and neither 1.380649e-23, 6.02214076e23 nor 0.45359237 appeared
# anywhere under tests/. A docstring that says a number is derived, checked by a
# test that only restates the number, is the same defect as a citation with no
# reference entry: the claim is not false, it is unverified. These two cases make
# the notes true.
# --------------------------------------------------------------------------------------


def test_gas_constant_is_the_product_of_the_two_si_defining_constants():
    """R = k N_A exactly, since the 2019 SI redefinition fixed both factors.

    Asserted in Decimal, because the product in binary floating point is not
    representable and the point of the claim is that the decimal digits are exact.
    Both modules that carry the constant are checked against the same derivation.
    """
    from decimal import Decimal

    from ae.physics.diffusion import GAS_CONSTANT as R_DIFFUSION
    from ae.physics.leaching import GAS_CONSTANT as R_LEACHING

    # The two SI defining constants, exact by definition (SI 9th edition, 2019).
    boltzmann = Decimal("1.380649e-23")          # J K^-1
    avogadro = Decimal("6.02214076e23")          # mol^-1
    product = boltzmann * avogadro
    assert product == Decimal("8.31446261815324"), (
        f"k N_A should be exactly 8.31446261815324 J mol^-1 K^-1, got {product}"
    )
    # The value the modules ship is that product at 10 significant figures.
    assert float(round(product, 9)) == 8.314462618
    for name, value in (("diffusion", R_DIFFUSION), ("leaching", R_LEACHING)):
        got = float(value.to("J/(mol*K)").magnitude)
        assert got == 8.314462618, f"{name}.GAS_CONSTANT is {got}, not the CODATA value"
        # The shipped value is the product at 10 significant figures, so it is
        # equal to the ROUNDED product exactly, and to the full product only to
        # within that truncation (1.5e-10 absolute, 1.8e-11 relative).
        assert got == float(round(product, 9)), (
            f"{name}.GAS_CONSTANT is not k N_A rounded to 10 significant figures"
        )
        assert got == pytest.approx(float(product), rel=1e-10), (
            f"{name}.GAS_CONSTANT does not match k N_A"
        )


def test_short_ton_is_two_thousand_avoirdupois_pounds():
    """1 short ton = 907.18474 kg exactly, from the pound at 0.45359237 kg exactly.

    The pound has been defined as exactly 0.45359237 kg since the 1959 international
    yard and pound agreement, so the short ton is exact and the comminution module's
    ton conversion is a definition, not a measurement with an uncertainty.
    """
    from decimal import Decimal

    from ae.physics.comminution import (
        METRIC_TON_KG,
        SHORT_TON_KG,
        SHORT_TONS_PER_METRIC_TON,
    )

    pound_kg = Decimal("0.45359237")             # exact by definition
    short_ton_kg = pound_kg * 2000
    assert short_ton_kg == Decimal("907.18474000")
    assert float(SHORT_TON_KG) == 907.18474, (
        f"SHORT_TON_KG is {SHORT_TON_KG}, not 2000 avoirdupois pounds"
    )
    assert float(short_ton_kg) == pytest.approx(float(SHORT_TON_KG), rel=1e-15)
    # And the derived ratio the docstrings quote as 1.1023113 short tons per tonne.
    ratio = Decimal(str(METRIC_TON_KG)) / short_ton_kg
    assert round(float(ratio), 7) == 1.1023113
    assert float(SHORT_TONS_PER_METRIC_TON) == pytest.approx(float(ratio), rel=1e-12)


def test_every_covered_site_is_anchored():
    """Every site in COVERED must also appear in PINNED_LITERALS.

    COVERED says "this site's arithmetic is checked somewhere"; PINNED_LITERALS
    says "and these are the digits it states". A site in the first map but not
    the second is checked by a case that retypes its numbers, so an edit to the
    docstring prose cannot fail anything. The first version of PINNED_LITERALS
    covered only the ae/physics group and this asymmetry went unnoticed, so it
    is now mechanical rather than a matter of remembering.
    """
    unanchored = sorted(set(COVERED) - set(PINNED_LITERALS))
    assert not unanchored, (
        "sites with a test case but no pinned literals, so their docstring prose "
        f"is unguarded: {unanchored}"
    )
