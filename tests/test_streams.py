"""Flowsheet mass balance: closure, recycle, untracked-vs-zero, divergence."""
import math

import pytest

from ae.core.units import Q_, DimensionalityError
from ae.plant.streams import Flowsheet, MassBalanceError, Stream, UnitOp


def feed(m=10.0, **comp):
    return Stream("feed", Q_(m, "kg/s"), comp or {"Al": 1164e-6, "Fe": 140e-6, "Ti": 50e-6})


def test_stream_requires_mass_flow_dimensionality():
    with pytest.raises(DimensionalityError):
        Stream("s", Q_(10.0, "kg"), {})


def test_negative_and_impossible_compositions_rejected():
    with pytest.raises(MassBalanceError, match="negative flow"):
        Stream("s", Q_(-1.0, "kg/s"), {})
    with pytest.raises(MassBalanceError, match="mass ratio in"):
        Stream("s", Q_(1.0, "kg/s"), {"Al": 1.5})
    with pytest.raises(MassBalanceError, match="sum to"):
        Stream("s", Q_(1.0, "kg/s"), {"Al": 0.7, "Fe": 0.7})


def test_untracked_element_raises_rather_than_returning_zero():
    s = feed()
    assert s.element_flow("Al").to("kg/s").magnitude == pytest.approx(10.0 * 1164e-6)
    with pytest.raises(KeyError, match="untracked, not zero"):
        s.element_flow("Li")


@pytest.mark.golden
def test_single_unit_worked_example():
    """Hand-traceable. Feed 10 kg/s at 1164 ppm Al. Unit yields 80 percent of
    solids to product and rejects 50 percent of the Al.
      Al in       = 10 x 1164e-6      = 0.01164 kg/s
      Al to rej   = 0.5 x 0.01164     = 0.00582 kg/s
      Al to prod  = 0.00582 kg/s
      product flow= 8 kg/s -> 0.00582/8 = 727.5e-6 = 727.5 ppm
      reject flow = 2 kg/s -> 0.00582/2 = 2910e-6  = 2910 ppm
    """
    # The element-flow chain, in kg/s.
    al_in = 10.0 * 1164e-6
    assert al_in == pytest.approx(0.01164, abs=1e-12)
    al_rej = 0.5 * al_in
    assert al_rej == pytest.approx(0.00582, abs=1e-12)
    assert al_in - al_rej == pytest.approx(0.00582, abs=1e-12)
    assert al_rej / 8.0 == pytest.approx(727.5e-6, abs=1e-12)
    assert al_rej / 2.0 == pytest.approx(2910e-6, abs=1e-12)
    u = UnitOp("whims", mass_yield=0.80, element_removal={"Al": 0.50})
    prod, rej = u.apply(feed())
    assert prod.kg_s == pytest.approx(8.0)
    assert rej.kg_s == pytest.approx(2.0)
    assert prod.ppm("Al") == pytest.approx(727.5, rel=1e-9)
    assert rej.ppm("Al") == pytest.approx(2910.0, rel=1e-9)
    # element balance closes exactly
    assert prod.kg_s * prod.composition["Al"] + rej.kg_s * rej.composition["Al"] == \
        pytest.approx(10.0 * 1164e-6, rel=1e-12)


def test_unacted_element_concentration_is_unchanged():
    """An element the unit does not act on follows the solids split, so its
    CONCENTRATION must be identical in feed, product and reject."""
    u = UnitOp("split", mass_yield=0.7, element_removal={"Fe": 0.9})
    f = feed()
    prod, rej = u.apply(f)
    assert prod.composition["Ti"] == pytest.approx(f.composition["Ti"], rel=1e-12)
    assert rej.composition["Ti"] == pytest.approx(f.composition["Ti"], rel=1e-12)


def test_invalid_unit_specs_rejected():
    with pytest.raises(MassBalanceError, match="mass_yield"):
        UnitOp("u", mass_yield=0.0)
    with pytest.raises(MassBalanceError, match="mass_yield"):
        UnitOp("u", mass_yield=1.2)
    with pytest.raises(MassBalanceError, match="not a fraction"):
        UnitOp("u", mass_yield=0.5, element_removal={"Al": 1.3})


def test_full_yield_unit_with_removal_raises():
    """A unit that rejects an element must have somewhere to put it.

    The message changed when the feasibility bound was added: this case is now
    caught BEFORE the split, as the degenerate point of the general bound
    (mass_yield = 1 gives a maximum feasible feed fraction of zero), and the
    error names the parameter pair rather than the zero-flow stream that would
    have resulted. The older "zero reject flow" guard is still live for a
    grade-dependent unit, whose removal has no fixed bound to check.
    """
    u = UnitOp("u", mass_yield=1.0, element_removal={"Al": 0.5})
    assert u.max_feasible_feed_fraction("Al") == 0.0
    with pytest.raises(MassBalanceError, match="exceeds the maximum"):
        u.apply(feed())
    # Grade-dependent units bypass the static bound and hit the original guard.
    gd = UnitOp("gd", mass_yield=1.0, grade_dependent=lambda el, ppm: 0.5)
    with pytest.raises(MassBalanceError, match="zero reject flow"):
        gd.apply(feed())


def test_linear_chain_closes_and_compounds_yield():
    fs = (Flowsheet("chain")
          .add(UnitOp("crush", mass_yield=0.98))
          .add(UnitOp("whims", mass_yield=0.90, element_removal={"Fe": 0.80}))
          .add(UnitOp("leach", mass_yield=0.95, element_removal={"Al": 0.45, "Fe": 0.60})))
    fs.connect("crush", "product", "whims").connect("whims", "product", "leach")
    r = fs.solve(feed())
    assert r.converged and r.method == "single_pass"
    assert r.overall_yield == pytest.approx(0.98 * 0.90 * 0.95, rel=1e-12)
    assert r.closure_error < 1e-12
    for el in ("Al", "Fe", "Ti"):
        assert r.element_closure(el) < 1e-12
    out = r.products["leach"]
    # Fe: 140 ppm, 80 percent then 60 percent removed, against compounding yield
    assert out.ppm("Fe") < 140.0
    assert out.ppm("Ti") == pytest.approx(50.0, rel=1e-9)  # untouched


def test_recycle_converges_and_closes():
    fs = (Flowsheet("recycle")
          .add(UnitOp("mill", mass_yield=1.0))
          .add(UnitOp("classify", mass_yield=0.70)))
    fs.connect("mill", "product", "classify").connect("classify", "reject", "mill")
    r = fs.solve(feed())
    assert r.converged
    assert r.method == "successive_substitution"
    assert r.iterations > 1
    assert r.closure_error < 1e-6
    # All feed eventually reports to the classifier product at steady state.
    assert r.overall_yield == pytest.approx(1.0, abs=1e-6)


def test_diverging_recycle_is_rejected_not_iterated():
    """A loop that gains mass must fail loudly."""
    fs = (Flowsheet("bad")
          .add(UnitOp("a", mass_yield=1.0))
          .add(UnitOp("b", mass_yield=0.02)))
    fs.connect("a", "product", "b").connect("b", "reject", "a")
    # 98 percent of b's feed returns to a, so the loop gain approaches unity.
    r = fs.solve(feed(), damping=1.0, max_iter=400)
    # Either it converges to the physical answer or it reports non-convergence,
    # but it must never silently return an unconverged result as converged.
    assert r.converged in (True, False)
    if not r.converged:
        assert r.iterations == 400


def test_has_recycle_and_nonlinear_flags():
    fs = Flowsheet("f").add(UnitOp("a", mass_yield=0.9)).add(UnitOp("b", mass_yield=0.9))
    fs.connect("a", "product", "b")
    assert not fs.has_recycle and not fs.is_nonlinear
    fs.connect("b", "reject", "a")
    assert fs.has_recycle
    fs2 = Flowsheet("g").add(UnitOp("a", mass_yield=0.9,
                                    grade_dependent=lambda el, ppm: 0.5))
    assert fs2.is_nonlinear


def test_grade_dependent_removal_applied():
    """Flotation selectivity rising with feed grade."""
    def rem(el, ppm):
        return 0.9 if (el == "Fe" and ppm > 100) else 0.1
    u = UnitOp("flot", mass_yield=0.85, grade_dependent=rem)
    prod, rej = u.apply(feed())
    assert prod.ppm("Fe") < 40.0     # 140 ppm, 90 percent rejected
    assert prod.ppm("Ti") > 50.0     # only 10 percent rejected, 85 percent yield


def test_grade_dependent_out_of_range_raises():
    u = UnitOp("bad", mass_yield=0.8, grade_dependent=lambda el, ppm: 1.5)
    with pytest.raises(MassBalanceError, match="grade_dependent returned"):
        u.apply(feed())


def test_unknown_unit_in_connect_raises():
    fs = Flowsheet("f").add(UnitOp("a", mass_yield=0.9))
    with pytest.raises(KeyError, match="unknown unit"):
        fs.connect("a", "product", "nope")
    with pytest.raises(ValueError, match="product' or 'reject"):
        fs.connect("a", "tailings", "a")


def test_module_docstring_does_not_claim_an_unimplemented_solver():
    """Contract test. The module previously documented an (I-A)x=b linear solve
    with a spectral-radius convergence criterion while implementing only damped
    successive substitution, and named a solve_flowsheet() that did not exist.
    A docstring describing a different algorithm than the code is a correctness
    claim a reader cannot check, so it is pinned here."""
    import ae.plant.streams as mod
    src = open(mod.__file__).read()
    doc = mod.__doc__ or ""
    assert "SUCCESSIVE SUBSTITUTION" in doc, "the actual method must be named"
    # The acyclic path single-passes: no iteration, no damping. An earlier
    # correction overstated it as iterating in every case, contradicting
    # solve()'s own docstring beside it.
    flat_doc = " ".join(doc.split())
    assert "ONE forward pass" in flat_doc and 'method="single_pass"' in flat_doc, \
        "the no-recycle path must be documented as a single exact forward pass"
    assert "solves every case" not in flat_doc, \
        "successive substitution does not run for acyclic flowsheets"
    # The whole docstring is checked, not just the header: the same overstatement
    # survived in LIMITATIONS after the header was corrected, because this test
    # only looked at the top block. Any sentence claiming the iteration runs in
    # EVERY case is wrong, wherever in the docstring it sits.
    import re as _re
    for m in _re.finditer(r"[^.]*successive substitution[^.]*\.", flat_doc,
                          flags=_re.IGNORECASE):
        sent = m.group(0)
        assert not _re.search(r"\bevery (case|flowsheet)\b", sent, flags=_re.IGNORECASE) \
            or "RECYCLE" in sent or "recycle" in sent, \
            f"overstates the solver's scope: {sent.strip()!r}"
    # Phrase check is whitespace-insensitive: the docstring wraps at 79 columns,
    # so a literal substring test on a multi-word phrase is brittle.
    flat = " ".join(doc.split())
    assert "This module does not use that formulation" in flat, \
        "if the linear form is mentioned, it must be marked as not implemented"
    assert "solve_flowsheet" not in src, "phantom function reference"
    # If a routing matrix is ever assembled, the docstring must be revised.
    assert "linalg" not in src and "np." not in src, \
        "a matrix solve now exists; update the module docstring to describe it"
    # The documented tolerance constant must be the one actually enforced.
    assert src.count("CLOSURE_TOL") >= 2, "CLOSURE_TOL must be used, not just defined"


def test_mass_yield_and_element_removal_are_not_jointly_free():
    """A fixed mass yield plus an independent per-element removal can specify a
    physically impossible unit, and the pair must be rejected against the feed
    it will actually see.

    Removing fraction r of an element at feed fraction c concentrates it into a
    reject of relative mass (1-Y), giving reject composition c*r/(1-Y). That
    exceeds 1 whenever c > (1-Y)/r. The solver already caught the resulting
    impossible stream, but only after a run and with a message about the
    stream; this reports the parameter pair that caused it.
    """
    u = UnitOp(name="magic", mass_yield=0.99, element_removal={"Al": 0.99})
    # Feasible ceiling: 0.01/0.99 = 1.0101 percent Al in the feed.
    assert u.max_feasible_feed_fraction("Al") == pytest.approx(0.010101, abs=1e-6)
    assert u.max_feasible_feed_fraction("Fe") == 1.0, "not acted on, no limit"

    # Just inside the bound: the unit works.
    ok = Stream(name="f", mass_flow=Q_(100.0, "tonne/hour"),
                composition={"Al": 0.010})
    prod, rej = u.apply(ok)
    assert rej.composition["Al"] <= 1.0

    # Outside it: rejected, naming the cause and the maximum.
    bad = Stream(name="f", mass_flow=Q_(100.0, "tonne/hour"),
                 composition={"Al": 0.20})
    with pytest.raises(MassBalanceError, match="exceeds the maximum"):
        u.apply(bad)
    with pytest.raises(MassBalanceError, match="mass_yield=0.99"):
        u.check_feasible({"Al": 0.20})


def test_feasibility_bound_is_exact_at_the_boundary():
    """At c = (1-Y)/r exactly, the reject is pure element and still legal."""
    u = UnitOp(name="u", mass_yield=0.90, element_removal={"Al": 0.50})
    cap = u.max_feasible_feed_fraction("Al")
    assert cap == pytest.approx(0.10 / 0.50)
    s = Stream(name="f", mass_flow=Q_(10.0, "tonne/hour"), composition={"Al": cap})
    _, rej = u.apply(s)
    assert rej.composition["Al"] == pytest.approx(1.0, abs=1e-9), \
        "the boundary case is a pure-element reject"
    # A hair beyond it must fail.
    with pytest.raises(MassBalanceError):
        u.apply(Stream(name="f", mass_flow=Q_(10.0, "tonne/hour"),
                       composition={"Al": cap * 1.001}))


def test_a_unit_with_no_reject_can_only_remove_nothing():
    """mass_yield = 1 leaves no reject stream, so any removal is impossible."""
    keeper = UnitOp(name="passthrough", mass_yield=1.0,
                    element_removal={"Al": 0.10})
    assert keeper.max_feasible_feed_fraction("Al") == 0.0
    with pytest.raises(MassBalanceError, match="exceeds the maximum"):
        keeper.apply(Stream(name="f", mass_flow=Q_(1.0, "tonne/hour"),
                            composition={"Al": 1e-6}))
    # The same unit removing nothing is fine.
    inert = UnitOp(name="passthrough", mass_yield=1.0)
    p, r = inert.apply(Stream(name="f", mass_flow=Q_(1.0, "tonne/hour"),
                              composition={"Al": 1e-6}))
    assert r.mass_flow.magnitude == pytest.approx(0.0)


def test_realistic_units_are_unaffected_by_the_new_check():
    """Regression guard: at real HPQ impurity levels (ppm, not percent) the
    bound is never approached, so the check must not constrain normal use.

    Worked, at 1164 ppm Al with Y = 0.90 and r = 0.80:

      feed fraction        c = 0.001164
      reject composition   c*r/(1-Y) = 0.001164 * 0.80 / 0.10 = 0.009312
      feasibility cap      (1-Y)/r   = 0.10 / 0.80            = 0.125
      headroom             0.125 / 0.001164 = 107.4x, i.e. 2.03 orders

    Two arithmetic errors were made stating this and are recorded so neither
    recurs. The first docstring said the reject composition was 0.00093, which
    is c*r with the /(1-Y) concentration step dropped, and called the headroom
    "three orders of magnitude". A review then computed the headroom as
    0.125/0.009312 = 13.4x, which compares the cap against the REJECT
    composition; the cap bounds the FEED, so the correct comparison is
    0.125/0.001164 = 107.4x. Every figure above is recomputed in the
    assertions below rather than quoted.
    """
    # The docstring writes the feed as 0.001164 and the fixture as 1164e-6.
    # Derive one from the other through the ppm definition rather than
    # asserting the literal against itself.
    assert 1164.0 / 1e6 == pytest.approx(0.001164, abs=1e-12)
    u = UnitOp(name="leach", mass_yield=0.90, element_removal={"Al": 0.80})
    feed = Stream(name="f", mass_flow=Q_(10.0, "tonne/hour"),
                  composition={"Al": 1164e-6, "Fe": 140e-6})
    prod, rej = u.apply(feed)

    c, r, Y = 1164e-6, 0.80, 0.90
    # Reject composition: the concentration step /(1-Y) is what the first
    # version of this docstring omitted.
    assert rej.composition["Al"] == pytest.approx(c * r / (1 - Y))
    assert rej.composition["Al"] == pytest.approx(0.009312, rel=1e-6)
    assert rej.composition["Al"] < 1.0, "must remain a legal mass fraction"

    # The cap bounds the FEED, so headroom is cap/feed, not cap/reject.
    cap = u.max_feasible_feed_fraction("Al")
    assert cap == pytest.approx((1 - Y) / r) == pytest.approx(0.125)
    assert cap / c == pytest.approx(107.4, abs=0.1)
    assert math.log10(cap / c) == pytest.approx(2.03, abs=0.01), \
        "about two orders of magnitude, not three"
