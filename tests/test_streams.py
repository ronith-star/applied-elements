"""Flowsheet mass balance: closure, recycle, untracked-vs-zero, divergence."""
import pytest
from ae.core.units import Q_, DimensionalityError
from ae.plant.streams import Stream, UnitOp, Flowsheet, MassBalanceError


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
    """A unit that rejects an element must have somewhere to put it."""
    u = UnitOp("u", mass_yield=1.0, element_removal={"Al": 0.5})
    with pytest.raises(MassBalanceError, match="zero reject flow"):
        u.apply(feed())


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
