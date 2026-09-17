"""Numbers claimed in a TEST docstring must appear in that test's own body.

Rationale, from a real defect. test_streams.py gained a docstring stating a
reject composition of 0.00093 and headroom of "three orders of magnitude"; the
correct figures are 0.009312 and 107.4x (2.03 orders). The assertions were
right, so the whole suite passed while the prose was wrong twice over: the
first figure dropped the /(1-Y) concentration step, and a later review
recomputed the headroom against the reject composition rather than the feed.

The src/ guard in test_docstring_arithmetic.py does not cover this, because it
scans src/ only, and extending its registry would mean hand-registering 45
sites. For tests the cheaper invariant holds: a number a test docstring states
should be verifiable by reading the same function, so it must appear in the
body (as a literal, or via the arithmetic that produces it). Numbers that are
genuinely descriptive rather than computed are listed explicitly below.
"""
import ast
import pathlib
import re

import pytest

TESTS = pathlib.Path(__file__).resolve().parent

#: Numbers in prose that are not claims about computed values: counts of
#: things, standard identifiers, dates, section references, tolerances stated
#: in words. Each entry is (filename, number-as-written) and suppresses ONE
#: number in ONE file, never a whole docstring.
_DESCRIPTIVE: set[tuple[str, str]] = {
    # Defect archaeology: figures recorded precisely because they were WRONG.
    ("test_streams.py", "0.00093"), ("test_streams.py", "13.4"),
    ("test_surrogate.py", "1.03"), ("test_surrogate.py", "1.2"),
    ("test_citations.py", "70"), ("test_citations.py", "1993"),
    ("test_capex.py", "30"), ("test_capex.py", "50"),
    # Counts of repository facts, asserted in dedicated tests elsewhere.
    ("test_citations.py", "6"), ("test_citations.py", "4"),
    ("test_citations.py", "2002"), ("test_citations.py", "2019"),
    ("test_citations.py", "1952"),
    ("test_feedstock.py", "7"),
    # Standard identifiers and published class numbers.
    ("test_capex.py", "18"), ("test_capex.py", "97"),
    ("test_capex.py", "4"), ("test_capex.py", "5"), ("test_capex.py", "3"),
    ("test_capex.py", "1"), ("test_capex.py", "2"),
}

#: Numbers in prose, INCLUDING scientific notation. A first version matched
#: only ``\d+(\.\d+)?`` and split "5.0e-05" into "5.0" and "05", reporting a
#: phantom claim of 05 that no test body could ever satisfy.
_NUM = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![\w.])"
)


def _is_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when the function's first statement is its docstring."""
    return bool(
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


def _test_functions() -> list[tuple[str, str, str, str]]:
    """(filename, test name, docstring, source of the function body)."""
    out = []
    for f in sorted(TESTS.rglob("test_*.py")):
        if f.name in ("test_docstring_arithmetic.py", __file__.split("/")[-1]):
            continue  # meta-modules: their prose is about other files' numbers
        src = f.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            # The docstring must be removed by AST SURGERY, not string
            # replacement. get_source_segment returns the function INCLUDING
            # its docstring, and `body.replace(doc, "")` never matched, for a
            # reason worth naming precisely: ast.get_docstring applies
            # inspect.cleandoc, which DEDENTS the text and strips the trailing
            # blank line, so what it returns is not a substring of the indented
            # source. Verified directly: for a plain indented docstring with no
            # escapes and no raw prefix, get_docstring gives
            # 'First line.\n\nsecond line' while the raw Constant holds
            # 'First line.\n\n    second line\n    '. The replace was
            # therefore a silent no-op, the docstring's own numbers counted as
            # "present in the body", and the check was self-satisfying: a
            # deliberately wrong figure planted as a control went undetected.
            # (An earlier version of this comment blamed escapes and raw
            # strings. That was wrong, and it mattered: it would have sent the
            # next reader looking for a quoting bug rather than an indentation
            # one.)
            stmts = node.body[1:] if _is_docstring(node) else node.body
            body = "\n".join(
                ast.get_source_segment(src, st) or "" for st in stmts
            )
            out.append((f.name, node.name, doc, body))
    return out


@pytest.mark.parametrize(
    "fname,tname,doc,body",
    _test_functions(),
    ids=[f"{f}::{t}" for f, t, _, _ in _test_functions()],
)
def test_docstring_numbers_appear_in_the_test_body(fname, tname, doc, body):
    prose = "\n".join(
        line for line in doc.splitlines() if not line.strip().startswith(">>>")
    )
    claimed = {m.group(1) for m in _NUM.finditer(prose)}
    missing = []
    for n in sorted(claimed):
        if (fname, n) in _DESCRIPTIVE:
            continue
        if n in body:
            continue
        # A figure may be written 0.125 in prose and 0.125 / 1 in code, or as
        # an unformatted float; compare numerically against every literal in
        # the body before reporting it.
        try:
            val = float(n)
        except ValueError:
            continue
        body_nums = [float(x) for x in _NUM.findall(body)]
        if any(abs(val - b) <= max(1e-9, abs(val) * 1e-6) for b in body_nums):
            continue
        missing.append(n)
    assert not missing, (
        f"{fname}::{tname} docstring states {missing} but the test body never "
        f"uses those values. Either assert the number or move it to "
        f"_DESCRIPTIVE with a reason. This is the check that the 0.00093 / "
        f"'three orders of magnitude' defect in test_streams.py evaded."
    )


def test_no_assertion_compares_a_literal_against_itself() -> None:
    """A guard against the way the guard above was first satisfied.

    Four docstring claims were "closed" by binding the quoted number to a name
    and asserting that name against the same literal, for example::

        ST_X3_ANALYTIC = 0.24
        assert ST_X3_ANALYTIC == pytest.approx(0.24, abs=0.005)

    That passes the number-presence check and verifies nothing: it is true by
    construction whatever the science says. It is the identical
    self-satisfying pattern that made the first version of this module blind,
    so it gets a guard of its own rather than a note.

    Detected structurally: an assertion whose two sides are each either a
    numeric literal or a name bound to a numeric literal in the same function,
    with no call, attribute access, or arithmetic on either side. ``approx``
    wrappers are unwrapped before the comparison is judged, since
    ``approx(0.24)`` is still just the literal.
    """
    import ast as _ast

    offenders: list[str] = []
    for f in sorted(TESTS.rglob("test_*.py")):
        tree = _ast.parse(f.read_text())
        for fn in _ast.walk(tree):
            if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            # Names bound to a bare numeric literal anywhere in this
            # function, MINUS any name that is also assigned a computed value
            # somewhere in the same function. A loop accumulator initialised
            # to a literal (``prev = 1e9`` then ``prev = floor``, ``tested = 0``
            # then ``tested += 1``) holds a computed value by the time it is
            # asserted, so treating it as a literal produced false positives on
            # two legitimate tests.
            lit_names: dict[str, float] = {}
            computed_names: set[str] = set()
            for st in _ast.walk(fn):
                if isinstance(st, (_ast.Assign, _ast.AnnAssign)):
                    tgts = (st.targets if isinstance(st, _ast.Assign)
                            else [st.target])
                    val = st.value
                    if isinstance(val, _ast.Tuple):
                        pairs = []
                        for t in tgts:
                            if isinstance(t, _ast.Tuple):
                                pairs = list(zip(t.elts, val.elts))
                        for tn, tv in pairs:
                            if (isinstance(tn, _ast.Name)
                                    and isinstance(tv, _ast.Constant)
                                    and isinstance(tv.value, (int, float))):
                                lit_names[tn.id] = float(tv.value)
                        continue
                    is_lit = (isinstance(val, _ast.Constant)
                              and isinstance(val.value, (int, float)))
                    for t in tgts:
                        if isinstance(t, _ast.Name):
                            if is_lit:
                                lit_names[t.id] = float(val.value)
                            else:
                                computed_names.add(t.id)
                if isinstance(st, _ast.AugAssign) and isinstance(st.target, _ast.Name):
                    computed_names.add(st.target.id)   # tested += 1
                if isinstance(st, (_ast.For, _ast.comprehension)):
                    tgt = st.target
                    for nm in _ast.walk(tgt):
                        if isinstance(nm, _ast.Name):
                            computed_names.add(nm.id)  # loop variables

            def _literal_value(node: _ast.AST) -> float | None:
                """The constant a node denotes, or None if it computes."""
                # pytest.approx(x, ...) is transparent for this purpose.
                if (isinstance(node, _ast.Call)
                        and isinstance(node.func, _ast.Attribute)
                        and node.func.attr == "approx"
                        and node.args):
                    return _literal_value(node.args[0])
                if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
                    inner = _literal_value(node.operand)
                    return None if inner is None else -inner
                if (isinstance(node, _ast.Constant)
                        and isinstance(node.value, (int, float))):
                    return float(node.value)
                if isinstance(node, _ast.Name):
                    if node.id in computed_names:
                        return None
                    return lit_names.get(node.id)
                return None

            for st in _ast.walk(fn):
                if not isinstance(st, _ast.Assert):
                    continue
                cmp = st.test
                if not isinstance(cmp, _ast.Compare) or len(cmp.ops) != 1:
                    continue
                if not isinstance(cmp.ops[0], (_ast.Eq, _ast.LtE, _ast.GtE)):
                    continue
                lv = _literal_value(cmp.left)
                rv = _literal_value(cmp.comparators[0])
                if lv is None or rv is None:
                    continue
                # Both sides are literals or literal-bound names: vacuous.
                offenders.append(
                    f"{f.name}::{fn.name} line {st.lineno}: "
                    f"{_ast.unparse(cmp)[:70]}"
                )

    assert not offenders, (
        "assertions comparing a literal against itself verify nothing; "
        "recompute the claim from the model or from its closed form:\n  "
        + "\n  ".join(offenders)
    )
