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
            # replacement. get_source_segment returns the function including
            # its docstring, and `body.replace(doc, "")` silently fails when
            # the source form differs from the parsed value (escapes, implicit
            # concatenation, raw strings). That made the docstring's own
            # numbers count as "present in the body", so the check was
            # self-satisfying: a deliberately wrong figure went undetected.
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
