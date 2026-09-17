"""The Excel mirror must recompute to the Python model's values.

A workbook that ships pasted Python results is a screenshot: it goes stale the
moment a driver changes, and a reader who edits an input and sees nothing move
learns not to trust the file. So every calculated cell in the mirror is an
Excel formula, and this module CHECKS THAT CLAIM by loading the workbook into
an independent formula engine, recalculating it, and comparing against the
Python model directly.

That is a different test from reading the numbers the build script wrote. The
build script could write a correct Python value into column C and a broken
formula into column B, and a test that only read the file would pass. Here the
formulas are evaluated by an engine that has never seen the Python model.

Two defects this caught while it was being written are worth recording. The
first version of the cost build omitted fixed costs entirely, summing
electricity, reagent and per-tonne labour to a cash cost of 158 USD/t: about
twenty times too low for HPQ purification, and every scenario looked
profitable. The platform's own cash_cost() takes labour, maintenance and
overhead as arguments precisely because a cash cost is not three line items.
The second was that after restructuring the cost rows, the hardcoded row
offsets in the downstream NPV and breakeven formulas still pointed at the old
rows, which only surfaced when the formulas were actually evaluated.
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

formulas = pytest.importorskip(
    "formulas", reason="the Excel formula engine is needed to recalculate the "
                       "mirror; without it this claim cannot be checked"
)

import build_workbook as bwb  # noqa: E402

WB = "AE-model-mirror.xlsx"


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory) -> dict:
    """Build the workbook, then recalculate it with an independent engine."""
    out = tmp_path_factory.mktemp("wb")
    warnings.filterwarnings("ignore")
    bwb.build(out / WB)
    xl = formulas.ExcelModel().loads(str(out / WB)).finish()
    return xl.calculate()


def _cell(sol: dict, sheet: str, ref: str):
    v = sol.get(f"'[{WB}]{sheet}'!{ref}")
    try:
        return v.value[0, 0]
    except Exception:
        return v


@pytest.mark.benchmark
def test_workbook_formulas_reproduce_the_python_model(evaluated) -> None:
    """Every reconciled quantity must agree to floating-point precision.

    The reconciliation sheet computes each quantity twice: column B by the
    workbook's own formulas, column C from the Python model. Agreement to
    1e-12 means the spreadsheet is a faithful mirror and not a parallel
    implementation that happens to be close. The lognormal off-spec fraction
    is the demanding one, since Excel's LOGNORM.DIST must be fed the
    method-of-moments parameters that scipy derives internally.
    """
    diffs = []
    for row in range(5, 10):
        name = _cell(evaluated, "RECONCILIATION", f"A{row}")
        wb_v = _cell(evaluated, "RECONCILIATION", f"B{row}")
        py_v = _cell(evaluated, "RECONCILIATION", f"C{row}")
        d = _cell(evaluated, "RECONCILIATION", f"D{row}")
        assert isinstance(name, str) and name, f"row {row} has no label"
        assert isinstance(wb_v, (int, float)), (
            f"{name}: the workbook cell did not evaluate to a number "
            f"(got {wb_v!r}), which means the formula is broken"
        )
        assert isinstance(py_v, (int, float)), f"{name}: no Python value"
        diffs.append((name, float(d)))

    assert len(diffs) == 5, f"expected 5 reconciled rows, read {len(diffs)}"
    worst_name, worst = max(diffs, key=lambda t: t[1])
    assert worst < 1e-12, (
        f"{worst_name} diverges by {worst:.3e}; the workbook and the Python "
        "model are no longer the same model"
    )


def test_status_cell_reports_reconciled(evaluated) -> None:
    """The workbook must say so itself, for a reader who never runs pytest."""
    assert _cell(evaluated, "RECONCILIATION", "D12") == "RECONCILED"


def test_cash_cost_is_physically_plausible(evaluated) -> None:
    """A regression guard on the omitted-fixed-cost defect.

    The first build summed electricity, reagent and per-tonne labour only and
    returned 158 USD/t. Published HPQ purification cash costs sit in the
    high hundreds to low thousands per tonne, so a figure under 400 means a
    cost category has gone missing again rather than that the plant is
    efficient.
    """
    cc = _cell(evaluated, "RECONCILIATION", "B9")
    assert isinstance(cc, (int, float))
    assert 400.0 < float(cc) < 5000.0, (
        f"cash cost {float(cc):,.1f} USD/t is outside any plausible range for "
        "HPQ purification; check whether fixed costs are being counted"
    )


def test_every_calculated_cell_is_a_formula_not_a_pasted_value() -> None:
    """The central claim of the workbook, checked structurally.

    Read with openpyxl WITHOUT computing values: a calculated cell must hold a
    string beginning with "=". If the build script ever pastes a Python result
    into the Model sheet's calculation block, the workbook silently stops
    being live and this test is what notices.
    """
    openpyxl = pytest.importorskip("openpyxl")
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = pathlib.Path(td) / WB
        bwb.build(path)
        wb = openpyxl.load_workbook(path)          # formulas, not values
        ws = wb["Model"]
        # Find the calculation block by its header, then require every
        # populated B cell below it to be a formula.
        start = None
        for row in ws.iter_rows(min_col=1, max_col=1):
            if row[0].value == "CALCULATED (live formulas)":
                start = row[0].row
                break
        assert start is not None, "the Model sheet lost its calculation block"

        checked = 0
        for r in range(start + 1, ws.max_row + 1):
            label = ws.cell(row=r, column=1).value
            val = ws.cell(row=r, column=2).value
            if not label or val is None:
                continue
            assert isinstance(val, str) and val.startswith("="), (
                f"Model!B{r} ({label!r}) holds {val!r}, a pasted value. Every "
                "cell in the calculation block must be a live formula."
            )
            checked += 1
        assert checked >= 15, (
            f"only {checked} calculated cells found; the block looks truncated"
        )
