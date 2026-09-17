"""Guards on the markdown-to-PDF converter in scripts/memo_pdf.py.

WHY THIS EXISTS. The converter's parser hung twice for 600 seconds before it
ever produced a page. Its paragraph branch treated any line starting with a
backtick as the start of a new block, while the dispatch chain only recognised
a TRIPLE backtick fence. A paragraph whose continuation line opens with an
inline code span therefore fell into the paragraph branch, collected zero
lines, reset the cursor to where it already was, and looped forever. How many
such lines the memo has is recorded in memo_pdf.BACKTICK_LINE_COUNT and
asserted below, not stated here, because that count drifted twice during
authoring while an unguarded docstring claimed otherwise.

The module docstring of memo_pdf.py claimed "test_memo_pdf.py exercises both"
before this file existed, which is a coverage claim with nothing behind it.
These tests make the claim true: they exercise the paragraph terminator set,
the cursor-advance assertion, the column-width floor that stopped "SOURCED"
rendering as "SOURC ED", and termination on the real memo.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from reportlab.pdfbase.pdfmetrics import stringWidth

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import memo_pdf as MP  # noqa: E402

MEMO = ROOT / "docs" / "decision-memo.md"


def test_paragraph_continuing_with_an_inline_code_span_terminates():
    """This exact shape hung the parser until the cursor guard was added."""
    md = ("Outputs: the file\n"
          "`docs/memo_numbers.csv`, one row per quantity, each with a tag.\n")
    flow = MP.build_flow(md)
    assert flow, "the paragraph produced no flowables"


def test_every_backtick_initial_line_in_the_real_memo_is_consumed():
    """The count is asserted, not merely required to be non-zero.

    An earlier version of this test asserted only that the offender list was
    non-empty, while memo_pdf.py's docstring stated a count of nine. That count
    was ten at the time and was not checked anywhere, so the prose and the file
    could disagree silently. The count now lives in
    memo_pdf.BACKTICK_LINE_COUNT and is compared against the memo here.
    """
    lines = MEMO.read_text().split("\n")
    offenders = [k + 1 for k, l in enumerate(lines)
                 if l.strip() and l.startswith("`") and not l.startswith("```")]
    assert offenders, (
        "this guard is vacuous unless the memo still contains a paragraph "
        "continuing with an inline code span")
    assert len(offenders) == MP.BACKTICK_LINE_COUNT, (
        f"memo_pdf.BACKTICK_LINE_COUNT is {MP.BACKTICK_LINE_COUNT} but the "
        f"memo has {len(offenders)} such lines, at {offenders}")
    flow = MP.build_flow(MEMO.read_text())
    assert len(flow) > 50, f"only {len(flow)} flowables from the full memo"


def test_memo_pdf_docstring_states_no_memo_line_numbers():
    """Line numbers in prose go stale the moment the memo gains a paragraph.

    The module docstring once listed the nine offending lines by number. An
    insertion elsewhere in the memo shifted all of them. Positions are derived
    at runtime by the test above, so they must not be restated in prose.
    """
    doc = MP.__doc__ or ""
    assert "at lines" not in doc, (
        "memo_pdf's docstring lists memo line numbers, which go stale on any "
        "edit to the memo; let the test report positions instead")


def test_bare_backtick_is_not_a_paragraph_terminator():
    """The fix is in the terminator tuple; assert on it directly."""
    src = (ROOT / "scripts" / "memo_pdf.py").read_text()
    m = re.search(r"not lines\[j\]\.startswith\(\(([^)]*)\)\)", src, re.S)
    assert m, "could not locate the paragraph terminator tuple"
    terms = re.findall(r'"([^"]*)"', m.group(1))
    assert "```" in terms, "a fenced code block must terminate a paragraph"
    assert "`" not in terms, (
        "a BARE backtick must not terminate a paragraph; that is the defect "
        "that made the parser loop forever")


def _broken_parser(paragraph_starts_at_i: bool, bare_backtick_terminates: bool):
    """Compile a variant of memo_pdf with one or both defects reintroduced.

    The original defect had TWO parts, and either alone is harmless:
      - the paragraph branch started collecting at j = i (so it could collect
        zero lines and reset the cursor to where it already was), AND
      - a bare backtick terminated a paragraph (so the zero-line case actually
        occurred).
    Only the combination loops. The fix changed both, so this helper can show
    that each part independently prevents the hang, and that the cursor
    assertion catches the combination instead of spinning.
    """
    src = (ROOT / "scripts" / "memo_pdf.py").read_text()
    if bare_backtick_terminates:
        src = src.replace(
            'and not lines[j].startswith(("#", "|", "!", "```", "- ",\n'
            '                                                "---"))',
            'and not lines[j].startswith(("#", "|", "!", "`", "- ", "---"))')
    if paragraph_starts_at_i:
        src = src.replace("            j = i + 1\n            buf = [lines[i]]",
                          "            j = i\n            buf = []")
    ns: dict = {"__name__": "memo_pdf_variant",
                "__file__": str(ROOT / "scripts" / "memo_pdf.py")}
    exec(compile(src, "memo_pdf_variant", "exec"), ns)
    return ns["build_flow"]


#: The paragraph shape that triggered the original hang.
HANG_MD = "Outputs: the file\n`docs/memo_numbers.csv`, one row per quantity.\n"


def test_cursor_advance_assertion_fires_on_the_original_defect():
    """Control: reintroduce BOTH parts; the guard must raise, not hang."""
    parser = _broken_parser(paragraph_starts_at_i=True,
                            bare_backtick_terminates=True)
    with pytest.raises(AssertionError, match="did not advance"):
        parser(HANG_MD)


@pytest.mark.parametrize("at_i,backtick", [(True, False), (False, True)])
def test_either_half_of_the_fix_alone_prevents_the_stall(at_i, backtick):
    parser = _broken_parser(paragraph_starts_at_i=at_i,
                            bare_backtick_terminates=backtick)
    assert parser(HANG_MD), "variant produced no flowables"


def test_built_pdf_is_six_pages():
    """The brief specifies a 6 page memo. This drifted three times.

    Each time I added a paragraph to the memo the PDF grew past the limit and I
    noticed only because I happened to re-read the build line. The build script
    warns on a loose range; this asserts the exact requirement against the
    shipped file, so growing the memo without re-tuning FIG_SCALE fails here.
    """
    pdf = ROOT / "docs" / "decision-memo.pdf"
    assert pdf.exists(), "build the PDF first: python scripts/memo_pdf.py"
    import pypdfium2 as pdfium
    n = len(pdfium.PdfDocument(str(pdf)))
    assert n == 6, (
        f"the brief specifies a 6 page memo; the built PDF has {n}. Re-tune "
        f"FIG_SCALE (currently {MP.FIG_SCALE}) or cut prose.")


def test_column_floor_fits_the_widest_word_in_every_memo_table():
    """"SOURCED" rendered as "SOURC ED" until the floors survived rescaling."""
    lines = MEMO.read_text().split("\n")
    checked = 0
    for idx, ln in enumerate(lines):
        if not ln.startswith("|") or (idx and lines[idx - 1].startswith("|")):
            continue
        src, k = [], idx
        while k < len(lines) and lines[k].startswith("|"):
            src.append(lines[k])
            k += 1
        cells = [[c.strip() for c in r.strip().strip("|").split("|")]
                 for r in src]
        cells = [c for c in cells
                 if not all(set(x) <= set("-: ") for x in c)]
        t = MP.table_block(src)
        assert sum(t._argW) == pytest.approx(MP.FRAME_W, abs=0.5), (
            f"table at line {idx+1} does not sum to the frame width")
        for i in range(len(cells[0])):
            widest = max((stringWidth(w, "Helvetica", MP.S["td"].fontSize)
                          for row in cells for w in row[i].split()), default=0)
            assert widest <= t._argW[i] - 6.0 + 1e-6, (
                f"table at line {idx+1} column {cells[0][i]!r}: widest word "
                f"{widest:.1f} pt exceeds usable {t._argW[i]-6.0:.1f} pt, so "
                f"it will wrap mid-word")
        checked += 1
    assert checked >= 5, f"expected several tables in the memo, saw {checked}"


def test_measured_font_metrics_not_a_per_character_estimate():
    """The first floor used a 4.0 pt per character estimate, too short here.

    The shortfall is asserted below rather than stated: the measured width of
    the longest provenance tag must exceed what that estimate predicts, which
    is why the first version of the column floor never bound.
    """
    fs = MP.S["td"].fontSize
    true_w = stringWidth("SOURCED", "Helvetica", fs)
    assert true_w > 4.0 * len("SOURCED"), (
        "the per-character estimate is meant to be an UNDERestimate here; if "
        "this ever flips, the comment in memo_pdf.py is wrong")
    src = (ROOT / "scripts" / "memo_pdf.py").read_text()
    assert "stringWidth" in src, "widths must come from measured font metrics"


def test_normalise_never_pushes_a_column_below_its_floor():
    floors = [40.0, 10.0, 10.0, 10.0]
    w = [40.0, 300.0, 300.0, 300.0]
    out = MP._normalise(list(w), floors)
    assert sum(out) == pytest.approx(MP.FRAME_W, abs=0.5)
    for k, (o, f) in enumerate(zip(out, floors)):
        assert o >= f - 1e-6, f"column {k} fell to {o:.1f} below floor {f:.1f}"


def test_normalise_degrades_gracefully_when_floors_exceed_the_frame():
    floors = [MP.FRAME_W] * 3
    out = MP._normalise([MP.FRAME_W] * 3, floors)
    assert sum(out) == pytest.approx(MP.FRAME_W, abs=0.5)
    assert all(o > 0 for o in out)
