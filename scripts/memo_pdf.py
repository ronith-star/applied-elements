"""Render docs/decision-memo.md to a paginated PDF with reportlab.

Headless Chrome is present on this machine but its --print-to-pdf writes
nothing from inside the sandbox (exit 0, no file), and pandoc, weasyprint and
a LaTeX toolchain are all absent, so the PDF is composed directly from the
markdown rather than via HTML.

The converter handles only the subset of markdown the memo uses: ATX headings,
paragraphs, pipe tables, fenced code, ordered and unordered lists, horizontal
rules, images with a following bold caption paragraph, and inline bold, code
and links. Anything not recognised is emitted as a literal paragraph.

DEFECT FIXED HERE, recorded because it cost two interrupted runs. The first
version of this loop had a paragraph branch that stopped collecting at any line
beginning with a backtick, while the dispatch chain only recognised a TRIPLE
backtick fence. A paragraph whose continuation line starts with an inline code
span, of which this memo has nine (at lines 9, 36, 39, 84, 192, 280, 381, 382
and 383), therefore fell into the paragraph branch, collected zero lines, set
the cursor back to where it already was, and looped forever: the cursor never
advanced past that line. The two 600 second runs were that infinite loop, not a
slow renderer and not a kernel problem.

Three changes prevent the class of bug rather than the instance. First, the
paragraph branch no longer treats a bare backtick as a terminator; only a
triple backtick fence stops it. Second, the paragraph branch now always
consumes its first line (j starts at i + 1), so it cannot collect zero lines
even if a future terminator is added carelessly. Third, and independently of
both, the main loop asserts on every iteration that the cursor strictly
advanced, so any branch that fails to consume input raises immediately with the
offending line number instead of hanging.

tests/test_memo_pdf.py measures all three. It reintroduces the two parser
defects, separately and together, and asserts that the combination raises
"did not advance" (a control that would hang without the third change) while
either part alone still parses. It also asserts the terminator tuple contains
the triple fence and not the bare backtick, that build_flow terminates on the
real memo including its nine backtick-initial continuation lines, and that
every table column in the memo is wide enough for its widest unbreakable word.
Measured: 10 passed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, Image,
                                KeepTogether, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MEMO = DOCS / "decision-memo.md"
OUT = DOCS / "decision-memo.pdf"
FIGDIR = DOCS / "figures"

NAVY = colors.HexColor("#1f4e79")
NAVY2 = colors.HexColor("#24537f")
GREY = colors.HexColor("#555555")
RULE = colors.HexColor("#c8d4e0")
MARGIN_X = 12 * mm
MARGIN_TOP = 11 * mm
MARGIN_BOT = 12 * mm
FRAME_W = A4[0] - 2 * MARGIN_X
#: Figure width as a fraction of the text frame. See the image branch.
FIG_SCALE = 0.68

S = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.6,
                         leading=16, spaceAfter=3),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.0,
                         leading=12, textColor=NAVY, spaceBefore=7.5,
                         spaceAfter=3),
    "h3": ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=8.8,
                         leading=10.6, textColor=NAVY2, spaceBefore=5,
                         spaceAfter=2),
    "p": ParagraphStyle("p", fontName="Helvetica", fontSize=7.9, leading=10.3,
                        alignment=TA_JUSTIFY, spaceAfter=3.8),
    "li": ParagraphStyle("li", fontName="Helvetica", fontSize=7.9, leading=10.3,
                         alignment=TA_JUSTIFY, leftIndent=12,
                         firstLineIndent=-12, spaceAfter=3.0),
    "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=6.9, leading=8.6,
                          textColor=GREY, alignment=TA_LEFT, spaceBefore=1.5,
                          spaceAfter=5.5),
    "code": ParagraphStyle("code", fontName="Courier", fontSize=6.8, leading=8.8,
                           backColor=colors.HexColor("#f6f8fa"), borderPadding=4,
                           leftIndent=4, spaceAfter=5),
    "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=6.8,
                         leading=8.2),
    "td": ParagraphStyle("td", fontName="Helvetica", fontSize=6.8, leading=8.2),
}


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier" size="7.0">\1</font>', t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t


def _normalise(w: list[float], floors: list[float]) -> list[float]:
    """Scale widths to exactly FRAME_W without pushing any below its floor.

    Columns already AT their floor are pinned; the slack is taken from, or
    given to, the columns that have room above their floor, in proportion to
    how much room each has. If the floors alone exceed FRAME_W there is no
    feasible solution, so the floors are scaled down uniformly and the caller
    gets a table that is tight but still sums to the frame width.
    """
    total_floor = sum(floors)
    if total_floor >= FRAME_W:
        k = FRAME_W / total_floor
        return [f * k for f in floors]
    pinned = [abs(w[i] - floors[i]) < 1e-9 for i in range(len(w))]
    for _ in range(len(w) + 1):
        free_now = sum(w[i] for i in range(len(w)) if not pinned[i])
        target_free = FRAME_W - sum(floors[i] for i in range(len(w)) if pinned[i])
        if free_now <= 0:
            break
        k = target_free / free_now
        out = [floors[i] if pinned[i] else w[i] * k for i in range(len(w))]
        newly = [i for i in range(len(w))
                 if not pinned[i] and out[i] < floors[i] - 1e-9]
        if not newly:
            return out
        for i in newly:
            pinned[i] = True
            w[i] = floors[i]
    return [FRAME_W * x / sum(w) for x in w]


def table_block(src: list[str]) -> Table:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in src]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    ncol = max(len(c) for c in cells)
    cells = [c + [""] * (ncol - len(c)) for c in cells]
    data = [[Paragraph(inline(x), S["th"] if r == 0 else S["td"])
             for x in row] for r, row in enumerate(cells)]
    # Column widths are proportional to content length, but each column also
    # gets a FLOOR wide enough for its widest unbreakable word. Without that
    # floor a narrow column mid-word-wraps: the provenance tables rendered
    # "SOURCED" as "SOURC ED" and "DERIVED" as "DERIVE D", which reads as two
    # different tags. The floor uses MEASURED font metrics via stringWidth, not
    # a characters-times-constant estimate: at 6.8 pt Helvetica "SOURCED" is
    # 34.00 pt while 4.0 pt per character predicts 28.0 pt, and that 20 percent
    # shortfall is exactly why the first attempt at this floor did not bind.
    def word_w(s: str) -> float:
        return stringWidth(s, "Helvetica-Bold", S["th"].fontSize)

    lens = [max(len(row[k]) for row in cells) for k in range(ncol)]
    floors = [min(FRAME_W * 0.32,
                  max((word_w(w_) for row in cells for w_ in row[k].split()),
                      default=8.0) + 7.0)
              for k in range(ncol)]
    tot = sum(lens) or 1
    w = [max(floors[k], FRAME_W * lens[k] / tot) for k in range(ncol)]
    # The floors must survive normalisation. Scaling all columns by
    # FRAME_W/sum(w) shrank floored columns back below their floor, which is
    # why the Tag column still wrapped at 38.3 pt against a 34.0 pt word plus
    # 6 pt padding. Pin the floored columns and distribute the remaining width
    # across the rest, in proportion to their content length.
    w = _normalise(w, floors)
    t = Table(data, colWidths=w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f7")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.9, NAVY),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, colors.HexColor("#dde3ea")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafbfd")]),
    ]))
    return t


def build_flow(md: str) -> list:
    lines = md.split("\n")
    flow: list = []
    i = 0
    prev = -1
    while i < len(lines):
        # Cursor-advance guard. Several branches below end in `continue` after
        # setting i themselves, so the check lives at the TOP of the loop: if
        # any branch failed to consume input, the next iteration raises here
        # naming the offending line, instead of spinning forever. This is the
        # structural fix for the defect described in the module docstring, and
        # it covers every branch rather than the one that happened to fail.
        assert i > prev, (
            f"markdown parser did not advance at line {i + 1}: "
            f"{lines[i][:70]!r}")
        prev = i
        start = i
        ln = lines[i]
        if ln.startswith("!["):
            m = re.search(r"\((figures/[^)]+)\)", ln)
            p = FIGDIR / Path(m.group(1)).name
            iw, ih = PILImage.open(p).size
            # Figures are scaled below full frame width: at 100 percent the five
            # of them push the memo to 8 pages against a 6 page requirement.
            # FIG_SCALE is the one knob that trades figure size for page count,
            # and it is applied to the width with the aspect ratio preserved.
            w = FRAME_W * FIG_SCALE
            h = w * ih / iw
            j = i + 1
            cap = ""
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**Figure"):
                cb = []
                while j < len(lines) and lines[j].strip():
                    cb.append(lines[j])
                    j += 1
                cap = " ".join(cb)
                i = j
            flow.append(KeepTogether([Image(str(p), width=w, height=h),
                                      Paragraph(inline(cap), S["cap"])]))
            i += 1
            continue
        if ln.startswith("# "):
            flow += [Paragraph(inline(ln[2:]), S["h1"]),
                     HRFlowable(width="100%", thickness=1.4, color=NAVY,
                                spaceBefore=1, spaceAfter=5)]
        elif ln.startswith("## "):
            flow += [Paragraph(inline(ln[3:]), S["h2"]),
                     HRFlowable(width="100%", thickness=0.4, color=RULE,
                                spaceBefore=0, spaceAfter=3.5)]
        elif ln.startswith("### "):
            flow.append(Paragraph(inline(ln[4:]), S["h3"]))
        elif ln.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            txt = "<br/>".join(
                x.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace(" ", "&nbsp;") for x in buf)
            flow.append(Paragraph(txt, S["code"]))
            i = j + 1
            continue
        elif ln.startswith("|"):
            j = i
            buf = []
            while j < len(lines) and lines[j].startswith("|"):
                buf.append(lines[j])
                j += 1
            flow += [table_block(buf), Spacer(1, 3)]
            i = j
            continue
        elif re.match(r"^\d+\. |^- ", ln):
            j = i
            items: list[str] = []
            while j < len(lines) and (re.match(r"^\d+\. |^- ", lines[j])
                                      or (lines[j].startswith("   ")
                                          and lines[j].strip())):
                if re.match(r"^\d+\. |^- ", lines[j]):
                    items.append(lines[j])
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            for it in items:
                mk = re.match(r"^(\d+)\. (.*)$", it)
                txt = (f"<b>{mk.group(1)}.</b>&nbsp;&nbsp;{inline(mk.group(2))}"
                       if mk else "&bull;&nbsp;&nbsp;" + inline(it[2:]))
                flow.append(Paragraph(txt, S["li"]))
            flow.append(Spacer(1, 1.5))
            i = j
            continue
        elif ln.startswith("---"):
            flow.append(HRFlowable(width="100%", thickness=0.4,
                                   color=colors.HexColor("#cccccc"),
                                   spaceBefore=4, spaceAfter=4))
        elif ln.strip():
            # A paragraph runs until a blank line or a line that STARTS a new
            # block. A bare backtick is NOT such a line: it opens an inline
            # code span, and treating it as a terminator is what made this
            # loop fail to advance. Only a triple backtick fence stops a
            # paragraph. The first line is always consumed, so j > i always.
            j = i + 1
            buf = [lines[i]]
            while (j < len(lines) and lines[j].strip()
                   and not lines[j].startswith(("#", "|", "!", "```", "- ",
                                                "---"))
                   and not re.match(r"^\d+\. ", lines[j])):
                buf.append(lines[j])
                j += 1
            flow.append(Paragraph(inline(" ".join(buf)), S["p"]))
            i = j
            assert i > start, f"paragraph branch did not advance at line {start+1}"
            continue
        i += 1
    return flow


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 6.4)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN_X, 7 * mm,
                      "Applied Elements: Vikarabad decision memo. All figures "
                      "computed in this repository; see docs/memo_numbers.csv.")
    canvas.drawRightString(A4[0] - MARGIN_X, 7 * mm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    flow = build_flow(MEMO.read_text())
    doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN_X,
                          rightMargin=MARGIN_X, topMargin=MARGIN_TOP,
                          bottomMargin=MARGIN_BOT, title="Vikarabad decision memo",
                          author="Applied Elements")
    frame = Frame(MARGIN_X, MARGIN_BOT, FRAME_W,
                  A4[1] - MARGIN_TOP - MARGIN_BOT, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
    # reportlab CONSUMES the story list during build, so len(flow) afterwards is
    # 0. A first version printed that as the flowable count, which was a false
    # zero next to a 1.1 MB output file. Count before the call.
    n_flow = len(flow)
    doc.build(flow)
    n_pages = doc.page
    print(f"[pdf] {n_flow} flowables -> {n_pages} pages -> {OUT} "
          f"({OUT.stat().st_size:,} bytes)")
    if not 5 <= n_pages <= 8:
        print(f"[warn] the brief asks for a 6 page memo; this build is "
              f"{n_pages} pages")


if __name__ == "__main__":
    main()
