"""Convert the RST-flavoured docstrings in src/ae into LaTeX for the model book.

Scope and intent. This is NOT a general RST implementation. It handles exactly
the constructs the platform's docstrings actually use, which were counted
before the converter was written (399 docstrings, 295,996 characters):

    :math: role               532     .. math:: directive       119
    :func:/:mod:/:class: role 210     simple (=====) tables      28
    section underlines        281     bullet lists              122
    numbered lists             99     doctest statements        154
    ``literal``               918     grid tables                 0
    .. code-block::             0     .. note::                   0

Those 154 doctest statements are distributed over 57 doctest BLOCKS (one block
per docstring that contains any). Both counts are measured, and they are not
interchangeable: the block count is the number of worked examples the book
reproduces, the statement count is the number of individual ``>>>`` lines
executed to reproduce them.

Constructs not on that list are passed through as escaped text rather than
silently dropped, and :func:`unhandled_constructs` reports leftovers, so a
docstring that grows a new construct is a visible failure rather than a
quietly mangled paragraph.

Why hand-rolled rather than docutils or pandoc. The docstrings mix RST with
free-form ASCII arithmetic blocks ("1/T = 1/373.15 = 2.6798874e-3 K^-1") that
are indented like literal blocks but are not marked as such. docutils warns and
reformats those, and pandoc's RST reader collapses the column alignment that
makes them readable. Both would need the same per-construct handling to produce
a faithful page, with less control over the result.
"""
from __future__ import annotations

import re

#: LaTeX special characters, escaped outside math and verbatim contexts.
#: Backslash must be first: escaping it after the others would double-escape
#: the backslashes they introduce.
_ESCAPES: list[tuple[str, str]] = [
    ("\\", "\\textbackslash{}"),
    ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
    ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
    ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
]


def escape(text: str) -> str:
    """Escape LaTeX specials in running prose."""
    for a, b in _ESCAPES:
        text = text.replace(a, b)
    return text


def dash_guard(text: str) -> str:
    """Remove en and em dashes, including the LaTeX ligatures that produce them.

    The platform brief forbids en and em dashes in output. The docstrings are
    already clean of the Unicode characters, but this runs regardless so that
    one introduced upstream cannot reach the typeset page. The ASCII ligatures
    matter just as much: a literal ``--`` in LaTeX source renders as an en dash
    and ``---`` as an em dash.

    The ligatures are broken with an empty group rather than collapsed to a
    single hyphen. An earlier version of this function collapsed them, which
    obeyed the dash rule but corrupted the text: every occurrence of a long
    command-line flag in the corpus, such as ``mypy --strict``, was printed as
    ``mypy -strict``, a flag that does not exist. Inserting ``{}`` between the
    hyphens suppresses the ligature while keeping both characters, so the
    reader sees the real flag and no dash is rendered.

    The ligature is real in typewriter as well as roman, which is why this
    applies everywhere and not only to prose. MEASURED with \\the\\wd on this
    TeX installation: ``\\texttt{-}`` is 5.24998pt and ``\\texttt{--}`` is also
    5.24998pt, so the pair collapses to one glyph, while ``\\texttt{-{}-}`` is
    10.49997pt, exactly two hyphens. In roman a hyphen is 3.33333pt and ``--``
    is 5.0pt, the en dash.

    Verbatim environments are exempt and must not be passed through here: they
    print their bytes literally with no ligature applied, so a doctest line
    containing ``--`` is already correct and inserting ``{}`` would put braces
    into code the reader is meant to copy.
    """
    text = text.replace("\u2014", ", ").replace("\u2013", " to ")
    text = text.replace("---", "-{}-{}-")
    text = text.replace("--", "-{}-")
    return text


def inline(text: str) -> str:
    """Convert inline RST roles and literals to LaTeX, escaping the rest.

    Order is load-bearing. Math and literal spans are lifted out to
    placeholders FIRST so that escaping cannot corrupt their contents: a
    ``\\Delta`` inside a :math: role must survive verbatim, while a stray
    backslash in prose must become ``\\textbackslash{}``.

    Stashed spans are dash-guarded too, with one exception. A stash bypasses
    the prose ``dash_guard`` call by construction, and an earlier version
    exploited that only accidentally: it meant an RST literal such as
    ```` ``mypy --strict`` ```` reached the page as a rendered en dash, because
    \\texttt applies the ligature just as roman does. Four such occurrences
    were in the built book. Math spans are exempt: inside ``$...$`` a ``-`` is
    a binary minus and no ligature is formed, and rewriting it would corrupt
    the expression.
    """
    # An inline role may be broken across a source line boundary, because the
    # docstrings wrap at 88 columns without regard to role boundaries, and the
    # single-line pattern below cannot match such a span. Joining the role's
    # own newlines into spaces before matching handles that case; LaTeX does
    # not care about the line break inside $...$, and RST treats the wrapped
    # role as one span too.
    #
    # This is a correctness fix for wrapped roles on its own terms, NOT the
    # cause of the 12 literal ":math:`..." strings that appeared on the built
    # page. It was written believing it was that cause, and the rebuild
    # immediately after refuted that: the leak count was 12 before the change
    # and 12 after it. The actual cause was the indented-block classifier in
    # convert(), documented at its own site.
    text = re.sub(r"(:(?:math|func|mod|class|meth|data|attr|obj|exc|ref):`)"
                  r"([^`]*)(`)",
                  lambda m: m.group(1) + re.sub(r"\s*\n\s*", " ", m.group(2))
                  + m.group(3), text)

    spans: list[str] = []

    def stash(rendered: str, guard: bool = True) -> str:
        spans.append(dash_guard(rendered) if guard else rendered)
        return "\x00%d\x00" % (len(spans) - 1)

    text = re.sub(r":math:`([^`]*)`",
                  lambda m: stash("$" + m.group(1) + "$", guard=False), text)
    text = re.sub(
        r":(?:func|mod|class|meth|data|attr|obj|exc|ref):`~?([^`]*)`",
        lambda m: stash("\\texttt{" + escape(m.group(1)) + "}"),
        text,
    )
    text = re.sub(r"``([^`]*)``",
                  lambda m: stash("\\texttt{" + escape(m.group(1)) + "}"), text)
    text = re.sub(r"(?<!`)`([^`]+)`_?(?!`)",
                  lambda m: stash("\\textit{" + escape(m.group(1)) + "}"), text)

    # ASCII unit exponents. The docstrings write units in running prose as
    # "mol m^-2 s^-1" and "J mol^-1 K^-1". Escaping the caret gives
    # "m\textasciicircum{}-2", which is correct but unreadable; these become
    # real superscripts. Only a letter or a closing paren followed by a caret
    # and an optionally signed integer is matched, so a caret used any other
    # way still escapes normally.
    text = re.sub(
        r"(?<=[A-Za-z\)])\^([+-]?\d+)",
        lambda m: stash("$^{" + m.group(1) + "}$"),
        text,
    )

    text = dash_guard(escape(text))
    for i, rendered in enumerate(spans):
        text = text.replace("\x00%d\x00" % i, rendered)
    return text


_UNDERLINE = re.compile(r"^([=\-~^\"'+*#`]){3,}\s*$")
_SIMPLE_TABLE_RULE = re.compile(r"^\s*=+(\s+=+)+\s*$")


def _is_underline(line: str, prev: str) -> bool:
    """True if ``line`` underlines ``prev`` as an RST section heading."""
    if not _UNDERLINE.match(line):
        return False
    if not prev.strip():
        return False
    # An underline must be at least as long as the text it underlines, less a
    # tolerance, and the text must not itself be a rule (which would make this
    # the bottom rule of a simple table).
    return len(line.rstrip()) >= len(prev.rstrip()) - 2 and not _SIMPLE_TABLE_RULE.match(prev)


def _table(rows: list[str], rule: str) -> str:
    """Render an RST simple table from its column-rule geometry.

    RST simple tables define columns by the position of the ``=`` runs in the
    rule line, not by a delimiter, so the rule is parsed for column spans and
    every body line is sliced at those offsets. The last column is taken to the
    end of the line, because RST permits the final cell to overrun the rule.

    A non-final cell that overruns its rule is an error in the table, not
    something to render as best it can. Silently truncating such a cell is how
    ``:math:`\\mathcal{R}``` in physics/leaching.py reached the built page as
    the mangled fragment ``:math:`\\textbackslash{}mathcal{R``: the rule gave
    that column 17 characters and the cell needed 19. This now raises, so a
    misaligned table stops the build instead of printing a broken symbol at a
    reader. Trailing whitespace is not an overrun, only non-space content in
    the gutter between one column's rule and the next column's start.
    """
    spans: list[tuple[int, int]] = []
    for m in re.finditer(r"=+", rule):
        spans.append((m.start(), m.end()))
    if not spans:
        return ""
    cells_rows: list[list[str]] = []
    for row in rows:
        if _SIMPLE_TABLE_RULE.match(row):
            continue
        for i, (_a, b) in enumerate(spans[:-1]):
            gutter = row[b:spans[i + 1][0]]
            if gutter.strip():
                raise ValueError(
                    f"RST simple table is misaligned: column {i + 1} of the row "
                    f"{row!r} overruns its rule, which ends at character {b}, "
                    f"and spills {gutter.strip()!r} into the gutter before "
                    f"column {i + 2} starts at {spans[i + 1][0]}. Widen the "
                    "'=' rule for that column in the docstring so the rule is "
                    "at least as wide as its widest cell.")
        cells = []
        for i, (a, b) in enumerate(spans):
            seg = row[a:] if i == len(spans) - 1 else row[a:b]
            cells.append(seg.strip())
        cells_rows.append(cells)
    if not cells_rows:
        return ""
    ncol = len(spans)
    head, body = cells_rows[0], cells_rows[1:]
    colspec = "@{}l" + "l" * (ncol - 2) + "p{0.42\\textwidth}@{}" if ncol >= 2 else "@{}l@{}"
    out = ["\\begin{center}", "\\footnotesize",
           "\\begin{tabular}{" + colspec + "}", "\\toprule"]
    out.append(" & ".join(inline(c) for c in head) + " \\\\")
    out.append("\\midrule")
    for r in body:
        out.append(" & ".join(inline(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(out)


def convert(doc: str, base_level: int = 0) -> str:
    """Convert one docstring to LaTeX.

    Parameters
    ----------
    doc
        The docstring text, already dedented by :func:`inspect.cleandoc` or
        :func:`ast.get_docstring`.
    base_level
        0 renders section headings as ``\\subsection``, 1 as
        ``\\subsubsection``, 2 as ``\\paragraph``. Docstring headings are
        nested under the chapter the module occupies, so they can never be
        allowed to open a chapter or section of their own.

    Returns
    -------
    str
        LaTeX body text, with no preamble and no chapter heading.
    """
    levels = ["subsection", "subsubsection", "paragraph"]
    lines = doc.expandtabs(4).split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    list_env: str | None = None

    def close_list() -> None:
        nonlocal list_env
        if list_env:
            out.append("\\end{%s}" % list_env)
            list_env = None

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Section heading: text followed by an underline rule.
        if i + 1 < n and _is_underline(lines[i + 1], line):
            close_list()
            lvl = levels[min(base_level, len(levels) - 1)]
            out.append("\n\\%s*{%s}\n" % (lvl, inline(stripped)))
            i += 2
            continue

        # Simple table: a rule line, then rows, then a rule, optionally a
        # header rule before the body.
        if _SIMPLE_TABLE_RULE.match(line):
            close_list()
            rule = line
            j = i + 1
            body: list[str] = []
            nrules = 0
            while j < n:
                if _SIMPLE_TABLE_RULE.match(lines[j]):
                    nrules += 1
                    if nrules >= 2:
                        break
                body.append(lines[j])
                j += 1
            out.append(_table([b for b in body if b.strip()], rule))
            i = j + 1
            continue

        # .. math:: directive, with its continuation lines.
        m = re.match(r"^(\s*)\.\.\s+math::\s*(.*)$", line)
        if m:
            close_list()
            indent = len(m.group(1))
            parts = [m.group(2).strip()] if m.group(2).strip() else []
            j = i + 1
            while j < n:
                if not lines[j].strip():
                    # A blank line ends the directive unless more indented
                    # content follows.
                    k = j + 1
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and (len(lines[k]) - len(lines[k].lstrip())) > indent:
                        parts.append("\\\\")
                        j = k
                        continue
                    break
                if (len(lines[j]) - len(lines[j].lstrip())) <= indent:
                    break
                parts.append(lines[j].strip())
                j += 1
            expr = " ".join(p for p in parts if p)
            expr = expr.replace("\\boxed{", "\\boxed{")
            out.append("\\begin{equation*}\n\\begin{gathered}\n%s\n\\end{gathered}\n\\end{equation*}" % expr)
            i = j
            continue

        # Doctest block: consecutive >>> / ... lines plus their output.
        if stripped.startswith(">>>"):
            close_list()
            indent = len(line) - len(line.lstrip())
            block: list[str] = []
            while i < n:
                cur = lines[i]
                if not cur.strip():
                    break
                if (len(cur) - len(cur.lstrip())) < indent:
                    break
                block.append(cur[indent:])
                i += 1
            out.append("\\begin{doctestblock}\n%s\n\\end{doctestblock}" % "\n".join(block))
            continue

        # Bullet list item.
        m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if m and not _UNDERLINE.match(stripped):
            if list_env != "itemize":
                close_list()
                out.append("\\begin{itemize}")
                list_env = "itemize"
            text = [m.group(2)]
            ind = len(m.group(1))
            j = i + 1
            while j < n and lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) > ind \
                    and not re.match(r"^\s*[-*+]\s", lines[j]) and not re.match(r"^\s*\d+\.\s", lines[j]):
                text.append(lines[j].strip())
                j += 1
            out.append("\\item " + inline(" ".join(text)))
            i = j
            continue

        # Enumerated list item.
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            if list_env != "enumerate":
                close_list()
                out.append("\\begin{enumerate}[leftmargin=*,label=\\arabic*.]")
                list_env = "enumerate"
            text = [m.group(3)]
            ind = len(m.group(1))
            j = i + 1
            while j < n and lines[j].strip() and (len(lines[j]) - len(lines[j].lstrip())) > ind \
                    and not re.match(r"^\s*\d+\.\s", lines[j]) and not re.match(r"^\s*[-*+]\s", lines[j]):
                text.append(lines[j].strip())
                j += 1
            out.append("\\item " + inline(" ".join(text)))
            i = j
            continue

        # Blank line: paragraph break, and closes any open list.
        if not stripped:
            close_list()
            out.append("")
            i += 1
            continue

        # An indented run of lines containing arithmetic or an ASCII layout is
        # preserved verbatim. This is what carries the hand-arithmetic blocks
        # in the golden-example docstrings, whose column alignment is the point.
        #
        # An indented line carrying RST markup is NOT such a block: it is
        # indented prose, and the "=" that matched is inside an equation
        # written as a :math: role. Classifying it verbatim is what put literal
        # ":math:`...`" strings on the built page, where a derivation note is
        # indented under its numbered equation.
        #
        # MEASURED both ways rather than asserted. Converting every docstring
        # in the package with and without this markup test: 38 role spans
        # across 9 modules are affected (leaching 11, thermal 10, diffusion 4,
        # streams 4, psd 3, chlorination 2, separation 2, provenance 1,
        # reagents 1), falling to 0 with the test in place. Of those, 12
        # reached the built page, because the book does not typeset every
        # docstring in the package; those 12 were in physics/psd.py and
        # physics/thermal.py. The markup test therefore runs first.
        has_rst_markup = bool(re.search(r":[a-z]+:`|``", line))
        if line.startswith("  ") and ("=" in line or "|" in line) \
                and not has_rst_markup:
            close_list()
            indent = len(line) - len(line.lstrip())
            block = []
            while i < n and lines[i].strip() and (len(lines[i]) - len(lines[i].lstrip())) >= indent:
                block.append(lines[i][min(indent, len(lines[i]) - len(lines[i].lstrip())):])
                i += 1
            out.append("\\begin{verbatimsmall}\n%s\n\\end{verbatimsmall}" % "\n".join(block))
            continue

        # Ordinary prose: gather the paragraph and convert inline markup.
        para = [stripped]
        j = i + 1
        while j < n and lines[j].strip():
            nxt = lines[j]
            if (j + 1 < n and _is_underline(lines[j + 1], nxt)) or _SIMPLE_TABLE_RULE.match(nxt) \
                    or nxt.strip().startswith(">>>") or re.match(r"^\s*\.\.\s+\w+::", nxt) \
                    or re.match(r"^\s*[-*+]\s", nxt) or re.match(r"^\s*\d+\.\s", nxt):
                break
            para.append(nxt.strip())
            j += 1
        text = " ".join(para)
        # RST's "::" literal-block marker becomes a plain colon.
        text = re.sub(r"\s*::\s*$", ":", text)
        out.append(inline(text))
        i = j

    close_list()
    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


_KNOWN = (
    r":math:`|``|\.\.\s+math::|:(?:func|mod|class|meth|data|attr|obj|exc|ref):`"
)


def unhandled_constructs(doc: str) -> list[str]:
    """Report RST directives and roles this converter does not implement.

    Returns the distinct offending tokens. An empty list means every construct
    in ``doc`` is one of the counted, handled forms. This is called over every
    docstring by the build, and a non-empty result aborts it, so the failure
    mode is a build error rather than a silently mangled page.
    """
    found: list[str] = []
    for m in re.finditer(r"^\s*\.\.\s+([a-z\-]+)::", doc, re.M):
        if m.group(1) != "math":
            found.append(".. %s::" % m.group(1))
    for m in re.finditer(r":([a-z]+):`", doc):
        role = m.group(1)
        if role not in {"math", "func", "mod", "class", "meth", "data", "attr", "obj", "exc", "ref"}:
            found.append(":%s:" % role)
    return sorted(set(found))
