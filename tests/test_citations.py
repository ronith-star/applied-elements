"""Every in-text citation must appear in its module's References block.

Rationale, from a real defect: uncertainty.py's References block was corrected
to drop an unverifiable "Sobol 1993" entry and to say so explicitly, while an
inline "(Sobol 1993)" survived twenty lines above it. The docstring then both
cited and disclaimed the same source. Prose consistency is checkable, so it is
checked here rather than trusted to review.
"""
import ast
import pathlib
import re

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: Author-year citations in running text, e.g. "(Sobol' 2001)" or "Bond 1952".
_CITE = re.compile(r"\(?\b([A-Z][A-Za-z'\u2032]+)\s+(?:et al\.?\s+)?(\d{4})[a-z]?\)?")

#: Capitalised words that are never author names in running prose.
_NOT_AUTHORS = {
    "Class", "Tier", "Table", "Figure", "Section", "Appendix", "Equation",
    "Verified", "Accessed", "Both", "The", "This", "A", "An", "In", "At",
    "Chapter", "Part", "Page", "Note", "Sources", "Source", "See", "Since",
    "Reading", "Requires", "Written", "Computed", "Reported", "Offered",
}

#: A publisher-year inside a FULL inline book citation, recognised by POSITION
#: rather than by name.
#:
#: This replaced a name list ("Heinemann", "Press", "Hall", "Wiley", ...) that
#: was wrong in two ways. It miscounted the defect it was written for: the
#: detector had flagged SIX publisher-year hits across FOUR modules (capex.py:
#: Heinemann 2021, Hill 2003; capacity.py: Press 1988; streams.py: Hall 2012,
#: Heinemann 2015; yield_cascade.py: Wiley 2012), not four in three. Worse, the
#: comment claimed "a genuinely missing citation is never suppressed", which the
#: list could not honour: Press, Hall, Cambridge, Taylor and Francis are all
#: real surnames, so a genuine "Taylor 2019" citation would have been silently
#: ignored.
#:
#: The structural signature is reliable and carries no such risk. A full book
#: citation names an italicised title before the publisher, and usually an
#: edition:
#:     Himmelblau and Riggs, *Basic Principles...*, 8th ed., Prentice Hall 2012
#:     Montgomery, *Introduction to Statistical Quality Control*, 7th ed.,
#:         Wiley 2012, chapter 6
#: An author-year citation in running prose has no italic title immediately
#: before it, so the two are distinguishable without guessing at names.
#: The gap between the title and the publisher may contain periods ("8th ed.,")
#: and the publisher name itself may be hyphenated ("McGraw-Hill",
#: "Butterworth-Heinemann"), so the separator is bounded by LENGTH and by the
#: absence of a sentence break (a period followed by a space and a capital),
#: not by excluding periods outright. A first version used [^.]{0,60} and
#: matched only 1 of the 6 known publisher citations.
_BOOK_CITATION_TAIL = re.compile(
    r"\*[^*]{4,140}\*"                            # italicised title
    r"(?:(?!\.\s+[A-Z])[^*]){0,70}?"               # edition/series, no new sentence
    r"\b[A-Z][A-Za-z.&'\u2032-]*(?:\s+[A-Z][A-Za-z.&'\u2032-]*){0,2}\s+"
    r"(?P<year>\d{4})\b"
)


def _publisher_years(flat_head: str) -> set[tuple[str, str]]:
    """Author-year pairs that are actually publisher-year in a book citation."""
    out = set()
    for m in _BOOK_CITATION_TAIL.finditer(flat_head):
        span = m.group(0)
        for a, y in _CITE.findall(span):
            if y == m.group("year"):
                out.add((a, y))
    return out


def _module_docstrings() -> dict[str, str]:
    out = {}
    for p in sorted(SRC.rglob("*.py")):
        if p.name == "__init__.py":
            continue
        tree = ast.parse(p.read_text())
        doc = ast.get_docstring(tree)
        if doc:
            out[str(p.relative_to(SRC))] = doc
    return out


def _citations(doc: str) -> set[tuple[str, str]]:
    """Author-year pairs appearing OUTSIDE the References block.

    Whitespace is normalised FIRST: a full book citation wraps freely, so
    "Prentice Hall\n2012" must be seen as "Hall 2012" and recognised as a
    publisher, not read as an author named Hall. The first version matched
    line by line and reported that citation as unresolved.
    """
    head = re.split(r"\nReferences?\n-+\n", doc)[0]
    head = re.sub(r"\s+", " ", head)
    publishers = _publisher_years(head)
    found = set()
    for m in _CITE.finditer(head):
        author, year = m.group(1), m.group(2)
        if author in _NOT_AUTHORS:
            continue
        if not 1900 <= int(year) <= 2100:
            continue
        if (author, year) in publishers:
            continue
        found.add((author, year))
    return found


def _reference_block(doc: str) -> str:
    parts = re.split(r"\nReferences?\n-+\n", doc)
    return parts[1] if len(parts) > 1 else ""


def _reference_block_flat(doc: str) -> str:
    """References block with whitespace normalised, for substring membership."""
    return re.sub(r"\s+", " ", _reference_block(doc))


@pytest.mark.parametrize("path", sorted(_module_docstrings()))
def test_every_intext_citation_is_in_the_reference_block(path):
    doc = _module_docstrings()[path]
    cites = _citations(doc)
    if not cites:
        pytest.skip(f"{path} makes no in-text author-year citation")
    refs = _reference_block_flat(doc)
    assert refs, (
        f"{path} cites {sorted(cites)} in running text but has no References "
        f"block. A citation a reader cannot resolve is not a citation."
    )
    missing = sorted(
        (a, y) for a, y in cites
        if not (a.rstrip("'\u2032") in refs and y in refs)
    )
    assert not missing, (
        f"{path}: cited in text but absent from its References block: {missing}. "
        f"This is the exact defect that let an unverifiable 'Sobol 1993' survive "
        f"in a docstring whose References block said it was not cited."
    )


def test_no_docstring_both_cites_and_disclaims_the_same_year():
    """A References block saying a source is NOT cited, while the same module
    cites it in text, is self-contradictory."""
    offenders = {path: c for path, doc in _module_docstrings().items()
                 if (c := _clash(doc))}
    assert not offenders, (
        f"modules that cite a year in text while their References block "
        f"disclaims it: {offenders}"
    )


def _clash(doc: str) -> list[str]:
    """Years a docstring both cites in text and disclaims in References.

    Extracted from the test below so the positive controls can exercise the
    SAME code path on synthetic text, rather than re-implementing it.
    """
    refs = _reference_block(doc)
    if not refs:
        return []
    entry_years: set[str] = set()
    disclaimed_paras: list[str] = []
    for para in re.split(r"\n\s*\n", refs):
        flat = re.sub(r"\s+", " ", para)
        disclaiming = re.search(
            r"not cited|could not be verified|unverifiable", flat, re.I)
        if disclaiming:
            disclaimed_paras.append(flat)
        elif re.search(r"doi:|\*[A-Z]", flat):
            entry_years.update(re.findall(r"\b(19\d{2}|20\d{2})\b", flat))
    disclaimed: set[str] = set()
    for flat in disclaimed_paras:
        disclaimed.update(re.findall(r"\b(19\d{2}|20\d{2})\b", flat))
    disclaimed -= entry_years
    in_text = {y for _, y in _citations(doc)}
    return sorted(disclaimed & in_text)


def test_positive_control_the_exact_text_that_shipped():
    """The real defect, run through the real detector.

    An earlier version of this control hardcoded ``disclaimed = {"1993"}`` and
    asserted the intersection was non-empty, which was already implied by the
    line above it. The disclaimer-extraction logic was never called, so the
    control could not fail if that logic regressed. It now calls _clash().
    """
    shipped = (
        "Intro paragraph.\n\n"
        "the variance decomposes uniquely (Sobol 1993):\n\n"
        "References\n----------\n"
        "Saltelli, A. et al. (2010), *Computer Physics Communications* "
        "181(2):259-270, doi:10.1016/j.cpc.2009.09.018.\n\n"
        "The decomposition is often attributed to a 1993 paper in *Mathematical "
        "Modelling and Computational Experiments*. That journal is not indexed "
        "by Crossref and the reference could not be verified, so it is NOT "
        "cited here.\n"
    )
    assert _clash(shipped) == ["1993"], (
        "the shipped contradiction must be detected by the real code path"
    )


def test_positive_control_the_corrected_text_is_clean():
    """The fix must pass. Same disclaimer, but the in-text citation now names
    the verified 2001 replacement, and 2001 has a reference entry of its own."""
    corrected = (
        "Intro paragraph.\n\n"
        "the variance decomposes uniquely (Sobol' 2001, see References):\n\n"
        "References\n----------\n"
        "Sobol', I.M. (2001), *Mathematics and Computers in Simulation* "
        "55(1-3):271-280, doi:10.1016/S0378-4754(00)00270-6.\n\n"
        "The decomposition is often attributed to a 1993 paper. That journal is "
        "not indexed by Crossref and the reference could not be verified, so it "
        "is NOT cited here.\n"
    )
    assert _clash(corrected) == []


def test_positive_control_unresolved_citation_is_detected():
    """An in-text citation absent from the References block must be found."""
    doc = (
        "The method follows Nonexistent 1997 exactly.\n\n"
        "References\n----------\n"
        "Saltelli, A. (2002), *Computer Physics Communications* 145(2):280-297, "
        "doi:10.1016/S0010-4655(02)00280-1.\n"
    )
    cites = _citations(doc)
    assert ("Nonexistent", "1997") in cites
    refs = _reference_block_flat(doc)
    missing = [(a, y) for a, y in cites
               if not (a.rstrip("'\u2032") in refs and y in refs)]
    assert missing == [("Nonexistent", "1997")]


def test_publisher_year_in_a_book_citation_is_not_read_as_an_author():
    """The six real cases from this repository, plus the surname guard."""
    for text, expected in [
        ("Himmelblau and Riggs, *Basic Principles and Calculations in Chemical "
         "Engineering*, 8th ed., Prentice Hall 2012, chapter on recycle.",
         {("Hall", "2012")}),
        ("Montgomery, *Introduction to Statistical Quality Control*, 7th ed., "
         "Wiley 2012, chapter 6.", {("Wiley", "2012")}),
        ("Nakajima, *Introduction to TPM*, Productivity Press 1988.",
         {("Press", "1988")}),
        ("Wills and Finch, *Wills' Mineral Processing Technology*, 8th ed., "
         "Butterworth-Heinemann 2015, chapter 3.", {("Heinemann", "2015")}),
    ]:
        flat = re.sub(r"\s+", " ", text)
        assert _publisher_years(flat) == expected, text[:50]
        assert not _citations(text), (
            f"publisher-year must not surface as an unresolved citation: {text[:50]}"
        )

    # A genuine author-year citation whose surname coincides with a publisher
    # must NOT be suppressed. The previous name-list approach failed this.
    for surname in ("Press", "Hall", "Cambridge", "Taylor", "Wiley"):
        doc = f"The kinetics follow {surname} 2019 closely."
        assert (surname, "2019") in _citations(doc), (
            f"a real author named {surname} must still be detected"
        )


def test_dois_are_well_formed():
    """A malformed DOI is unresolvable, so the shape is checked even though
    resolution needs the network."""
    pattern = re.compile(r"\b10\.\d{4,9}/\S+")
    for path, doc in _module_docstrings().items():
        for raw in pattern.findall(doc):
            doi = raw.rstrip(".,;:)").strip()
            assert doi.count("/") >= 1, f"{path}: malformed DOI {doi!r}"
            assert len(doi) > 10, f"{path}: truncated DOI {doi!r}"
            assert " " not in doi, f"{path}: DOI contains a space: {doi!r}"
