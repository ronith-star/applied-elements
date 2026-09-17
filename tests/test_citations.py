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

#: A publisher-year inside a FULL inline book citation, recognised by the shape
#: of the citation TAIL.
#:
#: Two earlier versions were wrong, in the same way each time: each asserted a
#: safety property it did not have.
#:
#: Version 1 was a name list (Heinemann, Press, Hall, Wiley, Cambridge, Taylor,
#: Francis, ...) commented "a genuinely missing citation is never suppressed".
#: Those are all real surnames, so a genuine "Taylor 2019" was silently
#: ignored. It also miscounted the defect it was written for: SIX
#: publisher-year hits across FOUR modules (capex.py: Heinemann 2021,
#: Hill 2003; capacity.py: Press 1988; streams.py: Hall 2012, Heinemann 2015;
#: yield_cascade.py: Wiley 2012), not four in three.
#:
#: Version 2 matched any capitalised token plus year within 70 characters after
#: an italicised title, commented "carries no such risk" and "an author-year
#: citation in running prose has no italic title immediately before it". Also
#: false. Measured counterexamples, all silently suppressed:
#:     "The approach in *Wills' Mineral Processing Technology* was later
#:      refined by Taylor 2019 for quartz."          -> Taylor 2019 suppressed
#:     "Following *Introduction to TPM*, the kinetics follow Bond 1952
#:      closely."                                    -> Bond 1952 suppressed
#:     "See *Some Book Title*, chapter 3, and also Saltelli 2002 for the
#:      method."                                     -> Saltelli 2002 suppressed
#: Proximity to a title is not the signal, because prose routinely names a book
#: and then cites a paper in the same sentence.
#:
#: The actual discriminator is that a publisher-year terminates a
#: COMMA-DELIMITED citation tail, with no verb or connective between the title
#: and the publisher, only edition and series noise:
#:     *Title*, 8th ed., Prentice Hall 2012
#:     *Title*, 7th ed., Wiley 2012, chapter 6
#:     *Title*, Productivity Press 1988
#: whereas the counterexamples all interpose running prose ("was later refined
#: by", "the kinetics follow", "chapter 3, and also"). Requiring the gap to
#: contain no lower-case word other than a short edition vocabulary separates
#: them. THIS IS STILL A HEURISTIC on prose, not a proof: it is verified by the
#: tests below against the six real cases in this repository, the three
#: counterexamples above, and a bare-surname probe. A new citation style may
#: need a new case, which is why the tests enumerate rather than assert a
#: general property.
#: Tokens allowed between the title and the publisher: edition and volume
#: wording only, no verbs or connectives. "8th ed." is TWO tokens, so the group
#: repeats over whitespace as well as commas; a first attempt allowed them only
#: comma-separated and matched just 1 of the 6 real cases.
_EDITION_NOISE = (
    r"(?:\d+(?:st|nd|rd|th)|ed\.?|edn\.?|eds\.?|vol\.?|rev\.?|"
    r"reprint|series|no\.?|\d+)"
)

_BOOK_CITATION_TAIL = re.compile(
    r"\*[^*]{4,140}\*"                          # italicised title
    r"(?:[\s,]*" + _EDITION_NOISE + r")*"        # edition noise only
    r"[\s,]*"
    r"\b(?P<pub>[A-Z][A-Za-z.&'\u2032-]*(?:\s+[A-Z][A-Za-z.&'\u2032-]*){0,2})\s+"
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
            r"not cited|could not be verified|unverifiable", flat, re.IGNORECASE)
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


#: The SIX real publisher-year citations in this repository, with the module
#: each came from. Enumerated rather than counted, after a docstring claimed
#: "the six real cases" while listing four.
_REAL_PUBLISHER_CITATIONS = [
    ("ae/econ/capex.py",
     "Peters and Timmerhaus, *Plant Design and Economics for Chemical "
     "Engineers*, 5th ed., McGraw-Hill 2003, chapters 6 and 12.",
     ("Hill", "2003")),
    ("ae/econ/capex.py",
     "Towler and Sinnott, *Chemical Engineering Design*, 3rd ed., "
     "Butterworth-Heinemann 2021, chapter 6.",
     ("Heinemann", "2021")),
    ("ae/plant/capacity.py",
     "Nakajima, *Introduction to TPM*, Productivity Press 1988.",
     ("Press", "1988")),
    ("ae/plant/streams.py",
     "Himmelblau and Riggs, *Basic Principles and Calculations in Chemical "
     "Engineering*, 8th ed., Prentice Hall 2012, chapter on recycle.",
     ("Hall", "2012")),
    ("ae/plant/streams.py",
     "Wills and Finch, *Wills' Mineral Processing Technology*, 8th ed., "
     "Butterworth-Heinemann 2015, chapter 3.",
     ("Heinemann", "2015")),
    ("ae/plant/yield_cascade.py",
     "Montgomery, *Introduction to Statistical Quality Control*, 7th ed., "
     "Wiley 2012, chapter 6.",
     ("Wiley", "2012")),
]

#: Genuine author-year citations that a proximity-based rule wrongly suppressed.
#: Each names a book in the SAME SENTENCE as the citation, which is why
#: "no italic title immediately before it" was not a safe discriminator.
_MUST_NOT_SUPPRESS = [
    "The approach in *Wills' Mineral Processing Technology* was later refined "
    "by Taylor 2019 for quartz.",
    "Following *Introduction to TPM*, the kinetics follow Bond 1952 closely.",
    "See *Some Book Title*, chapter 3, and also Saltelli 2002 for the method.",
    "Compare *Plant Design and Economics*, where Heinemann 2021 is discussed "
    "at length.",
]


@pytest.mark.parametrize("module,text,expected", _REAL_PUBLISHER_CITATIONS,
                         ids=[f"{m.split('/')[-1]}:{e[0]}"
                              for m, _, e in _REAL_PUBLISHER_CITATIONS])
def test_publisher_year_in_a_book_citation_is_not_read_as_an_author(
        module, text, expected):
    """All SIX real cases, one per parametrised run so a miss names itself."""
    flat = re.sub(r"\s+", " ", text)
    assert expected in _publisher_years(flat), (
        f"{module}: {expected} must be recognised as publisher-year"
    )
    assert not _citations(text), (
        f"{module}: publisher-year must not surface as an unresolved citation"
    )


def test_all_six_real_cases_are_enumerated():
    """Guards the count claim itself: 6 citations across 4 modules."""
    assert len(_REAL_PUBLISHER_CITATIONS) == 6
    assert len({m for m, _, _ in _REAL_PUBLISHER_CITATIONS}) == 4


@pytest.mark.parametrize("text", _MUST_NOT_SUPPRESS)
def test_a_genuine_citation_near_a_book_title_is_not_suppressed(text):
    """The counterexamples that broke the proximity rule.

    A previous version matched any capitalised token plus year within 70
    characters after an italicised title, and silently suppressed all of these.
    Prose names a book and cites a paper in the same sentence routinely, so the
    discriminator is the comma-delimited citation TAIL (no verbs or connectives
    between title and publisher), not proximity.
    """
    flat = re.sub(r"\s+", " ", text)
    assert _publisher_years(flat) == set(), (
        f"genuine author-year wrongly suppressed as a publisher: {text[:60]}"
    )
    assert _citations(text), "the citation must still be detected"


def test_a_real_author_sharing_a_publisher_surname_is_detected():
    """The failure mode of the original name list."""
    for surname in ("Press", "Hall", "Cambridge", "Taylor", "Wiley", "Hill"):
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
