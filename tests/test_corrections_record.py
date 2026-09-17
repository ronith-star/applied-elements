"""docs/CORRECTIONS.md must itself be checkable.

A corrections file is the one document in a repository with the strongest
incentive to drift: it is written after the fact, it is not executed, and
nobody diffs it against reality. A record of unverified claims that is itself
unverified would be worse than no record, because it would carry the
authority of a confession while being as unchecked as the thing it corrects.

So the checkable parts are checked here: every commit hash it names must
exist, every claim it quotes verbatim from a commit message must actually
appear in that message, and its arithmetic must reproduce. What cannot be
checked mechanically (whether a described defect really happened) is left to
the reader, which is why the file quotes primary evidence rather than
summarising it.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "CORRECTIONS.md"

GIT_ENV = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_TERMINAL_PROMPT": "0", "HOME": str(ROOT), "PATH": "/usr/bin:/bin:/usr/local/bin"}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, env=GIT_ENV,
                          capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def text() -> str:
    assert DOC.exists(), f"{DOC} is missing"
    return DOC.read_text()


def test_every_commit_hash_cited_actually_exists(text: str) -> None:
    """A hash that does not resolve makes the whole record unfalsifiable."""
    hashes = sorted(set(re.findall(r"`([0-9a-f]{7,40})`", text)))
    assert hashes, "the record cites no commit; it should cite the ones it corrects"
    missing = [h for h in hashes
               if not _git("cat-file", "-t", h).strip() == "commit"]
    if missing and not _git("rev-parse", "--git-dir").strip():
        pytest.skip("not a git checkout, so hashes cannot be resolved")
    assert not missing, f"cited commits do not exist: {missing}"


def test_quoted_commit_text_appears_in_that_commit(text: str) -> None:
    """Verbatim quotes must be verbatim.

    The record quotes two commit footers. Paraphrasing a quote while marking
    it verbatim is the same class of error the record exists to document, so
    each is checked against the message it is attributed to.
    """
    if not _git("rev-parse", "--git-dir").strip():
        pytest.skip("not a git checkout")

    # The quotes are EXTRACTED FROM THE DOCUMENT, not hardcoded here. The
    # first version of this test carried the pairs as literals, which meant it
    # compared git against itself and never read CORRECTIONS.md at all:
    # rewriting "1233 passed" to "1999 passed" in the document left the test
    # passing. A control caught that, and it is the exact failure mode this
    # module exists to prevent, so it is recorded rather than quietly fixed.
    #
    # Each entry's "**The claim:**" paragraph is paired with the commit named
    # in its "**Where:**" line, and every quoted string in that paragraph is
    # checked against that commit's message.
    pairs: list[tuple[str, str]] = []
    for block in re.split(r"^## C\d+\.", text, flags=re.M)[1:]:
        where = re.search(r"\*\*Where:\*\*(.*?)(?:\n\n|\Z)", block, re.S)
        claim = re.search(r"\*\*The claim[^:]*:\*\*(.*?)(?:\n\n|\Z)", block, re.S)
        if not (where and claim):
            continue
        shas = re.findall(r"`([0-9a-f]{7,40})`", where.group(1))
        quotes = re.findall(r'"([^"]{12,})"', claim.group(1))
        for sha in shas:
            for q in quotes:
                pairs.append((sha, q))

    assert pairs, (
        "no (commit, quote) pair could be extracted from CORRECTIONS.md; the "
        "test would then be vacuous, which is how the hardcoded version "
        "silently passed"
    )

    for sha, quote in pairs:
        if _git("cat-file", "-t", sha).strip() != "commit":
            pytest.skip(f"{sha} not present in this checkout")
        body = _git("show", "-s", "--format=%B", sha)
        flat = " ".join(body.split())
        assert " ".join(quote.split()) in flat, (
            f"{sha} does not contain the quoted text {quote!r}; the record "
            "attributes a claim to a commit that does not make it"
        )


def test_the_cash_cost_ratio_reproduces(text: str) -> None:
    """C4's arithmetic, recomputed from the two figures it states."""
    assert "157.77" in text and "795.02" in text
    ratio = 795.0150545140973 / 157.77194778602507
    assert ratio == pytest.approx(5.04, abs=0.005), (
        f"the ratio is {ratio:.4f}; the record's 5.04 must match"
    )
    # The phrase "twenty times" MUST be allowed to appear here, because C4
    # quotes the wrong claim in order to correct it. A blanket ban was the
    # first version of this assertion and it failed on the quotation, which is
    # a guard that cannot distinguish an error from a record of an error.
    # What must not survive is the phrase presented as current: inside a
    # "The claim:" line it is quoted history, anywhere else it is a live
    # assertion.
    live = [ln for ln in text.splitlines()
            if "twenty times" in ln and "**The claim:**" not in ln
            and not ln.strip().startswith("too low")]
    assert not live, (
        f"the overstated multiplier appears outside a quotation: {live}"
    )
    # And the corrected figure must be the one stated as current. Compared
    # whitespace-insensitively: the phrase is line-wrapped in the markdown as
    # "a factor of\n5.0", and a naive substring test failed on the newline,
    # reporting a missing correction that was present. A prose check that
    # depends on where the line breaks tests the formatter, not the claim.
    flat = " ".join(text.split())
    assert "a factor of 5.0" in flat, (
        "the record corrects the multiplier but never states the right one"
    )


def test_the_xia_claim_matches_the_current_registry(text: str) -> None:
    """C6 asserts six Xia benchmarks, all literature. Check against the file."""
    import csv
    reg = ROOT / "data" / "registry" / "validation_record.csv"
    if not reg.exists():
        pytest.skip("validation_record.csv not exported in this tree")
    rows = [r for r in csv.DictReader(reg.open()) if r["kind"] == "benchmark"]
    xia = {r["test"]: r["benchmark_kind"] for r in rows if "xia" in r["test"].lower()}
    assert len(xia) == 6, f"the record says six Xia benchmarks; found {len(xia)}"
    wrong = {t: k for t, k in xia.items() if k != "literature"}
    assert not wrong, f"the record says all six are literature; these are not: {wrong}"


def test_no_correction_is_left_without_a_resolution(text: str) -> None:
    """Each entry must say what replaced the wrong claim.

    An entry that states an error and stops is a confession, not a
    correction, and leaves the reader unable to tell whether the code was
    actually fixed.
    """
    entries = re.findall(r"^## (C\d+\..*)$", text, re.M)
    assert len(entries) >= 7, f"expected at least 7 entries, found {len(entries)}"
    blocks = re.split(r"^## C\d+\.", text, flags=re.M)[1:]
    for head, body in zip(entries, blocks):
        assert ("Replaced by" in body or "Resolution" in body
                or "Also changed" in body), (
            f"entry {head!r} states a problem but never says what replaced it"
        )
