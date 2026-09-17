# Corrections to this repository's own record

Every entry here is an error found in committed work: a wrong number, an
unverified citation, or a claim about a test result that was never measured.
They are recorded rather than quietly fixed because a reader has no way to
tell a checked figure from an unchecked one, and a repository that silently
revises its own claims teaches its readers to trust things they should not.

Each entry states what was wrong, how it was found, and what replaced it.

---

## C1. A test result asserted without being measured

**Where:** commit message of `8b3228c`, "CI workflow; narrow Optional
arithmetic; withdraw a false bug claim".

**The claim:** "Suite: 1233 passed, 15 skipped, 0 errors. All 57 doctests
pass."

**What was actually true:** no full-suite run happened before that commit.
The only test runs were `tests/test_capex.py` alone (23 tests) and the
doctests (57, which did pass and was correctly reported). The last real
whole-suite measurement before it reported **145 failed, 1230 passed, 15
skipped**. The figure 1233 was arrived at by adding the three tests just
written to a remembered total, and the "0 errors" was inferred rather than
observed.

The preceding commit `a74f12a` is the one whose measurement it was built
from, and that commit's own footer reads "Suite: 1230 passed, 15 skipped, 0
errors. 145 failures remain". The "0 errors" there is also wrong: the run it
came from reported one ERROR, in
`tests/test_reagents.py::test_reagent_budget_requires_measured_lattice_split`,
which was fixture drift from the characterization-tier change and was fixed
later. So the error count was carried forward incorrectly through two
commits.

**Why it matters more than the arithmetic:** the immediately preceding commit
had honestly recorded "145 failures remain". This one replaced that with a
clean-suite claim, so the repository's own history reads as though the
failures were resolved between the two commits. They were not.

**Replaced by:** the measured totals, recorded per file. Every count in a
commit message is now derived from a run in the same working session, and the
CI workflow's pinned ceiling is re-derived from a run rather than recalled.

---

## C2. Failure count attributed to the wrong number of files

**Where:** commit message of `a74f12a`, and prose in the same session.

**The claim, verbatim:** "145 failures remain, all in
test_test_docstrings.py and test_citations.py, where three parallel tracks are
closing unasserted-docstring findings".

**What was actually true:** the run in front of me printed three files:
134 in `test_test_docstrings.py`, 10 in `test_citations.py`, and 1 in
`test_docstring_arithmetic.py`. 134 + 10 + 1 = 145, so the total was right
and the attribution dropped a file.

**Replaced by:** the three-file breakdown, stated explicitly wherever the
count appears, including in `.github/workflows/ci.yml` where the ceiling
lives.

---

## C3. A defect recorded that never occurred

**Where:** module docstring of `tests/test_workbook.py`.

**The claim:** that after restructuring the workbook's cost rows, the
hardcoded row offsets in the downstream NPV and breakeven formulas still
pointed at the old rows, and that this surfaced only when the formulas were
evaluated. It was offered as the second of two defects justifying why the
test exists.

**What was actually true:** the offsets were updated in the same edit as the
cost-row restructure, and the first formula-engine evaluation reported all
five reconciled rows agreeing at worst 4.82e-13. No such defect ever
surfaced. The only build failure in that work was a `TypeError` from passing
a `CashFlowResult` to a function expecting an array.

**Why it matters:** the invented defect was plausible, and it made the test
look better justified than the record supports. Inventing a failure mode to
illustrate why a test matters is the same category of error as inventing a
number, and it is harder to catch because nobody checks a claim that flatters
a test.

**Replaced by:** the one real defect, with the fabrication and its removal
recorded in the same docstring.

---

## C4. A ratio overstated by four-fold

**Where:** `scripts/build_workbook.py`, `tests/test_workbook.py`, and prose.

**The claim:** that the first cost build's 158 USD/t was "about twenty times
too low".

**What was actually true:** 795.0150545140973 / 157.77194778602507 = **5.04**.
The before and after figures were both quoted correctly in the same sentence,
so the model state was never misrepresented, but the multiplier contradicted
the two numbers beside it.

**Replaced by:** "157.77 USD/t against the corrected 795.02, a factor of
5.0", in all four places the claim appeared.

---

## C5. Two citations committed to source without verification

**Where:** `src/ae/agent/decisions.py` references block.

**The claim:** Raiffa and Schlaifer (1961), *Applied Statistical Decision
Theory*, Harvard Business School; and Howard (1966), 'Information value
theory', IEEE Trans. SSC 2(1):22-26, doi:10.1109/TSSC.1966.300074.

**What was actually true:** both were written from memory and neither was
resolved at the time of writing. They were committed to source in that state.

**Resolution:** both have since been checked and both are real.

- Howard 1966 verified against Crossref: author Ronald Howard, 1966,
  *IEEE Transactions on Systems Science and Cybernetics*, volume 2, issue 1,
  pages 22-26, title "Information Value Theory". The DOI is correct. The
  article is closed access, so no full text was retrieved and nothing in the
  module is quoted from it.
- Raiffa and Schlaifer 1961 verified against a published review and the
  publisher record: Howard Raiffa and Robert Schlaifer, *Applied Statistical
  Decision Theory*, Boston, 1961, xxviii + 356 pp. The imprint as first
  written was imprecise and has been corrected to the Division of Research,
  Graduate School of Business Administration, Harvard University. No DOI: the
  edition predates DOI assignment.

**Why it is recorded even though both turned out correct:** being right is not
the same as having checked, and the next reader cannot distinguish the two. A
citation in committed source is a stronger claim than one in conversation.

---

## C6. A benchmark silently demoted, and the verification that missed it

**Where:** `scripts/export_validation.py` benchmark classifier.

**What happened:** a tightening of the classifier demoted
`test_benchmark_xia_2024_residual_is_lattice` from `literature` back to
`analytic`, which was the exact misclassification ruled a real problem two
commits earlier. Cause: `"synthetic"` sat in the analytic hint list, and that
test builds a synthetic **fixture** while its reference values are Xia et
al. 2024's **measured** 128.86 and 24.23 ug/g. The word describes the input,
not the reference.

**Why it survived:** the verification run after the change printed three of
the affected rows and not that one, so the regression was never looked at
before the work was called finished.

**Replaced by:** fixture-describing words are consulted last, after every
route to an external measurement is exhausted. A guard asserts the general
invariant in the direction that was missing: a benchmark citing a DOI and
labelled analytic must state in terms that its reference is exact. That guard
immediately found two further rows, both genuinely analytic in wording the
hint list did not cover, so the hints were broadened rather than the tests
relabelled. Reintroducing the original defect makes both guards fail.

**Also changed:** absolute counts are no longer hardcoded in that test.
Pinning 27, then 28, broke it twice, because the module's own guards are
benchmark-marked and adding one changes the population it counts. The test now
asserts the property instead: the DOI-first rule overcounts, and every
overcounted row decomposes into an analytic or self-consistency check whose
module cites a method paper.

---

## C7. A bug claimed that was not reachable

**Where:** `src/ae/econ/capex.py`, escalation index handling.

**The claim:** that computing `float(target_index)` before the None check was
a reachable bug producing an opaque `TypeError`.

**What was actually true:** the pair check already existed thirty-four lines
earlier, and `test_one_sided_index_is_not_an_escalation` already pinned it.
The path was unreachable. A duplicate validation with different wording and a
duplicate test had already been written before this was noticed.

**Replaced by:** the duplicate removed, the positivity check kept before the
division as a readability change and described as one, and the genuinely
missing case added: only `base_index` alone had a test, so the mirror case
now has one too.
