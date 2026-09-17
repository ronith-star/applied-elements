# CONTRIBUTING: the provenance discipline

Measured at commit `e926ac6`. The counts in this file come from
`python scripts/export_registry.py`, run in this repository.

Every physical parameter in `src/ae` carries a tag and, if it is external, a
source. This is enforced by validators in `src/ae/core/provenance.py`, not by
convention, so a value that does not comply fails at construction rather than at
review.

## The five tags

Defined as the `Tag` enum in `src/ae/core/provenance.py`.

**`MEASURED`** means measured on our own material, in a named laboratory. Nothing
in this repository currently carries this tag, because no Vikarabad sample has
been assayed. When the characterization campaign returns, its results are the
first `MEASURED` values in the platform.

**`SOURCED`** means taken from an external reference. A `Source` will not
construct without a DOI or a URL and an access date. Measured coverage: 203
provenance-tracked values, of which 134 are `SOURCED`, and 134 of 134 carry a DOI
or a URL. That ratio is checked by the exporter on every run, so it cannot drift
silently.

**`COMPUTED`** means derived inside the platform from tagged inputs by an explicit
calculation, where the calculation is the whole content of the value.

**`DERIVED`** means obtained from sourced values by a transformation that
introduces an assumption of its own, for example converting a published oxide
percentage to an element concentration using a stoichiometric factor, or
converting a short-ton basis to a metric basis. 8 values carry this tag. The
distinction from `COMPUTED` matters because a `DERIVED` value inherits both the
source's uncertainty and the transformation's.

**`ASSUMED`** means engineering judgement. **An `ASSUMED` value will not construct
without a stated `basis` string.** 61 values carry this tag, and the basis field
is what makes them auditable: an assumption with a stated basis can be attacked,
replaced, or bracketed, while a bare number cannot.

Every point estimate in the registry states an uncertainty. Measured: 0 point
estimates with no uncertainty stated.

## The three source tiers, and why a Tier 3 source may never be sole evidence

Defined as the `Tier` enum in `src/ae/core/provenance.py`.

**Tier 1** is peer-reviewed literature, curated databases, government data and
standards bodies. 133 registry rows.

**Tier 2** is patents, supplier datasheets, industry reports, theses, filings and
trade data. 1 registry row. These are real evidence with a known bias: a supplier
datasheet describes what a vendor is willing to warrant, a patent describes a
claim's scope rather than a working process, and an industry report's method is
usually not disclosed.

**Tier 3** is news, blogs, forums and trade press. **Zero registry rows, which is
the intended state.** A Tier 3 source is admissible as context, for example to
learn that a plant exists or that a price moved, and it is never admissible as the
sole evidence for a number in a model.

The reason is specific rather than snobbish. A Tier 3 source does not disclose its
method, so an error in it is undetectable from the source itself. When a trade
article reports a purity of 99.999 percent, the reader cannot tell whether that
refers to SiO2 by difference, a trace-element sum, a single-element
specification, or a marketing claim, and those four differ by orders of magnitude
on the quantity that matters. This platform already contains a measured instance
of exactly that ambiguity: `impurity_location::test_benchmark_qu_2025_two_ore_bodies`
found that by-difference SiO2 computed from the published cation sum exceeds the
paper's own published SiO2 upper bound in both ore bodies (HT +0.0281, PX +0.0035
percent relative), which proves the two conventions are not interchangeable even
in a peer-reviewed source where the method is disclosed. In a Tier 3 source the
same ambiguity exists and cannot be resolved at all.

A second reason is traceability. Tier 3 URLs rot, and the content behind a stable
URL changes without notice. A DOI resolves to a fixed version of record. The
access date on every `Source` exists so that a future reader can tell what was
seen, but only a Tier 1 or Tier 2 source makes that date meaningful.

The practical rule: a Tier 3 source may motivate a search, may be cited alongside
a Tier 1 or Tier 2 source that carries the number, and may appear in a market or
competitive-landscape narrative. It may not be the citation on a registry value.

## What a grade-conditional claim may say

The Vikarabad deposit is `unmeasured` in the citable record. Two source files say
so in the negative rather than implying otherwise:
`src/ae/physics/comminution.py:146` ("No work index in this module is measured on
Vikarabad ore") and `src/ae/physics/impurity_location.py:115` ("No partition data
exists for the Vikarabad deposit").

Therefore any statement of the form "the ore can reach 4N8" is a scenario gated on
a future assay campaign, and must be written as one. It is never a property of the
deposit. The tier gate in `src/ae/core/feedstock.py` enforces this in code:
`purification_ceiling`, `product_grade_claim` and `qualification_dossier` are
refused until the characterization tier reaches `located`, which requires spatially
resolved impurity data, not a bulk assay.

## What to do when a number cannot be sourced

Write `NOT SOURCED` and say what you tried. This is preferable to every
alternative, and specifically:

Do not substitute a number from a neighbouring system and leave the citation
pointing at the original context. Do not average two sources that measured
different quantities. Do not carry a number forward from an earlier draft whose
source you cannot now locate. Do not cite a review for a primary measurement
without reading the primary; the review's transcription error becomes yours.

If the number is needed to make a model run, tag it `ASSUMED`, state the basis,
and bracket it. The platform has a worked example of the value of doing this:
`separation::test_benchmark_whims_field_exponent_sensitivity` takes the `ASSUMED`
WHIMS field exponent `n_B = 2.0`, brackets it 1 to 3, and measures what the
assumption costs: recovery spreads 17.7 percent relative at 60 seconds residence
and 89.4 percent at 10 seconds. That is a usable input to a decision. A bare 2.0
with no bracket is not.

## Benchmarks: three kinds, and the one that is not external validation

`scripts/export_validation.py` classifies every `@pytest.mark.benchmark` test as
`literature`, `analytic` or `self_consistency`. Measured split of the 47
benchmarks: literature 28, analytic 12, self-consistency 7.

**An analytic check is not external validation.** Reproducing the Ishigami
function's closed-form Sobol indices to 0.21 percent proves the estimator is
implemented correctly; it says nothing about whether the model's inputs describe
real ore. A self-consistency check proves two routes through the platform agree,
which is weaker still: two routes can agree and both be wrong.

Do not quote the `literature` count of 28 without the caveat recorded in
`HANDOFF.md`. The classifier's last resort is to call a benchmark `literature` if
the module under test cites a DOI anywhere, and I measured that 22 of the 28
reach the class by that fallback alone, with no reported-value phrase in the
test's own docstring. Several of those are analytic checks that happen to live in
a module with a reference block.

When you add a benchmark, state in the docstring where its reference number comes
from, in words. The classifier reads those words, and more importantly so does the
next reader. "Reported:" and "measured value" and "the paper's" are the phrases
that mark an external measurement; "reference value is exact" and "known in
closed form" mark the analytic case.

## Numeric claims in docstrings

Every number in a test docstring must be reproduced by an assertion in that same
test body, or moved to a clearly marked descriptive note. The audit that enforces
this is `tests/test_test_docstrings.py`. Measured at `e926ac6`, the three audit
files report 597 collected, 570 passed, 11 failed, 16 skipped, the 11 failures all being unasserted
numbers in other tracks' test files written in the same window, so the backlog that made the
CI step pinned rather than fatal is closed. The pinned step and its ceiling of
149 remain in the workflow as slack; see the CI section of `README.md`.

The failure mode this audit exists to catch is real and is recorded in the repo.
`docs/CORRECTIONS.md` C4 records a multiplier that contradicted the two numbers
printed beside it: an early cost build's 157.77 USD per tonne against the
corrected 795.02 was described as "about twenty times too low" when the ratio is
5.04. Both endpoint figures were quoted correctly in the same sentence, so the
model state was never misrepresented, and the error was purely in the derived
multiplier, which is exactly the class of claim the arithmetic audit checks.
