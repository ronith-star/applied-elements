# Applied Elements platform

A physics, plant and economics model of quartz purification, with provenance
enforced on every physical parameter and the evidence behind every model stated
explicitly rather than implied.

**Read `HANDOFF.md` before trusting any output.** It states, per model, what is
validated against published measurements, what is scaffolding with no real data
behind it, and which measurements unlock which models. `PLAN.md` describes the
architecture. `CONTRIBUTING-provenance.md` describes the tagging discipline.
`docs/CORRECTIONS.md` is the repository's record of errors found in its own
committed work.

Every command and every count in this file was run in a clone with no editable
install at commit `c920e1d` and the output read. Nothing here is recalled.

## Requirements

Python 3.13.15 was used for the measurements below; `pyproject.toml` declares
`requires-python >=3.12`.

Runtime dependencies, as declared in `pyproject.toml`:
`numpy>=2.0`, `scipy>=1.14`, `pandas>=2.2`, `matplotlib>=3.9`, `sympy>=1.13`,
`pint>=0.24`, `pydantic>=2.8`, `SALib>=1.5`, `simpy>=4.1`, `PuLP>=2.8`,
`scikit-learn>=1.5`, `xgboost>=2.1`, `openpyxl>=3.1`, `XlsxWriter>=3.2`,
`PyKrige>=1.7`. Development extras: `pytest>=8.0`, `pytest-cov>=5.0`,
`mypy>=1.11`, `ruff>=0.6`.

## Install

```bash
git clone <this repo> ae-platform
cd ae-platform
python -m pip install numpy scipy pandas matplotlib sympy pint \
  SALib simpy PuLP scikit-learn xgboost openpyxl XlsxWriter \
  pydantic PyKrige pytest
```

**No editable install is required, and none should be performed.**
`tests/conftest.py` puts `src/` on `sys.path`, and the three scripts in
`scripts/` do the same for themselves. I verified this in a clean clone:
`python -m pip show ae-platform` reports `WARNING: Package(s) not found:
ae-platform`, and the suite still runs.

This is not incidental. `tests/conftest.py` carries its own regression test,
`test_src_is_importable_without_an_editable_install`, whose docstring records
that the suite passed for twenty-five commits only because an editable install
happened to exist in the environment, and that removing it produced 56
collection errors. That test is why the arrangement is now checked rather than
assumed.

One consequence worth knowing: `python -c "import ae"` fails with
`ModuleNotFoundError` outside pytest and outside those scripts, because nothing
has put `src/` on the path. That is expected. Within pytest and within the
scripts it resolves to `src/ae/`.

## Run the tests

### Full suite

```bash
python -m pytest tests/
```

Measured at `c920e1d`, counts parsed from `--junitxml`:

| | count |
| --- | --- |
| collected | 1,463 |
| passed | 1,445 |
| failed | 0 |
| errors | 0 |
| skipped | 18 |
| wall time | 168 s |

Exit code 0. These counts include the 11 guards in `tests/test_handoff_claims.py`,
which this track adds. The prose-audit backlog described below is now closed:
running the three audit files alone gives 532 collected, 516 passed, 0 failed, 16
skipped.

The 18 skips break down as 2 in `tests/test_workbook.py`, skipped at collection
by `pytest.importorskip("formulas")` because the Excel formula engine is not
installed, and 16 parametrized cases of
`test_every_intext_citation_is_in_the_reference_block`, skipped with the message
that the module in question makes no in-text author-year citation.

### The gate that CI actually enforces

```bash
python -m pytest tests/ \
  --ignore=tests/test_test_docstrings.py \
  --ignore=tests/test_citations.py \
  --ignore=tests/test_docstring_arithmetic.py
```

Measured: 931 collected, 929 passed, 0 failed, 0 errors, 2 skipped, 218 s, exit
code 0.

### Doctests

```bash
python -m pytest --doctest-modules src/ae
```

Measured: 57 collected, 57 passed, 0 failed, 1 s, exit code 0. Note that
`pyproject.toml` sets `addopts = "-q --strict-markers --doctest-modules"`, so
doctests run on any path you pass that contains modules.

### By marker

```bash
python -m pytest tests/ -m benchmark      # 48 collected, 0 failed, 2 skipped
python -m pytest tests/ -m golden         # 127 collected, 0 failed, 2 skipped
```

Add `-s` to see the benchmark output. Each benchmark prints its reference value,
the model value and the error, which is the fastest way to see what the platform
is actually checked against. Both counts above were measured from a
`--junitxml` run; both exceed the 47 and 125 rows in
`data/registry/validation_record.csv` because the validation-record guards in
`tests/test_validation_record.py` are themselves marker-carrying tests.

### One file

```bash
python -m pytest tests/test_thermal.py    # 49 collected, 0 failed, 0 skipped
```

## The CI arrangement, and why the prose audit is pinned rather than fatal

`.github/workflows/ci.yml` has four stages. Reading them in order tells you what
the project treats as a gate.

1. **Model suite (fatal).** Runs `pytest tests/` with the three prose-audit files
   ignored. This runs BEFORE any install of the package, which is what proves a
   fresh clone works.
2. **Documentation audit (pinned, must not regress).** Runs the three prose-audit
   files, counts `FAILED ` lines, and fails only if the count RISES above 149.
3. **Lint and type checks (advisory, not fatal).** Reported with the reason
   recorded in the workflow rather than silenced with `|| true`.
4. **Doctests (fatal),** then both registry exporters, then an upload of
   `data/registry/*.csv` as a build artifact.

The prose audit is pinned because of what those three files check and what they
found. `tests/test_test_docstrings.py` asserts that every number written in a
test docstring is reproduced by an assertion in that same test body.
`tests/test_citations.py` asserts that every in-text author-year citation appears
in the module's reference block. `tests/test_docstring_arithmetic.py` asserts
that arithmetic claimed in prose is reproduced in code. They were added to find
documentation that had drifted from the code, and they found a great deal of it.

Making them fatal today would gate every commit on a documentation backlog and
turn a red badge into background noise that nobody reads. Deleting them would
discard the finding. So the count is pinned: new undocumented numbers cannot
enter, while the existing backlog is worked down. The workflow's own comment
instructs whoever reduces the count to lower the ceiling, and at zero to delete
the step and move the three files into the fatal gate.

The workflow still enforces a ceiling of 149, and the comment above it still
records the split 138 plus 10 plus 1 that was measured when the ceiling was set.
**That backlog has since been closed.** Measured at `c920e1d`: the three audit
files give 532 collected, 516 passed, 0 failed, 16 skipped. The ceiling is now
slack rather than a debt register, and the workflow's own instruction applies:
at zero, lower the number, delete the pinned step, and move the three files into
the fatal gate above. I have not edited the workflow, because it belongs to
another track.

The repository has been bitten by a recalled count before, which is why the
pinned step was written to re-derive rather than recall: `docs/CORRECTIONS.md`
C1 and C2 record a commit message asserting "1233 passed, 15 skipped, 0 errors"
from a run that never happened, and a ceiling of 145 attributed to two files when
the run in front of the author printed three.
`scripts/commit_with_count.sh` exists to measure the count rather than type it,
and it refuses to commit on any failure.

## Build the workbook

```bash
python scripts/build_workbook.py            # writes data/AE-model-mirror.xlsx
python scripts/build_workbook.py /tmp/out    # or to a directory you name
```

Measured output, run in a clean clone:

```
AE-model-mirror.xlsx: 12,900 bytes
  3 ores x 3 sites = 9 scenarios
  base case: yield 0.7814, cash cost 795.0 USD/t, NPV +24.8 MUSD, breakeven 2,339.3 USD/t
```

Exit code 0. Every calculated cell in the mirror is an Excel formula, not a
pasted Python result, and a Reconciliation sheet computes the outputs both ways.
The workbook deliberately does not reproduce the Monte Carlo or the Sobol
decomposition; it imports P10, P50, P90 and the Sobol indices as clearly labelled
imported results, naming the producing script and commit.

The claim that the formulas reproduce the Python model is checked by
`tests/test_workbook.py` using an independent formula engine, and **that check
does not run in this environment**: `formulas` is not installed, so those 2 tests
skip. Install `formulas` to verify it.

## Run the exporters

```bash
python scripts/export_registry.py            # writes data/registry/*.csv
python scripts/export_validation.py
python scripts/export_registry.py /tmp/out   # or to a directory you name
```

Measured output of `export_registry.py`, exit code 0:

```
parameter_registry.csv       203 provenance-tracked values
definitional_constants.csv    92 rows, of which 40 are actual constants
  by role: {'numeric_table': 17, 'numeric_constant': 23, 'bibliography': 50, 'prose': 2}
  by tag: {'ASSUMED': 61, 'SOURCED': 134, 'DERIVED': 8}
  SOURCED/MEASURED with a DOI or URL: 134/134
  point estimates with no uncertainty stated: 0
```

Measured output of `export_validation.py`, exit code 0:

```
validation_record.csv  162 marked tests: {'golden': 115, 'benchmark': 47}
  literature benchmarks (cite a published measurement): 28
  analytic benchmarks (exact closed form or known generator): 12
  self-consistency benchmarks (two routes must agree): 7
  UNCLASSIFIED, needing a source or an analytic basis: 0
  benchmarks stating a numeric error: 22/47
  modules with at least one benchmark: 18
```

**Do not quote the 28 literature benchmarks without the caveat in `HANDOFF.md`.**
The classifier's last resort is to call a benchmark `literature` if the module
under test cites a DOI anywhere in its reference block, and I measured that 22 of
the 28 reach that class by the fallback alone, with no reported-value phrase in
the test's own docstring. Several of those are analytic checks.

Both exporters are idempotent against upstream code. `export_registry.py` run in
a clean clone left `git status --porcelain data/` empty, and `export_validation.py`
reproduces the committed `data/registry/validation_record.csv` exactly (162 marked
tests, 115 golden, 47 benchmark) when `tests/test_handoff_claims.py` is absent.
With that file present the regenerated CSV holds 173 marked tests and 126 golden,
because its 11 guards carry the `golden` marker and the exporter collects every
marked test. That is the exporter working, not drift. I have not committed a
regenerated CSV.

## Layout

```
src/ae/core/        units, provenance, feedstock, site, registry
src/ae/physics/     12 modules, 11,116 lines, the bulk of the platform
src/ae/plant/       streams, yield_cascade, capacity, scheduling
src/ae/econ/        unit_economics, capex, valuation, uncertainty
src/ae/ml/          surrogate (trained on synthetic data, see HANDOFF.md)
src/ae/agent/       decisions (EVPI; no test file, see HANDOFF.md)
tests/              36 files including conftest.py
scripts/            build_workbook.py, export_registry.py,
                    export_validation.py, commit_with_count.sh
data/registry/      parameter_registry.csv, definitional_constants.csv,
                    validation_record.csv, environment.json
docs/CORRECTIONS.md errors found in this repository's own committed work
docs/figures/       architecture.svg, architecture.png
```

27 modules, 17,017 source lines, measured with
`find src/ae -name '*.py' ! -name '__init__.py' | xargs wc -l`.

## Architecture in one figure

![Architecture](docs/figures/architecture.svg)

See `PLAN.md` for the module-by-module walk and the measured import graph. The
one fact to carry from it: physics does not import plant, plant does not import
econ, and econ does not import the decision layer. Those couplings exist only as
composition at a call site, and the call site that exercises all three is
`tests/test_integration_uncertainty.py`.

## Committing

Four agents share this remote. Use the environment prefix and rebase before
pushing:

```bash
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 GIT_TERMINAL_PROMPT=0
git pull --rebase origin main
git add <only the files you own>
git commit
git push origin main
```

Stage only your own files. Note that `scripts/commit_with_count.sh` ends with
`git add -A`, so it is unsuitable for a shared remote where tracks own disjoint
file sets; use it only if you genuinely intend to stage everything.
