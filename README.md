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

Every command and every count in this file was run and the output read. Nothing
here is recalled. Two kinds of run are reported and labelled as such: runs in an
INDEPENDENT CLONE of `origin/main` (a fresh `git clone`, no editable install,
`python -m pip show ae-platform` reporting the package not found, my four
documents and guard file copied in), and runs in the WORKING TREE, which also has
no editable install. Where they differ the difference is stated.

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

Measured in the working tree at `1d996ff`, counts parsed from `--junitxml`:

| | count |
| --- | --- |
| collected | 1,842 |
| passed | 1,812 |
| failed | 12 |
| errors | 0 |
| skipped | 18 |
| wall time | 129 s |

Exit code 1. These counts include the 13 guards in `tests/test_handoff_claims.py`,
which this track adds and which all pass.

The 12 failures are 11 in `tests/test_test_docstrings.py` (the docstring audit)
and 1 in `tests/test_registry_export.py`. None is in a file this track owns. The
11 audit failures are unasserted docstring numbers in other tracks' test files,
measured by parsing the parametrized test ids: `test_physics_audit.py` 9,
`test_diffusion.py` 1, `test_registry_export.py` 1.

For contrast, the independent-clone run at `c920e1d`, before those files landed,
measured 1,461 collected, 1,442 passed, 1 failed, 18 skipped, 324 s. The single
failure there was an unasserted number in my own guard docstring, which I then
removed. So this is work in flight by three other tracks rather than a
regression in anything documented here, and since commit `56e61ff` folded the
audit into the fatal gate, CI goes red on it, which is the intended behaviour.

The 18 skips break down as 2 in `tests/test_workbook.py`, skipped at collection
by `pytest.importorskip("formulas")` because the Excel formula engine is not
installed, and 16 parametrized cases of
`test_every_intext_citation_is_in_the_reference_block`, skipped with the message
that the module in question makes no in-text author-year citation.

### The gate that CI actually enforces

As of `56e61ff` the gate is the full suite, `python -m pytest tests/`, with no
files excluded. Before that commit the three prose-audit files were excluded and
pinned separately, and that narrower run is still useful when you want to know
whether a failure is a model defect or a documentation defect:

```bash
python -m pytest tests/ \
  --ignore=tests/test_test_docstrings.py \
  --ignore=tests/test_citations.py \
  --ignore=tests/test_docstring_arithmetic.py
```

Measured: 1,208 collected, 1,205 passed, 1 failed, 0 errors, 2 skipped, 129 s,
exit code 1. The gate is currently RED, and not from my track: the single
failure is `tests/test_registry_export.py::test_definitional_roles_are_reported_honestly`.

Running with and without the audit files is what separates a documentation
defect from a model defect, and at this commit there is one of each kind. The
full suite's 12 failures split 11 documentation (unasserted docstring numbers,
which only the audit files see) and 1 export (which the gate sees, and which is
the one above). Two earlier versions of this paragraph were wrong in the same
way: they described a failure set that a later re-measurement had already
changed, once saying all failures were in the audit files and once counting two
golden-vector failures that another track has since fixed.

### Doctests

```bash
python -m pytest --doctest-modules src/ae
```

Measured: 57 collected, 57 passed, 0 failed, 1 s, exit code 0. Note that
`pyproject.toml` sets `addopts = "-q --strict-markers --doctest-modules"`, so
doctests run on any path you pass that contains modules.

### By marker

```bash
python -m pytest tests/ -m benchmark      # 50 collected, 0 failed, 2 skipped
python -m pytest tests/ -m golden         # 299 collected, 0 failed, 2 skipped
```

Add `-s` to see the benchmark output. Each benchmark prints its reference value,
the model value and the error, which is the fastest way to see what the platform
is actually checked against. Both counts were measured from a `--junitxml` run
at `1d996ff`, and both exceed the committed
`data/registry/validation_record.csv` (162 marked tests, 115 golden, 47
benchmark). They exceed it for DIFFERENT reasons, and an earlier version of this
paragraph gave one cause for both.

The golden half is the exporter's scan scope.
`scripts/export_validation.py:190` globs `tests/test_*.py` non-recursively, so
it never sees `tests/golden/test_golden_vectors.py`. Measured decomposition of
the 299 golden collections: 168 from `tests/golden/`, invisible to the exporter;
116 from the flat `tests/test_*.py` files it does scan; 13 from the guards in
`tests/test_handoff_claims.py`, which are flat and therefore also scanned; and 2
skipped. The registry's golden count is a count of what the exporter can see,
not of golden-marked tests in the repository.

The benchmark half is drift, and scan scope contributes nothing to it, because
`tests/golden/` carries no benchmark marker at all (measured: 0 of the 50
collections). Running the exporter reports 49 benchmark-marked tests against the
committed 47, the two additions coming from other tracks' commits after the CSV
was last written.

An earlier measurement here found two failures in
`tests/golden/test_golden_vectors.py`; another track has since fixed them, and
the golden run is now clean.

### One file

```bash
python -m pytest tests/test_thermal.py    # 49 collected, 0 failed, 0 skipped
```

## The CI arrangement

`.github/workflows/ci.yml` at `56e61ff` has four stages. Reading them in order
tells you what the project treats as a gate.

1. **Model suite and documentation audit (fatal).** Runs `python -m pytest
   tests/ -q`, the whole suite, and it runs BEFORE any install of the package,
   which is what proves a fresh clone works.
2. **Lint and type checks (advisory, not fatal),** with the reason recorded in
   the workflow rather than silenced with a bare `|| true`.
3. **Doctests in modules (fatal).**
4. **Both registry exporters,** then an upload of `data/registry/*.csv` as a
   build artifact.

The prose audit is now inside the fatal gate, and that is a recent change worth
understanding, because the three files it comprises are unusual. They do not test
the models. `tests/test_test_docstrings.py` asserts that every number written in
a test docstring is reproduced by an assertion in that same test body.
`tests/test_citations.py` asserts that every in-text author-year citation appears
in the module's reference block. `tests/test_docstring_arithmetic.py` asserts
that arithmetic claimed in prose is reproduced in code.

They were added to find documentation that had drifted from the code, and they
found a great deal of it: at the time they were written the count was 148
failures, and making them fatal then would have gated every commit on a
documentation backlog and turned a red badge into noise. So the count was pinned
instead, with the rule that new undocumented numbers could not enter while the
backlog was worked down, and that at zero the step would be deleted and the three
files folded into the fatal suite. That is what happened. Measured at
`c920e1d`, the three files alone gave 532 collected, 516 passed, 0 failed, 16
skipped, and `56e61ff` removed the pinned step. Re-measured at `1d996ff` they
give 634 collected, 607 passed, 11 failed, 16 skipped. The 11 failures recorded in the
full-suite table above arrived after that, from test files another track is
still writing, and they are exactly what the now-fatal audit is meant to catch.

The consequence for a contributor: a new unasserted number in a docstring, an
unresolved citation, or an arithmetic claim not reproduced in code now fails the
build exactly like a wrong model. Treat that as the intended state rather than
an obstacle. The repository has been bitten by recalled numbers before, which is
why the audit exists at all: `docs/CORRECTIONS.md` C1 and C2 record a commit
message asserting "1233 passed, 15 skipped, 0 errors" from a run that never
happened, and a ceiling of 145 attributed to two files when the run in front of
the author printed three. `scripts/commit_with_count.sh` exists to measure the
count rather than type it, and it refuses to commit on any failure.

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

Measured stdout of `export_validation.py` at `1d996ff`, exit code 0. Note that
this command REWRITES the CSV before printing, so what it prints is the state of
the tree, not the state of the committed file:

```
validation_record.csv  178 marked tests: {'golden': 129, 'benchmark': 49}
  literature benchmarks (cite a published measurement): 28
  analytic benchmarks (exact closed form or known generator): 14
  self-consistency benchmarks (two routes must agree): 7
  UNCLASSIFIED, needing a source or an analytic basis: 0
  benchmarks stating a numeric error: 22/49
  modules with at least one benchmark: 18
```

The committed file holds 162 marked tests, 115 golden, 47 benchmark, with the
split 28 literature / 12 analytic / 7 self-consistency. Those are the figures
`HANDOFF.md` quotes, because they are what a fresh clone contains before anyone
runs the exporter. An earlier version of this block presented the committed
figures as this command's output, which they are not.

**Do not quote the 28 literature benchmarks without the caveat in `HANDOFF.md`.**
The classifier's last resort is to call a benchmark `literature` if the module
under test cites a DOI anywhere in its reference block, and I measured that 22 of
the 28 reach that class by the fallback alone, with no reported-value phrase in
the test's own docstring. Several of those are analytic checks.

`export_registry.py` is idempotent: running it in a clean clone left
`git status --porcelain data/` empty. `export_validation.py` is deterministic,
but the committed CSV is now behind the tree. Measured at `1d996ff` by moving my
guard file out and running the exporter: 165 marked tests, 116 golden, 49
benchmark, against the committed 162 / 115 / 47. So other tracks have added one
golden-marked and two benchmark-marked tests since the CSV was written.
Regenerating with my guard file present gives 178 / 129 / 49: exactly 13 more
golden rows, one per guard, which I confirmed by asking the exporter for the
rows it attributes to that file. The analytic count also moved, from 12 to 14. I have not committed a regenerated CSV, because that file
belongs to another track; regenerate it before quoting any of its totals.

## Layout

```
src/ae/core/        units, provenance, feedstock, site, registry
src/ae/physics/     12 modules, 11,358 lines, the bulk of the platform
src/ae/plant/       streams, yield_cascade, capacity, scheduling
src/ae/econ/        unit_economics, capex, valuation, uncertainty
src/ae/ml/          surrogate (trained on synthetic data, see HANDOFF.md)
src/ae/agent/       decisions (EVPI; no test file, see HANDOFF.md)
tests/              41 files including conftest.py
scripts/            build_workbook.py, export_registry.py,
                    export_validation.py, commit_with_count.sh
data/registry/      parameter_registry.csv, definitional_constants.csv,
                    validation_record.csv, environment.json
docs/CORRECTIONS.md errors found in this repository's own committed work
docs/figures/       architecture.svg, architecture.png
```

27 modules, 17,662 source lines, measured with
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
