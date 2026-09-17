"""Make ``src/`` importable without requiring an editable install.

Why this file exists. The suite passed for twenty-five commits while the
repository had no mechanism of its own to put ``src/`` on ``sys.path``: an
editable install (``pip install -e .``) happened to be present in the
development environment, so ``import ae`` resolved. When that install was
lost, EVERY test module failed to collect with ``ModuleNotFoundError: No
module named 'ae'``. A fresh clone would have failed the same way on the first
``pytest`` run, which for a repository whose point is reproducibility is the
defect that matters most.

The habitual command in this project was ``pytest tests/ src/``, and an
earlier version of this note claimed the ``src/`` argument had been masking
the problem by making pytest prepend ``src`` to ``sys.path``. THAT CLAIM WAS
WRONG and the runs in front of me disproved it: with the editable install
gone, ``pytest tests/ src/`` failed with 56 collection errors, and
``pytest tests/test_units.py src/ae/core/units.py`` failed identically to
``pytest tests/test_units.py`` alone. Whatever rootdir insertion pytest does
for src-tree arguments, it was not what made ``import ae`` resolve here. The
only mechanism actually evidenced is the editable install: ``pip`` reported
PackageNotFoundError for ``ae`` once the suite started failing, and this file
is what removes the dependency on it.

An editable install is still the right thing for development, and
``pyproject.toml`` configures one. This file guarantees the suite runs without
it, so the tests are a property of the repository rather than of one machine's
environment.
"""
from __future__ import annotations

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_src_is_importable_without_an_editable_install() -> None:
    """The guard's own regression test: ``ae`` must import from ``src/``.

    Placed here rather than in a test module because a failure of this
    invariant prevents every test module from being collected at all, which
    reports as a collection error rather than a test failure.
    """
    import ae.core.units

    assert pathlib.Path(ae.core.units.__file__).resolve().is_relative_to(SRC), (
        f"ae imported from {ae.core.units.__file__}, not from {SRC}; a stale "
        "installed copy is shadowing the working tree"
    )
