"""Export every provenance-tracked parameter in the platform to CSV.

Run: python scripts/export_registry.py [outdir]

Walks each module under ``src/ae``, finds every ``Value`` object (including
those nested in dicts, lists and tuples of priors), and writes one row per
parameter with its tag, quantity, uncertainty distribution, source, DOI or URL,
access date and evidence tier.

TWO CLASSES OF CONSTANT, deliberately treated differently.

``Value``-wrapped parameters are quantities that could have been measured
differently: a Bond work index, a diffusivity prefactor, a reagent price. They
carry uncertainty and provenance because a different source would give a
different number, and the Monte Carlo layer samples them.

Bare module constants are DEFINITIONAL and are not wrapped: molar masses
(IUPAC standard atomic weights), oxide stoichiometry, unit conversion factors
such as the short ton in kilograms, and published classification tables such
as the AACE accuracy bands. Wrapping these would imply a sampling distribution
over a definition. They are listed in the second output file so the split is
auditable rather than implicit, and so a constant that is quietly an estimate
cannot hide among them.

The second file is further split by ROLE, because "uppercase name containing a
digit" swept in three unrelated kinds of object. Of 92 rows, 42 were ``Source``
objects and 8 were access dates, which are bibliography, not constants; 2 were
prose strings documenting the basis of values held elsewhere. Reporting those
as definitional constants would misstate what the platform hardcodes. Only
numeric scalars, quantities and numeric tables are constants in the intended
sense, and the ``role`` column names which is which.
"""
from __future__ import annotations

import ast
import csv
import importlib
import pathlib
import sys
from typing import Any, Iterator

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ae.core.provenance import Value  # noqa: E402

MAX_DEPTH = 6


def _iter_values(obj: Any, depth: int = 0,
                 seen: set[int] | None = None) -> Iterator[Value]:
    """Every Value reachable from ``obj``, including through OBJECTS.

    Traversing only dicts, lists, tuples and sets was not enough and the gap
    was silent. ``PARTITION_PRIORS`` in impurity_location is a dict of
    ``ElementPartition`` objects, each HOLDING Values built by an ``_assumed``
    helper; because the walk stopped at the object boundary, none of those
    parameters reached the registry, and the constant was then misfiled as
    definitional. Any container that holds provenance must be walked, so this
    also descends into ``__dict__`` and ``__slots__`` attributes of ordinary
    objects and pydantic models.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > MAX_DEPTH:
        return
    seen.add(id(obj))
    if isinstance(obj, Value):
        yield obj
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_values(v, depth + 1, seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            yield from _iter_values(v, depth + 1, seen)
        return
    # Scalars and callables hold nothing; descending into them would walk the
    # whole interpreter. Everything else may be a record type holding Values.
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return
    if isinstance(obj, type) or callable(obj) and not hasattr(obj, "__dict__"):
        return
    attrs: dict[str, Any] = {}
    if hasattr(obj, "__dict__"):
        attrs.update(vars(obj))
    for slot in getattr(type(obj), "__slots__", ()) or ():
        if hasattr(obj, slot):
            attrs[slot] = getattr(obj, slot)
    for name, v in attrs.items():
        if name.startswith("__"):
            continue
        yield from _iter_values(v, depth + 1, seen)


def _modules() -> list[str]:
    out = []
    for f in sorted((ROOT / "src" / "ae").rglob("*.py")):
        if f.name == "__init__.py":
            continue
        out.append(str(f.relative_to(ROOT / "src")).replace("/", ".")[:-3])
    return out


def _dist_text(v: Value) -> str:
    d = v.dist
    if d is None:
        return "point estimate, no uncertainty stated"
    rel = " (relative to nominal)" if getattr(d, "relative", False) else ""
    if d.kind == "normal":
        return f"normal(loc={d.loc}, scale={d.scale}){rel}"
    if d.kind == "uniform":
        return f"uniform(low={d.low}, high={d.high}){rel}"
    if d.kind == "triangular":
        return f"triangular(low={d.low}, mode={d.mode}, high={d.high}){rel}"
    if d.kind == "lognormal":
        return f"lognormal(loc={d.loc}, scale={d.scale}){rel}"
    return f"{d.kind}{rel}"


def collect_values() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mod_name in _modules():
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:                        # pragma: no cover
            print(f"  SKIP {mod_name}: {type(exc).__name__}", file=sys.stderr)
            continue
        for container, obj in list(vars(mod).items()):
            if container.startswith("__"):
                continue
            for v in _iter_values(obj):
                s = v.source
                tag = v.tag.value if hasattr(v.tag, "value") else str(v.tag)
                tier = getattr(s, "tier", None) if s else None
                rows.append({
                    "module": mod_name,
                    "container": container,
                    "tag": tag,
                    "quantity": str(v.quantity),
                    "uncertainty": _dist_text(v),
                    "confidence": v.confidence or "",
                    "citation": (s.citation if s else ""),
                    "doi": (getattr(s, "doi", None) or "") if s else "",
                    "url": (getattr(s, "url", None) or "") if s else "",
                    "accessed": (str(getattr(s, "accessed", "")) if s else ""),
                    "tier": (tier.value if hasattr(tier, "value")
                             else str(tier) if tier else ""),
                    "basis": (v.basis or "").replace("\n", " "),
                })
    # Dedupe: the same Value can be referenced from several containers.
    seen: set[tuple[str, ...]] = set()
    out = []
    for r in rows:
        k = (r["module"], r["container"], r["quantity"], r["tag"], r["citation"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _role(obj: Any) -> str:
    """What kind of thing this constant is, by runtime type.

    ``bibliography`` for a Source or an access date, ``prose`` for a string
    documenting a basis, ``numeric_constant`` for a scalar or quantity, and
    ``numeric_table`` for a dict or sequence of numbers. Only the last two are
    constants in the sense the file claims to report.
    """
    import datetime as _dt

    from ae.core.provenance import Source as _Source

    if isinstance(obj, (_Source, _dt.date, _dt.datetime)):
        return "bibliography"
    if isinstance(obj, str):
        return "prose"
    if isinstance(obj, (int, float, bool)):
        return "numeric_constant"
    if type(obj).__name__ == "Quantity":
        return "numeric_constant"
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        return "numeric_table"
    return f"other:{type(obj).__name__}"


def collect_definitional() -> list[dict[str, str]]:
    """Uppercase module constants that carry NO provenance at all.

    Classified by RUNTIME CONTENT, not by a source-text substring. The first
    version tested each assignment's source for the literal ``"Value("``,
    which misfiled every constant built through a helper: ``PARTITION_PRIORS``
    (built from ``_assumed(...)``), ``FORMATION_ENTHALPY`` (from ``_v(...)``),
    ``TRANSITIONS`` and ``QUARTZ_LANDAU`` (from ``_sourced(...)`` and
    ``_hp(...)``) all landed in the "definitional" file despite carrying
    explicit ASSUMED bases or sourced measurements. ``FORMATION_ENTHALPY``
    appeared in BOTH files at once. That is the exact failure this second file
    was introduced to prevent, so the criterion is now the one that cannot
    disagree with the registry: import the constant and walk it. If the walk
    finds a single Value, it is a parameter and belongs only in the registry.
    """
    rows = []
    for f in sorted((ROOT / "src" / "ae").rglob("*.py")):
        if f.name == "__init__.py":
            continue
        tree = ast.parse(f.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
            else:
                continue
            if not targets or not targets[0].id.isupper():
                continue
            src = ast.unparse(node.value) if node.value else ""
            if not any(ch.isdigit() for ch in src):
                continue
            mod_name = str(f.relative_to(ROOT / "src")).replace("/", ".")[:-3]
            name = targets[0].id
            try:
                mod = importlib.import_module(mod_name)
                obj = getattr(mod, name)
            except Exception:
                # Cannot import or resolve: report it rather than assume.
                rows.append({"module": mod_name + ".py", "name": name,
                             "definition": src[:200].replace("\n", " "),
                             "provenance_values_found": "UNRESOLVED"})
                continue
            n_vals = sum(1 for _ in _iter_values(obj))
            if n_vals:
                continue        # a parameter, not a definition
            rows.append({
                "module": mod_name + ".py",
                "name": name,
                "role": _role(obj),
                "definition": src[:200].replace("\n", " "),
                "provenance_values_found": "0",
            })
    return rows


def main() -> None:
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "registry"
    outdir.mkdir(parents=True, exist_ok=True)

    vals = collect_values()
    p1 = outdir / "parameter_registry.csv"
    with p1.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(vals[0].keys()))
        w.writeheader()
        w.writerows(vals)

    defs = collect_definitional()
    from collections import Counter as _C
    roles = _C(r["role"] for r in defs)
    p2 = outdir / "definitional_constants.csv"
    with p2.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(defs[0].keys()))
        w.writeheader()
        w.writerows(defs)

    from collections import Counter
    by_tag = Counter(r["tag"] for r in vals)
    n_evidenced = sum(1 for r in vals if r["doi"] or r["url"])
    n_sourced = sum(1 for r in vals if r["tag"] in ("SOURCED", "MEASURED"))
    n_point = sum(1 for r in vals
                  if r["uncertainty"].startswith("point estimate"))
    print(f"parameter_registry.csv      {len(vals):4} provenance-tracked values")
    n_real = sum(1 for r in defs
                 if r["role"] in ("numeric_constant", "numeric_table"))
    print(f"definitional_constants.csv  {len(defs):4} rows, of which "
          f"{n_real} are actual constants")
    print(f"  by role: {dict(roles)}")
    print(f"  by tag: {dict(by_tag)}")
    print(f"  SOURCED/MEASURED with a DOI or URL: {n_evidenced}/{n_sourced}")
    print(f"  point estimates with no uncertainty stated: {n_point}")


if __name__ == "__main__":
    main()
