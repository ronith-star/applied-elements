"""Queryable registry of every parameter the platform uses.

One place holds every number, its unit, its uncertainty, its origin tag and its
source. Models look parameters up by key rather than embedding literals, so that

* the provenance of any result can be traced to the values that produced it,
* a coverage report can count how much of the model rests on ASSUMED values,
* a Monte Carlo run can sweep every uncertain parameter without hunting for them,
* and re-sourcing a value updates every model at once.

Registry keys are dotted paths: ``comminution.work_index.quartzite``,
``site.US-NM-ABQ.power.energy_price``. Lookups are exact; there is deliberately
no fuzzy matching, because silently resolving a typo to a neighbouring parameter
is worse than failing.

Coverage reporting
------------------
:meth:`Registry.coverage` returns the count of parameters by tag and by source
tier. The brief requires honesty about how much of a model is assumption, and a
number is easier to argue with than an adjective.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ae.core.provenance import Distribution, Source, Tag, Tier, Value
from ae.core.units import UREG, Quantity

__all__ = ["DuplicateParameter", "ParameterNotFound", "Registry"]


class ParameterNotFound(KeyError):
    """Raised when a key is absent. Carries near-miss keys to catch typos."""


class DuplicateParameter(ValueError):
    """Raised on a second registration of the same key without ``overwrite=True``."""


class Registry:
    """A mutable, serializable collection of provenance-tagged parameters."""

    def __init__(self) -> None:
        self._values: dict[str, Value] = {}

    # --- registration -------------------------------------------------------
    def add(self, key: str, value: Value, overwrite: bool = False) -> Value:
        """Register ``value`` under ``key``."""
        if not key or " " in key:
            raise ValueError(f"key must be a non-empty dotted path without spaces: {key!r}")
        if key in self._values and not overwrite:
            raise DuplicateParameter(
                f"{key!r} is already registered as {self._values[key]}; pass "
                f"overwrite=True to replace it deliberately"
            )
        self._values[key] = value
        return value

    def add_many(self, mapping: dict[str, Value], overwrite: bool = False) -> None:
        for k, v in mapping.items():
            self.add(k, v, overwrite=overwrite)

    # --- lookup -------------------------------------------------------------
    def __contains__(self, key: object) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def get(self, key: str) -> Value:
        """Fetch a Value, raising with near-miss suggestions if absent."""
        try:
            return self._values[key]
        except KeyError:
            near = [k for k in self._values if _close(k, key)]
            hint = f" Did you mean: {sorted(near)[:5]}?" if near else ""
            raise ParameterNotFound(
                f"{key!r} is not registered ({len(self._values)} parameters known).{hint}"
            ) from None

    def quantity(self, key: str, unit: str | None = None) -> Quantity:
        """Fetch a parameter as a Quantity, optionally converted to ``unit``."""
        q = self.get(key).quantity
        return q.to(unit) if unit else q

    def magnitude(self, key: str, unit: str) -> float:
        """Fetch a parameter's magnitude in an EXPLICIT unit.

        The unit is required rather than optional: reading a magnitude without
        naming the unit is how a kWh/short-ton work index becomes a kWh/tonne
        one.
        """
        return float(self.get(key).quantity.to(unit).magnitude)

    def find(self, prefix: str) -> dict[str, Value]:
        """All parameters whose key starts with ``prefix``."""
        return {k: v for k, v in sorted(self._values.items()) if k.startswith(prefix)}

    # --- uncertainty --------------------------------------------------------
    def uncertain_keys(self) -> list[str]:
        """Keys carrying a non-point distribution, i.e. the Monte Carlo inputs."""
        return sorted(k for k, v in self._values.items() if v.dist.kind != "point")

    def sample(self, keys: list[str], n: int, rng: np.random.Generator) -> dict[str, Quantity]:
        """Draw ``n`` correlated-free samples for each requested key."""
        return {k: self.get(k).sample(n, rng) for k in keys}

    # --- reporting ----------------------------------------------------------
    def coverage(self) -> dict[str, Any]:
        """Count parameters by tag and source tier, and list the weak spots.

        Returns a dict with exactly these keys: ``n_parameters``, ``by_tag``,
        ``by_tier``, ``assumed_keys``, ``assumed_fraction``,
        ``point_estimate_keys`` and ``tier3_keys``.

        ``point_estimate_keys`` matters because a parameter with no uncertainty
        is excluded from every sensitivity result, which can make a model look
        more robust than it is. ``tier3_keys`` matters because a tier 3 source
        may never be sole evidence.
        """
        by_tag = Counter(v.tag.value for v in self._values.values())
        by_tier = Counter(
            (v.source.tier.name if v.source else "NO_SOURCE") for v in self._values.values()
        )
        return {
            "n_parameters": len(self._values),
            "by_tag": dict(by_tag),
            "by_tier": dict(by_tier),
            "assumed_keys": sorted(k for k, v in self._values.items() if v.tag == Tag.ASSUMED),
            "assumed_fraction": (by_tag[Tag.ASSUMED.value] / len(self._values)
                                 if self._values else 0.0),
            "point_estimate_keys": sorted(
                k for k, v in self._values.items() if v.dist.kind == "point"
            ),
            "tier3_keys": sorted(
                k for k, v in self._values.items()
                if v.source is not None and v.source.tier == Tier.T3
            ),
        }

    def to_records(self) -> list[dict[str, Any]]:
        """Flatten to rows for a CSV or a DataFrame, one parameter per row."""
        rows: list[dict[str, Any]] = []
        for k, v in sorted(self._values.items()):
            s = v.source
            rows.append({
                "key": k,
                "magnitude": v.magnitude,
                "unit": v.units,
                "tag": v.tag.value,
                "confidence": v.confidence,
                "basis": v.basis,
                "dist_kind": v.dist.kind,
                "dist_loc": v.dist.loc,
                "dist_scale": v.dist.scale,
                "dist_low": v.dist.low,
                "dist_high": v.dist.high,
                "citation": s.citation if s else None,
                "tier": s.tier.value if s else None,
                "doi": s.doi if s else None,
                "url": s.url if s else None,
                "accessed": s.accessed.isoformat() if s and s.accessed else None,
                "extraction": s.extraction if s else None,
                "extraction_confidence": s.extraction_confidence if s else None,
            })
        return rows

    # --- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Write the registry to JSON, preserving units as strings."""
        payload = {
            k: {
                "magnitude": v.magnitude,
                "unit": v.units,
                "tag": v.tag.value,
                "basis": v.basis,
                "confidence": v.confidence,
                "dist": v.dist.model_dump(),
                "source": (v.source.model_dump(mode="json") if v.source else None),
            }
            for k, v in sorted(self._values.items())
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: str | Path) -> Registry:
        """Read a registry back, reconstructing units and provenance."""
        raw = json.loads(Path(path).read_text())
        reg = cls()
        for k, d in raw.items():
            src = Source(**d["source"]) if d.get("source") else None
            reg.add(k, Value(
                quantity=UREG.Quantity(d["magnitude"], d["unit"]),
                tag=Tag(d["tag"]),
                source=src,
                dist=Distribution(**d["dist"]),
                basis=d.get("basis"),
                confidence=d.get("confidence"),
            ))
        return reg


def _close(a: str, b: str) -> bool:
    """Cheap near-miss test for key suggestions: shared prefix or one edit apart."""
    if a == b:
        return False
    if a.split(".")[:-1] == b.split(".")[:-1]:
        return True
    if abs(len(a) - len(b)) <= 1:
        diffs = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        return diffs <= 2
    return False
