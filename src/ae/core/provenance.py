"""Provenance tagging and uncertainty for every value in the platform.

Hard rule from the platform brief: every stored datapoint carries source, DOI or
URL, access date, source tier and extraction method, and every value is tagged
MEASURED / SOURCED / COMPUTED / DERIVED / ASSUMED with a confidence. Missing
values are recorded as missing and raise on use rather than silently defaulting.

The central type is :class:`Value`, a quantity bundled with its uncertainty and
its origin. Models accept ``Value`` (or a plain ``Quantity`` for intermediates)
and Monte Carlo sampling reads the distribution off the ``Value`` directly, so an
uncertain input cannot be used as though it were exact.

Design note on why ``ASSUMED`` values do not raise
--------------------------------------------------
An assumption is legitimate input to a scenario; an *undocumented* assumption is
not. ``Value`` therefore requires a ``basis`` string for every ASSUMED entry, so
an assumption without a stated reason fails construction rather than review.
"""

from __future__ import annotations

import datetime as _dt
import enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ae.core.units import UREG, Quantity

__all__ = [
    "Tag",
    "Tier",
    "MissingValueError",
    "Source",
    "Distribution",
    "Value",
    "MISSING",
]


class Tag(str, enum.Enum):
    """Origin of a value. Ordered from strongest to weakest evidence."""

    MEASURED = "MEASURED"    # measured on our own material, in a named lab
    SOURCED = "SOURCED"      # taken from an external source, cited
    COMPUTED = "COMPUTED"    # output of a model in this platform
    DERIVED = "DERIVED"      # algebraic transform of other values (e.g. oxide to element)
    ASSUMED = "ASSUMED"      # engineering judgement, requires a stated basis


class Tier(int, enum.Enum):
    """Source reliability tier from the platform brief."""

    T1 = 1  # peer-reviewed, curated databases, government data, standards bodies
    T2 = 2  # patents, supplier datasheets, industry reports, theses, filings, trade data
    T3 = 3  # news, blogs, forums: context only, never sole evidence
    NONE = 0  # no external source (COMPUTED / DERIVED / ASSUMED)


class MissingValueError(RuntimeError):
    """Raised when a value recorded as missing is used in a calculation."""


class Source(BaseModel):
    """Bibliographic and retrieval provenance for one datapoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    citation: str = Field(min_length=3, description="Author, year, title or publisher")
    tier: Tier
    doi: str | None = None
    url: str | None = None
    accessed: _dt.date | None = None
    extraction: Literal["manual", "llm_assisted", "api", "computed"] = "manual"
    extraction_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    note: str | None = None

    @model_validator(mode="after")
    def _external_sources_are_locatable(self) -> "Source":
        if self.tier in (Tier.T1, Tier.T2, Tier.T3):
            if not (self.doi or self.url):
                raise ValueError(
                    f"tier {self.tier.name} source must carry a DOI or URL so the claim "
                    f"can be checked: {self.citation!r}"
                )
            if self.accessed is None:
                raise ValueError(
                    f"tier {self.tier.name} source must carry an access date (policies, "
                    f"prices and web pages change): {self.citation!r}"
                )
        if self.extraction == "llm_assisted" and self.extraction_confidence is None:
            raise ValueError(
                "llm_assisted extraction must carry extraction_confidence so low-confidence "
                "rows can be flagged for human review"
            )
        return self


class Distribution(BaseModel):
    """Uncertainty on a value, as a sampleable distribution.

    ``kind="point"`` means no uncertainty is claimed, which is different from an
    uncertainty of zero: it is recorded so that a Sobol analysis can report the
    value as unswept rather than as certain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["point", "normal", "lognormal", "uniform", "triangular"]
    # Interpretation depends on kind; all in the same unit as the Value.
    loc: float | None = None      # normal mean / lognormal log-mean / triangular mode
    scale: float | None = None    # normal sd / lognormal log-sd
    low: float | None = None      # uniform / triangular lower bound
    high: float | None = None     # uniform / triangular upper bound

    @model_validator(mode="after")
    def _params_match_kind(self) -> "Distribution":
        need = {
            "point": (),
            "normal": ("loc", "scale"),
            "lognormal": ("loc", "scale"),
            "uniform": ("low", "high"),
            "triangular": ("low", "loc", "high"),
        }[self.kind]
        for f in need:
            if getattr(self, f) is None:
                raise ValueError(f"{self.kind} distribution requires {f!r}")
        if self.kind in ("normal", "lognormal") and self.scale is not None and self.scale < 0:
            raise ValueError("scale must be non-negative")
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"low ({self.low}) exceeds high ({self.high})")
        if self.kind == "triangular" and not (self.low <= self.loc <= self.high):  # type: ignore[operator]
            raise ValueError("triangular mode must lie within [low, high]")
        return self

    def sample(self, n: int, rng: np.random.Generator, point: float) -> np.ndarray:
        """Draw ``n`` samples. ``point`` is the Value's nominal magnitude."""
        if self.kind == "point":
            return np.full(n, point, dtype=float)
        if self.kind == "normal":
            return rng.normal(self.loc, self.scale, n)
        if self.kind == "lognormal":
            return rng.lognormal(self.loc, self.scale, n)
        if self.kind == "uniform":
            return rng.uniform(self.low, self.high, n)
        return rng.triangular(self.low, self.loc, self.high, n)

    @classmethod
    def relative(cls, frac: float) -> "Distribution":
        """A symmetric normal with standard deviation ``frac`` of the nominal.

        The nominal is supplied at sample time, so this is a shape without a
        location and is the common case for "plus or minus 20 percent" inputs.
        """
        if frac < 0:
            raise ValueError("relative uncertainty must be non-negative")
        return cls(kind="normal", loc=0.0, scale=frac)


class Value(BaseModel):
    """A quantity with its uncertainty and its provenance.

    Examples
    --------
    >>> from ae.core.units import Q_
    >>> import datetime as dt
    >>> v = Value(
    ...     quantity=Q_(30.0, "ppm_mass"),
    ...     tag=Tag.SOURCED,
    ...     source=Source(citation="Muller et al. 2012, Quartz: Deposits, Mineralogy and Analytics",
    ...                   tier=Tier.T1, url="https://doi.org/10.1007/978-3-642-22161-3",
    ...                   accessed=dt.date(2026, 9, 16)),
    ... )
    >>> v.magnitude
    30.0
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    quantity: Quantity
    tag: Tag
    source: Source | None = None
    dist: Distribution = Field(default_factory=lambda: Distribution(kind="point"))
    basis: str | None = Field(
        default=None,
        description="Why this value was chosen. Required for ASSUMED and DERIVED.",
    )
    confidence: Literal["high", "medium", "low"] | None = None

    @field_validator("quantity", mode="before")
    @classmethod
    def _coerce_quantity(cls, v: Any) -> Any:
        if isinstance(v, str):
            return UREG.Quantity(v)
        return v

    @model_validator(mode="after")
    def _provenance_matches_tag(self) -> "Value":
        if self.tag in (Tag.MEASURED, Tag.SOURCED) and self.source is None:
            raise ValueError(
                f"{self.tag.value} value must carry a Source; an uncited measurement or "
                f"literature value is indistinguishable from an assumption"
            )
        if self.tag in (Tag.ASSUMED, Tag.DERIVED) and not self.basis:
            raise ValueError(
                f"{self.tag.value} value must state a basis, so an undocumented "
                f"assumption fails construction rather than review"
            )
        if self.tag == Tag.SOURCED and self.source is not None and self.source.tier == Tier.T3:
            if not self.basis:
                raise ValueError(
                    "tier 3 source may never be sole evidence; state in basis what "
                    "corroborates it"
                )
        return self

    @property
    def magnitude(self) -> float:
        return float(self.quantity.magnitude)

    @property
    def units(self) -> str:
        return str(self.quantity.units)

    def to(self, unit: str) -> Quantity:
        return self.quantity.to(unit)

    def sample(self, n: int, rng: np.random.Generator) -> Quantity:
        """Draw ``n`` samples as a Quantity array in this Value's own unit.

        A ``relative`` distribution is interpreted as a multiplicative factor on
        the nominal, so the returned samples are centred on the nominal value.
        """
        base = self.magnitude
        if self.dist.kind == "normal" and self.dist.loc == 0.0:
            draws = base * (1.0 + self.dist.sample(n, rng, 0.0))
        else:
            draws = self.dist.sample(n, rng, base)
        return UREG.Quantity(draws, self.quantity.units)

    def __str__(self) -> str:
        tail = f" [{self.tag.value}]"
        if self.source:
            tail += f" ({self.source.citation}, T{self.source.tier.value})"
        return f"{self.quantity:~P}{tail}"


class _Missing:
    """Sentinel for a value that is genuinely unknown.

    Using it in arithmetic raises :class:`MissingValueError` rather than
    propagating a NaN, because a NaN that reaches a report looks like a number
    that failed to compute rather than a measurement that was never made.
    """

    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "MISSING"

    def _raise(self, *_: Any, **__: Any) -> Any:
        raise MissingValueError(
            "this quantity is recorded as MISSING (never measured or sourced) and "
            "cannot be used in a calculation; measure it, source it, or replace it "
            "with an explicitly tagged ASSUMED Value carrying a basis"
        )

    __add__ = __radd__ = __sub__ = __rsub__ = _raise
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _raise
    __float__ = __lt__ = __gt__ = __le__ = __ge__ = _raise


#: Singleton sentinel for unmeasured quantities.
MISSING = _Missing()
