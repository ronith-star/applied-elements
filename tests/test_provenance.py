"""Provenance and uncertainty contract tests."""
import datetime as dt
import numpy as np
import pytest
from ae.core.units import Q_
from ae.core.provenance import (Tag, Tier, Source, Distribution, Value, MISSING,
                                MissingValueError)

T1 = Source(citation="Muller et al. 2012, Quartz: Deposits, Mineralogy and Analytics",
            tier=Tier.T1, doi="10.1007/978-3-642-22161-3", accessed=dt.date(2026, 9, 16))


def test_sourced_value_roundtrips():
    v = Value(quantity=Q_(30.0, "ppm_mass"), tag=Tag.SOURCED, source=T1)
    assert v.magnitude == pytest.approx(30.0)
    assert v.to("ug/g").magnitude == pytest.approx(30.0)


def test_measured_and_sourced_require_a_source():
    for tag in (Tag.MEASURED, Tag.SOURCED):
        with pytest.raises(ValueError, match="must carry a Source"):
            Value(quantity=Q_(1.0, "kg"), tag=tag)


def test_assumed_and_derived_require_a_basis():
    for tag in (Tag.ASSUMED, Tag.DERIVED):
        with pytest.raises(ValueError, match="must state a basis"):
            Value(quantity=Q_(1.0, "kg"), tag=tag)
    ok = Value(quantity=Q_(1.0, "kg"), tag=Tag.ASSUMED,
               basis="vendor-class estimate pending quotation")
    assert ok.basis


def test_external_source_must_be_locatable_and_dated():
    with pytest.raises(ValueError, match="DOI or URL"):
        Source(citation="Some paper 2024", tier=Tier.T1, accessed=dt.date(2026, 1, 1))
    with pytest.raises(ValueError, match="access date"):
        Source(citation="Some paper 2024", tier=Tier.T1, doi="10.1000/x")


def test_tier3_cannot_be_sole_evidence():
    t3 = Source(citation="Trade press item", tier=Tier.T3, url="https://example.org/x",
                accessed=dt.date(2026, 9, 16))
    with pytest.raises(ValueError, match="never be sole evidence"):
        Value(quantity=Q_(1.0, "USD/kg"), tag=Tag.SOURCED, source=t3)
    ok = Value(quantity=Q_(1.0, "USD/kg"), tag=Tag.SOURCED, source=t3,
               basis="corroborated by UN Comtrade unit value for the same HS code")
    assert ok.basis


def test_llm_extraction_requires_confidence():
    with pytest.raises(ValueError, match="extraction_confidence"):
        Source(citation="Paper", tier=Tier.T1, doi="10.1/x",
               accessed=dt.date(2026, 9, 16), extraction="llm_assisted")


def test_missing_raises_on_use_not_nan():
    """A NaN in a report reads as a failed computation; MISSING reads as unmeasured."""
    assert not MISSING
    for op in (lambda: MISSING + 1, lambda: 1 + MISSING, lambda: MISSING * 2.0,
               lambda: float(MISSING), lambda: MISSING < 1):
        with pytest.raises(MissingValueError):
            op()


@pytest.mark.parametrize("kind,missing", [
    ("normal", "scale"), ("lognormal", "scale"), ("uniform", "high"), ("triangular", "loc"),
])
def test_distribution_params_validated(kind, missing):
    kw = {"normal": dict(loc=1.0, scale=0.1), "lognormal": dict(loc=0.0, scale=0.1),
          "uniform": dict(low=0.0, high=1.0),
          "triangular": dict(low=0.0, loc=0.5, high=1.0)}[kind]
    kw.pop(missing)
    with pytest.raises(ValueError, match=missing):
        Distribution(kind=kind, **kw)


def test_triangular_mode_must_be_inside_bounds():
    with pytest.raises(ValueError, match="within"):
        Distribution(kind="triangular", low=0.0, loc=2.0, high=1.0)


def test_point_distribution_samples_constant():
    v = Value(quantity=Q_(2650.0, "kg/m**3"), tag=Tag.SOURCED, source=T1)
    s = v.sample(1000, np.random.default_rng(0))
    assert np.allclose(s.magnitude, 2650.0)
    assert str(s.units) == "kilogram / meter ** 3"


def test_relative_uncertainty_centres_on_nominal():
    v = Value(quantity=Q_(100.0, "USD/tonne"), tag=Tag.ASSUMED,
              basis="vendor-class estimate, plus or minus 20 percent",
              dist=Distribution.relative(0.20))
    s = v.sample(40000, np.random.default_rng(1)).magnitude
    assert s.mean() == pytest.approx(100.0, rel=0.01)
    assert s.std() == pytest.approx(20.0, rel=0.05)


def test_uniform_sampling_respects_bounds():
    v = Value(quantity=Q_(0.5, "dimensionless"), tag=Tag.ASSUMED, basis="scenario range",
              dist=Distribution(kind="uniform", low=0.3, high=0.9))
    s = v.sample(5000, np.random.default_rng(2)).magnitude
    assert s.min() >= 0.3 and s.max() <= 0.9


def test_value_is_frozen():
    v = Value(quantity=Q_(1.0, "kg"), tag=Tag.SOURCED, source=T1)
    with pytest.raises(Exception):
        v.quantity = Q_(2.0, "kg")
