"""Registry tests: exact lookup, unit discipline, coverage honesty, round-trip."""
import datetime as dt
import numpy as np
import pytest
from ae.core.units import Q_, DimensionalityError
from ae.core.provenance import Tag, Tier, Source, Distribution, Value
from ae.core.registry import Registry, ParameterNotFound, DuplicateParameter

SRC = Source(citation="Bond 1952, The Third Theory of Comminution", tier=Tier.T2,
             url="https://www.onemine.org/documents/the-third-theory-of-comminution",
             accessed=dt.date(2026, 9, 16))


def reg_with(**kw):
    r = Registry()
    r.add("comminution.work_index.quartzite",
          Value(quantity=Q_(12.18, "kWh/ton"), tag=Tag.SOURCED, source=SRC))
    r.add("comminution.work_index.vein_quartz",
          Value(quantity=Q_(13.57, "kWh/ton"), tag=Tag.SOURCED, source=SRC))
    r.add("leach.temperature",
          Value(quantity=Q_(200.0, "degC"), tag=Tag.ASSUMED,
                basis="mid-range of the 160 to 250 C band reported for mixed-acid leaching",
                dist=Distribution(kind="uniform", low=160.0, high=250.0)))
    return r


def test_exact_lookup_and_unit_conversion():
    r = reg_with()
    assert r.magnitude("comminution.work_index.quartzite", "kWh/ton") == pytest.approx(12.18)
    assert len(r) == 3
    assert "leach.temperature" in r


def test_missing_key_suggests_near_misses():
    r = reg_with()
    with pytest.raises(ParameterNotFound, match="Did you mean"):
        r.get("comminution.work_index.quartzit")


def test_missing_key_without_near_miss_still_raises():
    r = reg_with()
    with pytest.raises(ParameterNotFound, match="not registered"):
        r.get("totally.unrelated.key")


def test_no_fuzzy_resolution():
    """A typo must FAIL, not resolve to a neighbour."""
    r = reg_with()
    with pytest.raises(ParameterNotFound):
        r.get("comminution.work_index.quartzit")


def test_duplicate_registration_blocked():
    r = reg_with()
    v = Value(quantity=Q_(1.0, "kWh/ton"), tag=Tag.ASSUMED, basis="test")
    with pytest.raises(DuplicateParameter, match="overwrite=True"):
        r.add("comminution.work_index.quartzite", v)
    r.add("comminution.work_index.quartzite", v, overwrite=True)
    assert r.magnitude("comminution.work_index.quartzite", "kWh/ton") == pytest.approx(1.0)


def test_magnitude_requires_explicit_unit():
    """Reading a magnitude without naming the unit is how a kWh/short-ton work
    index silently becomes a kWh/tonne one."""
    r = reg_with()
    with pytest.raises(TypeError):
        r.magnitude("comminution.work_index.quartzite")  # type: ignore[call-arg]


def test_wrong_unit_raises_not_coerces():
    r = reg_with()
    with pytest.raises(DimensionalityError):
        r.magnitude("comminution.work_index.quartzite", "kg")


def test_find_by_prefix():
    r = reg_with()
    hits = r.find("comminution.work_index")
    assert set(hits) == {"comminution.work_index.quartzite", "comminution.work_index.vein_quartz"}


def test_uncertain_keys_are_the_monte_carlo_inputs():
    r = reg_with()
    assert r.uncertain_keys() == ["leach.temperature"]


def test_sampling_respects_registered_distribution():
    r = reg_with()
    s = r.sample(["leach.temperature"], 5000, np.random.default_rng(0))
    m = s["leach.temperature"].magnitude
    assert m.min() >= 160.0 and m.max() <= 250.0


def test_coverage_reports_assumption_load():
    r = reg_with()
    c = r.coverage()
    assert c["n_parameters"] == 3
    assert c["by_tag"] == {"SOURCED": 2, "ASSUMED": 1}
    assert c["by_tier"]["T2"] == 2
    assert c["by_tier"]["NO_SOURCE"] == 1
    assert c["assumed_keys"] == ["leach.temperature"]
    assert c["assumed_fraction"] == pytest.approx(1 / 3)
    # Point estimates are invisible to sensitivity analysis, so they are listed.
    assert set(c["point_estimate_keys"]) == {"comminution.work_index.quartzite",
                                             "comminution.work_index.vein_quartz"}


def test_records_flatten_provenance():
    r = reg_with()
    rows = {x["key"]: x for x in r.to_records()}
    row = rows["comminution.work_index.quartzite"]
    assert row["tag"] == "SOURCED" and row["tier"] == 2
    assert row["url"].startswith("https://")
    assert row["accessed"] == "2026-09-16"
    leach = rows["leach.temperature"]
    assert leach["dist_kind"] == "uniform" and leach["dist_low"] == 160.0
    assert "mid-range" in leach["basis"]


def test_roundtrip_preserves_units_tags_and_sources(tmp_path):
    r = reg_with()
    p = r.save(tmp_path / "reg.json")
    back = Registry.load(p)
    assert len(back) == len(r)
    for k in r:
        a, b = r.get(k), back.get(k)
        assert a.quantity.magnitude == pytest.approx(b.quantity.magnitude)
        assert str(a.quantity.units) == str(b.quantity.units)
        assert a.tag == b.tag
        assert a.dist.model_dump() == b.dist.model_dump()
        assert (a.source is None) == (b.source is None)
        if a.source:
            assert a.source.citation == b.source.citation
            assert a.source.tier == b.source.tier
            assert a.source.accessed == b.source.accessed


def test_key_format_validated():
    r = Registry()
    v = Value(quantity=Q_(1.0, "kg"), tag=Tag.ASSUMED, basis="x")
    for bad in ["", "has space"]:
        with pytest.raises(ValueError, match="dotted path"):
            r.add(bad, v)
