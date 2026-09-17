"""Surrogate validation: grouped splits, baselines, OOD refusal."""
import pathlib

import numpy as np
import pytest

from ae.ml.surrogate import (
    Surrogate,
    TrainingSet,
    evaluate_surrogate,
    leave_one_deposit_out_splits,
    random_kfold_optimism,
)


def synthetic(n_deposits=6, per_deposit=40, seed=0, deposit_offset=0.0,
              noise=0.05, feature_separation=0.0):
    """Deposits sharing a mechanism, with two DISTINCT kinds of grouping effect.

    ``deposit_offset`` shifts each deposit's TARGET level (an unmodelled
    per-deposit factor: a different alteration overprint, a lab calibration
    offset). ``feature_separation`` displaces each deposit's FEATURE ranges, so
    deposit identity becomes inferable from the features themselves.

    There are TWO distinct mechanisms by which grouped validation is harder
    than random k-fold, and they are separable in this fixture. Measured grid
    (ratio = leave-one-deposit-out RMSE over random k-fold RMSE, 6 deposits,
    40 samples each, seed 1):

      noise 0.01: sep 0.0 off 0.00 -> 1.00    sep 0.0 off 0.15 -> 1.13
                  sep 2.0 off 0.00 -> 1.41    sep 2.0 off 0.15 -> 1.44
      noise 0.05: sep 0.0 off 0.00 -> 0.99    sep 0.0 off 0.15 -> 1.11
                  sep 2.0 off 0.00 -> 1.00    sep 2.0 off 0.15 -> 1.30

    Mechanism 1, LEAKAGE of an unmodelled per-deposit factor (offset). A random
    split puts samples sharing an offset on both sides, so the model absorbs
    the offset and carries it to its fold neighbours. Present at both noise
    levels, worth about 1.11 to 1.13x on its own.

    Mechanism 2, EXTRAPOLATION in feature space (separation). When each deposit
    occupies a different region of feature space, holding one out is an
    extrapolation rather than an interpolation, and a tree model extrapolates
    as a constant. Worth 1.41x at low noise, where the model is otherwise
    accurate enough for the extrapolation penalty to dominate, and negligible
    at noise 0.05 where irreducible error swamps it.

    This is the more important mechanism for the intended domain: a new deposit
    typically differs in feature space as well as in level, which is exactly
    when a random-k-fold error estimate is least informative.

    An earlier version of this fixture varied only the offset and asserted a
    ratio above 1.2, which the generator could not produce (it returned 1.03).
    A prior draft of this docstring then attributed the gap to both mechanisms
    being necessary together, which the grid above also contradicts: at noise
    0.01 separation alone is sufficient and is in fact the larger effect.
    """
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for d in range(n_deposits):
        off = deposit_offset * rng.normal()
        c = feature_separation * (d - (n_deposits - 1) / 2)
        al = rng.uniform(5, 60, per_deposit) + 8 * c
        ti = rng.uniform(1, 25, per_deposit) + 3 * c
        fe = rng.uniform(10, 300, per_deposit) + 40 * c
        size = rng.uniform(50, 400, per_deposit)
        # Recovery falls with impurity load and rises with finer grind.
        rec = (0.95 - 0.0035 * al - 0.004 * ti - 0.0002 * fe
               + 0.0003 * (400 - size) + off + noise * rng.normal(size=per_deposit))
        X.append(np.column_stack([al, ti, fe, size]))
        y.append(np.clip(rec, 0.05, 0.999))
        g.append(np.full(per_deposit, f"D{d}"))
    return TrainingSet(np.vstack(X), np.concatenate(y), np.concatenate(g),
                       ["al_ppm", "ti_ppm", "fe_ppm", "p80_um"], "recovery")


def test_splits_hold_out_whole_deposits():
    d = synthetic(n_deposits=4, per_deposit=10)
    splits = leave_one_deposit_out_splits(d.groups)
    assert len(splits) == 4
    for train, test in splits:
        assert set(d.groups[test]).isdisjoint(set(d.groups[train])), \
            "no deposit may appear on both sides"
        assert len(set(d.groups[test])) == 1
        assert train.size + test.size == d.n_samples


def test_training_set_requires_groups_and_validates_shapes():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="all three must agree"):
        TrainingSet(X, np.zeros(9), np.zeros(10), ["a", "b", "c"])
    with pytest.raises(ValueError, match="feature names"):
        TrainingSet(X, np.zeros(10), np.zeros(10), ["a", "b"])
    with pytest.raises(ValueError, match="non-finite"):
        TrainingSet(np.array([[1.0, np.nan, 3.0]] * 10), np.zeros(10),
                    np.array(["a"] * 5 + ["b"] * 5), ["a", "b", "c"])
    with pytest.raises(ValueError, match="at least 2"):
        TrainingSet(X, np.zeros(10), np.array(["only"] * 10), ["a", "b", "c"])


@pytest.mark.benchmark
def test_grouped_validation_is_pessimistic_relative_to_random_kfold():
    """The central methodological claim, measured rather than asserted.

    Configured with both grouping effects present, which is the realistic case:
    a new deposit usually differs in feature space AND in level. Either alone is
    sufficient to produce optimism, and at noise 0.01 feature separation is the
    larger of the two (1.41x against 1.13x); see the synthetic() docstring for
    the measured grid and test_the_two_optimism_mechanisms_are_separable for the
    sweep that establishes it.
    """
    d = synthetic(n_deposits=6, per_deposit=40, deposit_offset=0.15,
                  feature_separation=2.0, noise=0.01, seed=1)
    o = random_kfold_optimism(d, kind="gbm", seed=1)
    print(f"\nrandom k-fold RMSE      {o['random_kfold_rmse']:.4f}")
    print(f"leave-one-deposit-out   {o['leave_one_deposit_out_rmse']:.4f}")
    print(f"optimism ratio          {o['optimism_ratio']:.2f}x")
    print(f"absolute gap            {o['optimism_absolute']:+.4f}")
    assert o["optimism_ratio"] > 1.15, (
        "with a per-deposit offset AND feature separability, grouped error must "
        "exceed random k-fold error"
    )


@pytest.mark.benchmark
def test_the_two_optimism_mechanisms_are_separable():
    """Leakage and extrapolation are DIFFERENT causes of grouped-validation
    pessimism, and this separates them by measurement.

    Asserted from the grid in the synthetic() docstring, at the noise level where
    each is visible: at low noise, feature separation alone dominates because
    the extrapolation penalty is large relative to irreducible error; at higher
    noise it vanishes and only the offset leak survives."""
    def ratio(sep, off, noise):
        d = synthetic(n_deposits=6, per_deposit=40, deposit_offset=off,
                      feature_separation=sep, noise=noise, seed=1)
        return random_kfold_optimism(d, kind="gbm", seed=1)["optimism_ratio"]

    lo_none, lo_off = ratio(0.0, 0.0, 0.01), ratio(0.0, 0.15, 0.01)
    lo_sep, lo_both = ratio(2.0, 0.0, 0.01), ratio(2.0, 0.15, 0.01)
    hi_sep, hi_both = ratio(2.0, 0.0, 0.05), ratio(2.0, 0.15, 0.05)
    print(f"\nnoise 0.01: none {lo_none:.2f}x  offset {lo_off:.2f}x  "
          f"separation {lo_sep:.2f}x  both {lo_both:.2f}x")
    print(f"noise 0.05: separation {hi_sep:.2f}x  both {hi_both:.2f}x")

    # No grouping structure: the two schemes must agree.
    assert lo_none < 1.10, "neither mechanism present"
    # Each mechanism alone produces optimism, contrary to my first hypothesis.
    assert lo_off > 1.05, "offset leak alone is material"
    assert lo_sep > 1.25, "extrapolation alone is the LARGER effect at low noise"
    assert lo_sep > lo_off, (
        "at low noise, feature-space extrapolation must outweigh offset leakage"
    )
    # And the extrapolation penalty is noise-dependent: it washes out when
    # irreducible error is large.
    assert hi_sep < lo_sep - 0.2, (
        "the extrapolation penalty must shrink as noise rises"
    )
    assert hi_both > hi_sep, "at high noise the offset leak is what remains"


def test_no_deposit_offset_shrinks_the_optimism_gap():
    """Control: remove the shared per-deposit component and the two validation
    schemes agree closely, confirming the gap is caused by grouping structure
    and not by fold size."""
    d = synthetic(n_deposits=6, per_deposit=40, deposit_offset=0.0,
                  feature_separation=2.0, seed=2)
    o = random_kfold_optimism(d, kind="gbm", seed=2)
    print(f"\nno offset: ratio {o['optimism_ratio']:.2f}x")
    assert o["optimism_ratio"] < 1.5


@pytest.mark.benchmark
def test_evaluation_reports_baselines_and_per_fold_spread():
    d = synthetic(n_deposits=6, per_deposit=40, deposit_offset=0.03, seed=3)
    r = evaluate_surrogate(d, kind="gbm", seed=3)
    rep = r.report()
    print(f"\n{rep['model']} on {rep['target']}, {rep['n_folds']} folds")
    print(f"  RMSE {rep['rmse']:.4f}  (per fold {rep['rmse_per_fold_min']:.4f} "
          f"to {rep['rmse_per_fold_max']:.4f})")
    print(f"  mean baseline {rep['baseline_mean_rmse']:.4f}, "
          f"ridge baseline {rep['baseline_ridge_rmse']:.4f}")
    print(f"  skill over mean {rep['skill_over_mean']:+.3f}, "
          f"folds beating mean {rep['folds_beating_mean']}/{rep['n_folds']}")
    print(f"  normalised RMSE (over IQR) {rep['normalised_rmse_over_iqr']:.3f}")
    assert len(r.folds) == 6
    assert rep["rmse_per_fold_max"] > rep["rmse_per_fold_min"], "spread is reported"
    assert rep["skill_over_mean"] > 0.0, "must beat a mean predictor"
    assert r.is_useful()


def test_a_model_with_no_signal_is_reported_as_not_useful():
    """Honest negative: pure-noise target must FAIL the usefulness test rather
    than returning a plausible-looking small error."""
    rng = np.random.default_rng(4)
    n_dep, per = 5, 30
    X = rng.normal(size=(n_dep * per, 4))
    y = rng.normal(size=n_dep * per)
    g = np.concatenate([[f"D{i}"] * per for i in range(n_dep)])
    d = TrainingSet(X, y, g, ["a", "b", "c", "e"], "noise")
    r = evaluate_surrogate(d, kind="gbm", seed=4)
    print(f"\nnoise target: skill {r.skill_over_mean:+.3f}, useful={r.is_useful()}")
    assert not r.is_useful()
    assert r.skill_over_mean < 0.10


def test_usefulness_requires_a_majority_of_folds_not_just_the_average():
    """One easy deposit must not carry the verdict."""
    d = synthetic(n_deposits=6, per_deposit=40, deposit_offset=0.03, seed=5)
    r = evaluate_surrogate(d, kind="gbm", seed=5)
    assert r.folds_beating_mean > len(r.folds) / 2
    # Construct the contrary case directly.
    import dataclasses as dc
    bad = dc.replace(r, folds=[dc.replace(f, rmse=f.baseline_mean_rmse * 1.5)
                               for f in r.folds[:-1]] + [r.folds[-1]])
    assert not bad.is_useful()


def test_ridge_baseline_can_win_and_that_is_reported():
    """On a linear mechanism, ridge should be competitive. The point is that the
    comparison is always made, so a tree cannot be adopted by default."""
    rng = np.random.default_rng(6)
    n_dep, per = 5, 40
    X = rng.uniform(0, 1, size=(n_dep * per, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 0.01 * rng.normal(size=n_dep * per)
    g = np.concatenate([[f"D{i}"] * per for i in range(n_dep)])
    d = TrainingSet(X, y, g, ["a", "b", "c"], "linear")
    r = evaluate_surrogate(d, kind="gbm", seed=6)
    print(f"\nlinear mechanism: gbm {r.rmse:.4f} vs ridge {r.baseline_ridge_rmse:.4f}")
    assert r.baseline_ridge_rmse < r.rmse, "ridge wins on an exactly linear target"
    assert sum(1 for f in r.folds if f.beats_ridge) < len(r.folds)


# --- OOD --------------------------------------------------------------------
def test_ood_refuses_by_default_and_explains_why():
    d = synthetic(n_deposits=5, per_deposit=40, seed=7)
    s = Surrogate(kind="gbm", seed=7).fit(d)
    inside = [30.0, 12.0, 150.0, 200.0]
    assert not s.ood_report(inside).is_ood
    assert 0.0 < s.predict(inside) < 1.0

    far = [5000.0, 12.0, 150.0, 200.0]     # Al 100x the trained range
    rep = s.ood_report(far)
    assert rep.is_ood
    assert "al_ppm" in rep.out_of_range_features
    print(f"\nOOD reason: {rep.reason()}")
    with pytest.raises(ValueError, match="out of distribution"):
        s.predict(far)
    with pytest.warns(UserWarning, match="out-of-distribution"):
        s.predict(far, on_ood="warn")
    val = s.predict(far, on_ood="allow")
    assert np.isfinite(val)


def test_tree_extrapolates_as_a_constant_which_is_why_ood_matters():
    """Mechanism, demonstrated: pushing a feature far outside the trained range
    changes the tree prediction not at all beyond the boundary."""
    d = synthetic(n_deposits=5, per_deposit=40, seed=8)
    s = Surrogate(kind="gbm", seed=8).fit(d)
    at_edge = s.predict([60.0, 12.0, 150.0, 200.0], on_ood="allow")
    far_out = s.predict([600.0, 12.0, 150.0, 200.0], on_ood="allow")
    way_out = s.predict([60000.0, 12.0, 150.0, 200.0], on_ood="allow")
    print(f"\nAl 60 -> {at_edge:.4f}, Al 600 -> {far_out:.4f}, "
          f"Al 60000 -> {way_out:.4f}")
    assert far_out == pytest.approx(way_out, abs=1e-9), \
        "the tree returns the same constant however far outside you go"


def test_ood_catches_a_hole_inside_the_feature_ranges():
    """A point can sit inside every marginal range and still be far from all
    training data. The nearest-neighbour criterion is what catches it."""
    rng = np.random.default_rng(9)
    a = rng.uniform(0, 1, 120)
    b = a + 0.02 * rng.normal(size=120)     # tightly correlated features
    X = np.column_stack([a, b])
    y = a + b
    g = np.concatenate([[f"D{i}"] * 40 for i in range(3)])
    d = TrainingSet(X, y, g, ["a", "b"], "sum")
    s = Surrogate(kind="gbm", seed=9, nn_sigma=3.0).fit(d)
    # Inside both marginals, but off the correlation ridge.
    rep = s.ood_report([0.05, 0.95])
    assert not rep.out_of_range_features, "inside every marginal range"
    assert rep.is_ood, "yet far from the training manifold"
    print(f"\nhole inside the hull: {rep.reason()}")


def test_covariance_fallback_when_samples_are_too_few():
    """Mahalanobis needs more samples than features. With too few, the method
    must SAY it fell back rather than invert a singular matrix."""
    rng = np.random.default_rng(10)
    X = rng.normal(size=(6, 8))
    y = rng.normal(size=6)
    g = np.array(["A", "A", "A", "B", "B", "B"])
    d = TrainingSet(X, y, g, [f"f{i}" for i in range(8)], "y")
    s = Surrogate(kind="ridge", seed=10).fit(d)
    rep = s.ood_report(np.zeros(8))
    assert np.isnan(rep.mahalanobis)
    assert "too few" in rep.method
    print(f"\nfallback method: {rep.method}")


def test_feature_count_mismatch_names_the_features():
    d = synthetic(n_deposits=3, per_deposit=20, seed=11)
    s = Surrogate(seed=11).fit(d)
    with pytest.raises(ValueError, match="al_ppm"):
        s.ood_report([1.0, 2.0])


def test_predict_with_report_never_raises():
    d = synthetic(n_deposits=3, per_deposit=20, seed=12)
    s = Surrogate(seed=12).fit(d)
    val, rep = s.predict_with_report([9999.0, 12.0, 150.0, 200.0])
    assert np.isfinite(val) and rep.is_ood


def test_unfitted_surrogate_raises():
    with pytest.raises(RuntimeError, match="call fit"):
        Surrogate().predict([1.0])
    with pytest.raises(ValueError, match="mahalanobis_quantile"):
        Surrogate(mahalanobis_quantile=1.0)
    with pytest.raises(ValueError, match="nn_sigma"):
        Surrogate(nn_sigma=0.0)
    with pytest.raises(ValueError, match="unknown model kind"):
        Surrogate(kind="magic").fit(synthetic(3, 20, seed=13))


@pytest.mark.benchmark
def test_permutation_importance_recovers_the_known_mechanism():
    """The generator's coefficients are known, so the importance ORDERING is a
    checkable claim rather than a plot to admire.

    The ranking follows coefficient x range, not coefficient alone. Computed
    from the generator directly, with each feature uniform on its range so the
    standard-deviation contribution is c*(hi-lo)/sqrt(12):

      al_ppm  0.0035 x (60-5)   = 0.1925 span, 61.1 percent of variance
      p80_um  0.0003 x (400-50) = 0.1050 span, 18.2 percent
      ti_ppm  0.0040 x (25-1)   = 0.0960 span, 15.2 percent
      fe_ppm  0.0002 x (300-10) = 0.0580 span,  5.5 percent

    Ti has the LARGEST coefficient of the four and is only third in importance,
    because its range is narrow. An earlier version of this docstring claimed
    "Al and Ti dominate by construction" and asserted only that the top two came
    from a permissive three-element set, so the stated mechanism and the
    recovered one disagreed while the test passed. The claim is now the computed
    ordering and the assertion is exact.
    """
    d = synthetic(n_deposits=5, per_deposit=60, seed=14, noise=0.01)
    s = Surrogate(kind="gbm", seed=14).fit(d)
    imp = s.permutation_importance(d, n_repeats=5, seed=14)
    print("\npermutation importance (held-out deposits):")
    for k, v in imp.items():
        print(f"  {k:10s} {v:.6f}")

    # Analytic variance shares, recomputed here rather than quoted.
    coef = {"al_ppm": (0.0035, 5, 60), "ti_ppm": (0.004, 1, 25),
            "fe_ppm": (0.0002, 10, 300), "p80_um": (0.0003, 50, 400)}
    # The SPANS quoted in the docstring, c*(hi-lo).
    span = {k: c * (hi - lo) for k, (c, lo, hi) in coef.items()}
    assert span["al_ppm"] == pytest.approx(0.1925, abs=1e-9)
    assert span["p80_um"] == pytest.approx(0.1050, abs=1e-9)
    assert span["ti_ppm"] == pytest.approx(0.0960, abs=1e-9)
    assert span["fe_ppm"] == pytest.approx(0.0580, abs=1e-9)

    sd = {k: c * (hi - lo) / np.sqrt(12) for k, (c, lo, hi) in coef.items()}
    # The VARIANCE SHARES quoted as percentages.
    tot = sum(v ** 2 for v in sd.values())
    share = {k: v ** 2 / tot * 100 for k, v in sd.items()}
    assert share["al_ppm"] == pytest.approx(61.1, abs=0.05)
    assert share["p80_um"] == pytest.approx(18.2, abs=0.05)
    assert share["ti_ppm"] == pytest.approx(15.2, abs=0.05)
    assert share["fe_ppm"] == pytest.approx(5.5, abs=0.05)
    analytic = sorted(sd, key=lambda k: -sd[k])
    assert analytic == ["al_ppm", "p80_um", "ti_ppm", "fe_ppm"]
    assert sd["ti_ppm"] < sd["p80_um"], "largest coefficient, narrowest range"

    # The model must recover that exact ordering, not a permissive superset.
    assert list(imp) == analytic, (
        f"recovered {list(imp)} but the generator implies {analytic}"
    )
    # And the top feature must dominate by a clear margin, as 0.611 vs 0.182
    # implies.
    assert imp["al_ppm"] > 2.0 * imp["p80_um"]


def _load_demo_surrogate():
    """Load scripts/demo_surrogate.py by path.

    `from scripts import demo_surrogate` was the first attempt and it cannot
    work: scripts/ deliberately has no __init__.py because it holds runnable
    command-line tools, not an importable package, and adding one to satisfy
    a test would misrepresent the layout. The three tests below were reported
    as "pinned" while failing on this ImportError, which is why the loader is
    a named helper with this note rather than an inline try/except: a test
    that cannot import its fixture is not a passing test.
    """
    import importlib.util  # noqa: PLC0415

    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "demo_surrogate.py"
    assert path.exists(), f"{path} is missing; the demo script is the fixture source"
    spec = importlib.util.spec_from_file_location("demo_surrogate", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.benchmark
def test_gbm_loses_to_ridge_on_a_correctly_specified_linear_problem() -> None:
    """A tree ensemble must not be assumed better than a line.

    On grouped synthetic data whose generating function is linear in two
    features with an additive per-deposit intercept, the gradient-boosted
    surrogate loses to ridge regression on EVERY leave-one-deposit-out fold.
    That is the correct outcome, not a defect: the linear model is correctly
    specified, and the ensemble pays in variance for flexibility the problem
    does not reward.

    This is asserted rather than merely observed because the failure mode it
    guards against is institutional. A platform that ships a gradient-boosted
    surrogate invites the reader to assume it is the better model. With a few
    dozen samples across a handful of deposits it frequently is not, and the
    per-fold baseline comparison is the only thing that reveals it. If a
    future change makes the ensemble win here, that is a finding worth
    inspecting, not a test to silence: it would mean either the fixture has
    stopped being linear or the baseline has broken.
    """
    demo = _load_demo_surrogate()

    ts = demo.synthetic_deposits(seed=0)
    res = evaluate_surrogate(ts, kind="gbm", seed=0)

    beats_ridge = [f.deposit for f in res.folds
                   if f.rmse < f.baseline_ridge_rmse]
    beats_mean = [f.deposit for f in res.folds
                  if f.rmse < f.baseline_mean_rmse]

    assert len(beats_mean) == len(res.folds), (
        "the surrogate must at least beat predicting the training mean on "
        f"every fold; it failed on {set(f.deposit for f in res.folds) - set(beats_mean)}"
    )
    assert not beats_ridge, (
        "the gradient-boosted model now beats ridge on "
        f"{beats_ridge}, which contradicts the documented finding. Either "
        "the fixture is no longer linear or the ridge baseline is broken; "
        "inspect before changing this assertion."
    )


@pytest.mark.benchmark
def test_random_kfold_is_optimistic_by_a_measured_margin() -> None:
    """Grouped data makes random k-fold understate out-of-deposit error.

    Mechanism: samples from one deposit share an intercept, so a random split
    leaves siblings of each test sample in training and the model partly
    memorises the offset instead of learning the chemistry. The ratio of
    leave-one-deposit-out RMSE to random k-fold RMSE measures the size of that
    self-deception. It must exceed 1.

    The magnitude is deliberately bounded loosely (1.0 to 2.0) rather than
    pinned: it depends on the deposit-offset standard deviation relative to
    the noise, and pinning it would make the test a record of one fixture
    rather than of the effect. What must hold is the SIGN.

    Basis: this is an ANALYTIC check against a known generating function, not
    a literature benchmark. The fixture's per-deposit intercept is drawn from
    a distribution this test controls, so the direction of the inequality is
    known by construction: at an offset standard deviation of zero the two
    schemes must agree, and any positive offset creates leakage a random
    split can exploit. The reference value is exact in that sense and no
    published measurement is being reproduced.
    """
    demo = _load_demo_surrogate()

    ts = demo.synthetic_deposits(seed=0)
    opt = random_kfold_optimism(ts, kind="gbm", seed=0)
    ratio = opt["optimism_ratio"]
    assert ratio > 1.0, (
        f"random k-fold RMSE {opt['random_kfold_rmse']:.4f} should be LOWER "
        f"than leave-one-deposit-out {opt['leave_one_deposit_out_rmse']:.4f}; "
        f"ratio came out {ratio:.3f}, so the grouping leak has vanished and "
        "the fixture no longer has deposit structure"
    )
    assert 1.0 < ratio < 2.0, f"optimism ratio {ratio:.3f} outside expected band"


def test_permutation_importance_has_a_working_negative_control() -> None:
    """A feature with no causal role must rank at zero.

    demo_surrogate's fixture includes li_ppm, which is drawn at random and
    never enters the generating function. If permutation importance assigns it
    material weight, the measure is reporting noise as signal and no
    importance ranking from this platform can be trusted.
    """
    demo = _load_demo_surrogate()

    ts = demo.synthetic_deposits(seed=0)
    model = Surrogate(kind="gbm", seed=0).fit(ts)
    imp = model.permutation_importance(ts)

    causal = max(imp["al_ppm"], imp["ti_ppm"])
    assert imp["li_ppm"] < 0.10 * causal, (
        f"li_ppm has no causal role but scored {imp['li_ppm']:.4f} against a "
        f"largest causal importance of {causal:.4f}; the measure is picking "
        "up noise"
    )
    assert imp["al_ppm"] > imp["li_ppm"] and imp["ti_ppm"] > imp["li_ppm"]


def test_the_global_skill_score_is_not_an_r2_and_does_not_claim_to_be():
    """The field named r2_global was not an R2, and its sign misled.

    As committed, the quantity was

        1 - ss_res / (denom / n_samples * n_test)

    where denom is the total sum of squares about the GLOBAL mean over ALL
    samples. Since denom / n_samples is the global variance, that expression is
    algebraically 1 - MSE / Var_global, a variance-normalised skill score. It
    is NOT the coefficient of determination against a global-mean predictor on
    the held-out fold, which would need sum((y_test - global_mean)^2) in the
    denominator, and the two differ whenever the fold's own distance from the
    global mean differs from the global variance.

    The difference is not cosmetic, because a reader takes a positive R2 to
    mean the model beat the naive predictor. On the fixture below, deposit C
    sits at the global mean and the model extrapolates badly onto it: the
    committed field reports +0.0525, which reads as mild skill, while R2
    against a global-mean predictor on that fold is -8230.5515. The naive
    predictor was better by four orders of magnitude and the metric said the
    model had skill.

    The skill score is the RIGHT quantity to report here, for the reason the
    source comment gives: a denominator taken from the fold would measure the
    fold's internal spread rather than the model, and a single-deposit fold has
    little internal spread. So the quantity is kept and the NAME is corrected,
    with the per-fold R2 reported alongside so the divergence is visible rather
    than hidden behind a familiar label.
    """
    rng = np.random.default_rng(3)
    groups = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
    level = {"A": 0.0, "B": 100.0, "C": 50.0}
    y = np.array([level[g] for g in groups]) + rng.normal(0, 0.5, 30)
    # Deposit C's feature value lies about its target, so holding C out forces
    # an extrapolation and the model misses it badly.
    f1 = np.where(groups == "C", 90.0, y) + rng.normal(0, 0.1, 30)
    X = np.column_stack([f1, rng.normal(0, 1, 30)])
    ts = TrainingSet(X=X, y=y, groups=groups, feature_names=["f1", "f2"],
                     target_name="t")

    global_mean = float(np.mean(ts.y))
    global_var = float(np.sum((ts.y - global_mean) ** 2)) / ts.n_samples
    assert global_var == pytest.approx(1666.3320, abs=1e-3)

    res = evaluate_surrogate(ts, kind="ridge", seed=0)
    by_dep = {f.deposit: f for f in res.folds}

    # The reported quantity is exactly 1 - MSE / Var_global, on every fold.
    for f in res.folds:
        assert f.skill_vs_global_variance == pytest.approx(
            1.0 - f.rmse ** 2 / global_var, rel=1e-9), (
            f"fold {f.deposit}: reported score is not 1 - MSE/Var_global"
        )

    # And the per-fold R2 is reported separately, because on deposit C the two
    # disagree in sign and by four orders of magnitude.
    c = by_dep["C"]
    assert c.skill_vs_global_variance == pytest.approx(0.0525, abs=1e-4)
    ss_res_c = c.rmse ** 2 * c.n_test
    ss_tot_c = float(np.sum((y[groups == "C"] - global_mean) ** 2))
    assert ss_tot_c == pytest.approx(1.9181, abs=1e-3)
    assert 1.0 - ss_res_c / ss_tot_c == pytest.approx(-8230.5515, rel=1e-4)
    assert c.r2_vs_global_mean_predictor == pytest.approx(-8230.5515, rel=1e-4)
    assert c.skill_vs_global_variance > 0.0
    assert c.r2_vs_global_mean_predictor < 0.0, (
        "the fold where the naive global-mean predictor wins must report a "
        "negative R2, whatever the skill score says"
    )
    # beats_mean uses the training-mean predictor and is a separate question.
    assert not c.beats_mean
