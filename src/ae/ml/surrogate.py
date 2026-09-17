r"""Surrogate models with leave-one-deposit-out validation and OOD detection.

Purpose and the failure mode it guards against
-----------------------------------------------
A surrogate is a cheap function fitted to expensive evaluations, used to replace
the physics inside an optimisation or a large Monte Carlo. The danger in a
minerals setting is not overfitting in the usual sense; it is that samples from
one deposit are NOT independent observations. Grains from a single drill core
share a formation history, an alteration overprint and an analytical batch, so
random k-fold cross-validation puts near-replicates on both sides of the split
and returns an optimistic error that collapses on a new deposit.

The validation here is therefore GROUPED: leave-one-deposit-out, so every
reported error is a genuine extrapolation to unseen geology.
:func:`random_kfold_optimism` quantifies the gap for a given dataset, and the
platform requires it to be reported alongside any surrogate error, because the
difference is what a reader needs to judge transferability.

The second guard is out-of-distribution detection. A tree model interpolates
inside its training hull and extrapolates as a constant outside it, so it
returns a confident-looking number for an ore unlike anything it has seen.
:class:`Surrogate` therefore refuses to predict silently on OOD input: every
prediction carries an OOD flag, a Mahalanobis distance and a nearest-neighbour
distance, and :meth:`Surrogate.predict` raises if asked for a point estimate on
flagged input unless the caller explicitly accepts it.

What a surrogate may and may not be used for
--------------------------------------------
Legitimate: replacing a slow physics call inside a 10,000-draw Monte Carlo
after the surrogate's leave-one-deposit-out error has been shown small relative
to the input uncertainty; screening a large design space to shortlist candidates
for full simulation.

NOT legitimate: predicting a lattice impurity ceiling for an uncharacterized
deposit. Lattice-bound Al, Ti, Li and B are set by crystallisation conditions
and are not recoverable from bulk assay or process response, so a model fitted
on other deposits has no mechanism by which to know them. The platform states
this as a limitation rather than relying on the OOD flag to catch it, because a
new deposit can sit inside the training hull on every measured feature and still
have a different lattice chemistry.

Method notes
------------
* Baselines are mandatory, not optional. :func:`evaluate_surrogate` always
  reports the error of a mean predictor and of ridge regression alongside the
  model, because a gradient-boosted tree that cannot beat a linear fit on
  leave-one-deposit-out is not a surrogate, it is an expensive mean.
* :math:`R^2` is computed against the GLOBAL mean of the held-out fold's parent
  dataset, not the fold's own mean. Using the fold mean makes a single-deposit
  fold's :math:`R^2` depend on that deposit's internal spread and can return
  large negative values that say nothing about the model.
* Errors are reported as RMSE and MAE in the target's own units, plus a
  normalised error (RMSE over the target's interquartile range) so that models
  of different targets can be compared without implying a common scale.

LIMITATIONS
-----------
* Leave-one-deposit-out with :math:`n` deposits gives :math:`n` error estimates.
  With fewer than about five deposits the spread of those estimates is wide and
  the mean is weak evidence; :func:`evaluate_surrogate` returns the per-fold
  values so a reader can see the spread rather than a single averaged number.
* Mahalanobis distance assumes an approximately elliptical training
  distribution and needs more samples than features to estimate a covariance.
  The implementation falls back to a nearest-neighbour criterion and SAYS SO
  when that condition fails, rather than inverting a singular matrix.
* No calibrated predictive interval is provided. Quantile regression or a
  conformal wrapper would give one; until that is implemented, an interval from
  this module would be a guess with error bars drawn on it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "EvaluationResult",
    "FoldResult",
    "OODReport",
    "Surrogate",
    "TrainingSet",
    "evaluate_surrogate",
    "leave_one_deposit_out_splits",
    "random_kfold_optimism",
]


@dataclass
class TrainingSet:
    """Feature matrix, target and DEPOSIT GROUPS.

    ``groups`` is required, not optional. A training set without deposit labels
    cannot be validated honestly in this domain, so the type system says so.
    """

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    target_name: str = "y"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.groups = np.asarray(self.groups)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {self.X.shape}")
        if not (self.X.shape[0] == self.y.size == self.groups.size):
            raise ValueError(
                f"X has {self.X.shape[0]} rows, y has {self.y.size} and groups has "
                f"{self.groups.size}; all three must agree"
            )
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"{len(self.feature_names)} feature names for {self.X.shape[1]} columns"
            )
        if not np.isfinite(self.X).all():
            raise ValueError("X contains non-finite values; impute or drop before fitting")
        if not np.isfinite(self.y).all():
            raise ValueError("y contains non-finite values")
        if self.n_deposits < 2:
            raise ValueError(
                f"only {self.n_deposits} deposit group present; leave-one-deposit-out "
                f"needs at least 2, and meaningful evidence needs about 5"
            )

    @property
    def n_deposits(self) -> int:
        return int(np.unique(self.groups).size)

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    def deposit_sizes(self) -> dict[str, int]:
        u, c = np.unique(self.groups, return_counts=True)
        return {str(k): int(v) for k, v in zip(u, c)}


@dataclass
class OODReport:
    """Whether a query point lies outside the training distribution."""

    is_ood: bool
    mahalanobis: float
    mahalanobis_threshold: float
    nn_distance: float
    nn_threshold: float
    method: str
    out_of_range_features: dict[str, tuple[float, float, float]] = field(
        default_factory=dict)

    def reason(self) -> str:
        if not self.is_ood:
            return "inside the training distribution on all implemented criteria"
        parts = []
        if self.mahalanobis > self.mahalanobis_threshold:
            parts.append(f"Mahalanobis {self.mahalanobis:.2f} exceeds "
                         f"{self.mahalanobis_threshold:.2f}")
        if self.nn_distance > self.nn_threshold:
            parts.append(f"nearest training point {self.nn_distance:.2f} standard "
                         f"deviations away, above {self.nn_threshold:.2f}")
        for f, (v, lo, hi) in self.out_of_range_features.items():
            parts.append(f"{f} = {v:.4g} outside the trained range "
                         f"[{lo:.4g}, {hi:.4g}]")
        return "; ".join(parts)


@dataclass
class FoldResult:
    """Error on one held-out deposit."""

    deposit: str
    n_test: int
    n_train: int
    rmse: float
    mae: float
    bias: float
    r2_global: float
    baseline_mean_rmse: float
    baseline_ridge_rmse: float

    @property
    def beats_mean(self) -> bool:
        return self.rmse < self.baseline_mean_rmse

    @property
    def beats_ridge(self) -> bool:
        return self.rmse < self.baseline_ridge_rmse


@dataclass
class EvaluationResult:
    """Leave-one-deposit-out evaluation, per fold and aggregated."""

    folds: list[FoldResult]
    target_name: str
    target_iqr: float
    model_label: str = "model"

    @property
    def rmse(self) -> float:
        """Sample-weighted RMSE across folds, pooled on squared error."""
        n = np.array([f.n_test for f in self.folds], dtype=float)
        se = np.array([f.rmse ** 2 for f in self.folds], dtype=float)
        return float(np.sqrt(np.sum(n * se) / np.sum(n)))

    @property
    def mae(self) -> float:
        n = np.array([f.n_test for f in self.folds], dtype=float)
        return float(np.sum(n * np.array([f.mae for f in self.folds])) / np.sum(n))

    @property
    def rmse_spread(self) -> tuple[float, float]:
        v = [f.rmse for f in self.folds]
        return (float(min(v)), float(max(v)))

    @property
    def normalised_rmse(self) -> float:
        """RMSE over the target's interquartile range."""
        return self.rmse / self.target_iqr if self.target_iqr > 0 else float("nan")

    @property
    def baseline_mean_rmse(self) -> float:
        n = np.array([f.n_test for f in self.folds], dtype=float)
        se = np.array([f.baseline_mean_rmse ** 2 for f in self.folds], dtype=float)
        return float(np.sqrt(np.sum(n * se) / np.sum(n)))

    @property
    def baseline_ridge_rmse(self) -> float:
        n = np.array([f.n_test for f in self.folds], dtype=float)
        se = np.array([f.baseline_ridge_rmse ** 2 for f in self.folds], dtype=float)
        return float(np.sqrt(np.sum(n * se) / np.sum(n)))

    @property
    def skill_over_mean(self) -> float:
        """1 - RMSE/RMSE_mean. Zero or below means the model adds nothing."""
        b = self.baseline_mean_rmse
        return float(1.0 - self.rmse / b) if b > 0 else float("nan")

    @property
    def folds_beating_mean(self) -> int:
        return sum(1 for f in self.folds if f.beats_mean)

    def is_useful(self, min_skill: float = 0.10) -> bool:
        """Whether the surrogate earns its complexity.

        Requires BOTH positive skill over the mean predictor and a majority of
        folds beating it, so a single easy deposit cannot carry the verdict.
        """
        return (self.skill_over_mean >= min_skill
                and self.folds_beating_mean > len(self.folds) / 2)

    def report(self) -> dict[str, Any]:
        lo, hi = self.rmse_spread
        return {
            "model": self.model_label,
            "target": self.target_name,
            "n_folds": len(self.folds),
            "rmse": self.rmse,
            "mae": self.mae,
            "rmse_per_fold_min": lo,
            "rmse_per_fold_max": hi,
            "normalised_rmse_over_iqr": self.normalised_rmse,
            "baseline_mean_rmse": self.baseline_mean_rmse,
            "baseline_ridge_rmse": self.baseline_ridge_rmse,
            "skill_over_mean": self.skill_over_mean,
            "folds_beating_mean": self.folds_beating_mean,
            "folds_beating_ridge": sum(1 for f in self.folds if f.beats_ridge),
            "useful": self.is_useful(),
        }


def leave_one_deposit_out_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index pairs holding out one deposit at a time."""
    groups = np.asarray(groups)
    out = []
    for g in np.unique(groups):
        test = np.flatnonzero(groups == g)
        train = np.flatnonzero(groups != g)
        out.append((train, test))
    return out


def _make_model(kind: str, seed: int) -> Any:
    if kind == "gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08,
                                             max_depth=4, random_state=seed)
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                     random_state=seed, n_jobs=1)
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 3, 25)))
    if kind == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        k = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-3)
        return make_pipeline(StandardScaler(),
                             GaussianProcessRegressor(kernel=k, normalize_y=True,
                                                      random_state=seed))
    raise ValueError(f"unknown model kind {kind!r}; use gbm, rf, ridge or gp")


def evaluate_surrogate(
    data: TrainingSet,
    kind: str = "gbm",
    seed: int = 0,
) -> EvaluationResult:
    """Leave-one-deposit-out evaluation against mandatory baselines.

    Every fold reports the model's error AND the error of a mean predictor and
    of ridge regression fitted on the same training deposits, because a model
    that cannot beat a linear fit on unseen geology is not a surrogate.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    global_mean = float(np.mean(data.y))
    denom = float(np.sum((data.y - global_mean) ** 2))
    folds: list[FoldResult] = []

    for train, test in leave_one_deposit_out_splits(data.groups):
        Xtr, ytr, Xte, yte = data.X[train], data.y[train], data.X[test], data.y[test]
        m = _make_model(kind, seed)
        m.fit(Xtr, ytr)
        pred = np.asarray(m.predict(Xte), dtype=float)

        ridge = _make_model("ridge", seed)
        ridge.fit(Xtr, ytr)
        pred_ridge = np.asarray(ridge.predict(Xte), dtype=float)
        pred_mean = np.full(yte.shape, float(np.mean(ytr)))

        # R2 against the GLOBAL mean, not the fold's own mean: a single-deposit
        # fold has little internal spread, so fold-mean R2 measures that spread
        # rather than the model.
        ss_res = float(np.sum((yte - pred) ** 2))
        r2 = 1.0 - ss_res / (denom / data.n_samples * yte.size) if denom > 0 else float("nan")

        folds.append(FoldResult(
            deposit=str(np.unique(data.groups[test])[0]),
            n_test=int(yte.size), n_train=int(ytr.size),
            rmse=float(np.sqrt(mean_squared_error(yte, pred))),
            mae=float(mean_absolute_error(yte, pred)),
            bias=float(np.mean(pred - yte)),
            r2_global=float(r2),
            baseline_mean_rmse=float(np.sqrt(mean_squared_error(yte, pred_mean))),
            baseline_ridge_rmse=float(np.sqrt(mean_squared_error(yte, pred_ridge))),
        ))

    q75, q25 = np.percentile(data.y, [75, 25])
    return EvaluationResult(folds=folds, target_name=data.target_name,
                            target_iqr=float(q75 - q25), model_label=kind)


def random_kfold_optimism(
    data: TrainingSet,
    kind: str = "gbm",
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, float]:
    """How much random k-fold flatters the model relative to grouped validation.

    Reported alongside every surrogate error. A large ratio means samples within
    a deposit are near-replicates, so a random split leaks information across
    the fold boundary and its error is not an estimate of performance on a new
    deposit.
    """
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    se, n = 0.0, 0
    for train, test in kf.split(data.X):
        m = _make_model(kind, seed)
        m.fit(data.X[train], data.y[train])
        p = m.predict(data.X[test])
        se += float(mean_squared_error(data.y[test], p)) * test.size
        n += test.size
    random_rmse = float(np.sqrt(se / n))
    grouped = evaluate_surrogate(data, kind=kind, seed=seed).rmse
    return {
        "random_kfold_rmse": random_rmse,
        "leave_one_deposit_out_rmse": grouped,
        "optimism_ratio": grouped / random_rmse if random_rmse > 0 else float("nan"),
        "optimism_absolute": grouped - random_rmse,
    }


class Surrogate:
    """A fitted surrogate that refuses to predict silently out of distribution.

    Parameters
    ----------
    kind
        ``"gbm"``, ``"rf"``, ``"ridge"`` or ``"gp"``.
    mahalanobis_quantile
        Training-set quantile of squared Mahalanobis distance used as the OOD
        threshold. 0.99 means the flag fires on input further from the training
        centroid than 99 percent of the training data.
    nn_sigma
        Nearest-neighbour threshold in standardised units. A query whose closest
        training point is further than this is flagged even if it sits near the
        centroid, which catches holes inside the hull.
    """

    def __init__(self, kind: str = "gbm", seed: int = 0,
                 mahalanobis_quantile: float = 0.99, nn_sigma: float = 3.0) -> None:
        if not 0.5 < mahalanobis_quantile < 1.0:
            raise ValueError("mahalanobis_quantile must be in (0.5, 1.0)")
        if nn_sigma <= 0:
            raise ValueError("nn_sigma must be positive")
        self.kind = kind
        self.seed = seed
        self.mahalanobis_quantile = mahalanobis_quantile
        self.nn_sigma = nn_sigma
        self._fitted = False

    def fit(self, data: TrainingSet) -> Surrogate:
        self.feature_names = list(data.feature_names)
        self.target_name = data.target_name
        self._model = _make_model(self.kind, self.seed)
        self._model.fit(data.X, data.y)

        self._mu = data.X.mean(axis=0)
        self._sd = data.X.std(axis=0, ddof=1)
        self._sd[self._sd == 0] = 1.0
        self._Xz = (data.X - self._mu) / self._sd
        self._lo = data.X.min(axis=0)
        self._hi = data.X.max(axis=0)

        # Covariance needs more samples than features; say so rather than
        # inverting a singular matrix.
        n, k = data.X.shape
        if n > k + 2:
            cov = np.cov(self._Xz, rowvar=False)
            try:
                self._prec = np.linalg.inv(cov + 1e-8 * np.eye(k))
                d2 = np.einsum("ij,jk,ik->i", self._Xz, self._prec, self._Xz)
                self._maha_thresh = float(np.sqrt(
                    np.quantile(d2, self.mahalanobis_quantile)))
                self._ood_method = "mahalanobis + nearest neighbour + feature range"
            except np.linalg.LinAlgError:
                self._prec = None
                self._maha_thresh = float("inf")
                self._ood_method = "nearest neighbour + feature range (singular covariance)"
        else:
            self._prec = None
            self._maha_thresh = float("inf")
            self._ood_method = (
                f"nearest neighbour + feature range ({n} samples for {k} features "
                f"is too few to estimate a covariance)"
            )
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit() before predicting")

    def ood_report(self, x: Sequence[float]) -> OODReport:
        """OOD assessment for a single query point."""
        self._check_fitted()
        x = np.asarray(x, dtype=float).ravel()
        if x.size != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features "
                f"({', '.join(self.feature_names)}), got {x.size}"
            )
        z = (x - self._mu) / self._sd
        maha = (float(np.sqrt(z @ self._prec @ z)) if self._prec is not None
                else float("nan"))
        nn = float(np.min(np.linalg.norm(self._Xz - z, axis=1)))
        oor = {name: (float(x[i]), float(self._lo[i]), float(self._hi[i]))
               for i, name in enumerate(self.feature_names)
               if x[i] < self._lo[i] or x[i] > self._hi[i]}
        is_ood = bool(
            (self._prec is not None and maha > self._maha_thresh)
            or nn > self.nn_sigma
            or bool(oor)
        )
        return OODReport(is_ood=is_ood, mahalanobis=maha,
                         mahalanobis_threshold=self._maha_thresh,
                         nn_distance=nn, nn_threshold=self.nn_sigma,
                         method=self._ood_method, out_of_range_features=oor)

    def predict(self, x: Sequence[float],
                on_ood: Literal["raise", "warn", "allow"] = "raise") -> float:
        """Point prediction, refusing OOD input by default.

        A tree model extrapolates as a constant outside its training hull and
        so returns a confident-looking number for an ore unlike anything it has
        seen. Requiring the caller to opt in makes that explicit.
        """
        self._check_fitted()
        rep = self.ood_report(x)
        if rep.is_ood:
            if on_ood == "raise":
                raise ValueError(
                    f"query is out of distribution: {rep.reason()}. Pass "
                    f"on_ood='allow' to accept an extrapolation, or measure the "
                    f"case directly."
                )
            if on_ood == "warn":
                import warnings
                warnings.warn(f"out-of-distribution prediction: {rep.reason()}",
                              UserWarning, stacklevel=2)
        x = np.asarray(x, dtype=float).reshape(1, -1)
        return float(self._model.predict(x)[0])

    def predict_with_report(self, x: Sequence[float]) -> tuple[float, OODReport]:
        """Prediction and its OOD assessment together, never raising.

        For batch screening, where the caller filters on the flag afterwards.
        """
        rep = self.ood_report(x)
        val = float(self._model.predict(
            np.asarray(x, dtype=float).reshape(1, -1))[0])
        return val, rep

    def permutation_importance(self, data: TrainingSet, n_repeats: int = 10,
                               seed: int = 0) -> dict[str, float]:
        """Permutation importance on held-out deposits, not on training data.

        Training-set importance rewards a feature the model has memorised.
        Computed here by leave-one-deposit-out, so importance means importance
        for generalisation.
        """
        self._check_fitted()
        from sklearn.metrics import mean_squared_error
        rng = np.random.default_rng(seed)
        totals = {n: 0.0 for n in self.feature_names}
        weight = 0.0
        for train, test in leave_one_deposit_out_splits(data.groups):
            m = _make_model(self.kind, self.seed)
            m.fit(data.X[train], data.y[train])
            base = mean_squared_error(data.y[test], m.predict(data.X[test]))
            for i, name in enumerate(self.feature_names):
                drops = []
                for _ in range(n_repeats):
                    Xp = data.X[test].copy()
                    Xp[:, i] = rng.permutation(Xp[:, i])
                    drops.append(mean_squared_error(data.y[test], m.predict(Xp)) - base)
                totals[name] += float(np.mean(drops)) * test.size
            weight += test.size
        return {k: v / weight for k, v in
                sorted(totals.items(), key=lambda kv: -kv[1])}
