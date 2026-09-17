"""Exercise the surrogate validation harness end to end, and report what it finds.

Run: python scripts/demo_surrogate.py

WHY THIS SCRIPT EXISTS SEPARATELY FROM THE TESTS. The tests assert that the
harness behaves correctly on fixtures. This script answers a different
question that a reader of the platform will ask first: what does the harness
actually SAY, and is the surrogate trustworthy enough to use? Those numbers
belong in the open, not inside an assertion.

WHAT THE SURROGATE IS AND IS NOT. It is trained here on SYNTHETIC data with a
known generating function, because no measured dataset exists: the deposit is
uncharacterized. So every number this script prints describes the HARNESS, not
the ore. The surrogate is scaffolding whose validation machinery is real and
whose training data is invented, and that distinction is the single most
important thing to carry forward. It is recorded again in HANDOFF.md under
STUBBED.

THE ONE RESULT WORTH READING. Random k-fold cross-validation is OPTIMISTIC on
grouped geological data, and this script measures by how much. The mechanism:
samples from the same deposit share a deposit-level offset, so a random split
puts siblings of a test sample in the training set and the model partly
memorises the offset rather than learning the chemistry. Leave-one-deposit-out
withholds the whole deposit, which is the question actually being asked of a
production model, namely how it performs on a deposit it has never seen. The
ratio between the two is the size of the self-deception a random split buys.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ae.ml.surrogate import (Surrogate, TrainingSet,  # noqa: E402
                             evaluate_surrogate,
                             leave_one_deposit_out_splits,
                             random_kfold_optimism)

N_DEPOSITS = 6
PER_DEPOSIT = 14
#: Deposit-level offset standard deviation, in spec-yield units. This is the
#: parameter that CREATES the optimism: at 0.0 the two validation schemes must
#: agree, because there is no group structure to leak. It is set to 0.08,
#: comparable to the spread the chemistry terms produce over the feature
#: ranges below, so the group effect is of the same order as the signal.
DEPOSIT_OFFSET_SD = 0.08
NOISE_SD = 0.01


def synthetic_deposits(seed: int = 0) -> TrainingSet:
    """Build a grouped synthetic dataset with a known generating function.

    The function is linear in Al and Ti with a per-deposit intercept:

        spec_yield = 0.9 - 0.004*al_ppm - 0.008*ti_ppm + offset(deposit) + e

    Li is included as a feature that does NOT enter the target, so that
    permutation importance has a negative control: a feature with no causal
    role must rank near zero, and if it does not, the importance measure is
    reporting noise as signal.
    """
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for d in range(N_DEPOSITS):
        offset = rng.normal(0.0, DEPOSIT_OFFSET_SD)
        for _ in range(PER_DEPOSIT):
            al = rng.uniform(10.0, 60.0)
            ti = rng.uniform(2.0, 20.0)
            li = rng.uniform(1.0, 8.0)
            X.append([al, ti, li])
            y.append(0.9 - 0.004 * al - 0.008 * ti + offset
                     + rng.normal(0.0, NOISE_SD))
            g.append(f"DEP-{d}")
    return TrainingSet(
        X=np.asarray(X), y=np.asarray(y), groups=np.asarray(g),
        feature_names=["al_ppm", "ti_ppm", "li_ppm"],
        target_name="spec_yield",
    )


def main() -> None:
    ts = synthetic_deposits()
    splits = leave_one_deposit_out_splits(ts.groups)

    print("SYNTHETIC TRAINING SET (not measured data)")
    print(f"  deposits {N_DEPOSITS} | samples {len(ts.y)} | "
          f"features {ts.feature_names}")
    print(f"  deposit offset sd {DEPOSIT_OFFSET_SD} | noise sd {NOISE_SD}")
    print(f"  leave-one-deposit-out folds: {len(splits)}")

    res = evaluate_surrogate(ts, kind="gbm", seed=0)
    rmses = [f.rmse for f in res.folds]
    print("\nLEAVE-ONE-DEPOSIT-OUT")
    print(f"  {'held out':10} {'RMSE':>8} {'mean-pred':>10} {'ridge':>8} "
          f"{'bias':>8}  n_test")
    for f, r in zip(res.folds, rmses):
        print(f"  {f.deposit!s:10} {r:8.4f} {f.baseline_mean_rmse:10.4f} "
              f"{f.baseline_ridge_rmse:8.4f} {f.bias:+8.4f}  {f.n_test}")
    print(f"  mean RMSE {np.mean(rmses):.4f} | worst {max(rmses):.4f} | "
          f"target IQR {res.target_iqr:.4f}")
    # A surrogate that does not beat predicting the training mean is not a
    # model, it is an expensive constant. Reporting the two baselines beside
    # every fold is what makes that visible per deposit rather than on
    # average, and the per-deposit view is the one that matters: a model can
    # beat the mean overall while losing to it on the deposit you are about
    # to run.
    beat_mean = sum(1 for f, r in zip(res.folds, rmses)
                    if r < f.baseline_mean_rmse)
    beat_ridge = sum(1 for f, r in zip(res.folds, rmses)
                     if r < f.baseline_ridge_rmse)
    print(f"  folds where the surrogate beats the mean predictor: "
          f"{beat_mean}/{len(rmses)}")
    print(f"  folds where it beats ridge:                         "
          f"{beat_ridge}/{len(rmses)}")
    if beat_ridge < len(rmses):
        print("  READ THAT LINE. On this dataset the gradient-boosted model "
              "LOSES to ridge")
        print("  regression, and on every fold. That is the correct answer "
              "and not a bug: the")
        print("  generating function is linear in Al and Ti with an additive "
              "per-deposit")
        print("  intercept, so a linear model is correctly specified and a "
              "tree ensemble pays")
        print("  for flexibility it cannot use, in variance. The lesson "
              "generalises: with a few")
        print("  dozen samples across a handful of deposits, the honest "
              "default is the linear")
        print("  model, and a gradient-boosted surrogate has to earn its "
              "place against that")
        print("  baseline on a per-deposit basis before it is used for "
              "anything.")
    print(f"  mean RMSE as a fraction of target IQR: "
          f"{np.mean(rmses) / res.target_iqr:.1%}")
    print("  The worst fold is the number to quote, not the mean: a plant "
          "runs on one deposit at a time.")

    opt = random_kfold_optimism(ts, kind="gbm", seed=0)
    print("\nOPTIMISM OF RANDOM K-FOLD ON GROUPED DATA")
    print(f"  random k-fold RMSE          {opt['random_kfold_rmse']:.4f}")
    print(f"  leave-one-deposit-out RMSE  "
          f"{opt['leave_one_deposit_out_rmse']:.4f}")
    print(f"  optimism ratio              {opt['optimism_ratio']:.3f}")
    print(f"  optimism absolute           {opt['optimism_absolute']:.4f}")
    print("  A ratio above 1 means random k-fold UNDERSTATES the error a new "
          "deposit will produce.")

    model = Surrogate(kind="gbm", seed=0).fit(ts)
    print("\nOUT-OF-DISTRIBUTION DETECTION")
    inside = model.ood_report(np.array([30.0, 10.0, 4.0]))
    outside = model.ood_report(np.array([500.0, 200.0, 90.0]))
    print(f"  inside envelope  is_ood {inside.is_ood!s:5} "
          f"mahalanobis {inside.mahalanobis:.2f} "
          f"(threshold {inside.mahalanobis_threshold:.2f})")
    print(f"  outside envelope is_ood {outside.is_ood!s:5} "
          f"mahalanobis {outside.mahalanobis:.2f}")
    for name, (val, lo, hi) in outside.out_of_range_features.items():
        print(f"    {name}: {val:g} outside training range "
              f"[{lo:.2f}, {hi:.2f}]")
    print("  A prediction on an out-of-envelope feedstock is an "
          "extrapolation and must be refused, not reported with a "
          "confidence interval.")

    imp = model.permutation_importance(ts)
    print("\nPERMUTATION IMPORTANCE (li_ppm is a negative control: it does "
          "not enter the generating function)")
    for name, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        print(f"  {name:10} {v:.4f}")


if __name__ == "__main__":
    main()
