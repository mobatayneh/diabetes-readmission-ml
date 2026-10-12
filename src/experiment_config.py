"""Prespecified, bounded CPU experiment; never adapt these to test scores."""

from sklearn.model_selection import ParameterSampler

MODELS = ("logistic", "lightgbm", "xgboost")
SEED = 42
FOLDS = 3
RANDOM_CANDIDATES = 6
BOOTSTRAPS = 1000
THREADS = 2
SEARCH_SPACES = {
    "logistic": {
        "classifier__estimator__C": [0.03, 0.1, 0.3, 1.0, 3.0],
        "classifier__estimator__l1_ratio": [0.0, 1.0],
        "classifier__weighting": [None, "balanced"],
    },
    "lightgbm": {
        "classifier__estimator__n_estimators": [100, 200, 350],
        "classifier__estimator__learning_rate": [0.03, 0.06, 0.1],
        "classifier__estimator__max_depth": [3, 5, -1],
        "classifier__estimator__num_leaves": [7, 15, 31],
        "classifier__estimator__min_child_samples": [20, 60, 120],
        "classifier__estimator__subsample": [0.8, 1.0],
        "classifier__estimator__colsample_bytree": [0.8, 1.0],
        "classifier__estimator__reg_alpha": [0.0, 0.5],
        "classifier__estimator__reg_lambda": [0.0, 2.0, 10.0],
        "classifier__weighting": [None, "balanced"],
    },
    "xgboost": {
        "classifier__estimator__n_estimators": [100, 200, 350],
        "classifier__estimator__learning_rate": [0.03, 0.06, 0.1],
        "classifier__estimator__max_depth": [2, 3, 5],
        "classifier__estimator__min_child_weight": [1, 5, 15],
        "classifier__estimator__subsample": [0.8, 1.0],
        "classifier__estimator__colsample_bytree": [0.8, 1.0],
        "classifier__estimator__gamma": [0.0, 0.5, 2.0],
        "classifier__estimator__reg_alpha": [0.0, 0.5],
        "classifier__estimator__reg_lambda": [1.0, 5.0, 15.0],
        "classifier__weighting": [None, "balanced"],
    },
}


def candidates(model, count=RANDOM_CANDIDATES, seed=SEED):
    """Include the untuned configuration, then fixed seeded random candidates."""
    return [{}] + list(ParameterSampler(SEARCH_SPACES[model], n_iter=count, random_state=seed))


def protocol(smoke=False):
    return {
        "name": "controlled_readmission_benchmark_v1",
        "seed": SEED, "cv_folds": 2 if smoke else FOLDS,
        "random_candidates_per_model": 1 if smoke else RANDOM_CANDIDATES,
        "threads_per_fit": THREADS, "cv_jobs": 1,
        "bootstrap_replicates": 20 if smoke else BOOTSTRAPS,
        "smoke": smoke, "max_patients": 600 if smoke else None,
        "search_metric": "mean training-fold average precision",
        "threshold_rule": "maximum validation F1 over 0.05..0.95 step 0.01; highest threshold on ties",
        "primary_models": [m + "_tuned" for m in MODELS],
        "secondary_models": [m + suffix for m in MODELS for suffix in ("_initial", "_calibrated")],
        "calibration": "sigmoid; nested training-only hyperparameter search within each calibration fold; ensemble=False",
        "development_ablations": ["toggle class weighting", "pool categories with training count <20", "omit discharge disposition"],
        "ablation_policy": "report development-only sensitivity; do not promote to final models",
        "robustness": "selected fixed configurations on a second training CV partition (seed 43); exploratory, not nested",
        "prediction_time": "end of index hospitalization, assuming final encounter fields are available",
        "test_history": "same patient holdout previously reported for the portfolio logistic baseline; not pristine external confirmation",
        "search_spaces": SEARCH_SPACES,
        "candidates": {m: candidates(m, 1 if smoke else RANDOM_CANDIDATES) for m in MODELS},
    }
