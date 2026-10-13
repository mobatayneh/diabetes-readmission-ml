"""Cloneable estimators and audited training searches for the benchmark."""
import logging
import json
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from src.baseline import make_pipeline

class FoldWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Identical imbalance option for all algorithms, computed inside each fit.

    Balanced sample weights have mean 1; no class counts from another fold or
    partition enter this estimator. No synthetic categorical rows are created.
    """
    def __init__(self, estimator, weighting="balanced"):
        self.estimator = estimator
        self.weighting = weighting

    def fit(self, X, y):
        if self.weighting not in (None, "balanced"):
            raise ValueError("weighting must be None or balanced")
        self.estimator_ = clone(self.estimator)
        weights = None if self.weighting is None else compute_sample_weight("balanced", y)
        self.estimator_.fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


def make_model(name, seed=42, threads=2, smoke=False):
    # Clone the reference pipeline so median imputation / one-hot semantics match.
    pipe = make_pipeline(seed)
    if name == "logistic":
        estimator = LogisticRegression(solver="liblinear", l1_ratio=0, C=1.0,
                                       max_iter=1000, random_state=seed)
    elif name == "lightgbm":
        estimator = LGBMClassifier(n_estimators=20 if smoke else 100, learning_rate=.06,
            num_leaves=15, max_depth=-1, min_child_samples=20, subsample=1.0,
            subsample_freq=1, n_jobs=threads, random_state=seed, verbosity=-1,
            deterministic=True, force_col_wise=True)
    elif name == "xgboost":
        estimator = XGBClassifier(n_estimators=20 if smoke else 100, learning_rate=.06,
            max_depth=3, tree_method="hist", device="cpu", n_jobs=threads,
            random_state=seed, eval_metric="logloss", verbosity=0)
    else:
        raise ValueError(f"Unknown model: {name}")
    if name != "logistic":
        pipe.named_steps["features"].transformers[0][1].set_params(scaler="passthrough")
    pipe.set_params(classifier=FoldWeightedClassifier(estimator))
    return pipe



class AuditedSearchCV(GridSearchCV):
    """Keep the nested calibration search trace in the experiment execution log."""
    def fit(self, X, y=None, **params):
        super().fit(X, y, **params)
        record = {"training_rows": len(X), "selected": self.best_params_, "trials": []}
        for i, candidate in enumerate(self.cv_results_["params"]):
            record["trials"].append({"parameters": candidate,
                "mean_cv_ap": float(self.cv_results_["mean_test_ap"][i]),
                "fold_ap": [float(self.cv_results_[f"split{f}_test_ap"][i]) for f in range(self.n_splits_)]})
        logging.getLogger(__name__).info("SEARCH_TRACE %s", json.dumps(record))
        return self
