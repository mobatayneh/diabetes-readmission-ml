"""Shared metrics and paired patient bootstrap for frozen predictions."""

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score
from src.baseline import evaluate, choose_threshold

INTERVAL_METRICS = ("roc_auc", "average_precision", "f1", "precision", "recall", "specificity", "balanced_accuracy", "brier_score", "log_loss")


def validate_predictions(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Labels and probabilities must be non-empty aligned vectors")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("Both binary classes are required")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return y, p


def metrics(y, probabilities, threshold):
    y, p = validate_predictions(y, probabilities)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0, 1]")
    result = evaluate(y, p, threshold)
    cm = result.pop("confusion_matrix")
    return {**result, **cm, "specificity": cm["tn"] / (cm["tn"] + cm["fp"]),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean())}


def threshold_curve(y, p):
    validate_predictions(y, p)
    return pd.DataFrame([{
        "threshold": float(t),
        "precision": precision_score(y, np.asarray(p) >= t, zero_division=0),
        "recall": recall_score(y, np.asarray(p) >= t, zero_division=0),
        "f1": f1_score(y, np.asarray(p) >= t, zero_division=0),
    } for t in np.linspace(0.05, 0.95, 91)])


def paired_bootstrap(y, predictions, thresholds, primary, replicates=1000, seed=2042):
    """Ordinary patient bootstrap, shared resamples; fixed models and thresholds.

    Each row is one unique patient. Intervals are percentile 95%, unadjusted
    for multiple comparisons and conditional on this development procedure.
    No hyperparameters or thresholds are refitted in a bootstrap resample.
    """
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates required")
    y = np.asarray(y)
    for p in predictions.values():
        validate_predictions(y, p)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in predictions}
    accepted = 0
    while accepted < replicates:
        index = rng.integers(0, len(y), len(y))
        if len(np.unique(y[index])) < 2:
            continue
        for name, p in predictions.items():
            m = metrics(y[index], np.asarray(p)[index], thresholds[name])
            draws[name].append([m[k] for k in INTERVAL_METRICS])
        accepted += 1
    arrays = {k: np.asarray(v) for k, v in draws.items()}
    intervals, paired = [], []
    for name, values in arrays.items():
        point = metrics(y, predictions[name], thresholds[name])
        for j, key in enumerate(INTERVAL_METRICS):
            low, high = np.quantile(values[:, j], [.025, .975])
            intervals.append(dict(model=name, metric=key, estimate=point[key], low=low, high=high))
    comparisons = list(combinations(primary, 2))
    # Prespecified within-algorithm contrasts in addition to primary pairs.
    for name in primary:
        root = name.removesuffix("_tuned")
        comparisons += [(name, root + "_initial"), (root + "_calibrated", name)]
    for left, right in comparisons:
        for j, key in enumerate(INTERVAL_METRICS):
            diff = arrays[left][:, j] - arrays[right][:, j]
            low, high = np.quantile(diff, [.025, .975])
            delta = metrics(y, predictions[left], thresholds[left])[key] - metrics(y, predictions[right], thresholds[right])[key]
            paired.append(dict(left=left, right=right, metric=key, difference=delta, low=low, high=high))
    return pd.DataFrame(intervals), pd.DataFrame(paired)
