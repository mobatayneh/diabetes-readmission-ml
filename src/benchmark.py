"""Separate development and frozen evaluation on the original patient split.

Run `python -m src.benchmark --stage develop`, then `--stage evaluate`.
Raw records, fitted estimators and row-level predictions remain in ignored runs.
Only aggregate evidence is written to reports/benchmark.
"""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import time
import warnings
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from threadpoolctl import threadpool_limits

from src.baseline import FEATURES, prepare_cohort, split_cohort, choose_threshold
from src.evaluation import metrics, paired_bootstrap, threshold_curve
from src.experiment_config import MODELS, protocol
from src.benchmark_models import make_model, AuditedSearchCV
from src.paths import PROJECT_ROOT, RAW_DATA_DIR

LOG = logging.getLogger(__name__)
SOURCE_FILES = ("src/baseline.py", "src/benchmark.py", "src/evaluation.py", "src/experiment_config.py",
                "src/benchmark_reporting.py", "src/benchmark_models.py", "requirements-benchmark.txt", "requirements-core.txt")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n")


def environment():
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost-cpu", "joblib", "matplotlib", "threadpoolctl")
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": {p: version(p) for p in packages}}



def load_partitions(raw_path, config):
    raw = pd.read_csv(raw_path, keep_default_na=False, na_values=["?"], low_memory=False)
    if raw.encounter_id.duplicated().any():
        raise ValueError("Duplicate encounter IDs make encounter selection ambiguous")
    cohort = prepare_cohort(raw)
    if config["max_patients"]:
        cohort, _ = train_test_split(cohort, train_size=config["max_patients"],
            stratify=cohort.readmitted_binary, random_state=config["seed"])
    parts = dict(zip(("train", "validation", "test"), split_cohort(cohort, config["seed"])))
    ids = {k: set(v.patient_nbr) for k, v in parts.items()}
    overlap = {a + "_" + b: len(ids[a] & ids[b]) for a, b in
               (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(overlap.values()):
        raise ValueError("Patient overlap detected")
    manifest = {
        "raw_rows": len(raw), "cohort_rows": len(cohort), "raw_sha256": digest(raw_path),
        "patient_overlap": overlap,
        "partitions": {k: {"n": len(v), "positives": int(v.readmitted_binary.sum())}
            for k, v in parts.items()},
        "features": FEATURES,
        "missing_counts_by_partition": {k: v[FEATURES].isna().sum().to_dict() for k, v in parts.items()},
    }
    return parts, manifest


def private_partition_fingerprints(parts):
    """Identifier-derived checks are local integrity records, never public data."""
    return {k: hashlib.sha256(v.patient_nbr.to_csv(index=False, header=False).encode()).hexdigest()
            for k, v in parts.items()}


def cv_splits(frame, folds, seed):
    if frame.patient_nbr.duplicated().any():
        raise ValueError("CV requires unique patients")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(frame[FEATURES], frame.readmitted_binary))
    for tr, va in splits:
        if set(frame.iloc[tr].patient_nbr) & set(frame.iloc[va].patient_nbr):
            raise ValueError("Patient overlap in CV")
    return splits


def search(estimator, candidate_list, folds):
    grid = [{key: [value] for key, value in candidate.items()} for candidate in candidate_list]
    return AuditedSearchCV(estimator, grid, scoring={"ap": "average_precision", "roc_auc": "roc_auc"},
                        refit="ap", cv=folds, n_jobs=1, error_score="raise", return_train_score=False)


def fold_records(result, model, experiment, seed):
    return [{"model": model, "experiment": experiment, "cv_seed": seed, "fold": i,
             "average_precision": float(ap), "roc_auc": float(result["test_roc_auc"][i]),
             "fit_seconds": float(result["fit_time"][i])}
            for i, ap in enumerate(result["test_ap"])]


def fit_timed(model, X, y):
    start = time.perf_counter()
    model.fit(X, y)
    return time.perf_counter() - start


def develop_model(name, train, validation, config, private):
    """This function has no test data argument, including test labels."""
    X, y = train[FEATURES], train.readmitted_binary
    Xv, yv = validation[FEATURES], validation.readmitted_binary
    folds = cv_splits(train, config["cv_folds"], config["seed"])
    base = make_model(name, threads=config["threads_per_fit"], smoke=config["smoke"])
    candidates = config["candidates"][name]
    LOG.info("%s: fitting initial model and %d training-CV candidates", name, len(candidates))
    initial = clone(base)
    initial_seconds = fit_timed(initial, X, y)
    selector = search(clone(base), candidates, folds)
    search_seconds = fit_timed(selector, X, y)
    tuned = selector.best_estimator_
    raw_cv = selector.cv_results_
    trials = []
    for i, params in enumerate(raw_cv["params"]):
        trials.append({"model": name, "candidate": i, "parameters": params,
            "mean_cv_ap": float(raw_cv["mean_test_ap"][i]), "std_cv_ap": float(raw_cv["std_test_ap"][i]),
            "mean_cv_roc_auc": float(raw_cv["mean_test_roc_auc"][i]),
            "mean_fit_seconds": float(raw_cv["mean_fit_time"][i]),
            "fold_ap": [float(raw_cv[f"split{f}_test_ap"][i]) for f in range(config["cv_folds"])],
            "selected": i == selector.best_index_})
    LOG.info("%s: selected candidate %d (training CV AP %.4f)", name, selector.best_index_, selector.best_score_)
    # Calibration folds repeat hyperparameter selection on their own training
    # subsets. Their held-out labels never select a model or fit preprocessing.
    # ensemble=False also refits this search on all training rows for prediction.
    inner = StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["seed"])
    calibrated = CalibratedClassifierCV(search(clone(base), candidates, inner),
        method="sigmoid", cv=folds, ensemble=False, n_jobs=1)
    LOG.info("%s: fitting nested-CV sigmoid calibration", name)
    calibration_seconds = fit_timed(calibrated, X, y)
    # The full-data search uses the same folds as tuned; calibration folds still
    # repeat selection independently on their own smaller training subsets.
    cal_search = calibrated.calibrated_classifiers_[0].estimator
    if cal_search.best_params_ != selector.best_params_:
        raise AssertionError("Calibrated and tuned full-training selections differ")
    selections = {"tuned": selector.best_params_, "calibrated_full_train": cal_search.best_params_}
    records, threshold_tables, artifacts = [], [], {}
    for variant, model, fit_seconds in (("initial", initial, initial_seconds),
            ("tuned", tuned, selector.refit_time_), ("calibrated", calibrated, calibration_seconds)):
        key = name + "_" + variant
        p = model.predict_proba(Xv)[:, 1]
        threshold, _ = choose_threshold(yv, p)
        records.append({"model": key, **metrics(yv, p, threshold), "fit_seconds": float(fit_seconds)})
        threshold_tables.append(threshold_curve(yv, p).assign(model=key))
        path = private / (key + ".joblib")
        joblib.dump(model, path)
        artifacts[key] = {"filename": path.name, "sha256": digest(path), "threshold": threshold,
                          "fit_seconds": float(fit_seconds)}
    # Controlled single-factor sensitivity analyses, never promoted using test.
    ablations, robustness = [], []
    for label in config["development_ablations"]:
        model = clone(tuned)
        if label == "toggle class weighting":
            current = model.get_params()["classifier__weighting"]
            model.set_params(classifier__weighting=None if current else "balanced")
        elif label == "pool categories with training count <20":
            model.set_params(features__categorical__encoder__min_frequency=20)
        elif label == "omit discharge disposition":
            trans = model.named_steps["features"].transformers
            model.named_steps["features"].transformers = [(n, t, [c for c in cols if c != "discharge_disposition_id"]) for n, t, cols in trans]
        cv = cross_validate(model, X, y, cv=folds, scoring={"ap": "average_precision", "roc_auc": "roc_auc"}, n_jobs=1, error_score="raise")
        seconds = fit_timed(model, X, y)
        pv = model.predict_proba(Xv)[:, 1]
        threshold, _ = choose_threshold(yv, pv)
        ablations.append({"model": name, "factor": label, "cv_mean_ap": float(cv["test_ap"].mean()),
            "fit_seconds": seconds, **metrics(yv, pv, threshold)})
        robustness.extend(fold_records(cv, name, label, 42))
    second = cv_splits(train, config["cv_folds"], config["seed"] + 1)
    cv = cross_validate(clone(tuned), X, y, cv=second, scoring={"ap": "average_precision", "roc_auc": "roc_auc"}, n_jobs=1, error_score="raise")
    robustness.extend(fold_records(cv, name, "selected configuration, second CV partition", 43))
    # Validation-based permutation analysis: underlying raw predictors, no test.
    sample = validation if len(validation) <= 2000 else train_test_split(validation, train_size=2000,
        stratify=validation.readmitted_binary, random_state=config["seed"])[0]
    importance = permutation_importance(tuned, sample[FEATURES], sample.readmitted_binary,
        scoring="average_precision", n_repeats=2 if config["smoke"] else 5,
        random_state=config["seed"], n_jobs=1)
    importance_rows = [{"model": name, "feature": f, "mean_ap_decrease": float(importance.importances_mean[i]),
        "sd_across_permutations": float(importance.importances_std[i]), "validation_sample_n": len(sample)} for i, f in enumerate(FEATURES)]
    coefficients = []
    if name == "logistic":
        names = tuned.named_steps["features"].get_feature_names_out()
        coefs = tuned.named_steps["classifier"].estimator_.coef_[0]
        # Publish strongest associations only, not fitted model files.
        order = np.unique(np.r_[np.argsort(coefs)[:15], np.argsort(coefs)[-15:]])
        coefficients = [{"encoded_feature": names[i], "coefficient": float(coefs[i])} for i in order]
    return {"artifacts": artifacts, "validation": records, "trials": trials, "selections": selections,
        "search_seconds": search_seconds, "calibration_seconds": calibration_seconds,
        "thresholds": pd.concat(threshold_tables), "ablations": ablations, "robustness": robustness,
        "importance": importance_rows, "coefficients": coefficients}


def develop(raw_path, output, private, smoke=False):
    if private.resolve().is_relative_to(output.resolve()):
        raise ValueError("--private must be outside the public aggregate --output directory")
    config = protocol(smoke)
    if (output / "freeze.json").exists():
        raise FileExistsError("Frozen experiment exists; choose a new --output and --private directory")
    output.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    parts, cohort = load_partitions(raw_path, config)
    save_json(private / "partition_checks.json", private_partition_fingerprints(parts))
    save_json(output / "experiment_config.json", config)
    save_json(output / "cohort.json", cohort)
    all_results = []
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=config["threads_per_fit"]):
        warnings.simplefilter("always")
        for name in MODELS:
            result = develop_model(name, parts["train"], parts["validation"], config, private)
            all_results.append(result)
            save_json(output / (name + "_search.json"), {k: result[k] for k in ("trials", "selections", "search_seconds", "calibration_seconds")})
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    warning_counts = {}
    for warning in caught:
        key = warning.category.__name__ + ": " + str(warning.message)
        warning_counts[key] = warning_counts.get(key, 0) + 1
    save_json(output / "warnings.json", warning_counts)
    if convergence:
        raise RuntimeError("Convergence warnings occurred; development must be reviewed before freezing")
    for key in ("validation", "ablations", "robustness", "importance", "coefficients"):
        pd.DataFrame([row for result in all_results for row in result[key]]).to_csv(output / (key + ".csv"), index=False)
    pd.concat([r["thresholds"] for r in all_results]).to_csv(output / "validation_threshold_curves.csv", index=False)
    artifacts = {key: value for r in all_results for key, value in r["artifacts"].items()}
    freeze = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "source_sha256": {p: digest(PROJECT_ROOT / p) for p in SOURCE_FILES},
        "config_sha256": digest(output / "experiment_config.json"), "cohort": cohort,
        "environment": environment(), "artifacts": artifacts,
        "development_seconds": time.perf_counter() - start,
        "status": "development complete; thresholds and estimators frozen before this experiment's test evaluation"}
    save_json(output / "freeze.json", freeze)
    LOG.info("Development complete: %s; test evaluation has not run", output / "freeze.json")
    return freeze


def verify_freeze(raw_path, output, private):
    freeze = json.loads((output / "freeze.json").read_text())
    if digest(output / "experiment_config.json") != freeze["config_sha256"]:
        raise ValueError("Experiment configuration changed after freeze")
    for p, expected in freeze["source_sha256"].items():
        if digest(PROJECT_ROOT / p) != expected:
            raise ValueError(f"Source changed after freeze: {p}")
    config = json.loads((output / "experiment_config.json").read_text())
    parts, cohort = load_partitions(raw_path, config)
    private_checks = json.loads((private / "partition_checks.json").read_text())
    if private_partition_fingerprints(parts) != private_checks:
        raise ValueError("Private patient partition checks changed after freeze")
    if cohort != freeze["cohort"]:
        raise ValueError("Dataset or patient partitions changed after freeze")
    if environment() != freeze["environment"]:
        raise ValueError("Execution environment changed after freeze")
    for item in freeze["artifacts"].values():
        if digest(private / item["filename"]) != item["sha256"]:
            raise ValueError("Frozen model artifact changed")
    return freeze, config, parts


def evaluate_frozen(raw_path, output, private):
    if (output / "evaluation.json").exists():
        raise FileExistsError("Evaluation already exists; use its results, do not overwrite it")
    freeze, config, parts = verify_freeze(raw_path, output, private)
    test = parts["test"]
    predictions, thresholds, rows = {}, {}, []
    with threadpool_limits(limits=config["threads_per_fit"]):
        for name, item in freeze["artifacts"].items():
            # These are our locally trained, hash-verified artifacts, not the
            # untrusted/version-dependent historical best_classifier.pkl.
            model = joblib.load(private / item["filename"])
            model.predict_proba(test[FEATURES].iloc[:10])  # warm-up, no labels
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                p = model.predict_proba(test[FEATURES])[:, 1]
                elapsed.append(time.perf_counter() - start)
            predictions[name], thresholds[name] = p, item["threshold"]
            rows.append({"model": name, **metrics(test.readmitted_binary, p, item["threshold"]),
                "fit_seconds": item["fit_seconds"], "inference_seconds_median_3": float(np.median(elapsed))})
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({"patient_nbr": test.patient_nbr, "y": test.readmitted_binary, **predictions}).to_csv(private / "test_predictions.csv", index=False)
    LOG.info("Frozen evaluation complete; calculating paired patient-bootstrap intervals")
    ci, paired = paired_bootstrap(test.readmitted_binary, predictions, thresholds,
        config["primary_models"], config["bootstrap_replicates"])
    ci.to_csv(output / "confidence_intervals.csv", index=False)
    paired.to_csv(output / "paired_differences.csv", index=False)
    from src.benchmark_reporting import render_results, error_analysis
    error_analysis(test, predictions, thresholds, config["primary_models"]).to_csv(output / "error_analysis.csv", index=False)
    render_results(test.readmitted_binary.to_numpy(), predictions, thresholds, output, config["primary_models"])
    evaluation = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": digest(output / "freeze.json"), "environment": environment(),
        "bootstrap": {"replicates": config["bootstrap_replicates"], "seed": 2042,
            "method": "paired ordinary patient percentile bootstrap; fixed models and thresholds; pointwise unadjusted 95% intervals"},
        "test_role": config["test_history"], "test_patients": len(test),
        "reference_reproduction": None}
    if not config["smoke"]:
        reference = json.loads((PROJECT_ROOT / "reports/portfolio/baseline/metrics.json").read_text())["test_at_selected_threshold"]
        observed = table.set_index("model").loc["logistic_initial"]
        evaluation["reference_reproduction"] = {k: {"reference": reference[k], "observed": float(observed[k]),
            "absolute_difference": float(abs(observed[k] - reference[k]))} for k in
            ("roc_auc", "average_precision", "f1", "precision", "recall", "brier_score", "threshold")}
    save_json(output / "evaluation.json", evaluation)
    LOG.info("Evaluation and aggregate evidence saved to %s", output)
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["develop", "evaluate"], required=True)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_DIR / "diabetic_data.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/runs/benchmark/aggregate")
    parser.add_argument("--private", type=Path, default=PROJECT_ROOT / "reports/runs/benchmark/private")
    parser.add_argument("--smoke", action="store_true", help="600 real patients; mechanics only, never portfolio scores")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.stage == "develop":
        develop(args.raw, args.output, args.private, args.smoke)
    else:
        evaluate_frozen(args.raw, args.output, args.private)


if __name__ == "__main__":
    main()
