"""Reproducible patient-disjoint logistic baseline on the original UCI CSV.

This is a new, deliberately compact experiment, separate from the original
notebook benchmarks. Threshold selection uses validation labels only.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.paths import RAW_DATA_DIR, REPORTS_DIR

NUMERIC_FEATURES = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
]
CATEGORICAL_FEATURES = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "diag_1", "diag_2", "diag_3", "max_glu_serum",
    "A1Cresult", "insulin", "change", "diabetesMed",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_DISCHARGES = {11, 13, 14, 19, 20, 21}


def prepare_cohort(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep one eligible encounter per patient and create the binary target."""
    required = set(FEATURES + ["patient_nbr", "encounter_id", "readmitted"])
    if missing := required.difference(raw.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = raw.copy()
    if df[["patient_nbr", "encounter_id", "readmitted"]].isna().any().any():
        raise ValueError("Patient IDs, encounter IDs and readmission labels must be present")
    if not set(df["readmitted"]).issubset({"NO", ">30", "<30"}):
        raise ValueError("Unexpected readmission label; expected NO, >30 or <30")
    discharge = pd.to_numeric(df["discharge_disposition_id"], errors="raise")
    df = df.loc[~discharge.isin(EXCLUDED_DISCHARGES)].copy()
    df["encounter_id"] = pd.to_numeric(df["encounter_id"], errors="raise")
    df = df.sort_values("encounter_id", kind="stable").drop_duplicates("patient_nbr", keep="first")
    df["readmitted_binary"] = df["readmitted"].eq("<30").astype(int)
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return df.reset_index(drop=True)


def split_cohort(cohort: pd.DataFrame, seed=42):
    """Stratify 70/15/15 splits after enforcing one row per patient."""
    if cohort["patient_nbr"].duplicated().any():
        raise ValueError("Repeated patients must be resolved before splitting")
    train, remainder = train_test_split(
        cohort, test_size=0.30, random_state=seed, stratify=cohort["readmitted_binary"],
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=seed, stratify=remainder["readmitted_binary"],
    )
    return train, validation, test


def make_pipeline(seed=42):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ])),
        ("classifier", LogisticRegression(
            solver="liblinear", class_weight="balanced", max_iter=1000, random_state=seed,
        )),
    ])


def choose_threshold(y_validation, probabilities):
    """Choose highest validation F1; break ties toward the higher threshold."""
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_validation, np.asarray(probabilities) >= t, zero_division=0) for t in candidates]
    best = max(range(len(candidates)), key=lambda i: (scores[i], candidates[i]))
    return float(candidates[best]), float(scores[best])


def evaluate(y_true, probabilities, threshold):
    predictions = np.asarray(probabilities) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def run(raw_path: Path, output: Path, seed=42, max_rows=None):
    raw_path, output = Path(raw_path), Path(output)
    raw = pd.read_csv(raw_path, keep_default_na=False, na_values=["?"], low_memory=False)
    cohort = prepare_cohort(raw)
    eligible_rows = len(cohort)
    if max_rows is not None:
        if max_rows < 200:
            raise ValueError("--max-rows must be at least 200")
        if max_rows < len(cohort):
            cohort, _ = train_test_split(
                cohort, train_size=max_rows, random_state=seed, stratify=cohort["readmitted_binary"],
            )
    train, validation, test = split_cohort(cohort, seed)
    model = make_pipeline(seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train[FEATURES], train["readmitted_binary"])
    validation_prob = model.predict_proba(validation[FEATURES])[:, 1]
    threshold, validation_f1 = choose_threshold(validation["readmitted_binary"], validation_prob)
    # Test labels are first used for reporting after all fitting and selection.
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    splits = {"train": train, "validation": validation, "test": test}
    result = {
        "experiment": "patient_disjoint_logistic_baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": "https://doi.org/10.24432/C5230J",
        "raw_csv_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed,
        "raw_rows": len(raw),
        "eligible_unique_patients": eligible_rows,
        "used_rows": len(cohort),
        "features": FEATURES,
        "split_counts": {name: {"rows": len(frame), "positive": int(frame.readmitted_binary.sum())} for name, frame in splits.items()},
        "patient_overlap": {
            "train_validation": len(set(train.patient_nbr) & set(validation.patient_nbr)),
            "train_test": len(set(train.patient_nbr) & set(test.patient_nbr)),
            "validation_test": len(set(validation.patient_nbr) & set(test.patient_nbr)),
        },
        "selection": {"metric": "validation_f1", "threshold": threshold, "validation_f1": validation_f1},
        "test_at_selected_threshold": evaluate(test.readmitted_binary, test_prob, threshold),
        "test_at_default_threshold": evaluate(test.readmitted_binary, test_prob, 0.5),
        "convergence_warnings": [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)],
        "environment": {"python": platform.python_version(), **{p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")}},
        "limitations": [
            "Single random patient-disjoint split; no temporal or external validation.",
            "The smallest encounter_id is used to select one eligible encounter per patient, not a verified timestamp.",
            "Class weighting does not guarantee calibrated probabilities.",
            "This new baseline is not directly comparable to the historical notebook CV summaries.",
        ],
    }
    if any(result["patient_overlap"].values()):
        raise AssertionError("Patient overlap detected")
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": len(cohort), "test": result["test_at_selected_threshold"], "output": str(output)}, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DATA_DIR / "diabetic_data.csv")
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "runs" / "baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)
    if not args.raw.exists():
        parser.error("Raw dataset missing. Run: python -m src.download_data")
    run(args.raw, args.output, args.seed, args.max_rows)


if __name__ == "__main__":
    main()
