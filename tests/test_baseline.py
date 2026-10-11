"""Regression checks for patient separation, label handling and fitted preprocessing."""

import unittest
import numpy as np
import pandas as pd

from src.baseline import (
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    prepare_cohort, split_cohort, make_pipeline, choose_threshold, evaluate,
)


def sample_encounters(n=300):
    # Synthetic fixtures verify mechanics only; they are never reported as model results.
    df = pd.DataFrame({c: np.arange(n) % 7 + 1 for c in NUMERIC_FEATURES})
    for column in CATEGORICAL_FEATURES:
        df[column] = "A"
    df["admission_type_id"] = "1"
    df["discharge_disposition_id"] = "1"
    df["admission_source_id"] = "1"
    df["max_glu_serum"] = "None"
    df["A1Cresult"] = "None"
    df["patient_nbr"] = np.arange(n)
    df["encounter_id"] = np.arange(n) + 1000
    df["readmitted"] = np.where(np.arange(n) % 5 == 0, "<30", "NO")
    return df


class BaselineTests(unittest.TestCase):
    def test_labels_are_explicit(self):
        df = sample_encounters(3)
        df["readmitted"] = ["NO", ">30", "<30"]
        self.assertEqual(prepare_cohort(df).readmitted_binary.tolist(), [0, 0, 1])
        df.loc[0, "readmitted"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unexpected readmission"):
            prepare_cohort(df)

    def test_discharge_exclusions_and_first_eligible_encounter(self):
        df = sample_encounters(5)
        df["patient_nbr"] = [1, 1, 2, 3, 4]
        df["discharge_disposition_id"] = ["11", "1", "13.0", "1", "21"]
        result = prepare_cohort(df)
        self.assertEqual(result.patient_nbr.tolist(), [1, 3])
        self.assertEqual(result.encounter_id.tolist(), [1001, 1003])

    def test_missing_identifiers_fail(self):
        df = sample_encounters()
        df.loc[0, "patient_nbr"] = np.nan
        with self.assertRaisesRegex(ValueError, "must be present"):
            prepare_cohort(df)

    def test_splits_are_disjoint_reproducible_and_exhaustive(self):
        cohort = prepare_cohort(sample_encounters())
        split1 = split_cohort(cohort)
        split2 = split_cohort(cohort)
        ids = [set(frame.patient_nbr) for frame in split1]
        self.assertEqual([len(f) for f in split1], [210, 45, 45])
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(set.union(*ids), set(cohort.patient_nbr))
        for first, second in zip(split1, split2):
            self.assertEqual(first.index.tolist(), second.index.tolist())
            self.assertEqual(first.readmitted_binary.nunique(), 2)
        with self.assertRaisesRegex(ValueError, "Repeated patients"):
            split_cohort(pd.concat([cohort, cohort.iloc[[0]]]))

    def test_preprocessing_learns_only_training_values(self):
        cohort = prepare_cohort(sample_encounters())
        train, validation, _ = split_cohort(cohort)
        validation = validation.copy()
        validation["race"] = "unseen-in-training"
        validation["num_medications"] = 1000000
        model = make_pipeline().fit(train[FEATURES], train.readmitted_binary)
        transformer = model.named_steps["features"]
        categories = transformer.named_transformers_["categorical"].named_steps["encoder"].categories_[0]
        self.assertNotIn("unseen-in-training", categories)
        scaler = transformer.named_transformers_["numeric"].named_steps["scaler"]
        position = NUMERIC_FEATURES.index("num_medications")
        self.assertAlmostEqual(scaler.mean_[position], train.num_medications.mean())
        probabilities = model.predict_proba(validation[FEATURES])[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertNotIn("patient_nbr", FEATURES)
        self.assertNotIn("readmitted", FEATURES)

    def test_threshold_and_metrics(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        threshold, score = choose_threshold(labels, probabilities)
        self.assertAlmostEqual(score, 1.0)
        metrics = evaluate(labels, probabilities, threshold)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 0, "tp": 2})
        self.assertEqual(metrics["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
