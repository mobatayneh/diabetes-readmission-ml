"""Scientific integrity checks with synthetic fixtures, never claimed as results."""

import json
import tempfile
import unittest
import warnings
import subprocess
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from test_baseline import sample_encounters
from src.baseline import FEATURES, NUMERIC_FEATURES, prepare_cohort, split_cohort, make_pipeline
from src.benchmark import cv_splits, search, verify_freeze, digest
from src.benchmark_models import make_model
from src.evaluation import metrics, paired_bootstrap, threshold_curve
from src.experiment_config import protocol, MODELS
from src.data_preprocessing import load_raw_data


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cohort = prepare_cohort(sample_encounters(300))
        self.train, self.validation, _ = split_cohort(self.cohort)

    def test_three_estimators_train_only_and_persist(self):
        val = self.validation.copy()
        val["race"] = "validation-only-category"
        val["num_medications"] = 99999
        for name in MODELS:
            with self.subTest(model=name), warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                m = make_model(name, threads=1, smoke=True)
                m.fit(self.train[FEATURES], self.train.readmitted_binary)
                prep = m.named_steps["features"]
                enc = prep.named_transformers_["categorical"].named_steps["encoder"]
                self.assertNotIn("validation-only-category", enc.categories_[0])
                numeric = prep.named_transformers_["numeric"].named_steps["imputer"]
                index = NUMERIC_FEATURES.index("num_medications")
                self.assertEqual(numeric.statistics_[index], self.train.num_medications.median())
                p = m.predict_proba(val[FEATURES])[:, 1]
                self.assertTrue(np.isfinite(p).all())
                with tempfile.TemporaryDirectory() as temp:
                    path = Path(temp) / "model.joblib"
                    joblib.dump(m, path)
                    np.testing.assert_allclose(joblib.load(path).predict_proba(val[FEATURES])[:, 1], p)

    def test_weighted_logistic_matches_reference(self):
        X, y = self.train[FEATURES], self.train.readmitted_binary
        old = make_pipeline().fit(X, y)
        new = make_model("logistic").fit(X, y)
        np.testing.assert_allclose(old.predict_proba(self.validation[FEATURES]),
                                   new.predict_proba(self.validation[FEATURES]), atol=1e-10)

    def test_fold_patient_isolation_and_duplicate_rejection(self):
        for tr, va in cv_splits(self.train, 3, 42):
            self.assertFalse(set(self.train.iloc[tr].patient_nbr) & set(self.train.iloc[va].patient_nbr))
        duplicate = self.train.copy()
        duplicate.iloc[1, duplicate.columns.get_loc("patient_nbr")] = duplicate.iloc[0].patient_nbr
        with self.assertRaises(ValueError):
            cv_splits(duplicate, 3, 42)

    def test_nested_calibration_runs_without_validation_fit(self):
        cv = cv_splits(self.train, 2, 42)
        estimator = search(make_model("logistic"), [{}, {"classifier__estimator__C": .1}], 2)
        model = CalibratedClassifierCV(estimator, cv=cv, method="sigmoid", ensemble=False, n_jobs=1)
        model.fit(self.train[FEATURES], self.train.readmitted_binary)
        p = model.predict_proba(self.validation[FEATURES])[:, 1]
        self.assertTrue(np.isfinite(p).all())
        selected = model.calibrated_classifiers_[0].estimator.best_estimator_
        self.assertLess(selected.named_steps["features"].named_transformers_["numeric"].named_steps["scaler"].mean_.max(), 10)

    def test_metrics_check_invalid_input_and_specificity(self):
        m = metrics([0, 0, 1, 1], [.1, .8, .2, .9], .5)
        self.assertEqual((m["tn"], m["fp"], m["fn"], m["tp"]), (1, 1, 1, 1))
        self.assertEqual(m["specificity"], .5)
        for p in ([0, 1], [.1, .2, np.nan, .5], [.1, .2, .3, 1.5]):
            with self.assertRaises(ValueError):
                metrics([0, 0, 1, 1], p, .5)

    def test_bootstrap_is_paired_and_reproducible(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([.1, .2, .6, .4, .8, .9])
        names = ["a_tuned", "a_initial", "a_calibrated", "b_tuned", "b_initial", "b_calibrated"]
        predictions = {k: p for k in names}
        thresholds = {k: .5 for k in names}
        ci, paired = paired_bootstrap(y, predictions, thresholds, ["a_tuned", "b_tuned"], replicates=8)
        self.assertTrue((paired[["difference", "low", "high"]] == 0).all().all())
        again, _ = paired_bootstrap(y, predictions, thresholds, ["a_tuned", "b_tuned"], replicates=8)
        self.assertTrue(ci.equals(again))

    def test_threshold_curve_has_only_validation_candidates(self):
        curve = threshold_curve([0, 0, 1, 1], [.1, .2, .3, .4])
        self.assertEqual(len(curve), 91)
        self.assertEqual(curve.f1.max(), 1)

    def test_modified_config_cannot_reach_test_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            config = output / "experiment_config.json"
            config.write_text("{}")
            (output / "freeze.json").write_text(json.dumps({"config_sha256": digest(config)}))
            config.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                verify_freeze(output / "never_read.csv", output, output)

    def test_candidate_budget_and_information_are_fixed(self):
        cfg = protocol()
        self.assertEqual(cfg, protocol())
        self.assertTrue(all(len(v) == 7 for v in cfg["candidates"].values()))
        self.assertEqual(len(FEATURES), 22)
        self.assertFalse(set(FEATURES) & {"patient_nbr", "encounter_id", "readmitted", "readmitted_binary"})

    def test_generic_loader_preserves_uci_literal_none(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "diabetic_data.csv"
            path.write_text("A1Cresult,race\nNone,?\n>8,Caucasian\n")
            loaded = load_raw_data(str(path))
            self.assertEqual(loaded.A1Cresult.iloc[0], "None")
            self.assertTrue(loaded.race.isna().iloc[0])

    def test_two_process_cli_round_trip_and_no_overwrite(self):
        # Independent interpreters catch persistence/entry-point failures that
        # in-process estimator checks cannot. All records here are synthetic.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "synthetic.csv"
            sample_encounters(900).to_csv(raw, index=False)
            output, private = root / "aggregate", root / "private"
            command = [sys.executable, "-m", "src.benchmark", "--raw", str(raw),
                       "--output", str(output), "--private", str(private)]
            for stage in ("develop", "evaluate"):
                result = subprocess.run(command + ["--stage", stage, "--smoke"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            self.assertTrue((output / "evaluation.json").exists())
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 6)
            self.assertTrue((private / "test_predictions.csv").exists())
            self.assertFalse((output / "test_predictions.csv").exists())
            again = subprocess.run(command + ["--stage", "evaluate"], capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("Evaluation already exists", again.stderr)


if __name__ == "__main__":
    unittest.main()
