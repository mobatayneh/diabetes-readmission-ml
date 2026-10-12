# Controlled benchmark methodology

## Question and prediction time

Predict the UCI label `readmitted == "<30"` versus `NO` / `>30`, **at the end of the index hospitalization**, assuming final encounter fields can then be obtained. This is an offline retrospective study, not an admission-time alert or a deployed medical device. The source records contain no reliable field-level timestamps with which to verify real operational availability.

| Predictive information | Availability assumption and limitation |
| --- | --- |
| `race`, `gender`, `age` | Recorded demographics; age is a band. Categories can be missing, inaccurate, or reflect social measurement rather than biological mechanisms. |
| `admission_type_id`, `admission_source_id` | Recorded admission context, assumed available by discharge. |
| `number_outpatient`, `number_emergency`, `number_inpatient` | Encounters in the year preceding hospitalization, according to the data dictionary; incomplete capture across institutions may bias them. |
| `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_diagnoses` | Totals over the index encounter; require encounter completion. Not legitimate admission-time predictors. |
| `diag_1`, `diag_2`, `diag_3` | Index-encounter diagnoses. Final coding may arrive after discharge in a real system; operational timing is unverified. |
| `max_glu_serum`, `A1Cresult` | Index-encounter test categories. Literal `None` is preserved; `?` is missing. Assume values are finalized by the prediction time. |
| `insulin`, `change`, `diabetesMed` | Index-encounter medication information; require completed treatment records. Association is not evidence of medication effects. |
| `discharge_disposition_id` | Final discharge destination/outcome code. Included for a discharge-time study, with a development-only omission sensitivity. It is not assumed available earlier. |

The target, target-derived columns, `patient_nbr` and `encounter_id` never enter the feature matrix. No feature selection based on outcomes outside training folds occurs.

## Cohort and partitions

Reuse `src.baseline.prepare_cohort` and `split_cohort`: load the official CSV with `keep_default_na=False, na_values=["?"]`; reject missing IDs/unknown labels; exclude discharge codes 11, 13, 14, 19, 20 and 21; sort numeric encounter IDs stably and keep one eligible encounter per patient. This produces 69,990 patients from 101,766 encounters. Selecting the smallest ID is reproducible **but is not proof of chronological ordering**. Exclusions define the study population and affect generalizability.

Two stratified splits with seed 42 give training 48,993; validation 10,498; test 10,499. IDs are retained only for partition checks and ignored local prediction files. All pairwise patient overlaps and all CV-fold patient overlaps must be zero. With one row per patient, stratified folds are patient-disjoint; a future multiple-encounter protocol would require grouped splitting.

The test set is the **previously inspected portfolio holdout**. It supports a controlled comparison using the same patients, not a newly untouched external confirmation. In this extension, all configurations, estimators and thresholds are frozen before new test scores are calculated. Stratification and partition-count checks necessarily access outcome labels during split construction; those are not outcome-driven model selection.

## Models, selection and preprocessing

- Logistic Regression: liblinear, L1/L2 via `l1_ratio` 1/0, maximum 1,000 iterations; numeric median imputation and standard scaling.
- LightGBM and XGBoost: bounded CPU tree ensembles; numeric median imputation without scaling. LightGBM uses deterministic column-wise training; XGBoost uses CPU histogram trees.
- All three: the same 22 raw predictors, constant categorical missing token, training-fitted one-hot encoding with unseen categories ignored. Sparse one-hot matrices are used. XGBoost routes absent sparse entries as missing/default branches; this representation choice is part of its preprocessing, not extra patient information.
- Class balancing uses per-fit sample weights computed only from that fold's labels. The mean weight is one. No SMOTE or other synthetic categorical records are used.
- Seven candidates per algorithm: the initial configuration plus six seeded random draws from the documented search space. `GridSearchCV` evaluates this fixed sampled candidate list over the same three stratified **training** folds. Selection maximizes mean average precision; ties follow candidate order. Search candidates run sequentially, with two computational threads per fit. Equal candidate/fold budgets do not imply equal wall time.
- Save all candidate settings, fold AP, CV means, fit timings and selections. The development execution trace also records searches nested inside calibration. The search is deliberately small; unsampled combinations and other solvers remain untested.

Initial and tuned models are fitted on the training partition only. Validation labels select a separate threshold for every model using maximum F1 over 0.05–0.95 in 0.01 steps; exact ties prefer the higher threshold, matching the original reference. This grid can miss an optimum outside its range. F1 is a research objective, not an established clinical utility function.

The **primary comparison is the three tuned models**. Initial and calibrated variants are prespecified secondary analyses; all nine are reported, with no winner selected from test scores. No refitting on training plus validation occurs.

## Calibration and development sensitivity

Sigmoid calibration uses `CalibratedClassifierCV(ensemble=False)` around the complete parameter-search estimator. Each of three calibration folds repeats the same candidate search using only that fold's training subset. Its held-out labels fit the sigmoid calibrator, without entering preprocessing or hyperparameter choice for those predictions. The final search is refitted on the full training partition and must select the same settings as the primary tuned model. Validation is used only for its numerical threshold. Calibration is prespecified for all three algorithms; it is not selected using test performance.

Three independent, single-factor sensitivities start from each selected tuned configuration: toggle class weighting; pool categorical levels observed fewer than 20 times in the fitting data; remove discharge disposition. Each gets three-fold training CV and validation evaluation; all outcomes, including losses, are reported. They are exploratory and are never promoted to the frozen test models.

Selected configurations are also evaluated using a second training CV partition, seed 43. The estimator seed remains 42, isolating partition variability. These are **not nested or unbiased generalization estimates**, because the configuration was selected using the training data. This is a limited robustness check, not a repeated external validation.

## Metrics, uncertainty and explanations

Average precision (AP) is `average_precision_score`, the recall-weighted precision summary; it is not trapezoidal area under the PR curve. Report ROC-AUC, AP, precision, recall, specificity, F1, balanced accuracy, Brier score, log loss, confusion counts, prevalence and the frozen threshold. Fit and inference timings are local wall times, with median inference over three repetitions after warm-up; hardware comparisons are not implied.

Use 1,000 ordinary bootstrap resamples of test patients, seed 2042. The same sampled indices are used for all models, yielding paired differences and pointwise 95% percentile intervals. Models and thresholds remain fixed. These intervals quantify test-sample uncertainty conditional on training and selection, not training/split/threshold uncertainty. They assume patient-level independence; site dependence cannot be assessed. No multiplicity correction or formal significance declaration is made.

Permutation importance measures AP decrease on a stratified 2,000-patient validation sample with five repeats. Error bars show variability across permutations, **not** confidence intervals over independent studies. Correlated predictors can mask or share importance. Logistic coefficients describe conditional model associations (numeric predictors per training SD; categorical indicators), not causal effects or validated individual explanations. No new SHAP or partial-dependence study is claimed.

Test error analysis publishes only group counts/metrics and feature medians for true/false positive/negative groups. Subgroup metrics require at least 50 patients, 10 positives and 10 negatives; otherwise only counts are retained. These are descriptive slices without subgroup uncertainty estimates or fairness certification. No row-level patient records or fitted models are committed.

## Reproduction and evidence

From the repository root using Python 3.12:

```bash
python -m pip install -r requirements-benchmark.txt
python -m src.download_data
python -m src.benchmark --stage develop
python -m src.benchmark --stage evaluate
python -m unittest discover -s tests -v
```

Default aggregate and private outputs are under ignored `reports/runs/benchmark/`. The committed reference aggregates live in `reports/benchmark/`. A completed run is not silently overwritten: use new `--output` and `--private` directories for a different experiment. The evaluation verifies source/config/data/model/environment hashes from `freeze.json` first. Use `--smoke` during development for 600 real patients; it tests mechanics and its results must not replace full-cohort scores.

Direct computational dependencies are pinned in `requirements-benchmark.txt` plus core/summary requirements. `environment.json`/freeze metadata records the measured environment; this is not a complete transitive package or OS lock. Linux/Python 3.12 is the validated target; Windows/macOS are not independently verified.

## Sources

- [Official UCI dataset and variable documentation](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008), [dataset DOI](https://doi.org/10.24432/C5230J), CC BY 4.0.
- [scikit-learn probability calibration](https://scikit-learn.org/1.8/modules/calibration.html), [cross-validation](https://scikit-learn.org/1.8/modules/cross_validation.html), and [average precision](https://scikit-learn.org/1.8/modules/generated/sklearn.metrics.average_precision_score.html).
- [LightGBM 4.6 classifier documentation](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html).
- [XGBoost parameters](https://xgboost.readthedocs.io/en/release_3.0.0/parameter.html).
