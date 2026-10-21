# Resume wording and interview preparation

These descriptions summarize the implemented project. The original academic work was collaborative. Choose wording consistent with your actual role and understanding; these are project descriptions, not claims that one person independently performed every experiment or original contribution. No CV file was changed.

## Short project entry

**Diabetes Readmission ML — Collaborative Academic Project & Portfolio Extension**  
Python · scikit-learn · LightGBM · XGBoost · GitHub Actions

Reproducible comparison of three classifiers on 69,990 eligible patients, with patient-disjoint splits, training-only preprocessing, cross-validated tuning, validation-selected thresholds and paired bootstrap evaluation. Tuned models achieved approximately 0.65 ROC-AUC; results include calibration diagnostics, error analysis and documented limitations.

## Longer project explanation

This project investigates readmission within 30 days using the UCI Diabetes 130-US Hospitals dataset. The original academic team—Christopher Indris, Mojtaba Mohammadi and Mohammad Al Batayneh—combined research, exploratory analysis, supervised modeling and patient segmentation. A later portfolio extension preserves that work and adds an executable three-model comparison on a common cohort and split.

The extension compares Logistic Regression, LightGBM and XGBoost using fold-local preprocessing, bounded AP-based hyperparameter searches, validation-F1 decision thresholds and 1,000 paired patient bootstrap resamples. It includes nested training-only probability calibration, sensitivity studies, permutation importance, aggregate error analysis and automated checks. The tuned models have similar held-out discrimination, around ROC-AUC 0.65, without a clear pairwise advantage. This is a retrospective research exercise with a previously inspected holdout and no external or prospective clinical validation.

## A concise technical walkthrough

1. **Problem:** identify recorded readmissions in fewer than 30 days, at discharge, while treating false positives and missed cases as explicit evaluation concerns.
2. **Data:** 101,766 encounters; exclude specified terminal/hospice discharge codes and retain one eligible encounter per patient using a deterministic identifier ordering proxy, leaving 69,990 patients.
3. **Features:** 22 recorded predictors; remove patient/encounter identifiers and target-derived information. Preserve literal laboratory `None` categories and treat `?` as missing.
4. **Preprocessing:** median imputation for numeric fields; scaling for logistic regression; missing categorical token and training-fitted one-hot encoding for all models. Fit these operations within each training fold.
5. **Models:** logistic regression offers a linear reference; LightGBM and XGBoost allow nonlinear tree interactions using the same predictive information.
6. **Selection:** seven configurations per algorithm, three-fold stratified training CV and mean AP. Class weighting is a candidate choice. No SMOTE on one-hot clinical categories.
7. **Decision rule:** validation F1 selects each model's threshold; the primary tuned models independently select 0.11. Freeze these choices before evaluating the test set.
8. **Evaluation:** the common 10,499-patient test set has 943 positives. Report discrimination, probability quality, confusion counts and paired uncertainty. The holdout was already used for an earlier baseline, which limits confirmatory interpretation.
9. **Findings:** tuned logistic improved ROC-AUC/AP over its initial baseline; the tuned algorithms are close to one another. Calibration did not reduce Brier point estimates. Discharge disposition and previous inpatient visits were prominent validation permutation associations.
10. **Limits:** historical data, modest recall/precision, unknown operational timestamps, limited tuning, one reused random holdout, no external/temporal/clinical validation. Further development needs a new evaluation design.

## Questions and evidence-based answers

| Interview question | Answer grounded in this repository |
| --- | --- |
| Why not report accuracy as the main success measure? | About 9% of eligible patients are positive. Predicting everyone negative would be about 91% accurate but have zero positive recall. AP, ROC-AUC and operating-point precision/recall are more informative here. |
| How did you prevent patient leakage? | Deduplicate eligible encounters before splitting, retain IDs only for checks, assert zero overlap across partitions and folds, and exclude identifiers from predictors. Multiple-encounter work would need group-aware splitting. |
| Is the earliest encounter really chronological? | No. The smallest numeric encounter ID is a reproducible proxy, not a verified date. The limitation is explicit. |
| Could discharge variables leak future information? | They would be unavailable for admission-time prediction. This task is defined at discharge, but final coding delays are unverified; omission sensitivity shows how much results depend on disposition. |
| How are hyperparameters and thresholds selected? | Training CV AP selects settings; a separate validation set selects F1 thresholds over a fixed grid. Test scores do not feed either selection. |
| Was the test set completely untouched? | Not historically: the initial portfolio baseline already used it. The extension froze decisions before its new evaluation, but makes no pristine confirmatory claim. |
| Why keep Logistic Regression? | It is a reproducibility reference and competitive under this protocol. Tuned test AP is 0.1706 versus 0.1766 for LightGBM; the paired difference interval includes zero. Complexity alone did not establish a clear gain. |
| Why not apply SMOTE automatically? | Ordinary SMOTE can interpolate one-hot categorical values into implausible fractional categories. The controlled benchmark evaluates per-fit weighting instead; no oversampling touches validation or test data. |
| What does AP mean? | Average precision is recall-increment-weighted precision. It differs from trapezoidal PR area. Historical `auprc_mean` refers to AP, while `pr_auc_mean` refers to a different calculation. |
| How does paired bootstrap help? | Each resample selects the same test patients for every model, preserving correlation between their errors when estimating a difference. It does not capture retraining or tuning uncertainty. |
| Which model won? | None established a clear pairwise advantage in ROC-AUC, AP or F1. LightGBM has the largest AP point estimate; XGBoost the largest primary F1 point estimate. The differences are small and their intervals include zero. |
| Did calibration improve the models? | Not by test Brier point estimates in this run. The sigmoid was fitted using nested training-only out-of-fold predictions; ranking metrics stayed unchanged and thresholded predictions changed. Negative results remain documented. |
| Do feature importance and coefficients explain clinical causes? | No. They describe model associations and can reflect correlated variables, treatment decisions or data capture. We did not establish causal effects or validate individual clinical explanations. |
| How was software verified? | Unit and synthetic integration tests cover all three estimators, train-only transformations, partition checks, serialization, nested calibration, metrics, configuration tampering and separate-process execution. The full real-data benchmark was also executed; hosted CI evidence is linked in the validation record. |
| What would you do next? | Secure suitable independent data and verified timing, design a new evaluation protocol, then assess external generalization, calibration, subgroup uncertainty and operational capacity before considering any clinical use. |

For exact numbers and uncertainty, use [model comparison](model_comparison.md), not the original academic CV table as a substitute for new results.
