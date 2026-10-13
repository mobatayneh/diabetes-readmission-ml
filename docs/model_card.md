# Model card: controlled diabetes readmission benchmark

## Intended use and provenance

Research and engineering education: compare Logistic Regression, LightGBM and XGBoost for the observed UCI `<30` readmission label. Prediction is defined at the end of an index hospitalization, assuming final fields are available. The repository is not a clinical decision system and does not support admission-time prediction with these inputs.

The original academic work belongs to **Christopher Indris, Mojtaba Mohammadi, and Mohammad Al Batayneh**. This extension was developed with AI coding assistance and verified through the recorded experiments and tests; it does not establish any individual contributor's sole authorship or independent mastery. Original PDFs, slides, stored model and aggregate artifacts are retained. Data attribution: Clore, Cios, DeShazo and Strack, UCI, [DOI 10.24432/C5230J](https://doi.org/10.24432/C5230J), CC BY 4.0. No new repository-wide license is asserted over the team's work.

## Data and scope

- 101,766 encounters from the UCI 1999–2008 dataset; 69,990 eligible unique patients after the documented restrictions.
- One record per patient, chosen using the smallest eligible encounter identifier; no verified temporal order.
- Positive: recorded readmission in fewer than 30 days. `NO` and `>30` form the negative class; this label does not prove complete follow-up across every healthcare provider.
- Twenty-two predictors, including demographics, prior utilization, diagnosis, hospitalization totals, tests, treatment and discharge disposition.
- The data are old and reflect participating institutions and historical care. Selection, coding, access-to-care and missingness biases can affect generalization.

## Training and evaluation

Fixed seed 42, patient-disjoint training/validation/test partitions of 70%/15%/15%. Train-fitted preprocessing is inside every CV pipeline. Hyperparameters maximize training-fold average precision. Validation F1 selects thresholds; initial, tuned and nested-CV calibrated variants are all retained. See [methodology](methodology.md) for budgets, weighting, calibration, exact prediction-time assumptions and interval construction.

[Measured model comparison](model_comparison.md) links the full test tables, paired bootstrap intervals, calibration curves, validation sensitivity analyses and aggregate error analysis. A training-prevalence constant probability is an additional calibration reference in that report, not a tuned predictor.

## Appropriate interpretation

- Test scores describe one reused random holdout, previously inspected for the portfolio baseline. It is not an external, temporal, prospective or pristine confirmatory validation.
- Model differences are conditional on this data, split, finite candidate search and threshold rule. Bootstrap intervals exclude training and tuning uncertainty and are unadjusted for multiple comparisons.
- Important predictors and logistic coefficients are model associations. They do not identify causes, treatment effects or reasons an individual patient was readmitted.
- Aggregate false-positive/false-negative profiles identify patterns for further investigation; clinical error causes are not available in this dataset.
- Subgroup slices omit groups below 50 patients, report remaining sample sizes, and suppress metrics for groups with fewer than 10 positive or negative cases. They do not establish fairness or safety, and are not sufficiently powered for strong intersectional conclusions.
- Numeric zeros in sparse XGBoost input follow its default missing-branch semantics. Different encodings and model families can alter results even with the same underlying information.
- Probability calibration is assessed empirically; a low Brier score alone does not prove clinically usable risk estimates. Compare it with the prevalence-only reference and reliability diagrams.

## Deployment boundary and future work

No deployment, patient intervention, economic savings, prospective utility or clinical certification is claimed. Before any real use, independent clinical governance and validation would need to establish target capture, feature timing, transportability, calibration, subgroup behavior and intervention consequences. Suitable independent contemporary data would be needed for external validation; reliable dates and institution information would be needed for temporal or site-based validation. They cannot be reconstructed from encounter IDs.

Future methodological work can use a **new evaluation protocol** to explore richer features, broader hyperparameter searches, nested repeated evaluation, missingness indicators, operational capacity constraints, or additional explanations. Do not continue optimizing against the already reported test set.

## Artifact handling

Committed artifacts are code, configuration, source hashes, aggregate metrics and plots. Raw/processed patient data, per-patient predictions and new fitted models stay in ignored local directories. `freeze.json` records their hashes. The old pickle remains an unvalidated historical artifact; the new experiment does not load it. Only deserialize trusted locally generated models.
