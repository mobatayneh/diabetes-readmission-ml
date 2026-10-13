# Controlled benchmark execution and verification

Executed locally on **2026-10-12**, Python **3.12.14**, Linux x86_64. The reference data and original portfolio baseline were verified before new development. The benchmark is a separate branch extension; the original academic results remain historical.

## Evidence actually obtained

| Check | Evidence / result |
| --- | --- |
| Reference logistic reproduction | Full real-data run reproduced all stored metrics and threshold exactly. [Numeric differences](../reports/benchmark/evaluation.json) are zero. |
| Real cohort | 101,766 raw encounters; 69,990 eligible unique patients; 22 predictors. [Aggregate cohort manifest](../reports/benchmark/cohort.json). |
| Partitions | Train 48,993 / validation 10,498 / test 10,499; all three pairwise patient overlaps zero. Training CV requires unique patients and checks fold separation. |
| Full development | Three algorithms; seven candidates each; three-fold training CV; initial and selected full-training fits; nested training-only sigmoid calibration; nine one-factor sensitivities; second CV partition; validation permutation analysis. Completed in 258.857 seconds locally. |
| Selection evidence | Full candidate settings, fold AP and selected configurations in three search JSON files; nested calibration searches in [development trace](../reports/benchmark/development_trace.txt). No convergence warnings. |
| Frozen evaluation | Nine prespecified variants evaluated on the same 10,499 patients after configuration, estimators and thresholds were frozen. [All metrics](../reports/benchmark/metrics.csv). |
| Uncertainty | 1,000 shared patient bootstrap draws, fixed models/thresholds, pointwise 95% intervals and paired differences; no significance or equivalence certification. |
| Diagnostic artifacts | Six figures generated and visually inspected: ROC/PR, confusion matrices, reliability diagrams, validation threshold curves, metric intervals and permutation importance. |
| Automated tests | 21 tests pass locally, including all three estimators, train-only transformations, reference equivalence, CV isolation, nested calibration, paired bootstrap, tamper rejection, UCI missing tokens and a synthetic CLI round trip across separate Python processes. |
| Real-data smoke run | Separate 600-patient development/evaluation completed; its scores are not used as portfolio results. |
| Compilation / notebooks | Python compilation passed. All five notebooks validated structurally and code cells compiled. Notebook 04 executed all six code cells from committed aggregate references. |
| Historical asset preservation | All 84 tracked files under `reports/` and `presentation/` at `681b9859da75f3da4a4eea610072e46c24d8958c` are byte-identical. This includes the original PDFs, LaTeX/bibliographies, model, figures, tables and initial baseline evidence. The user's team-collaboration paragraph is retained verbatim. |
| Public output separation | No patient-level CSV, new fitted model or patient-ID fingerprint is published. Partition identity checks and original integrity records stay local. |
| Hosted CI on executed source snapshot | [Run 38300649378](https://github.com/mobatayneh/diabetes-readmission-ml/actions/runs/38300649378), commit `fd33684c2b45f9c295ee2e2140c404d3ac27b77c`: successful installation, compilation, tests and notebook 04 execution. This run covers that source snapshot; later publication maintenance is recorded separately when verified. |

The reference freeze/evaluation JSON includes exact package versions: NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost CPU 3.0.5, joblib 1.5.3, matplotlib 3.10.8, threadpoolctl 3.7.0. Direct pins do not constitute a complete transitive/OS lock. Timings include normal local system variability and are not controlled hardware benchmarks.

## Resumption audit, 2026-10-13

Before final publication, GitHub contained the executed source snapshot `fd33684`; the final aggregate results and documentation were still local. The resumed review checked those existing artifacts without retraining or changing the protocol, models, thresholds or results. The machine-readable [verification record](../reports/benchmark/resumption_verification.json) records the checks:

- Original source, raw-data and fitted-model hashes match their frozen records; reconstructed patient partitions and labels match the saved predictions, with zero overlap.
- All nine fitted models reproduce their saved test predictions. CSV probabilities are restored to each estimator's native floating-point dtype, including XGBoost float32. Every recorded test metric matches exactly; validation metrics and the previously frozen thresholds also match.
- All 81 interval rows and 81 paired-difference rows reproduce from the original 1,000 shared bootstrap draws within absolute tolerance `1e-12`. No new model fitting or test-driven selection is involved.
- All six figures regenerate byte-for-byte. README interval tables, 27 additional numeric comparison-table rows, the constant reference and subgroup/error summaries match the underlying evidence.
- All 21 tests pass again (16.302 seconds); Python modules compile, and all five notebooks validate structurally with compilable code cells. Historical notebooks were not newly executed end to end.
- Documentation links resolve, all 84 original report/presentation files remain byte-identical, and private patient fingerprints are absent from tracked text artifacts.

Hosted checks and their exact commits are available in the review pull request's Checks tab. The earlier hosted run above remains evidence for its specifically identified source snapshot.

## Source versions and publication maintenance

Run-directory protection was verified on 2026-10-13: development now rejects nonempty private directories before reading data or writing outputs. A regression test protects interrupted-run files, and the two-process integration test attempts a second development run with a new aggregate path but the previous private directory. The attempt is rejected, every existing private file retains its hash, and no new aggregate directory is created. Existing empty private directories remain supported. All 22 tests passed locally in 18.783 seconds; Python compilation passed. This guard does not change any recorded scientific results or rerun the real-data experiment.

The executed code was recorded under local source commit `027c369` and published as [`fd33684c2b45f9c295ee2e2140c404d3ac27b77c`](https://github.com/mobatayneh/diabetes-readmission-ml/commit/fd33684c2b45f9c295ee2e2140c404d3ac27b77c). Their source trees are identical: `b78f7070f0453748b0c80f38b6c29d580920add3`. Every source SHA-256 in the reference freeze manifest matches that executed source snapshot.

The original locally frozen integrity records included fingerprints of ordered patient-ID lists. Publication review rejected exporting those identifier-derived values. They were retained privately and removed from the public cohort/freeze copies. The public evaluation record explicitly labels this sanitization rather than claiming byte identity with the original private freeze file.

The current pipeline writes these fingerprints only to `--private/partition_checks.json`, verifies them before evaluation, and rejects placing the private directory inside aggregate output. This post-run maintenance changes serialization/privacy boundaries and omits subgroups below 50 patients from publication only; it does not change cohort membership, model fitting, thresholds, predictions or any measured metric. The complete real-data experiment was **not retuned or rerun after looking at its results**. Current code was checked through the synthetic process-level integration test. The executed source remains available for audit.

Additional maintenance corrected the historical notebook's signed-value threshold direction and misleading cost labels, case-sensitive documentation links, an imbalanced-target description, and UCI `None` handling in the generic loader. Original historical output artifacts were not regenerated to appear newly executed.

## Known warnings and limits

LightGBM emitted 237 development warnings about unnamed transformed arrays versus internally generated feature names. The same fitted column transformer supplies train/validation/test arrays, so column ordering is preserved; all warnings are counted in [warnings.json](../reports/benchmark/warnings.json). No convergence warnings occurred. The evaluation trace retains corresponding warnings rather than silently hiding them.

Not independently rerun: complete historical notebooks 01, 03 and 05; historical optional research stack; old pickle compatibility; native Jupyter kernel startup in this environment; Windows/macOS; SHAP or partial-dependence analysis; external/temporal/prospective validation. Notebook 02's prior successful execution remains documented in [the earlier validation record](validation.md); the new benchmark starts directly from the raw CSV.

Calibration is an empirical secondary analysis, not a guarantee. Clinical benefit, monetary savings, medical fairness and causal explanations are unverified. Statistical uncertainty is conditional on one reused holdout and does not include all training/selection uncertainty. See [methodology](methodology.md) and [model card](model_card.md).
