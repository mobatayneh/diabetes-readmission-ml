# Validation and Evaluation Limits

Verified locally on **2026-10-11** with Python **3.12.14**. Package versions for the new experiment are recorded in [metrics.json](../reports/portfolio/baseline/metrics.json); the tested direct dependencies are pinned in `requirements-core.txt` and `requirements-summary.txt`.

## What Was Run

| Check | Result |
| --- | --- |
| Official UCI downloader | Downloaded the original CSV and verified the existing mapping file; source URL, UTC timestamp, sizes, and hashes recorded locally. |
| Full new baseline | 101,766 raw rows → 69,990 eligible unique patients; completed with no convergence warnings. |
| Patient partitions | Train 48,993; validation 10,498; test 10,499; all three pairwise patient overlaps are zero. |
| Unit checks | 9 tests passed: labels, cohort exclusions, encounter selection, missing IDs, split separation/reproducibility, train-only preprocessing, thresholds/metrics, paths/mappings, and safe archive extraction. |
| Python syntax | `python -m compileall -q src scripts` passed. |
| Notebook 02 | All 68 code cells completed sequentially on the original UCI data; produced 69,990 rows × 2,473 columns, no missing values, and no identifier columns. |
| Notebook 04 | All 6 code cells completed sequentially using committed aggregate results; tables and two figures rendered and inspected. |
| Notebook structure | All five files validated with nbformat; code cells compiled after IPython transformation. |
| Historical report artifacts | Original report/slides PDFs, model, figures, and CSV tables retained unchanged. |
| GitHub Actions | Passed on GitHub with Ubuntu and Python 3.12: installation, compilation, unit checks, and notebook 04 execution. [Verified run](https://github.com/mobatayneh/CP8305FP/actions/runs/38119312537), commit `f410832666ce1590224e84a8f508ad62d0104fc7`. |

Notebook execution used `scripts/run_notebook.py --in-process`, which runs trusted cells sequentially with IPython and captures their outputs. The preparation environment could not open Jupyter kernel sockets. Native Jupyter kernel startup and Windows execution were therefore not validated here. The default runner uses the active Python interpreter for its kernel on environments that allow sockets.

Notebook 02's large processed CSV and executed copy remain ignored local outputs. Notebook 04 includes its executed outputs for review directly on GitHub. Notebooks 01, 03, and 05 were checked structurally but **not rerun end to end**; their old embedded outputs were cleared to avoid presenting stale results as output of modified code. Original runs remain in Git history and committed report assets.

## Reproducing the New Experiment

From the repository root, with Python 3.12 and an activated environment:

```bash
python -m pip install -r requirements-summary.txt
python -m src.download_data
python -m src.baseline
python -m unittest discover -s tests -v
python -m compileall -q src scripts
python scripts/run_notebook.py notebooks/04_analysis_findings.ipynb --in-process
```

The default baseline output is `reports/runs/baseline/metrics.json`. Notebook 04 intentionally reviews the **committed** reference run under `reports/portfolio/baseline/`; it does not automatically switch to the most recent local experiment. To deliberately replace that reference after reviewing a new run:

```bash
python -m src.baseline --output reports/portfolio/baseline
python scripts/run_notebook.py notebooks/04_analysis_findings.ipynb --output notebooks/04_analysis_findings.ipynb --in-process
```

For notebook 02, set `CP8305_REPORTS_DIR` to an isolated output directory before execution, as described in the README. The pinned core and notebook dependencies are sufficient for that run. The full optional research stack in `requirements.txt` has not been installed and validated as a locked environment.

## Changes to the Historical Workflow

- Data loading uses the original local UCI CSV and preserves the literal laboratory category `None`.
- Processed data paths now agree across the shared helper and notebooks.
- Removed machine-specific paths and the credential-file setup cell.
- Fixed a reference to a disabled outlier-filter variable and pandas categorical mapping of missing descriptions in notebook 02. Outliers remain in the data.
- Notebook 03 defines its CPU limit, uses separate training/validation/test partitions, and selects F1/cost thresholds with validation labels instead of test labels. Full training has not been rerun after these changes.
- The old independent Python export now launches the canonical notebook, avoiding conflicting fold settings.
- Notebook 04 replaces placeholders with sourced tables, explicit metric definitions, and an independent baseline review.
- Shared figure/model paths respect `CP8305_REPORTS_DIR`; it is optional, and the default remains `reports/`.

These changes do **not** demonstrate that all leakage or statistical issues in the historical workflow have been resolved. Some preprocessing and encoding in notebook 02 occurs before splitting; notebook 03's broader feature selection/CV protocol needs further review before interpreting its intervals as unbiased evidence. The compact baseline uses a distinct train-fitted pipeline.

## How to Interpret the Results

The new baseline uses one random patient-disjoint split, not repeated or nested validation. F1 threshold selection occurs only on validation labels; the held-out test set is used for reporting. The comparison protocol, cohort transformations, and predictors differ from historical experiments, so the new and archived scores do not establish a performance improvement or regression.

The baseline's 0.156 average precision should be interpreted alongside the 8.98% positive prevalence. At the validation-selected 0.56 threshold, recall is 0.403 and precision is 0.153. Class weighting can impair probability calibration; the recorded Brier score is 0.218. No claim of well-calibrated clinical risk is made.

Historical table field `auprc_mean` was computed with `average_precision_score`; `pr_auc_mean` is a separate trapezoidal PR-area calculation. Notebook 04 preserves this distinction. Stored 95% intervals are inherited summaries, not independently verified confidence intervals. Initial and tuned tables are separate protocols and are not combined to claim tuning improved performance.

Clustering profiles provide feature averages. They do not by themselves establish cluster sizes, outcome rates, causal factors, or clinically validated risk groups. Existing monetary cost assumptions are illustrative, not observed savings or official penalty estimates.

Further research could assess temporal/external generalization, repeated patient-disjoint evaluation, calibration on validation data, demographic subgroup performance, and a clearly defined clinical prediction time. Those studies have not been performed in this edition.
