# Data Sources

## Primary Dataset

| Field | Details |
| --- | --- |
| Name | Diabetes 130-US Hospitals for Years 1999–2008 |
| Publisher | UCI Machine Learning Repository |
| Dataset page | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| DOI | https://doi.org/10.24432/C5230J |
| Coverage | Diabetes-related inpatient encounters in the United States, 1999–2008 |
| Download verified | 2026-10-11; each download records its own exact UTC time locally |
| Shape of downloaded CSV | 101,766 rows × 50 columns, including 2 identifiers and the target |
| Companion file | `IDS_mapping.csv`: admission type, discharge disposition, and admission source lookup tables |
| Dataset license | CC BY 4.0, as stated by UCI; this does not license all repository contents |

Run `python -m src.download_data` to fetch the [official archive](https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip). Only the two expected CSV names are extracted. Existing files with different contents are not silently overwritten.

The data directory is `data/diabetes130/raw/`. The full encounter CSV and generated manifest remain local; the small mapping file is tracked. `download_manifest.json` records the source URL, UTC timestamp, sizes, and SHA-256 hashes.

Verified original CSV hash:

```text
0689e7ec031237dc63031b938805c48377748761a3b26acab621567afa24df97
```

## Variables and Target

- `readmitted`: `<30`, `>30`, or `NO`. The binary task uses `<30` as class 1 and both other labels as class 0.
- `patient_nbr` and `encounter_id`: identifiers used for cohort construction and overlap checks, never baseline predictors.
- Numeric utilization features include length of stay, medication/procedure/lab counts, and previous inpatient, outpatient, and emergency visits.
- Categorical variables include age bracket, demographic categories, admission/discharge/source IDs, diagnoses, laboratory result categories, and medication indicators.
- The companion mappings describe coded admission/discharge/source values; they are not a second patient dataset.

## Data Quality and Cohort Rules

The downloader preserves raw bytes. The baseline reads `?` as missing and keeps the literal laboratory category `None` as a category. Missing numerical values are imputed with training-set medians; categorical missingness becomes `Unknown`. Encoding and scaling are fitted on the training split only.

The baseline excludes discharge codes 11, 13, 14, 19, 20, and 21 (expired/hospice categories), then retains the smallest eligible `encounter_id` per patient. This is a deterministic selection rule, not proof of chronological ordering: the dataset does not provide a verified encounter timestamp for this use. The resulting cohort contains **69,990 patients**, including **6,285 positives (8.98%)**. Outliers are retained.

The split is stratified 70/15/15 with seed 42. The 22-predictor baseline uses a train-fitted pipeline and validation-only F1 threshold selection. Its cohort and transformations are separate from the feature-engineered historical notebook workflow. See [implementation](../src/baseline.py) and [recorded run](../reports/portfolio/baseline/metrics.json).

Notebook 02 retains one eligible record per patient and exports its larger engineered table to `data/diabetes130/processed/processed_dataset.csv`. Some historical preparation choices are performed before splitting; the new baseline is the preferred reproducibility example. Neither workflow should be described as externally validated.

## Use and Limitations

These older records do not establish performance in present-day hospitals. Class imbalance, selection rules, missingness, available discharge information, and demographic representation can affect conclusions. The baseline is intended for retrospective research using encounter/discharge information, not a claim of prediction at admission. It has no subgroup fairness, prospective, external, or temporal validation.

No patient-level exports, credentials, or private hospital records are needed to review the committed results notebook. Predictions and illustrative cost assumptions are not clinical recommendations or measured financial savings.

## Citation

Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999–2008* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J
