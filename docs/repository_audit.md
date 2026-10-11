# Original README Audit

Scope: [chrisindris/CP8305FP](https://github.com/chrisindris/CP8305FP), `main` at **d340d61aa0aa7f7828409dfeae4f27c437c50d4a**, dated **2026-09-26**. This describes the original snapshot before portfolio maintenance.

## Confirmed Contents

The snapshot contains 163 tracked files. Its top-level project directories are `data`, `docs`, `notebooks`, `notes`, `presentation`, `reports`, `scripts`, and `src`, plus editor/GitHub configuration directories. It contains five notebooks, not four.

| Location | Confirmed contents at the source commit |
| --- | --- |
| `data/diabetes130/` | `raw/IDS_mapping.csv` and raw/processed `.gitkeep` files; no tracked `diabetic_data.csv` or processed dataset. |
| `docs/` | `data_sources.md`, still a template. |
| `notebooks/` | `01_data_exploration.ipynb`, `02_data_preprocessing.ipynb`, `03_modeling.ipynb`, `04_analysis_findings.ipynb`, `05_unsupervised_modelling.ipynb`; a Python export; three Slurm launchers; 27 output logs. |
| `notes/` | `02_data_preprocessing_checklist.md` and `03_modeling_checklist.md`. |
| `presentation/` | `slides.pdf`, `slides.tex`, `slides.bib`, `build.zsh`, figures, reference resources, and generated LaTeX intermediates. No `.ppt` or `.pptx`. |
| `reports/` | `bare_jrnl.pdf`, `bare_jrnl.tex`, `references.bib`, `build.zsh`, `best_classifier.pkl`, 38 figures, 8 CSV tables, and generated intermediates. |
| `scripts/refactor/` | Six Python scripts for model evaluation, cross-validation, and imbalance experiments. |
| `src/` | `__init__.py`, `data_preprocessing.py`, `feature_engineering.py`, `modeling.py`, `visualization.py`. |

## README Comparison

| Original claim/section | Status | Correction |
| --- | --- | --- |
| Team members and reusable Python modules | Matches | Preserve all three authors and existing modules. |
| Project Topic | Missing | Replace placeholder with the diabetes 30-day readmission task and binary target. |
| Four notebooks | Incomplete | Include notebook 05 and distinguish unfinished original notebook 04 from a completed analysis. |
| `data/raw` and `data/processed` layout | Inaccurate | Use actual `data/diabetes130/raw` and `data/diabetes130/processed` paths. |
| Repository map | Incomplete | Include notes, presentation, refactor scripts, report/model/CSV assets, and launchers. |
| Deliverables checklist | Inaccurate/incomplete | Report and slides PDFs already exist; slides source is Beamer, not PowerPoint. Data documentation was still a template. |
| Pinned dependencies | Inaccurate | Original `requirements.txt` contains unpinned names. |
| Generic preprocessing entry point | Incomplete | It does not reproduce the complete diabetes-specific notebook workflow; original processed paths also differed. |
| References | Missing | Replace five placeholder items with actual dataset and literature references. |

The revised README fills these gaps. Original academic artifacts and history are preserved; generated logs and LaTeX build intermediates are removed from the current tree but remain available at the source commit. Newly executed baseline results are stored separately under `reports/portfolio/`.
