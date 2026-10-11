"""Repository-relative paths shared by notebooks and command-line entry points."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "diabetes130"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = Path(os.environ.get("CP8305_REPORTS_DIR", PROJECT_ROOT / "reports")).resolve()
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
