"""Run the canonical modeling notebook with the current Python environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_notebook import main

if __name__ == "__main__":
    main(["notebooks/03_modeling.ipynb", *sys.argv[1:]])
