#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --job-name=05_unsupervised_modelling
#SBATCH --output=%x-%j.out
set -euo pipefail
# Load any cluster-specific Python/compiler modules before submitting this job.
# Submit with sbatch from the repository root, or set CP8305_PROJECT_ROOT.
PROJECT_ROOT="${CP8305_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [[ ! -f "$PROJECT_ROOT/src/paths.py" ]]; then
    echo "Submit from the repository root or set CP8305_PROJECT_ROOT." >&2
    exit 1
fi
VENV_DIR="${CP8305_VENV:-$PROJECT_ROOT/.venv}"
source "$VENV_DIR/bin/activate"
export CP8305_CPUS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_NUM_THREADS="$CP8305_CPUS"
export MKL_NUM_THREADS="$CP8305_CPUS"
export OPENBLAS_NUM_THREADS="$CP8305_CPUS"
export PYTHONUNBUFFERED=1
cd "$PROJECT_ROOT"
python scripts/run_notebook.py notebooks/05_unsupervised_modelling.ipynb
