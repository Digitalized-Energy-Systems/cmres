#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM array job: CMRES RQMC resilience simulations
#
# Submits all 18 (scenario × grid) experiments as independent array tasks,
# each running on its own node/allocation simultaneously.
#
# Submit:
#   sbatch slurm_run_simulations.sh
#
# Resume (skip already-completed experiments):
#   sbatch slurm_run_simulations.sh --resume
#
# Run a single task manually (for testing):
#   sbatch --array=1 slurm_run_simulations.sh
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cmres_rqmc
#SBATCH --array=1-18                  # one task per (scenario, grid) pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # JAX / numba / torch use internal threads
#SBATCH --mem=64G
#SBATCH --time=16:00:00              # generous; most runs finish in <2 h
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mail-type=FAIL             # notify only on failure
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export JAX_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT="./cmres/experiments/re/run_simulation.py"
DATA_DIR="./data/res"
LOG_DIR="./logs"

mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# ── Optional --resume flag forwarded from sbatch command line ─────────────────
RESUME_FLAG=""
for arg in "$@"; do
    [[ "$arg" == "--resume" ]] && RESUME_FLAG="--resume"
done

# ── Run ───────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID}  array task: ${SLURM_ARRAY_TASK_ID}"
echo "Host        : $(hostname)"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Experiment  : ${SLURM_ARRAY_TASK_ID} / 18"
echo "Started at  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

module load hpc-env/13.1
module load Miniforge3/26.1.0-0

conda activate cmres_env

# cd ./cmres
# pip install -e .
# cd ..

python "${SCRIPT}" "${SLURM_ARRAY_TASK_ID}" ${RESUME_FLAG}

EXIT_CODE=$?

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
