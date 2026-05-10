#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CMRES RQMC simulation worker.
#
# Submitted by submit_simulations.sh.  Don't invoke directly via SLURM —
# use ``bash submit_simulations.sh`` (see header of that script).
#
# Each invocation runs ONE of:
#   * a single shard       — when CMRES_MERGE_PHASE is unset
#   * a single grid merge  — when CMRES_MERGE_PHASE=1
#
# The phase is selected by submit_simulations.sh via the environment, and the
# (grid_idx, shard_idx) pair is derived from SLURM_ARRAY_TASK_ID.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cmres_rqmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de
#SBATCH --nodelist=mpcs001,mpcb001

# Tunable knobs (must match the launcher's defaults).
N_GRIDS=${N_GRIDS:-9}
N_SHARDS=${N_SHARDS:-6}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export JAX_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

SCRIPT="./experiments/re/run_simulation.py"
mkdir -p ./data/res ./logs

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: this is a worker script; submit it via ``bash submit_simulations.sh``." >&2
    exit 2
fi

# ── --resume passthrough ──────────────────────────────────────────────────────
RESUME_FLAG=""
for arg in "$@"; do
    [[ "$arg" == "--resume" ]] && RESUME_FLAG="--resume"
done

echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID}  array task: ${SLURM_ARRAY_TASK_ID}"
echo "Host        : $(hostname)"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Phase       : ${CMRES_MERGE_PHASE:+MERGE}${CMRES_MERGE_PHASE:-RUN}"
echo "Started at  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

module load hpc-env/13.1
module load Miniforge3/26.1.0-0
conda activate cmres_env

TASK_ID="${SLURM_ARRAY_TASK_ID}"

if [[ -n "${CMRES_MERGE_PHASE:-}" ]]; then
    GRID_IDX="${TASK_ID}"
    echo "Merging shards for grid #${GRID_IDX}"
    python "${SCRIPT}" "${GRID_IDX}" --merge
else
    GRID_IDX=$(( (TASK_ID - 1) / N_SHARDS + 1 ))
    SHARD_IDX=$(( (TASK_ID - 1) % N_SHARDS + 1 ))
    echo "Running grid #${GRID_IDX}, shard ${SHARD_IDX}/${N_SHARDS}"
    python "${SCRIPT}" "${GRID_IDX}" \
        --shard "${SHARD_IDX}" --n-shards "${N_SHARDS}" \
        ${RESUME_FLAG}
fi

EXIT_CODE=$?

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
