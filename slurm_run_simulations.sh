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
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de
#SBATCH --nodelist=mpcs046,mpcs047

# Tunable knobs (must match the launcher's defaults).
N_GRIDS=${N_GRIDS:-24}
N_SHARDS=${N_SHARDS:-96}

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

# When the launcher pins a single grid by name, CMRES_GRID_NAME is set and
# the TASK_ID encodes the shard index directly (no grid dimension).  In that
# case run_simulation.py resolves the name via ALL_GRIDS itself.
if [[ -n "${CMRES_GRID_NAME:-}" ]]; then
    GRID_SELECTOR=( --name "${CMRES_GRID_NAME}" )
    GRID_LABEL="grid=${CMRES_GRID_NAME}"
    SHARD_IDX="${TASK_ID}"
else
    if [[ -n "${CMRES_MERGE_PHASE:-}" ]]; then
        GRID_IDX="${TASK_ID}"
    else
        GRID_IDX=$(( (TASK_ID - 1) / N_SHARDS + 1 ))
        SHARD_IDX=$(( (TASK_ID - 1) % N_SHARDS + 1 ))
    fi
    GRID_SELECTOR=( "${GRID_IDX}" )
    GRID_LABEL="grid #${GRID_IDX}"
fi

if [[ -n "${CMRES_MERGE_PHASE:-}" ]]; then
    echo "Merging shards for ${GRID_LABEL}"
    python "${SCRIPT}" "${GRID_SELECTOR[@]}" --merge
else
    echo "Running ${GRID_LABEL}, shard ${SHARD_IDX}/${N_SHARDS}"
    python "${SCRIPT}" "${GRID_SELECTOR[@]}" \
        --shard "${SHARD_IDX}" --n-shards "${N_SHARDS}" \
        ${RESUME_FLAG}
fi

EXIT_CODE=$?

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
