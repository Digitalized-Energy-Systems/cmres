#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM array job: CMRES RQMC resilience simulations  (sharded)
#
# Each (grid × shard) pair is one independent SLURM array task.  After all
# shards of a grid finish, a dependent merge job concatenates the per-shard
# ``per_run`` arrays and recomputes the final statistics, producing
# ``data/res/<EXPERIMENT_NAME>-<grid>/mc_result.npz``.
#
# With N_GRIDS grids and N_SHARDS shards per grid:
#   - parallel run tasks  : N_GRIDS × N_SHARDS
#   - dependent merge tasks: 1 (handles all grids in series)
#
# Submit:
#   sbatch slurm_run_simulations.sh
#
# Run only the run-phase array (skip auto-submitting merge):
#   SUBMIT_MERGE=0 sbatch slurm_run_simulations.sh
#
# Run a single shard manually (for testing):
#   sbatch --array=1 slurm_run_simulations.sh
#
# Adjust shard count via env var:
#   N_SHARDS=16 sbatch slurm_run_simulations.sh
# ─────────────────────────────────────────────────────────────────────────────

# ── Tunable knobs (override via env vars on submit) ───────────────────────────
N_GRIDS=${N_GRIDS:-2}        # Must match len(ALL_GRIDS) in test_grids.py
N_SHARDS=${N_SHARDS:-8}      # Shards per grid → total tasks = N_GRIDS * N_SHARDS
SUBMIT_MERGE=${SUBMIT_MERGE:-1}

TOTAL_TASKS=$(( N_GRIDS * N_SHARDS ))

#SBATCH --job-name=cmres_rqmc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00              # per-shard wall time; shorter than the
                                     # un-sharded 16h since each shard runs
                                     # only MC_MAX_RUNS / N_SHARDS scenarios
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de

# Note: --array is set on the command line below via re-submission, since the
# script must compute TOTAL_TASKS from env vars before sbatch sees the
# directive.  See the dispatch block at the bottom.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export JAX_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT="./cmres/experiments/re/run_simulation.py"
DATA_DIR="./data/res"
LOG_DIR="./logs"
mkdir -p "${DATA_DIR}" "${LOG_DIR}"

# ── Re-submission dispatch ────────────────────────────────────────────────────
# When invoked without a SLURM array context, this script re-submits itself
# with the computed array bounds + a dependent merge job, then exits.
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" && -z "${CMRES_MERGE_PHASE:-}" ]]; then
    echo "Submitting RUN phase: array=1-${TOTAL_TASKS}  (N_GRIDS=${N_GRIDS} × N_SHARDS=${N_SHARDS})"
    RUN_JOBID=$(sbatch --parsable --array="1-${TOTAL_TASKS}" "$0" "$@" | tr -d '\n')
    echo "  RUN job id    : ${RUN_JOBID}"

    if [[ "${SUBMIT_MERGE}" == "1" ]]; then
        echo "Submitting MERGE phase: depends on afterok:${RUN_JOBID}"
        MERGE_JOBID=$(CMRES_MERGE_PHASE=1 sbatch --parsable \
            --dependency=afterok:"${RUN_JOBID}" \
            --time=00:30:00 \
            --array="1-${N_GRIDS}" \
            "$0" "$@" | tr -d '\n')
        echo "  MERGE job id  : ${MERGE_JOBID}"
    else
        echo "MERGE phase skipped (SUBMIT_MERGE=0)."
        echo "Run manually after RUN phase finishes:"
        for i in $(seq 1 ${N_GRIDS}); do
            echo "  python ${SCRIPT} ${i} --merge"
        done
    fi
    exit 0
fi

# ── --resume passthrough ──────────────────────────────────────────────────────
RESUME_FLAG=""
for arg in "$@"; do
    [[ "$arg" == "--resume" ]] && RESUME_FLAG="--resume"
done

# ── Run a single shard or a single merge ──────────────────────────────────────
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

# Map SLURM_ARRAY_TASK_ID → (grid_idx, shard_idx).
# RUN phase  : tasks 1..(N_GRIDS*N_SHARDS), grid_idx = ((id-1) / N_SHARDS) + 1
# MERGE phase: tasks 1..N_GRIDS, grid_idx = id
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
