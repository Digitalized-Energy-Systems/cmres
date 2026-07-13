#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CMRES single-removal load-shed worker.
#
# Submitted by submit_single_removal_shed.sh. Don't invoke directly via
# SLURM — use ``bash submit_single_removal_shed.sh`` instead.
#
# Each invocation runs ONE of:
#   * a single (grid, shard)  — when CMRES_MERGE_PHASE is unset
#   * a single grid merge      — when CMRES_MERGE_PHASE=1
#
# The phase is selected by the launcher via the environment; the
# (grid_idx, shard_idx) pair is derived from SLURM_ARRAY_TASK_ID:
#
#     grid_idx  = SLURM_ARRAY_TASK_ID / N_SHARDS    # 0-based
#     shard_idx = SLURM_ARRAY_TASK_ID % N_SHARDS    # 0-based
#
# The component slice for each shard is a round-robin stripe across the
# component list (so wall-clock imbalance from heterogeneous solve times
# averages out). See single_removal_shed.py::_slice_targets.
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cmres_srs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/srs_%A_%a.out
#SBATCH --error=logs/srs_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de
#SBATCH --nodelist=mpcb014,mpcb016

# Tunable knobs — must match the launcher's defaults.
N_GRIDS=${N_GRIDS:-24}
N_SHARDS=${N_SHARDS:-8}

# SLURM array index → (grid_idx, shard_idx) — both 0-based.
# Networks are built per-task from experiments/re/test_grids.py::ALL_GRIDS
# (same path the MC simulation uses), so no pre-built pickle directory is
# required any more — only the output dir.
OUTPUT_DIR=${OUTPUT_DIR:-data/out/single_removal_shed}

# Grid catalogue — derived dynamically from
# experiments/re/test_grids.py::ALL_GRIDS via sed so this list cannot drift
# from the Python truth (insertion order matters: index → name must match
# what slurm_run_simulations.sh / submit_simulations.sh use).
GRIDS=( $(
    sed -n \
        '/^ALL_GRIDS[[:space:]]*=[[:space:]]*{/,/^}/ s/^[[:space:]]*"\([^"]*\)"[[:space:]]*:.*/\1/p' \
        experiments/re/test_grids.py
) )
if (( ${#GRIDS[@]} == 0 )); then
    echo "ERROR: could not extract ALL_GRIDS keys from experiments/re/test_grids.py" >&2
    exit 2
fi
if (( ${#GRIDS[@]} != N_GRIDS )); then
    echo "WARN: N_GRIDS=$N_GRIDS but ALL_GRIDS has ${#GRIDS[@]} keys — using the file's count." >&2
    N_GRIDS=${#GRIDS[@]}
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export JAX_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

SCRIPT="./experiments/re/single_removal_shed.py"
mkdir -p "$OUTPUT_DIR" ./logs

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: this is a worker script; submit it via ``bash submit_single_removal_shed.sh``." >&2
    exit 2
fi

echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID}  array task: ${SLURM_ARRAY_TASK_ID}"
echo "Host        : $(hostname)"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Phase       : ${CMRES_MERGE_PHASE:+MERGE}${CMRES_MERGE_PHASE:-RUN}"
echo "Started at  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── HPC environment activation (must match slurm_run_simulations.sh) ─────────
# Without these, SLURM falls back to the system Python (≤ 3.6 on this
# cluster), which can't parse ``from __future__ import annotations``,
# triggering an immediate SyntaxError and a non-zero exit that breaks the
# afterok dependency on the merge array.
module load hpc-env/13.1
module load Miniforge3/26.1.0-0
conda activate cmres_env

# ── Merge phase ───────────────────────────────────────────────────────────────
# Each merge task handles exactly one grid (by SLURM_ARRAY_TASK_ID = grid_idx).
if [[ "${CMRES_MERGE_PHASE:-0}" == "1" ]]; then
    GRID_IDX=$SLURM_ARRAY_TASK_ID
    GRID=${GRIDS[$GRID_IDX]}
    if [[ -z "$GRID" ]]; then
        echo "ERROR: no grid for index $GRID_IDX" >&2
        exit 2
    fi
    echo "[merge] grid=$GRID"
    python -u "$SCRIPT" "$GRID" \
        --output-dir "$OUTPUT_DIR" \
        --merge \
        --n-shards "$N_SHARDS"
    EXIT_CODE=$?
else
    # ── Run phase ─────────────────────────────────────────────────────────────
    GRID_IDX=$(( SLURM_ARRAY_TASK_ID / N_SHARDS ))
    SHARD_IDX=$(( SLURM_ARRAY_TASK_ID % N_SHARDS ))
    SHARD=$(( SHARD_IDX + 1 ))     # single_removal_shed.py expects 1-based shards
    GRID=${GRIDS[$GRID_IDX]}

    if [[ -z "$GRID" ]]; then
        echo "ERROR: no grid for index $GRID_IDX (array=$SLURM_ARRAY_TASK_ID, N_SHARDS=$N_SHARDS)" >&2
        exit 2
    fi

    echo "[run] grid=$GRID shard=$SHARD/$N_SHARDS  array_id=$SLURM_ARRAY_TASK_ID"
    python -u "$SCRIPT" "$GRID" \
        --output-dir "$OUTPUT_DIR" \
        --shard      "$SHARD" \
        --n-shards   "$N_SHARDS"
    EXIT_CODE=$?
fi

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
