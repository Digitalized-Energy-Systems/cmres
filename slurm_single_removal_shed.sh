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
#SBATCH --time=04:00:00
#SBATCH --output=logs/srs_%A_%a.out
#SBATCH --error=logs/srs_%A_%a.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de
#SBATCH --nodelist=mpcs001,mpcb001

# Tunable knobs — must match the launcher's defaults.
N_GRIDS=${N_GRIDS:-6}
N_SHARDS=${N_SHARDS:-8}

# SLURM array index → (grid_idx, shard_idx) — both 0-based.
INPUT_DIR=${INPUT_DIR:-/home/rschrage/experiments/0508/res}
OUTPUT_DIR=${OUTPUT_DIR:-data/out/single_removal_shed}

# Grid catalogue MUST stay aligned with experiments/re/test_grids.py::ALL_GRIDS.
GRIDS=(
    "simbench_lv_no"
    "simbench_lv_low"
    "simbench_lv"
    "simbench_lv_centralized"
    "simbench_lv_high"
    "simbench_lv_max"
)

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
    exit $?
fi

# ── Run phase ─────────────────────────────────────────────────────────────────
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
    --input-dir  "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --shard      "$SHARD" \
    --n-shards   "$N_SHARDS"
