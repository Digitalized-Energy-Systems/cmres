#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Launcher for the CMRES RQMC sharded simulation array.
#
# This script is *not* a SLURM job — invoke it directly from a login shell:
#
#     bash submit_simulations.sh
#     N_SHARDS=16 bash submit_simulations.sh
#     SUBMIT_MERGE=0 bash submit_simulations.sh   # don't auto-submit merge
#
# Run a single grid only:
#     GRID_NAME=simbench_lv_high bash submit_simulations.sh
#     GRID_IDX=5                 bash submit_simulations.sh
#
# It submits two SLURM jobs:
#   1. the RUN array       (N_GRIDS × N_SHARDS tasks, each runs one shard)
#   2. the MERGE array     (N_GRIDS tasks, each merges one grid's shards)
#      with --dependency=afterok:<RUN_JOBID>
#
# All actual workload runs in slurm_run_simulations.sh, which contains no
# nested sbatch calls and so passes through site-wide submit filters that
# reject scripts containing the ``sbatch`` keyword.
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

N_GRIDS=${N_GRIDS:-11}
N_SHARDS=${N_SHARDS:-96}
SUBMIT_MERGE=${SUBMIT_MERGE:-1}
GRID_NAME=${GRID_NAME:-}
GRID_IDX=${GRID_IDX:-}

# ── Single-grid mode ──────────────────────────────────────────────────────────
# Resolve GRID_NAME → GRID_IDX via AST parse of test_grids.py (avoids the
# heavy side-effects of importing the module).
if [[ -n "${GRID_NAME}" && -n "${GRID_IDX}" ]]; then
    echo "ERROR: set either GRID_NAME or GRID_IDX, not both." >&2
    exit 1
fi
if [[ -n "${GRID_NAME}" ]]; then
    GRID_IDX=$(CMRES_GRID_NAME="${GRID_NAME}" \
               CMRES_TEST_GRIDS="${SCRIPT_DIR}/experiments/re/test_grids.py" \
               python3 -c '
import ast, os, sys
src = open(os.environ["CMRES_TEST_GRIDS"]).read()
keys = None
for node in ast.parse(src).body:
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "ALL_GRIDS" for t in node.targets
    ):
        keys = [k.value for k in node.value.keys]
        break
if keys is None:
    sys.exit("ALL_GRIDS not found in test_grids.py")
name = os.environ["CMRES_GRID_NAME"]
if name not in keys:
    sys.exit(f"Grid {name!r} not in ALL_GRIDS.\nAvailable: {keys}")
print(keys.index(name) + 1)
') || exit 1
fi

if [[ -n "${GRID_IDX}" ]]; then
    if ! [[ "${GRID_IDX}" =~ ^[0-9]+$ ]] || (( GRID_IDX < 1 || GRID_IDX > N_GRIDS )); then
        echo "ERROR: GRID_IDX=${GRID_IDX} must be in [1, ${N_GRIDS}]." >&2
        exit 1
    fi
    RUN_ARRAY_START=$(( (GRID_IDX - 1) * N_SHARDS + 1 ))
    RUN_ARRAY_END=$(( GRID_IDX * N_SHARDS ))
    RUN_ARRAY="${RUN_ARRAY_START}-${RUN_ARRAY_END}"
    MERGE_ARRAY="${GRID_IDX}-${GRID_IDX}"
    TOTAL_TASKS=${N_SHARDS}
else
    TOTAL_TASKS=$(( N_GRIDS * N_SHARDS ))
    RUN_ARRAY="1-${TOTAL_TASKS}"
    MERGE_ARRAY="1-${N_GRIDS}"
fi

WORKER="${SCRIPT_DIR}/slurm_run_simulations.sh"
if [[ ! -r "${WORKER}" ]]; then
    echo "ERROR: cannot read worker script at ${WORKER}" >&2
    exit 1
fi

mkdir -p logs data/res

echo "============================================================"
echo "Submitting CMRES RQMC array job"
echo "  Worker        : ${WORKER}"
echo "  N_GRIDS       : ${N_GRIDS}"
echo "  N_SHARDS      : ${N_SHARDS}"
if [[ -n "${GRID_IDX}" ]]; then
    echo "  Single grid   : #${GRID_IDX}${GRID_NAME:+ (${GRID_NAME})}"
fi
echo "  RUN array     : ${RUN_ARRAY}"
echo "  MERGE array   : ${MERGE_ARRAY}"
echo "  Total tasks   : ${TOTAL_TASKS}"
echo "  Auto-merge    : ${SUBMIT_MERGE}"
echo "============================================================"

# ── RUN phase ─────────────────────────────────────────────────────────────────
RUN_OUT=$(sbatch -p rosa_express.p --parsable --array="${RUN_ARRAY}" "${WORKER}" "$@")
RUN_JOBID=$(echo "${RUN_OUT}" | tr -d '\n')
if [[ -z "${RUN_JOBID}" ]]; then
    echo "ERROR: RUN-phase sbatch failed.  See SLURM error above." >&2
    exit 1
fi
echo "  RUN job id    : ${RUN_JOBID}"

# ── MERGE phase ───────────────────────────────────────────────────────────────
if [[ -n "${GRID_IDX}" ]]; then
    MERGE_INDICES=( "${GRID_IDX}" )
else
    MERGE_INDICES=( $(seq 1 ${N_GRIDS}) )
fi

if [[ "${SUBMIT_MERGE}" != "1" ]]; then
    echo
    echo "MERGE phase skipped (SUBMIT_MERGE=0)."
    echo "Run manually after the RUN array finishes:"
    for i in "${MERGE_INDICES[@]}"; do
        echo "  python ./experiments/re/run_simulation.py ${i} --merge"
    done
    exit 0
fi

MERGE_OUT=$(CMRES_MERGE_PHASE=1 sbatch -p rosa_express.p --parsable \
    --dependency=afterok:"${RUN_JOBID}" \
    --time=00:30:00 \
    --array="${MERGE_ARRAY}" \
    "${WORKER}" "$@")
MERGE_JOBID=$(echo "${MERGE_OUT}" | tr -d '\n')
if [[ -z "${MERGE_JOBID}" ]]; then
    echo "ERROR: MERGE-phase sbatch failed.  RUN job ${RUN_JOBID} continues." >&2
    echo "       Run merge manually after RUN finishes:" >&2
    for i in "${MERGE_INDICES[@]}"; do
        echo "         python ./experiments/re/run_simulation.py ${i} --merge" >&2
    done
    exit 1
fi
echo "  MERGE job id  : ${MERGE_JOBID}  (depends on afterok:${RUN_JOBID})"
