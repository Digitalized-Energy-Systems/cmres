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

N_GRIDS=${N_GRIDS:-3}
N_SHARDS=${N_SHARDS:-8}
SUBMIT_MERGE=${SUBMIT_MERGE:-1}
TOTAL_TASKS=$(( N_GRIDS * N_SHARDS ))

WORKER="$(cd "$(dirname "$0")" && pwd)/slurm_run_simulations.sh"
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
echo "  Total tasks   : ${TOTAL_TASKS}"
echo "  Auto-merge    : ${SUBMIT_MERGE}"
echo "============================================================"

# ── RUN phase ─────────────────────────────────────────────────────────────────
RUN_OUT=$(sbatch -p rosa.p --parsable --array="1-${TOTAL_TASKS}" "${WORKER}" "$@")
RUN_JOBID=$(echo "${RUN_OUT}" | tr -d '\n')
if [[ -z "${RUN_JOBID}" ]]; then
    echo "ERROR: RUN-phase sbatch failed.  See SLURM error above." >&2
    exit 1
fi
echo "  RUN job id    : ${RUN_JOBID}"

# ── MERGE phase ───────────────────────────────────────────────────────────────
if [[ "${SUBMIT_MERGE}" != "1" ]]; then
    echo
    echo "MERGE phase skipped (SUBMIT_MERGE=0)."
    echo "Run manually after the RUN array finishes:"
    for i in $(seq 1 ${N_GRIDS}); do
        echo "  python ./cmres/experiments/re/run_simulation.py ${i} --merge"
    done
    exit 0
fi

MERGE_OUT=$(CMRES_MERGE_PHASE=1 sbatch -p rosa.p --parsable \
    --dependency=afterok:"${RUN_JOBID}" \
    --time=00:30:00 \
    --array="1-${N_GRIDS}" \
    "${WORKER}" "$@")
MERGE_JOBID=$(echo "${MERGE_OUT}" | tr -d '\n')
if [[ -z "${MERGE_JOBID}" ]]; then
    echo "ERROR: MERGE-phase sbatch failed.  RUN job ${RUN_JOBID} continues." >&2
    echo "       Run merge manually after RUN finishes:" >&2
    for i in $(seq 1 ${N_GRIDS}); do
        echo "         python ./cmres/experiments/re/run_simulation.py ${i} --merge" >&2
    done
    exit 1
fi
echo "  MERGE job id  : ${MERGE_JOBID}  (depends on afterok:${RUN_JOBID})"
