#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Launcher for the single-removal-shed analytical ground truth.
#
# Run from a login shell:
#
#     bash submit_single_removal_shed.sh
#     N_SHARDS=16 bash submit_single_removal_shed.sh
#     SUBMIT_MERGE=0 bash submit_single_removal_shed.sh   # don't auto-submit merge
#
# Submits two SLURM jobs:
#   1. RUN array     N_GRIDS × N_SHARDS tasks; each runs one shard
#                    of one grid via single_removal_shed.py --shard …
#   2. MERGE array   N_GRIDS tasks; each merges one grid's shards
#                    with --dependency=afterok:<RUN_JOBID>
#
# Both jobs share the same worker (slurm_single_removal_shed.sh); the
# phase is selected by env var CMRES_MERGE_PHASE.
#
# Usage notes
# -----------
# - INPUT_DIR    where MoneeResilienceExperiment-<grid>/network.p lives
#                (default: /user/towo7024/cmres_new/cmres/data/res)
# - OUTPUT_DIR   per-shard CSV destination
#                (default: data/out/single_removal_shed)
# - GRIDS list   defined inside slurm_single_removal_shed.sh; keep aligned
#                with experiments/re/test_grids.py::ALL_GRIDS.
# ─────────────────────────────────────────────────────────────────────────────

set -e

N_GRIDS=${N_GRIDS:-9}
N_SHARDS=${N_SHARDS:-8}
SUBMIT_MERGE=${SUBMIT_MERGE:-1}
INPUT_DIR=${INPUT_DIR:-/user/towo7024/cmres_new/cmres/data_0512/res}
OUTPUT_DIR=${OUTPUT_DIR:-data/out/single_removal_shed}

mkdir -p logs "$OUTPUT_DIR"

# ── RUN phase ────────────────────────────────────────────────────────────────
# Array size = N_GRIDS × N_SHARDS. Worker computes (grid_idx, shard_idx)
# from SLURM_ARRAY_TASK_ID.
N_RUN_TASKS=$(( N_GRIDS * N_SHARDS ))
echo "Submitting RUN: $N_GRIDS grids × $N_SHARDS shards = $N_RUN_TASKS tasks"

RUN_JID=$(
    sbatch \
        -p rosa.p \
        --parsable \
        --array=0-$((N_RUN_TASKS - 1)) \
        --export=ALL,N_GRIDS=$N_GRIDS,N_SHARDS=$N_SHARDS,INPUT_DIR=$INPUT_DIR,OUTPUT_DIR=$OUTPUT_DIR \
        slurm_single_removal_shed.sh
)
echo "RUN array submitted: jobid=$RUN_JID"

# ── MERGE phase (afterok) ────────────────────────────────────────────────────
if [[ "$SUBMIT_MERGE" == "1" ]]; then
    echo "Submitting MERGE: $N_GRIDS grids (afterok:$RUN_JID)"
    MERGE_JID=$(
        sbatch \
        -p rosa.p \
            --parsable \
            --dependency=afterok:$RUN_JID \
            --array=0-$((N_GRIDS - 1)) \
            --export=ALL,N_GRIDS=$N_GRIDS,N_SHARDS=$N_SHARDS,OUTPUT_DIR=$OUTPUT_DIR,CMRES_MERGE_PHASE=1 \
            slurm_single_removal_shed.sh
    )
    echo "MERGE array submitted: jobid=$MERGE_JID"
fi

echo "Done. Watch with: squeue -u \$USER"
