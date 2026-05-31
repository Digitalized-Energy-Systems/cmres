#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Launcher for the single-removal-shed analytical ground truth.
#
# Run from a login shell:
#
#     bash submit_single_removal_shed.sh
#     N_SHARDS=16 bash submit_single_removal_shed.sh
#     SUBMIT_MERGE=0 bash submit_single_removal_shed.sh   # don't auto-submit merge
#     SUBMIT_EVAL=0  bash submit_single_removal_shed.sh   # don't auto-submit eval
#
# Submits up to three SLURM jobs:
#   1. RUN array     N_GRIDS × N_SHARDS tasks; each runs one shard
#                    of one grid via single_removal_shed.py --shard …
#   2. MERGE array   N_GRIDS tasks; each merges one grid's shards
#                    with --dependency=afterok:<RUN_JOBID>
#   3. EVAL job      one task; renders the full plot/report battery
#                    (per-grid + cross-grid pooled) via
#                    slurm_single_removal_shed_eval.sh,
#                    --dependency=afterok:<MERGE_JOBID>.
#
# RUN and MERGE share the worker (slurm_single_removal_shed.sh); the phase
# is selected by env var CMRES_MERGE_PHASE. EVAL uses its own dedicated
# worker (slurm_single_removal_shed_eval.sh).
#
# Usage notes
# -----------
# - OUTPUT_DIR   per-shard CSV destination
#                (default: data/out/single_removal_shed)
# - GRIDS list   defined inside slurm_single_removal_shed.sh; keep aligned
#                with experiments/re/test_grids.py::ALL_GRIDS — that file is
#                also the network source-of-truth (no pre-built pickle is
#                needed any more).
# ─────────────────────────────────────────────────────────────────────────────

set -e

N_GRIDS=${N_GRIDS:-22}
N_SHARDS=${N_SHARDS:-8}
SUBMIT_MERGE=${SUBMIT_MERGE:-1}
SUBMIT_EVAL=${SUBMIT_EVAL:-1}
OUTPUT_DIR=${OUTPUT_DIR:-data/out/single_removal_shed}
# E16 (in the eval job) needs the MC-experiment outputs; default mirrors the
# DEFAULT_INPUT_DIR baked into experiments/re/e16_plots.py. Override when the
# MC data lives elsewhere.
INPUT_DIR=${INPUT_DIR:-data/res}
E16_OUT_DIR=${E16_OUT_DIR:-data/out/cmres}
SKIP_E16=${SKIP_E16:-0}

mkdir -p logs "$OUTPUT_DIR"

# ── RUN phase ────────────────────────────────────────────────────────────────
# Array size = N_GRIDS × N_SHARDS. Worker computes (grid_idx, shard_idx)
# from SLURM_ARRAY_TASK_ID.
N_RUN_TASKS=$(( N_GRIDS * N_SHARDS ))
echo "Submitting RUN: $N_GRIDS grids × $N_SHARDS shards = $N_RUN_TASKS tasks"

RUN_JID=$(
    sbatch \
        -p rosa_express.p \
        --parsable \
        --array=0-$((N_RUN_TASKS - 1)) \
        --export=ALL,N_GRIDS=$N_GRIDS,N_SHARDS=$N_SHARDS,OUTPUT_DIR=$OUTPUT_DIR \
        slurm_single_removal_shed.sh
)
echo "RUN array submitted: jobid=$RUN_JID"

# ── MERGE phase (afterok) ────────────────────────────────────────────────────
MERGE_JID=""
if [[ "$SUBMIT_MERGE" == "1" ]]; then
    echo "Submitting MERGE: $N_GRIDS grids (afterok:$RUN_JID)"
    MERGE_JID=$(
        sbatch \
        -p rosa_express.p \
            --parsable \
            --dependency=afterok:$RUN_JID \
            --array=0-$((N_GRIDS - 1)) \
            --export=ALL,N_GRIDS=$N_GRIDS,N_SHARDS=$N_SHARDS,OUTPUT_DIR=$OUTPUT_DIR,CMRES_MERGE_PHASE=1 \
            slurm_single_removal_shed.sh
    )
    echo "MERGE array submitted: jobid=$MERGE_JID"
fi

# ── EVAL phase (afterok) ─────────────────────────────────────────────────────
# Plots run after the merge so every grid has a final
# ``single_removal_shed_<grid>.csv`` to read. With SUBMIT_MERGE=0 we still
# fall back to afterok:<RUN_JID> so the eval doesn't start mid-shard-write;
# the plotter itself just skips any *_shard_*.csv it sees.
if [[ "$SUBMIT_EVAL" == "1" ]]; then
    DEP_JID="${MERGE_JID:-$RUN_JID}"
    echo "Submitting EVAL: 1 task (afterok:$DEP_JID)"
    EVAL_JID=$(
        sbatch \
            -p rosa_express.p \
            --parsable \
            --dependency=afterok:$DEP_JID \
            --export=ALL,OUTPUT_DIR=$OUTPUT_DIR,INPUT_DIR=$INPUT_DIR,E16_OUT_DIR=$E16_OUT_DIR,SKIP_E16=$SKIP_E16 \
            slurm_single_removal_shed_eval.sh
    )
    echo "EVAL job submitted: jobid=$EVAL_JID"
fi

echo "Done. Watch with: squeue -u \$USER"
