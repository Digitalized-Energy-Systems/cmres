#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CMRES single-removal-shed evaluation.
#
# Two-step job that produces every artefact a reader needs from the merged
# ``single_removal_shed_<grid>.csv`` files:
#
#   1. ``single_removal_shed_plots.py`` — per-grid HTML report (with PDFs
#      under ``single/``) plus the cross-grid pooled report
#      ``pooled_report.html``.
#   2. ``e16_plots.py``                 — cmres_eval experiment E16
#      (single-removal-shed validation): rebuilds df_eval per scenario from
#      the MC outputs, joins on the shed CSV, then runs
#      ``experiment_e16_single_removal_validation`` + the matching plot
#      routines. Writes ``E16_*.csv`` and ``E16_single_removal.html`` into
#      ``$E16_OUT_DIR``.
#
# Both plotters skip ``*_shard_<i>_of_<k>.csv`` files in the same
# directory (those are pre-merge artefacts), so this job is safe to
# submit as soon as at least one grid has finished merging.
#
# Submit standalone::
#
#     sbatch slurm_single_removal_shed_eval.sh
#     OUTPUT_DIR=data/out/single_removal_shed sbatch slurm_single_removal_shed_eval.sh
#
#     # Standalone with custom MC-output dir + E16 destination:
#     INPUT_DIR=data_0512/res E16_OUT_DIR=data/out/cmres \
#         sbatch slurm_single_removal_shed_eval.sh
#
# …or chain it after the merge phase via ``submit_single_removal_shed.sh``
# (which sets ``--dependency=afterok:<MERGE_JID>`` automatically when
# ``SUBMIT_EVAL=1``).
#
# Re-runs are idempotent and cheap; the slow part of the pipeline (the
# per-component LP solves) is already amortised into the merged CSVs. E16
# also calls ``cp_metric_vs_actual_impact`` per grid which is the only
# moderately expensive step here (≤ a few minutes per grid).
#
# Environment knobs
# -----------------
#   OUTPUT_DIR      where the merged shed CSVs live, AND where the per-grid
#                   + pooled shed reports land
#                   (default: data/out/single_removal_shed)
#   INPUT_DIR       MC-experiment dir with MoneeResilienceExperiment-<grid>/
#                   subfolders (network.p, performance.csv, failure.csv).
#                   Required by E16; if empty / missing the E16 step is
#                   skipped with a warning so the shed plots still publish.
#                   (default: data/res)
#   E16_OUT_DIR     where E16_*.csv and E16_single_removal.html land
#                   (default: data/out/cmres)
#   SKIP_E16        set to 1 to run only the shed plots
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cmres_srs_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/srs_eval_%j.out
#SBATCH --error=logs/srs_eval_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de

# Configurable directories — defaults match the rest of the SRS pipeline.
OUTPUT_DIR=${OUTPUT_DIR:-data/out/single_removal_shed}
INPUT_DIR=${INPUT_DIR:-data/res}
E16_OUT_DIR=${E16_OUT_DIR:-data/out/cmres}
SKIP_E16=${SKIP_E16:-0}

# Match the worker thread-count caps used elsewhere so plotly/Kaleido /
# numpy don't oversubscribe the node.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

SHED_PLOTS="./experiments/re/single_removal_shed_plots.py"
E16_PLOTS="./experiments/re/e16_plots.py"
mkdir -p ./logs "$OUTPUT_DIR" "$E16_OUT_DIR"

echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Host        : $(hostname)"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-?}"
echo "Shed dir    : $OUTPUT_DIR"
echo "MC dir      : $INPUT_DIR"
echo "E16 dir     : $E16_OUT_DIR"
echo "Started at  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── HPC environment activation (must match slurm_single_removal_shed.sh) ─────
# Same caveat as the worker: without these, SLURM falls back to the system
# Python (≤ 3.6 on this cluster) and chokes on ``from __future__ import
# annotations`` before the script even runs.
module load hpc-env/13.1
module load Miniforge3/26.1.0-0
conda activate cmres_env

EXIT_CODE=0

# ── Step 1: shed plots (per-grid + pooled) ────────────────────────────────────
# Auto-discovers ``single_removal_shed_*.csv`` and filters out shard CSVs
# (``*_shard_<i>_of_<k>.csv``) so this works as soon as the merge phase
# has produced at least one merged file.
echo "------------------------------------------------------------"
echo "[1/2] shed plots → $OUTPUT_DIR"
echo "------------------------------------------------------------"
python -u "$SHED_PLOTS" --dir "$OUTPUT_DIR"
SHED_EXIT=$?
if [[ $SHED_EXIT -ne 0 ]]; then
    echo "[srs_eval] shed plots failed (exit=$SHED_EXIT)" >&2
    EXIT_CODE=$SHED_EXIT
fi

# ── Step 2: cmres_eval E16 (single-removal-shed validation) ──────────────────
# E16 needs the MC outputs (perf/fail/network.p per grid) to rebuild
# df_eval. When INPUT_DIR is missing / empty we skip with a warning so
# the shed plots produced in step 1 still publish.
if [[ "$SKIP_E16" == "1" ]]; then
    echo "[srs_eval] SKIP_E16=1 set — not running E16."
elif [[ ! -d "$INPUT_DIR" ]]; then
    echo "[srs_eval] INPUT_DIR='$INPUT_DIR' not found — skipping E16."
elif ! compgen -G "$INPUT_DIR/MoneeResilienceExperiment-*" > /dev/null; then
    echo "[srs_eval] no MoneeResilienceExperiment-* folders in $INPUT_DIR — skipping E16."
else
    echo "------------------------------------------------------------"
    echo "[2/2] cmres_eval E16 → $E16_OUT_DIR"
    echo "------------------------------------------------------------"
    python -u "$E16_PLOTS" \
        --input-dir  "$INPUT_DIR" \
        --shed-dir   "$OUTPUT_DIR" \
        --output-dir "$E16_OUT_DIR"
    E16_EXIT=$?
    if [[ $E16_EXIT -ne 0 ]]; then
        echo "[srs_eval] E16 failed (exit=$E16_EXIT)" >&2
        # Don't overwrite a shed-plots failure with a later E16 success/failure;
        # surface the first non-zero step.
        if [[ $EXIT_CODE -eq 0 ]]; then EXIT_CODE=$E16_EXIT; fi
    fi
fi

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
