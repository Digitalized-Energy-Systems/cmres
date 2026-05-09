#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CMRES full evaluation worker.
#
# Single-shot SLURM job that runs the full ``cp_cn_evaluation.evaluate(INPUT)``
# pipeline end-to-end:
#
#   - per-scenario eval (cp_metric_vs_actual_impact, cp_only_*,
#     resilience_per_scenario, impact_*)
#   - pooled views (pooled_metric_comparison, cp_only_pooled_*)
#   - CMRES experiments (E2..E16 in cmres_eval.run_cmres_block)
#
# Submit directly — there's no array, no shards, no merge phase:
#
#     sbatch slurm_evaluation.sh
#
# Output:  data/out/<grid>/*.html, data/out/pooled/*.html,
#          data/out/cmres/*.csv  (under whatever OUTPUT directory
#          cp_cn_evaluation.py points at).
#
# Inputs (read by the eval):
#   - <INPUT_DIR>/MoneeResilienceExperiment-<grid>/{network.p,
#       performance.csv, failure.csv, mc_result.npz}
#     where INPUT_DIR is set in cp_cn_evaluation.py::INPUT.
#   - data/out/single_removal_shed/single_removal_shed_<grid>.csv
#     (only used by E16; missing files are skipped cleanly).
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=cmres_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=rico.schrage@uni-oldenburg.de

mkdir -p logs data/out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export JAX_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "============================================================"
echo "Job ID      : ${SLURM_JOB_ID}"
echo "Host        : $(hostname)"
echo "CPUs        : ${SLURM_CPUS_PER_TASK}"
echo "Started at  : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"

# ── HPC environment activation (must match slurm_run_simulations.sh) ────────
module load hpc-env/13.1
module load Miniforge3/26.1.0-0
conda activate cmres_env

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Run the full evaluation. ``INPUT`` and ``OUTPUT`` are module-level
# constants in cp_cn_evaluation.py; edit there to point at a different
# simulation output / output directory.
python -u -c "
import sys
sys.path.insert(0, 'experiments/re')
from cp_cn_evaluation import evaluate, INPUT
evaluate(INPUT)
"
EXIT_CODE=$?

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
