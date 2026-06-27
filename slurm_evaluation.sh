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

# The eval segfaulted in native code (no faulthandler output) and the crash
# vanished under gdb — a heisenbug, i.e. a race/corruption in a multithreaded
# numeric library (BLAS/OpenMP). Pin the native libs to a single thread to
# remove the oversubscription race. Override via CMRES_NUM_THREADS if needed.
NTHREADS=${CMRES_NUM_THREADS:-1}
export OMP_NUM_THREADS=${NTHREADS}
export MKL_NUM_THREADS=${NTHREADS}
export OPENBLAS_NUM_THREADS=${NTHREADS}
export JAX_NUM_THREADS=${NTHREADS}
export NUMBA_NUM_THREADS=${NTHREADS}

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
# simulation output / output directory. CMRES_INPUT (env) overrides
# the module default for one-off runs without editing the .py.
# Capture a real core dump (lands in CWD = SLURM_SUBMIT_DIR) and print the
# kernel core pattern so the file can be located after a native crash.
ulimit -c unlimited
echo "ulimit -c    : $(ulimit -c)"
echo "core_pattern : $(cat /proc/sys/kernel/core_pattern 2>/dev/null)"

# PYTHONFAULTHANDLER makes the interpreter dump a Python+C traceback to stderr
# (the eval_%j.err file) on a segfault instead of dying silently.
export PYTHONFAULTHANDLER=1

# gdb batch mode dumps a native C backtrace into the .err, naming the offending
# shared library. It also serializes execution enough to MASK the threading race
# above — so it is OFF by default (the single-thread pin is the actual fix). Set
# CMRES_GDB=1 to re-enable it for diagnosing any future native crash.
PY_SCRIPT='
import os, sys, faulthandler
faulthandler.enable()
sys.path.insert(0, "experiments/re")
from cp_cn_evaluation import evaluate, INPUT
evaluate(os.environ.get("CMRES_INPUT") or INPUT)
'
if [ -n "${CMRES_GDB:-}" ] && command -v gdb >/dev/null 2>&1; then
    gdb -q -batch \
        -ex "set pagination off" \
        -ex "run" \
        -ex "echo \n==== NATIVE BACKTRACE (faulting thread) ====\n" \
        -ex "bt" \
        -ex "echo \n==== ALL THREADS ====\n" \
        -ex "thread apply all bt" \
        --args python -u -X faulthandler -c "$PY_SCRIPT"
    EXIT_CODE=$?
else
    python -u -X faulthandler -c "$PY_SCRIPT"
    EXIT_CODE=$?
fi

echo "============================================================"
echo "Finished at : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Exit code   : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
