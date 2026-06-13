"""
RQMC resilience simulation runner.

Runs one RQMC resilience experiment per test grid and writes per-run results
to disk.  Designed to run overnight — each individual experiment can take
tens of minutes depending on hardware.

Usage
-----
Run ALL experiments sequentially:
    python experiments/re/run_simulation.py

Run one experiment by 1-based index (for parallelism across terminals):
    python experiments/re/run_simulation.py 1
    python experiments/re/run_simulation.py 2

Skip already-completed experiments (idempotent):
    python experiments/re/run_simulation.py --resume

Show planned experiments without running:
    python experiments/re/run_simulation.py --list

Shard one experiment across K parallel workers
(each worker runs ``MC_MAX_RUNS / K`` runs from a contiguous slice of the
shared Sobol sequence; convergence is **not** checked — the full budget is
spent so all shards finish at the same point):
    python experiments/re/run_simulation.py 1 --shard 1 --n-shards 8
    python experiments/re/run_simulation.py 1 --shard 2 --n-shards 8
    ...

Merge all shards of a finished experiment into the final ``mc_result.npz``:
    python experiments/re/run_simulation.py 1 --merge

Experiment grid
---------------
One experiment per test grid (see experiments/re/test_grids.py): the 15-grid
roster spanning the ``_backup`` / ``_loadbearing`` / ``_control`` scenario
families over six CP-density stems.

Note: earlier revisions multiplied this by 6 "impact scenarios".  Those
scenarios turned out to be unconsumed by the failure model and contributed
no additional scientific signal — see discussion in methodology.tex §2.7.

Output
------
data/res/<EXPERIMENT_NAME>/
    network.p           — pickled monee Network
    performance.csv     — per-timestep carrier performance (appended per run)
    failure.csv         — failure events
    mc_result.npz       — MCResult summary (mean, CI, per_run array)
    mc_summary.txt      — human-readable MCResult.summary()
    run.log             — full DEBUG log for this experiment
    shard_<I>_of_<K>.npz — per-shard ``per_run`` array (sharded mode only;
                           merged into ``mc_result.npz`` by ``--merge``)
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402

from test_grids import ALL_GRIDS  # noqa: E402

import cmres.log  # noqa: E402
from cmres.resilience.mc import (
    ComponentRegistry,
    FailureScenario,
    MCEngine,
    RQMCSampler,
)  # noqa: E402
from cmres.resilience.metric import SimpleResilienceMetric  # noqa: E402
from cmres.resilience.model import SimpleResilienceModel  # noqa: E402
from cmres.simulation.scenarios import start_res_simulation  # noqa: E402

log = logging.getLogger("cmres.run")

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "res"
EXPERIMENT_NAME = "MoneeResilienceExperiment"
SEED = 101

# Simulation time budget.
# Failures persist from f.time to TIME_STEPS-1 (no repair), so TIME_STEPS sets
# the integration horizon for energy-not-served per scenario.
INCIDENT_TIME_STEPS = 3
INCIDENT_SHIFT = 0
TIME_STEPS = 16

# MC convergence
MC_REL_TOL = 0.05  # 5 % relative CI
MC_MAX_RUNS = 2**14  # power of 2
MC_MIN_RUNS = 200  # warm-up before convergence check
MC_ANTITHETIC = True

# ── Experiment grid ───────────────────────────────────────────────────────────

# One experiment per grid: ["urban_district", "industrial_hub", "regional_mes"]
EXPERIMENTS = list(ALL_GRIDS.keys())


# ── Network factory ───────────────────────────────────────────────────────────


def build_network_and_timeseries(grid_name: str):
    """Construct the named test grid and its demand profiles."""
    create_fn, timeseries_fn = ALL_GRIDS[grid_name]
    container = create_fn()
    net = container.network
    # The test_grids factory already applies the formulation set
    # (EL_MISOCP + McCormick heat); re-applying here was redundant and
    # kept run_simulation out of sync with single_removal_shed.
    td = timeseries_fn(net, n_steps=TIME_STEPS, seed=SEED)
    return container, td


# ── Per-experiment output path ────────────────────────────────────────────────


def exp_dir(grid_name: str) -> Path:
    return DATA_DIR / f"{EXPERIMENT_NAME}-{grid_name}"


# ── Simulation runner for one experiment ─────────────────────────────────────


def run_experiment(grid_name: str, shard: int = 0, n_shards: int = 1):
    """Run RQMC simulation for one test grid.

    Parameters
    ----------
    grid_name : str
        Key into ``ALL_GRIDS``.
    shard : int, default 0
        1-based shard index, or 0 for unsharded (full-experiment) mode.
    n_shards : int, default 1
        Total number of shards.  When ``> 1``, each shard runs a contiguous
        slice of the shared Sobol sequence and writes ``shard_<I>_of_<K>.npz``
        instead of the consolidated ``mc_result.npz``.  Convergence checking
        is disabled in shard mode (every shard spends its full budget) so
        shards finish at predictable wall-times for SLURM scheduling.
    """
    is_shard = n_shards > 1
    if is_shard and not (1 <= shard <= n_shards):
        raise ValueError(f"shard {shard} out of range [1, {n_shards}]")

    out_dir = exp_dir(grid_name)
    out_name = str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-(experiment|shard) log file so concurrent shards don't fight over it.
    log_filename = (
        f"shard_{shard}_of_{n_shards}.log" if is_shard else "run.log"
    )
    cmres.log.setup(log_file=out_dir / log_filename)

    log.info("─" * 60)
    log.info("Experiment : grid=%s", grid_name)
    log.info("Output     : %s", out_dir)
    if is_shard:
        log.info("Shard      : %d / %d", shard, n_shards)
    log.info("─" * 60)

    # ── Network + timeseries ──────────────────────────────────────────────────
    container, td = build_network_and_timeseries(grid_name)
    net = container.network
    if not is_shard or shard == 1:
        # Network is identical across shards; only one writer needed.
        with (out_dir / "network.p").open("wb") as fp:
            pickle.dump(net, fp)

    # ── RQMC setup ───────────────────────────────────────────────────────────
    registry = ComponentRegistry(net)
    n_sobol = (MC_MAX_RUNS + 1) // 2 if MC_ANTITHETIC else MC_MAX_RUNS
    sobol_d = registry.n_components * INCIDENT_TIME_STEPS * FailureScenario.N_DIM

    log.info(
        "Components : %d   Sobol d=%d   n_sobol=%d",
        registry.n_components,
        sobol_d,
        n_sobol,
    )

    sampler = RQMCSampler(
        n_components=registry.n_components,
        T_incident=INCIDENT_TIME_STEPS,
        n_scenarios=n_sobol,
        base_seed=SEED,
    )

    if is_shard:
        # Each shard owns a contiguous slice of the Sobol prefix.  Sobol points
        # are paired (point + antithetic twin) so shard slicing is done in
        # *Sobol-point* units.
        shard_sobol = n_sobol // n_shards
        if shard_sobol == 0:
            raise ValueError(
                f"n_shards={n_shards} too large for n_sobol={n_sobol}"
            )
        start_idx = (shard - 1) * shard_sobol
        # Last shard absorbs the remainder so no points are lost.
        end_idx = shard * shard_sobol if shard < n_shards else n_sobol
        sampler._idx = start_idx
        shard_runs = (end_idx - start_idx) * (2 if MC_ANTITHETIC else 1)
        # Disable convergence by setting min_runs > max_runs.
        engine = MCEngine(
            rel_tol=MC_REL_TOL,
            max_runs=shard_runs,
            min_runs=shard_runs + 1,
            antithetic_variates=MC_ANTITHETIC,
            sampler=sampler,
        )
        log.info(
            "Sobol slice: [%d, %d)  shard_runs=%d",
            start_idx, end_idx, shard_runs,
        )
    else:
        engine = MCEngine(
            rel_tol=MC_REL_TOL,
            max_runs=MC_MAX_RUNS,
            min_runs=MC_MIN_RUNS,
            antithetic_variates=MC_ANTITHETIC,
            sampler=sampler,
        )

    # ── Models (created once, reused across runs) ─────────────────────────────
    resilience_model = SimpleResilienceModel(
        incident_shift=INCIDENT_SHIFT,
        incident_timesteps=INCIDENT_TIME_STEPS,
    )

    # Each shard tags its own runs with a globally-unique id range so the
    # CSV-side artefacts (performance.csv, failure.csv, incidents/) don't
    # collide across shards running on different nodes.
    run_id_offset = (shard - 1) * (n_sobol // n_shards) * (2 if MC_ANTITHETIC else 1) if is_shard else 0
    run_counter = [run_id_offset]

    def run_func(scenario):
        run_id = run_counter[0]
        run_counter[0] += 1
        _perf_sum, carrier_sums = start_res_simulation(
            net.copy(),
            td,
            resilience_model=resilience_model,
            resilience_measurement_model=SimpleResilienceMetric(),
            time_steps=TIME_STEPS,
            name=f"{out_name}-{run_id}",
            out_name=out_name,
            id=run_id,
            registry=registry,
            scenario=scenario,
            ext_grid_el_bounds=container.ext_grid_el_bounds,
            ext_grid_gas_bounds=container.ext_grid_gas_bounds,
            ext_grid_heat_bounds=container.ext_grid_heat_bounds,
            include_coupling_points=container.include_coupling_points,
        )
        return carrier_sums  # shape (3,): [power, heat, gas]

    # ── Run ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    result = engine.run(run_func)
    elapsed = time.time() - t0

    log.info(
        "Finished in %.1f min  (n_runs=%d  converged=%s)",
        elapsed / 60,
        result.n_runs,
        result.converged,
    )
    log.info(result.summary())

    # Zero-truncation bias diagnostic. ``SimpleResilienceModel`` guarantees
    # ≥1 failure per scenario, so MC means are conditional on "≥1 failure"
    # and biased upward by ~1/(1 − p_all_zero). ``forced_failure_count``
    # is the empirical estimate of n × P(all-zero). Logged here and
    # persisted in the saved artefacts so downstream eval can correct or
    # at least disclose the bias.
    sc = max(resilience_model.scenario_count, 1)
    p_all_zero = resilience_model.forced_failure_count / sc
    log.info(
        "Zero-truncation bias: %d / %d scenarios needed a forced failure "
        "(P_hat(all-zero)=%.4f, upward-bias factor ~ %.4f)",
        resilience_model.forced_failure_count,
        resilience_model.scenario_count,
        p_all_zero,
        1.0 / (1.0 - p_all_zero) if p_all_zero < 1.0 else float("inf"),
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    if is_shard:
        np.savez(
            out_dir / f"shard_{shard}_of_{n_shards}.npz",
            per_run=result.per_run,
            shard=np.array([shard]),
            n_shards=np.array([n_shards]),
            shard_n_runs=np.array([result.n_runs]),
            elapsed_s=np.array([elapsed]),
            run_id_offset=np.array([run_id_offset]),
            # Bias / accumulator diagnostics per shard. ``merge_shards``
            # sums these across shards before computing P_hat(all-zero).
            n_skipped_nonfinite=np.array([result.n_skipped_nonfinite]),
            forced_failure_count=np.array([resilience_model.forced_failure_count]),
            scenario_count=np.array([resilience_model.scenario_count]),
        )
        log.info(
            "Shard %d/%d saved (%d runs, %.1f min).  "
            "Run ``--merge`` after all shards finish.",
            shard, n_shards, result.n_runs, elapsed / 60,
        )
    else:
        np.savez(
            out_dir / "mc_result.npz",
            mean=result.mean,
            std=result.std,
            ci_lower=result.ci_lower,
            ci_upper=result.ci_upper,
            rel_half_width=result.rel_half_width,
            ess=np.array([result.ess]),
            n_runs=np.array([result.n_runs]),
            converged=np.array([result.converged]),
            per_run=result.per_run,
            # Bias / accumulator diagnostics — see [run] log block above.
            n_skipped_nonfinite=np.array([result.n_skipped_nonfinite]),
            forced_failure_count=np.array([resilience_model.forced_failure_count]),
            scenario_count=np.array([resilience_model.scenario_count]),
            # E7: convergence checkpoints (n, mean[3], rhw[3], ess) per row.
            # Empty array if MCEngine wasn't recording (very old code path).
            convergence=(
                result.convergence
                if result.convergence is not None
                else np.empty((0, 8))
            ),
        )
        (out_dir / "mc_summary.txt").write_text(
            f"grid={grid_name}\n"
            f"elapsed={elapsed:.1f}s\n"
            f"forced_failure_count={resilience_model.forced_failure_count}\n"
            f"scenario_count={resilience_model.scenario_count}\n"
            f"p_all_zero={p_all_zero:.6f}\n"
            f"n_skipped_nonfinite={result.n_skipped_nonfinite}\n\n"
            + result.summary()
        )

    return result


# ── Shard merge ───────────────────────────────────────────────────────────────


def merge_shards(grid_name: str) -> None:
    """Concatenate ``shard_*_of_K.npz`` files into a final ``mc_result.npz``.

    Statistics (mean, std, CI, rel_half_width, ESS) are recomputed from the
    full per-run vector via ``WeightedAccumulator`` so the merged result is
    indistinguishable from a single non-sharded run modulo the missing
    sequential-stopping decision.
    """
    from cmres.resilience.mc import WeightedAccumulator

    out_dir = exp_dir(grid_name)
    shard_files = sorted(out_dir.glob("shard_*_of_*.npz"))
    if not shard_files:
        log.error("No shard files in %s — nothing to merge.", out_dir)
        sys.exit(1)

    log.info("Merging %d shard files for grid=%s", len(shard_files), grid_name)
    per_run_chunks = []
    total_elapsed = 0.0
    forced_total = 0
    scenario_total = 0
    for fp in shard_files:
        z = np.load(fp)
        per_run_chunks.append(z["per_run"])
        total_elapsed += float(z["elapsed_s"][0])
        # Sum bias counters across shards. Older shard files predate this
        # field — fall back to 0 so old runs still merge cleanly.
        if "forced_failure_count" in z.files:
            forced_total += int(z["forced_failure_count"][0])
        if "scenario_count" in z.files:
            scenario_total += int(z["scenario_count"][0])
        log.info(
            "  %s : n_runs=%d  elapsed=%.1f min",
            fp.name, len(z["per_run"]), float(z["elapsed_s"][0]) / 60,
        )
    per_run = np.concatenate(per_run_chunks, axis=0)
    n_runs = len(per_run)

    # Recompute statistics from scratch.
    acc = WeightedAccumulator(n_carriers=per_run.shape[1])
    for x in per_run:
        acc.update(x)
    lo, hi = acc.confidence_interval()
    rhw = acc.relative_half_width()
    converged = bool(np.all(rhw <= MC_REL_TOL))
    p_all_zero = (
        forced_total / scenario_total if scenario_total else 0.0
    )

    np.savez(
        out_dir / "mc_result.npz",
        mean=acc.mean,
        std=acc.std,
        ci_lower=lo,
        ci_upper=hi,
        rel_half_width=rhw,
        ess=np.array([acc.ess]),
        n_runs=np.array([n_runs]),
        converged=np.array([converged]),
        per_run=per_run,
        n_skipped_nonfinite=np.array([acc.n_skipped]),
        forced_failure_count=np.array([forced_total]),
        scenario_count=np.array([scenario_total]),
    )
    summary_lines = [
        f"grid={grid_name}",
        f"merged_from={len(shard_files)} shards",
        f"sum_elapsed_s={total_elapsed:.1f}",
        f"n_runs={n_runs}",
        f"ess={acc.ess:.1f}",
        f"converged={converged}",
        f"forced_failure_count={forced_total}",
        f"scenario_count={scenario_total}",
        f"p_all_zero={p_all_zero:.6f}",
        f"n_skipped_nonfinite={acc.n_skipped}",
        "",
        f"{'Carrier':<8} {'Mean':>10} {'Std':>10} {'CI 95% lo':>12} "
        f"{'CI 95% hi':>12} {'RHW':>8}",
        "-" * 62,
    ]
    for i, name in enumerate(["power", "heat", "gas"]):
        summary_lines.append(
            f"{name:<8} {acc.mean[i]:>10.4f} {acc.std[i]:>10.4f} "
            f"{lo[i]:>12.4f} {hi[i]:>12.4f} {rhw[i]:>8.4f}"
        )
    (out_dir / "mc_summary.txt").write_text("\n".join(summary_lines) + "\n")
    log.info("Merged → %s  (n_runs=%d, ess=%.1f, converged=%s)",
             out_dir / "mc_result.npz", n_runs, acc.ess, converged)


# ── Resume helper ─────────────────────────────────────────────────────────────


def is_done(grid_name: str) -> bool:
    return (exp_dir(grid_name) / "mc_result.npz").exists()


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run RQMC resilience simulations for the MES paper."
    )
    parser.add_argument(
        "index",
        nargs="?",
        type=int,
        default=None,
        help=f"1-based experiment index (1-{len(EXPERIMENTS)}).  Omit to run all.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Grid name (key in ALL_GRIDS).  Alternative to the positional "
        "index; mutually exclusive with it.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose mc_result.npz already exists.",
    )
    parser.add_argument(
        "--list", action="store_true", help="Print planned experiments and exit."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set console log level to DEBUG (very verbose).",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="1-based shard index when running in sharded mode "
        "(requires --n-shards > 1).  0 = unsharded (default).",
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="Total number of shards.  When > 1, this shard runs "
        "MC_MAX_RUNS / N_SHARDS scenarios from a contiguous slice of "
        "the shared Sobol sequence and writes shard_<I>_of_<K>.npz.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all shard_*.npz files for the experiment specified by "
        "INDEX into the final mc_result.npz / mc_summary.txt.  Run once "
        "per experiment after all shards have finished.",
    )
    args = parser.parse_args()

    # Initialise console logging before any experiment runs.
    console_level = logging.DEBUG if args.debug else logging.INFO
    cmres.log.setup(console_level=console_level)

    # Resolve --name to a 1-based index so the rest of the code stays unchanged.
    if args.name is not None:
        if args.index is not None:
            log.error("Pass either INDEX or --name, not both.")
            sys.exit(1)
        if args.name not in EXPERIMENTS:
            log.error(
                "Grid %r not in ALL_GRIDS. Available: %s",
                args.name, EXPERIMENTS,
            )
            sys.exit(1)
        args.index = EXPERIMENTS.index(args.name) + 1

    if args.list:
        print(f"{'#':>3}  {'grid':<24}  {'done?':>6}")
        print("-" * 40)
        for i, grid in enumerate(EXPERIMENTS, 1):
            done = "✓" if is_done(grid) else ""
            print(f"{i:>3}  {grid:<24}  {done:>6}")
        return

    if args.merge:
        if args.index is None:
            # Merge all experiments.
            for grid in EXPERIMENTS:
                merge_shards(grid)
        else:
            idx = args.index - 1
            if not (0 <= idx < len(EXPERIMENTS)):
                log.error("index must be 1-%d", len(EXPERIMENTS))
                sys.exit(1)
            merge_shards(EXPERIMENTS[idx])
        return

    if args.n_shards < 1:
        log.error("--n-shards must be >= 1")
        sys.exit(1)
    if args.n_shards > 1 and not (1 <= args.shard <= args.n_shards):
        log.error(
            "When --n-shards=%d, --shard must be in [1, %d]",
            args.n_shards, args.n_shards,
        )
        sys.exit(1)

    if args.index is not None:
        idx = args.index - 1
        if not (0 <= idx < len(EXPERIMENTS)):
            log.error("index must be 1-%d", len(EXPERIMENTS))
            sys.exit(1)
        grid = EXPERIMENTS[idx]
        if args.resume and is_done(grid):
            log.info("Skipping #%d (grid=%s) — already done.", args.index, grid)
            return
        run_experiment(grid, shard=args.shard, n_shards=args.n_shards)
    else:
        if args.n_shards > 1:
            log.error("--n-shards >1 requires explicit INDEX (one shard per call)")
            sys.exit(1)
        total = len(EXPERIMENTS)
        done_n = 0
        skip_n = 0
        for i, grid in enumerate(EXPERIMENTS, 1):
            if args.resume and is_done(grid):
                log.info("[%d/%d] Skipping grid=%s (already done)", i, total, grid)
                skip_n += 1
                continue
            log.info("[%d/%d] Starting grid=%s", i, total, grid)
            run_experiment(grid)
            done_n += 1

        log.info("Done.  Ran %d, skipped %d/%d.", done_n, skip_n, total)


if __name__ == "__main__":
    main()
