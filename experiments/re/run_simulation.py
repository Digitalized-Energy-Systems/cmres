"""
RQMC resilience simulation runner.

Runs all experiment configurations (impact scenarios × test grids) and
writes per-run results to disk.  Designed to run overnight — each individual
experiment can take tens of minutes depending on hardware.

Usage
-----
Run ALL experiments sequentially:
    python experiments/re/run_simulation.py

Run one experiment by 1-based index (for parallelism across terminals):
    python experiments/re/run_simulation.py 1
    python experiments/re/run_simulation.py 2
    ...
    python experiments/re/run_simulation.py 18   # 6 scenarios × 3 grids

Skip already-completed experiments (idempotent):
    python experiments/re/run_simulation.py --resume

Show planned experiments without running:
    python experiments/re/run_simulation.py --list

Experiment grid
---------------
6 impact scenarios × 3 test grids = 18 configurations.

Impact scenarios (Table 1 in the paper):
    high_el   : (power=3, heat=0, gas=0, mes=0)
    high_heat : (power=0, heat=3, gas=0, mes=0)
    high_gas  : (power=0, heat=0, gas=3, mes=0)
    high_cp   : (power=0, heat=0, gas=0, mes=3)
    low_all   : (power=1, heat=1, gas=1, mes=1)
    med_all   : (power=2, heat=2, gas=2, mes=2)

Test grids (see experiments/re/test_grids.py):
    urban_district  — 20 kV / gas / heat, 4 CPs, high coupling density
    industrial_hub  — 110 kV / gas only,  5 CPs, gas-backup focus
    regional_mes    — 120 kV / gas / heat, 7 CPs, all CP types, ring topology

Output
------
data/res/<EXPERIMENT_NAME>/
    network.p        — pickled monee Network
    performance.csv  — per-timestep carrier performance (appended per run)
    failure.csv      — failure events
    repair.csv       — repair events
    mc_result.npz    — MCResult summary (mean, CI, per_run array)
    mc_summary.txt   — human-readable MCResult.summary()
    run.log          — full DEBUG log for this experiment
"""

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402

from monee.model.formulation import MISOCP_NETWORK_FORMULATION  # noqa: E402

from test_grids import ALL_GRIDS  # noqa: E402

import cmres.log  # noqa: E402
from cmres.resilience.mc import (
    ComponentRegistry,
    FailureScenario,
    MCEngine,
    RQMCSampler,
)  # noqa: E402
from cmres.resilience.metric import SimpleResilienceMetric  # noqa: E402
from cmres.resilience.model import SimpleRepairModel, SimpleResilienceModel  # noqa: E402
from cmres.simulation.scenarios import start_res_simulation  # noqa: E402

log = logging.getLogger("cmres.run")

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "res"
EXPERIMENT_NAME = "MoneeResilienceExperiment"
SEED = 101

# Simulation time budget
TIME_STEPS = 4 * 8  # 32 timesteps per run
INCIDENT_TIME_STEPS = 3
INCIDENT_SHIFT = 0
REPAIR_DELAY = 5

# MC convergence
MC_REL_TOL = 0.05  # 5 % relative CI
MC_MAX_RUNS = 2**14  # power of 2
MC_MIN_RUNS = 200  # warm-up before convergence check
MC_ANTITHETIC = True

# ── Experiment grid ───────────────────────────────────────────────────────────

SCENARIOS = [
    # (label,         power, heat, gas, mes)
    ("high_el", 3, 0, 0, 0),
    ("high_heat", 0, 3, 0, 0),
    ("high_gas", 0, 0, 3, 0),
    ("high_cp", 0, 0, 0, 3),
    ("low_all", 1, 1, 1, 1),
    ("med_all", 2, 2, 2, 2),
]

GRIDS = list(ALL_GRIDS.keys())  # ["urban_district", "industrial_hub", "regional_mes"]

# All 18 configs: (scenario_label, power, heat, gas, mes, grid_name)
EXPERIMENTS = [
    (label, p, h, g, m, grid) for (label, p, h, g, m) in SCENARIOS for grid in GRIDS
]


# ── Network factory ───────────────────────────────────────────────────────────


def build_network_and_timeseries(grid_name: str):
    """Construct the named test grid and its demand profiles."""
    create_fn, timeseries_fn = ALL_GRIDS[grid_name]
    net = create_fn()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    td = timeseries_fn(net, n_steps=TIME_STEPS, seed=SEED)
    return net, td


# ── Per-experiment output path ────────────────────────────────────────────────


def exp_dir(
    label: str, power: int, heat: int, gas: int, mes_impact: int, grid_name: str
) -> Path:
    tag = f"{EXPERIMENT_NAME}-{label}-{power}-{heat}-{gas}-{mes_impact}-{grid_name}"
    return DATA_DIR / tag


# ── Simulation runner for one experiment ─────────────────────────────────────


def run_experiment(
    label: str,
    power: int,
    heat: int,
    gas: int,
    mes_impact: int,
    grid_name: str,
):
    """Run RQMC simulation for one (scenario, grid) pair."""

    out_dir = exp_dir(label, power, heat, gas, mes_impact, grid_name)
    out_name = str(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Attach a per-experiment file handler so every DEBUG record is captured.
    cmres.log.setup(log_file=out_dir / "run.log")

    log.info("─" * 60)
    log.info("Experiment : %s  grid=%s", label, grid_name)
    log.info(
        "Impact     : power=%d  heat=%d  gas=%d  mes=%d", power, heat, gas, mes_impact
    )
    log.info("Output     : %s", out_dir)
    log.info("─" * 60)

    # ── Network + timeseries ──────────────────────────────────────────────────
    net, td = build_network_and_timeseries(grid_name)
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
        power_impact=power,
        heat_impact=heat,
        gas_impact=gas,
        mes_impact=mes_impact,
    )
    repair_model = SimpleRepairModel(
        delay_for_repair=REPAIR_DELAY,
        incident_timesteps=INCIDENT_TIME_STEPS,
        incident_shift=INCIDENT_SHIFT,
    )

    run_counter = [0]

    def run_func(scenario):
        run_id = run_counter[0]
        run_counter[0] += 1
        # Deep-copy net so fault inject/repair mutations don't leak between runs.
        # td is read-only, so it can be shared.
        # perf_sum == carrier_sums.sum() — redundant, not saved separately.
        _perf_sum, carrier_sums = start_res_simulation(
            net.copy(),
            td,
            resilience_model=resilience_model,
            repair_model=repair_model,
            resilience_measurement_model=SimpleResilienceMetric(),
            time_steps=TIME_STEPS,
            name=f"{out_name}-{run_id}",
            out_name=out_name,
            id=run_id,
            registry=registry,
            scenario=scenario,
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

    # ── Save MCResult ─────────────────────────────────────────────────────────
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
    )
    (out_dir / "mc_summary.txt").write_text(
        f"label={label}  power={power}  heat={heat}  gas={gas}  "
        f"mes={mes_impact}  grid={grid_name}\n"
        f"elapsed={elapsed:.1f}s\n\n" + result.summary()
    )

    return result


# ── Resume helper ─────────────────────────────────────────────────────────────


def is_done(label, power, heat, gas, mes_impact, grid_name) -> bool:
    return (
        exp_dir(label, power, heat, gas, mes_impact, grid_name) / "mc_result.npz"
    ).exists()


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
        help=f"1-based experiment index (1–{len(EXPERIMENTS)}).  Omit to run all.",
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
    args = parser.parse_args()

    # Initialise console logging before any experiment runs.
    console_level = logging.DEBUG if args.debug else logging.INFO
    cmres.log.setup(console_level=console_level)

    if args.list:
        print(
            f"{'#':>3}  {'label':<12} {'p':>2} {'h':>2} {'g':>2} {'m':>2}  "
            f"{'grid':<16}  {'done?':>6}"
        )
        print("-" * 60)
        for i, (lbl, p, h, g, m, grid) in enumerate(EXPERIMENTS, 1):
            done = "✓" if is_done(lbl, p, h, g, m, grid) else ""
            print(
                f"{i:>3}  {lbl:<12} {p:>2} {h:>2} {g:>2} {m:>2}  {grid:<16}  {done:>6}"
            )
        return

    if args.index is not None:
        idx = args.index - 1
        if not (0 <= idx < len(EXPERIMENTS)):
            log.error("index must be 1–%d", len(EXPERIMENTS))
            sys.exit(1)
        lbl, p, h, g, m, grid = EXPERIMENTS[idx]
        if args.resume and is_done(lbl, p, h, g, m, grid):
            log.info(
                "Skipping #%d (%s, grid=%s) — already done.", args.index, lbl, grid
            )
            return
        run_experiment(lbl, p, h, g, m, grid)
    else:
        total = len(EXPERIMENTS)
        done_n = 0
        skip_n = 0
        for i, (lbl, p, h, g, m, grid) in enumerate(EXPERIMENTS, 1):
            if args.resume and is_done(lbl, p, h, g, m, grid):
                log.info(
                    "[%d/%d] Skipping %s  grid=%s (already done)", i, total, lbl, grid
                )
                skip_n += 1
                continue
            log.info("[%d/%d] Starting %s  grid=%s", i, total, lbl, grid)
            run_experiment(lbl, p, h, g, m, grid)
            done_n += 1

        log.info("Done.  Ran %d, skipped %d/%d.", done_n, skip_n, total)


if __name__ == "__main__":
    main()
