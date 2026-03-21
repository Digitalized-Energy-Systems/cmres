"""
MES resilience experiment entry point.

Run as:
    python experiments/re/cp_cn.py [RUN_ID]

RUN_ID is 1-based and selects (impact_scenario, cp_density) from the
cartesian product of NUM_TO_EXP_MAP × DENSITY_MAP.  Multiple parallel
processes with different RUN_IDs accumulate results into the same CSV files
(writes are serialised with a file lock).

RQMC methodology
----------------
Each run uses a distinct Owen-scrambled Sobol point as its random input.
All failure decisions and repair-time draws for a run come from that single
Sobol point (reshaped to [n_components × T_incident × 4]) rather than from
the global pseudo-random number generator.  The antithetic twin of each
Sobol point (1 − u) is evaluated in the immediately following run, giving
maximally negatively correlated pairs.

This combination of RQMC + antithetic variates typically halves the number
of simulation runs required to reach a given confidence-interval width
compared with plain pseudo-random Monte Carlo.
"""

import sys
import pickle
import numpy as np
import random
from pathlib import Path

from cmres.resilience.mc import (
    MCEngine,
    ComponentRegistry,
    FailureScenario,
    RQMCSampler,
)
from cmres.resilience.metric import SimpleResilienceMetric
from cmres.resilience.model import SimpleRepairModel, SimpleResilienceModel
from cmres.simulation.scenarios import start_res_simulation

import monee.network.mes as mes
import monee.model as mm
from monee.io.from_simbench import obtain_simbench_net
from monee import TimeseriesData

# ── Experiment constants ────────────────────────────────────────────────────

EXPERIMENT_NAME = "data/res/MoneeResilienceExperiment"
SEED = 101  # deterministic seed for the initial network build
TIME_STEPS = 4 * 8  # total simulation time steps
REPAIR_DELAY = 5
INCIDENT_TIME_STEPS = 3
INCIDENT_SHIFT = 0

# MC convergence settings
MC_REL_TOL = 0.05  # stop when 95 % CI relative half-width ≤ 5 % for all carriers
MC_MAX_RUNS = 2000  # hard upper bound
MC_MIN_RUNS = 200  # warm-up before convergence is checked
MC_ANTITHETIC = True  # antithetic-variates variance reduction
MC_VERBOSE = True

# Parameter grid
NUM_TO_EXP_MAP = [
    # (power_impact, gas_impact, heat_impact, mes_impact)
    (5, 5, 5, 5),
]
DENSITY_MAP = [0, 0.5, 1, 1.5, 2]


# ── Network construction ────────────────────────────────────────────────────


def create_common_network(simbench_id, cp_density_coeff):
    net_simbench = obtain_simbench_net(simbench_id)
    new_mes = net_simbench.copy()
    for child in net_simbench.childs_by_type(mm.PowerGenerator):
        child.model.p_mw = child.model.p_mw * 4
    mes.create_gas_net_for_power(
        net_simbench,
        new_mes,
        1,
        source_scaling=1,
        default_diameter_m=0.64,
        length_scale=0.001,
        default_length=100000,
    )
    return mes.create_monee_benchmark_net()


# ── Per-run simulation function ─────────────────────────────────────────────


def make_run_func(
    net,
    power_impact,
    heat_impact,
    gas_impact,
    mes_impact,
    out_name,
    process_id,
    num_iter,
    registry,
):
    """Return a run_func compatible with MCEngine.run() (RQMC mode).

    The closure captures fixed simulation parameters and the ComponentRegistry.
    The returned function accepts a single FailureScenario; all stochastic
    decisions come from that scenario's pre-sampled Sobol uniforms.

    The models are created once and reused across runs — in scenario mode their
    internal RNG is bypassed for failure/repair decisions, so re-instantiation
    is not required.
    """
    run_counter = [0]

    # Create models once; stochasticity comes from the FailureScenario.
    resilience_model = SimpleResilienceModel(
        incident_shift=INCIDENT_SHIFT,
        incident_timesteps=INCIDENT_TIME_STEPS,
        power_impact=power_impact,
        heat_impact=heat_impact,
        gas_impact=gas_impact,
        mes_impact=mes_impact,
    )
    repair_model = SimpleRepairModel(
        delay_for_repair=REPAIR_DELAY,
        incident_timesteps=INCIDENT_TIME_STEPS,
        incident_shift=INCIDENT_SHIFT,
    )

    def run_func(scenario: FailureScenario) -> np.ndarray:
        run_id = process_id * num_iter + run_counter[0]
        run_counter[0] += 1

        _perf_sum, carrier_sums = start_res_simulation(
            net,
            TimeseriesData(),
            resilience_model=resilience_model,
            repair_model=repair_model,
            resilience_measurement_model=SimpleResilienceMetric(),
            time_steps=TIME_STEPS,
            name=(
                f"{out_name}-"
                f"{power_impact}-{heat_impact}-{gas_impact}-{mes_impact}"
                f"-{run_id}"
            ),
            out_name=out_name,
            id=run_id,
            registry=registry,
            scenario=scenario,
        )
        # Return per-carrier loss vector: [power, heat, gas]
        return carrier_sums

    return run_func


# ── Main experiment function ─────────────────────────────────────────────────


def start_test_sim(
    simbench_id,
    power_impact=1,
    heat_impact=1,
    gas_impact=1,
    mes_impact=1,
    cp_density=1,
    process_id=0,
):
    # Build network with a fixed seed so topology is deterministic.
    np.random.seed(SEED if cp_density != 1 else SEED + 50)
    random.seed(SEED if cp_density != 1 else SEED + 50)
    net = create_common_network(simbench_id, cp_density)
    print(net.as_dataframe_dict_str())

    out_name = (
        f"{EXPERIMENT_NAME}"
        f"-{power_impact}-{heat_impact}-{gas_impact}-{mes_impact}-{cp_density}"
    )
    out_path = Path(out_name)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / Path("network.p")).open("wb") as fp:
        pickle.dump(net, fp)

    # Build component registry from the network for RQMC scenario indexing.
    registry = ComponentRegistry(net)
    print(
        f"[MC] ComponentRegistry: {registry.n_components} components, "
        f"Sobol d = {registry.n_components * INCIDENT_TIME_STEPS * 4}"
    )

    # Derive a unique, reproducible per-process base seed.
    base_seed = SEED * 1000 + process_id

    # Pre-generate Owen-scrambled Sobol scenarios.
    # With antithetic variates each Sobol point yields two runs (original +
    # reflected), so we need ceil(max_runs / 2) points.  Use ceiling division
    # so odd max_runs values don't leave us one scenario short.
    n_sobol = ((MC_MAX_RUNS + 1) // 2) if MC_ANTITHETIC else MC_MAX_RUNS
    sampler = RQMCSampler(
        n_components=registry.n_components,
        T_incident=INCIDENT_TIME_STEPS,
        n_scenarios=n_sobol,
        base_seed=base_seed,
    )

    engine = MCEngine(
        rel_tol=MC_REL_TOL,
        max_runs=MC_MAX_RUNS,
        min_runs=MC_MIN_RUNS,
        antithetic_variates=MC_ANTITHETIC,
        sampler=sampler,
        verbose=MC_VERBOSE,
    )

    run_func = make_run_func(
        net=net,
        power_impact=power_impact,
        heat_impact=heat_impact,
        gas_impact=gas_impact,
        mes_impact=mes_impact,
        out_name=out_name,
        process_id=process_id,
        num_iter=MC_MAX_RUNS,
        registry=registry,
    )

    result = engine.run(run_func)

    print("\n" + result.summary())

    return result


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    num = 1
    if len(sys.argv) > 1:
        num = int(sys.argv[1])

    impacts = NUM_TO_EXP_MAP[(num - 1) % len(NUM_TO_EXP_MAP)]
    density = DENSITY_MAP[((num - 1) // len(NUM_TO_EXP_MAP)) % len(DENSITY_MAP)]

    start_test_sim(
        "1-LV-urban6--2-no_sw",
        power_impact=impacts[0],
        gas_impact=impacts[1],
        heat_impact=impacts[2],
        mes_impact=impacts[3],
        cp_density=density,
        process_id=num - 1,
    )
