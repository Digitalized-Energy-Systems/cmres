"""
Test multi-energy system (MES) networks for resilience simulation research.

Three grids of increasing size and coupling diversity:

  Grid 1 – Urban residential district  (20 kV / gas / heat, high CP density)
  Grid 2 – Industrial energy hub        (110 kV / gas only, gas-backup focus)
  Grid 3 – Regional integrated MES      (120 kV / gas / heat, ring, all CP types)

Each ``create_*`` function returns a monee Network ready for energy-flow or
resilience simulation.  Companion ``make_*_timeseries`` functions return a
matching TimeseriesData with sinusoidal demand profiles.

Physical parameters are chosen to match real infrastructure:
  - 20 kV cable:     r = x = 3e-4 Ω/m, max_i_ka = 0.30 kA  → ~10 MVA/line
  - 110 kV OHL:      r = x = 7e-5 Ω/m, max_i_ka = 0.40 kA  → ~76 MVA/line
  - 120 kV cable:    r = x = 3e-4 Ω/m, max_i_ka = 0.30 kA  → ~62 MVA/line
  - Gas pipes:  d = 0.10–0.50 m, Weymouth-friction regime
  - Heat pipes: d = 0.10–0.20 m, Darcy-Weisbach regime

Coupling point parameters:
  - CHP:  η_el = 0.40, η_th = 0.40  (80 % total, typical micro-CHP)
  - G2P:  η    = 0.88               (gas turbine CCGT)
  - P2G:  η    = 0.70               (PEM electrolyser)
  - P2H:  η    = 0.95               (heat pump / resistance heater)
"""

import numpy as np
import monee.model as mm
from monee import TimeseriesData
from monee import (
    PyomoSolver,
    run_energy_flow_optimization,
)
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
import monee.problem as mp
from monee.network import (
    create_urban_district_net,
    create_resilient_urban_mes_net,
    create_balanced_urban_mes_timeseries,
    create_industrial_hub_net,
    create_regional_mes_net,
)

# =============================================================================
# Helpers
# =============================================================================


def _sinusoidal_profile(
    n_steps: int,
    base: float,
    amplitude: float = 0.25,
    noise: float = 0.04,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Bell-shaped daily demand curve with small Gaussian noise.

    Profile peaks near the midpoint (noon) and troughs at the edges (night).
    Clipped to [50 %, 200 %] of base to avoid unphysical values.
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    profile = base * (1.0 + amplitude * np.sin(t - np.pi / 2))
    profile += rng.normal(0, noise * base, n_steps)
    return np.clip(profile, 0.5 * base, 2.0 * base)


def make_urban_district_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Demand profiles for Grid 1 (96 steps ≈ 24 h at 15-min resolution)."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id, "p_mw", _sinusoidal_profile(n_steps, base, amplitude=0.25, rng=rng)
        )
    for c in net.childs_by_type(mm.Sink):
        if c.grid.name == "gas":
            amp = 0.30
            base = float(mm.value(c.model.mass_flow))
            td.add_child_series(
                c.id,
                "mass_flow",
                _sinusoidal_profile(n_steps, base, amplitude=amp, rng=rng),
            )
    return td


def make_industrial_hub_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Flat industrial demand profiles with small daytime variation."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id,
            "p_mw",
            _sinusoidal_profile(n_steps, base, amplitude=0.15, noise=0.02, rng=rng),
        )
    for c in net.childs_by_type(mm.Sink):
        base = float(mm.value(c.model.mass_flow))
        td.add_child_series(
            c.id,
            "mass_flow",
            _sinusoidal_profile(n_steps, base, amplitude=0.20, noise=0.03, rng=rng),
        )
    return td


def make_regional_mes_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Sinusoidal demand profiles for Grid 3 with per-carrier amplitude tuning."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id, "p_mw", _sinusoidal_profile(n_steps, base, amplitude=0.25, rng=rng)
        )
    for c in net.childs_by_type(mm.Sink):
        if c.grid.name == "gas":
            amp = 0.30
            base = float(mm.value(c.model.mass_flow))
            td.add_child_series(
                c.id,
                "mass_flow",
                _sinusoidal_profile(n_steps, base, amplitude=amp, rng=rng),
            )
    return td


# =============================================================================
# Convenience registry
# =============================================================================

ALL_GRIDS = {
    "large_urban_balanced": (create_resilient_urban_mes_net, create_balanced_urban_mes_timeseries),
    "urban_district": (create_urban_district_net, make_urban_district_timeseries),
    "industrial_hub": (create_industrial_hub_net, make_industrial_hub_timeseries),
    "regional_mes": (create_regional_mes_net, make_regional_mes_timeseries),
}

BOUND_EL = ("vm_pu", 1, 0.1)
BOUND_GAS = ("pressure_pu", 1, 0.1)
BOUND_HEAT = ("t_pu", 1, 0.1)
BOUND_LP = ("loading_percent", 1, 1)


def solve(network):
    optimization_problem = None
    bounds_el = (
        BOUND_EL[1] * (1 - BOUND_EL[2]),
        BOUND_EL[1] * (1 + BOUND_EL[2]),
    )
    bounds_heat = (
        BOUND_HEAT[1] * (1 - BOUND_HEAT[2]),
        BOUND_HEAT[1] * (1 + BOUND_HEAT[2]),
    )
    bounds_gas = (
        BOUND_GAS[1] * (1 - BOUND_GAS[2]),
        BOUND_GAS[1] * (1 + BOUND_GAS[2]),
    )

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        use_ext_grid_bounds=False,
    )

    return run_energy_flow_optimization(
        network,
        solver=PyomoSolver(),
        optimization_problem=optimization_problem,
        exclude_unconnected_nodes=True,
    )


if __name__ == "__main__":
    import monee.model as mm

    print("URBAN")
    print("-------")
    net = create_urban_district_net()
    td = make_urban_district_timeseries(net)
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)

    # for step in range(20):
    #     td.apply_to_network(net, step)
    #     print(solve(net, optimization_problem=None, solver=PyomoSolver()))
    net.deactivate(net.branches_by_type(mm.GasPipe)[0])
    # run_timeseries(net, td, solver=PyomoSolver())
    print(solve(net))

    # print("Industrial")
    # print("-------")
    # net = create_industrial_hub_net()
    # net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # print(run_energy_flow(net, solver=PyomoSolver()))

    # print("Regional")
    # print("-------")
    # net = create_regional_mes_net()
    # net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # print(run_energy_flow(net, solver=PyomoSolver()))
