import monee.problem as mp
from monee import PyomoSolver, run_energy_flow_optimization

BOUND_EL = ("vm_pu", 1, 0.1)
BOUND_GAS = ("pressure_pu", 1, 0.1)
BOUND_HEAT = ("t_pu", 1, 0.1)


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
