from monee import run_energy_flow_optimization
import monee.problem as mp


BOUND_EL = ("vm_pu", 1, 0.2)
BOUND_GAS = ("pressure_pa", 500000, 0.5)
BOUND_HEAT = ("t_k", 352, 0.1)
BOUND_LP = ("loading_percent", 0.75, 1)


def solve(network, without_load=False):
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
    bounds_lp = (
        BOUND_LP[1] * (1 - BOUND_LP[2]),
        BOUND_LP[1] * (1 + BOUND_LP[2]),
    )
    if without_load:
        optimization_problem = mp.create_ls_init_optimization_problem(
            bounds_el=bounds_el,
            bounds_heat=bounds_heat,
            bounds_gas=bounds_gas,
            bounds_lp=bounds_lp,
            target_ext_grid_el_feed=0,
            target_ext_grid_gas_feed=0.3,
        )
    else:
        optimization_problem = mp.create_load_shedding_optimization_problem(
            bounds_el=bounds_el,
            bounds_heat=bounds_heat,
            bounds_gas=bounds_gas,
            bounds_lp=bounds_lp,
            ext_grid_el_bounds=(0.0, 0.0),
            ext_grid_gas_bounds=(0.0, 0.0),
        )

    return run_energy_flow_optimization(
        network, optimization_problem=optimization_problem
    )
