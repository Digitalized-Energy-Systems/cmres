import monee.problem as mp
from monee import PyomoSolver, run_energy_flow_optimization

BOUND_EL = ("vm_pu", 1, 0.1)
BOUND_GAS = ("pressure_pu", 1, 0.1)
BOUND_HEAT = ("t_pu", 1, 0.1)


def solve(network):

    optimization_problem = mp.create_min_load_shedding_problem(
        bounds_el=(0.9, 1.1),
        bounds_gas=(0.9, 1.1),
        bounds_heat=(0.8, 1.15),
        ext_grid_el_bounds=(-0.01, 0.01),
        ext_grid_gas_bounds=(-0.01, 0.01),
        ext_grid_heat_bounds=(-100, 100),
        include_ext_grids=True,
        check_vm=True,
        check_pressure=True,
        check_temperature=True,
        check_line_loading=True,
    )

    return run_energy_flow_optimization(
        network,
        solver=PyomoSolver(),
        solver_name="gurobi",
        optimization_problem=optimization_problem,
        exclude_unconnected_nodes=True,
    )

