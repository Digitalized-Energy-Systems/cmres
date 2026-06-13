import monee.problem as mp
from monee import PyomoSolver, run_energy_flow_optimization

BOUND_EL = ("vm_pu", 1, 0.1)
BOUND_GAS = ("pressure_pu", 1, 0.1)
BOUND_HEAT = ("t_pu", 1, 0.1)

DEFAULT_EXT_GRID_EL_BOUNDS = (-0.05, 0.05)
DEFAULT_EXT_GRID_GAS_BOUNDS = (-0.006, 0.006)
DEFAULT_EXT_GRID_HEAT_BOUNDS = (-6.0, 6.0)


def solve(
    network,
    ext_grid_el_bounds=DEFAULT_EXT_GRID_EL_BOUNDS,
    ext_grid_gas_bounds=DEFAULT_EXT_GRID_GAS_BOUNDS,
    ext_grid_heat_bounds=DEFAULT_EXT_GRID_HEAT_BOUNDS,
    include_coupling_points=False,
):

    optimization_problem = mp.create_min_load_shedding_problem(
        bounds_vm=(0.9, 1.1),
        bounds_pressure=(0.9, 1.1),
        bounds_t=(0.8, 1.15),
        bounds_ext_el=ext_grid_el_bounds,
        bounds_ext_gas=ext_grid_gas_bounds,
        bounds_ext_heat=ext_grid_heat_bounds,
        include_ext_grids=True,
        include_coupling_points=include_coupling_points,
        check_vm=True,
        check_pressure=True,
        check_t=True,
        check_lp=True,
        priority_safety_factor=1000.0,
    )

    return run_energy_flow_optimization(
        network,
        solver=PyomoSolver(),
        solver_name="gurobi",
        optimization_problem=optimization_problem,
        exclude_unconnected_nodes=True,
    )
