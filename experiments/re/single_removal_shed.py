"""Analytical single-removal load-shed tool.

For each component on a solved monee network, deactivate it, run a
min-load-shedding optimisation, record the total load shed, then reactivate.
The resulting (component_id → total_shed_mw) table is the deterministic,
MC-independent ground truth that bounds what any structural criticality
metric could achieve on this grid.

CLI for SLURM
-------------

Per-shard run::

    python single_removal_shed.py simbench_lv \\
        --input-dir /path/to/run_simulation_output \\
        --output-dir data/out/single_removal_shed \\
        --shard 1 --n-shards 8

Each shard takes a contiguous slice of the component list, runs its
deactivate-solve-reactivate loop, and writes
``single_removal_shed_<grid>_shard_<I>_of_<K>.csv``.

Final merge::

    python single_removal_shed.py simbench_lv \\
        --output-dir data/out/single_removal_shed \\
        --merge --n-shards 8

Output: ``single_removal_shed_<grid>.csv`` ready to join against df_eval
on ``cp_id``.

The component slicing is deterministic (sorted by id), so re-running a
particular shard reproduces its slice.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import monee.model as mm
import monee.problem as mp
from monee import PyomoSolver, run_energy_flow_optimization

# Cap per-solve Gurobi runtime and loosen the MIP gap slightly. With
# ``demand_weight=1e3`` the LP objective is O(1e2-1e3), so the default
# ``MIPGap=1e-3`` demands sub-mW precision — far below the analysis's
# noise floor and prone to multi-hour B&B on degenerate single-removal
# cases. ``MIPGap=5e-3`` gives ~0.5 % precision (~few mW absolute) and
# the ``TimeLimit`` is a safety net so a single stuck solve cannot
# block the whole sweep.
from monee.solver.pyo import PER_SOLVER_OPTIONS as _MONEE_GUROBI_OPTS
_MONEE_GUROBI_OPTS["gurobi"]["MIPGap"] = 5e-3
_MONEE_GUROBI_OPTS["gurobi"]["TimeLimit"] = 300

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Component enumeration
# ─────────────────────────────────────────────────────────────────────────────


def _enumerate_targets(monee_net) -> List[Tuple[str, str, object]]:
    """Return ``(cp_id_str, kind, component)`` triples for everything we'd
    deactivate. Mirrors the enumeration used by ``mes_all_components_metric``
    so the cp_id strings match df_eval's keys 1-1.
    """
    targets: List[Tuple[str, str, object]] = []

    # Compound CPs (CHP / CHPHG / PowerToHeat) → cp_id "compound:{id}"
    for cls in (mm.CHP, mm.CHPHG, mm.PowerToHeat):
        for cp in monee_net.compounds_by_type(cls):
            targets.append((f"compound:{cp.id}", "compound", cp))

    # Branch CPs (PowerToGas / GasToPower / PowerToHeatHG / GasToHeatHG)
    # → cp_id "from→to" (matches mes_cp_metric's branch CP rows)
    for cls in (mm.PowerToGas, mm.GasToPower, mm.PowerToHeatHG, mm.GasToHeatHG):
        for b in monee_net.branches_by_type(cls):
            targets.append(
                (f"{b.from_node_id}→{b.to_node_id}", "branch_cp", b)
            )

    # Non-CP branches → cp_id is the str of the branch id (e.g. "(5, 134, 0)")
    cp_branch_ids = set()
    for cls in (mm.PowerToGas, mm.GasToPower, mm.PowerToHeatHG, mm.GasToHeatHG):
        cp_branch_ids.update(b.id for b in monee_net.branches_by_type(cls))
    for cls in (mm.GenericPowerBranch, mm.GasPipe, mm.WaterPipe, mm.HeatExchanger):
        for b in monee_net.branches_by_type(cls):
            if b.id in cp_branch_ids:
                continue
            targets.append((str(b.id), "branch", b))

    return targets


# ─────────────────────────────────────────────────────────────────────────────
# One-component analytical shed
# ─────────────────────────────────────────────────────────────────────────────


def _solve_load_shed(
    monee_net,
    ext_grid_el_bounds: Tuple[float, float],
    ext_grid_gas_bounds: Tuple[float, float],
    ext_grid_heat_bounds: Tuple[float, float],
):
    """Run the same min-load-shedding problem the resilience model uses
    when the hard solve goes infeasible. Returns the result object on
    success, ``None`` on failure.

    The three ext-grid bound tuples MUST match what the resilience model
    used during the MC simulation (see ``MESContainer`` in test_grids.py),
    otherwise the analytical shed and the MC actuals will use different
    external slack capacities and the comparison is biased.
    """
    # Weight only demand shed, and scale it so it dwarfs the
    # formulation-level tightening terms (MISOC ``current·br_r`` on
    # power branches, NL ``1e-5·mass_flow²`` on pipes) that the
    # solver-side formulations add to ``pm.obj``. Those terms total
    # O(0.2) on simbench_lv; with the default weights the LP will
    # happily shift O(0.4) of demand shed into formulation slack
    # when a CP is removed, which breaks monotonicity for
    # ``_shed_from_solved`` (it only sees the demand side).
    opt = mp.create_min_load_shedding_problem(
        demand_weight=1e3,
        generator_weight=0.1,
        ext_grid_weight=0.1,
        bounds_el=(0.9, 1.1),
        bounds_gas=(0.9, 1.1),
        bounds_heat=(0.7, 1.3),
        ext_grid_el_bounds=ext_grid_el_bounds,
        ext_grid_gas_bounds=ext_grid_gas_bounds,
        ext_grid_heat_bounds=ext_grid_heat_bounds,
        include_ext_grids=True,
        check_vm=True,
        check_pressure=True,
        check_temperature=True,
        check_line_loading=True,
    )
    try:
        return run_energy_flow_optimization(
            monee_net,
            solver=PyomoSolver(),
            solver_name="gurobi",
            optimization_problem=opt,
            exclude_unconnected_nodes=True,
        )
    except Exception:
        traceback.print_exc()
        return None


def _shed_from_solved(net) -> Tuple[float, float, float, float]:
    """Extract per-carrier and total load shed from a solved network.
    Mirrors ``GeneralResiliencePerformanceMetric.calc(network)``.

    Returns (power_mw, heat_mw, gas_mw, total_mw).
    """
    passive_hx = getattr(mm, "PassiveHeatExchangerLoad", mm.HeatExchangerLoad)

    power = 0.0
    for c in net.childs:
        m = c.model
        if c.ignored or not c.active:
            if isinstance(m, mm.PowerLoad):
                power += float(mm.upper(m.p_mw) or 0.0)
            continue
        if isinstance(m, mm.PowerLoad):
            try:
                power += float(mm.upper(m.p_mw) or 0.0) - float(
                    mm.value(m.p_mw) or 0.0
                ) * float(mm.value(m.regulation) or 1.0)
            except Exception:
                pass

    heat = 0.0
    for c in net.childs:
        m = c.model
        if c.ignored or not c.active:
            if isinstance(m, mm.HeatLoad):
                heat += float(mm.upper(m.q_mw_heat) or 0.0)
            elif isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
                heat += float(mm.upper(m.q_mw) or 0.0)
            continue
        if isinstance(m, mm.HeatLoad):
            try:
                heat += float(mm.upper(m.q_mw_heat) or 0.0) - float(
                    mm.value(m.q_mw_heat) or 0.0
                ) * float(mm.value(m.regulation) or 1.0)
            except Exception:
                pass
        elif isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
            try:
                heat += float(mm.upper(m.q_mw) or 0.0) - float(
                    mm.value(m.q_mw) or 0.0
                ) * float(mm.value(m.regulation) or 1.0)
            except Exception:
                pass
    for b in net.branches:
        m = b.model
        if isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
            if not b.active or b.ignored:
                heat += float(mm.upper(m.q_mw) or 0.0)
            else:
                try:
                    heat += float(mm.upper(m.q_mw) or 0.0) - float(
                        mm.value(m.q_mw) or 0.0
                    ) * float(mm.value(m.regulation) or 1.0)
                except Exception:
                    pass

    gas = 0.0
    for c in net.childs:
        m = c.model
        if not isinstance(m, mm.Sink):
            continue
        grid = getattr(c, "grid", None)
        # Only gas-grid Sinks contribute to gas shedding. Water Sinks live on
        # WaterGrid which has no ``higher_heating_value`` — defaulting to a
        # gas HHV here would conjure huge fake gas energy from water mass
        # flows (mirrors model.py::_max_load_shedding).
        if grid is None or not hasattr(grid, "higher_heating_value"):
            continue
        hhv = float(grid.higher_heating_value)
        if c.ignored or not c.active:
            try:
                gas += float(mm.upper(m.mass_flow) or 0.0) * 3.6 * hhv
            except Exception:
                pass
            continue
        try:
            gas += (
                float(mm.upper(m.mass_flow) or 0.0)
                - float(mm.value(m.mass_flow) or 0.0)
                * float(mm.value(m.regulation) or 1.0)
            ) * 3.6 * hhv
        except Exception:
            pass

    total = power + heat + gas
    return float(power), float(heat), float(gas), float(total)


def compute_single_removal_shed(
    net_factory,
    ext_grid_bounds: dict,
    targets: Optional[List[Tuple[str, str, object]]] = None,
) -> pd.DataFrame:
    """Run deactivate-solve for each target on a *factory-fresh* network.

    Why a factory (callable) instead of a single Network object + ``.copy()``:
    monee's Pyomo solver mutates the network's Var state in place. A single
    successful solve replaces every per-component Var with its solved float
    value, after which ``optimization_problem._apply(network)`` crashes
    with ::

        AttributeError: 'float' object has no attribute 'max'

    because ``_apply`` expects each variable it sets bounds on to still be
    a Pyomo Var. ``Network.copy()`` doesn't reliably reconstitute every
    Var — under our pickled+formulated networks the copied Vars sometimes
    survive as floats. Calling ``net_factory()`` (typically
    ``pickle.loads(cached_bytes)``) returns a guaranteed-fresh formulated-
    but-unsolved network for every iteration.

    ``net_factory`` is also accepted as a Network instance for backwards
    compat — that path uses ``net.copy()`` and inherits the brittleness
    above.

    Returns a DataFrame with ``cp_id, kind, power_shed, heat_shed, gas_shed,
    total_shed, solve_status, elapsed_s``.
    """
    if not callable(net_factory):
        # Backwards-compat: a Network was passed; wrap it in a copy factory.
        _net_obj = net_factory
        net_factory = _net_obj.copy

    bounds = ext_grid_bounds

    # Use the first fresh net for target enumeration so component refs
    # remain valid across iterations (deactivate only reads type+id).
    targets = targets or _enumerate_targets(net_factory())

    # Baseline shed (no faults).
    t0 = time.time()
    base = _solve_load_shed(net_factory(), **bounds)
    base_p, base_h, base_g, base_t = (
        _shed_from_solved(base.network) if base is not None else (0.0, 0.0, 0.0, 0.0)
    )
    log.info(
        "baseline: total_shed=%.4f MW (p=%.4f, h=%.4f, g=%.4f) in %.1fs",
        base_t, base_p, base_h, base_g, time.time() - t0,
    )

    rows = [
        {
            "cp_id": "_baseline_",
            "kind": "baseline",
            "power_shed": base_p,
            "heat_shed": base_h,
            "gas_shed": base_g,
            "total_shed": base_t,
            "solve_status": "ok" if base is not None else "fail",
            "elapsed_s": float(time.time() - t0),
        }
    ]

    for cp_id, kind, comp in targets:
        t1 = time.time()
        # Fresh state per iteration. deactivate only reads type(comp) and
        # comp.id, both preserved across pickle round-trips; the
        # ``comp`` reference can be reused even though the new network
        # is a different Python object.
        net_iter = net_factory()
        try:
            net_iter.deactivate(comp)
        except Exception:
            log.exception("deactivate failed for %s", cp_id)
            rows.append({
                "cp_id": cp_id, "kind": kind,
                "power_shed": float("nan"), "heat_shed": float("nan"),
                "gas_shed": float("nan"), "total_shed": float("nan"),
                "solve_status": "deactivate_fail",
                "elapsed_s": float(time.time() - t1),
            })
            continue
        try:
            res = _solve_load_shed(net_iter, **bounds)
            if res is None:
                p, h, g, tot = float("nan"), float("nan"), float("nan"), float("nan")
                status = "solve_fail"
            else:
                p, h, g, tot = _shed_from_solved(res.network)
                status = "ok"
        except Exception:
            log.exception("shed solve failed for %s", cp_id)
            p, h, g, tot = float("nan"), float("nan"), float("nan"), float("nan")
            status = "solve_exception"
        rows.append({
            "cp_id": cp_id, "kind": kind,
            "power_shed": p, "heat_shed": h, "gas_shed": g, "total_shed": tot,
            "solve_status": status,
            "elapsed_s": float(time.time() - t1),
        })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _slice_targets(
    targets: List[Tuple[str, str, object]], shard: int, n_shards: int
):
    """Deterministic shard slicing. ``shard`` is 1-based."""
    if n_shards <= 1:
        return targets
    if not (1 <= shard <= n_shards):
        raise ValueError(f"shard {shard} out of [1, {n_shards}]")
    # Sort by cp_id for stability across runs.
    targets = sorted(targets, key=lambda t: t[0])
    n = len(targets)
    # Round-robin striping so wall-clock-imbalance from heterogeneous
    # solve times averages out across shards.
    return [t for i, t in enumerate(targets) if (i % n_shards) == (shard - 1)]


# NOTE: we intentionally do NOT pre-solve the network here. A successful
# Pyomo solve replaces the network's per-component Vars with their solved
# float values, after which the next ``optimization_problem._apply()``
# crashes with ``AttributeError: 'float' object has no attribute 'max'``.
# Each per-component iteration runs on a fresh ``net.copy()`` instead;
# see ``compute_single_removal_shed``.


def _resolve_ext_grid_bounds(grid_name: str) -> dict:
    """Return the ext-grid bound dict the resilience MC used for *grid_name*.

    Reads ``MESContainer.ext_grid_*_bounds`` from ``test_grids.ALL_GRIDS``
    so the analytical shed solve has the same external slack as the MC
    simulation it is being validated against. Builds the network once,
    which is moderately expensive, but only happens at startup.
    """
    from test_grids import ALL_GRIDS  # local import keeps CLI startup cheap

    if grid_name not in ALL_GRIDS:
        raise KeyError(
            f"Grid {grid_name!r} not in test_grids.ALL_GRIDS "
            f"({sorted(ALL_GRIDS)})."
        )
    create_fn, _ = ALL_GRIDS[grid_name]
    container = create_fn()
    return {
        "ext_grid_el_bounds": container.ext_grid_el_bounds,
        "ext_grid_gas_bounds": container.ext_grid_gas_bounds,
        "ext_grid_heat_bounds": container.ext_grid_heat_bounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "grid",
        help="Grid name as in run_simulation.py (e.g. simbench_lv).",
    )
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("/user/towo7024/cmres_new/cmres/data/res"),
        help="Directory containing MoneeResilienceExperiment-<grid>/network.p",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/out/single_removal_shed"),
        help="Where to write per-shard CSVs and the merged result.",
    )
    parser.add_argument("--shard", type=int, default=0,
                        help="1-based shard index (0 = unsharded).")
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true",
                        help="Merge all shards into <grid>.csv and exit.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = args.output_dir / f"single_removal_shed_{args.grid}.csv"

    if args.merge:
        parts: List[pd.DataFrame] = []
        for k in range(1, args.n_shards + 1):
            p = args.output_dir / f"single_removal_shed_{args.grid}_shard_{k}_of_{args.n_shards}.csv"
            if p.exists():
                parts.append(pd.read_csv(p))
        if not parts:
            print(f"no shard CSVs found in {args.output_dir}", file=sys.stderr)
            return 1
        merged = pd.concat(parts, ignore_index=True)
        # Drop duplicate baseline rows (each shard records one).
        baseline = merged[merged["cp_id"] == "_baseline_"].head(1)
        rest = merged[merged["cp_id"] != "_baseline_"]
        out = pd.concat([baseline, rest], ignore_index=True)
        out.to_csv(final_csv, index=False)
        print(f"wrote merged: {final_csv}  ({len(out)} rows)")
        return 0

    # Single-shard or unsharded run.
    network_pkl = args.input_dir / f"MoneeResilienceExperiment-{args.grid}" / "network.p"
    # Read the pickle bytes ONCE; deserialise per iteration to get a
    # guaranteed-fresh formulated network. ``Network.copy()`` does NOT
    # restore Pyomo Var state cleanly under our pickled+formulated setup —
    # the copied network's Vars come back as floats, and
    # ``optimization_problem._apply`` crashes on the first
    # ``var.max = max_value`` call. Re-pickling per iteration mirrors what
    # cp_cn_evaluation does (``pickle.load(network.p)`` → load-shed solve
    # works), so it sidesteps the broken-copy problem entirely.
    with open(network_pkl, "rb") as f:
        net_bytes = f.read()

    def net_factory():
        return pickle.loads(net_bytes)

    net = net_factory()
    log.info("loaded %s (%s)", network_pkl, net.statistics())

    targets = _enumerate_targets(net)
    log.info(
        "targets: %d total; shard %d/%d → %d this run",
        len(targets), args.shard, args.n_shards,
        len(_slice_targets(targets, args.shard, args.n_shards)) if args.n_shards > 1 else len(targets),
    )

    sliced = _slice_targets(targets, args.shard, args.n_shards) if args.n_shards > 1 else targets
    ext_bounds = _resolve_ext_grid_bounds(args.grid)
    log.info("ext-grid bounds for %s: %s", args.grid, ext_bounds)
    df = compute_single_removal_shed(net_factory, ext_bounds, targets=sliced)
    if args.shard and args.n_shards > 1:
        path = args.output_dir / f"single_removal_shed_{args.grid}_shard_{args.shard}_of_{args.n_shards}.csv"
    else:
        path = final_csv
    df.to_csv(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    sys.exit(main())
