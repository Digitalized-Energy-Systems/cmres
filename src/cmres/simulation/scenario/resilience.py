from pathlib import Path
import logging
import os
import fcntl

import numpy as np

from cmres.resilience.world import CentralFaultyMoneeWorld
from cmres.resilience.fault import FaultGenerator
from cmres.resilience.core import ResilienceMetric, ResilienceModel
from cmres.resilience.model import CascadingModel

import cmres.data.observer as observer
from cmres.data.observer import Observer

from monee import Network, TimeseriesData
import pandas

log = logging.getLogger(__name__)


def _dump_infeasibility_incident(out_name, run_id, name, scenario, faults, incidents):
    """Write a self-contained reproduction packet for a run that hit one or
    more infeasible solves.

    Layout::

        <out_name>/
            incidents.log                  ← one append-only summary line per run
            incidents/run_<id>/
                scenario.npz               ← uniforms + log_weight (replay input)
                faults.txt                  ← all generated faults (sanity check)
                REPORT.txt                  ← per-incident diagnostics + report
                incidents.json              ← machine-readable list

    Replay
    ------
    Re-build the same grid and seed, load ``scenario.npz`` into a
    :class:`FailureScenario`, and pass it through
    :func:`start_resilience_simulation`.  Faults are regenerated
    deterministically from the scenario + registry, so they need not be
    serialised for replay — ``faults.txt`` is for human eyes only.
    """
    import json

    out_root = Path(out_name)
    inc_dir = out_root / "incidents" / f"run_{run_id}"
    inc_dir.mkdir(parents=True, exist_ok=True)

    if scenario is not None:
        np.savez(
            inc_dir / "scenario.npz",
            uniforms=scenario.uniforms,
            log_weight=np.array(scenario.log_weight),
        )

    (inc_dir / "faults.txt").write_text(
        "\n".join(str(f) for f in (faults or [])) or "<no faults>"
    )

    lines = [
        f"Run id        : {run_id}",
        f"Run name      : {name}",
        f"Faults        : {len(faults or [])}",
        f"Incidents     : {len(incidents)}",
        f"Scenario file : {(inc_dir / 'scenario.npz') if scenario is not None else '<no scenario — legacy RNG mode>'}",
        "",
    ]
    for i, inc in enumerate(incidents, 1):
        lines.append(f"--- Incident {i} ---")
        lines.append(f"  step           : {inc.get('step')}")
        lines.append(f"  kind           : {inc.get('kind')}")
        lines.append(f"  n_active_faults: {inc.get('n_active_faults')}")
        report = inc.get("report") or ""
        if report:
            lines.append("  solver report  :")
            for rep_line in str(report).splitlines():
                lines.append(f"      {rep_line}")
        lines.append("")
    (inc_dir / "REPORT.txt").write_text("\n".join(lines))

    with (inc_dir / "incidents.json").open("w") as fp:
        json.dump(
            [
                {
                    "step": inc.get("step"),
                    "kind": inc.get("kind"),
                    "n_active_faults": inc.get("n_active_faults"),
                    "active_faults": inc.get("active_faults"),
                    "report": inc.get("report"),
                }
                for inc in incidents
            ],
            fp,
            indent=2,
        )

    summary_line = (
        f"run_id={run_id}\tname={name}\tn_incidents={len(incidents)}"
        f"\tsteps={[inc.get('step') for inc in incidents]}"
        f"\tdir={inc_dir}\n"
    )
    summary_lock = out_root / ".incidents.log.lock"
    out_root.mkdir(parents=True, exist_ok=True)
    with open(summary_lock, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (out_root / "incidents.log").open("a") as fp:
            fp.write(summary_line)
        fcntl.flock(lock, fcntl.LOCK_UN)

    log.warning(
        "Recorded %d infeasibility incident(s) for run id=%s -> %s",
        len(incidents), run_id, inc_dir,
    )


def write_in_one_html(figures, name):
    Path(name).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{name}.html", "w") as file:
        file.write(figures[0].to_html(include_plotlyjs="cdn"))
        for fig in figures[1:]:
            file.write(fig.to_html(full_html=False, include_plotlyjs=False))


def flush_observed_data(experiment_name, id, obs=None):
    """Flush gathered events to per-key CSVs.

    ``obs`` selects which Observer to read. Defaults to the current
    thread-local observer (legacy global behavior); explicit-pass an
    Observer instance to flush a specific run's state regardless of
    which context happens to be active.
    """
    source = obs if obs is not None else observer
    for key, value_list in source.data().items():
        out_path = Path(experiment_name)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / Path(f"{key}.csv")
        dataframe = []
        for value in value_list:
            if type(value) is dict:
                dataframe.append({**value, **{"id": id}})
            if isinstance(value, (list, tuple)):
                dataframe.append(
                    {**{str(i): v for i, v in enumerate(value)}, **{"id": id}}
                )

        out_file_lock = out_path / Path(f".{key}.lock")
        with open(out_file_lock, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            pandas.DataFrame(dataframe).to_csv(
                out_file, mode="a", header=not os.path.exists(out_file)
            )
            fcntl.flock(lock, fcntl.LOCK_UN)


def start_resilience_simulation(
    net: Network,
    timeseries_data: TimeseriesData,
    resilience_model: ResilienceModel,
    resilience_measurement_model: ResilienceMetric,
    time_steps=96,
    name="RES_SIM",
    out_name="RES_SIM",
    id=0,
    registry=None,
    scenario=None,
    ext_grid_el_bounds=None,
    ext_grid_gas_bounds=None,
    ext_grid_heat_bounds=None,
    include_coupling_points=False,
):
    def iteration_step(net, step, step_state, step_result, base_net):
        resilience_measurement_model.gather(net, step)

    cm_kwargs = {"include_coupling_points": include_coupling_points}
    if ext_grid_el_bounds is not None:
        cm_kwargs["ext_grid_el_bounds"] = ext_grid_el_bounds
    if ext_grid_gas_bounds is not None:
        cm_kwargs["ext_grid_gas_bounds"] = ext_grid_gas_bounds
    if ext_grid_heat_bounds is not None:
        cm_kwargs["ext_grid_heat_bounds"] = ext_grid_heat_bounds
    cascading_model = CascadingModel(**cm_kwargs)

    def init_func(net):
        _, __ = cascading_model.calc_performance(net, 0)

    fault_gen = FaultGenerator(
        resilience_model, registry=registry, scenario=scenario
    )
    sim = CentralFaultyMoneeWorld(
        iteration_step,
        init_func,
        net,
        timeseries_data,
        max_steps=time_steps,
        name=name,
        fault_generator=fault_gen,
    )
    sim.add_step_hook(cascading_model.step)
    # Per-run observer context: any ``observer.gather`` call inside the
    # ``with`` block targets THIS Observer only — never the shared default
    # or another concurrent run's state. Replaces the prior global-dict +
    # ``observer.clear()``-in-finally pattern.
    with Observer() as obs:
        try:
            try:
                sim.prepare()
                cascading_model._faults = sim.faults
                log.debug("Starting simulation  faults=%s", [str(f) for f in sim.faults])
                sim.run()
            finally:
                # Capture incidents BEFORE flush_observed_data, since that
                # path is CSV-friendly but discards rich per-incident
                # structure (active fault list, solver report). flush still
                # runs afterwards so the CSV ``infeasibility.csv`` artefact
                # is preserved alongside the dump.
                incidents = list(obs.data().get("infeasibility", []))
                if incidents:
                    try:
                        _dump_infeasibility_incident(
                            out_name=out_name,
                            run_id=id,
                            name=name,
                            scenario=scenario,
                            faults=getattr(sim, "faults", None),
                            incidents=incidents,
                        )
                    except Exception:
                        log.exception(
                            "Failed to dump infeasibility incident packet for run id=%s",
                            id,
                        )
                log.debug("Flushing observer data  id=%s", id)
                flush_observed_data(out_name, id, obs=obs)

            # Per-carrier performance sums: index 0=power, 1=heat, 2=gas.
            # Each entry in obs.data()["performance"] is a 3-tuple (or scalar).
            # ``skipna=False`` so a single NaN step (e.g. CascadingModel.step
            # gathered NaN performance after a solver time-limit abort)
            # propagates into the carrier sum — the MC accumulator then
            # drops the whole run via its non-finite skip path, which is
            # exactly what we want for non-converged samples.
            raw_perfs = obs.data().get("performance", [])
            per_carrier = pandas.DataFrame(
                [t if isinstance(t, (list, tuple)) else [t, 0.0, 0.0] for t in raw_perfs],
                columns=["power", "heat", "gas"],
            )
            carrier_sums = per_carrier.sum(skipna=False).to_numpy()
            performance_sum = float(carrier_sums.sum())  # scalar total
            return performance_sum, carrier_sums
        finally:
            # Observer state goes out of scope with the ``with`` block —
            # no global clear required. The legacy ``observer`` module
            # still gets cleared defensively in case any hook captured the
            # free function from outside the context.
            observer.clear()
