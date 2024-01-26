from pathlib import Path
import os
import fcntl

from secmes.agent.world import CentralFaultyMoneeWorld
from secmes.resilience.fault import FaultGenerator
from secmes.resilience.core import ResilienceMetric, ResilienceModel, RepairModel
from secmes.resilience.model import CascadingModel

import secmes.data.observer as observer

from monee import Network, TimeseriesData
import pandas


def write_in_one_html(figures, name):
    Path(name).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{name}.html", "w") as file:
        file.write(figures[0].to_html(include_plotlyjs="cdn"))
        for fig in figures[1:]:
            file.write(fig.to_html(full_html=False, include_plotlyjs=False))


def flush_observed_data(experiment_name, id):
    for key, value_list in observer.data().items():
        out_path = Path(experiment_name)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / Path(f"{key}.csv")
        dataframe = []
        for value in value_list:
            if type(value) == dict:
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
    repair_model: RepairModel,
    resilience_measurement_model: ResilienceMetric,
    time_steps=96,
    name="RES_SIM",
    out_name="RES_SIM",
    id=0,
):
    networks = []

    def iteration_step(net, _, time):
        resilience_measurement_model.gather(net, time)
        networks.append(net)

    cascading_model = CascadingModel()

    def init_func(net):
        _, __ = cascading_model.calc_performance(net, 0)

    fault_gen = FaultGenerator(resilience_model, repair_model)
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
    try:
        sim.prepare()
        cascading_model._faults = sim.faults
        sim.run()
    finally:
        flush_observed_data(out_name, id)
    performance_sum = sum([sum(t) for t in observer.data()["performance"]])
    observer.clear()
    return performance_sum
