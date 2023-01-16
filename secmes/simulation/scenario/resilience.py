from pathlib import Path

from secmes.agent.world import CentralFaultyWorld
from secmes.resilience.fault import FaultGenerator
from secmes.resilience.core import ResilienceMetric, ResilienceModel, RepairModel
from secmes.resilience.model import CascadingModel
from secmes.omef.eval import OMEFEvaluator

import secmes.data.observer as observer

from secmes.cn.network import (
    to_phys_bus_junc_networkx_graph,
)


def write_in_one_html(figures, name):
    Path(name).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{name}.html", "w") as file:
        file.write(figures[0].to_html(include_plotlyjs="cdn"))
        for fig in figures[1:]:
            file.write(fig.to_html(full_html=False, include_plotlyjs=False))


def flush_observed_data(experiment_name):
    import json

    Path(experiment_name).mkdir(parents=True, exist_ok=True)
    with open(experiment_name + "/readable_result.json", "w") as fp:
        json.dump(observer.data(), fp)


def start_resilience_simulation(
    multinet,
    resilience_model: ResilienceModel,
    repair_model: RepairModel,
    resilience_measurement_model: ResilienceMetric,
    time_steps=96,
    name="RES_SIM",
):
    networks = []

    def iteration_step(me_network, time):
        resilience_measurement_model.gather(me_network, time)
        networks.append(
            to_phys_bus_junc_networkx_graph(
                me_network, only_include_active_elements=True
            )
        )

    cascading_model = CascadingModel()

    def init_func(me_network):
        _, best = cascading_model.calc_performance(
            me_network, 0, OMEFEvaluator(without_load=True)
        )
        cascading_model.apply_setpoints(me_network, best, without_load=True)

    fault_gen = FaultGenerator(resilience_model, repair_model)
    sim = CentralFaultyWorld(
        iteration_step,
        init_func,
        multinet,
        max_steps=time_steps,
        name=name,
        fault_generator=fault_gen,
    )
    sim.add_post_step_hook(cascading_model.step)
    sim.add_post_step_hook(lambda _, __: flush_observed_data(name))
    sim.run()
    observer.clear()
