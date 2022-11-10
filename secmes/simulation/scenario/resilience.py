from pathlib import Path

from secmes.agent.world import CentralFaultyWorld
from secmes.simulation.fault import FaultGenerator
from secmes.resilience.core import ResilienceMetric, ResilienceModel, RepairModel
import plotly.graph_objects as go

import networkx.drawing.nx_agraph as nxd

from secmes.cn.network import (
    to_phys_bus_junc_networkx_graph,
    create_networkx_topology_plot,
)


def write_in_one_html(figures, name):
    Path(name).parent.mkdir(parents=True, exist_ok=True)

    with open(f"{name}.html", "w") as file:
        file.write(figures[0].to_html(include_plotlyjs="cdn"))
        for fig in figures[1:]:
            file.write(fig.to_html(full_html=False, include_plotlyjs=False))


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

    fault_gen = FaultGenerator(resilience_model, repair_model)
    sim = CentralFaultyWorld(
        iteration_step,
        lambda x: x,
        multinet,
        max_steps=time_steps,
        name=name,
        fault_generator=fault_gen,
    )
    sim.run()

    print(fault_gen.failures)
    # write
    figs = []
    for label, data in resilience_measurement_model.calc():
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=list(range(len(data))), y=data, mode="lines", name="lines")
        )
        fig.update_layout(
            title=f"Resilience Metric {type(resilience_measurement_model)}",
            xaxis_title="time",
            yaxis_title=f"resilience ({label})",
        )
        figs.append(fig)
    write_in_one_html(figs, name + "/Resilience")

    graph_figs = []
    pos = nxd.pygraphviz_layout(networks[0], prog="neato")
    for network in networks:
        graph_figs.append(create_networkx_topology_plot(network, pos=pos))
    write_in_one_html(graph_figs, name + "/Graph")
