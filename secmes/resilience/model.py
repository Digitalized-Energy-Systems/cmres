from secmes.resilience.core import (
    ResilienceModel,
    RepairModel,
    Failure,
    Effect,
    StepModel,
)
from secmes.resilience.metric import (
    GeneralResiliencePerformanceMetric,
    CascadingResilienceMetric,
)
from secmes.omef.solver.eval import OMEFEvaluator, calc_relative_violation, is_productive
from secmes.omef.solver.ea import EASolver
from secmes.cn.network import name_of
from typing import List
import random

from peext.node import *
from peext.edge import *
from peext.network import MENetwork

import numpy as np
import scipy.stats as stats

import secmes.data.observer as observer

FAIL_BASE_PROBABILITY_MAP = {
    SourceNode: 0.2,
    SinkNode: 0.05,
    P2GNode: 0.2,
    P2HNode: 0.2,
    CHPNode: 0.2,
    G2PNode: 0.2,
    SGeneratorNode: 0.2,
    GeneratorNode: 0.2,
    HeatExchangerNode: 0.2,
    LineEdge: 0.1,
    TrafoEdge: 0.00,
    PipeEdge: 0.1,
    PumpEdge: 0.03,
    EmptyBusNode: 0,
    EmptyJunctionNode: 0,
    BusNode: 0.01,
    JunctionNode: 0.01,
}

FAILURE_PROBABILITY_MODEL = lambda base_prob: base_prob * np.random.normal(1, scale=0.1)

FAILURE_TIME_MODEL = lambda incident_time_steps, time: stats.norm.pdf(
    time, loc=incident_time_steps / 2
)
FAILURE_SPATIAL_MODEL = lambda coords: 1


class SimpleResilienceModel(ResilienceModel):
    def __init__(
        self,
        incident_shift=5,
        incident_timesteps=10,
        heat_impact=0.1,
        gas_impact=0.1,
        power_impact=5,
        mes_impact=1,
        base_fail_probability_map=FAIL_BASE_PROBABILITY_MAP,
        fail_probability_model=FAILURE_PROBABILITY_MODEL,
        time_model=FAILURE_TIME_MODEL,
        spatial_model=FAILURE_SPATIAL_MODEL,
    ) -> None:
        self._incident_shift = incident_shift
        self._incident_timesteps = incident_timesteps
        self._base_fail_probability_map = base_fail_probability_map
        self._heat_impact = heat_impact
        self._gas_impact = gas_impact
        self._power_impact = power_impact
        self._mes_impact = mes_impact
        self._fail_probability_model = fail_probability_model
        self._time_model = lambda time: time_model(
            self._incident_timesteps + incident_shift, time
        )
        self._spatial_model = (
            lambda coords: spatial_model(coords) if coords is not None else 1
        )

    def _read_impact(self, model):
        if model.network.name == "heat":
            return self._heat_impact
        elif model.network.name == "power":
            return self._power_impact
        elif model.network.name == "gas":
            return self._gas_impact
        return self._mes_impact

    def calc_fail(self, model, time):
        model_type = type(model)
        if model_type not in self._base_fail_probability_map:
            return 0
        base_failure_probability = self._base_fail_probability_map[model_type]
        coords = model.coords()
        return (
            self._fail_probability_model(base_failure_probability)
            * self._read_impact(model)
            * self._time_model(time)
            * self._spatial_model(coords)
        )

    def has_failed(self, model, time):
        return random.random() < self.calc_fail(model, time)

    def generate_failures(self, me_network):
        failures = []
        for i in range(self._incident_timesteps):
            time = i + self._incident_shift
            for node in me_network.nodes:
                fail_prob = self.calc_fail(node, time)
                if random.random() < fail_prob:
                    failures.append(Failure(time, node, fail_prob, Effect.DEAD, -1))
            for virtual_node in me_network.virtual_nodes:
                fail_prob = self.calc_fail(virtual_node, time)
                if random.random() < fail_prob:
                    failures.append(
                        Failure(time, virtual_node, fail_prob, Effect.DEAD, -1)
                    )
            for edge in me_network.edges:
                fail_prob = self.calc_fail(edge, time)
                if random.random() < fail_prob:
                    failures.append(Failure(time, edge, fail_prob, Effect.DEAD, -1))
        return failures


DMG_COEFF_FUNC_MAP = {
    SourceNode: lambda source: source.mdot_kg_per_s_capability(),
    SinkNode: lambda sink: sink.mdot_kg_per_s_capability(),
    P2GNode: lambda p2g: p2g.active_power_capability(),
    P2HNode: lambda p2h: p2h.active_power_capability(),
    CHPNode: lambda chp: chp.mdot_kg_per_s_capability(),
    G2PNode: lambda g2p: g2p.mdot_kg_per_s_capability(),
    SGeneratorNode: lambda sgen: sgen.active_power_capability(),
    GeneratorNode: lambda sgen: sgen.active_power_capability(),
    HeatExchangerNode: lambda he: he.q_capability(),
    LineEdge: lambda line: line.properties_as_dict()["length_km"],
    TrafoEdge: lambda _: 1,
    PipeEdge: lambda pipe: pipe.properties_as_dict()["length_km"],
    PumpEdge: lambda _: 1,
    BusNode: lambda _: 1,
    JunctionNode: lambda _: 1,
}
DMG_COEFF_VARIANCE_MODEL = lambda dmg_coeff: dmg_coeff * np.random.normal(1, scale=0.1)


class SimpleRepairModel(RepairModel):
    def __init__(
        self,
        delay_for_repair=10,
        dmg_coeff_func_map=DMG_COEFF_FUNC_MAP,
        dmg_coeff_variance_model=DMG_COEFF_VARIANCE_MODEL,
        incident_timesteps=10,
    ) -> None:
        self._dmg_coeff_func_map = dmg_coeff_func_map
        self._dmg_coeff_variance_model = dmg_coeff_variance_model
        self._delay_for_repair = delay_for_repair
        self._incident_timesteps = incident_timesteps

    def generate_repairs(self, _, failures: List[Failure]):
        for failure in failures:
            f: Failure = failure
            dmg = (
                self._dmg_coeff_variance_model(
                    self._dmg_coeff_func_map[type(f.node)](f.node)
                )
                * f.severity
            )
            time_needed = max(
                np.random.normal(5, 5) + self._delay_for_repair + dmg / 10,
                self._incident_timesteps,
            )
            f.repaired_time = int(f.time + time_needed)
        return failures


def to_failure_probability(relative_violation, ramp=0, steepness=1, exponent=1.5):
    return steepness * ((relative_violation - ramp) * 100) ** exponent / 100


def deactivate_node(node):
    node.network[node.component_type()].at[node.id, "in_service"] = False


def activate_node(node):
    node.network[node.component_type()].at[node.id, "in_service"] = True


class CascadingModel(StepModel):
    def __init__(self, performance_accuracy=100) -> None:
        self._cascading_metric = CascadingResilienceMetric()
        self._performance_metric = GeneralResiliencePerformanceMetric()
        self._omef = OMEFEvaluator()
        self._current_failures = []
        self._iteration_number_omef = performance_accuracy

    def calc_performance(self, me_network, step, omef: OMEFEvaluator):
        solver = EASolver(
            omef,
            iteration_number=self._iteration_number_omef,
            population_size=16,
            generation_size=8,
            parent_number=2,
        )
        best, _, _ = solver.solve(me_network, step, without_load=omef._without_load)
        return (
            self._performance_metric.calc_using_setpoints(
                me_network, best, without_load=omef._without_load
            ),
            best,
        )

    def apply_setpoints(self, me_network, new_setpoints, without_load=False):
        all_regulatable_nodes = [
            node
            for node in me_network.nodes
            if is_productive(node, without_load=without_load)
        ]
        for i, node in enumerate(all_regulatable_nodes):
            node.regulate(new_setpoints[i])

    def process_nodes(self, nodes, attribute, target, allowed_diff, step, network_name):
        for i in range(len(self._current_failures) - 1, -1, -1):
            node, failure = self._current_failures[i]
            if (
                node.network.name != network_name
                or attribute not in node.values_as_dict()
            ):
                continue
            if failure["step"] + failure["min_duration"] >= step:
                relative_violation = to_failure_probability(
                    calc_relative_violation(node, attribute, target, allowed_diff)
                )
                if relative_violation < np.random.random():
                    activate_node(node)
                    del self._current_failures[i]
                    observer.gather(
                        "cascading repair",
                        {
                            "step": step,
                            "node": name_of(node),
                            "probability": relative_violation,
                            "min_duration": -1,
                            "type": "repair",
                        },
                    )

        for node in nodes:
            relative_violation = to_failure_probability(
                calc_relative_violation(node, attribute, target, allowed_diff)
            )
            if relative_violation > np.random.random():
                min_duration = int(relative_violation * np.random.random() * 10)
                deactivate_node(node)
                failure_description = {
                    "step": step,
                    "node": name_of(node),
                    "probability": relative_violation,
                    "min_duration": min_duration,
                    "type": "failure",
                }
                self._current_failures.append((node, failure_description))

                observer.gather(
                    "cascading failure",
                    failure_description,
                )

    def process_network_state(self, me_network: MENetwork, step):
        for network, nodes in me_network.virtual_nodes_as_dict.items():
            if network == "gas":
                self.process_nodes(nodes, "p_bar", 60, 0.3, step, network)
            if network == "heat":
                self.process_nodes(nodes, "t_k", 360, 0.2, step, network)
            if network == "power":
                self.process_nodes(nodes, "vm_pu", 1, 0.2, step, network)

        for network, edges in me_network.edges_as_dict.items():
            if network == "power":
                self.process_nodes(
                    edges, "loading_percent", 50, 1, step=step, network_name=network
                )

    def step(self, me_network, step):
        performance, _ = self.calc_performance(me_network, step, self._omef)

        observer.gather("performance", performance)

        self.process_network_state(me_network, step)

        # calc performance and base performance
        performance_after_cascade, new_setpoints = self.calc_performance(
            me_network, step, self._omef
        )
        observer.gather("performance_after_cascade", performance_after_cascade)

        # apply new setpoints
        self.apply_setpoints(me_network, new_setpoints)

        self._cascading_metric.gather(
            me_network,
            step,
            performance=performance,
            performance_after_cascade=performance_after_cascade,
        )

    def calc_metric(self):
        return self._cascading_metric.calc()
