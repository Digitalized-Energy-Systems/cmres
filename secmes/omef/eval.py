from abc import ABC, abstractmethod
from typing import List
import math

import pandapipes.multinet.control as ppmc

from peext.node import (
    RegulatableMESModel,
    BusNode,
    JunctionNode,
    ExtGasGrid,
    ExtPowerGrid,
    SwitchNode,
    SinkNode,
    PowerLoadNode,
    HeatExchangerNode,
    SGeneratorNode,
    GeneratorNode,
    SourceNode,
)
from peext.edge import LineEdge
from peext.network import MENetwork

from secmes.mes.common import conversion_factor_kgps_to_mw


def is_broken(node):
    broken = False
    properties = node.properties_as_dict()
    if "bus" in properties:
        broken = broken or math.isnan(
            node.network.res_bus.at[properties["bus"], "vm_pu"]
        )
    elif "junction" in properties:
        broken = broken or math.isnan(
            node.network.res_junction.at[properties["junction"], "p_bar"]
        )
    elif "from_junction" in properties:
        broken = broken or math.isnan(
            node.network.res_junction.at[properties["from_junction"], "t_k"]
        )
    return broken or not properties["in_service"]


def is_productive(node, without_load=False):
    productive = True
    if without_load:
        productive = productive and not (
            type(node) == PowerLoadNode
            or type(node) == SinkNode
            or type(node) == HeatExchangerNode
            and node.qext_w() > 0
        )

    productive = productive and not is_broken(node)

    productive = productive and (
        isinstance(node, RegulatableMESModel)
        and not (
            isinstance(node, BusNode)
            or isinstance(node, JunctionNode)
            or isinstance(node, ExtPowerGrid)
            or isinstance(node, ExtGasGrid)
            or isinstance(node, SwitchNode)
        )
    )
    return productive


class Evaluator(ABC):
    @abstractmethod
    def evaluate(solution, me_network):
        pass


def step_net(me_network, step):
    for _, net in me_network.multinet["nets"].items():
        for _, row in net.controller.iterrows():
            controller = row.object
            controller.time_step(net, step)

    try:
        ppmc.run_control_multinet.run_control(
            me_network.multinet, max_iter=30, mode="all"
        )
    except:
        pass


def to_penalty(rel_diff):
    return (rel_diff * 100) ** 1.5


def simple_node_to_cost(node):
    if (
        type(node) == SinkNode
        or type(node) == PowerLoadNode
        or type(node) == HeatExchangerNode
        and node.qext_w() > 0
    ):
        return -100
    else:
        return 25


def calc_relative_violation(node, attribute, target, rel_allowed_diff):
    return max(
        (abs((node.values_as_dict()[attribute] - target)) - rel_allowed_diff * target)
        / (target * rel_allowed_diff),
        0,
    )


class OMEFEvaluator:
    def __init__(
        self,
        regulatable_node_to_cost=simple_node_to_cost,
        highest_cost=1,
        target_voltage=1,
        voltage_rel_allowed_diff=0.1,
        target_pressure=60,
        pressure_rel_allowed_diff=0.3,
        target_temp=360,
        temp_rel_allowed_diff=0.1,
        acceptable_balance_diff=10,
        without_load=False,
    ) -> None:
        self._node_to_cost = regulatable_node_to_cost
        self._target_voltage = target_voltage
        self._voltage_rel_allwed_diff = voltage_rel_allowed_diff
        self._target_pressure = target_pressure
        self._pressure_rel_allowed_diff = pressure_rel_allowed_diff
        self._target_temp = target_temp
        self._temp_rel_allowed_diff = temp_rel_allowed_diff
        self._max_node_cost = highest_cost
        self._acceptable_balance_diff = acceptable_balance_diff
        self._without_load = without_load

    def _fetch_node_energy(self, node):
        if isinstance(node, PowerLoadNode):
            return node.active_power_capability()
        if isinstance(node, SinkNode):
            return node.mdot_kg_per_s_capability() * conversion_factor_kgps_to_mw(
                node.network
            )
        if isinstance(node, HeatExchangerNode):
            return node.q_capability() * 1e-6
        if isinstance(node, SourceNode):
            return node.mdot_kg_per_s_capability() * conversion_factor_kgps_to_mw(
                node.network
            )
        if isinstance(node, SGeneratorNode):
            return node.active_power_capability()
        if isinstance(node, GeneratorNode):
            return node.active_power_capability()
        return 0

    def _calc_cost(self, regulatable_nodes):
        cost = 0
        for node in regulatable_nodes:
            cost += (
                self._fetch_node_energy(node)
                * self._node_to_cost(node)
                * node.regulation_factor()
            )
        return cost

    def _calc_penalty(self, me_network: MENetwork):
        penalty = 0

        # general physical network constraints
        for network_name, nodes in me_network.virtual_nodes_as_dict.items():
            for node in nodes:
                if not node.properties_as_dict()["in_service"]:
                    continue
                if network_name == "gas":
                    rel_diff_pressure = calc_relative_violation(
                        node,
                        "p_bar",
                        target=self._target_pressure,
                        rel_allowed_diff=self._pressure_rel_allowed_diff,
                    )
                    penalty += to_penalty(rel_diff_pressure)

                if network_name == "heat":
                    if node.id == 1:
                        continue
                    rel_diff_heat = calc_relative_violation(
                        node,
                        "t_k",
                        target=self._target_temp,
                        rel_allowed_diff=self._temp_rel_allowed_diff,
                    )
                    penalty += to_penalty(rel_diff_heat)

                if network_name == "power":
                    rel_diff_voltage = calc_relative_violation(
                        node,
                        "vm_pu",
                        target=self._target_voltage,
                        rel_allowed_diff=self._voltage_rel_allwed_diff,
                    )
                    penalty += to_penalty(rel_diff_voltage)

        # line load constraint
        for edge in me_network.edges_as_dict["power"]:
            if not edge.properties_as_dict()["in_service"]:
                continue
            if isinstance(edge, LineEdge):
                problem_diff = max(edge.loading_percent() - 100, 0) / 100
                penalty += to_penalty(problem_diff)

        slack_gas = [
            node for node in me_network.nodes_as_dict["gas"] if type(node) == ExtGasGrid
        ][0]
        slack_power = [
            node
            for node in me_network.nodes_as_dict["power"]
            if type(node) == ExtPowerGrid
        ][0]

        penalty += max(0, abs(slack_power.values_as_dict()["p_mw"]) - 0.01) * 1000
        penalty += (
            max(
                0,
                abs(
                    slack_gas.values_as_dict()["mdot_kg_per_s"]
                    * conversion_factor_kgps_to_mw(slack_gas.network)
                )
                - 0.05,
            )
            * 1000
        )

        return -penalty

    def _calc_penalty_coeff(self):
        return self._max_node_cost

    def evaluate(
        self,
        solution,
        me_network: MENetwork,
        nodes_to_regulate: List[RegulatableMESModel],
        step: int,
    ):

        for i, node in enumerate(nodes_to_regulate):
            node.regulate(solution[i])

        step_net(me_network, step)

        # calculate some objective
        return -self._calc_cost(
            nodes_to_regulate
        ) + self._calc_penalty_coeff() * self._calc_penalty(me_network)
