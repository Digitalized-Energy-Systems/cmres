from abc import ABC, abstractmethod

from peext.node import RegulatableMESModel, SinkNode, PowerLoadNode, HeatExchangerNode
from peext.network import MENetwork

import pandapipes.multinet.control as ppmc


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
        return -1
    else:
        return 1


class OMEFEvaluator:
    def __init__(
        self,
        regulatable_node_to_cost=simple_node_to_cost,
        highest_cost=1,
        target_voltage=10,
        voltage_rel_allowed_diff=0.1,
        target_pressure=5,
        pressure_rel_allowed_diff=0.1,
        target_temp=380,
        temp_rel_allowed_diff=0.2,
    ) -> None:
        self._node_to_cost = regulatable_node_to_cost
        self._target_voltage = target_voltage
        self._voltage_rel_allwed_diff = voltage_rel_allowed_diff
        self._target_pressure = target_pressure
        self._pressure_rel_allowed_diff = pressure_rel_allowed_diff
        self._target_temp = target_temp
        self._temp_rel_allowed_diff = temp_rel_allowed_diff
        self._max_node_cost = highest_cost

    def _calc_cost(self, regulatable_nodes):
        cost = 0
        for node in regulatable_nodes:
            cost += self._node_to_cost(node) * node.regulation_factor()
        return cost

    def _calc_penalty(self, me_network: MENetwork):
        penalty = 0

        # general physical network constraints
        for network_name, nodes in me_network.virtual_nodes_as_dict.items():
            for node in nodes:
                if not node.properties_as_dict()["in_service"]:
                    continue
                if network_name == "gas":
                    rel_diff_pressure = max(
                        abs(node.values_as_dict()["p_bar"] - self._target_pressure)
                        / self._target_pressure
                        - self._pressure_rel_allowed_diff,
                        0,
                    )
                    penalty += to_penalty(rel_diff_pressure)

                if network_name == "heat":
                    rel_diff_heat = max(
                        abs(node.values_as_dict()["t_k"] - self._target_temp)
                        / self._target_temp
                        - self._temp_rel_allowed_diff,
                        0,
                    )
                    penalty += to_penalty(rel_diff_heat)

                if network_name == "power":
                    rel_diff_voltage = max(
                        abs(node.values_as_dict()["vm_pu"] - self._target_voltage)
                        / self._target_voltage
                        - self._voltage_rel_allwed_diff,
                        0,
                    )
                    penalty += to_penalty(rel_diff_voltage)

        # line load constraint
        for edge in me_network.edges_as_dict["power"]:
            if not edge.properties_as_dict()["in_service"]:
                continue
            problem_diff = max(edge.loading_percent() - 100, 0) / 100
            penalty += to_penalty(problem_diff)

        return -penalty

    def _calc_penalty_coeff(self):
        return self._max_node_cost

    def evaluate(self, solution, me_network: MENetwork, step: int):

        all_regulatable_nodes = [
            node for node in me_network.nodes if isinstance(node, RegulatableMESModel)
        ]

        for i, node in enumerate(all_regulatable_nodes):
            node.regulate(solution[i])

        step_net(me_network, step)

        # calculate some objective
        return -self._calc_cost(
            all_regulatable_nodes
        ) + self._calc_penalty_coeff() * self._calc_penalty(me_network)
