from secmes.resilience.core import ResilienceMetric, PerformanceMetric

from peext.network import *
from secmes.mes.common import conversion_factor_kgps_to_mw
from secmes.omef.solver.eval import is_productive, is_broken


class rlist(list):
    def __init__(self, default):
        self._default = default

    def __setitem__(self, key, value):
        if key >= len(self):
            self += [self._default] * (key - len(self) + 1)
        super(rlist, self).__setitem__(key, value)


def is_load_node(node):
    return (
        isinstance(node, PowerLoadNode)
        or isinstance(node, SinkNode)
        or isinstance(node, HeatExchangerNode)
        and node.qext_w() > 0
    )


class GeneralResiliencePerformanceMetric(PerformanceMetric):
    def get_relevant_nodes(self, me_network, without_load=False):
        return [
            node
            for node in me_network.nodes
            if is_productive(node, without_load=without_load)
        ]

    def get_broken_nodes(self, me_network):
        return [
            node for node in me_network.nodes if is_broken(node) and is_load_node(node)
        ]

    def calc_using_setpoints(self, me_network, set_points, without_load=False):
        load_nodes = self.get_relevant_nodes(me_network, without_load=without_load)
        broken_nodes = self.get_broken_nodes(me_network)
        power_load_curtailed = 0
        heat_load_curtailed = 0
        gas_load_curtailed = 0

        assert len(set_points) == len(load_nodes)

        for i, node in enumerate(load_nodes + broken_nodes):

            # if broken (out of service or not connectable),
            # set_point of 0 is assumed
            if i >= len(load_nodes):
                set_point = 0
            else:
                set_point = set_points[i]

            curtailment_rel = 1 - set_point
            if isinstance(node, PowerLoadNode):
                power_load_curtailed += node.active_power_capability() * curtailment_rel
            if isinstance(node, SinkNode):
                gas_load_curtailed += (
                    node.mdot_kg_per_s_capability()
                    * conversion_factor_kgps_to_mw(node.network)[0]
                    * curtailment_rel
                )
            if isinstance(node, HeatExchangerNode) and node.q_capability() >= 0:
                heat_load_curtailed += node.q_capability() * 1e-6 * curtailment_rel

        return (power_load_curtailed, heat_load_curtailed, gas_load_curtailed)

    def calc(self, me_network: MENetwork):
        current_set_points = [
            load_node.regulation_factor()
            for load_node in self.get_load_nodes(me_network)
        ]
        return self.calc_using_setpoints(me_network, current_set_points)


class CascadingResilienceMetric(ResilienceMetric):
    def __init__(self) -> None:
        self._performances = rlist(0)
        self._performances_after_cascade = rlist(0)

    def gather(self, _, step, **kwargs):
        self._performances[step] = kwargs["performance"]
        self._performances_after_cascade = kwargs["performance_after_cascade"]

    def calc(self):
        pass


class SimpleResilienceMetric(ResilienceMetric):
    def __init__(self) -> None:
        self.gas_balance_measurements = []
        self.power_balance_measurements = []

    def gather(self, me_network, time):
        nodes_as_dict = me_network.nodes_as_dict
        gas_nodes = nodes_as_dict["gas"]
        for gas_node in gas_nodes:
            if type(gas_node) == ExtGasGrid:
                self.gas_balance_measurements.append(
                    gas_node.values_as_dict()["mdot_kg_per_s"]
                )
                break

        power_nodes = nodes_as_dict["power"]
        for power_node in power_nodes:
            if type(power_node) == ExtPowerGrid:
                self.power_balance_measurements.append(
                    power_node.values_as_dict()["p_mw"]
                )
                break

    def calc(self):
        return ("power_balance", self.power_balance_measurements), (
            "gas balance",
            self.gas_balance_measurements,
        )
