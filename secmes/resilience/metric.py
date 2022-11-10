from secmes.resilience.core import ResilienceMetric

from peext.network import *


class SimpleResilienceMetric(ResilienceMetric):
    def __init__(self) -> None:
        self.gas_balance_measurements = []
        self.power_balance_measurements = []

    def gather(self, me_network, _):
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
