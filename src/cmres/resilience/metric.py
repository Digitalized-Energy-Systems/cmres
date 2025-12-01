from cmres.resilience.core import ResilienceMetric, PerformanceMetric

from monee import Network
import monee.model as md
import monee.problem as mp

class rlist(list):
    def __init__(self, default):
        self._default = default

    def __setitem__(self, key, value):
        if key >= len(self):
            self += [self._default] * (key - len(self) + 1)
        super(rlist, self).__setitem__(key, value)


def is_load(component):
    model = component.model
    grid = component.grid
    return (
        isinstance(model, md.PowerLoad)
        or isinstance(model, md.Sink)
        and isinstance(grid, md.GasGrid)
        or isinstance(model, md.HeatExchangerLoad)
    )


class GeneralResiliencePerformanceMetric(PerformanceMetric):
    def get_relevant_components(self, network: Network):
        return [
            component
            for component in network.childs + network.branches
            if is_load(component)
        ]

    def calc(self, network):
        return mp.calc_general_resilience_performance(network)

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

    def gather(self, network: Network, time):
        ext_hydr_grids = network.childs_by_type(md.ExtHydrGrid)
        for grid in ext_hydr_grids:
            self.gas_balance_measurements.append(grid.model.mass_flow)

        ext_power_grids = network.childs_by_type(md.ExtPowerGrid)
        for grid in ext_power_grids:
            self.power_balance_measurements.append(grid.model.p_mw)

    def calc(self):
        return ("power_balance", self.power_balance_measurements), (
            "gas balance",
            self.gas_balance_measurements,
        )
