from cmres.resilience.core import (
    ResilienceModel,
    RepairModel,
    Failure,
    Effect,
    StepModel,
)
from cmres.resilience.metric import (
    GeneralResiliencePerformanceMetric,
    CascadingResilienceMetric,
)
from cmres.resilience.fault import name_of
from typing import List
import random

import numpy as np
import scipy.stats as stats

import cmres.data.observer as observer
import cmres.omef.solver.monee as ms

import monee.model as mm
from monee import Network


FAIL_BASE_PROBABILITY_MAP = {
    mm.Source: 0.1,
    mm.Sink: 0.0,
    mm.PowerToGas: 0.1,
    mm.PowerToHeat: 0.1,
    mm.CHP: 0.1,
    mm.GasToPower: 0.1,
    mm.PowerGenerator: 0.1,
    mm.HeatExchanger: 0.1,
    mm.HeatExchangerGenerator: 0.1,
    mm.HeatExchangerLoad: 0.1,
    mm.GenericPowerBranch: 0.1,
    mm.PowerLine: 0.1,
    mm.Trafo: 0.00,
    mm.WaterPipe: 0.1,
    mm.GasPipe: 0.1,
    mm.Bus: 0.00,
    mm.Junction: 0.00,
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
        spatial_mode=FAILURE_SPATIAL_MODEL,
        base_fail=0.3,
    ) -> None:
        self._base_fail = base_fail
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
        self._spatial_model = lambda coords: (
            spatial_model(coords) if coords is not None else 1
        )

    def _read_impact(self, component):
        if (
            not hasattr(component, "grid")
            or component.grid == None
            or type(component.grid) == dict
        ):
            return self._mes_impact

        if component.grid.name == "water":
            return self._heat_impact
        elif component.grid.name == "power":
            return self._power_impact
        elif component.grid.name == "gas":
            return self._gas_impact

        return self._mes_impact

    def calc_fail(self, network: Network, component, time):
        model_type = type(component.model)
        if model_type not in self._base_fail_probability_map:
            return 0
        base_failure_probability = self._base_fail_probability_map[model_type]
        #coords = mm.calc_coordinates(network, component)
        return (
            self._fail_probability_model(base_failure_probability)
            * self._read_impact(component)
            * self._time_model(time)
            * self._base_fail
         #   * self._spatial_model(coords)
        )

    def has_failed(self, model, time):
        return random.random() < self.calc_fail(model, time)

    def generate_failures(self, net: Network):
        failures = []
        for i in range(self._incident_timesteps):
            time = i + self._incident_shift
            for node in net.nodes:
                if not node.independent:
                    continue
                fail_prob = self.calc_fail(net, node, time)
                if random.random() < fail_prob:
                    failures.append(Failure(time, node, fail_prob, Effect.DEAD, -1))
            for branch in net.branches:
                if not branch.independent:
                    if type(branch.model) != mm.HeatExchanger:
                        continue
                fail_prob = self.calc_fail(net, branch, time)
                if random.random() < fail_prob:
                    failures.append(Failure(time, branch, fail_prob, Effect.DEAD, -1))
            for child in net.childs:
                if not child.independent:
                    continue
                fail_prob = self.calc_fail(net, child, time)
                if random.random() < fail_prob:
                    failures.append(Failure(time, child, fail_prob, Effect.DEAD, -1))
            for compound in net.compounds:
                fail_prob = self.calc_fail(net, compound, time)
                if random.random() < fail_prob:
                    failures.append(Failure(time, compound, fail_prob, Effect.DEAD, -1))
        return failures


DMG_COEFF_FUNC_MAP = {
    mm.Source: lambda model: mm.upper(model.mass_flow),
    mm.Sink: lambda model: mm.upper(model.mass_flow),
    mm.PowerToGas: lambda model: mm.upper(model.to_mass_flow),
    mm.PowerToHeat: lambda model: mm.upper(model.heat_energy_mw),
    mm.CHP: lambda model: mm.upper(model.mass_flow),
    mm.GasToPower: lambda model: mm.upper(model.from_mass_flow),
    mm.PowerGenerator: lambda model: mm.upper(model.p_mw),
    mm.HeatExchanger: lambda model: mm.upper(model.q_w),
    mm.HeatExchangerGenerator: lambda model: mm.upper(model.q_w),
    mm.HeatExchangerLoad: lambda model: mm.upper(model.q_w),
    mm.GenericPowerBranch: lambda model: model.br_r,
    mm.PowerLine: lambda model: model.br_r,
    mm.Trafo: lambda _: 1,
    mm.WaterPipe: lambda model: model.length_m,
    mm.GasPipe: lambda model: model.length_m,
    mm.Bus: lambda _: 1,
    mm.Junction: lambda _: 1,
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
            if not type(f.component.model) in self._dmg_coeff_func_map:
                raise Exception(
                    f"There is no dmg coeff defined for {type(f.component.model)}!"
                )
            dmg = (
                self._dmg_coeff_variance_model(
                    self._dmg_coeff_func_map[type(f.component.model)](f.component.model)
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


def deactivate_node(network: Network, node):
    for child_id in node.child_ids:
        child = network.child_by_id(child_id)
        network.deactivate(child)


def activate_node(network: Network, node):
    for child_id in node.child_ids:
        child = network.child_by_id(child_id)
        network.activate(child)


def calc_relative_violation(component, attribute, target, rel_allowed_diff):
    return max(
        (
            abs((mm.value(getattr(component.model, attribute)) - target))
            - rel_allowed_diff * target
        )
        / (target * rel_allowed_diff),
        0,
    )


class CascadingModel(StepModel):
    def __init__(self, performance_accuracy=100) -> None:
        self._cascading_metric = CascadingResilienceMetric()
        self._performance_metric = GeneralResiliencePerformanceMetric()
        self._current_failures = []
        self._iteration_number_omef = performance_accuracy
        self._faults = None
        self._last_performance = None

    def calc_performance(self, network: Network, without_load=False):
        result = ms.solve(network)
        pm = self._performance_metric.calc(result.network)
        return (
            self._performance_metric.calc(result.network),
            result,
        )

    def check_repairs(self, network, bound_tuple, step, network_name):
        attribute, target, allowed_diff = bound_tuple
        for i in range(len(self._current_failures) - 1, -1, -1):
            node, failure = self._current_failures[i]
            if node.grid.name != network_name or attribute not in node.values_as_dict():
                continue
            if failure["step"] + failure["min_duration"] >= step:
                relative_violation = to_failure_probability(
                    calc_relative_violation(node, attribute, target, allowed_diff)
                )
                if relative_violation < np.random.random():
                    activate_node(network, node)
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

    def process_node(self, network: Network, node, bound_tuple, step):
        attribute, target, allowed_diff = bound_tuple
        relative_violation = to_failure_probability(
            calc_relative_violation(node, attribute, target, allowed_diff)
        )
        if relative_violation > np.random.random():
            min_duration = int(relative_violation * np.random.random() * 10)
            deactivate_node(network, node)
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

    def process_network_state(self, network: Network, step):
        self.check_repairs(network, ms.BOUND_GAS, step, "gas")
        self.check_repairs(network, ms.BOUND_HEAT, step, "heat")
        self.check_repairs(network, ms.BOUND_EL, step, "power")

        for node in network.nodes:
            if not node.independent:
                continue
            if node.grid.name == "gas":
                self.process_node(network, node, ms.BOUND_GAS, step)
            if node.grid.name == "heat":
                self.process_node(network, node, ms.BOUND_HEAT, step)
            if node.grid.name == "power":
                self.process_node(network, node, ms.BOUND_EL, step)

        for branch in network.branches_by_type(mm.GenericPowerBranch):
            if not branch.independent:
                continue
            self.process_node(network, branch, ms.BOUND_LP, step)

    def fault_delta_exists(self, step):
        for fault in self._faults:
            if fault.start_time == step or fault.stop_time == step:
                return True
        return False

    def step(self, net, base_net, step):
        if self.fault_delta_exists(step) or self._last_performance is None:
            try:
                performance, res = self.calc_performance(net)
            except:
                performance = float("inf")
            self._last_performance = performance
        else:
            performance = self._last_performance

        observer.gather("performance", performance)

        """
        self.process_network_state(net, step)

        # calc performance and base performance
        performance_after_cascade, _ = self.calc_performance(net)
        """
        observer.gather("performance_after_cascade", performance)

        self._cascading_metric.gather(
            net,
            step,
            performance=performance,
            performance_after_cascade=performance,
        )

    def calc_metric(self):
        return self._cascading_metric.calc()
