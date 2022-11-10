from secmes.resilience.core import ResilienceModel, RepairModel, Failure, Effect
from typing import List
import random

from peext.node import *
from peext.edge import *

import numpy as np
import scipy.stats as stats

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
    LineEdge: 0.05,
    TrafoEdge: 0.05,
    PipeEdge: 0.05,
}

FAILURE_PROBABILITY_MODEL = lambda base_prob: base_prob * np.random.normal(1, scale=0.1)

FAILURE_TIME_MODEL = lambda incident_time_steps, time: stats.norm.pdf(
    time, loc=incident_time_steps / 2
)
FAILURE_SPATIAL_MODEL = None


class SimpleResilienceModel(ResilienceModel):
    def __init__(
        self,
        incident_shift=5,
        incident_timesteps=10,
        heat_impact=1,
        gas_impact=1,
        power_impact=1,
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
        self._spatial_model = spatial_model

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
        return (
            self._fail_probability_model(base_failure_probability)
            * self._read_impact(model)
            * self._time_model(time)
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
}
DMG_COEFF_VARIANCE_MODEL = lambda dmg_coeff: dmg_coeff * np.random.normal(1, scale=0.1)


class SimpleRepairModel(RepairModel):
    def __init__(
        self,
        delay_for_repair=10,
        dmg_coeff_func_map=DMG_COEFF_FUNC_MAP,
        dmg_coeff_variance_model=DMG_COEFF_VARIANCE_MODEL,
    ) -> None:
        self._dmg_coeff_func_map = dmg_coeff_func_map
        self._dmg_coeff_variance_model = dmg_coeff_variance_model
        self._delay_for_repair = delay_for_repair

    def generate_repairs(self, _, failures: List[Failure]):
        for failure in failures:
            f: Failure = failure
            dmg = (
                self._dmg_coeff_variance_model(
                    self._dmg_coeff_func_map[type(f.node)](f.node)
                )
                * f.severity
            )
            time_needed = np.random.normal(5, 5) + self._delay_for_repair + dmg / 10
            f.repaired_time = int(f.time + time_needed)
        return failures
