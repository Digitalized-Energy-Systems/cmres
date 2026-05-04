from abc import ABC, abstractmethod
from typing import List

from monee import StepHook, Network

import cmres.data.observer as observer
from cmres.resilience.core import Failure, Effect, ResilienceModel


def gen_id(node):
    return f"{node.name}:{node.model}:{node.id}"


def name_of(node):
    return gen_id(node)


class FaultExecutor(ABC):
    @abstractmethod
    def inject_fault(self, multinet):
        pass


class DeadEffectFaultExecutor(FaultExecutor):
    def __init__(self, component, severity) -> None:
        self._affected_component = component
        self._severity = severity

    def inject_fault(self, net: Network, time):
        net.deactivate(self._affected_component)

        observer.gather(
            "failure",
            {
                "step": time,
                "node": name_of(self._affected_component),
                "type": "failure",
            },
        )

    def __str__(self):
        return f"DeadEffectFaultExecutor({name_of(self._affected_component)}, severity={self._severity})"


class Fault:
    def __init__(self, fault_executor: FaultExecutor, start_time: int) -> None:
        self._fault_executor = fault_executor
        self._start_time = start_time

    @property
    def fault_executor(self):
        return self._fault_executor

    @property
    def start_time(self):
        return self._start_time

    def __str__(self):
        return f"Fault(executor={self._fault_executor}, start_time={self._start_time})"


class FaultGenerator:
    def __init__(
        self,
        resilience_model: ResilienceModel,
        registry=None,
        scenario=None,
    ) -> None:
        self._resilience_model = resilience_model
        self._registry = registry
        self._scenario = scenario

    @staticmethod
    def create_fault_executor(
        effect: Effect, severity: float, component
    ) -> "FaultExecutor":
        if effect == Effect.DEAD:
            return DeadEffectFaultExecutor(component=component, severity=severity)
        raise NotImplementedError(f"No FaultExecutor defined for effect {effect!r}")

    @staticmethod
    def to_fault_obj(failure: Failure) -> Fault:
        return Fault(
            FaultGenerator.create_fault_executor(
                failure.effect, failure.severity, failure.component
            ),
            failure.time,
        )

    def generate(self, network) -> List[Fault]:
        failures = self.failures = self._resilience_model.generate_failures(
            network, registry=self._registry, scenario=self._scenario
        )
        return [FaultGenerator.to_fault_obj(failure) for failure in failures]


class FaultInjector(StepHook):
    def __init__(
        self,
        faults: List[Fault],
    ):
        self._faults = faults

    def pre_run(self, base_net, step, step_state):
        if self._faults is not None:
            for fault in self._faults:
                if step == fault.start_time:
                    fault.fault_executor.inject_fault(base_net, step)
