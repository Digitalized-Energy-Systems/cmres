from abc import ABC, abstractmethod
from typing import List
from cmres.resilience.core import *
import cmres.data.observer as observer

from monee import StepHook


def gen_id(node):
    return f"{node.name}:{node.model}:{node.id}"


def name_of(node):
    return gen_id(node)


class FaultExecutor(ABC):
    @abstractmethod
    def inject_fault(self, multinet):
        pass

    @abstractmethod
    def reverse_fault(self, multinet):
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

    def reverse_fault(self, net: Network, time):
        net.activate(self._affected_component)

        observer.gather(
            "repair",
            {"step": time, "node": name_of(self._affected_component), "type": "repair"},
        )


class Fault:
    def __init__(
        self, fault_executor: FaultExecutor, start_time: int, stop_time: int
    ) -> None:
        self._fault_executor = fault_executor
        self._start_time = start_time
        self._stop_time = stop_time

    @property
    def fault_executor(self):
        return self._fault_executor

    @property
    def start_time(self):
        return self._start_time

    @property
    def stop_time(self):
        return self._stop_time


class FaultGenerator:
    def __init__(
        self, resilience_model: ResilienceModel, repair_model: RepairModel
    ) -> None:
        self._resilience_model = resilience_model
        self._repair_model = repair_model

    @staticmethod
    def create_fault_executor(effect: Effect, severity: float, component) -> Fault:
        if effect == Effect.DEAD:
            return DeadEffectFaultExecutor(component=component, severity=severity)

    @staticmethod
    def to_fault_obj(failure: Failure) -> Fault:
        return Fault(
            FaultGenerator.create_fault_executor(
                failure.effect, failure.severity, failure.component
            ),
            failure.time,
            failure.repaired_time,
        )

    def generate(self, network) -> List[Fault]:
        failures = self.failures = self._resilience_model.generate_failures(network)
        self._repair_model.generate_repairs(network, failures)
        return [FaultGenerator.to_fault_obj(failure) for failure in failures]


class FaultInjector(StepHook):
    def __init__(
        self,
        faults: List[Fault],
    ):
        self._faults = faults

    def pre_run(self, base_net, step):
        if self._faults is not None:
            for fault in self._faults:
                if step == fault.start_time:
                    fault.fault_executor.inject_fault(base_net, step)
                if step == fault.stop_time:
                    fault.fault_executor.reverse_fault(base_net, step)
