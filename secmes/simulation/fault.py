from abc import ABC, abstractmethod
from typing import List, Tuple
from pandapower.control.basic_controller import Controller
from secmes.resilience.core import *

from peext.node import RegulatableController


class FaultExecutor(ABC):
    @abstractmethod
    def inject_fault(self, multinet):
        pass

    @abstractmethod
    def reverse_fault(self, multinet):
        pass


class DeadEffectFaultExecutor(FaultExecutor):
    def __init__(self, node, severity) -> None:
        self._affected_node = node
        self._severity = severity

    def inject_fault(self, _):
        if isinstance(self._affected_node, RegulatableController):
            self._affected_node.regulate(0)
            return
        self._affected_node.network[self._affected_node.component_type()].at[
            self._affected_node.id, "in_service"
        ] = False

    def reverse_fault(self, _):
        if isinstance(self._affected_node, RegulatableController):
            self._affected_node.regulate(1)
            return
        self._affected_node.network[self._affected_node.component_type()].at[
            self._affected_node.id, "in_service"
        ] = True


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
    def create_fault_executor(effect: Effect, severity: float, node: MESModel) -> Fault:
        if effect == Effect.DEAD:
            return DeadEffectFaultExecutor(node=node, severity=severity)

    @staticmethod
    def to_fault_obj(failure: Failure) -> Fault:
        return Fault(
            FaultGenerator.create_fault_executor(
                failure.effect, failure.severity, failure.node
            ),
            failure.time,
            failure.repaired_time,
        )

    def generate(self, me_network) -> List[Fault]:
        failures = self.failures = self._resilience_model.generate_failures(me_network)
        self._repair_model.generate_repairs(me_network, failures)
        return [FaultGenerator.to_fault_obj(failure) for failure in failures]


class FaultInjector(Controller):
    """Interface to the pandapipes/power controller system to overcome the need to have
    a mango agent. Useful for time-series simulations without the need of communication
    between real agents.
    """

    def __init__(
        self,
        multinet,
        faults: List[Fault],
        in_service=True,
        order=0,
        level=0,
        drop_same_existing_ctrl=False,
        initial_run=True,
        **kwargs
    ):
        super().__init__(
            multinet,
            in_service,
            order,
            level,
            drop_same_existing_ctrl=drop_same_existing_ctrl,
            initial_run=initial_run,
            **kwargs
        )

        self._faults = faults
        self._names = multinet["nets"].keys()

    def initialize_control(self, _):
        self.applied = False

    def get_all_net_names(self):
        return self._names

    def time_step(self, mn, time):

        if self._faults is not None:
            for fault in self._faults:
                if time == fault.start_time:
                    fault.fault_executor.inject_fault(mn)
                if time == fault.stop_time:
                    fault.fault_executor.reverse_fault(mn)

    def control_step(self, _):
        self.applied = True

    def is_converged(self, _):
        return self.applied
