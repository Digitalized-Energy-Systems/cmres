from dataclasses import dataclass
from abc import abstractmethod, ABC
from enum import Enum
from typing import List

from monee.model import Component, Network


class PerformanceMetric(ABC):
    @abstractmethod
    def calc(self, network: Network):
        pass


class ResilienceMetric(ABC):
    @abstractmethod
    def gather(self, network: Network, step, **kwargs):
        pass

    @abstractmethod
    def calc(self):
        pass


class Effect(Enum):
    DEAD = 0
    DECREASED_CAPACITY = 1


@dataclass
class Failure:
    time: int
    component: Component
    severity: float
    effect: Effect
    repaired_time: int

    def __str__(self) -> str:
        return f"{self.time}-{self.repaired_time}.{self.severity}.{self.effect}: {self.component.grid.name}.{type(self.component.model)}.{self.component.id}"


class RepairModel(ABC):
    @abstractmethod
    def generate_repairs(
        self, network, failures: List[Failure], registry=None, scenario=None
    ):
        pass


class ResilienceModel(ABC):
    @abstractmethod
    def generate_failures(self, network, registry=None, scenario=None):
        pass


class StepModel(ABC):
    @abstractmethod
    def step(self, network, step):
        pass
