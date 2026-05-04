from dataclasses import dataclass
from abc import abstractmethod, ABC
from enum import Enum

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

    def __str__(self) -> str:
        return f"{self.time}.{self.severity}.{self.effect}: {self.component.grid.name}.{type(self.component.model)}.{self.component.id}"


class ResilienceModel(ABC):
    @abstractmethod
    def generate_failures(self, network, registry=None, scenario=None):
        pass


class StepModel(ABC):
    @abstractmethod
    def step(self, network, step):
        pass
