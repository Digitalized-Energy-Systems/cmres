from dataclasses import dataclass
from abc import abstractmethod, ABC
from typing import List

from peext.core import MESModel

from peext.network import MENetwork


class PerformanceMetric(ABC):
    @abstractmethod
    def calc(self, me_network: MENetwork):
        pass


class ResilienceMetric(ABC):
    @abstractmethod
    def gather(self, me_network: MENetwork, step, **kwargs):
        pass

    @abstractmethod
    def calc(self):
        pass


class Effect(enumerate):
    DEAD = 0
    DECREASED_CAPACITY = 1


@dataclass
class Failure:
    time: int
    node: MESModel
    severity: float
    effect: Effect
    repaired_time: int

    def __str__(self) -> str:
        return f"{self.time}-{self.repaired_time}.{self.severity}.{self.effect}: {self.node.network.name}.{self.node.component_type()}.{self.node.id}"


class RepairModel(ABC):
    @abstractmethod
    def generate_repairs(self, me_network, failures: List[Failure]):
        pass


class ResilienceModel(ABC):
    @abstractmethod
    def generate_failures(self, me_network):
        pass


class StepModel(ABC):
    @abstractmethod
    def step(self, me_network, step):
        pass
