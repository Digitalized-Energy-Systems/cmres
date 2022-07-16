# dynamic structure in adaptive network-topologies

from abc import ABC
from dataclasses import dataclass
import random

from overrides import overrides

from secmes.agent.cell_agent import CellAgentRole
from secmes.agent.core import SecmesAgentRouter, SecmesRegionManager, SyncAgentRole


@dataclass
class RegionDisbandedMessage:
    region_id: int
    time: int


class DynamicCoalitionAdaptionTopologyAgent(CellAgentRole, ABC):
    def setup(self):
        super().setup()
        self.context.subscribe_message(
            self,
            self.handle_region_disbanded,
            lambda c, _: isinstance(c, RegionDisbandedMessage),
        )

    def handle_region_disbanded(self, msg, _):
        self.control(msg.time)

    @overrides
    def execute_operation_point(self, time):
        # no operation point adjustment here
        pass


class DATCouplingPointRole(SyncAgentRole):
    def __init__(self, nc, probablity) -> None:
        self._local_model = nc
        self._toggle = True
        self._probability = probablity

    def control(self, time):
        region_m: SecmesRegionManager = self.region_manager
        router: SecmesAgentRouter = self.router

        if random.random() < self._probability:
            self._toggle = not self._toggle
            if self._toggle:
                # Switch model on again
                self._local_model.regulate(1)
                router.link_cp(self.context.aid)
            else:
                # The region will be closed to avoid unoptimal region
                # structures.
                region = region_m.get_agent_region(self.context.aid)
                agents_in_own_region = region_m.get_agents_region(region)

                # shut down CP, remove region, remove self from agent
                # topology
                self._local_model.regulate(0)
                router.unlink_cp(self.context.aid)
                region_m.remove_own_region(self.context.aid)

                # All agents will be notified
                # to give them a chance to join another region asap
                for aid in agents_in_own_region:
                    if aid != self.context.aid:
                        router.dispatch_message_sync(
                            RegionDisbandedMessage(region, time), aid, self.context.aid
                        )
