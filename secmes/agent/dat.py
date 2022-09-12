# dynamic structure in adaptive network-topologies

from abc import ABC
from dataclasses import dataclass
from enum import Enum
import random
from typing import List

from overrides import overrides

import networkx as nx

from secmes.agent.cell_agent import CellAgentRole
from secmes.agent.core import SecmesAgentRouter, SecmesRegionManager, SyncAgentRole


@dataclass
class RegionDisbandedMessage:
    region_id: int
    time: int


class SplittingStrategy(Enum):
    CONNECTED_COMPONENTS = 1
    DISINTEGRATE = 2


def execute_splitting_strategy(
    strategy: SplittingStrategy,
    router: SecmesAgentRouter,
    region_manager: SecmesRegionManager,
    cp_aid: str,
    networks: List[str],
    time: int,
):
    if strategy == SplittingStrategy.DISINTEGRATE:
        # The region will be closed to avoid not optimal region
        # structures.
        region = region_manager.get_agent_region(cp_aid)
        agents_in_own_region = region_manager.get_agents_region(region)
        agents_in_own_region.remove(cp_aid)

        # shut down CP, remove region, remove self from agent
        # topology
        router.unlink_cp(cp_aid, network_names=networks)
        agents_as_subgraph = router.get_agents_as_subgraph(agents_in_own_region)

        # It is possible that the region is still a connected component
        if len(list(nx.connected_components(agents_as_subgraph))) == 1:
            region_manager.remove_assigned_agent(cp_aid, region)
        else:
            region_manager.remove_region(region)

            # All agents will be notified
            # to give them a chance to join another region asap
            for aid in agents_in_own_region:
                router.dispatch_message_sync(
                    RegionDisbandedMessage(region, time), aid, cp_aid
                )
    else:
        region = region_manager.get_agent_region(cp_aid)
        agents_in_own_region = region_manager.get_agents_region(region)
        agents_in_own_region.remove(cp_aid)

        region_manager.remove_region(region)
        router.unlink_cp(cp_aid, network_names=networks)

        agents_as_subgraph = router.get_agents_as_subgraph(agents_in_own_region)

        for component in nx.connected_components(agents_as_subgraph):
            neighbor_regions = set()
            for node in component:
                for neighbor in router.lookup_neighbors(node):
                    neighbor_region = region_manager.get_agent_region(neighbor)
                    if neighbor_region is not None:
                        neighbor_regions |= {neighbor_region}
            region_manager.add_region(neighbor_regions, component)


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
    def __init__(
        self,
        nc,
        probablity: float,
        splitting_strategy: SplittingStrategy = SplittingStrategy.DISINTEGRATE,
    ) -> None:
        self._local_model = nc
        self._toggle = True
        self._probability = probablity
        self._splitting_strategy = splitting_strategy

    def control(self, time):
        if time > 2:
            region_m: SecmesRegionManager = self.region_manager
            router: SecmesAgentRouter = self.router

            if random.random() < self._probability:
                self._toggle = not self._toggle
                if self._toggle:
                    # Switch model on again
                    self._local_model.regulate(1)
                    router.link_cp(self.context.aid)
                    region_m.register_region(set(), self.context.aid)
                else:
                    # shut down physical
                    self._local_model.regulate(0)
                    # execute splitting strategy
                    execute_splitting_strategy(
                        self._splitting_strategy,
                        router,
                        region_m,
                        self.context.aid,
                        self._local_model.networks,
                        time,
                    )
