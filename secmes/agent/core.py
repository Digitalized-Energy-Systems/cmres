from abc import abstractmethod
from cgitb import lookup
import itertools
from typing import Any, Dict, Optional, Tuple, Union
from mango.role.api import Role
from mango.role.core import RoleAgentContext
from pandapower.control.basic_controller import Controller
import networkx as nx

NETS_ACCESS = "nets"


class SecmesAgent:
    @abstractmethod
    def control(self):
        pass

    @abstractmethod
    def get_context(self) -> RoleAgentContext:
        pass

    def secmes_setup(self, region_manager, sync_router):
        self.region_manager = region_manager
        self.router = sync_router


class SecmesRegionManager:
    def __init__(self) -> None:
        self._region_graph = nx.Graph()

    def register_region(self, neighbor_regions, initial_aid):
        region_id = (
            0
            if len(self._region_graph.nodes) == 0
            else max(self._region_graph.nodes.keys()) + 1
        )

        self._region_graph.add_node(region_id, assigned_agents=[initial_aid])
        for region_neighbor in neighbor_regions:
            self._region_graph.add_edge(region_id, region_neighbor)
        return region_id

    def register_agent(self, agent_id, region_id):
        # remove agent from current region and delete region if empty
        for node in self._region_graph.nodes:
            assigend_agents = self._region_graph.nodes[node]["assigned_agents"]
            if agent_id in assigend_agents:
                assigend_agents.remove(agent_id)
                if not assigend_agents:
                    self._region_graph.remove_node(node)
                    break

        self._region_graph.nodes[region_id]["assigned_agents"] += [agent_id]
        # print(f"{agent_id} registered as member of {region_id}")

    def get_agent_region(self, agent_id):
        for node in self._region_graph.nodes:
            if agent_id in self._region_graph.nodes[node]["assigned_agents"]:
                return node
        return None

    def get_agents_region(self, region_id):
        return self._region_graph.nodes[region_id]["assigned_agents"]

    def get_data_as_copy(self):
        return self._region_graph.copy()

    def remove_region(self, region_id):
        self._region_graph.remove_node(region_id)

    def remove_own_region(self, agent_id):
        self.remove_region(self.get_agent_region(agent_id))


VIRTUAL_NODE_CONTAIN_STR = ["junction", "bus"]


class SecmesAgentRouter:
    def __init__(self, agent_topology) -> None:
        self._agent_topology = agent_topology
        self._neighbors_removed_cp = {}
        self._cp_agent = {}

    def dispatch_message_sync(self, content, agent_id, sender_agent_id):
        agent: SecmesAgent = self._agent_topology.nodes[agent_id]["agent"]
        agent.get_context().handle_msg(content, {"sender_agent_id": sender_agent_id})

    def unlink_cp(self, cp_id):
        self._neighbors_removed_cp[cp_id] = nx.neighbors(self._agent_topology, cp_id)
        self._cp_agent[cp_id] = self._agent_topology.nodes[cp_id]["agent"]
        self._agent_topology.remove_node(cp_id)

    def link_cp(self, cp_id):
        self._agent_topology.add_node(cp_id, agent=self._cp_agent[cp_id])
        for n in self._neighbors_removed_cp[cp_id]:
            if n in self._agent_topology.nodes:
                self._agent_topology.add_edge(cp_id, n)

    def is_virtual_node(self, agent_id):
        for contain_str in VIRTUAL_NODE_CONTAIN_STR:
            if contain_str in agent_id:
                return True
        return False

    def lookup_neighbors(self, agent_id, include_virtual_nodes=False, blacklist=None):
        if blacklist is None:
            blacklist = []
        direct_neighbors = set(
            [
                n
                for n in nx.neighbors(self._agent_topology, agent_id)
                if n not in blacklist
            ]
        )
        virtual_neighbors = set(
            [n for n in direct_neighbors if self.is_virtual_node(n)]
        )
        direct_neighbors_with_indirect_virtual_neighbors = (
            direct_neighbors
            if include_virtual_nodes
            else direct_neighbors - virtual_neighbors
            | set(
                itertools.chain.from_iterable(
                    [
                        self.lookup_neighbors(
                            vn,
                            include_virtual_nodes=include_virtual_nodes,
                            blacklist=blacklist + [vn],
                        )
                        for vn in virtual_neighbors
                    ]
                )
            )
        )

        return direct_neighbors_with_indirect_virtual_neighbors

    def exists(self, agent_id):
        return agent_id in self._agent_topology.nodes

    def get_data_as_copy(self):
        copy = self._agent_topology.copy()

        # remove agents to enable serializability
        for node_key in copy.nodes:
            copy.nodes[node_key]["agent"] = None
            copy.nodes[node_key]["roles"] = None

        return copy


class SecmesRoleAgentContext(RoleAgentContext):
    def __init__(
        self,
        role_handler,
        router: SecmesAgentRouter,
        region_manager: SecmesRegionManager,
        agent_id: int,
    ):
        super().__init__(None, role_handler, agent_id, None, None)

        self._region_manager = region_manager
        self._router = router

    async def send_message(
        self,
        content,
        _: Union[str, Tuple[str, int]],
        *,
        receiver_id: Optional[str] = None,
        __: bool = False,
        ___: Optional[Dict[str, Any]] = None,
        ____: Optional[Dict[str, Any]] = None,
    ):
        """Send a message to another agent. Delegates the call to the agent-container.

        :param content: the message you want to send
        :param receiver_addr: address of the recipient
        :param receiver_id: ip of the recipient. Defaults to None.
        :param create_acl: set true if you want to create an acl. Defaults to False.
        :param acl_metadata: Metadata of the acl. Defaults to None
        :param mqtt_kwargs: Args for mqtt. Defaults to None.
        """
        return await self._router.dispatch_message(
            content=content, agent_id=receiver_id
        )

    def send_message_sync(
        self,
        content,
        _: Union[str, Tuple[str, int]],
        *,
        receiver_id: Optional[str] = None,
        __: bool = False,
        ___: Optional[Dict[str, Any]] = None,
        ____: Optional[Dict[str, Any]] = None,
    ):
        """Send a message to another agent. Delegates the call to the agent-container.

        :param content: the message you want to send
        :param receiver_addr: address of the recipient
        :param receiver_id: ip of the recipient. Defaults to None.
        :param create_acl: set true if you want to create an acl. Defaults to False.
        :param acl_metadata: Metadata of the acl. Defaults to None
        :param mqtt_kwargs: Args for mqtt. Defaults to None.
        """
        return self._router.dispatch_message_sync(content=content, agent_id=receiver_id)

    @property
    def router(self):
        return self._router

    @property
    def region_manager(self):
        return self._region_manager


class SyncAgentRole(Role, SecmesAgent):
    def get_context(self):
        return self.context


class AgentController(Controller):
    """Interface to the pandapipes/power controller system to overcome the need to have
    a mango agent. Useful for time-series simulations without the need of communication
    between real agents.
    """

    def __init__(
        self,
        multinet,
        names,
        agents,
        region_manager: SecmesRegionManager = None,
        router: SecmesAgentRouter = None,
        in_service=True,
        order=0,
        level=0,
        drop_same_existing_ctrl=False,
        initial_run=True,
        **kwargs,
    ):
        super().__init__(
            multinet,
            in_service,
            order,
            level,
            drop_same_existing_ctrl=drop_same_existing_ctrl,
            initial_run=initial_run,
            **kwargs,
        )

        self._agents = agents
        self._names = names
        self._region_manager = region_manager
        self._region_history = [None, None]
        self._agent_state_data_list = {}
        self._router = router
        self._topology_graph_history = []

    def initialize_control(self, _):
        self.applied = False

    def get_all_net_names(self):
        return self._names

    def time_step(self, _, time):
        if time > 1:
            self.applied = False

            for agent in self._agents:
                if (
                    not hasattr(agent, "time_pause")
                    or time % (agent.time_pause() + 1) == 0
                ):
                    if agent.context.aid not in self._agent_state_data_list:
                        self._agent_state_data_list[agent.context.aid] = []
                    agent_states = agent.control(time)
                    if agent_states is not None:
                        self._agent_state_data_list[agent.context.aid].append(
                            (time, agent_states)
                        )
            for agent in self._agents:
                if hasattr(agent, "post_control"):
                    agent.post_control(time)
            if self._region_manager is not None:
                self._region_history.append(self._region_manager.get_data_as_copy())
        if self._router is not None:
            self._topology_graph_history.append(self._router.get_data_as_copy())

    def control_step(self, _):
        self.applied = True

    def is_converged(self, _):
        return self.applied

    @property
    def result(self):
        return (
            self._region_history,
            self.get_agent_state_data(),
            self._topology_graph_history,
        )

    def get_agent_state_data(self):
        result = {}
        for aid, agent_state_list in self._agent_state_data_list.items():
            x = {}
            y = {}
            txt = []
            for time, agent_states in agent_state_list:
                if len(agent_states) <= 0:
                    continue
                for k, v in agent_states.items():
                    if k not in x:
                        x[k] = []
                        if isinstance(v, list):
                            y[k] = {}
                            for v_el in v:
                                y[k][v_el[0]] = []
                        else:
                            y[k] = []
                        txt.append(k)
                    x[k].append(time)
                    if isinstance(v, list):
                        for v_el in v:
                            if v_el[0] in y[k]:
                                y[k][v_el[0]].append(v_el[1])
                            else:
                                y[k][v_el[0]] = [None for _ in range(len(x[k]) - 1)] + [
                                    v_el[1]
                                ]
                        for _, vyk in y[k].items():
                            if len(vyk) != len(x[k]):
                                vyk.append(None)
                    else:
                        y[k].append(v)
            result[aid] = to_list_datas(x, y, txt)
        return result


def to_list_datas(x, y, label):
    x_l = []
    y_l = []
    label_l = []
    for k in label:
        x_l.append(x[k])
        y_l.append(y[k])
        label_l.append(k)
    return x_l, y_l, label_l
