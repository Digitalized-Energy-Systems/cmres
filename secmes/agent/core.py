from abc import abstractmethod
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, Union
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

    def _generate_region_id(self):
        return (
            0
            if len(self._region_graph.nodes) == 0
            else max(self._region_graph.nodes.keys()) + 1
        )

    def register_region(self, neighbor_regions, initial_aid):
        region_id = self._generate_region_id()

        self._region_graph.add_node(region_id, assigned_agents={initial_aid})
        for region_neighbor in neighbor_regions:
            self._region_graph.add_edge(region_id, region_neighbor)
        return region_id

    def add_region(self, neighbor_regions, aid_set):
        region_id = self._generate_region_id()

        self._region_graph.add_node(region_id, assigned_agents=aid_set)
        for region_neighbor in neighbor_regions:
            self._region_graph.add_edge(region_id, region_neighbor)
        return region_id

    def register_agent(self, agent_id, region_id):
        # remove agent from current region and delete region if empty
        for node in self._region_graph.nodes:
            assigend_agents: Set = self._region_graph.nodes[node]["assigned_agents"]
            if agent_id in assigend_agents:
                assigend_agents.remove(agent_id)
                if not assigend_agents:
                    self._region_graph.remove_node(node)
                    break

        self._region_graph.nodes[region_id]["assigned_agents"] |= {agent_id}
        # print(f"{agent_id} registered as member of {region_id}")

    def get_agent_region(self, agent_id):
        for node in self._region_graph.nodes:
            if agent_id in self._region_graph.nodes[node]["assigned_agents"]:
                return node
        return None

    def get_agents_region(self, region_id) -> Set:
        return self._region_graph.nodes[region_id]["assigned_agents"]

    def get_data_as_copy(self):
        region_graph_copy = self._region_graph.copy()
        for node in region_graph_copy.nodes:
            region_graph_copy.nodes[node]["assigned_agents"] = region_graph_copy.nodes[
                node
            ]["assigned_agents"].copy()
        return region_graph_copy

    def remove_assigned_agent(self, aid, region_id):
        self._region_graph.nodes[region_id]["assigned_agents"] -= {aid}

    def remove_region(self, region_id):
        self._region_graph.remove_node(region_id)

    def remove_own_region(self, agent_id):
        self.remove_region(self.get_agent_region(agent_id))


VIRTUAL_NODE_CONTAIN_STR = ["junction", "bus"]


def to_aid(node_id):
    return node_id.split("#")[0] if node_id.find("#") != -1 else node_id


def to_node_ids(aid, network_names):
    return [aid + "#" + network_name for network_name in network_names]


def merge_length_dicts(first, second):
    shortest_path_length_dict = {}
    for node in first.keys() | second.keys():
        if node in second and node in first:
            shortest_path_length_dict[node] = min(first[node], second[node])
        else:
            shortest_path_length_dict[node] = (
                second[node] if node in second else first[node]
            )
    return shortest_path_length_dict


def is_virtual_node(agent_id):
    for contain_str in VIRTUAL_NODE_CONTAIN_STR:
        if contain_str in agent_id:
            return True
    return False


def calc_k_nearest_neighbors_excluding_virtual(node_to_shortest_length_map, k):
    shortest_path_length_tuple_list: List = list(node_to_shortest_length_map.items())
    shortest_path_length_tuple_list.sort(key=lambda v: v[1])
    return [
        to_aid(node)
        for node, _ in shortest_path_length_tuple_list[:k]
        if not is_virtual_node(node)
    ]


class SecmesAgentRouter:
    def __init__(self, agent_topology, neighborhood_size=10) -> None:
        self._agent_topology: nx.Graph = agent_topology
        self._neighbors_removed_cp = {}
        self._subgraph_removed_cp = {}
        self._cp_agent = {}
        self._neighborhood_size = neighborhood_size

    def dispatch_message_sync(self, content, agent_id, sender_agent_id):
        all_nodes = self._agent_topology.nodes
        if agent_id not in all_nodes:
            for nid in all_nodes:
                if "aid" in all_nodes[nid] and all_nodes[nid]["aid"] == agent_id:
                    agent = all_nodes[nid]["agent"]
                    break
        else:
            agent: SecmesAgent = all_nodes[agent_id]["agent"]
        agent.get_context().handle_msg(content, {"sender_agent_id": sender_agent_id})

    def unlink_cp(self, cp_id, network_names=None):
        all_cp_ids = to_node_ids(cp_id, network_names)
        self._subgraph_removed_cp[cp_id] = nx.Graph(
            self._agent_topology.subgraph(all_cp_ids)
        )

        self._neighbors_removed_cp[cp_id] = {}
        for actual_cp_id in all_cp_ids:
            self._neighbors_removed_cp[cp_id][actual_cp_id] = list(
                nx.neighbors(self._agent_topology, actual_cp_id)
            )
        self._agent_topology.remove_nodes_from(all_cp_ids)

    def link_cp(self, cp_id):
        self._agent_topology = nx.compose(
            self._subgraph_removed_cp[cp_id], self._agent_topology
        )

        for actual_cp_id, neighbors in self._neighbors_removed_cp[cp_id].items():
            for n in neighbors:
                if n in self._agent_topology.nodes:
                    self._agent_topology.add_edge(actual_cp_id, n)

    def calc_cp_neighborhood(self, cp_id, network_names, cutoff_length=0.1):
        node_ids = to_node_ids(cp_id, network_names)
        for node_id in node_ids:
            if not node_id in self._agent_topology.nodes:
                return []
        shortest_path_length_dict = None
        last_dict = None
        for node_id in node_ids:
            shortest_path_length_dict = nx.single_source_dijkstra_path_length(
                self._agent_topology, node_id, cutoff=cutoff_length, weight="weight"
            )
            if last_dict is not None:
                current_dict = shortest_path_length_dict
                shortest_path_length_dict = merge_length_dicts(current_dict, last_dict)
            last_dict = shortest_path_length_dict

        return calc_k_nearest_neighbors_excluding_virtual(
            shortest_path_length_dict, self._neighborhood_size
        )

    def calc_neighborhood(self, agent_id, cutoff_length=0.1):
        shortest_path_length_dict = nx.single_source_dijkstra_path_length(
            self._agent_topology, agent_id, cutoff=cutoff_length, weight="weight"
        )
        return calc_k_nearest_neighbors_excluding_virtual(
            shortest_path_length_dict, self._neighborhood_size
        )

    def lookup_direct_neighbors(
        self, agent_id, include_virtual_nodes=False, blacklist=None
    ):
        if blacklist is None:
            blacklist = []
        direct_neighbors = set(
            [
                n
                for n in nx.neighbors(self._agent_topology, agent_id)
                if n not in blacklist
            ]
        )
        virtual_neighbors = set([n for n in direct_neighbors if is_virtual_node(n)])
        direct_neighbors_with_indirect_virtual_neighbors = (
            direct_neighbors
            if include_virtual_nodes
            else direct_neighbors - virtual_neighbors
            | set(
                itertools.chain.from_iterable(
                    [
                        self.lookup_direct_neighbors(
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
        return any([agent_id in node for node in self._agent_topology.nodes])

    def get_agents_as_subgraph(self, agent_ids: List[str]):
        return self._agent_topology.subgraph(agent_ids)

    def get_data_as_copy(self):
        copy = self._agent_topology.copy()

        # remove agents to enable serializability
        for node_key in copy.nodes:
            copy.nodes[node_key]["agent"] = None
            copy.nodes[node_key]["roles"] = None

        return copy

    def get_data_as_ref(self):
        return self._agent_topology


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
