
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
from mango.role.api import Role
from mango.role.core import RoleAgentContext
from pandapower.control.basic_controller import Controller
import networkx as nx

NETS_ACCESS = 'nets'

class SecmesAgent:
    @abstractmethod
    def control(self):
        pass
    @abstractmethod
    def get_context(self) -> RoleAgentContext:
        pass

class SecmesRegionManager:

    def __init__(self) -> None:
        self._region_graph = nx.Graph()

    def register_region(self, neighbor_regions):
        region_id = 0 if len(self._region_graph.nodes) == 0 else max(self._region_graph.nodes.keys()) + 1

        self._region_graph.add_node(region_id, assigned_agents=[])
        for region_neighbor in neighbor_regions:
            self._region_graph.add_edge(region_id, region_neighbor)
        return region_id

    def register_agent(self, agent_id, region_id):
        # remove agent from current region and delete region if empty
        for node in self._region_graph.nodes:
            assigend_agents = self._region_graph.nodes[node]['assigned_agents']
            if agent_id in assigend_agents:
                assigend_agents.remove(agent_id)
                if not assigend_agents:
                    self._region_graph.remove_node(node)
                    break

        self._region_graph.nodes[region_id]['assigned_agents'] += [agent_id]
        print(f"{agent_id} registered as member of {region_id}")

    def get_agent_region(self, agent_id):
        for node in self._region_graph.nodes:
            if agent_id in self._region_graph.nodes[node]['assigned_agents']:
                return node
        return None

    def get_agents_region(self, region_id):
        return self._region_graph.nodes[region_id]['assigned_agents']
    
    def get_data_as_copy(self):
        return self._region_graph.copy()

class SecmesAgentRouter:
    
    def __init__(self, agent_topology) -> None:
        self._agent_topology = agent_topology
        pass

    def dispatch_message(self, content, agent_id, sender_agent_id):
        agent: SecmesAgent = self._agent_topology.nodes[agent_id]['agent']
        agent.get_context().handle_msg(content, {'sender_agent_id': sender_agent_id})

    def lookup_neighbors(self, agent_id):
        return [n for n in nx.neighbors(self._agent_topology, agent_id)]


class SyncRoleAgentContext(RoleAgentContext):
    def __init__(self, role_handler, router: SecmesAgentRouter, region_manager: SecmesRegionManager, agent_id: int):
        super().__init__(None, role_handler, agent_id, None, None)

        self._region_manager = region_manager
        self._router = router

    def send_message_sync(
            self, content,
            _: Union[str, Tuple[str, int]], *,
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
        return self._router.dispatch_message(
            content=content,
            agent_id=receiver_id)

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
    def __init__(self, multinet, names, agents, region_manager: SecmesRegionManager = None, in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        super().__init__(multinet, in_service, order, level,
                        drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                        **kwargs)

        self._agents = agents
        self._names = names
        self._region_manager = region_manager
        self._region_history = [None, None]
        self._agent_state_data_list = {}

    def initialize_control(self, _):
        self.applied = False

    def get_all_net_names(self):
        return self._names

    def time_step(self, _, time):
        if time > 1:
            self.applied = False

            for agent in self._agents:
                if not hasattr(agent, 'time_pause') or time % (agent.time_pause() + 1) == 0:
                    if agent.context.aid not in self._agent_state_data_list:
                        self._agent_state_data_list[agent.context.aid] = []
                    self._agent_state_data_list[agent.context.aid].append((time, agent.control(time)))
            if self._region_manager is not None:
                self._region_history.append(self._region_manager.get_data_as_copy())



    def control_step(self, _):
        self.applied = True

    def is_converged(self, _):
        return self.applied

    @property
    def result(self):
        return self._region_history, self.get_agent_state_data()

    def get_agent_state_data(self):
        result = {}
        for aid, agent_state_list in self._agent_state_data_list.items():
            x = []
            y = []
            txt = []
            for time, agent_states in agent_state_list:
                for j, (k, v) in enumerate(agent_states.items()):
                    if len(x) <= j:
                        x.append([])
                        y.append([])
                        txt.append(k)
                    x[j].append(time)
                    y[j].append(v)
            result[aid] = (x, y, txt)
        return result
