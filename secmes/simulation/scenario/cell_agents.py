import asyncio
from posixpath import split
from typing import List
from secmes.agent.cell_agent import (
    CellAgentRole,
    DummyHeatCA,
    GasCA,
    HeatCA,
    PowerCA,
    PowerGasCA,
    PowerGasHeatCA,
    PowerHeatCA,
)
from secmes.agent.core import SecmesAgent, SecmesAgentRouter
from secmes.agent.world import AsyncWorld, gen_id

from peext.network import MENetwork
from peext.node import (
    CHPNode,
    P2GNode,
    G2PNode,
    P2HNode,
    ExtGasGrid,
    ExtPowerGrid,
    HeatExchangerNode,
    EmptyBusNode,
    EmptyJunctionNode,
    SwitchNode,
    CouplingPoint,
)

from mango.role.core import RoleAgent
from mango.core.container import Container

import networkx as nx

COMP_TO_CA_TYPE = {
    CHPNode: PowerGasHeatCA,
    P2GNode: PowerGasCA,
    G2PNode: PowerGasCA,
    P2HNode: PowerHeatCA,
    HeatExchangerNode: HeatCA,
}
NETWORK_TO_CA_DEFAULT = {"power": PowerCA, "gas": GasCA, "heat": DummyHeatCA}
BLACKLIST_NODES = {ExtGasGrid, ExtPowerGrid, SwitchNode}
BLACKLIST_AGENT_NODES = {EmptyJunctionNode, EmptyBusNode}


def get_agent_type(node):
    if type(node) in COMP_TO_CA_TYPE:
        return COMP_TO_CA_TYPE[type(node)]
    if node.network.name in NETWORK_TO_CA_DEFAULT:
        return NETWORK_TO_CA_DEFAULT[node.network.name]
    raise Exception("Sad.")


def gen_id_cp(node, network_name):
    return f"{gen_id(node)}#{network_name}"


def topology_from_me_network(
    me_network: MENetwork, agent_instantiator, additional_roles_instantiator
):
    agent_topology = nx.MultiGraph()
    for i, node in enumerate(me_network.nodes):
        if type(node) not in BLACKLIST_NODES:
            agent = None
            if type(node) not in BLACKLIST_AGENT_NODES:
                agent = agent_instantiator(node)
            if isinstance(node, CouplingPoint):
                edge_desc = node.edge_description()
                networks = node.networks
                roles = additional_roles_instantiator(node)
                for network_name in networks:
                    agent_topology.add_node(
                        gen_id_cp(node, network_name),
                        agent=agent,
                        roles=roles,
                        aid=gen_id(node),
                    )
                for from_net, to_net, efficiency in edge_desc:
                    first_cp_id = gen_id_cp(node, from_net)
                    second_cp_id = gen_id_cp(node, to_net)
                    if not agent_topology.has_node(
                        first_cp_id
                    ) or not agent_topology.has_node(first_cp_id):
                        raise Exception(edge_desc)
                    agent_topology.add_edge(
                        first_cp_id,
                        second_cp_id,
                        weight=efficiency,
                    )
            else:
                agent_topology.add_node(
                    gen_id(node),
                    agent=agent,
                    roles=additional_roles_instantiator(node),
                )

    for edge in me_network.edges:
        possible_connected = list(edge.nodes)
        for edge_tuple in edge.nodes:
            first_node = edge_tuple[0]
            possible_connected.remove(edge_tuple)
            for second_node, _, __ in possible_connected:
                if (
                    type(first_node) not in BLACKLIST_NODES
                    and type(second_node) not in BLACKLIST_NODES
                ):
                    first_id = (
                        gen_id_cp(first_node, edge.network.name)
                        if isinstance(first_node, CouplingPoint)
                        else gen_id(first_node)
                    )
                    second_id = (
                        gen_id_cp(second_node, edge.network.name)
                        if isinstance(second_node, CouplingPoint)
                        else gen_id(second_node)
                    )
                    if not agent_topology.has_node(
                        first_id
                    ) or not agent_topology.has_node(second_id):
                        raise Exception(first_id + second_id)

                    agent_topology.add_edge(first_id, second_id, edge_id=gen_id(edge))
    return agent_topology


def create_agents_coro(additional_roles_instantiator, agent_instantiator, port_add=0):
    async def create_agents(me_network, region_manager):
        # create global resources
        agent_graph = topology_from_me_network(
            me_network, agent_instantiator, additional_roles_instantiator
        )
        router = SecmesAgentRouter(agent_graph)
        container = await Container.factory(addr=("127.0.0.2", 5555 + port_add))

        # create results from gr
        agent_list = []
        agent_processed = set()
        for node in agent_graph.nodes:
            agent: SecmesAgent = agent_graph.nodes[node]["agent"]
            if agent is not None and agent not in agent_processed:
                agent_processed.add(agent)
                roles: List = agent_graph.nodes[node]["roles"]
                agent.secmes_setup(region_manager, router)

                # if coupling point cut off #... as there is only one agent for all
                # subnodes cps consist of
                aid = node.split("#")[0] if node.find("#") != -1 else node

                # inject context with gr to agents
                agent_list.append(agent)
                a = RoleAgent(container, suggested_aid=aid)
                a.add_role(agent)
                for role in roles:
                    role.secmes_setup(region_manager, router)
                    a.add_role(role)
                    agent_list.append(role)

        return agent_list, router, container

    return create_agents


def start_cell_simulation(
    multinet,
    additional_roles_instantiator,
    agent_instantiator=lambda nc: CellAgentRole(nc, get_agent_type(nc)(nc)),
    time_steps: int = 96,
    name="MESCellPoC",
    no_energy_flow=False,
    port_add=0,
):
    """Start a cell style simulation using an AsyncWorld and provided network topology +
       provided agent data

    Args:
        multinet (_type_): _description_
        additional_roles_instantiator (_type_): _description_
        agent_instantiator (_type_, optional): _description_. Defaults to lambdanc:CellAgentRole(nc, get_agent_type(nc)(nc)).
        time_steps (int, optional): _description_. Defaults to 96.
    """
    world = AsyncWorld(
        create_agents_coro(
            additional_roles_instantiator, agent_instantiator, port_add=port_add
        ),
        multinet,
        max_steps=time_steps,
        name=name,
        no_energy_flow=no_energy_flow,
    )
    loop = asyncio.get_event_loop()
    loop.run_until_complete(future=world.run())
