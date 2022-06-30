"""Module for simulating the world. Contains the main loop and asyncio loop handling.
"""
import asyncio
import pickle
import os
import tempfile
from typing import List

from pandapower.control.run_control import NetCalculationNotConverged
from pandapower.powerflow import LoadflowNotConverged
from pandapower.timeseries.output_writer import OutputWriter
import pandapipes.multinet.timeseries as multinettimeseries
import pandapipes

from peext.network import MENetwork
from peext.world.core import MASWorld
from secmes.agent.core import AgentController, SecmesAgent, SecmesAgentRouter, SecmesRegionManager
from peext.data.core import DataSink, StaticPlottingController

from secmes.rl.drl.dqn import DQNAgent
from secmes.rl.drl.role import DQNRole
from datetime import datetime

from secmes.scenario.fault import Fault, FaultInjector

class SyncWorld:
    """Base for any simulation in secmes. Defines the simulation steps.
    """
    def __init__(self, 
                    agents: List[SecmesAgent], 
                    router: SecmesAgentRouter, 
                    me_network: MENetwork, 
                    faults: List[Fault] = None,
                    region_manager: SecmesRegionManager = None,
                    steps = 96 * 30, 
                    data_sink: DataSink = None, 
                    name = None) -> None:
        self._agents = agents
        self._router = router
        self._region_manager = region_manager
        self._me_network = me_network
        self._data_sink = data_sink
        self._steps = steps
        self._name = f"sim-{datetime.now()}" if name is None else name
        self._faults = faults
        
    @property
    def agents(self):
        return self._agents

    @property
    def name(self):
        return self._name
        
    def step(self):
        try:        
            time_steps = range(self._steps)
            OutputWriter(self._me_network.multinet['nets']['heat'], time_steps=time_steps, output_path=tempfile.gettempdir(), log_variables=[])

            multinettimeseries.run_timeseries(self._me_network.multinet, time_steps=time_steps, continue_on_divergence=True, mode='all', verbose=False)
        except (NetCalculationNotConverged, LoadflowNotConverged) as ex:
            print("Multi-Net did not converge. This may be caused by invalid controller states: {0}".format(ex))

    def write_results(self, static_plotting_controller: StaticPlottingController, agent_controller: AgentController):
        if not os.path.isdir(self.name):
            os.mkdir(self.name)
        pandapipes.to_pickle(self._me_network.multinet, f"{self.name}/network.p")
        with open(f"{self.name}/network-result.p", "wb") as output_file:
            pickle.dump(static_plotting_controller.result, output_file)
        with open(f"{self.name}/agent-result.p", "wb") as output_file:
            pickle.dump(agent_controller.result, output_file)

    def run(self):
        """Running the world simulation
        """
        # include controller to execute agents
        mn_names = self._me_network.multinet['nets'].keys()
        agent_controller = AgentController(self._me_network.multinet, mn_names, self._agents, region_manager=self._region_manager)
        static_plotting_controller = StaticPlottingController(self._me_network.multinet, mn_names, self._me_network, self._steps-1, only_collect=True)
        fault_controller = FaultInjector(self._me_network.multinet, self._faults)

        # calculate new network results
        self.step()

        # write result to filesystem
        # Remove plotting controller first, as it contains the me_network, which we dont want to serialize
        self._me_network.multinet.controller.drop(static_plotting_controller.index, inplace=True)
        self._me_network.multinet.controller.drop(agent_controller.index, inplace=True)
        self._me_network.multinet.controller.drop(fault_controller.index, inplace=True)
        self.write_results(static_plotting_controller, agent_controller)


class DRLWorld(MASWorld):
    """World for simulating multiple agents using deep reinforcement learning to tackle some kind of control problem in 
    multi-energy context.
    """

    def learning_agents(self):
        """Define the learning agents, which use DRL internally to determine the control action

        :return: list of roles
        :rtype: List
        """
        learning_roles = []
        for agent in self.agents:
            for role in agent.roles:
                if isinstance(role, DQNRole):
                    learning_roles.append(role)
                    break
                    
        return learning_roles
    
    def calc_reward(self, obs, set_points):
        """Calculate the reward for some observation given the set points.

        :param obs: the observation
        :type obs: List
        :param set_points: the set points
        :type set_points: List
        :return: the reward
        :rtype: float
        """
        fitness = 0

        for i in range(len(set_points)):
            fitness -= abs(obs[i]- set_points[i][0])
        return fitness

    def train_agents(self, episodes=10000, duration=200):
        """Train the agents together.

        :param episodes: number of episodes, defaults to 10000
        :type episodes: int, optional
        :param duration: duration, defaults to 200
        :type duration: int, optional
        """
        learning_roles = self.learning_agents()
        for episode in range(episodes):
            obs = {}
            print('new episode')
            for i in range(duration):
                for role in learning_roles:
                    drl_agent: DQNAgent = role.drl_agent
                    
                    # before episode start 
                    # initial observation
                    if i == 0:
                        obs[role] = [value for _, value in role.observation.items()]
                    
                    # best learned action
                    action = drl_agent.predict(obs[role])
                    # set action to agent and step environment
                    role.action = action.item()
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(future=self.step())
                    # fetch reward and new state
                    new_obs = [value for _, value in role.observation.items()]
                    reward = self.calc_reward(new_obs, role.set_points)

                    drl_agent.train_one_step(reward, new_obs, obs[role], action)
                    obs[role] = new_obs
                    
                    # after episode ends
                    drl_agent.on_after(i, duration, episode)