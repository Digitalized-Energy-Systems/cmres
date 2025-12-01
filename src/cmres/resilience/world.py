"""Module for simulating the world. Contains the main loop and asyncio loop handling.
"""

import cmres.data.observer as observer

from cmres.resilience.fault import FaultInjector

from monee import Network, TimeseriesData, run_energy_flow, run_timeseries, StepHook
import monee.model as mm


def gen_id(el):
    return f"{el.network.name}:{el.component_type()}:{el.id}"


def calc_loss(edge):
    if edge.network.name == "heat":
        return edge.loss_perc()[1]
    elif edge.network.name == "gas":
        return edge.loss_perc()[0]
    return edge.loss_perc()


def to_eid_edge_map(me_network):
    edge_map = {}
    for edge in me_network.edges:
        edge_map[gen_id(edge)] = edge
    return edge_map


class CentralFaultyMoneeWorldStepHook(StepHook):
    def pre_run(self, base_net, step):
        pass

    def post_run(self, net, base_net, step):
        observer.gather(
            "balance_power",
            net.childs_by_type(mm.ExtPowerGrid)[0].model.p_mw,
        )
        observer.gather(
            "balance_gas",
            net.childs_by_type(mm.ExtHydrGrid)[1].model.mass_flow,
        )


class CentralFaultyMoneeWorld:
    def __init__(
        self,
        iteration_step_hook: StepHook,
        init_func,
        net: Network,
        timeseries_data: TimeseriesData,
        max_steps=None,
        name="CentralFaultyMoneeWorld",
        fault_generator=None,
    ) -> None:
        self._iteration_step_hook = iteration_step_hook
        self._fault_generator = fault_generator
        self._init_func = init_func
        self.__net = net
        self.__td = timeseries_data
        self._max_steps = max_steps
        self._name = name
        self._step_hooks = []

    def add_step_hook(self, step_hook: StepHook):
        self._step_hooks.append(step_hook)

    def run(self):
        """start asyncio event loop"""

        self.run_loop()

    def prepare(self):
        self.faults = self._fault_generator.generate(self.__net)
        if self._fault_generator is not None:
            self._step_hooks.append(
                FaultInjector(
                    self.faults,
                )
            )

        # initial single run for initial observation
        run_energy_flow(self.__net)

        self._init_func(self.__net)

    def run_loop(self):
        """Mainloop of the world simulation

        :param me_network: the MES network
        :type me_network: network.MENetwork
        :param panda_multinet: panda multinet
        :type panda_multinet: multinet
        :param plot_config: configuration of the plotting
        :type plot_config: Dict
        """
        run_timeseries(
            self.__net,
            self.__td,
            self._max_steps,
            [self._iteration_step_hook] + self._step_hooks,
            solve_flag=False,
        )
