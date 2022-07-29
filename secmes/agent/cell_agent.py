# jeder agent registriert sich eine zelle
# agent prüft alle nachbarn
# auf eine vereinigungsscore, score hoch genug -> vereinigen, sonst -> nix
# bei vereinigung degregistrierung der alten zelle und registrierung einer neuen
# regulation entsprechend der region balance
# wird in einem iterativen PID verfahren erledigt (Dynamik ist hier gegeben durch die Agenten, nicht das Netz!)
# Spannung/Druck: ?

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import local

import numpy as np

from secmes.agent.core import SecmesAgentRouter, SecmesRegionManager, SyncAgentRole
from secmes.da.dgd import (
    generate_linear_desc_array,
    generate_random_doubly_stoch_mat,
    iteration_step,
)

import peext.network as pn
from peext.node import (
    RegulatableMESModel,
    PowerLoadNode,
    P2GNode,
    P2HNode,
    CHPNode,
    G2PNode,
    SinkNode,
    HeatExchangerNode,
)


@dataclass
class BalanceSumRequest:
    pass


@dataclass
class JoinRequest:
    region_id: int
    region_attraction: int


@dataclass
class BalanceSumAnswer:
    sum: float


@dataclass
class CalcOperationPointIteration:
    time: int
    w: object
    alpha: object
    x: object


INVERT_POWER_BALANCE_RATE = {PowerLoadNode, P2GNode, P2HNode}
INVERT_GAS_BALANCE_RATE = {CHPNode, G2PNode, SinkNode}


class CellAgentRole(SyncAgentRole, ABC):
    def __init__(self, network_component, common_nc_data_access) -> None:
        super().__init__()

        self._local_model = network_component
        self._common_nc_data_access = common_nc_data_access
        self._environment_perceptions = {}
        self._operation_point = {}

    def setup(self):
        self.context.subscribe_message(
            self, self.handle_join_request, lambda c, _: isinstance(c, JoinRequest)
        )
        self.context.subscribe_message(
            self,
            self.handle_balance_answer,
            lambda c, _: isinstance(c, BalanceSumAnswer),
        )
        self.context.subscribe_message(
            self,
            self.handle_balance_request,
            lambda c, _: isinstance(c, BalanceSumRequest),
        )
        self.context.subscribe_message(
            self,
            self.handle_calc_op,
            lambda c, _: isinstance(c, CalcOperationPointIteration),
        )

        self.create_initial_region()

    def create_initial_region(self):
        region_m: SecmesRegionManager = self.region_manager
        router: SecmesAgentRouter = self.router
        if router.exists(self.context.aid):
            neighbors = router.lookup_neighbors(self.context.aid)
            neighbor_regions = set(
                [
                    region_m.get_agent_region(neighbor)
                    for neighbor in neighbors
                    if region_m.get_agent_region(neighbor) is not None
                ]
            )
            region_id = region_m.register_region(neighbor_regions, self.context.aid)
            return region_id
        return None

    def handle_join_request(self, content: JoinRequest, meta):

        region_m: SecmesRegionManager = self.region_manager
        region = region_m.get_agent_region(self.context.aid)
        if region is not None:

            region_agents = region_m.get_agents_region(region)
            calculated_region_balance = self.calc_region_balance(region_agents)

            sender_aid = meta["sender_agent_id"]

            attraction_towards_requesting_region = self.calc_agent_attraction(
                sender_aid, calculated_region_balance
            )
            attraction_towards_current_region = self.calc_agent_attraction(
                self.context.aid, self._common_nc_data_access.calc_balance()
            )

            if (
                attraction_towards_requesting_region
                >= attraction_towards_current_region
            ).all():
                region_m.register_agent(self.context.aid, content.region_id)

    def _get_sender_id(self, meta):
        return meta["sender_agent_id"]

    def _assign_perception(self, id, key, value):
        if id not in self._environment_perceptions:
            self._environment_perceptions[id] = {}
        self._environment_perceptions[id][key] = value

    def handle_balance_answer(self, content, meta):
        self._assign_perception(self._get_sender_id(meta), "balance", content.sum)

    def handle_balance_request(self, _, meta):
        router: SecmesAgentRouter = self.router
        router.dispatch_message_sync(
            BalanceSumAnswer(self._common_nc_data_access.calc_balance()),
            self._get_sender_id(meta),
            self.context.aid,
        )
        pass

    def get_or_request_balance(self, agent_id):
        if agent_id == self.context.aid:
            return self._common_nc_data_access.calc_balance()
        router: SecmesAgentRouter = self.router
        router.dispatch_message_sync(BalanceSumRequest(), agent_id, self.context.aid)
        return self._environment_perceptions[agent_id]["balance"]

    def calc_region_balance(self, region_agents):
        sum = 0
        for region_agent in region_agents:
            if region_agent == self.context.aid:
                sum += self._common_nc_data_access.calc_balance()
            else:
                sum += self.get_or_request_balance(region_agent)
        return sum

    def calc_agent_attraction(self, other, calculated_balance):
        other_balance = self.get_or_request_balance(other)
        return -np.sign(calculated_balance) * other_balance - self.calc_cost_gradient(
            other
        )

    def project(self, x, r):
        # cons_x = self._common_nc_data_access.define_local_constraints()(x/(r/10 + 1))
        # return np.clip(x if cons_x is None else cons_x, 0, 1)
        # return x if cons_x is None else cons_x
        return x

    def handle_calc_op(self, content: CalcOperationPointIteration, _):
        r = self._local_model.regulation_factor()
        time = content.time
        w = content.w
        x = content.x
        region_agents = self.region_manager.get_agents_region(
            self.region_manager.get_agent_region(self.context.aid)
        )
        i = region_agents.index(self.context.aid)
        # f = lambda x: self._common_nc_data_access.max_energy() * x
        new_x = iteration_step(
            w,
            x,
            lambda x: self._common_nc_data_access.max_energy(),
            len(region_agents),
            i,
            time,
        )
        x[i] = self.project(new_x, r)
        self._operation_point[time] = x[i]

    def calculate_operation_point(self, time):
        if time in self._operation_point:
            return self._operation_point[time]
        router: SecmesAgentRouter = self.router
        region_agents = self.region_manager.get_agents_region(
            self.region_manager.get_agent_region(self.context.aid)
        )
        r = self._local_model.regulation_factor()
        m = len(region_agents)
        x = np.array([np.ones(3) for i in range(m)])
        # f = lambda x: self._common_nc_data_access.max_energy() * x
        i = region_agents.index(self.context.aid)
        alpha = generate_linear_desc_array(100)
        w = generate_random_doubly_stoch_mat((m, m))

        for j in range(100):
            new_x = iteration_step(
                w, x, lambda x: self._common_nc_data_access.max_energy(), m, i, j
            )
            for region_agent in region_agents:
                router.dispatch_message_sync(
                    CalcOperationPointIteration(time, w, alpha, x),
                    region_agent,
                    self.context.aid,
                )
            x[i] = self.project(new_x, r)

        self._operation_point[time] = x[i]
        return x[i]

    def execute_operation_point(self, time):
        if isinstance(self._local_model, RegulatableMESModel):
            operation_point = self.calculate_operation_point(time)
            # todo op aggregation
            self._local_model.regulate(operation_point[0])

    def calc_cost_gradient(self, neighbor):
        return [0, 0, 0]

    def control(self, time):
        region_m: SecmesRegionManager = self.region_manager
        router: SecmesAgentRouter = self.router
        agent_control_values = {}
        aid = self.context.aid
        region = region_m.get_agent_region(aid)
        if region is None:
            region = self.create_initial_region()

        if region is not None:
            region_agents = region_m.get_agents_region(region)
            calculated_region_balance = self.calc_region_balance(region_agents)
            agent_control_values["region_balance_power"] = calculated_region_balance[0]
            agent_control_values["region_balance_heat"] = calculated_region_balance[1]
            agent_control_values["region_balance_gas"] = calculated_region_balance[2]

            agent_control_values["attraction_power"] = []
            agent_control_values["attraction_heat"] = []
            agent_control_values["attraction_gas"] = []
            neighbors = router.lookup_neighbors(aid)
            for neighbor in neighbors:
                attraction = self.calc_agent_attraction(
                    neighbor, calculated_region_balance
                )
                agent_control_values["attraction_power"].append(
                    (neighbor, attraction[0])
                )
                agent_control_values["attraction_heat"].append(
                    (neighbor, attraction[1])
                )
                agent_control_values["attraction_gas"].append((neighbor, attraction[2]))
                if neighbor in region_agents:
                    continue
                if (attraction >= 0).all():
                    router.dispatch_message_sync(
                        JoinRequest(region, attraction), neighbor, aid
                    )

            self.execute_operation_point(time)

        agent_control_values["region"] = (
            -1
            if region_m.get_agent_region(aid) is None
            else region_m.get_agent_region(aid)
        )
        return agent_control_values


def to_multi_energy(power=0, heat=0, gas=0):
    if (
        not isinstance(power, (float, int))
        or not isinstance(heat, (float, int))
        or not isinstance(gas, (float, int))
    ):
        raise Exception(f"Input values are suspicious! {power}.{heat}.{gas}")
    return np.array([power, heat, gas])


VM_PU_REF = 1
P_BAR_REF = 60
TEMP_K_REF = 375


def power_constraint(bus_data, r):
    if np.abs(VM_PU_REF - bus_data["power"][0]["vm_pu"]) * (r[0] / 10 + 1) <= 0.1:
        if bus_data["power"][0]["vm_pu"] > VM_PU_REF:
            return 1.1 / bus_data["power"][0]["vm_pu"]
        else:
            return 0.9 / bus_data["power"][0]["vm_pu"]


def heat_constraint(junc_data, r):
    if (
        TEMP_K_REF * 0.9
        <= junc_data["heat"][1]["t_k"] * (r[1] / 10 + 1)
        <= TEMP_K_REF * 1.1
    ):
        if junc_data["heat"][0]["t_k"] > TEMP_K_REF:
            return 1.1 / junc_data["heat"][0]["t_k"]
        else:
            return 0.9 / junc_data["heat"][0]["t_k"]


def gas_constraint(junc_data, r):
    if (
        P_BAR_REF * 0.9
        <= junc_data["gas"][0]["p_bar"] * (r[2] / 10 + 1)
        <= P_BAR_REF * 1.1
    ):
        if junc_data["gas"][0]["p_bar"] > P_BAR_REF:
            return 1.1 / junc_data["gas"][0]["p_bar"]
        else:
            return 0.9 / junc_data["gas"][0]["p_bar"]


class CommonNetworkComponentAccess(ABC):
    def __init__(self, local_model) -> None:
        super().__init__()

        self._local_model = local_model

    @abstractmethod
    def calc_balance(self):
        pass

    @abstractmethod
    def max_energy(self):
        pass

    @abstractmethod
    def define_local_constraints(self):
        pass


INVERT_CA_POWER = {PowerLoadNode, P2GNode, P2HNode}
INVERT_CA_GAS_FLOW = {CHPNode, G2PNode, SinkNode}


def calc_power_balance(local_model):
    if type(local_model) in INVERT_CA_POWER:
        return -local_model.active_power()
    return local_model.active_power()


def calc_gas_balance(local_model):
    if type(local_model) in INVERT_CA_GAS_FLOW:
        return -local_model.mdot_kg_per_s()
    return local_model.mdot_kg_per_s()


class PowerCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy(self._local_model.active_power_capability())

    def define_local_constraints(self):
        return lambda r: power_constraint(
            pn.get_bus_junc_res_data(self._local_model), r
        )

    def calc_balance(self):
        model = self._local_model
        return to_multi_energy(power=calc_power_balance(model))


class GasCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy(gas=self._local_model.mdot_kg_per_s_capability())

    def define_local_constraints(self):
        return lambda r: gas_constraint(pn.get_bus_junc_res_data(self._local_model), r)

    def calc_balance(self):
        return to_multi_energy(gas=calc_gas_balance(self._local_model))


class DummyHeatCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy()

    def define_local_constraints(self):
        return lambda _: 0

    def calc_balance(self):
        return to_multi_energy(heat=0)


class HeatCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy(self._local_model.q_capability())

    def define_local_constraints(self):
        return lambda r: heat_constraint(pn.get_bus_junc_res_data(self._local_model), r)

    def calc_balance(self):
        return to_multi_energy(heat=self._local_model.qext_w())


class PowerGasCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy(
            power=self._local_model.active_power_capability(),
            gas=self._local_model.mdot_kg_per_s_capability(),
        )

    def define_local_constraints(self):
        data = pn.get_bus_junc_res_data(self._local_model)
        return lambda r: gas_constraint(data, r) and power_constraint(data, r)

    def calc_balance(self):

        return to_multi_energy(
            power=calc_power_balance(self._local_model),
            gas=calc_gas_balance(self._local_model),
        )


class PowerGasHeatCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy(
            power=self._local_model.active_power_capability(),
            gas=self._local_model.mdot_kg_per_s_capability(),
            heat=self._local_model.q_capability(),
        )

    def define_local_constraints(self):
        data = pn.get_bus_junc_res_data(self._local_model)
        return (
            lambda r: gas_constraint(data, r)
            and power_constraint(data, r)
            and heat_constraint(data, r)
        )

    def calc_balance(self):
        return to_multi_energy(
            power=calc_power_balance(self._local_model),
            heat=self._local_model.qext_w(),
            gas=calc_gas_balance(self._local_model),
        )


class PowerHeatCA(CommonNetworkComponentAccess):
    def max_energy(self):
        return to_multi_energy(
            power=self._local_model.active_power_capability(),
            heat=self._local_model.q_capability(),
        )

    def define_local_constraints(self):
        data = pn.get_bus_junc_res_data(self._local_model)
        return lambda r: power_constraint(data, r) and heat_constraint(data, r)

    def calc_balance(self):
        return to_multi_energy(
            power=calc_power_balance(self._local_model),
            heat=self._local_model.qext_w(),
        )
