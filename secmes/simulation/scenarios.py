# simulation scenario for DAT

from peext.node import CHPNode, P2GNode, G2PNode, P2HNode

import secmes.simulation.scenario.cell_agents as ssc
import secmes.simulation.scenario.resilience as ssr
import secmes.simulation.profiles as ssp
from secmes.resilience.core import ResilienceMetric, ResilienceModel, RepairModel

from secmes.agent.dat import (
    DATCouplingPointRole,
    DynamicCoalitionAdaptionTopologyAgent,
    SplittingStrategy,
)
from monee import Network, TimeseriesData

CP_CHANGE_PROB = 0.2
TIME_STEPS = 96
DAT_SIM_NAME = "DATSIM"
RES_SIM_NAME = "RESSIM"

COUPLING_POINTS = [CHPNode, P2GNode, G2PNode, P2HNode]


# calc electr distance
# import pandapower.pd2ppc as ppppc
# from pandapower.pypower.makeYbus import makeYbus

# ppc, ppci = ppppc._pd2ppc(test_n)
# Ybus = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])[0]
# np.real(makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])[0].todense())
#
# mapping pandapower bus id to ybusindex
# imap = net["_pd2ppc_lookups"]['bus']
# ->
# Resistance between bus i and j:
# rij = Ybus[imap[i], imap[j]]
#
# calc_rij_for_all_add_as_weight
# edge_weights = nx.distance_resistance(g)


def create_initiator(prob: float, strategy: SplittingStrategy):
    def add_roles_initiator(nc):
        if type(nc) in COUPLING_POINTS:
            return [DATCouplingPointRole(nc, prob, strategy)]
        return []

    return add_roles_initiator


def start_dat_simulation(
    multinet,
    demand_attacher_function=ssp.create_and_attach_random_profiles_all_demands_mn,
    cp_change_prob=CP_CHANGE_PROB,
    splitting_strategy=SplittingStrategy.DISINTEGRATE,
    time_steps=TIME_STEPS,
    name=DAT_SIM_NAME,
    no_energy_flow=False,
    port_add=0,
):
    if demand_attacher_function is not None:
        demand_attacher_function(multinet, time_steps)
    ssc.start_cell_simulation(
        multinet,
        create_initiator(cp_change_prob, splitting_strategy),
        agent_instantiator=lambda nc: DynamicCoalitionAdaptionTopologyAgent(
            nc, ssc.get_agent_type(nc)(nc)
        ),
        time_steps=time_steps,
        name=name,
        no_energy_flow=no_energy_flow,
        port_add=port_add,
    )


def start_res_simulation(
    net: Network,
    timeseries_data: TimeseriesData,
    resilience_model: ResilienceModel,
    repair_model: RepairModel,
    resilience_measurement_model: ResilienceMetric,
    time_steps=TIME_STEPS,
    name=RES_SIM_NAME,
    out_name=RES_SIM_NAME,
    id=0,
):
    ssr.start_resilience_simulation(
        net,
        timeseries_data,
        resilience_model,
        repair_model,
        resilience_measurement_model,
        time_steps=time_steps,
        name=name,
        out_name=out_name,
        id=id,
    )
