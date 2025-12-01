# simulation scenario for RES

import cmres.simulation.scenario.resilience as ssr
import cmres.simulation.profiles as ssp
from cmres.resilience.core import ResilienceMetric, ResilienceModel, RepairModel

from monee import Network, TimeseriesData

TIME_STEPS = 96
RES_SIM_NAME = "RESSIM"


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
    return ssr.start_resilience_simulation(
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
