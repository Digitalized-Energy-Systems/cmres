# simulation scenario for RES

import cmres.simulation.scenario.resilience as ssr
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
    registry=None,
    scenario=None,
):
    """Run one resilience simulation.

    Parameters
    ----------
    registry : ComponentRegistry | None
        Canonical component ordering; required when *scenario* is provided.
    scenario : FailureScenario | None
        Pre-sampled uniform[0,1] inputs from an RQMC sampler.  When provided,
        all stochastic failure and repair decisions come from the scenario
        rather than the global RNG.

    Returns
    -------
    (performance_sum, carrier_sums)
        performance_sum : float  — total loss across all carriers and timesteps
        carrier_sums    : ndarray shape (3,) — [power_loss, heat_loss, gas_loss]
    """
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
        registry=registry,
        scenario=scenario,
    )
