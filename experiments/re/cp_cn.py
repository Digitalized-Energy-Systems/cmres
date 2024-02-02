import sys
import numpy as np
import random
import pickle
from pathlib import Path

from cmres.resilience.metric import SimpleResilienceMetric
from cmres.resilience.model import SimpleRepairModel, SimpleResilienceModel
from cmres.simulation.scenarios import start_res_simulation
from cmres.resilience.stopping import createMGBMStoppingCriterion

import monee.network as mn
import monee.model as mm
from monee.io.from_simbench import obtain_simbench_net
from monee import TimeseriesData

EXPERIMENT_NAME = "data/res/MoneeResilienceExperiment"
SEED = 101
SEED2 = 151
TIME_STEPS = 4 * 8
REPAIR_DELAY = 5
INCIDENT_TIME_STEPS = 3
INCIDENT_SHIFT = 0


def create_common_network(simbench_id, cp_density_coeff):
    net_simbench = obtain_simbench_net(simbench_id)
    for child in net_simbench.childs_by_type(mm.PowerGenerator):
        child.model.p_mw = child.model.p_mw * 4
    return mn.generate_mes_based_on_power_net(
        net_simbench,
        heat_deployment_rate=1,
        gas_deployment_rate=1,
        p2g_density=0.1 * cp_density_coeff,
        p2h_density=0.2 * cp_density_coeff,
        chp_density=0.2 * cp_density_coeff,
    )


def start_test_sim(
    simbench_id,
    power_impact=1,
    heat_impact=1,
    gas_impact=1,
    mes_impact=1,
    cp_density=1,
    process_id=0,
):
    seed = SEED
    if cp_density == 1:
        seed = SEED2
    np.random.seed(seed)
    random.seed(seed)

    net = create_common_network(simbench_id, cp_density)

    """
    td = (
        obtain_simbench_profile(simbench_id)
        + ssp.create_usa_gas_profiles_td(net, time_steps=TIME_STEPS)
        + ssp.attach_usa_heat_profiles_td(net, time_steps=TIME_STEPS)
    )
    """
    out_name = f"{EXPERIMENT_NAME}-{power_impact}-{heat_impact}-{gas_impact}-{mes_impact}-{cp_density}"
    out_path = Path(out_name)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / Path("network.p")).open("wb") as fp:
        pickle.dump(net, fp)

    # reset seed for more variety when used over multiple processes for more samples
    np.random.seed()
    random.seed()

    num_iter = 2000
    i = 0
    mgbm = createMGBMStoppingCriterion(0.1, 1, 100)
    perf_list = []
    while True:
        perf = start_res_simulation(
            net,
            TimeseriesData(),
            resilience_model=SimpleResilienceModel(
                incident_shift=INCIDENT_SHIFT,
                incident_timesteps=INCIDENT_TIME_STEPS,
                power_impact=power_impact,
                heat_impact=heat_impact,
                gas_impact=gas_impact,
                mes_impact=mes_impact,
            ),
            repair_model=SimpleRepairModel(
                delay_for_repair=REPAIR_DELAY, incident_timesteps=INCIDENT_TIME_STEPS
            ),
            resilience_measurement_model=SimpleResilienceMetric(),
            time_steps=TIME_STEPS,
            name=f"{cp_density}-{EXPERIMENT_NAME}-{power_impact}-{heat_impact}-{gas_impact}-{mes_impact}-{i}-{cp_density}",
            out_name=out_name,
            id=process_id * num_iter + i,
        )
        perf_list.append(perf)
        i += 1
        if i >= num_iter and mgbm.stop(perf_list):
            break


NUM_TO_EXP_MAP = [
    (0, 0, 3, 0),
    (3, 0, 0, 0),
    (0, 3, 0, 0),
    (0, 0, 0, 3),
    (2, 2, 2, 2),
    (1, 1, 1, 1),
]
DENSITY_MAP = [0, 0.5, 1, 1.5, 2]

if __name__ == "__main__":
    num = 1
    if len(sys.argv) > 1:
        num = int(sys.argv[1])

    impacts = NUM_TO_EXP_MAP[(num - 1) % (len(NUM_TO_EXP_MAP))]
    density = DENSITY_MAP[((num - 1) // len(NUM_TO_EXP_MAP)) % len(DENSITY_MAP)]
    start_test_sim(
        "1-LV-urban6--2-no_sw",
        power_impact=impacts[0],
        gas_impact=impacts[1],
        heat_impact=impacts[2],
        mes_impact=impacts[3],
        cp_density=density,
        process_id=num - 1,
    )
    """

    np.random.seed(SEED)
    random.seed(SEED)
    net = create_common_network("1-LV-urban6--2-no_sw", 1, 1)

    out_name = f"{EXPERIMENT_NAME}-{1}-{1}-{1}-{1}"
    out_path = Path(out_name)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / Path("network.p")).open("wb") as fp:
        pickle.dump(net, fp)
    """
