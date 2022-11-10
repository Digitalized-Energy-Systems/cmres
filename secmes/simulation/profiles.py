from pandapower.timeseries.data_sources.frame_data import DFData

import pandapower.control as powercontrol

import pandas as pd
import numpy as np
import simbench as sb
import glob
import pandas


def create_random_profile(
    element_len, time_steps, center=0, dev=0.1, only_positive=False
):
    random_load_matr = np.random.normal(center, dev, (time_steps, element_len))
    if only_positive:
        random_load_matr = abs(random_load_matr)
    return random_load_matr


def attach_load_profiles(net, profile, time_steps):
    attach_profiles(
        net, profile, time_steps, net.load.index, net.load.p_mw.values, "load", "p_mw"
    )


def attach_sink_profiles(net, profile, time_steps):
    attach_profiles(
        net,
        profile,
        time_steps,
        net.sink.index,
        net.sink.mdot_kg_per_s.values,
        "sink",
        "mdot_kg_per_s",
    )


def attach_he_profiles(net, profile, time_steps):
    attach_profiles(
        net,
        profile,
        time_steps,
        net.heat_exchanger.index,
        net.heat_exchanger.qext_w.values,
        "heat_exchanger",
        "qext_w",
    )


def attach_profiles(
    net, profile, time_steps, element_index, element_values, element_name, element_var
):
    df = (
        pd.DataFrame(profile, index=list(range(time_steps)), columns=element_index)
        * element_values
    )
    ds = DFData(df)
    powercontrol.ConstControl(
        net,
        element=element_name,
        element_index=element_index,
        variable=element_var,
        data_source=ds,
        profile_name=element_index,
    )


def create_and_attach_random_profiles_all_demands_mn(mn, time_steps):
    create_and_attach_random_profiles_all_demands(
        mn["nets"]["power"], mn["nets"]["heat"], mn["nets"]["gas"], time_steps
    )


def create_and_attach_random_profiles_load_mn(mn, time_steps):
    create_and_attach_random_profiles_all_demands(mn["nets"]["power"], time_steps)


def create_and_attach_random_profiles_load(power_net, time_steps):
    load_mat = create_random_profile(
        len(power_net.load.index), time_steps, only_positive=True
    )

    attach_load_profiles(power_net, load_mat, time_steps)


def attach_all_profiles_mn(mn, power_profile, heat_profile, gas_profile, time_steps):
    attach_load_profiles(mn["nets"]["power"], power_profile, time_steps)
    attach_he_profiles(mn["nets"]["heat"], heat_profile, time_steps)
    attach_sink_profiles(mn["nets"]["gas"], gas_profile, time_steps)


def create_and_attach_random_profiles_all_demands(
    power_net, heat_net, gas_net, time_steps
):
    load_mat = create_random_profile(
        len(power_net.load.index), time_steps, only_positive=True
    )
    he_mat = create_random_profile(
        len(heat_net.heat_exchanger.index), time_steps, only_positive=False
    )
    sink_mat = create_random_profile(
        len(gas_net.sink.index), time_steps, only_positive=True
    )

    attach_load_profiles(power_net, load_mat, time_steps)
    attach_he_profiles(heat_net, he_mat, time_steps)
    attach_sink_profiles(gas_net, sink_mat, time_steps)


def create_demand_mat_usa(column_names, element_len, time_steps, sub_column=None):
    path_to_data = "data/input/profiles/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/RESIDENTIAL_LOAD_DATA_E_PLUS_OUTPUT/BASE/"
    profile_names_csv = glob.glob(path_to_data + "*.csv")[:element_len]

    big_df = None
    column_names_full = list(
        set(column_names + ([] if sub_column is None else [sub_column]))
    )
    for profile_name in profile_names_csv:
        df = pd.read_csv(
            profile_name,
            names=column_names_full,
            nrows=time_steps / 4,
            skiprows=1,
        )
        df["demand"] = sum([df[cn] for cn in column_names])
        if sub_column is not None:
            df["demand"] = df["demand"] - df[sub_column]
        df = (
            df.drop(column_names_full, axis=1)
            .loc[df.index.repeat(4)]
            .reset_index(drop=True)
            .transpose()
        )
        if big_df is not None:
            big_df = pandas.concat([big_df, df]).reset_index(drop=True)
        else:
            big_df = df

    return big_df.to_numpy().transpose()


def create_usa_heat_profiles(heat_net, time_steps):
    return create_demand_mat_usa(
        ["Heating:Gas [kW](Hourly)", "Heating:Electricity [kW](Hourly)"],
        element_len=len(heat_net.heat_exchanger.index),
        time_steps=time_steps,
    )


def create_and_attach_usa_heat_profiles(heat_net, time_steps):
    load_mat = create_demand_mat_usa(
        ["Heating:Gas [kW](Hourly)", "Heating:Electricity [kW](Hourly)"],
        element_len=len(heat_net.heat_exchanger.index),
        time_steps=time_steps,
    )
    attach_he_profiles(heat_net, load_mat, time_steps)


def create_usa_gas_profiles(gas_net, time_steps):
    return create_demand_mat_usa(
        ["Gas:Facility [kW](Hourly)"],
        element_len=len(gas_net.sink.index),
        time_steps=time_steps,
    )


def create_and_attach_usa_gas_profiles(gas_net, time_steps):
    load_mat = create_demand_mat_usa(
        ["Gas:Facility [kW](Hourly)"],
        element_len=len(gas_net.sink.index),
        time_steps=time_steps,
    )
    attach_sink_profiles(gas_net, load_mat, time_steps)


def create_simbench_profiles(power_net):
    assert not sb.profiles_are_missing(power_net)
    return sb.get_absolute_values(power_net, profiles_instead_of_study_cases=True)


def create_and_attach_all_simbench_profiles(power_net):
    sb.apply_const_controllers(power_net, create_simbench_profiles(power_net))
