from typing import Dict
import os
import pickle
from pathlib import PurePath, Path
from statistics import mean

import cmres.evaluation.evaluation as eval
from monee import Network, run_energy_flow
from monee.model.core import Node

import pandas
import networkx as nx

OUTPUT = "data/out/res"
SMALL_NUMBER = 0.00000000001

TYPE_TO_CARRIER = {
    "Junction": "heat/gas",
    "Bus": "electricity",
    "CHP": "multi",
    "GasPipe": "gas",
    "GenericPowerBranch": "electricity",
    "PowerGenerator": "electricity",
    "PowerToGas": "multi",
    "PowerToHeat": "multi",
    "WaterPipe": "heat",
    "PowerToHeatControlNode": "multi",
    "CHPControlNode": "multi",
    "PowerLoad": "electricity",
    "Sink": "gas",
    "Source": "gas",
    "ExtPowerGrid": "electricity",
    "ExtHydrGrid": "gas",
    "GenericTransferBranch": "multi",
    "HeatExchangerLoad": "heat",
    "HeatExchanger": "heat",
}

EXPERIMENT_ID_TO_STR = {
    "0-0-0-3": "high cp",
    "0-0-3-0": "high gas",
    "0-3-0-0": "high heating",
    "3-0-0-0": "high electricity",
    "1-1-1-1": "low overall",
    "2-2-2-2": "medium overall",
}


def get_id_type(id_str: str):
    if "CHP" in id_str or "PowerToHeat" in id_str:
        return "compound"
    if ".child." in id_str:
        return "child"
    if id_str.endswith(")"):
        return "branch"
    return "node"


def extend_impact_df(cpd_to_net: Dict[float, Network], metrics_df, impact_df):
    new_impact_df = pandas.DataFrame()
    for _, row in metrics_df[
        metrics_df.apply(lambda row: row["id"].startswith("node"), axis=1)
    ].iterrows():
        node_id = int(row["id"].split(":")[1])
        node = cpd_to_net[row["cp_density"]].node_by_id(node_id)
        child_impact_df = impact_df[
            impact_df.apply(
                lambda row_impact: row_impact["id"].startswith("child")
                and int(row_impact["id"].split(":")[1]) in node.child_ids
                and row_impact["cp_density"] == row["cp_density"],
                axis=1,
            )
        ]

        node_impacts_df = (
            child_impact_df.groupby(["carrier", "cp_density"]).mean().reset_index()
        )
        node_impacts_df["id"] = row["id"]
        node_impacts_df["type"] = type(node.model)
        new_impact_df = pandas.concat([new_impact_df, node_impacts_df])
    return pandas.concat([impact_df, new_impact_df])


def create_impact_df(perf_df: pandas.DataFrame, repair_df, fail_df, metrics_df):
    global_rows = []
    perf_df = perf_df.rename(columns={"Unnamed: 0": "step"}).dropna()
    usable_fail_df = fail_df
    usable_repair_df = repair_df
    usable_fail_df["node"] = usable_fail_df["node"].apply(
        lambda c: get_id_type(c) + ":" + c.split(":")[-1]
    )
    usable_repair_df["node"] = usable_repair_df["node"].apply(
        lambda c: get_id_type(c) + ":" + c.split(":")[-1]
    )
    # knoten/edge resilience in zeiträumen in denen aktiv / resilience
    # in zeiträumen in denen inaktiv -> impact/influence
    for _, row in metrics_df.iterrows():
        node_edge_id = row["id"]
        type = row["type"]

        node_usable_fail_df = usable_fail_df.query(f"node == '{node_edge_id}'")
        node_usable_repair_df = usable_repair_df.query(f"node == '{node_edge_id}'")

        concated_fail_node_df = pandas.concat(
            [node_usable_fail_df, node_usable_repair_df]
        )

        rules = {}
        cp_density = row["cp_density"]
        concated_fail_node_df_with_cpd = concated_fail_node_df[
            concated_fail_node_df["cp_density"] == cp_density
        ]
        perf_df_with_cpd = perf_df[perf_df["cp_density"] == cp_density]
        for name, group in concated_fail_node_df_with_cpd.groupby(["experiment", "id"]):
            if len(group) != 2:
                continue
            failure_row = group.query("type == 'failure'")
            repair_row = group.query("type == 'repair'")
            if len(repair_row["step"]) == 0 or len(failure_row["step"]) == 0:
                continue
            rules[(name[1], name[0])] = (
                failure_row["step"].iloc[0],
                repair_row["step"].iloc[0],
            )

        node_in_perf_df = perf_df_with_cpd[
            perf_df_with_cpd.apply(
                lambda row: (row["id"], row["experiment"]) in rules
                and row["step"] >= rules[(row["id"], row["experiment"])][0]
                and row["step"] <= rules[(row["id"], row["experiment"])][1],
                axis=1,
            )
        ]
        node_not_in_perf_df = perf_df_with_cpd[
            perf_df_with_cpd.apply(
                lambda row: (row["id"], row["experiment"]) not in rules
                or row["step"] < rules[(row["id"], row["experiment"])][0]
                or row["step"] > rules[(row["id"], row["experiment"])][1],
                axis=1,
            )
        ]

        node_in_perf_df_mean = node_in_perf_df[["0", "1", "2"]].mean()
        node_not_in_perf_df_mean = node_not_in_perf_df[["0", "1", "2"]].mean()
        for carrier in [("heat", "1"), ("gas", "2"), ("electricity", "0")]:
            impact = (
                (node_in_perf_df_mean[carrier[1]] + SMALL_NUMBER)
                / (node_not_in_perf_df_mean[carrier[1]] + SMALL_NUMBER)
                if node_in_perf_df_mean[carrier[1]]
                > node_not_in_perf_df_mean[carrier[1]]
                else -(node_not_in_perf_df_mean[carrier[1]] + SMALL_NUMBER)
                / (node_in_perf_df_mean[carrier[1]] + SMALL_NUMBER)
            )
            global_rows.append(
                {
                    "id": node_edge_id,
                    "carrier": carrier[0],
                    "impact": impact,
                    "type": type,
                    "cp_density": cp_density,
                }
            )
    return pandas.DataFrame(global_rows)


def create_or_load_impact_df(fail_df, perf_df, repair_df, metrics_df, folder_id):
    impact_out = OUTPUT + f"/{folder_id}/impact.csv"
    if Path(impact_out).exists():
        return pandas.read_csv(impact_out)
    impact_df = create_impact_df(perf_df, repair_df, fail_df, metrics_df)
    impact_df.to_csv(Path(impact_out))
    return impact_df


def create_metrics_df(monee_net: Network, cp_density: float):
    monee_net = run_energy_flow(monee_net).network

    # id, type, metric... (betweenness_centrality)
    for edge in monee_net._network_internal.edges:
        branch_model = monee_net.branch_by_id(edge).model
        monee_net._network_internal.edges[edge]["weight"] = branch_model.loss_percent()

    node_to_bc = nx.betweenness_centrality(monee_net._network_internal, weight="weight")
    edge_to_bc = nx.edge_betweenness_centrality(
        monee_net._network_internal, weight="weight"
    )
    node_to_degree = nx.degree(monee_net._network_internal)
    edge_to_degree = {}
    for edge_id, _ in edge_to_bc.items():
        edge_to_degree[edge_id] = (
            node_to_degree[edge_id[0]] + node_to_degree[edge_id[1]]
        )
    node_to_vital = nx.closeness_vitality(monee_net._network_internal, weight="weight")
    edge_to_vital = {}
    for edge_id, _ in edge_to_bc.items():
        edge_to_vital[edge_id] = (
            node_to_vital[edge_id[0]] + node_to_vital[edge_id[1]]
        ) / 2
    node_to_katz = nx.katz_centrality(
        nx.Graph(monee_net._network_internal), weight="weight"
    )
    edge_to_katz = {}
    for edge_id, _ in edge_to_bc.items():
        edge_to_katz[edge_id] = (
            node_to_katz[edge_id[0]] + node_to_katz[edge_id[1]]
        ) / 2
    all_rows = []
    for node_id, cb in node_to_bc.items():
        all_rows.append(
            {
                "id": f"node:{node_id}",
                "type": type(monee_net.node_by_id(node_id).model),
                "betweenness_centrality": cb,
                "degree": node_to_degree[node_id],
                "vc": node_to_vital[node_id],
                "katz": node_to_katz[node_id],
                "cp_density": cp_density,
            }
        )
    for child in monee_net.childs:
        bc = node_to_bc[child.node_id]
        degree = node_to_degree[child.node_id]
        vc = node_to_vital[child.node_id]
        katz = node_to_katz[child.node_id]
        all_rows.append(
            {
                "id": f"child:{child.id}",
                "type": type(monee_net.child_by_id(child.id).model),
                "betweenness_centrality": bc,
                "degree": degree,
                "vc": vc,
                "katz": katz,
                "cp_density": cp_density,
            }
        )
    for compound in monee_net.compounds:
        bc = nx.group_betweenness_centrality(
            monee_net._network_internal,
            [comp.id for comp in compound.component_of_type(Node)],
            weight="weight",
        )
        degree = sum(
            [node_to_degree[node_id] for node_id in compound.connected_to.values()]
        )
        vc = mean(
            [node_to_vital[node_id] for node_id in compound.connected_to.values()]
        )
        katz = mean(
            [node_to_katz[node_id] for node_id in compound.connected_to.values()]
        )
        all_rows.append(
            {
                "id": f"compound:{compound.id}",
                "type": type(monee_net.compound_by_id(compound.id).model),
                "betweenness_centrality": bc,
                "degree": degree,
                "vc": vc,
                "katz": katz,
                "cp_density": cp_density,
            }
        )
    for edge_id, cb in edge_to_bc.items():
        all_rows.append(
            {
                "id": f"branch:{edge_id}",
                "type": type(monee_net.branch_by_id(edge_id).model),
                "betweenness_centrality": cb,
                "degree": edge_to_degree[edge_id],
                "vc": edge_to_vital[edge_id],
                "katz": edge_to_katz[edge_id],
                "cp_density": cp_density,
            }
        )
    return pandas.DataFrame(all_rows)


def create_full_metrics_df(key_to_net):
    dfs = []
    for cpd, net in key_to_net.items():
        dfs.append(create_metrics_df(net, cpd))
    return pandas.concat(dfs)


def create_or_load_metrics_df(key_to_net, folder_id):
    metrics_out = OUTPUT + f"/{folder_id}/metrics.csv"
    if Path(metrics_out).exists():
        return pandas.read_csv(metrics_out)
    metrics_df = create_full_metrics_df(key_to_net)
    Path(OUTPUT + f"/{folder_id}").mkdir(exist_ok=True, parents=True)
    metrics_df.to_csv(Path(metrics_out))
    return metrics_df


def append_desc_df(
    single_df, identifier, power_impact, heat_impact, gas_impact, mes_impact, cp_density
):
    single_df["experiment"] = identifier
    single_df["power_impact"] = power_impact
    single_df["heat_impact"] = heat_impact
    single_df["gas_impact"] = gas_impact
    single_df["mes_impact"] = mes_impact
    single_df["cp_density"] = cp_density


def load_dfs(folder_id):
    all_folders = [f.path for f in os.scandir(folder_id) if f.is_dir()]

    failure_dfs = []
    performance_dfs = []
    repair_dfs = []

    cpd_to_net = {}

    for experiment_desc in all_folders:
        experiment_desc_name = PurePath(experiment_desc).name
        experiment_attributes = experiment_desc_name.split("-")
        power_impact = experiment_attributes[1]
        heat_impact = experiment_attributes[2]
        gas_impact = experiment_attributes[3]
        mes_impact = experiment_attributes[4]
        cp_density = float(experiment_attributes[5])

        if not cp_density in cpd_to_net:
            with open(Path(experiment_desc) / Path("network.p"), "rb") as network_file:
                monee_net = pickle.load(network_file)
                print(monee_net.statistics())
                cpd_to_net[cp_density] = monee_net

        # failure
        failure_path = Path(experiment_desc) / Path("failure.csv")
        if failure_path.exists():
            failure_df = pandas.read_csv(failure_path)
            append_desc_df(
                failure_df,
                experiment_desc,
                power_impact,
                heat_impact,
                gas_impact,
                mes_impact,
                cp_density,
            )
            failure_dfs.append(failure_df)

        # performance
        performance_df = pandas.read_csv(
            Path(experiment_desc) / Path("performance.csv")
        )
        append_desc_df(
            performance_df,
            experiment_desc,
            power_impact,
            heat_impact,
            gas_impact,
            mes_impact,
            cp_density,
        )
        performance_dfs.append(performance_df)

        # repair
        repair_path = Path(experiment_desc) / Path("repair.csv")
        if repair_path.exists():
            repair_df = pandas.read_csv(repair_path)
            append_desc_df(
                repair_df,
                experiment_desc,
                power_impact,
                heat_impact,
                gas_impact,
                mes_impact,
                cp_density,
            )
            repair_dfs.append(repair_df)

    return (
        pandas.concat(failure_dfs),
        pandas.concat(performance_dfs),
        pandas.concat(repair_dfs),
        create_or_load_metrics_df(cpd_to_net, folder_id),
        cpd_to_net,
    )


COLUMN_TIMESTEP = "Unnamed: 0"
COLUMN_EL = "0"
COLUMN_HEAT = "1"
COLUMN_GAS = "2"
COLUMN_ID = "id"
COLUMN_EL_IMP = "power_impact"
COLUMN_HEAT_IMP = "heat_impact"
COLUMN_GAS_IMP = "gas_impact"
COLUMN_EXPERIMENT_NAME = "experiment"

CARRIER_REPLACE_MAP = {"0": "electricity", "1": "heat", "2": "gas"}


def resilience_per_scenario(perf_df: pandas.DataFrame, folder_id):
    # experiment, id 0 1 2
    resilience_per_carrier_per_scenario = (
        perf_df.groupby(["cp_density", "experiment", "id"])[["0", "1", "2"]]
        .sum()
        .reset_index()
        .groupby(["cp_density", "experiment"])
        .mean()
        .reset_index()
        .melt(
            id_vars=["cp_density", "experiment", "id"],
            var_name="carrier",
            value_name="resilience_mean",
        )
    )
    resilience_per_carrier_per_scenario[
        "experiment"
    ] = resilience_per_carrier_per_scenario["experiment"].apply(
        lambda v: "-".join(v.split("/")[2].split("-")[1:-1])
    )
    resilience_per_carrier_per_scenario[
        "carrier"
    ] = resilience_per_carrier_per_scenario["carrier"].apply(
        lambda v: CARRIER_REPLACE_MAP[v]
    )
    resilience_per_carrier_per_scenario_hist = eval.create_bar(
        resilience_per_carrier_per_scenario,
        x_label="experiment",
        y_label="resilience_mean",
        color="carrier",
        color_discrete_map=eval.NETWORK_COLOR_MAP,
        pattern_shape_map=eval.NETWORK_PATTERN_MAP,
        legend_text="carrier",
        template="plotly_white+publish3",
        yaxis_title="mean performance loss in MW",
        xaxis_title="scenario",
        title="Performance drop by scenario, by carrier",
        barmode="group",
        width=1200,
        height=450,
    )
    unique_cp_densitites = pandas.unique(
        resilience_per_carrier_per_scenario["cp_density"]
    )
    unique_cp_densitites.sort()
    unique_experiments = list(
        pandas.unique(resilience_per_carrier_per_scenario["experiment"])
    )

    resilience_per_carrier_per_scenario_hist_2 = (
        eval.create_multilevel_grouped_bar_chart(
            [
                list(
                    resilience_per_carrier_per_scenario[
                        resilience_per_carrier_per_scenario["carrier"] == carrier
                    ].sort_values(by=["cp_density", "experiment"])["resilience_mean"]
                )
                for carrier in ["electricity", "heat", "gas"]
            ],
            ["#ffa000", "#d32f2f", "#388e3c"],
            ["electricity", "heat", "gas"],
            [f"<b>{cpd}</b>" for cpd in unique_cp_densitites],
            len(unique_experiments),
            [f"{EXPERIMENT_ID_TO_STR[exp]}" for exp in unique_experiments]
            * len(unique_cp_densitites),
            yaxis_title="<b>mean performance loss in MW</b>",
            multi_level_distance=-0.4,
        )
    )

    eval.write_all_in_one(
        [
            resilience_per_carrier_per_scenario_hist_2,
        ],
        "Figure",
        Path("."),
        OUTPUT + f"/{folder_id}/resilience_per_carrier_per_scenario.html",
        titles=["Performance drop by scenario, by carrier grouped by cp density"],
    )


TYPE_SPECIALS_CN = {
    "branches": ["GenericPowerBranch", "WaterPipe", "GasPipe"],
    "nodes": ["PowerGenerator", "Source", "HeatExchanger"],
    "CPs": ["PowerToHeat", "CHP", "PowerToGas"],
    "power lines": ["GenericPowerBranch"],
    "gas pipes": ["GasPipe"],
    "water pipes": ["WaterPipe"],
}


def impact_over_metrics(
    cpd_to_net: Dict[float, Network],
    impact_df,
    metrics_df,
    folder_id,
    metric_ids,
):
    metric_impact_df: pandas.DataFrame = impact_df.astype({"id": "string"}).merge(
        metrics_df.astype({"id": "string"}), on=["id", "cp_density"]
    )
    metric_impact_df["type_y"] = (
        metric_impact_df["type_y"].astype(str).apply(lambda v: v.split(".")[-1][:-2])
    )
    metric_impact_df["impact"] = metric_impact_df["impact"].apply(lambda v: abs(v))
    figures = []
    titles = []
    metric_impact_df = metric_impact_df[metric_impact_df["impact"].notnull()]
    for carrier in pandas.unique(metric_impact_df["carrier"]):
        metric_impact_df_carrier = metric_impact_df[
            metric_impact_df["carrier"] == carrier
        ]
        for metric in metric_ids:
            # carrier_name = carrier.split("'")[1]
            carrier_name = carrier
            figures.append(
                eval.create_scatter_with_df(
                    metric_impact_df_carrier,
                    metric,
                    "impact",
                    color_label="type_y",
                    yaxis_title=f"{carrier_name}-impact",
                    xaxis_title=metric,
                    legend_text="type",
                )
            )
            titles.append(f"{metric} to the components' {carrier_name}-impact")
            for key, value in TYPE_SPECIALS_CN.items():
                metric_impact_df_carrier_with_types = metric_impact_df_carrier.query(
                    f"type_y in {value}"
                )
                figures.append(
                    eval.create_scatter_with_df(
                        metric_impact_df_carrier_with_types,
                        metric,
                        "impact",
                        color_label="type_y",
                        yaxis_title=f"{carrier_name}-impact",
                        xaxis_title=metric,
                        legend_text="type",
                    )
                )
                titles.append(f"{metric} to the {key}' {carrier_name}-impact")

        for cpd, monee_net in cpd_to_net.items():
            metric_impact_df_carrier_cpd = metric_impact_df_carrier[
                metric_impact_df_carrier["cp_density"] == cpd
            ]
            figures.append(
                eval.create_networkx_plot(
                    monee_net,
                    metric_impact_df_carrier_cpd,
                    color_name="impact",
                    color_legend_text=f"{carrier_name}-impact",
                    template="plotly_white+publish",
                )
            )
            titles.append(f"graph of the components' {carrier_name}-impact ({cpd})")
            figures.append(
                eval.create_networkx_plot(
                    monee_net,
                    metric_impact_df_carrier_cpd,
                    color_name="impact",
                    color_legend_text=f"{carrier_name}-impact",
                    template="plotly_white+publish",
                    without_nodes=True,
                )
            )
            titles.append(
                f"edge-graph of the components' {carrier_name}-impact ({cpd})"
            )

    # aggregated all carrier impacts
    metric_impact_df_all_carrier = (
        metric_impact_df.groupby(["type_y", "cp_density", "id"] + metric_ids)
        .sum()
        .reset_index()
    )
    for metric in metric_ids:
        figures.append(
            eval.create_scatter_with_df(
                metric_impact_df_all_carrier,
                metric,
                "impact",
                color_label="type_y",
                yaxis_title=f"impact",
                xaxis_title=metric,
                legend_text="type",
            )
        )
        titles.append(f"{metric} to the components' impact")
        for key, value in TYPE_SPECIALS_CN.items():
            metric_impact_df_carrier_with_types = metric_impact_df_all_carrier.query(
                f"type_y in {value}"
            )
            figures.append(
                eval.create_scatter_with_df(
                    metric_impact_df_carrier_with_types,
                    metric,
                    "impact",
                    color_label="type_y",
                    yaxis_title=f"impact",
                    xaxis_title=metric,
                    legend_text="type",
                )
            )
            titles.append(f"{metric} to the {key}' {carrier_name}-impact")

    for cpd, monee_net in cpd_to_net.items():
        metric_impact_df_carrier_cpd = metric_impact_df_all_carrier[
            metric_impact_df_all_carrier["cp_density"] == cpd
        ]
        figures.append(
            eval.create_networkx_plot(
                monee_net,
                metric_impact_df_carrier_cpd,
                color_name="impact",
                color_legend_text=f"impact",
                template="plotly_white+publish",
            )
        )
        titles.append(f"graph of the components' impact ({cpd})")
        figures.append(
            eval.create_networkx_plot(
                monee_net,
                metric_impact_df_carrier_cpd,
                color_name="impact",
                color_legend_text=f"impact",
                template="plotly_white+publish",
                without_nodes=True,
            )
        )
        titles.append(f"edge-graph of the components' impact ({cpd})")

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{folder_id}/metric_to_impact.html",
        titles=titles,
    )


def impact_aggregated_component_carrier(impact_df: pandas.DataFrame, folder_id):
    new_impact_df = impact_df.copy()
    new_impact_df["impact"] = new_impact_df["impact"].apply(lambda v: abs(v))
    new_impact_df["type"] = (
        impact_df["type"].astype(str).apply(lambda v: v.split(".")[-1][:-2])
    )
    new_impact_df["type_carrier"] = new_impact_df.apply(
        lambda row: TYPE_TO_CARRIER[row["type"]], axis=1
    )
    new_impact_df = new_impact_df[new_impact_df["impact"].notnull()]
    """
    new_impact_df["carrier"] = (
        new_impact_df["carrier"].astype(str).apply(lambda v: v.split("'")[1])
    )
    """
    average_impact_per_carrier = (
        new_impact_df.groupby(["type_carrier", "carrier"]).mean().reset_index()
    )
    average_impact_per_component = (
        new_impact_df.groupby(["type", "carrier"]).mean().reset_index()
    )
    impact_per_carrier = (
        new_impact_df.groupby(["type_carrier", "carrier"]).sum().reset_index()
    )
    impact_per_component = (
        new_impact_df.groupby(["type", "carrier"]).sum().reset_index()
    )
    figures = []
    titles = []
    # component type by carrier impacts
    figures += [
        eval.create_bar(
            average_impact_per_component,
            x_label="type",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="type",
            showlegend=False,
        )
    ]
    titles.append(f"Average impacts by component type")
    figures += [
        eval.create_bar(
            impact_per_component,
            x_label="type",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="type",
            showlegend=False,
        )
    ]
    titles.append(f"Total impacts by component type")
    # carrier type with carrier impacts
    figures += [
        eval.create_bar(
            average_impact_per_carrier,
            x_label="type_carrier",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="carrier",
            showlegend=False,
        )
    ]
    titles.append(f"Average impacts by carrier type")
    figures += [
        eval.create_bar(
            impact_per_carrier,
            x_label="type_carrier",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="carrier",
            showlegend=False,
        )
    ]
    titles.append(f"Total impacts by carrier type")

    average_impact_per_carrier_cpd = (
        new_impact_df.groupby(["type_carrier", "carrier", "cp_density"])
        .mean()
        .reset_index()
    )
    average_impact_per_carrier_cpd["carrier_cpd"] = (
        average_impact_per_carrier_cpd["type_carrier"].astype(str)
        + "-"
        + average_impact_per_carrier_cpd["cp_density"].astype(str)
    )
    impact_per_carrier_cpd = (
        new_impact_df.groupby(["type_carrier", "carrier", "cp_density"])
        .sum()
        .reset_index()
    )
    impact_per_carrier_cpd["carrier_cpd"] = (
        impact_per_carrier_cpd["type_carrier"].astype(str)
        + "-"
        + impact_per_carrier_cpd["cp_density"].astype(str)
    )
    figures += [
        eval.create_bar(
            average_impact_per_carrier_cpd,
            x_label="carrier_cpd",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="carrier-density",
            showlegend=False,
        )
    ]
    titles.append(f"Average impacts by carrier type and density")
    figures += [
        eval.create_bar(
            impact_per_carrier_cpd,
            x_label="carrier_cpd",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="carrier-density",
            showlegend=False,
        )
    ]
    titles.append(f"Total impacts by carrier type and density")

    average_impact_per_cpd = (
        new_impact_df.groupby(["carrier", "cp_density"]).mean().reset_index()
    )
    average_impact_per_cpd["cp_density"] = average_impact_per_cpd["cp_density"].astype(
        str
    )
    total_impact_per_cpd = (
        new_impact_df.groupby(["carrier", "cp_density"]).sum().reset_index()
    )
    total_impact_per_cpd["cp_density"] = total_impact_per_cpd["cp_density"].astype(str)

    figures += [
        eval.create_bar(
            average_impact_per_cpd,
            x_label="cp_density",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="density",
            showlegend=False,
        )
    ]
    titles.append(f"Average impacts by density")
    figures += [
        eval.create_bar(
            total_impact_per_cpd,
            x_label="cp_density",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template="plotly_white+publish3",
            yaxis_title="impact",
            xaxis_title="density",
            showlegend=False,
        )
    ]
    titles.append(f"Total impacts by density")

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{folder_id}/impact_aggregated_component_carrier.html",
        titles=titles,
    )


def evaluate(folder_id):
    fail_df, perf_df, repair_df, metrics_df, cpd_to_net = load_dfs(folder_id)
    impact_df = create_or_load_impact_df(
        fail_df, perf_df, repair_df, metrics_df, folder_id
    )
    resilience_per_scenario(perf_df, folder_id)
    impact_df = extend_impact_df(cpd_to_net, metrics_df, impact_df)
    impact_aggregated_component_carrier(impact_df, folder_id)
    impact_over_metrics(
        cpd_to_net,
        impact_df,
        metrics_df,
        folder_id,
        ["betweenness_centrality", "degree", "vc", "katz"],
    )


def main():
    evaluate("data/res_31_01_24")

if __name__ == "__main__":
    main()
