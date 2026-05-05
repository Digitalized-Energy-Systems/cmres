from typing import Dict
import os
import sys
import pickle
import traceback
from pathlib import PurePath, Path
from statistics import mean

import cmres.evaluation.evaluation as eval
from monee import Network, run_energy_flow, PyomoSolver
from monee.model.core import Node
import scipy.stats

import pandas
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from cp_metric import mes_all_components_metric, CPMetricConfig

INPUT = "/home/rschrage/experiments/0503/res"
OUTPUT = "data/out"
SMALL_NUMBER = 0.00000000001

TYPE_TO_CARRIER = {
    "Junction": "heat/gas",
    "Bus": "electricity",
    "CHP": "multi",
    "GasPipe": "gas",
    "GenericPowerBranch": "electricity",
    "PowerLine": "electricity",
    "PowerGenerator": "electricity",
    "PowerToGas": "multi",
    "GasToPower": "multi",
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
    "SubHE": "heat",
}

def get_id_type(id_str: str):
    if "CHP" in id_str or "PowerToHeat" in id_str:
        return "compound"
    if ".child." in id_str:
        return "child"
    if id_str.endswith(")"):
        return "branch"
    return "node"


def extend_impact_df(net_type_to_net: Dict[str, Network], metrics_df, impact_df):
    impact_df = impact_df.reset_index(drop=True)
    impact_id_str = impact_df["id"].astype(str)
    is_child = impact_id_str.str.startswith("child")
    if not is_child.any():
        return impact_df

    child_impact_df = impact_df.loc[is_child, ["id", "carrier", "network_type", "impact"]].copy()
    child_impact_df["_child_id"] = (
        child_impact_df["id"].astype(str).str.split(":").str[1].astype(int)
    )

    metrics_id_str = metrics_df["id"].astype(str)
    node_metrics = metrics_df[metrics_id_str.str.startswith("node")]
    if node_metrics.empty:
        return impact_df

    pair_rows = []
    parent_type = {}
    for _, row in node_metrics.iterrows():
        nt = row["network_type"]
        try:
            node_id = int(str(row["id"]).split(":")[1])
            node = net_type_to_net[nt].node_by_id(node_id)
        except (KeyError, ValueError, IndexError):
            continue
        parent_type[row["id"]] = type(node.model)
        for cid in node.child_ids:
            pair_rows.append(
                {"_parent_id": row["id"], "network_type": nt, "_child_id": int(cid)}
            )
    if not pair_rows:
        return impact_df

    pairs_df = pandas.DataFrame(pair_rows)
    merged = child_impact_df.merge(pairs_df, on=["network_type", "_child_id"], how="inner")
    if merged.empty:
        return impact_df

    agg = (
        merged.groupby(["_parent_id", "carrier", "network_type"], as_index=False)["impact"]
        .mean()
        .rename(columns={"_parent_id": "id"})
    )
    agg["type"] = agg["id"].map(parent_type)
    return pandas.concat([impact_df, agg], ignore_index=True)


def create_impact_df(perf_df: pandas.DataFrame, fail_df, metrics_df):
    # Vectorised: replaces a per-metric-row double-axis-1 apply over perf_df.
    # Failures now persist from f.step to the end of the run (no repair) — see
    # SimpleResilienceModel.step where process_network_state is disabled.
    perf_df = perf_df.rename(columns={"Unnamed: 0": "step"}).dropna()
    perf_df = perf_df.copy()
    perf_df["id"] = perf_df["id"].astype(str)
    perf_df["experiment"] = perf_df["experiment"].astype(str)
    perf_df["network_type"] = perf_df["network_type"].astype(str)

    usable_fail_df = fail_df.copy()
    usable_fail_df["node"] = usable_fail_df["node"].apply(
        lambda c: get_id_type(c) + ":" + c.split(":")[-1]
    )

    # Build map from fail-df node id → canonical metrics_df id (handles branch
    # tuple-order reversal: "(9, 2, 0)" in fail csv ↔ "(2, 9, 0)" in metrics_df).
    metric_id_map = {}
    for mid in metrics_df["id"].astype(str).unique():
        metric_id_map.setdefault(mid, mid)
        if mid.startswith("branch:"):
            inner = mid[len("branch:"):].strip("()")
            parts = [p.strip() for p in inner.split(",")]
            if len(parts) >= 3:
                reversed_id = f"branch:({parts[1]}, {parts[0]}, {parts[2]})"
                metric_id_map.setdefault(reversed_id, mid)

    usable_fail_df["metric_id"] = usable_fail_df["node"].map(metric_id_map)
    fail_matched = usable_fail_df.dropna(subset=["metric_id"])
    failure_only = fail_matched[fail_matched["type"] == "failure"]

    rules_df = (
        failure_only
        .groupby(["metric_id", "network_type", "experiment", "id"], as_index=False)["step"]
        .min()
        .rename(columns={"step": "fail_step"})
    )
    rules_df["id"] = rules_df["id"].astype(str)
    rules_df["experiment"] = rules_df["experiment"].astype(str)
    rules_df["network_type"] = rules_df["network_type"].astype(str)

    # Totals per network_type — basis for "not during fault" stats.
    totals = perf_df.groupby("network_type")[["0", "1", "2"]].agg(["sum", "count"])

    # "In fault" rows: perf rows that match a rule and have step ≥ fail_step.
    if not rules_df.empty:
        in_merge = perf_df.merge(
            rules_df, on=["network_type", "experiment", "id"], how="inner"
        )
        in_active = in_merge[in_merge["step"] >= in_merge["fail_step"]]
        in_agg = (
            in_active.groupby(["metric_id", "network_type"])[["0", "1", "2"]]
            .agg(["sum", "count"])
        )
    else:
        in_agg = pandas.DataFrame()

    # Convert to plain dicts for fast lookup in the per-metric emit loop.
    totals_dict = {nt: totals.loc[nt] for nt in totals.index}
    in_agg_dict = {idx: in_agg.loc[idx] for idx in in_agg.index} if not in_agg.empty else {}

    carrier_pairs = (("heat", "1"), ("gas", "2"), ("electricity", "0"))
    global_rows = []
    for row in metrics_df.itertuples(index=False):
        metric_id = str(row.id)
        network_type = row.network_type
        type_ = row.type
        total_for_nt = totals_dict.get(network_type)
        if total_for_nt is None:
            continue

        in_for_metric = in_agg_dict.get((metric_id, network_type))
        for carrier_name, carrier_col in carrier_pairs:
            t_sum = total_for_nt[(carrier_col, "sum")]
            t_cnt = total_for_nt[(carrier_col, "count")]
            if in_for_metric is not None:
                in_sum = in_for_metric[(carrier_col, "sum")]
                in_cnt = in_for_metric[(carrier_col, "count")]
            else:
                in_sum = 0.0
                in_cnt = 0
            out_sum = t_sum - in_sum
            out_cnt = t_cnt - in_cnt
            in_mean = in_sum / in_cnt if in_cnt > 0 else float("nan")
            out_mean = out_sum / out_cnt if out_cnt > 0 else float("nan")
            # Absolute degradation: out - in. Positive = component caused loss.
            impact = out_mean - in_mean
            global_rows.append(
                {
                    "id": metric_id,
                    "carrier": carrier_name,
                    "impact": impact,
                    "type": type_,
                    "network_type": network_type,
                }
            )
    return pandas.DataFrame(global_rows)


def create_or_load_impact_df(fail_df, perf_df, metrics_df, folder_id):
    impact_out = OUTPUT + f"/{folder_id}/impact.csv"
    if Path(impact_out).exists():
        return pandas.read_csv(impact_out)
    impact_df = create_impact_df(perf_df, fail_df, metrics_df)
    impact_df.to_csv(Path(impact_out))
    return impact_df


def create_metrics_df(monee_net: Network, network_type: str):
    try:
        result = run_energy_flow(monee_net, solver=PyomoSolver(), solver_name="gurobi")
        monee_net = result.network
        for edge in monee_net._network_internal.edges:
            branch_model = monee_net.branch_by_id(edge).model
            monee_net._network_internal.edges[edge]["weight"] = branch_model.loss_percent()
    except Exception as e:
        traceback.print_exc()

        print(f"Warning: energy flow failed for {network_type}, using uniform weights: {e}")
        for edge in monee_net._network_internal.edges:
            monee_net._network_internal.edges[edge]["weight"] = 1.0

    # id, type, metric... (betweenness_centrality)

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
                "network_type": network_type,
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
                "network_type": network_type,
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
                "network_type": network_type,
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
                "network_type": network_type,
            }
        )
    return pandas.DataFrame(all_rows)


def create_full_metrics_df(key_to_net):
    dfs = []
    for net_type, net in key_to_net.items():
        dfs.append(create_metrics_df(net, net_type))
    return pandas.concat(dfs)


def create_or_load_metrics_df(key_to_net, folder_id):
    metrics_out = OUTPUT + f"/{folder_id}/metrics.csv"
    if Path(metrics_out).exists():
        return pandas.read_csv(metrics_out)
    metrics_df = create_full_metrics_df(key_to_net)
    Path(OUTPUT + f"/{folder_id}").mkdir(exist_ok=True, parents=True)
    metrics_df.to_csv(Path(metrics_out))
    return metrics_df


def append_desc_df(single_df, identifier, network_type):
    single_df["experiment"] = identifier
    single_df["network_type"] = network_type


def load_dfs(folder_id):
    all_folders = [f.path for f in os.scandir(folder_id) if f.is_dir()]

    failure_dfs = []
    performance_dfs = []

    net_type_to_net = {}

    for experiment_desc in all_folders:
        if not (Path(experiment_desc) / "performance.csv").exists():
            print(f"Skipping incomplete folder: {experiment_desc}")
            continue
        experiment_desc_name = PurePath(experiment_desc).name
        # Folder layout: <EXPERIMENT_NAME>-<grid>
        # The grid name may itself contain dashes ("urban_district" etc. do not,
        # but guard anyway), so take everything after the first hyphen.
        network_type = experiment_desc_name.split("-", 1)[1]

        if network_type not in net_type_to_net:
            with open(Path(experiment_desc) / Path("network.p"), "rb") as network_file:
                monee_net = pickle.load(network_file)
                print(monee_net.statistics())
                net_type_to_net[network_type] = monee_net

        # failure
        failure_path = Path(experiment_desc) / Path("failure.csv")
        if failure_path.exists():
            failure_df = pandas.read_csv(failure_path)
            append_desc_df(failure_df, experiment_desc, network_type)
            failure_dfs.append(failure_df)

        # performance
        performance_df = pandas.read_csv(
            Path(experiment_desc) / Path("performance.csv")
        )
        append_desc_df(performance_df, experiment_desc, network_type)
        performance_dfs.append(performance_df)

    return (
        pandas.concat(failure_dfs),
        pandas.concat(performance_dfs),
        create_or_load_metrics_df(net_type_to_net, folder_id),
        net_type_to_net,
    )


COLUMN_TIMESTEP = "Unnamed: 0"
COLUMN_EL = "0"
COLUMN_HEAT = "1"
COLUMN_GAS = "2"
COLUMN_ID = "id"
COLUMN_EXPERIMENT_NAME = "experiment"

CARRIER_REPLACE_MAP = {"0": "electricity", "1": "heat", "2": "gas"}


def resilience_per_scenario(perf_df: pandas.DataFrame, folder_id):
    # experiment, id 0 1 2
    resilience_per_carrier_per_scenario = (
        perf_df.groupby(["network_type", "experiment", "id"])[["0", "1", "2"]]
        .sum()
        .reset_index()
        .groupby(["network_type", "experiment"])
        .mean()
        .reset_index()
        .melt(
            id_vars=["network_type", "experiment"],
            value_vars=["0", "1", "2"],
            var_name="carrier",
            value_name="resilience_mean",
        )
    )
    # Experiment label is now the grid name (one experiment per grid).
    # Extract from the folder path "<EXPERIMENT_NAME>-<grid>".
    resilience_per_carrier_per_scenario["experiment"] = (
        resilience_per_carrier_per_scenario["experiment"].apply(
            lambda v: v.split("/")[-1].split("-", 1)[1]
        )
    )
    resilience_per_carrier_per_scenario["carrier"] = (
        resilience_per_carrier_per_scenario["carrier"].apply(
            lambda v: CARRIER_REPLACE_MAP[v]
        )
    )
    eval.create_bar(
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
    unique_network_types = sorted(pandas.unique(
        resilience_per_carrier_per_scenario["network_type"]
    ))
    unique_experiments = list(
        pandas.unique(resilience_per_carrier_per_scenario["experiment"])
    )

    resilience_per_carrier_per_scenario_hist_2 = (
        eval.create_multilevel_grouped_bar_chart(
            [
                list(
                    resilience_per_carrier_per_scenario[
                        resilience_per_carrier_per_scenario["carrier"] == carrier
                    ].sort_values(by=["network_type", "experiment"])["resilience_mean"]
                )
                for carrier in ["electricity", "heat", "gas"]
            ],
            ["#ffa000", "#d32f2f", "#388e3c"],
            ["electricity", "heat", "gas"],
            [f"<b>{net_type}</b>" for net_type in unique_network_types],
            len(unique_experiments),
            [str(exp) for exp in unique_experiments] * len(unique_network_types),
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
    net_type_to_net: Dict[str, Network],
    impact_df,
    metrics_df,
    folder_id,
    metric_ids,
):
    metric_impact_df: pandas.DataFrame = impact_df.astype({"id": "string"}).merge(
        metrics_df.astype({"id": "string"}), on=["id", "network_type"]
    )
    metric_impact_df["type_y"] = (
        metric_impact_df["type_y"].astype(str).str.split(".").str[-1].str[:-2]
    )
    metric_impact_df["impact"] = metric_impact_df["impact"].abs()
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

        for net_type, monee_net in net_type_to_net.items():
            metric_impact_df_carrier_net_type = metric_impact_df_carrier[
                metric_impact_df_carrier["network_type"] == net_type
            ]
            figures.append(
                eval.create_networkx_plot(
                    monee_net,
                    metric_impact_df_carrier_net_type,
                    color_name="impact",
                    color_legend_text=f"{carrier_name}-impact",
                    template="plotly_white+publish",
                )
            )
            titles.append(f"graph of the components' {carrier_name}-impact ({net_type})")
            figures.append(
                eval.create_networkx_plot(
                    monee_net,
                    metric_impact_df_carrier_net_type,
                    color_name="impact",
                    color_legend_text=f"{carrier_name}-impact",
                    template="plotly_white+publish",
                    without_nodes=True,
                )
            )
            titles.append(
                f"edge-graph of the components' {carrier_name}-impact ({net_type})"
            )

    # aggregated all carrier impacts
    metric_impact_df_all_carrier = (
        metric_impact_df.groupby(["type_y", "network_type", "id"] + metric_ids)
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
                yaxis_title="impact",
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
                    yaxis_title="impact",
                    xaxis_title=metric,
                    legend_text="type",
                )
            )
            titles.append(f"{metric} to the {key}' {carrier_name}-impact")

    for net_type, monee_net in net_type_to_net.items():
        metric_impact_df_carrier_net_type = metric_impact_df_all_carrier[
            metric_impact_df_all_carrier["network_type"] == net_type
        ]
        figures.append(
            eval.create_networkx_plot(
                monee_net,
                metric_impact_df_carrier_net_type,
                color_name="impact",
                color_legend_text="impact",
                template="plotly_white+publish",
            )
        )
        titles.append(f"graph of the components' impact ({net_type})")
        figures.append(
            eval.create_networkx_plot(
                monee_net,
                metric_impact_df_carrier_net_type,
                color_name="impact",
                color_legend_text="impact",
                template="plotly_white+publish",
                without_nodes=True,
            )
        )
        titles.append(f"edge-graph of the components' impact ({net_type})")

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{folder_id}/metric_to_impact.html",
        titles=titles,
    )


def impact_aggregated_component_carrier(impact_df: pandas.DataFrame, folder_id):
    new_impact_df = impact_df.copy()
    new_impact_df["impact"] = new_impact_df["impact"].abs()
    new_impact_df["type"] = (
        impact_df["type"].astype(str).str.split(".").str[-1].str[:-2]
    )
    new_impact_df["type_carrier"] = new_impact_df["type"].map(TYPE_TO_CARRIER)
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
    titles.append("Average impacts by component type")
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
    titles.append("Total impacts by component type")
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
    titles.append("Average impacts by carrier type")
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
    titles.append("Total impacts by carrier type")

    average_impact_per_carrier_net_type = (
        new_impact_df.groupby(["type_carrier", "carrier", "network_type"])
        .mean()
        .reset_index()
    )
    average_impact_per_carrier_net_type["carrier_net_type"] = (
        average_impact_per_carrier_net_type["type_carrier"].astype(str)
        + "-"
        + average_impact_per_carrier_net_type["network_type"].astype(str)
    )
    impact_per_carrier_net_type = (
        new_impact_df.groupby(["type_carrier", "carrier", "network_type"])
        .sum()
        .reset_index()
    )
    impact_per_carrier_net_type["carrier_net_type"] = (
        impact_per_carrier_net_type["type_carrier"].astype(str)
        + "-"
        + impact_per_carrier_net_type["network_type"].astype(str)
    )
    figures += [
        eval.create_bar(
            average_impact_per_carrier_net_type,
            x_label="carrier_net_type",
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
    titles.append("Average impacts by carrier type and density")
    figures += [
        eval.create_bar(
            impact_per_carrier_net_type,
            x_label="carrier_net_type",
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
    titles.append("Total impacts by carrier type and density")

    average_impact_per_net_type = (
        new_impact_df.groupby(["carrier", "network_type"]).mean().reset_index()
    )
    average_impact_per_net_type["network_type"] = average_impact_per_net_type["network_type"].astype(
        str
    )
    total_impact_per_net_type = (
        new_impact_df.groupby(["carrier", "network_type"]).sum().reset_index()
    )
    total_impact_per_net_type["network_type"] = total_impact_per_net_type["network_type"].astype(str)

    figures += [
        eval.create_bar(
            average_impact_per_net_type,
            x_label="network_type",
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
    titles.append("Average impacts by density")
    figures += [
        eval.create_bar(
            total_impact_per_net_type,
            x_label="network_type",
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
    titles.append("Total impacts by density")

    eval.write_all_in_one(
        figures,
        "Figure",
        Path("."),
        OUTPUT + f"/{folder_id}/impact_aggregated_component_carrier.html",
        titles=titles,
    )


def _build_branch_lookup(impact_ids):
    """Map (a_str, b_str) → impact_id for every branch impact id (both directions)."""
    branch_lookup = {}
    for iid in impact_ids:
        if not iid.startswith("branch:"):
            continue
        inner = iid[len("branch:"):].strip("()")
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
            branch_lookup.setdefault((a, b), iid)
            branch_lookup.setdefault((b, a), iid)
    return branch_lookup


def _match_impact_id(cp_id, cp_type, impact_ids, branch_lookup=None):
    """Find the impact_df id string for a given metric entry."""
    if cp_type in ("CHP", "PowerToHeat"):
        candidate = f"compound:{cp_id}"
        return candidate if candidate in impact_ids else None

    if branch_lookup is None:
        branch_lookup = _build_branch_lookup(impact_ids)

    # Non-CP branches: cp_id is already str(edge_tuple) e.g. "(10, 4, 0)"
    if cp_type in ("PowerLine", "GasPipe", "WaterPipe", "HeatExchanger"):
        candidate = f"branch:{cp_id}"
        if candidate in impact_ids:
            return candidate
        inner = str(cp_id).strip("()")
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 2:
            return branch_lookup.get((parts[0], parts[1]))
        return None

    # branch CPs (GasToPower, PowerToGas): cp_id = "from→to"
    try:
        from_id, to_id = str(cp_id).split("→")
        from_id, to_id = from_id.strip(), to_id.strip()
    except ValueError:
        return None
    return branch_lookup.get((from_id, to_id))


def cp_metric_vs_actual_impact(monee_net, impact_df_nt, network_type):
    try:
        df_scores, _ = mes_all_components_metric(monee_net, cfg=CPMetricConfig())
    except Exception as e:
        print(f"CP metric failed for {network_type}: {e}")
        raise e

    # Aggregate actual impact per component across carriers
    impact_abs = impact_df_nt.copy()
    impact_abs["impact"] = impact_abs["impact"].abs()
    actual_total = (
        impact_abs.groupby("id")["impact"].sum()
        .reset_index()
        .rename(columns={"impact": "actual_total"})
    )
    actual_per_carrier = (
        impact_abs.pivot_table(index="id", columns="carrier", values="impact", aggfunc="sum")
        .reset_index()
    )

    impact_ids = set(actual_total["id"])
    branch_lookup = _build_branch_lookup(impact_ids)
    total_lookup = dict(zip(actual_total["id"], actual_total["actual_total"]))
    per_carrier_lookup = {
        col: dict(zip(actual_per_carrier["id"], actual_per_carrier[col]))
        for col in actual_per_carrier.columns
        if col != "id"
    }
    rows = []
    for score_row in df_scores.itertuples(index=False):
        impact_id = _match_impact_id(
            score_row.cp_id, score_row.cp_type, impact_ids, branch_lookup
        )
        if impact_id is None:
            continue
        actual_total_val = total_lookup.get(impact_id)
        if actual_total_val is None:
            continue
        score_dict = score_row._asdict()
        entry = {
            "cp_id": str(score_row.cp_id),
            "cp_type": score_row.cp_type,
            "predicted_score": score_row.score,
            "predicted_stress": score_row.total_stress,
            "topo_factor": score_row.topo_factor,
            "topo_bc": score_row.topo_bc,
            "stress_bc": score_dict.get("stress_bc", 0.0),
            "stress_score": score_dict.get("stress_score", score_row.score),
            "local_score": score_dict.get("local_score", score_row.score),
            "self_score": score_dict.get("self_score", score_row.score),
            "katz_score": score_dict.get("katz_score", 0.0),
            "vitality_score": score_dict.get("vitality_score", 0.0),
            "actual_total": actual_total_val,
        }
        for metric_col, carrier_col in [
            ("power_stress", "electricity"),
            ("gas_stress", "gas"),
            ("heat_stress", "heat"),
        ]:
            entry[f"predicted_{metric_col}"] = score_dict.get(metric_col, 0.0)
            carrier_map = per_carrier_lookup.get(carrier_col)
            entry[f"actual_{carrier_col}"] = (
                carrier_map.get(impact_id, 0.0) if carrier_map is not None else 0.0
            )
        rows.append(entry)

    if not rows:
        print(f"No metric/impact matches found for {network_type}")
        return

    df = pandas.DataFrame(rows)
    figures = []
    titles = []

    import numpy as _np2

    def _spearman_with_ci(a, b, alpha=0.05):
        """Returns (rho, pval, ci_lo, ci_hi) using Fisher z-transform CI."""
        res = scipy.stats.spearmanr(a, b)
        rho, pval = res.statistic, res.pvalue
        n = len(a)
        if n > 3:
            z = _np2.arctanh(rho)
            se = 1.0 / _np2.sqrt(n - 3)
            z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
            ci_lo = float(_np2.tanh(z - z_crit * se))
            ci_hi = float(_np2.tanh(z + z_crit * se))
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        return float(rho), float(pval), ci_lo, ci_hi

    def _rho_label(rho, pval, ci_lo, ci_hi):
        return f"ρ={rho:.2f} [{ci_lo:.2f}, {ci_hi:.2f}], p={pval:.3f}"

    # Spearman correlation annotation
    rho_str = ""
    if len(df) >= 3:
        rho, pval, ci_lo, ci_hi = _spearman_with_ci(
            df["predicted_score"], df["actual_total"]
        )
        rho_str = f" (Spearman {_rho_label(rho, pval, ci_lo, ci_hi)})"

    # Scatter: predicted score vs actual total impact
    figures.append(eval.create_scatter_with_df(
        df, "predicted_score", "actual_total",
        color_label="cp_type",
        yaxis_title="Actual Total Impact (MC)",
        xaxis_title="Predicted CP Score",
        legend_text="CP Type",
    ))
    titles.append(f"Predicted Component Score vs Actual Impact{rho_str}")

    # Per-carrier: predicted stress vs actual carrier impact
    for pred_col, actual_col, label in [
        ("predicted_power_stress", "actual_electricity", "Electricity"),
        ("predicted_gas_stress", "actual_gas", "Gas"),
        ("predicted_heat_stress", "actual_heat", "Heat"),
    ]:
        if actual_col not in df.columns:
            continue
        figures.append(eval.create_scatter_with_df(
            df, pred_col, actual_col,
            color_label="cp_type",
            yaxis_title=f"Actual {label} Impact (MC)",
            xaxis_title=f"Predicted {label} Stress",
            legend_text="CP Type",
        ))
        titles.append(f"Predicted {label} Stress vs Actual {label} Impact")

    # Topo factor vs actual total impact
    figures.append(eval.create_scatter_with_df(
        df, "topo_factor", "actual_total",
        color_label="cp_type",
        yaxis_title="Actual Total Impact (MC)",
        xaxis_title="Topology Factor (1 + α·BC)",
        legend_text="CP Type",
    ))
    titles.append("Topology Factor vs Actual Total Impact")

    figures.append(eval.create_scatter_with_df(
        df, "topo_bc", "actual_total",
        color_label="cp_type",
        yaxis_title="Actual Total Impact (MC)",
        xaxis_title="Group Betweenness Centrality",
        legend_text="CP Type",
    ))
    titles.append("Betweenness Centrality vs Actual Total Impact")

    # Topology benefit: compare score with vs without topo factor
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    df["score_no_topo"] = df["predicted_score"] / df["topo_factor"].replace(0, float("nan"))
    # topo_only: use raw BC as the sole predictor
    df["score_topo_only"] = df["topo_bc"]

    rows_with_data = df.dropna(subset=["score_no_topo", "actual_total"])
    if len(rows_with_data) >= 3:
        def _sr(col):
            return _spearman_with_ci(rows_with_data[col], rows_with_data["actual_total"])

        rho_with,       pval_with,       ci_lo_with,       ci_hi_with       = _sr("predicted_score")
        rho_without,    pval_without,    ci_lo_without,    ci_hi_without    = _sr("score_no_topo")
        rho_topo_only,  pval_topo_only,  ci_lo_topo_only,  ci_hi_topo_only  = _sr("score_topo_only")
        rho_stress_bc,  pval_stress_bc,  ci_lo_stress_bc,  ci_hi_stress_bc  = _sr("stress_bc")
        rho_local,      pval_local,      ci_lo_local,      ci_hi_local      = _sr("local_score")
        rho_self,       pval_self,       ci_lo_self,       ci_hi_self       = _sr("self_score")
        rho_katz,       pval_katz,       ci_lo_katz,       ci_hi_katz       = _sr("katz_score")
        rho_vitality,   pval_vitality,   ci_lo_vitality,   ci_hi_vitality   = _sr("vitality_score")

        cp_types = rows_with_data["cp_type"].unique().tolist()
        colors = px.colors.qualitative.Plotly
        type_color = {t: colors[i % len(colors)] for i, t in enumerate(cp_types)}

        panels = [
            # x_col, subplot title (with ρ + 95% CI), x-axis label (exact formula)
            ("predicted_score",
             f"Full: PTDF stress + phys. BC<br>{_rho_label(rho_with, pval_with, ci_lo_with, ci_hi_with)}",
             "p_fail · τ · PTDF_stress · (1 + α·BC_phys)"),
            ("score_no_topo",
             f"PTDF stress only, no BC<br>{_rho_label(rho_without, pval_without, ci_lo_without, ci_hi_without)}",
             "p_fail · τ · PTDF_stress"),
            ("score_topo_only",
             f"Phys. BC only, no stress<br>{_rho_label(rho_topo_only, pval_topo_only, ci_lo_topo_only, ci_hi_topo_only)}",
             "Phys. betweenness centrality"),
            ("stress_bc",
             f"Stress-weighted BC only<br>{_rho_label(rho_stress_bc, pval_stress_bc, ci_lo_stress_bc, ci_hi_stress_bc)}",
             "Stress-weighted betweenness centrality"),
            ("local_score",
             f"1-hop local: loading + critical neighbours<br>{_rho_label(rho_local, pval_local, ci_lo_local, ci_hi_local)}",
             "p_fail · loading · (1 + crit.nbrs) · n_carriers"),
            ("self_score",
             f"0-hop self: own loading only<br>{_rho_label(rho_self, pval_self, ci_lo_self, ci_hi_self)}",
             "p_fail · loading · n_carriers"),
            ("katz_score",
             f"Katz centrality only<br>{_rho_label(rho_katz, pval_katz, ci_lo_katz, ci_hi_katz)}",
             "Katz centrality (phys. graph)"),
            ("vitality_score",
             f"Closeness vitality only<br>{_rho_label(rho_vitality, pval_vitality, ci_lo_vitality, ci_hi_vitality)}",
             "Closeness vitality W(G)−W(G\\v) (phys. graph)"),
        ]

        topo_fig = make_subplots(
            rows=1, cols=len(panels),
            subplot_titles=[p[1] for p in panels],
            shared_yaxes=True,
        )
        for col_idx, (x_col, _title, x_label) in enumerate(panels, start=1):
            for cp_type in cp_types:
                sub = rows_with_data[rows_with_data["cp_type"] == cp_type]
                color = type_color[cp_type]
                topo_fig.add_trace(go.Scatter(
                    x=sub[x_col], y=sub["actual_total"],
                    mode="markers", name=cp_type,
                    marker=dict(color=color, size=8),
                    legendgroup=cp_type, showlegend=(col_idx == 1),
                ), row=1, col=col_idx)
            topo_fig.update_xaxes(title_text=x_label, row=1, col=col_idx)

        topo_fig.update_yaxes(title_text="Actual Total Impact (MC)", row=1, col=1)
        topo_fig.update_layout(
            height=420, width=1600,
            template="plotly_white",
            margin={"l": 50, "b": 50, "r": 20, "t": 60},
            legend={"title": "Component Type"},
        )
        figures.append(topo_fig)
        titles.append(
            f"Metric scatter comparison — "
            f"ρ(PTDF+pBC)={rho_with:.2f} | ρ(PTDF)={rho_without:.2f} | "
            f"ρ(pBC)={rho_topo_only:.2f} | ρ(sBC)={rho_stress_bc:.2f} | "
            f"ρ(1-hop)={rho_local:.2f} | ρ(0-hop)={rho_self:.2f} | ρ(katz)={rho_katz:.2f} | ρ(vi)={rho_vitality:.2f}"
        )

    # ── Metric comparison figures ──────────────────────────────────────────
    import numpy as _np

    METRICS = [
        ("predicted_score",  "PTDF stress + phys. BC"),   # p_fail·τ·PTDF·(1+α·BC_phys)
        ("score_no_topo",    "PTDF stress only"),          # p_fail·τ·PTDF  (BC removed)
        ("score_topo_only",  "Phys. BC only"),             # raw betweenness centrality, no stress
        ("stress_bc",        "Stress BC only"),            # raw stress-weighted betweenness centrality, no PTDF
        ("local_score",      "1-hop local"),               # p_fail·loading·(1+crit.nbrs)·n_carriers
        ("self_score",       "0-hop self"),                # p_fail·loading·n_carriers
        ("katz_score",       "Katz BC only"),              # raw Katz centrality (phys. graph), no stress
        ("vitality_score",   "Closeness vitality"),        # W(G) - W(G\v), phys. weights
        ("actual_total",     "Actual (MC)"),               # ground truth
    ]
    df = df.copy()
    df["network_type"] = network_type

    # Compute ranks for every metric (1 = highest)
    for col, _label in METRICS:
        df[f"rank_{col}"] = df[col].rank(ascending=False, method="min").astype(int)

    valid = df.dropna(subset=[col for col, _ in METRICS])

    # 1. ρ bar chart with 95% CI error bars
    rho_rows = []
    for col, label in METRICS[:-1]:  # exclude actual vs itself
        rho, pval, ci_lo, ci_hi = _spearman_with_ci(valid[col], valid["actual_total"])
        rho_rows.append({
            "Metric": label, "Spearman ρ": rho, "p-value": pval,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "err_lo": rho - ci_lo, "err_hi": ci_hi - rho,
        })
    rho_df = pandas.DataFrame(rho_rows).sort_values("Spearman ρ", ascending=False)

    rho_bar = go.Figure(go.Bar(
        x=rho_df["Metric"],
        y=rho_df["Spearman ρ"],
        error_y=dict(
            type="data", symmetric=False,
            array=rho_df["err_hi"].tolist(),
            arrayminus=rho_df["err_lo"].tolist(),
        ),
        text=[f"ρ={r:.2f} [{lo:.2f},{hi:.2f}]<br>p={p:.3f}"
              for r, p, lo, hi in zip(
                  rho_df["Spearman ρ"], rho_df["p-value"],
                  rho_df["ci_lo"], rho_df["ci_hi"])],
        textposition="outside",
        marker_color=px.colors.qualitative.Plotly[:len(rho_df)],
    ))
    rho_bar.update_layout(
        height=450, width=800,
        template="plotly_white",
        yaxis=dict(title="Spearman ρ vs Actual (95% CI)", range=[-1.15, 1.15],
                   zeroline=True, zerolinecolor="black", zerolinewidth=1),
        xaxis_title="Metric",
        margin={"l": 50, "b": 100, "r": 20, "t": 40},
        showlegend=False,
    )
    figures.append(rho_bar)
    titles.append("Predictive Power: Spearman ρ vs Actual Impact")

    # 2. Pairwise rank-correlation heatmap
    all_labels = [label for _, label in METRICS]
    all_cols   = [col   for col, _ in METRICS]
    n = len(all_cols)
    rho_matrix = _np.ones((n, n))
    for i, ci in enumerate(all_cols):
        for j, cj in enumerate(all_cols):
            if i != j:
                rho_matrix[i, j] = float(scipy.stats.spearmanr(valid[ci], valid[cj]).statistic)

    heatmap_fig = go.Figure(go.Heatmap(
        z=rho_matrix,
        x=all_labels, y=all_labels,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=[[f"{rho_matrix[i,j]:.2f}" for j in range(n)] for i in range(n)],
        texttemplate="%{text}",
        colorbar=dict(title=dict(text="Spearman ρ", side="right")),
    ))
    heatmap_fig.update_layout(
        height=480, width=620,
        template="plotly_white",
        margin={"l": 120, "b": 120, "r": 20, "t": 40},
        xaxis=dict(title="Metric"),
        yaxis=dict(title="Metric"),
    )
    figures.append(heatmap_fig)
    titles.append("Pairwise Rank Correlation (Spearman ρ) Between All Metrics")

    # 3. Bump chart – rank of each component across all metrics
    metric_names = [label for _, label in METRICS]
    rank_cols    = [f"rank_{col}" for col, _ in METRICS]
    cp_colors    = px.colors.qualitative.Plotly
    cp_type_color = {t: cp_colors[i % len(cp_colors)]
                     for i, t in enumerate(df["cp_type"].unique())}

    seen_cp_types = set()
    bump_fig = go.Figure()
    for _, row in df.iterrows():
        ranks = [row[rc] for rc in rank_cols]
        cp_type = row["cp_type"]
        first_of_type = cp_type not in seen_cp_types
        seen_cp_types.add(cp_type)
        bump_fig.add_trace(go.Scatter(
            x=metric_names, y=ranks,
            mode="lines+markers",
            name=str(row["cp_id"]),
            legendgroup=cp_type,
            legendgrouptitle=dict(text=cp_type) if first_of_type else None,
            line=dict(color=cp_type_color.get(cp_type, "grey"), width=1.5),
            marker=dict(size=7),
            hovertemplate=f"<b>{row['cp_id']}</b> ({cp_type})<br>%{{x}}: rank %{{y}}<extra></extra>",
            showlegend=True,
        ))

    bump_fig.update_layout(
        height=max(400, 20 * len(df)),
        width=900,
        template="plotly_white",
        yaxis=dict(title="Rank (1 = highest impact)", autorange="reversed",
                   dtick=1, gridcolor="lightgrey"),
        xaxis=dict(title="Metric"),
        margin={"l": 60, "b": 60, "r": 180, "t": 40},
        legend=dict(title="Component", groupclick="toggleitem",
                    x=1.01, y=1, xanchor="left"),
    )
    figures.append(bump_fig)
    titles.append("Rank Bump Chart: Each Component's Rank Across All Metrics")

    # ── Ranking accuracy figures ───────────────────────────────────────────
    # Helpers (no external dependencies beyond numpy/scipy)

    def _ndcg(actual_vals, predicted_scores):
        """Normalised Discounted Cumulative Gain.
        Relevance = max(actual, 0) so components with zero/negative impact get
        relevance 0 — they are not 'relevant' to identify as critical.
        Using actual - min() would inflate all scores when the range is wide and
        arbitrarily reward metrics that rank the least-harmful component last.
        """
        actual_arr = _np.array(actual_vals, dtype=float)
        pred_order = _np.argsort(predicted_scores)[::-1]
        ideal_order = _np.argsort(actual_arr)[::-1]
        gains = _np.maximum(actual_arr, 0.0)           # negative impact → relevance 0
        dcg  = sum(gains[pred_order[i]]  / _np.log2(i + 2) for i in range(len(gains)))
        idcg = sum(gains[ideal_order[i]] / _np.log2(i + 2) for i in range(len(gains)))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def _precision_at_k(actual_vals, predicted_scores, k):
        """Fraction of true top-k components that appear in the predicted top-k."""
        actual_top = set(_np.argsort(actual_vals)[-k:])
        pred_top   = set(_np.argsort(predicted_scores)[-k:])
        return len(actual_top & pred_top) / k

    def _bootstrap_ci(stat_fn, actual_arr, pred_arr, n_boot=1000, alpha=0.05, rng=None):
        """Bootstrap percentile CI for any scalar statistic.
        stat_fn(actual, predicted) → float
        """
        if rng is None:
            rng = _np.random.default_rng(42)
        n = len(actual_arr)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boot.append(stat_fn(actual_arr[idx], pred_arr[idx]))
        return (float(_np.percentile(boot, 100 * alpha / 2)),
                float(_np.percentile(boot, 100 * (1 - alpha / 2))))

    pred_metrics = [(col, label) for col, label in METRICS if col != "actual_total"]
    actual_vals  = valid["actual_total"].values

    # 4a. Summary bar chart: Kendall τ and NDCG per metric, with bootstrap 95% CIs
    _rng = _np.random.default_rng(42)
    acc_rows = []
    for col, label in pred_metrics:
        scores = valid[col].values
        tau  = float(scipy.stats.kendalltau(scores, actual_vals).statistic)
        ndcg = _ndcg(actual_vals, scores)
        tau_lo, tau_hi   = _bootstrap_ci(
            lambda a, p: float(scipy.stats.kendalltau(p, a).statistic),
            actual_vals, scores, rng=_rng)
        ndcg_lo, ndcg_hi = _bootstrap_ci(
            lambda a, p: _ndcg(a, p),
            actual_vals, scores, rng=_rng)
        acc_rows.append({
            "Metric": label,
            "Kendall τ": tau,  "τ_lo": tau_lo,  "τ_hi": tau_hi,
            "NDCG":      ndcg, "ndcg_lo": ndcg_lo, "ndcg_hi": ndcg_hi,
        })

    acc_df = pandas.DataFrame(acc_rows).sort_values("NDCG", ascending=True)

    acc_fig = make_subplots(rows=1, cols=2, subplot_titles=["Kendall τ (95% CI)", "NDCG (95% CI)"])
    metric_colors = {row["Metric"]: px.colors.qualitative.Plotly[i % 10]
                     for i, row in acc_df.iterrows()}

    for col_idx, (measure, lo_col, hi_col) in enumerate(
        [("Kendall τ", "τ_lo", "τ_hi"), ("NDCG", "ndcg_lo", "ndcg_hi")], start=1
    ):
        vals   = acc_df[measure].values
        err_lo = (vals - acc_df[lo_col].values).clip(0)
        err_hi = (acc_df[hi_col].values - vals).clip(0)
        acc_fig.add_trace(go.Bar(
            x=vals,
            y=acc_df["Metric"],
            orientation="h",
            marker_color=[metric_colors[m] for m in acc_df["Metric"]],
            error_x=dict(type="data", symmetric=False,
                         array=err_hi.tolist(), arrayminus=err_lo.tolist()),
            text=[f"{v:.3f} [{lo:.2f},{hi:.2f}]"
                  for v, lo, hi in zip(vals, acc_df[lo_col], acc_df[hi_col])],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)
        ref_range = [-1.05, 1.05] if measure == "Kendall τ" else [-0.05, 1.05]
        acc_fig.update_xaxes(title_text=measure, range=ref_range,
                             zeroline=True, zerolinecolor="black", zerolinewidth=1,
                             row=1, col=col_idx)
        acc_fig.update_yaxes(title_text="Metric", row=1, col=col_idx)

    acc_fig.update_layout(
        height=80 + 40 * len(acc_df), width=1000,
        template="plotly_white",
        margin={"l": 160, "b": 50, "r": 120, "t": 50},
    )
    figures.append(acc_fig)
    best = acc_df.iloc[-1]["Metric"]
    titles.append(f"Ranking Accuracy: Kendall τ and NDCG with Bootstrap 95% CI (best NDCG: {best})")

    # 4b. Precision@k curve: how well does each metric identify the true top-k?
    n_comp = len(valid)
    k_values = list(range(1, n_comp + 1))
    prec_fig = go.Figure()
    for col, label in pred_metrics:
        scores = valid[col].values
        prec_at_k = [_precision_at_k(actual_vals, scores, k) for k in k_values]
        prec_fig.add_trace(go.Scatter(
            x=k_values, y=prec_at_k,
            mode="lines+markers",
            name=label,
            marker=dict(size=5),
        ))
    # Baseline: random predictor expected precision = k/n
    prec_fig.add_trace(go.Scatter(
        x=k_values, y=[k / n_comp for k in k_values],
        mode="lines", name="Random baseline",
        line=dict(dash="dash", color="grey", width=1),
    ))
    prec_fig.update_layout(
        height=450, width=750,
        template="plotly_white",
        xaxis=dict(title="k (number of top components considered)", dtick=1),
        yaxis=dict(title="Precision@k", range=[-0.05, 1.05],
                   zeroline=True, zerolinecolor="lightgrey"),
        margin={"l": 60, "b": 60, "r": 20, "t": 40},
        legend=dict(title="Metric", x=1.01, xanchor="left"),
    )
    figures.append(prec_fig)
    titles.append("Precision@k: Fraction of True Top-k Components Correctly Identified")

    eval.write_all_in_one(
        figures, "Figure", Path("."),
        OUTPUT + f"/{network_type}/cp_metric_vs_actual.html",
        titles=titles,
    )
    print(f"Written cp_metric_vs_actual.html for {network_type}")
    return df


def pooled_metric_comparison(pooled_df, output_dir):
    """Run metric comparison figures on data pooled across all network types.

    pooled_df must have the same columns as the per-network df produced by
    cp_metric_vs_actual_impact, plus a 'network_type' column.
    """
    import numpy as _np
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    METRICS = [
        ("predicted_score",  "PTDF stress + phys. BC"),
        ("score_no_topo",    "PTDF stress only"),
        ("score_topo_only",  "Phys. BC only"),
        ("stress_bc",        "Stress BC only"),
        ("local_score",      "1-hop local"),
        ("self_score",       "0-hop self"),
        ("katz_score",       "Katz BC only"),
        ("vitality_score",   "Closeness vitality"),
        ("actual_total",     "Actual (MC)"),
    ]

    df = pooled_df.copy()
    # Compute score_no_topo if not already present
    if "score_no_topo" not in df.columns:
        df["score_no_topo"] = df["predicted_score"] / df["topo_factor"].replace(0, float("nan"))
    if "score_topo_only" not in df.columns:
        df["score_topo_only"] = df["topo_bc"]

    valid = df.dropna(subset=[col for col, _ in METRICS])
    n_total = len(valid)
    net_types = sorted(valid["network_type"].unique())
    print(f"Pooled analysis: {n_total} components across {len(net_types)} network types: {net_types}")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _spearman_with_ci(a, b, alpha=0.05):
        res = scipy.stats.spearmanr(a, b)
        rho, pval = res.statistic, res.pvalue
        n = len(a)
        if n > 3:
            z = _np.arctanh(rho)
            se = 1.0 / _np.sqrt(n - 3)
            z_crit = scipy.stats.norm.ppf(1 - alpha / 2)
            ci_lo = float(_np.tanh(z - z_crit * se))
            ci_hi = float(_np.tanh(z + z_crit * se))
        else:
            ci_lo, ci_hi = float("nan"), float("nan")
        return float(rho), float(pval), ci_lo, ci_hi

    def _rho_label(rho, pval, ci_lo, ci_hi):
        return f"ρ={rho:.2f} [{ci_lo:.2f},{ci_hi:.2f}], p={pval:.3f}"

    def _ndcg(actual_vals, predicted_scores):
        actual_arr = _np.array(actual_vals, dtype=float)
        pred_order  = _np.argsort(predicted_scores)[::-1]
        ideal_order = _np.argsort(actual_arr)[::-1]
        gains = _np.maximum(actual_arr, 0.0)
        dcg  = sum(gains[pred_order[i]]  / _np.log2(i + 2) for i in range(len(gains)))
        idcg = sum(gains[ideal_order[i]] / _np.log2(i + 2) for i in range(len(gains)))
        return float(dcg / idcg) if idcg > 0 else 0.0

    def _precision_at_k(actual_vals, predicted_scores, k):
        actual_top = set(_np.argsort(actual_vals)[-k:])
        pred_top   = set(_np.argsort(predicted_scores)[-k:])
        return len(actual_top & pred_top) / k

    def _bootstrap_ci(stat_fn, actual_arr, pred_arr, n_boot=1000, alpha=0.05, rng=None):
        if rng is None:
            rng = _np.random.default_rng(42)
        n = len(actual_arr)
        boot = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            boot.append(stat_fn(actual_arr[idx], pred_arr[idx]))
        return (float(_np.percentile(boot, 100 * alpha / 2)),
                float(_np.percentile(boot, 100 * (1 - alpha / 2))))

    figures = []
    titles  = []
    actual_vals = valid["actual_total"].values
    pred_metrics = [(col, label) for col, label in METRICS if col != "actual_total"]
    net_colors = {nt: px.colors.qualitative.Plotly[i % 10]
                  for i, nt in enumerate(net_types)}

    # ── 1. Scatter panels (one per metric, colored by network type) ────────
    panels = [
        ("predicted_score",  "PTDF stress + phys. BC",  "p_fail · τ · PTDF_stress · (1 + α·BC_phys)"),
        ("score_no_topo",    "PTDF stress only",         "p_fail · τ · PTDF_stress"),
        ("score_topo_only",  "Phys. BC only",            "Phys. betweenness centrality"),
        ("stress_bc",        "Stress BC only",           "Stress-weighted betweenness centrality"),
        ("local_score",      "1-hop local",              "p_fail · loading · (1 + crit.nbrs) · n_carriers"),
        ("self_score",       "0-hop self",               "p_fail · loading · n_carriers"),
        ("katz_score",       "Katz BC only",             "Katz centrality (phys. graph)"),
        ("vitality_score",   "Closeness vitality",       "W(G) − W(G\\v) (phys. graph)"),
    ]
    scatter_fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=[
            f"{label}<br>{_rho_label(*_spearman_with_ci(valid[col], valid['actual_total']))}"
            for col, label, _ in panels
        ],
        shared_yaxes=False,
    )
    for idx, (x_col, _label, x_axis_label) in enumerate(panels):
        r, c = divmod(idx, 4)
        for nt in net_types:
            sub = valid[valid["network_type"] == nt]
            scatter_fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub["actual_total"],
                mode="markers", name=nt,
                marker=dict(color=net_colors[nt], size=7),
                legendgroup=nt, showlegend=(idx == 0),
            ), row=r + 1, col=c + 1)
        scatter_fig.update_xaxes(title_text=x_axis_label, row=r + 1, col=c + 1)
        scatter_fig.update_yaxes(title_text="Actual Impact (MW)", row=r + 1, col=c + 1)
    scatter_fig.update_layout(
        height=700, width=1800,
        template="plotly_white",
        margin={"l": 60, "b": 60, "r": 20, "t": 80},
        legend={"title": "Network type"},
    )
    figures.append(scatter_fig)
    titles.append(f"Pooled metric scatter (n={n_total} across {len(net_types)} networks)")

    # ── 2. ρ bar chart with 95% CI ─────────────────────────────────────────
    rho_rows = []
    for col, label in pred_metrics:
        rho, pval, ci_lo, ci_hi = _spearman_with_ci(valid[col], actual_vals)
        rho_rows.append({
            "Metric": label, "Spearman ρ": rho, "p-value": pval,
            "ci_lo": ci_lo, "ci_hi": ci_hi,
            "err_lo": rho - ci_lo, "err_hi": ci_hi - rho,
        })
    rho_df = pandas.DataFrame(rho_rows).sort_values("Spearman ρ", ascending=True)

    rho_bar = go.Figure(go.Bar(
        x=rho_df["Spearman ρ"],
        y=rho_df["Metric"],
        orientation="h",
        error_x=dict(type="data", symmetric=False,
                     array=rho_df["err_hi"].tolist(),
                     arrayminus=rho_df["err_lo"].tolist()),
        text=[f"ρ={r:.2f} [{lo:.2f},{hi:.2f}]<br>p={p:.3f}"
              for r, lo, hi, p in zip(rho_df["Spearman ρ"], rho_df["ci_lo"],
                                      rho_df["ci_hi"], rho_df["p-value"])],
        textposition="outside",
    ))
    rho_bar.update_layout(
        height=80 + 40 * len(rho_df), width=750,
        template="plotly_white",
        xaxis=dict(title="Spearman ρ", range=[-1.05, 1.05],
                   zeroline=True, zerolinecolor="black"),
        yaxis=dict(title="Metric"),
        margin={"l": 160, "b": 50, "r": 160, "t": 50},
    )
    figures.append(rho_bar)
    titles.append(f"Pooled Spearman ρ with Fisher z 95% CI (n={n_total})")

    # ── 3. Spearman ρ per network type (small multiples) ──────────────────
    if len(net_types) > 1:
        nt_rho_fig = go.Figure()
        for col, label in pred_metrics:
            nt_rhos, nt_errs_lo, nt_errs_hi = [], [], []
            for nt in net_types:
                sub = valid[valid["network_type"] == nt]
                if len(sub) < 4:
                    nt_rhos.append(float("nan"))
                    nt_errs_lo.append(0)
                    nt_errs_hi.append(0)
                    continue
                rho, _, ci_lo, ci_hi = _spearman_with_ci(sub[col], sub["actual_total"])
                nt_rhos.append(rho)
                nt_errs_lo.append(rho - ci_lo)
                nt_errs_hi.append(ci_hi - rho)
            nt_rho_fig.add_trace(go.Bar(
                name=label,
                x=net_types,
                y=nt_rhos,
                error_y=dict(type="data", symmetric=False,
                             array=nt_errs_hi, arrayminus=nt_errs_lo),
            ))
        nt_rho_fig.update_layout(
            barmode="group",
            height=450, width=200 + 160 * len(net_types),
            template="plotly_white",
            xaxis=dict(title="Network type"),
            yaxis=dict(title="Spearman ρ", range=[-1.05, 1.05],
                       zeroline=True, zerolinecolor="black"),
            legend=dict(title="Metric"),
            margin={"l": 60, "b": 60, "r": 20, "t": 50},
        )
        figures.append(nt_rho_fig)
        titles.append("Spearman ρ per network type — check for heterogeneity")

    # ── 4. Kendall τ and NDCG with bootstrap CI ───────────────────────────
    _rng = _np.random.default_rng(42)
    acc_rows = []
    for col, label in pred_metrics:
        scores = valid[col].values
        tau  = float(scipy.stats.kendalltau(scores, actual_vals).statistic)
        ndcg = _ndcg(actual_vals, scores)
        tau_lo,  tau_hi  = _bootstrap_ci(
            lambda a, p: float(scipy.stats.kendalltau(p, a).statistic),
            actual_vals, scores, rng=_rng)
        ndcg_lo, ndcg_hi = _bootstrap_ci(
            lambda a, p: _ndcg(a, p),
            actual_vals, scores, rng=_rng)
        acc_rows.append({
            "Metric": label,
            "Kendall τ": tau,  "τ_lo": tau_lo,  "τ_hi": tau_hi,
            "NDCG":      ndcg, "ndcg_lo": ndcg_lo, "ndcg_hi": ndcg_hi,
        })
    acc_df = pandas.DataFrame(acc_rows).sort_values("NDCG", ascending=True)
    metric_colors = {row["Metric"]: px.colors.qualitative.Plotly[i % 10]
                     for i, row in acc_df.iterrows()}

    acc_fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Kendall τ (95% CI)", "NDCG (95% CI)"])
    for col_idx, (measure, lo_col, hi_col) in enumerate(
        [("Kendall τ", "τ_lo", "τ_hi"), ("NDCG", "ndcg_lo", "ndcg_hi")], start=1
    ):
        vals   = acc_df[measure].values
        err_lo = (vals - acc_df[lo_col].values).clip(0)
        err_hi = (acc_df[hi_col].values - vals).clip(0)
        acc_fig.add_trace(go.Bar(
            x=vals, y=acc_df["Metric"], orientation="h",
            marker_color=[metric_colors[m] for m in acc_df["Metric"]],
            error_x=dict(type="data", symmetric=False,
                         array=err_hi.tolist(), arrayminus=err_lo.tolist()),
            text=[f"{v:.3f} [{lo:.2f},{hi:.2f}]"
                  for v, lo, hi in zip(vals, acc_df[lo_col], acc_df[hi_col])],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)
        ref_range = [-1.05, 1.05] if measure == "Kendall τ" else [-0.05, 1.05]
        acc_fig.update_xaxes(title_text=measure, range=ref_range,
                             zeroline=True, zerolinecolor="black", row=1, col=col_idx)
        acc_fig.update_yaxes(title_text="Metric", row=1, col=col_idx)
    acc_fig.update_layout(
        height=80 + 40 * len(acc_df), width=1000,
        template="plotly_white",
        margin={"l": 160, "b": 50, "r": 120, "t": 50},
    )
    figures.append(acc_fig)
    best = acc_df.iloc[-1]["Metric"]
    titles.append(
        f"Pooled Ranking Accuracy: Kendall τ and NDCG with Bootstrap 95% CI "
        f"(n={n_total}, best NDCG: {best})"
    )

    # ── 5. Precision@k ─────────────────────────────────────────────────────
    n_comp = len(valid)
    k_values = list(range(1, n_comp + 1))
    prec_fig = go.Figure()
    for col, label in pred_metrics:
        scores = valid[col].values
        prec_fig.add_trace(go.Scatter(
            x=k_values,
            y=[_precision_at_k(actual_vals, scores, k) for k in k_values],
            mode="lines+markers", name=label, marker=dict(size=5),
        ))
    prec_fig.add_trace(go.Scatter(
        x=k_values, y=[k / n_comp for k in k_values],
        mode="lines", name="Random baseline",
        line=dict(dash="dash", color="grey", width=1),
    ))
    prec_fig.update_layout(
        height=450, width=750, template="plotly_white",
        xaxis=dict(title="k (number of top components considered)", dtick=max(1, n_comp // 20)),
        yaxis=dict(title="Precision@k", range=[-0.05, 1.05]),
        legend=dict(title="Metric", x=1.01, xanchor="left"),
        margin={"l": 60, "b": 60, "r": 20, "t": 40},
    )
    figures.append(prec_fig)
    titles.append(f"Pooled Precision@k (n={n_total})")

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    eval.write_all_in_one(
        figures, "Figure", Path("."),
        output_dir + "/cp_metric_vs_actual_pooled.html",
        titles=titles,
    )
    print(f"Written pooled metric comparison (n={n_total}) to {output_dir}/cp_metric_vs_actual_pooled.html")


def evaluate(folder_id):
    fail_df, perf_df, metrics_df, net_type_to_net = load_dfs(folder_id)
    impact_df = create_or_load_impact_df(
        fail_df, perf_df, metrics_df, folder_id
    )
    impact_df = extend_impact_df(net_type_to_net, metrics_df, impact_df)

    per_network_dfs = []

    for network_type, monee_net in net_type_to_net.items():
        # if network_type == "large_urban_balanced":
        #     result = run_energy_flow(monee_net, solver=PyomoSolver(), solver_name="gurobi")
        #     print(result.full())
        #     print(result.network.as_result_dataframe_dict_str())
        #     monee_net = result.network
        # else:
        #     continue

        Path(OUTPUT + f"/{network_type}").mkdir(exist_ok=True, parents=True)

        perf_df_nt = perf_df[perf_df["network_type"] == network_type]
        impact_df_nt = impact_df[impact_df["network_type"] == network_type]
        metrics_df_nt = metrics_df[metrics_df["network_type"] == network_type]

        resilience_per_scenario(perf_df_nt, network_type)
        impact_aggregated_component_carrier(impact_df_nt, network_type)
        impact_over_metrics(
            {network_type: monee_net},
            impact_df_nt,
            metrics_df_nt,
            network_type,
            ["betweenness_centrality", "degree", "vc", "katz"],
        )
        net_df = cp_metric_vs_actual_impact(monee_net, impact_df_nt, network_type)
        if net_df is not None and len(net_df) > 0:
            per_network_dfs.append(net_df)

    if len(per_network_dfs) > 1:
        pooled_df = pandas.concat(per_network_dfs, ignore_index=True)
        pooled_metric_comparison(pooled_df, OUTPUT + "/pooled")
    elif len(per_network_dfs) == 1:
        print("Only one network type found — skipping pooled analysis.")


def main():
    evaluate(INPUT)


if __name__ == "__main__":
    main()
