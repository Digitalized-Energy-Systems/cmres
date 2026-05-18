from typing import Dict
import argparse
import os
import sys
import pickle
import traceback
from pathlib import PurePath, Path
from statistics import mean

import cmres.evaluation.evaluation as eval
from monee import Network, run_energy_flow, run_energy_flow_optimization, PyomoSolver
from monee.model.core import Node
import monee.problem as mp
import scipy.stats

import pandas
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent))
from cp_metric import mes_all_components_metric, CPMetricConfig
import eval_common as _ec
# Shared plot-style helpers (sector legend, bar marker, error-bar shape)
# so every bar produced by this module matches the E16 bars 1-to-1.
from cmres_eval_plots import (
    SECTOR_COLORS, SECTOR_PRETTY,
    bar_error_kwargs, outlined_marker,
)

# Default simulation-output directory used when no --input-dir is passed
# on the CLI. The slurm worker submits without arguments and inherits this.
INPUT = "/home/rschrage/experiments/0508/res"
OUTPUT = "data/out"
SMALL_NUMBER = 0.00000000001

# Re-exports from eval_common so existing code (and external callers that
# imported these names from this module) keep working unchanged. The full
# rationale for MC_FAILED_EPS lives in eval_common's docstring.
MC_FAILED_EPS = _ec.MC_FAILED_EPS
CP_TYPE_SET = _ec.CP_TYPE_SET

TYPE_TO_CARRIER = {
    "Junction": "heat/gas",
    "Bus": "electricity",
    "CHP": "multi",
    "CHPHG": "multi",
    "GasPipe": "gas",
    "GenericPowerBranch": "electricity",
    "PowerLine": "electricity",
    "PowerGenerator": "electricity",
    "PowerToGas": "multi",
    "GasToPower": "multi",
    "PowerToHeat": "multi",
    "PowerToHeatHG": "multi",
    "GasToHeatHG": "multi",
    "WaterPipe": "heat",
    "PowerToHeatControlNode": "multi",
    "CHPControlNode": "multi",
    "CHPHGControlNode": "multi",
    "SubHG": "heat",
    "PowerLoad": "electricity",
    "HeatLoad": "heat",
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
    # Branches first: their id is rendered as a tuple "(from, to, key)" so the
    # whole string ends with ")". Must precede the CHP / PowerToHeat substring
    # check because HG-variant branches (PowerToHeatHG, GasToHeatHG) contain
    # "PowerToHeat" / "Heat" in their class repr.
    if id_str.endswith(")"):
        return "branch"
    if ".child." in id_str:
        return "child"
    if "CHP" in id_str or "PowerToHeat" in id_str:
        return "compound"
    return "node"


def _type_name(t) -> str:
    """Normalise an ``impact_df.type`` value to a bare class name.

    ``create_metrics_df`` populates ``type`` with the actual Python
    ``type`` object — but ``impact_df`` is round-tripped through CSV
    by ``create_or_load_impact_df``, so on subsequent runs the same
    column comes back as ``"<class 'monee.model.child.PowerLine'>"``.
    Neither form equals the bare ``"PowerLine"`` key in ``TYPE_TO_CARRIER``.
    This helper accepts both and returns the bare class name. NaN /
    ``None`` (e.g. the rows ``extend_impact_df`` appends without a
    ``type``) return an empty string so they map to ``NaN`` in
    ``TYPE_TO_CARRIER`` lookups and get filtered out cleanly.
    """
    if t is None:
        return ""
    # NaN floats from pandas concat.
    try:
        if isinstance(t, float) and t != t:  # NaN check w/o numpy import
            return ""
    except Exception:
        pass
    if isinstance(t, type):
        return t.__name__
    s = str(t)
    if s.startswith("<class '") and s.endswith("'>"):
        return s[len("<class '"):-len("'>")].rsplit(".", 1)[-1]
    return s


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
            # Signed degradation: out_mean − in_mean. With in_mean = average
            # carrier loss while this component is faulted and out_mean = same
            # while it is not, a component that *causes* extra loss when it
            # fails has in_mean > out_mean and therefore a NEGATIVE impact.
            # Downstream consumers (cp_metric_vs_actual_impact,
            # impact_aggregated_component_carrier, impact_over_metrics) all
            # take |impact|, so the sign is informational only — we keep it as
            # documentation of which side of the comparison is larger.
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
    for edge in monee_net._network_internal.edges:
        monee_net._network_internal.edges[edge]["weight"] = 1.0

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

# Single source of truth for human-readable scenario labels is
# ``grid_topology_table.SCENARIO_LABEL`` — the same short tags used in
# the dissertation LaTeX table ("LV-no", "LV-s", "LV-m-eq", …). The
# import is lazy / guarded so this module stays importable in headless
# / minimal environments that don't pull in ``test_grids``.
try:
    from grid_topology_table import SCENARIO_LABEL as _GTT_SCENARIO_LABEL  # noqa: E402
except Exception:  # pragma: no cover
    _GTT_SCENARIO_LABEL = {
        "simbench_lv_no":                "LV-no",
        "simbench_lv_low":               "LV-s",
        "simbench_lv":                   "LV-m",
        "simbench_lv_high":              "LV-l",
        "simbench_lv_xl":                "LV-xl",
        "simbench_lv_xxl":               "LV-xxl",
        "simbench_lv_low_same_cap":      "LV-s-eq",
        "simbench_lv_same_cap":          "LV-m-eq",
        "simbench_lv_high_same_cap":     "LV-l-eq",
        "simbench_lv_xl_same_cap":       "LV-xl-eq",
        "simbench_lv_xxl_same_cap":      "LV-xxl-eq",
    }

# Public alias kept for backwards compatibility — downstream modules
# (cmres_eval_plots, single_removal_shed_plots) import this name to derive
# the canonical scenario ordering and to map scenario keys → labels.
SCENARIO_NAME_MAP = dict(_GTT_SCENARIO_LABEL)


def pretty_scenario(name) -> str:
    """Map a technical scenario / network-type id to its display label.

    Returns the input unchanged when no mapping is registered, so file paths
    and join keys (which still use the raw id) keep working as before.
    """
    if name is None:
        return ""
    return SCENARIO_NAME_MAP.get(str(name), str(name))


def _all_grids_order():
    """Canonical scenario ordering from ``test_grids.ALL_GRIDS``.

    Returns a ``{grid_name: position}`` map. Imported lazily so this module
    works in environments where ``test_grids`` (and its heavy simbench /
    monee deps) hasn't been put on the path.
    """
    try:
        from test_grids import ALL_GRIDS  # type: ignore
        return {k: i for i, k in enumerate(ALL_GRIDS.keys())}
    except Exception:
        # Fallback: derive from SCENARIO_NAME_MAP insertion order. Same
        # ordering as long as the map is kept in sync with ALL_GRIDS.
        return {k: i for i, k in enumerate(SCENARIO_NAME_MAP.keys())}


def scenario_sort_key(name) -> tuple:
    """Sort key that orders network-type ids by their position in
    ``ALL_GRIDS`` (i.e. matching ``test_grids.py``). Unknown ids land at
    the end in lexicographic order, so a stray scenario name doesn't
    crash the figure pipeline."""
    order = _all_grids_order()
    s = str(name)
    return (order.get(s, len(order)), s)


def sort_scenarios(names):
    """Return ``names`` sorted by :func:`scenario_sort_key`."""
    return sorted(names, key=scenario_sort_key)


def resilience_per_scenario(perf_df: pandas.DataFrame, folder_id):
    # experiment, id 0 1 2
    # Per run: average instantaneous load shed across the 16-step horizon (MW).
    # Across runs: MC expectation. Result is bounded by total grid demand.
    resilience_per_carrier_per_scenario = (
        perf_df.groupby(["network_type", "experiment", "id"])[["0", "1", "2"]]
        .mean()
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
    # Replace technical scenario keys with display labels (file paths still
    # use the raw key, only the chart-visible columns are remapped).
    resilience_per_carrier_per_scenario["experiment"] = (
        resilience_per_carrier_per_scenario["experiment"].map(pretty_scenario)
    )
    eval.create_bar(
        resilience_per_carrier_per_scenario,
        x_label="experiment",
        y_label="resilience_mean",
        color="carrier",
        color_discrete_map=eval.NETWORK_COLOR_MAP,
        pattern_shape_map=eval.NETWORK_PATTERN_MAP,
        legend_text="carrier",
        template=eval.CMRES_TEMPLATE,
        yaxis_title="mean performance loss in MW",
        xaxis_title="scenario",
        title="Performance drop by scenario, by carrier",
        barmode="group",
        width=1200,
        height=450,
    )
    unique_network_types = sort_scenarios(
        pandas.unique(resilience_per_carrier_per_scenario["network_type"])
    )
    unique_experiments = list(
        pandas.unique(resilience_per_carrier_per_scenario["experiment"])
    )

    # Sort the inner per-carrier rows by the canonical grid order so the
    # values line up with ``unique_network_types`` when zipped into the
    # grouped-bar chart.
    _sort_keys = (
        resilience_per_carrier_per_scenario["network_type"].map(scenario_sort_key)
    )
    resilience_per_carrier_per_scenario = (
        resilience_per_carrier_per_scenario.assign(_sort_key=_sort_keys)
        .sort_values(by=["_sort_key", "experiment"])
        .drop(columns="_sort_key")
    )

    resilience_per_carrier_per_scenario_hist_2 = (
        eval.create_multilevel_grouped_bar_chart(
            [
                list(
                    resilience_per_carrier_per_scenario[
                        resilience_per_carrier_per_scenario["carrier"] == carrier
                    ]["resilience_mean"]
                )
                for carrier in ["electricity", "heat", "gas"]
            ],
            ["#ffa000", "#d32f2f", "#388e3c"],
            ["electricity", "heat", "gas"],
            [f"<b>{pretty_scenario(net_type)}</b>" for net_type in unique_network_types],
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
    "CPs": [
        "PowerToHeat", "PowerToHeatHG", "GasToHeatHG",
        "CHP", "CHPHG",
        "PowerToGas", "GasToPower",
    ],
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
    # Filter out per-carrier rows whose component never failed in MC (or only
    # had a vanishing share of impact in this carrier). Keeps scatter and
    # correlation views from being dominated by the long flat tail at impact=0
    # — same rationale as cp_metric_vs_actual_impact. The graph/network view
    # later in the function uses the network layout itself, so unfiltered
    # components just don't get coloured rather than being misrepresented.
    n_before = len(metric_impact_df)
    metric_impact_df = metric_impact_df[
        metric_impact_df["impact"] > MC_FAILED_EPS
    ].reset_index(drop=True)
    print(
        f"impact_over_metrics[{folder_id}]: filtered to MC-sampled rows "
        f"({len(metric_impact_df)} of {n_before}, eps={MC_FAILED_EPS:g})"
    )
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
                metric_impact_df_carrier_with_types = metric_impact_df_carrier[
                    metric_impact_df_carrier["type_y"].isin(value)
                ]
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
                    template=eval.CMRES_TEMPLATE,
                )
            )
            titles.append(f"graph of the components' {carrier_name}-impact ({net_type})")
            figures.append(
                eval.create_networkx_plot(
                    monee_net,
                    metric_impact_df_carrier_net_type,
                    color_name="impact",
                    color_legend_text=f"{carrier_name}-impact",
                    template=eval.CMRES_TEMPLATE,
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
            metric_impact_df_carrier_with_types = metric_impact_df_all_carrier[
                metric_impact_df_all_carrier["type_y"].isin(value)
            ]
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
            titles.append(f"{metric} to the {key}' impact (all carriers)")

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
                template=eval.CMRES_TEMPLATE,
            )
        )
        titles.append(f"graph of the components' impact ({net_type})")
        figures.append(
            eval.create_networkx_plot(
                monee_net,
                metric_impact_df_carrier_net_type,
                color_name="impact",
                color_legend_text="impact",
                template=eval.CMRES_TEMPLATE,
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
            template=eval.CMRES_TEMPLATE,
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
            template=eval.CMRES_TEMPLATE,
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
            template=eval.CMRES_TEMPLATE,
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
            template=eval.CMRES_TEMPLATE,
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
        + average_impact_per_carrier_net_type["network_type"].map(pretty_scenario)
    )
    impact_per_carrier_net_type = (
        new_impact_df.groupby(["type_carrier", "carrier", "network_type"])
        .sum()
        .reset_index()
    )
    impact_per_carrier_net_type["carrier_net_type"] = (
        impact_per_carrier_net_type["type_carrier"].astype(str)
        + "-"
        + impact_per_carrier_net_type["network_type"].map(pretty_scenario)
    )
    figures += [
        eval.create_bar(
            average_impact_per_carrier_net_type,
            x_label="carrier_net_type",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template=eval.CMRES_TEMPLATE,
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
            template=eval.CMRES_TEMPLATE,
            yaxis_title="impact",
            xaxis_title="carrier-density",
            showlegend=False,
        )
    ]
    titles.append("Total impacts by carrier type and density")

    average_impact_per_net_type = (
        new_impact_df.groupby(["carrier", "network_type"]).mean().reset_index()
    )
    average_impact_per_net_type["network_type"] = (
        average_impact_per_net_type["network_type"].map(pretty_scenario)
    )
    total_impact_per_net_type = (
        new_impact_df.groupby(["carrier", "network_type"]).sum().reset_index()
    )
    total_impact_per_net_type["network_type"] = (
        total_impact_per_net_type["network_type"].map(pretty_scenario)
    )

    figures += [
        eval.create_bar(
            average_impact_per_net_type,
            x_label="network_type",
            y_label="impact",
            color="carrier",
            color_discrete_map=eval.NETWORK_COLOR_MAP,
            legend_text="by carrier",
            template=eval.CMRES_TEMPLATE,
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
            template=eval.CMRES_TEMPLATE,
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


# Backwards-compat re-exports. The canonical implementations now live in
# ``eval_common`` so cmres_eval and this module use the same logic.
_COMPOUND_CP_TYPES = _ec._COMPOUND_CP_TYPES
_NON_CP_BRANCH_TYPES = _ec._NON_CP_BRANCH_TYPES
_build_branch_lookup = _ec.build_branch_lookup
_match_impact_id = _ec.match_impact_id


def cp_metric_vs_actual_impact(monee_net, impact_df_nt, network_type):
    """E1 / E5: per-network ρ + ranking battery.

    Implements CMRES evaluation experiment **E1 (validation)** and feeds
    ``cp_only_metric_comparison`` (**E5**, per-CP-type breakdown). Returns
    the unfiltered matched dataframe so cmres_eval can reuse it for
    E2 ablations, E3 density, E4 distribution, etc., without re-running
    ``mes_all_components_metric``.
    """
    try:
        df_scores, _ = mes_all_components_metric(monee_net, cfg=CPMetricConfig())
    except Exception as e:
        print(f"CP metric failed for {network_type}: {e}")
        raise e

    df_all = _ec.build_matched_df(df_scores, impact_df_nt)
    if df_all.empty:
        print(f"No metric/impact matches found for {network_type}")
        return
    # Tag with the scenario name BEFORE the filtered slice is taken, so the
    # ``network_type`` column survives both into the local filtered ``df``
    # AND into the ``df_all`` we return. ``pooled_metric_comparison`` later
    # concatenates these returned dataframes and groups on ``network_type``;
    # without this assignment it crashes with KeyError: 'network_type'.
    df_all = df_all.copy()
    df_all["network_type"] = network_type
    # Primary view: only components the MC failure model actually sampled (and
    # therefore can produce a non-trivial actual_total). Components with zero
    # impact are unrankable and inflate NDCG / suppress P@k toward random; see
    # MC_FAILED_EPS docstring above.
    df = df_all[df_all["actual_total"] > MC_FAILED_EPS].reset_index(drop=True)
    n_all = len(df_all)
    n_mc = len(df)
    if n_mc < 3:
        print(
            f"[{network_type}] only {n_mc} components have actual>0; falling "
            f"back to full set (n={n_all}) for ranking metrics"
        )
        df = df_all.copy()
    figures = []
    titles = []

    # Aliased so existing call sites (ρ-bar, scatter panels, comparison
    # panel) don't have to be touched.
    _spearman_with_ci = _ec.spearman_with_ci

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
    from plotly.subplots import make_subplots

    # "score_no_topo" = pure PTDF stress (carrier-weighted total_stress) with
    # NO throughput, NO topo factor, NO input-adequacy gate. The earlier
    # derivation `predicted_score / topo_factor` silently picked up throughput
    # and (for CPs) the input_adequacy gate, so the label "PTDF stress only"
    # didn't actually mean what it said.
    df["score_no_topo"] = df["predicted_stress"]
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
        colors = eval.PALETTE_QUAL
        type_color = {t: colors[i % len(colors)] for i, t in enumerate(cp_types)}

        panels = [
            # x_col, subplot title (with ρ + 95% CI), x-axis label (exact formula)
            ("predicted_score",
             f"Full: PTDF stress + phys. BC<br>{_rho_label(rho_with, pval_with, ci_lo_with, ci_hi_with)}",
             "τ · PTDF_stress · (1 + α·BC_phys) · adequacy"),
            ("score_no_topo",
             f"PTDF stress only<br>{_rho_label(rho_without, pval_without, ci_lo_without, ci_hi_without)}",
             "PTDF_stress (carrier-weighted)"),
            ("score_topo_only",
             f"Phys. BC only, no stress<br>{_rho_label(rho_topo_only, pval_topo_only, ci_lo_topo_only, ci_hi_topo_only)}",
             "Phys. betweenness centrality"),
            ("stress_bc",
             f"Stress-weighted BC only<br>{_rho_label(rho_stress_bc, pval_stress_bc, ci_lo_stress_bc, ci_hi_stress_bc)}",
             "Stress-weighted betweenness centrality"),
            ("local_score",
             f"1-hop local: loading + critical neighbours<br>{_rho_label(rho_local, pval_local, ci_lo_local, ci_hi_local)}",
             "loading · (1 + crit.nbrs) · n_carriers"),
            ("self_score",
             f"0-hop self: own loading only<br>{_rho_label(rho_self, pval_self, ci_lo_self, ci_hi_self)}",
             "loading · n_carriers"),
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
            template=eval.CMRES_TEMPLATE,
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

    # Canonical 10-metric set shared with E16 (cmres_eval) — see
    # eval_common.CORE_METRICS for the single source of truth. ``actual_total``
    # is appended here as the MC ground truth referenced by the rank-accuracy
    # plots below.
    METRICS = list(_ec.CORE_METRICS) + [("actual_total", "Actual (MC)")]
    # network_type is already set on df_all (and inherited by this filtered
    # df) right after build_matched_df. ``df.copy()`` here keeps the original
    # safe from the inplace ``df[metric_cols] = df[metric_cols].replace(...)``
    # that follows.
    df = df.copy()

    # Replace any ±inf with NaN so dropna catches them too — without this they
    # would survive the dropna, then crash plotly / Spearman.
    metric_cols = [col for col, _ in METRICS]
    df[metric_cols] = df[metric_cols].replace([_np.inf, -_np.inf], _np.nan)

    # Diagnostic: where do the non-finite metric values come from? Report
    # per-column counts and the offending cp_id / cp_type so the issue can be
    # traced back into mes_all_components_metric (PTDF stress, throughput,
    # topology factor, etc.).
    nan_per_col = {c: int(df[c].isna().sum()) for c in metric_cols if df[c].isna().any()}
    if nan_per_col:
        print(
            f"[warn] cp_metric_vs_actual_impact[{network_type}]: "
            f"NaN/inf in metric columns → {nan_per_col}"
        )
        bad = df[df[metric_cols].isna().any(axis=1)][["cp_id", "cp_type"] + metric_cols]
        print(f"[warn] offending rows (showing up to 8):\n{bad.head(8).to_string(index=False)}")

    # Drop rows missing any metric *before* ranking — otherwise the rank()
    # output keeps NaN slots and astype(int) raises IntCastingNaNError.
    valid = df.dropna(subset=metric_cols).copy()

    for col, _label in METRICS:
        valid[f"rank_{col}"] = valid[col].rank(ascending=False, method="min").astype(int)

    # Carry ranks back onto df for the bump-chart / df-iteration code below.
    # Rows that were dropped get NaN ranks (Int64 nullable so we can persist).
    for col, _label in METRICS:
        df[f"rank_{col}"] = (
            df[col].rank(ascending=False, method="min").astype("Int64")
        )

    if len(valid) < len(df):
        print(
            f"[info] cp_metric_vs_actual_impact[{network_type}]: "
            f"{len(df) - len(valid)}/{len(df)} components dropped from rank/ρ analysis "
            f"due to non-finite metric values."
        )

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

    _palette = eval.PALETTE_QUAL[:len(rho_df)]
    rho_bar = go.Figure(go.Bar(
        x=rho_df["Metric"],
        y=rho_df["Spearman ρ"],
        error_y=bar_error_kwargs(
            err_hi=rho_df["err_hi"].tolist(),
            err_lo=rho_df["err_lo"].tolist(),
        ),
        text=[f"ρ={r:.2f} [{lo:.2f},{hi:.2f}]<br>p={p:.3f}"
              for r, p, lo, hi in zip(
                  rho_df["Spearman ρ"], rho_df["p-value"],
                  rho_df["ci_lo"], rho_df["ci_hi"])],
        textposition="outside",
        marker=outlined_marker(_palette[0]),  # uniform fill — palette via individual bars below
    ))
    # Recolour bars per-metric using the qualitative palette while keeping the
    # E16 outline.
    rho_bar.data[0].marker.color = list(_palette)

    rho_bar.update_layout(
        # Match E16 vertical-bar dim: ~460h × (120 + 110·N)w so per-metric
        # bar width is the same as in cmres_eval_plots' E16 sector bars.
        height=460, width=120 + 110 * max(len(rho_df), 1),
        template=eval.CMRES_TEMPLATE,
        yaxis=dict(title="Spearman ρ vs Actual (95% CI)", range=[-1.15, 1.15],
                   gridcolor="#e5e5e5",
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
        template=eval.CMRES_TEMPLATE,
        margin={"l": 120, "b": 120, "r": 20, "t": 40},
        xaxis=dict(title="Metric"),
        yaxis=dict(title="Metric"),
    )
    figures.append(heatmap_fig)
    titles.append("Pairwise Rank Correlation (Spearman ρ) Between All Metrics")

    # 3. Bump chart – rank of each component across all metrics
    metric_names = [label for _, label in METRICS]
    rank_cols    = [f"rank_{col}" for col, _ in METRICS]
    cp_colors    = eval.PALETTE_QUAL
    cp_type_color = {t: cp_colors[i % len(cp_colors)]
                     for i, t in enumerate(valid["cp_type"].unique())}

    seen_cp_types = set()
    bump_fig = go.Figure()
    # Bump chart only plots components with finite ranks across every metric;
    # `valid` already excludes rows containing NaN/inf in any metric column.
    for _, row in valid.iterrows():
        ranks = [int(row[rc]) for rc in rank_cols]
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
        height=max(400, 20 * len(valid)),
        width=900,
        template=eval.CMRES_TEMPLATE,
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
    # Aliased re-exports so the call sites stay short.
    _ndcg = _ec.ndcg
    _rndcg = _ec.random_normalized_ndcg
    _default_k = _ec.default_ndcg_k
    _precision_at_k = _ec.precision_at_k
    _bootstrap_ci = _ec.bootstrap_ci
    _bootstrap_ndcg_ci = _ec.bootstrap_ndcg_ci

    pred_metrics = [(col, label) for col, label in METRICS if col != "actual_total"]
    actual_vals  = valid["actual_total"].values
    # NDCG@k cutoff scales with the per-network component count so the
    # top-quintile rule applies consistently (clamped 5..20 — see
    # eval_common.default_ndcg_k for the rationale).
    _k_cut = _default_k(len(actual_vals))

    # 4a. Summary bar chart: Kendall τ, NDCG@k, and rNDCG per metric (the
    # full-list NDCG saturates on heavy-tailed ``actual_total`` and is
    # kept only for backwards compatibility — see eval_common.ndcg docs).
    # NDCG / rNDCG CIs use the vectorised ``bootstrap_ndcg_ci`` — the
    # earlier Python-loop bootstrap with an inner 80-permutation MC was
    # the dominant cost of this block.
    _rng = _np.random.default_rng(42)
    acc_rows = []
    for col, label in pred_metrics:
        scores = valid[col].values
        tau  = float(scipy.stats.kendalltau(scores, actual_vals).statistic)
        ndcg = _ndcg(actual_vals, scores)
        ndcg_at_k = _ndcg(actual_vals, scores, k=_k_cut)
        rndcg = _rndcg(actual_vals, scores, k=_k_cut)
        tau_lo, tau_hi = _bootstrap_ci(
            lambda a, p: float(scipy.stats.kendalltau(p, a).statistic),
            actual_vals, scores, rng=_rng)
        ndcg_lo, ndcg_hi = _bootstrap_ndcg_ci(actual_vals, scores, rng=_rng)
        ndcgk_lo, ndcgk_hi = _bootstrap_ndcg_ci(
            actual_vals, scores, k=_k_cut, rng=_rng,
        )
        rndcg_lo, rndcg_hi = _bootstrap_ndcg_ci(
            actual_vals, scores, k=_k_cut, normalize_random=True, rng=_rng,
        )
        acc_rows.append({
            "Metric": label,
            "Kendall τ": tau, "τ_lo": tau_lo, "τ_hi": tau_hi,
            "NDCG": ndcg, "ndcg_lo": ndcg_lo, "ndcg_hi": ndcg_hi,
            "NDCG@k": ndcg_at_k, "ndcgk_lo": ndcgk_lo, "ndcgk_hi": ndcgk_hi,
            "rNDCG@k": rndcg, "rndcg_lo": rndcg_lo, "rndcg_hi": rndcg_hi,
        })

    # Sort by rNDCG@k (the most discriminating of the three NDCG variants
    # — strips out the random-baseline inflation that masks differences
    # between metrics on heavy-tailed actual_total).
    acc_df = pandas.DataFrame(acc_rows).sort_values("rNDCG@k", ascending=True)

    panels = [
        ("Kendall τ",       "τ_lo",     "τ_hi",     [-1.05, 1.05]),
        (f"NDCG@{_k_cut}",  "ndcgk_lo", "ndcgk_hi", [-0.05, 1.05]),
        (f"rNDCG@{_k_cut}", "rndcg_lo", "rndcg_hi", [-1.05, 1.05]),
        ("NDCG",            "ndcg_lo",  "ndcg_hi",  [-0.05, 1.05]),
    ]
    measure_to_col = {
        "Kendall τ": "Kendall τ",
        f"NDCG@{_k_cut}": "NDCG@k",
        f"rNDCG@{_k_cut}": "rNDCG@k",
        "NDCG": "NDCG",
    }
    acc_fig = make_subplots(
        rows=1, cols=len(panels),
        subplot_titles=[f"{m} (95% CI)" for m, *_ in panels],
    )
    metric_colors = {row["Metric"]: eval.PALETTE_QUAL[i % 10]
                     for i, row in acc_df.iterrows()}
    for col_idx, (measure, lo_col, hi_col, ref_range) in enumerate(panels, start=1):
        data_col = measure_to_col[measure]
        vals   = acc_df[data_col].values
        err_lo = (vals - acc_df[lo_col].values).clip(0)
        err_hi = (acc_df[hi_col].values - vals).clip(0)
        acc_fig.add_trace(go.Bar(
            x=vals,
            y=acc_df["Metric"],
            orientation="h",
            marker=dict(
                color=[metric_colors[m] for m in acc_df["Metric"]],
                line=dict(color="#222", width=0.4),
            ),
            error_x=bar_error_kwargs(
                err_hi=err_hi.tolist(), err_lo=err_lo.tolist(),
            ),
            text=[f"{v:.3f} [{lo:.2f},{hi:.2f}]"
                  for v, lo, hi in zip(vals, acc_df[lo_col], acc_df[hi_col])],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)
        acc_fig.update_xaxes(title_text=measure, range=ref_range,
                             gridcolor="#e5e5e5",
                             zeroline=True, zerolinecolor="black", zerolinewidth=1,
                             row=1, col=col_idx)
        acc_fig.update_yaxes(title_text="Metric" if col_idx == 1 else "",
                             row=1, col=col_idx)

    acc_fig.update_layout(
        # 4 horizontal-bar panels side by side. Per-panel width matches
        # the E16 single-bar figure (~460 px); per-bar height bumped to
        # 60 px so individual bars are visually similar to E16's bars.
        height=120 + 60 * len(acc_df), width=460 * len(panels) + 60,
        template=eval.CMRES_TEMPLATE,
        margin={"l": 160, "b": 50, "r": 80, "t": 60},
    )
    figures.append(acc_fig)
    best = acc_df.iloc[-1]["Metric"]
    titles.append(
        f"Ranking Accuracy: Kendall τ, NDCG@{_k_cut}, rNDCG@{_k_cut}, NDCG "
        f"with Bootstrap 95% CI (best rNDCG@{_k_cut}: {best})"
    )

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
        template=eval.CMRES_TEMPLATE,
        xaxis=dict(title="k (number of top components considered)", dtick=1),
        yaxis=dict(title="Precision@k", range=[-0.05, 1.05],
                   zeroline=True, zerolinecolor="lightgrey"),
        margin={"l": 60, "b": 60, "r": 20, "t": 40},
        legend=dict(title="Metric", x=1.01, xanchor="left"),
    )
    figures.append(prec_fig)
    titles.append(
        f"Precision@k: Fraction of True Top-k Components Correctly Identified "
        f"(n={len(valid)} MC-sampled of {n_all} matched)"
    )

    # ── Filtered-vs-unfiltered ρ comparison ────────────────────────────────
    # Each metric scored against actual_total on (a) all matched components,
    # which includes the long flat tail of zero-impact rows from the MC, and
    # (b) only MC-sampled components (actual > MC_FAILED_EPS). The filtered
    # value is the one used in the figures above; the unfiltered value is
    # what you'd see if the never-failed mass were left in. Same metrics,
    # same df, just different population — the gap shows how much the zero
    # tail was distorting things.
    if n_mc >= 3 and n_all > n_mc:
        df_all_local = df_all.copy()
        # score_no_topo = pure PTDF stress (carrier-weighted total_stress).
        # See the comment on the primary derivation above for why this is no
        # longer `predicted_score / topo_factor`.
        if "score_no_topo" not in df_all_local.columns:
            df_all_local["score_no_topo"] = df_all_local["predicted_stress"]
        if "score_topo_only" not in df_all_local.columns:
            df_all_local["score_topo_only"] = df_all_local["topo_bc"]

        cmp_rows = []
        for col, label in METRICS[:-1]:
            try:
                rho_all, _, lo_all, hi_all = _spearman_with_ci(
                    df_all_local[col], df_all_local["actual_total"]
                )
            except Exception:
                rho_all, lo_all, hi_all = float("nan"), float("nan"), float("nan")
            try:
                rho_mc, _, lo_mc, hi_mc = _spearman_with_ci(
                    df[col], df["actual_total"]
                )
            except Exception:
                rho_mc, lo_mc, hi_mc = float("nan"), float("nan"), float("nan")
            cmp_rows.append({
                "Metric": label,
                "rho_all": rho_all, "ci_lo_all": lo_all, "ci_hi_all": hi_all,
                "rho_mc": rho_mc, "ci_lo_mc": lo_mc, "ci_hi_mc": hi_mc,
            })
        cmp_df = pandas.DataFrame(cmp_rows).sort_values("rho_mc", ascending=True)

        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Bar(
            name=f"All matched (n={n_all})",
            x=cmp_df["Metric"], y=cmp_df["rho_all"],
            error_y=bar_error_kwargs(
                err_hi=(cmp_df["ci_hi_all"] - cmp_df["rho_all"]).tolist(),
                err_lo=(cmp_df["rho_all"] - cmp_df["ci_lo_all"]).tolist(),
            ),
            marker=outlined_marker("lightgrey"),
        ))
        cmp_fig.add_trace(go.Bar(
            name=f"MC-sampled (n={n_mc})",
            x=cmp_df["Metric"], y=cmp_df["rho_mc"],
            error_y=bar_error_kwargs(
                err_hi=(cmp_df["ci_hi_mc"] - cmp_df["rho_mc"]).tolist(),
                err_lo=(cmp_df["rho_mc"] - cmp_df["ci_lo_mc"]).tolist(),
            ),
            marker=outlined_marker(eval.PALETTE_QUAL[0]),
        ))
        cmp_fig.update_layout(
            barmode="group",
            # Match E16 grouped-bar dim: 460h × (120 + 110·N)w.
            height=460, width=120 + 110 * max(len(cmp_df), 1),
            template=eval.CMRES_TEMPLATE,
            yaxis=dict(title="Spearman ρ vs Actual (95% CI)", range=[-1.15, 1.15],
                       gridcolor="#e5e5e5",
                       zeroline=True, zerolinecolor="black"),
            xaxis=dict(title="Metric"),
            margin={"l": 60, "b": 100, "r": 20, "t": 50},
            legend=dict(title="Population"),
        )
        figures.append(cmp_fig)
        titles.append(
            f"Filtered vs unfiltered: ρ when including all matched components "
            f"(n={n_all}) vs only MC-sampled (n={n_mc}, actual > {MC_FAILED_EPS:g})"
        )

    eval.write_all_in_one(
        figures, "Figure", Path("."),
        OUTPUT + f"/{network_type}/cp_metric_vs_actual.html",
        titles=titles,
    )
    print(
        f"Written cp_metric_vs_actual.html for {network_type} "
        f"(MC-sampled n={n_mc} of {n_all} matched)"
    )
    # Return the full set so the pooled comparison can apply its own filter
    # and still emit an all-vs-MC comparison panel.
    return df_all


# ──────────────────────────────────────────────────────────────────────────────
# CP-only metric comparison (per-network and pooled) — same metric battery as
# cp_metric_vs_actual_impact / pooled_metric_comparison, restricted to coupling
# points and broken out by cp_type so per-type predictability is visible.
# ──────────────────────────────────────────────────────────────────────────────


def _cp_only_metric_comparison_core(
    df_all: pandas.DataFrame,
    label: str,
    output_path: str,
):
    """Shared core for both the per-network and pooled CP-only views.

    Parameters
    ----------
    df_all : DataFrame
        Full df returned by ``cp_metric_vs_actual_impact``. Must contain
        ``cp_id``, ``cp_type``, ``actual_total`` and the predicted-score columns
        used by the metric battery.
    label : str
        Identifier shown in figure titles (network name or ``"pooled"``).
    output_path : str
        Destination HTML.
    """
    import numpy as _np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # CP-only view uses the same canonical 10 metrics as cp_metric_vs_actual_impact
    # / pooled_metric_comparison / E16 (eval_common.CORE_METRICS), plus the
    # ``actual_total`` MC ground truth and two CP-only diagnostics — the
    # input-adequacy gate (which only fires on CPs) and the no-gate variant
    # of the composite, so the gate's effect can be read off the bar chart.
    METRICS = (
        list(_ec.CORE_METRICS)
        + [
            ("score_no_adequacy", "PTDF + BC (no input gate)"),
            ("input_adequacy",    "Input adequacy alone"),
            ("actual_total",      "Actual (MC)"),
        ]
    )

    df = df_all.copy()
    # "score_no_topo" = pure PTDF stress (carrier-weighted total_stress) —
    # NO throughput, NO topo factor, NO input gate. Use predicted_stress
    # directly so the label "PTDF stress only" actually reflects the formula.
    if "score_no_topo" not in df.columns:
        df["score_no_topo"] = df["predicted_stress"]
    if "score_topo_only" not in df.columns:
        df["score_topo_only"] = df["topo_bc"]
    # score_no_adequacy = the full CP score WITHOUT the input-adequacy gate,
    # i.e. predicted_score / input_adequacy. Lets the heatmap / ρ-bar show how
    # much the conditional gate is doing on top of the existing PTDF + BC.
    if "score_no_adequacy" not in df.columns:
        if "input_adequacy" in df.columns:
            adq = df["input_adequacy"].where(df["input_adequacy"] > 0, float("nan"))
            df["score_no_adequacy"] = df["predicted_score"] / adq
        else:
            df["score_no_adequacy"] = df["predicted_score"]
    if "input_adequacy" not in df.columns:
        df["input_adequacy"] = 1.0

    # 1) Restrict to coupling-point rows.
    df = df[df["cp_type"].astype(str).isin(CP_TYPE_SET)].reset_index(drop=True)
    n_cps_total = len(df)
    if n_cps_total == 0:
        print(f"[cp_only:{label}] no CP rows in matched df — skipping")
        return

    # 2) Drop NaN metric rows so the rank-based metrics don't blow up.
    valid_all = df.dropna(subset=[col for col, _ in METRICS]).reset_index(drop=True)
    valid_mc = valid_all[valid_all["actual_total"] > MC_FAILED_EPS].reset_index(drop=True)
    n_all = len(valid_all)
    n_mc = len(valid_mc)
    primary = valid_mc if n_mc >= 3 else valid_all
    if n_mc < 3 and n_all >= 3:
        print(
            f"[cp_only:{label}] only {n_mc} CPs have actual>0; falling back "
            f"to full CP set (n={n_all}) for ranking metrics"
        )
    elif n_all < 3:
        print(f"[cp_only:{label}] only {n_all} CPs valid — too few for ranking metrics")
        return

    cp_types_present = sorted(primary["cp_type"].unique())
    print(
        f"[cp_only:{label}] CPs: {n_cps_total} matched, {n_all} valid, "
        f"{n_mc} MC-sampled (used). Types: {cp_types_present}"
    )

    # ── Helpers (re-exported from eval_common) ───────────────────────────
    _spearman_with_ci = _ec.spearman_with_ci
    _ndcg = _ec.ndcg
    _precision_at_k = _ec.precision_at_k

    figures = []
    titles  = []
    pred_metrics = [(col, lab) for col, lab in METRICS if col != "actual_total"]
    actual_vals = primary["actual_total"].values

    cp_color_map = {
        t: eval.PALETTE_QUAL[i % 10]
        for i, t in enumerate(cp_types_present)
    }

    # ── 1. Scatter: predicted_score vs actual, coloured by cp_type ────────
    scatter_fig = go.Figure()
    for ct in cp_types_present:
        sub = primary[primary["cp_type"] == ct]
        scatter_fig.add_trace(go.Scatter(
            x=sub["predicted_score"], y=sub["actual_total"],
            mode="markers", name=f"{ct} (n={len(sub)})",
            marker=dict(color=cp_color_map[ct], size=9,
                        line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
            hovertext=sub["cp_id"].astype(str),
        ))
    scatter_fig.update_layout(
        height=500, width=900, template=eval.CMRES_TEMPLATE,
        xaxis=dict(title="Predicted CP Score"),
        yaxis=dict(title="Actual Total Impact (MC)"),
        legend=dict(title="CP Type"),
        margin={"l": 60, "b": 60, "r": 20, "t": 50},
    )
    figures.append(scatter_fig)
    titles.append(f"CP-only [{label}] scatter (n={n_mc} MC-sampled, by CP type)")

    # ── 2. Per-CP-type ρ heatmap: rows = cp_type, cols = metrics ──────────
    # The cell is Spearman ρ for that CP type's predicted-vs-actual; cells with
    # n<3 are NaN (shown blank). This is the KEY view: it makes per-type
    # predictability visible at a glance.
    rho_matrix = _np.full((len(cp_types_present), len(pred_metrics)), _np.nan)
    n_matrix = _np.zeros((len(cp_types_present), len(pred_metrics)), dtype=int)
    for i, ct in enumerate(cp_types_present):
        sub = primary[primary["cp_type"] == ct]
        for j, (col, _lab) in enumerate(pred_metrics):
            n_matrix[i, j] = len(sub)
            if len(sub) >= 3:
                rho, _, _, _ = _spearman_with_ci(sub[col], sub["actual_total"])
                rho_matrix[i, j] = rho

    text_matrix = [
        [
            (
                f"ρ={rho_matrix[i, j]:.2f}<br>n={n_matrix[i, j]}"
                if _np.isfinite(rho_matrix[i, j])
                else f"n={n_matrix[i, j]}<br>(too few)"
            )
            for j in range(len(pred_metrics))
        ]
        for i in range(len(cp_types_present))
    ]
    heatmap_fig = go.Figure(go.Heatmap(
        z=rho_matrix,
        x=[lab for _, lab in pred_metrics],
        y=cp_types_present,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=text_matrix,
        texttemplate="%{text}",
        colorbar=dict(title=dict(text="Spearman ρ", side="right")),
    ))
    heatmap_fig.update_layout(
        height=80 + 60 * len(cp_types_present),
        width=200 + 110 * len(pred_metrics),
        template=eval.CMRES_TEMPLATE,
        margin={"l": 140, "b": 140, "r": 60, "t": 50},
        xaxis=dict(title="Metric", tickangle=-30),
        yaxis=dict(title="CP type"),
    )
    figures.append(heatmap_fig)
    titles.append(
        f"CP-only [{label}] per-type Spearman ρ "
        f"(NaN = fewer than 3 components of that type)"
    )

    # ── 3. Combined ρ bar chart for all CPs together ──────────────────────
    rho_rows = []
    for col, lab in pred_metrics:
        rho, pval, lo, hi = _spearman_with_ci(primary[col], primary["actual_total"])
        rho_rows.append({
            "Metric": lab, "rho": rho, "p": pval, "lo": lo, "hi": hi,
        })
    rho_df = pandas.DataFrame(rho_rows).sort_values("rho", ascending=True)
    rho_fig = go.Figure(go.Bar(
        x=rho_df["rho"], y=rho_df["Metric"], orientation="h",
        error_x=bar_error_kwargs(
            err_hi=(rho_df["hi"] - rho_df["rho"]).clip(lower=0).tolist(),
            err_lo=(rho_df["rho"] - rho_df["lo"]).clip(lower=0).tolist(),
        ),
        text=[f"ρ={r:.2f} [{lo:.2f},{hi:.2f}]<br>p={p:.3f}"
              for r, lo, hi, p in zip(
                  rho_df["rho"], rho_df["lo"], rho_df["hi"], rho_df["p"])],
        textposition="outside",
        marker=outlined_marker(eval.PALETTE_QUAL[0]),
    ))
    rho_fig.update_layout(
        # Horizontal-bar equivalent of the E16 vertical-bar dim — bar
        # thickness 60 px so individual bars match E16 visual weight.
        height=120 + 60 * max(len(rho_df), 1), width=900,
        template=eval.CMRES_TEMPLATE,
        xaxis=dict(title="Spearman ρ vs Actual (95% CI)", range=[-1.05, 1.05],
                   gridcolor="#e5e5e5",
                   zeroline=True, zerolinecolor="black"),
        yaxis=dict(title="Metric"),
        margin={"l": 200, "b": 50, "r": 160, "t": 50},
    )
    figures.append(rho_fig)
    titles.append(f"CP-only [{label}] combined Spearman ρ (n={len(primary)})")

    # ── 4. Kendall τ + NDCG@k + rNDCG@k (+ legacy NDCG) with bootstrap CI ─
    # NDCG / rNDCG CIs use the vectorised ``bootstrap_ndcg_ci``.
    _bootstrap_ci = _ec.bootstrap_ci
    _bootstrap_ndcg_ci = _ec.bootstrap_ndcg_ci
    _rndcg = _ec.random_normalized_ndcg
    _k_cut = _ec.default_ndcg_k(len(actual_vals))

    rng = _np.random.default_rng(42)
    acc_rows = []
    for col, lab in pred_metrics:
        scores = primary[col].values
        if len(scores) < 3:
            continue
        tau = float(scipy.stats.kendalltau(scores, actual_vals).statistic)
        ndcg = _ndcg(actual_vals, scores)
        ndcgk = _ndcg(actual_vals, scores, k=_k_cut)
        rndcg = _rndcg(actual_vals, scores, k=_k_cut)
        tau_lo, tau_hi = _bootstrap_ci(
            lambda a, p: float(scipy.stats.kendalltau(p, a).statistic),
            actual_vals, scores, rng=rng,
        )
        ndcg_lo, ndcg_hi = _bootstrap_ndcg_ci(actual_vals, scores, rng=rng)
        ndcgk_lo, ndcgk_hi = _bootstrap_ndcg_ci(
            actual_vals, scores, k=_k_cut, rng=rng,
        )
        rndcg_lo, rndcg_hi = _bootstrap_ndcg_ci(
            actual_vals, scores, k=_k_cut, normalize_random=True, rng=rng,
        )
        acc_rows.append({
            "Metric": lab,
            "tau": tau, "tau_lo": tau_lo, "tau_hi": tau_hi,
            "ndcg": ndcg, "ndcg_lo": ndcg_lo, "ndcg_hi": ndcg_hi,
            "ndcgk": ndcgk, "ndcgk_lo": ndcgk_lo, "ndcgk_hi": ndcgk_hi,
            "rndcg": rndcg, "rndcg_lo": rndcg_lo, "rndcg_hi": rndcg_hi,
        })

    if acc_rows:
        acc_df = pandas.DataFrame(acc_rows).sort_values("rndcg", ascending=True)
        panels = [
            ("Kendall τ",       "tau",   "tau_lo",   "tau_hi",   [-1.05, 1.05]),
            (f"NDCG@{_k_cut}",  "ndcgk", "ndcgk_lo", "ndcgk_hi", [-0.05, 1.05]),
            (f"rNDCG@{_k_cut}", "rndcg", "rndcg_lo", "rndcg_hi", [-1.05, 1.05]),
            ("NDCG",            "ndcg",  "ndcg_lo",  "ndcg_hi",  [-0.05, 1.05]),
        ]
        acc_fig = make_subplots(
            rows=1, cols=len(panels),
            subplot_titles=[f"{m} (95% CI)" for m, *_ in panels],
        )
        for col_idx, (title_, data_col, lo_col, hi_col, ref_range) in enumerate(
            panels, start=1,
        ):
            vals   = acc_df[data_col].values
            err_lo = (vals - acc_df[lo_col].values).clip(0)
            err_hi = (acc_df[hi_col].values - vals).clip(0)
            acc_fig.add_trace(go.Bar(
                x=vals, y=acc_df["Metric"], orientation="h",
                marker=outlined_marker(eval.PALETTE_QUAL[col_idx - 1]),
                error_x=bar_error_kwargs(
                    err_hi=err_hi.tolist(), err_lo=err_lo.tolist(),
                ),
                text=[f"{v:.3f} [{lo:.2f},{hi:.2f}]"
                      for v, lo, hi in zip(vals, acc_df[lo_col], acc_df[hi_col])],
                textposition="outside",
                showlegend=False,
            ), row=1, col=col_idx)
            acc_fig.update_xaxes(
                title_text=title_,
                range=ref_range, gridcolor="#e5e5e5",
                zeroline=True, zerolinecolor="black",
                row=1, col=col_idx,
            )
        acc_fig.update_layout(
            # 4 horizontal-bar panels; per-panel sized to match the E16
            # bar figure (~460 px each in the value direction).
            height=120 + 60 * len(acc_df), width=460 * len(panels) + 100,
            template=eval.CMRES_TEMPLATE,
            margin={"l": 200, "b": 50, "r": 80, "t": 60},
        )
        figures.append(acc_fig)
        titles.append(
            f"CP-only [{label}] Kendall τ + NDCG@{_k_cut} + rNDCG@{_k_cut} "
            f"+ NDCG (n={len(primary)})"
        )

    # ── 5. Precision@k (combined) ─────────────────────────────────────────
    n_comp = len(primary)
    if n_comp >= 2:
        k_values = list(range(1, n_comp + 1))
        prec_fig = go.Figure()
        for col, lab in pred_metrics:
            scores = primary[col].values
            prec_fig.add_trace(go.Scatter(
                x=k_values,
                y=[_precision_at_k(actual_vals, scores, k) for k in k_values],
                mode="lines+markers", name=lab, marker=dict(size=5),
            ))
        prec_fig.add_trace(go.Scatter(
            x=k_values, y=[k / n_comp for k in k_values],
            mode="lines", name="Random baseline",
            line=dict(dash="dash", color="grey", width=1),
        ))
        prec_fig.update_layout(
            height=450, width=850, template=eval.CMRES_TEMPLATE,
            xaxis=dict(title="k (number of top components considered)",
                       dtick=max(1, n_comp // 20)),
            yaxis=dict(title="Precision@k", range=[-0.05, 1.05]),
            legend=dict(title="Metric", x=1.01, xanchor="left"),
            margin={"l": 60, "b": 60, "r": 20, "t": 40},
        )
        figures.append(prec_fig)
        titles.append(f"CP-only [{label}] Precision@k (combined, n={n_comp})")

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    eval.write_all_in_one(
        figures, "Figure", Path("."),
        output_path,
        titles=titles,
    )
    print(
        f"[cp_only:{label}] written {output_path} "
        f"(n_mc={n_mc}, n_all={n_all}, types={cp_types_present})"
    )


def cp_only_metric_comparison(df_all: pandas.DataFrame, network_type: str):
    """Per-network CP-only view of the metric battery, broken out by cp_type.

    Mirrors ``cp_metric_vs_actual_impact`` but with only coupling-point rows
    so non-CP branches don't dominate the ranking metrics. Adds a per-CP-type
    Spearman-ρ heatmap so you can see at a glance which CP types each predictor
    handles well and which it doesn't.
    """
    out = OUTPUT + f"/{network_type}/cp_metric_vs_actual_cp_only.html"
    _cp_only_metric_comparison_core(df_all, network_type, out)


def cp_only_pooled_metric_comparison(pooled_df: pandas.DataFrame, output_dir: str):
    """Pooled CP-only view across all network types, broken out by cp_type."""
    out = output_dir + "/cp_metric_vs_actual_cp_only_pooled.html"
    _cp_only_metric_comparison_core(pooled_df, "pooled", out)


def pooled_resilience_per_scenario(perf_df: pandas.DataFrame, output_dir: str):
    """Pooled performance loss per carrier across all scenarios / network types.

    One figure showing the mean per-carrier performance loss for every scenario
    (= folder = grid) found in *perf_df*, grouped on the x-axis by network type.
    Mirrors the per-network ``resilience_per_scenario`` aggregation but
    consolidates everything into a single output.
    """
    if perf_df.empty:
        print("Pooled resilience plot: empty perf_df — skipping.")
        return

    pooled = (
        perf_df.groupby(["network_type", "experiment", "id"])[["0", "1", "2"]]
        .mean()
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
    pooled["experiment"] = pooled["experiment"].apply(
        lambda v: v.split("/")[-1].split("-", 1)[1]
    )
    pooled["carrier"] = pooled["carrier"].map(CARRIER_REPLACE_MAP)
    pooled["scenario"] = pooled["network_type"].map(pretty_scenario)
    # Order rows (and therefore the x-axis categories below) by the
    # canonical ALL_GRIDS scheme so figures are consistent across runs.
    pooled = (
        pooled.assign(
            _sort_key=pooled["network_type"].map(scenario_sort_key)
        )
        .sort_values(["_sort_key", "experiment", "carrier"])
        .drop(columns="_sort_key")
        .reset_index(drop=True)
    )

    fig = eval.create_bar(
        pooled,
        x_label="scenario",
        y_label="resilience_mean",
        color="carrier",
        color_discrete_map=eval.NETWORK_COLOR_MAP,
        pattern_shape_map=eval.NETWORK_PATTERN_MAP,
        legend_text="carrier",
        template=eval.CMRES_TEMPLATE,
        yaxis_title="mean performance loss in MW",
        xaxis_title="scenario",
        title="Pooled performance drop by scenario, by carrier",
        barmode="group",
        width=max(800, 80 * len(pooled["scenario"].unique())),
        height=480,
    )

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    eval.write_all_in_one(
        [fig],
        "Figure",
        Path("."),
        output_dir + "/resilience_per_carrier_per_scenario_pooled.html",
        titles=[
            f"Pooled performance drop by scenario "
            f"(n_scenarios={pooled['scenario'].nunique()}, "
            f"n_network_types={pooled['network_type'].nunique()})"
        ],
    )
    print(
        f"Pooled resilience plot: "
        f"{pooled['scenario'].nunique()} scenarios across "
        f"{pooled['network_type'].nunique()} network types → {output_dir}"
    )


def cross_carrier_impact_per_scenario(
    impact_df: pandas.DataFrame, output_dir: str
):
    """Cross-sector impact figure: one panel per scenario, no averaging.

    For each scenario (= grid in ``test_grids.ALL_GRIDS``), shows the total
    |impact| (MW) attributable to failures originating in each source
    carrier (electricity / heat / gas / multi-carrier CPs), broken down
    by the impacted target carrier (electricity / heat / gas).

    Reading the figure:
      * x-axis of each subplot = **source** carrier (where the failed
        component lives).
      * grouped bars (color) = **target/impacted** carrier (where the
        load-shed shows up).
      * each subplot keeps the scenario's raw cross-sector pattern intact
        — nothing is pooled across scenarios.

    Source classification uses ``TYPE_TO_CARRIER``. Components mapped to
    ``"multi"`` (CHP / P2G / G2P / P2H / GasToHeatHG / their control
    nodes) get their own ``multi`` source bucket so cross-carrier
    coupling-point impacts don't disappear into the three plain sectors.
    Components mapped to ``"heat/gas"`` (generic junctions) are dropped
    because the static type→carrier map cannot disambiguate them without
    the grid object.
    """
    if impact_df is None or impact_df.empty:
        print("Cross-carrier impact: empty impact_df — skipping.")
        return

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = impact_df.copy()
    # ``impact_df.type`` is a Python ``type`` object on fresh builds and a
    # ``"<class '…PowerLine'>"`` string on CSV round-trips — neither equals
    # the bare ``"PowerLine"`` key in ``TYPE_TO_CARRIER``. Normalise both
    # forms to the bare class name before the lookup, otherwise every row
    # drops out and ``nets`` ends up empty even though impact_df is populated.
    df["_type_name"] = df["type"].map(_type_name)
    df["source_carrier"] = df["_type_name"].map(TYPE_TO_CARRIER)
    df = df[df["source_carrier"].notna()]
    # Drop the ambiguous "heat/gas" source bucket (generic Junction); the
    # static map can't say which grid the junction lives on.
    df = df[df["source_carrier"] != "heat/gas"]
    df["impact_abs"] = df["impact"].abs()

    source_order = ["electricity", "heat", "gas", "multi"]
    target_order = ["electricity", "heat", "gas"]

    # Carrier-key → sector helpers. ``SECTOR_*`` uses ``power`` for the
    # electricity bucket; cross-carrier data still labels that carrier
    # ``electricity``. Translate once so both legends render identically.
    _CARRIER_TO_SECTOR = {
        "electricity": "power", "heat": "heat", "gas": "gas", "multi": "multi",
    }

    def _sector_label(c: str) -> str:
        return SECTOR_PRETTY[_CARRIER_TO_SECTOR[c]]

    def _sector_color(c: str) -> str:
        return SECTOR_COLORS[_CARRIER_TO_SECTOR[c]]

    agg = (
        df.groupby(["network_type", "source_carrier", "carrier"], as_index=False)[
            "impact_abs"
        ]
        .sum()
    )

    nets = sort_scenarios(agg["network_type"].unique())
    if not nets:
        # Diagnostic for the empty-after-filter case so the next person
        # doesn't have to repeat this debugging session.
        n_in = len(impact_df)
        unmapped_types = (
            impact_df["type"].map(_type_name).value_counts().head(5)
            if "type" in impact_df.columns else "<no type column>"
        )
        print(
            f"Cross-carrier impact: 0 scenarios after filtering "
            f"(impact_df had {n_in} rows). Top unmapped types: "
            f"{unmapped_types.to_dict() if hasattr(unmapped_types, 'to_dict') else unmapped_types}"
        )
        return

    n = len(nets)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[pretty_scenario(nt) for nt in nets],
        shared_yaxes=False,
        horizontal_spacing=0.07, vertical_spacing=0.18,
    )

    # Find a global y-range so subplots are comparable at a glance.
    y_max = float(agg["impact_abs"].max() or 0.0) * 1.10
    if y_max <= 0:
        y_max = 1.0

    legend_seen: set = set()
    for idx, nt in enumerate(nets):
        r = idx // n_cols + 1
        c = idx % n_cols + 1
        sub = agg[agg["network_type"] == nt]
        for target in target_order:
            sub_t = sub[sub["carrier"] == target]
            # Reindex over source_order so empty buckets render as zero
            # and the x-axis is identical across every subplot.
            ys = [
                float(sub_t.loc[sub_t["source_carrier"] == s, "impact_abs"].sum())
                for s in source_order
            ]
            show_legend = target not in legend_seen
            legend_seen.add(target)
            target_label = _sector_label(target)
            fig.add_trace(
                go.Bar(
                    x=[_sector_label(s) for s in source_order], y=ys,
                    name=f"→ {target_label}",
                    marker=dict(
                        color=_sector_color(target),
                        line=dict(color="#222", width=0.4),
                    ),
                    legendgroup=target,
                    showlegend=show_legend,
                    hovertemplate=(
                        f"scenario={pretty_scenario(nt)}<br>"
                        "source=%{x}<br>"
                        f"target={target_label}<br>"
                        "Σ |impact|=%{y:.4f} MW<extra></extra>"
                    ),
                ),
                row=r, col=c,
            )
        fig.update_xaxes(title_text="Source carrier", row=r, col=c)
        fig.update_yaxes(
            title_text="Σ |impact| (MW)", range=[0, y_max], row=r, col=c,
        )

    fig.update_layout(
        barmode="group",
        height=300 * n_rows + 80,
        width=420 * n_cols,
        template=eval.CMRES_TEMPLATE,
        legend=dict(title="Impacted carrier"),
        margin={"l": 70, "b": 60, "r": 30, "t": 70},
    )

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    eval.write_all_in_one(
        [fig], "Figure", Path("."),
        output_dir + "/cross_carrier_impact_per_scenario.html",
        titles=[
            f"Cross-carrier impact per scenario "
            f"(n_scenarios={n}, source × target, no averaging)"
        ],
    )
    print(
        f"Cross-carrier impact: {n} scenarios → "
        f"{output_dir}/cross_carrier_impact_per_scenario.html"
    )


def cross_carrier_impact_aggregated(
    impact_df: pandas.DataFrame, output_dir: str
):
    """Cross-sector impact, averaged across all scenarios.

    Companion to :func:`cross_carrier_impact_per_scenario` that collapses
    the per-scenario small-multiples into a single bar chart. For each
    (source carrier, target carrier) cell we first compute the
    per-scenario Σ |impact| (same partitioning as the per-scenario plot),
    then take the **mean across scenarios** as the bar height and the
    **std across scenarios** as the error bar.

    Reading the figure:
      * x-axis = **source** carrier (where the failed component lives).
      * grouped bars (color) = **target/impacted** carrier (load-shed).
      * bar height = mean over scenarios of Σ |impact|.
      * error bar = std over scenarios of Σ |impact|.

    Same source classification rules as the per-scenario variant
    (``TYPE_TO_CARRIER``; ``"heat/gas"`` Junctions dropped;
    coupling-point components in their own ``"multi"`` source bucket).
    """
    if impact_df is None or impact_df.empty:
        print("Cross-carrier impact (aggregated): empty impact_df — skipping.")
        return

    import numpy as _np
    import plotly.graph_objects as go

    df = impact_df.copy()
    # See `_type_name` rationale in cross_carrier_impact_per_scenario:
    # impact_df.type is the Python type object on fresh builds and the
    # ``"<class '…'>"`` string on CSV round-trips; normalise to the bare
    # class name before the TYPE_TO_CARRIER lookup.
    df["_type_name"] = df["type"].map(_type_name)
    df["source_carrier"] = df["_type_name"].map(TYPE_TO_CARRIER)
    df = df[df["source_carrier"].notna()]
    df = df[df["source_carrier"] != "heat/gas"]
    df["impact_abs"] = df["impact"].abs()

    source_order = ["electricity", "heat", "gas", "multi"]
    target_order = ["electricity", "heat", "gas"]

    # Carrier-key → sector helpers — same translation as the per-scenario
    # variant so both figures share legend colour + text.
    _CARRIER_TO_SECTOR = {
        "electricity": "power", "heat": "heat", "gas": "gas", "multi": "multi",
    }

    def _sector_label(c: str) -> str:
        return SECTOR_PRETTY[_CARRIER_TO_SECTOR[c]]

    def _sector_color(c: str) -> str:
        return SECTOR_COLORS[_CARRIER_TO_SECTOR[c]]

    # Per-scenario Σ |impact| for each (network_type, source, target) cell,
    # reindexed against the canonical (source, target) grid so missing
    # combinations contribute 0 (otherwise the mean would silently exclude
    # scenarios where, say, no gas-side component failed).
    per_scenario_sum = (
        df.groupby(["network_type", "source_carrier", "carrier"], as_index=False)[
            "impact_abs"
        ]
        .sum()
    )
    nets = sort_scenarios(per_scenario_sum["network_type"].unique())
    if not nets:
        n_in = len(impact_df)
        unmapped_types = (
            impact_df["type"].map(_type_name).value_counts().head(5)
            if "type" in impact_df.columns else "<no type column>"
        )
        print(
            f"Cross-carrier impact (aggregated): 0 scenarios after filtering "
            f"(impact_df had {n_in} rows). Top unmapped types: "
            f"{unmapped_types.to_dict() if hasattr(unmapped_types, 'to_dict') else unmapped_types}"
        )
        return

    # Per-source component count per scenario — needed for the
    # per-component figure below. Each component has one impact_df row
    # per impacted carrier (3 rows / component), so we de-dup by unique
    # ``id`` rather than counting rows.
    comp_count = (
        df.groupby(["network_type", "source_carrier"])["id"]
        .nunique()
        .rename("n_components")
        .reset_index()
    )
    per_scenario_sum = per_scenario_sum.merge(
        comp_count, on=["network_type", "source_carrier"], how="left"
    )
    per_scenario_sum["impact_per_component"] = (
        per_scenario_sum["impact_abs"] / per_scenario_sum["n_components"].clip(lower=1)
    )

    # Build a (source × target) DataFrame per scenario and stack into a
    # 3-D array (n_scenarios, |source|, |target|) for mean/std. We stack
    # both the totals and the per-component normalised values so the two
    # figures below share the same scenario-axis ordering.
    full_index = pandas.MultiIndex.from_product(
        [source_order, target_order], names=["source_carrier", "carrier"],
    )

    def _stack_value(value_col):
        layers = []
        for nt in nets:
            sub = (
                per_scenario_sum[per_scenario_sum["network_type"] == nt]
                .set_index(["source_carrier", "carrier"])[value_col]
                .reindex(full_index, fill_value=0.0)
                .unstack("carrier")
                .reindex(index=source_order, columns=target_order)
                .fillna(0.0)
            )
            layers.append(sub.to_numpy(dtype=float))
        return _np.stack(layers, axis=0)

    total_stack = _stack_value("impact_abs")
    per_comp_stack = _stack_value("impact_per_component")

    def _mean_std(stack):
        mean = stack.mean(axis=0)
        std = (
            stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else _np.zeros_like(mean)
        )
        return mean, std

    total_mean, total_std = _mean_std(total_stack)
    per_comp_mean, per_comp_std = _mean_std(per_comp_stack)

    # Component counts averaged across scenarios so the per-component
    # figure's hover can disclose what we normalised by.
    n_comp_per_source = (
        comp_count.groupby("source_carrier")["n_components"]
        .mean()
        .reindex(source_order, fill_value=0)
        .to_numpy(dtype=float)
    )

    def _build_fig(mean_arr, std_arr, y_title, unit, show_n_components):
        fig = go.Figure()
        for j, target in enumerate(target_order):
            extra_hover = ""
            if show_n_components:
                extra_hover = "<br>mean n_components=%{customdata:.1f}"
            target_label = _sector_label(target)
            fig.add_trace(go.Bar(
                x=[_sector_label(s) for s in source_order],
                y=mean_arr[:, j],
                name=f"→ {target_label}",
                marker=dict(
                    color=_sector_color(target),
                    line=dict(color="#222", width=0.4),
                ),
                customdata=n_comp_per_source if show_n_components else None,
                error_y=dict(
                    type="data",
                    array=std_arr[:, j].tolist(),
                    thickness=1.2, width=3, color="#333",
                    visible=bool(total_stack.shape[0] > 1),
                ),
                hovertemplate=(
                    "source=%{x}<br>"
                    f"target={target_label}<br>"
                    "%{y:.4f}" + f" {unit}"
                    + extra_hover
                    + f"<br>n_scenarios={total_stack.shape[0]}<extra></extra>"
                ),
            ))
        fig.update_layout(
            barmode="group",
            height=520,
            width=820,
            template=eval.CMRES_TEMPLATE,
            legend=dict(title="Impacted carrier"),
            xaxis=dict(title="Source carrier"),
            yaxis=dict(title=y_title),
            margin={"l": 70, "b": 60, "r": 30, "t": 70},
        )
        return fig

    fig_total = _build_fig(
        total_mean, total_std,
        y_title="Mean Σ |impact| across scenarios (MW)",
        unit="MW",
        show_n_components=False,
    )
    fig_per_comp = _build_fig(
        per_comp_mean, per_comp_std,
        y_title="Mean |impact| per source-side component (MW)",
        unit="MW / component",
        show_n_components=True,
    )

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    eval.write_all_in_one(
        [fig_total, fig_per_comp], "Figure", Path("."),
        output_dir + "/cross_carrier_impact_aggregated.html",
        titles=[
            f"Cross-carrier impact aggregated across all scenarios "
            f"(n_scenarios={total_stack.shape[0]}, mean ± std)",
            f"Cross-carrier impact per source-side component "
            f"(n_scenarios={total_stack.shape[0]}, "
            "Σ |impact| ÷ n_components in that source bucket)",
        ],
    )
    print(
        f"Cross-carrier impact (aggregated): {total_stack.shape[0]} scenarios → "
        f"{output_dir}/cross_carrier_impact_aggregated.html (total + per-component)"
    )


def pooled_metric_comparison(pooled_df, output_dir):
    """Run metric comparison figures on data pooled across all network types.

    pooled_df must have the same columns as the per-network df produced by
    cp_metric_vs_actual_impact, plus a 'network_type' column.
    """
    import numpy as _np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Same canonical 10-metric set as cp_metric_vs_actual_impact and E16;
    # see eval_common.CORE_METRICS for the single source of truth.
    METRICS = list(_ec.CORE_METRICS) + [("actual_total", "Actual (MC)")]

    df_all_full = pooled_df.copy()
    # score_no_topo = pure PTDF stress. See cp_metric_vs_actual_impact for why
    # this is no longer `predicted_score / topo_factor` — that derivation
    # silently included throughput and (for CPs) the input-adequacy gate.
    if "score_no_topo" not in df_all_full.columns:
        df_all_full["score_no_topo"] = df_all_full["predicted_stress"]
    if "score_topo_only" not in df_all_full.columns:
        df_all_full["score_topo_only"] = df_all_full["topo_bc"]

    valid_all = df_all_full.dropna(subset=[col for col, _ in METRICS])
    # Primary view: pooled MC-sampled components only. Same rationale as in
    # cp_metric_vs_actual_impact.
    valid_mc = valid_all[valid_all["actual_total"] > MC_FAILED_EPS].reset_index(drop=True)
    n_all = len(valid_all)
    n_mc = len(valid_mc)
    if n_mc < 3:
        print(
            f"Pooled: only {n_mc} components have actual>0; falling back to "
            f"full set (n={n_all}) for ranking metrics"
        )
        valid = valid_all
    else:
        valid = valid_mc

    n_total = len(valid)
    net_types = sort_scenarios(valid["network_type"].unique())
    print(
        f"Pooled analysis: {n_total} components (MC-sampled, of {n_all} matched) "
        f"across {len(net_types)} network types: {net_types}"
    )

    # ── Helpers ────────────────────────────────────────────────────────────

    # Stat helpers: re-exported from eval_common (same logic as the
    # per-network and CP-only views).
    _spearman_with_ci = _ec.spearman_with_ci
    _ndcg = _ec.ndcg
    _precision_at_k = _ec.precision_at_k
    _bootstrap_ci = _ec.bootstrap_ci

    def _rho_label(rho, pval, ci_lo, ci_hi):
        return f"ρ={rho:.2f} [{ci_lo:.2f},{ci_hi:.2f}], p={pval:.3f}"

    figures = []
    titles  = []
    actual_vals = valid["actual_total"].values
    pred_metrics = [(col, label) for col, label in METRICS if col != "actual_total"]
    net_colors = {nt: eval.PALETTE_QUAL[i % 10]
                  for i, nt in enumerate(net_types)}

    # ── 1. Scatter panels (one per metric, colored by network type) ────────
    # Per-metric x-axis formula annotations — falls back to the metric
    # label when no special formula has been registered.
    _METRIC_FORMULAS = {
        "predicted_score":          "τ · PTDF_stress · (1 + α·BC_phys) · adequacy",
        "predicted_score_cp_aware": "CP-aware variant of predicted_score",
        "predicted_score_balanced": "Balanced S1 + C1 + C2 + C3 composite",
        "predicted_stress":         "PTDF stress (carrier-weighted)",
        "topo_bc":                  "Phys. betweenness centrality",
        "stress_bc":                "Stress-weighted betweenness centrality",
        "katz_score":               "Katz centrality (phys. graph)",
        "vitality_score":           "W(G) − W(G\\v) (phys. graph)",
        "local_score":              "loading · (1 + crit.nbrs) · n_carriers",
        "self_score":               "loading · n_carriers",
    }
    panels = [
        (col, label, _METRIC_FORMULAS.get(col, label))
        for col, label in pred_metrics
    ]
    # Plotly subplot grid auto-sized to the canonical metric count
    # (10 metrics → 2 rows × 5 cols; auto-adjusts if CORE_METRICS shrinks).
    n_panels = len(panels)
    n_cols = max(1, (n_panels + 1) // 2)
    n_rows = (n_panels + n_cols - 1) // n_cols
    scatter_fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[
            f"{label}<br>{_rho_label(*_spearman_with_ci(valid[col], valid['actual_total']))}"
            for col, label, _ in panels
        ],
        shared_yaxes=False,
    )
    for idx, (x_col, _label, x_axis_label) in enumerate(panels):
        r, c = divmod(idx, n_cols)
        for nt in net_types:
            sub = valid[valid["network_type"] == nt]
            scatter_fig.add_trace(go.Scatter(
                x=sub[x_col], y=sub["actual_total"],
                mode="markers", name=pretty_scenario(nt),
                marker=dict(color=net_colors[nt], size=7),
                legendgroup=nt, showlegend=(idx == 0),
            ), row=r + 1, col=c + 1)
        scatter_fig.update_xaxes(title_text=x_axis_label, row=r + 1, col=c + 1)
        scatter_fig.update_yaxes(title_text="Actual Impact (MW)", row=r + 1, col=c + 1)
    scatter_fig.update_layout(
        height=380 * n_rows + 80, width=420 * n_cols,
        template=eval.CMRES_TEMPLATE,
        margin={"l": 60, "b": 60, "r": 20, "t": 80},
        legend={"title": "Network type"},
    )
    figures.append(scatter_fig)
    titles.append(f"Pooled metric scatter (n={n_total} across {len(net_types)} networks)")

    # ── 1b. Pairwise Spearman ρ heatmap across all metrics + actual ──────────
    # Companion to the per-metric scatter panels above: shows pairwise
    # rank correlation between every predicted score and the MC actual
    # impact, so redundant / co-linear scores stand out at a glance.
    # One C call to ``DataFrame.corr(method='spearman')`` replaces the
    # earlier nested Python loop over ``scipy.stats.spearmanr``.
    heat_cols = [col for col, _ in METRICS]
    heat_labels = [label for _, label in METRICS]
    n_h = len(heat_cols)
    corr_df = valid[heat_cols].corr(method="spearman")
    rho_matrix = corr_df.reindex(index=heat_cols, columns=heat_cols).to_numpy(
        dtype=float, na_value=_np.nan
    )
    corr_fig = go.Figure(go.Heatmap(
        z=rho_matrix,
        x=heat_labels,
        y=heat_labels,
        zmin=-1.0, zmax=1.0,
        colorscale="RdBu_r",
        reversescale=False,
        colorbar=dict(title="Spearman ρ", tickvals=[-1, -0.5, 0, 0.5, 1]),
        hovertemplate="x=%{x}<br>y=%{y}<br>ρ=%{z:.2f}<extra></extra>",
    ))
    # Cell annotations (small numeric labels) for quick visual scanning.
    annotations = []
    for i in range(n_h):
        for j in range(n_h):
            v = rho_matrix[i, j]
            text_color = "white" if abs(v) > 0.55 else "black"
            annotations.append(dict(
                x=heat_labels[j], y=heat_labels[i],
                text=("—" if v != v else f"{v:.2f}"),
                xref="x", yref="y", showarrow=False,
                font=dict(size=10, color=text_color),
            ))
    corr_fig.update_layout(
        height=80 + 38 * n_h, width=140 + 38 * n_h + 260,
        template=eval.CMRES_TEMPLATE,
        margin={"l": 200, "b": 180, "r": 60, "t": 50},
        annotations=annotations,
        xaxis=dict(tickangle=-35, automargin=True),
        yaxis=dict(autorange="reversed", automargin=True),
    )
    figures.append(corr_fig)
    titles.append(
        f"Pairwise Spearman ρ across metrics and the MC actual (n={n_total})"
    )

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
        marker=outlined_marker(eval.PALETTE_QUAL[0]),
        error_x=bar_error_kwargs(
            err_hi=rho_df["err_hi"].tolist(),
            err_lo=rho_df["err_lo"].tolist(),
        ),
        text=[f"ρ={r:.2f} [{lo:.2f},{hi:.2f}]<br>p={p:.3f}"
              for r, lo, hi, p in zip(rho_df["Spearman ρ"], rho_df["ci_lo"],
                                      rho_df["ci_hi"], rho_df["p-value"])],
        textposition="outside",
    ))
    rho_bar.update_layout(
        # Same horizontal-bar dim as the per-network rho_fig — 60 px per
        # bar so each bar is visually comparable to an E16 bar.
        height=120 + 60 * max(len(rho_df), 1), width=900,
        template=eval.CMRES_TEMPLATE,
        xaxis=dict(title="Spearman ρ", range=[-1.05, 1.05],
                   gridcolor="#e5e5e5",
                   zeroline=True, zerolinecolor="black"),
        yaxis=dict(title="Metric"),
        margin={"l": 160, "b": 50, "r": 160, "t": 50},
    )
    figures.append(rho_bar)
    titles.append(f"Pooled Spearman ρ with Fisher z 95% CI (n={n_total})")

    # ── 3. Spearman ρ per network type (small multiples) ──────────────────
    if len(net_types) > 1:
        nt_rho_fig = go.Figure()
        for m_idx, (col, label) in enumerate(pred_metrics):
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
                x=[pretty_scenario(nt) for nt in net_types],
                y=nt_rhos,
                marker=outlined_marker(eval.PALETTE_QUAL[m_idx % 10]),
                error_y=bar_error_kwargs(err_hi=nt_errs_hi, err_lo=nt_errs_lo),
            ))
        nt_rho_fig.update_layout(
            barmode="group",
            # Match E16 grouped-bar dim: 460h × (120 + 110·N)w where the
            # x-axis category is the network here.
            height=460, width=120 + 110 * max(len(net_types), 1),
            template=eval.CMRES_TEMPLATE,
            xaxis=dict(title="Network type"),
            yaxis=dict(title="Spearman ρ", range=[-1.05, 1.05],
                       gridcolor="#e5e5e5",
                       zeroline=True, zerolinecolor="black"),
            legend=dict(title="Metric"),
            margin={"l": 60, "b": 60, "r": 20, "t": 50},
        )
        figures.append(nt_rho_fig)
        titles.append("Spearman ρ per network type — check for heterogeneity")

    # ── 4. Kendall τ + NDCG@k + rNDCG@k (+ legacy NDCG) with bootstrap CI ─
    # NDCG / rNDCG CIs use the vectorised ``bootstrap_ndcg_ci``.
    _rndcg = _ec.random_normalized_ndcg
    _bootstrap_ndcg_ci = _ec.bootstrap_ndcg_ci
    _k_cut = _ec.default_ndcg_k(len(actual_vals))
    _rng = _np.random.default_rng(42)
    acc_rows = []
    for col, label in pred_metrics:
        scores = valid[col].values
        tau  = float(scipy.stats.kendalltau(scores, actual_vals).statistic)
        ndcg = _ndcg(actual_vals, scores)
        ndcgk = _ndcg(actual_vals, scores, k=_k_cut)
        rndcg = _rndcg(actual_vals, scores, k=_k_cut)
        tau_lo, tau_hi = _bootstrap_ci(
            lambda a, p: float(scipy.stats.kendalltau(p, a).statistic),
            actual_vals, scores, rng=_rng)
        ndcg_lo, ndcg_hi = _bootstrap_ndcg_ci(actual_vals, scores, rng=_rng)
        ndcgk_lo, ndcgk_hi = _bootstrap_ndcg_ci(
            actual_vals, scores, k=_k_cut, rng=_rng,
        )
        rndcg_lo, rndcg_hi = _bootstrap_ndcg_ci(
            actual_vals, scores, k=_k_cut, normalize_random=True, rng=_rng,
        )
        acc_rows.append({
            "Metric": label,
            "Kendall τ": tau, "τ_lo": tau_lo, "τ_hi": tau_hi,
            "NDCG": ndcg, "ndcg_lo": ndcg_lo, "ndcg_hi": ndcg_hi,
            "NDCG@k": ndcgk, "ndcgk_lo": ndcgk_lo, "ndcgk_hi": ndcgk_hi,
            "rNDCG@k": rndcg, "rndcg_lo": rndcg_lo, "rndcg_hi": rndcg_hi,
        })
    acc_df = pandas.DataFrame(acc_rows).sort_values("rNDCG@k", ascending=True)
    metric_colors = {row["Metric"]: eval.PALETTE_QUAL[i % 10]
                     for i, row in acc_df.iterrows()}

    panels = [
        ("Kendall τ",       "Kendall τ", "τ_lo",     "τ_hi",     [-1.05, 1.05]),
        (f"NDCG@{_k_cut}",  "NDCG@k",    "ndcgk_lo", "ndcgk_hi", [-0.05, 1.05]),
        (f"rNDCG@{_k_cut}", "rNDCG@k",   "rndcg_lo", "rndcg_hi", [-1.05, 1.05]),
        ("NDCG",            "NDCG",      "ndcg_lo",  "ndcg_hi",  [-0.05, 1.05]),
    ]
    acc_fig = make_subplots(
        rows=1, cols=len(panels),
        subplot_titles=[f"{m} (95% CI)" for m, *_ in panels],
    )
    for col_idx, (title_, data_col, lo_col, hi_col, ref_range) in enumerate(
        panels, start=1,
    ):
        vals   = acc_df[data_col].values
        err_lo = (vals - acc_df[lo_col].values).clip(0)
        err_hi = (acc_df[hi_col].values - vals).clip(0)
        acc_fig.add_trace(go.Bar(
            x=vals, y=acc_df["Metric"], orientation="h",
            marker=dict(
                color=[metric_colors[m] for m in acc_df["Metric"]],
                line=dict(color="#222", width=0.4),
            ),
            error_x=bar_error_kwargs(
                err_hi=err_hi.tolist(), err_lo=err_lo.tolist(),
            ),
            text=[f"{v:.3f} [{lo:.2f},{hi:.2f}]"
                  for v, lo, hi in zip(vals, acc_df[lo_col], acc_df[hi_col])],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)
        acc_fig.update_xaxes(title_text=title_, range=ref_range,
                             gridcolor="#e5e5e5",
                             zeroline=True, zerolinecolor="black", row=1, col=col_idx)
        acc_fig.update_yaxes(title_text="Metric" if col_idx == 1 else "",
                             row=1, col=col_idx)
    acc_fig.update_layout(
        # 4 horizontal-bar panels; per-panel width matches the E16 single
        # bar plot (~460 px), per-bar height bumped to 60 px.
        height=120 + 60 * len(acc_df), width=460 * len(panels) + 100,
        template=eval.CMRES_TEMPLATE,
        margin={"l": 160, "b": 50, "r": 80, "t": 60},
    )
    figures.append(acc_fig)
    best = acc_df.iloc[-1]["Metric"]
    titles.append(
        f"Pooled Ranking Accuracy: Kendall τ + NDCG@{_k_cut} + rNDCG@{_k_cut} "
        f"+ NDCG with Bootstrap 95% CI (n={n_total}, best rNDCG@{_k_cut}: {best})"
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
        height=450, width=750, template=eval.CMRES_TEMPLATE,
        xaxis=dict(title="k (number of top components considered)", dtick=max(1, n_comp // 20)),
        yaxis=dict(title="Precision@k", range=[-0.05, 1.05]),
        legend=dict(title="Metric", x=1.01, xanchor="left"),
        margin={"l": 60, "b": 60, "r": 20, "t": 40},
    )
    figures.append(prec_fig)
    titles.append(f"Pooled Precision@k (n={n_total} MC-sampled of {n_all} matched)")

    # ── Filtered-vs-unfiltered ρ comparison (pooled) ───────────────────────
    if n_mc >= 3 and n_all > n_mc:
        cmp_rows = []
        for col, label in pred_metrics:
            try:
                rho_all, _, lo_all, hi_all = _spearman_with_ci(
                    valid_all[col], valid_all["actual_total"]
                )
            except Exception:
                rho_all, lo_all, hi_all = float("nan"), float("nan"), float("nan")
            try:
                rho_mc, _, lo_mc, hi_mc = _spearman_with_ci(
                    valid_mc[col], valid_mc["actual_total"]
                )
            except Exception:
                rho_mc, lo_mc, hi_mc = float("nan"), float("nan"), float("nan")
            cmp_rows.append({
                "Metric": label,
                "rho_all": rho_all, "ci_lo_all": lo_all, "ci_hi_all": hi_all,
                "rho_mc": rho_mc, "ci_lo_mc": lo_mc, "ci_hi_mc": hi_mc,
            })
        cmp_df = pandas.DataFrame(cmp_rows).sort_values("rho_mc", ascending=True)

        cmp_fig = go.Figure()
        cmp_fig.add_trace(go.Bar(
            name=f"All matched (n={n_all})",
            x=cmp_df["Metric"], y=cmp_df["rho_all"],
            error_y=bar_error_kwargs(
                err_hi=(cmp_df["ci_hi_all"] - cmp_df["rho_all"]).tolist(),
                err_lo=(cmp_df["rho_all"] - cmp_df["ci_lo_all"]).tolist(),
            ),
            marker=outlined_marker("lightgrey"),
        ))
        cmp_fig.add_trace(go.Bar(
            name=f"MC-sampled (n={n_mc})",
            x=cmp_df["Metric"], y=cmp_df["rho_mc"],
            error_y=bar_error_kwargs(
                err_hi=(cmp_df["ci_hi_mc"] - cmp_df["rho_mc"]).tolist(),
                err_lo=(cmp_df["rho_mc"] - cmp_df["ci_lo_mc"]).tolist(),
            ),
            marker=outlined_marker(eval.PALETTE_QUAL[0]),
        ))
        cmp_fig.update_layout(
            barmode="group",
            # E16 grouped-bar dim: 460h × (120 + 110·N)w.
            height=460, width=120 + 110 * max(len(cmp_df), 1),
            template=eval.CMRES_TEMPLATE,
            yaxis=dict(title="Spearman ρ vs Actual (95% CI)", range=[-1.15, 1.15],
                       gridcolor="#e5e5e5",
                       zeroline=True, zerolinecolor="black"),
            xaxis=dict(title="Metric"),
            margin={"l": 60, "b": 100, "r": 20, "t": 50},
            legend=dict(title="Population"),
        )
        figures.append(cmp_fig)
        titles.append(
            f"Filtered vs unfiltered (pooled): ρ on all matched (n={n_all}) vs "
            f"only MC-sampled (n={n_mc}, actual > {MC_FAILED_EPS:g})"
        )

    Path(output_dir).mkdir(exist_ok=True, parents=True)
    eval.write_all_in_one(
        figures, "Figure", Path("."),
        output_dir + "/cp_metric_vs_actual_pooled.html",
        titles=titles,
    )
    print(
        f"Written pooled metric comparison (MC-sampled n={n_total} of {n_all} matched) "
        f"to {output_dir}/cp_metric_vs_actual_pooled.html"
    )


def _scenario_density_distribution(network_type: str):
    """Map a simbench scenario name to (CP density, distribution label).

    Returns ``(None, None)`` for scenarios that don't follow the
    ``simbench_lv[...]`` naming convention so the CMRES E3/E4
    experiments can simply skip them.
    """
    name = str(network_type)
    distribution = "centralized" if "centralized" in name else "distributed"
    if name.endswith("_no") or name == "simbench_lv_no":
        return 0.0, distribution
    if name.endswith("_low_high"):
        return 0.875, distribution  # informal mid between low and high
    if name.endswith("_low"):
        return 0.25, distribution
    if name.endswith("_high"):
        return 0.75, distribution
    if name.endswith("_max"):
        return 1.0, distribution
    if name.endswith("_centralized"):
        return 0.5, "centralized"
    if name == "simbench_lv":
        return 0.5, distribution
    return None, distribution


def _make_cmres_artefact(
    label, df_eval, monee_net, mc_npz_path, density, distribution
):
    """Lazy-import wrapper so cp_cn_evaluation.py doesn't hard-depend on
    cmres_eval at module load (the CMRES evaluation experiments are
    optional)."""
    from cmres_eval import ScenarioArtefacts
    return ScenarioArtefacts(
        label=label,
        df_eval=df_eval,
        monee_net=monee_net,
        mc_npz_path=mc_npz_path,
        density=density,
        distribution=distribution,
    )


def evaluate(folder_id):
    fail_df, perf_df, metrics_df, net_type_to_net = load_dfs(folder_id)
    impact_df = create_or_load_impact_df(
        fail_df, perf_df, metrics_df, folder_id
    )
    impact_df = extend_impact_df(net_type_to_net, metrics_df, impact_df)

    per_network_dfs = []
    # Bundle of per-scenario inputs the CMRES eval block consumes after
    # the per-scenario loop completes. Local to evaluate() so re-runs don't
    # carry state.
    cmres_artefacts = []


    pooled_resilience_per_scenario(perf_df, OUTPUT + "/pooled")
    cross_carrier_impact_per_scenario(impact_df, OUTPUT + "/pooled")
    cross_carrier_impact_aggregated(impact_df, OUTPUT + "/pooled")

    if len(per_network_dfs) > 1:
        pooled_df = pandas.concat(per_network_dfs, ignore_index=True)
        pooled_metric_comparison(pooled_df, OUTPUT + "/pooled")
        cp_only_pooled_metric_comparison(pooled_df, OUTPUT + "/pooled")
    elif len(per_network_dfs) == 1:
        print("Only one network type found — skipping pooled metric analysis.")
        cp_only_pooled_metric_comparison(per_network_dfs[0], OUTPUT + "/pooled")
        
    for network_type, monee_net in net_type_to_net.items():
        print(network_type)
        # Plain run_energy_flow is a hard feasibility solve and goes infeasible
        # whenever the demanded load exceeds what the (sparse) coupling points
        # plus the heat slack can deliver — e.g. simbench_lv_low: ~0.534 MW
        # heat demand vs ~0.012 MW heat injection from 6 CHPHGs + 1 P2HHG. Try
        # the hard solve first (fast path); if it reports infeasible, fall back
        # to the same load-shedding optimisation that produced the pickle in
        # the first place, so the metric layer sees a well-defined operating
        # point instead of NaN-laden Pyomo Vars.
        try:
            result = run_energy_flow(
                monee_net, solver=PyomoSolver(), solver_name="gurobi"
            )
            monee_net = result.network
            print(f"  energy flow: feasible (objective={getattr(result, 'objective', '?')})")
        except Exception as e_hard:
            print(f"  plain energy flow failed ({type(e_hard).__name__}: {e_hard}) "
                  f"— falling back to min-load-shedding optimisation")
            opt = mp.create_min_load_shedding_problem(
                bounds_el=(0.9, 1.1),
                bounds_gas=(0.9, 1.1),
                bounds_heat=(0.7, 1.3),
                ext_grid_el_bounds=(-0.25, 0.25),
                ext_grid_gas_bounds=(-1.5, 1.5),
                ext_grid_heat_bounds=(-100, 100),
                include_ext_grids=True,
                check_vm=True,
                check_pressure=True,
                check_temperature=True,
                check_line_loading=True,
                priority_safety_factor=1000.0,
            )
            try:
                result = run_energy_flow_optimization(
                    monee_net,
                    solver=PyomoSolver(),
                    solver_name="gurobi",
                    optimization_problem=opt,
                    exclude_unconnected_nodes=True,
                )
                monee_net = result.network
                print(f"  load-shedding solve: objective={getattr(result, 'objective', '?')} "
                      f"(non-zero ⇒ load was shed)")
            except Exception as e_opt:
                print(f"  load-shedding solve also failed ({type(e_opt).__name__}: {e_opt}) "
                      f"— continuing with pickled network state as-is")
        print("Finished")
        
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
            cp_only_metric_comparison(net_df, network_type)
            per_network_dfs.append(net_df)
            # Per-scenario artefact for the CMRES eval block. Density and
            # distribution map directly off the simbench scenario name; the
            # mc_result.npz path mirrors the layout used by run_simulation.
            density, distribution = _scenario_density_distribution(network_type)
            mc_npz = Path(folder_id) / f"MoneeResilienceExperiment-{network_type}" / "mc_result.npz"
            cmres_artefacts.append(
                _make_cmres_artefact(
                    label=network_type,
                    df_eval=net_df,
                    monee_net=monee_net,
                    mc_npz_path=mc_npz if mc_npz.exists() else None,
                    density=density,
                    distribution=distribution,
                )
            )

    # ── CMRES evaluation experiments (E2..E16) ─────────────────────────
    # Run the full CMRES evaluation battery on the per-scenario artefacts
    # we collected during the loop. Each experiment writes its own CSV (and
    # HTML where applicable) under data/out/cmres/.
    if cmres_artefacts:
        try:
            from cmres_eval import run_cmres_block
            run_cmres_block(
                cmres_artefacts,
                impact_df,
                Path(OUTPUT) / "cmres",
            )
        except Exception as e:
            traceback.print_exc()
            print(f"[cmres-eval] block failed: {type(e).__name__}: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the full CMRES evaluation pipeline (per-scenario eval, pooled "
            "views, CMRES experiment battery E2..E16) on a directory of "
            "simulation outputs."
        ),
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=INPUT,
        help=(
            "Path to the directory containing "
            "MoneeResilienceExperiment-<grid>/ subfolders produced by "
            "experiments/re/run_simulation.py. Defaults to the module-level "
            f"INPUT constant ({INPUT!r}). Accepts both positional and "
            "--input-dir forms."
        ),
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir_flag",
        default=None,
        help="Alias for the positional argument; --input-dir wins when both are given.",
    )
    args = parser.parse_args(argv)
    folder_id = args.input_dir_flag or args.input_dir
    print(f"[cp_cn_evaluation] input_dir = {folder_id}")
    evaluate(folder_id)


if __name__ == "__main__":
    main()
