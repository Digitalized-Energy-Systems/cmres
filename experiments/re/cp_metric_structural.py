"""Structural-metric module: load-aware criticality features.

Adds the per-component features identified by the post-evaluation diagnostic
as missing from the existing metric battery:

  - **DDaR** (Downstream Demand at Risk): per-carrier per-component sum of
    load demand whose nominal supply path passes through this component.
    Strict generalisation of betweenness centrality with load weighting.
  - **CP substitutability**: rated capacity / total rated capacity of all
    CPs of the same kind in the same connected sub-network. High = the CP
    is irreplaceable.
  - **Source-sink BC**: betweenness restricted to source→load pairs. Removes
    the "load↔load" and "source↔source" path counts that contribute noise
    to plain BC on grids where most nodes are loads.
  - **k-shortest-path redundancy**: per-component fraction of the top-k
    shortest source→load paths it lies on. High = bottleneck (no alternates).
  - **Min-cut criticality**: per-component reduction in source→sink max-flow
    when the component is removed. Most expensive of the suite; off by
    default and gated by ``enable_min_cut=True``.
  - **Carrier-load assignment**: per-component breakdown of how many loads
    of each carrier are downstream + per-carrier total demand. DDaR's raw
    output by carrier.

The functions live here, side-effect free; ``attach_structural_metrics``
joins them onto a matched-df produced by ``eval_common.build_matched_df``
so the existing metric battery picks them up automatically.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import monee.model as mm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — carrier-specific demand / source / load enumeration
# ─────────────────────────────────────────────────────────────────────────────


def _gas_demand_mw(model, grid) -> float:
    """Convert a gas Sink's mass_flow to MW using the same convention as
    the resilience metric (kg/s × HHV [kWh/kg] × 3.6).
    """
    hhv = float(getattr(grid, "higher_heating_value", 15.3))
    try:
        m = float(mm.upper(model.mass_flow))
    except Exception:
        m = 0.0
    if not math.isfinite(m):
        return 0.0
    return abs(m) * 3.6 * hhv


def _carrier_loads(monee_net):
    """Return ``{carrier: [(node_id, demand_mw, comp_id, comp_kind), ...]}``.

    ``comp_kind`` is one of ``"child"`` / ``"branch"`` / ``"compound"`` so
    callers can later tag the load itself with its consuming component if
    needed. Demand is in MW per carrier:
      - power: PowerLoad.p_mw
      - heat:  HeatLoad.q_mw_heat (children) and HeatExchangerLoad.q_mw
              (children + branches)
      - gas:   Sink.mass_flow × HHV × 3.6 (gas-grid sinks only)
    """
    out: Dict[str, List[Tuple[int, float, int, str]]] = {
        "power": [], "heat": [], "gas": [],
    }
    passive_hx = getattr(mm, "PassiveHeatExchangerLoad", mm.HeatExchangerLoad)

    for c in monee_net.childs:
        if not c.active or c.ignored:
            continue
        m = c.model
        if isinstance(m, mm.PowerLoad):
            try:
                p = float(mm.upper(m.p_mw))
            except Exception:
                p = 0.0
            if math.isfinite(p) and p > 0:
                out["power"].append((c.node_id, abs(p), c.id, "child"))
        elif isinstance(m, mm.HeatLoad):
            try:
                q = float(mm.upper(m.q_mw_heat))
            except Exception:
                q = 0.0
            if math.isfinite(q) and q > 0:
                out["heat"].append((c.node_id, abs(q), c.id, "child"))
        elif isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
            try:
                q = float(mm.upper(m.q_mw))
            except Exception:
                q = 0.0
            if math.isfinite(q) and q > 0:
                out["heat"].append((c.node_id, abs(q), c.id, "child"))
        elif isinstance(m, mm.Sink):
            grid = getattr(c, "grid", None)
            if grid is not None and getattr(grid, "name", None) == "gas":
                d = _gas_demand_mw(m, grid)
                if d > 0:
                    out["gas"].append((c.node_id, d, c.id, "child"))

    for b in monee_net.branches:
        if not b.active or b.ignored:
            continue
        m = b.model
        if isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
            try:
                q = float(mm.upper(m.q_mw))
            except Exception:
                q = 0.0
            if math.isfinite(q) and q > 0:
                out["heat"].append((b.from_node_id, abs(q), b.id, "branch"))

    return out


def _carrier_sources(monee_net) -> Dict[str, List[int]]:
    """Per-carrier list of node ids that act as supply slacks.

    For nominal-flow source→load path computations:
      - power: ExtPowerGrid + PowerGenerator + power-output side of
              GasToPower / CHP / CHPHG
      - heat:  ExtHydrGrid (water grid) + heat-output side of
              CHP / CHPHG / PowerToHeat / PowerToHeatHG / GasToHeatHG
      - gas:   ExtHydrGrid (gas grid) + Source + gas-output of PowerToGas

    The carrier label of a CP is its OUTPUT carrier (the CP injects into
    that carrier under nominal flow).
    """
    out: Dict[str, List[int]] = {"power": [], "heat": [], "gas": []}

    for c in monee_net.childs_by_type(mm.ExtPowerGrid):
        out["power"].append(c.node_id)
    for c in monee_net.childs_by_type(mm.PowerGenerator):
        out["power"].append(c.node_id)
    for c in monee_net.childs_by_type(mm.ExtHydrGrid):
        grid = getattr(c, "grid", None)
        gname = getattr(grid, "name", None)
        if gname == "gas":
            out["gas"].append(c.node_id)
        elif gname == "water":
            out["heat"].append(c.node_id)
    for c in monee_net.childs_by_type(mm.Source):
        out["gas"].append(c.node_id)

    # Branch CPs — the carrier of the *output* endpoint is the source side.
    branch_cp_outputs = [
        (mm.PowerToGas, "gas"),
        (mm.GasToPower, "power"),
        (mm.PowerToHeatHG, "heat"),
        (mm.GasToHeatHG, "heat"),
    ]
    for cls, out_carrier in branch_cp_outputs:
        for b in monee_net.branches_by_type(cls):
            if not b.active:
                continue
            # Identify the output endpoint by its grid type.
            for nid in (b.from_node_id, b.to_node_id):
                try:
                    n = monee_net.node_by_id(nid)
                    grid_name = getattr(getattr(n, "grid", None), "name", None)
                    if (out_carrier == "gas" and grid_name == "gas") or \
                       (out_carrier == "power" and grid_name == "power") or \
                       (out_carrier == "heat" and grid_name == "water"):
                        out[out_carrier].append(nid)
                        break
                except Exception:
                    continue

    # Compound CPs — heat + power outputs.
    for cp in monee_net.compounds_by_type(mm.CHP):
        for k, nid in cp.connected_to.items():
            if "power" in k:
                out["power"].append(nid)
            elif "heat" in k and "return" not in k:
                out["heat"].append(nid)
    for cp in monee_net.compounds_by_type(mm.CHPHG):
        for k, nid in cp.connected_to.items():
            if "power" in k:
                out["power"].append(nid)
            elif "heat" in k and "return" not in k:
                out["heat"].append(nid)
    for cp in monee_net.compounds_by_type(mm.PowerToHeat):
        for k, nid in cp.connected_to.items():
            if "heat" in k and "return" not in k:
                out["heat"].append(nid)

    # Dedupe while preserving order (mostly cosmetic).
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def _per_carrier_graph(monee_net, carrier: str) -> Tuple[nx.Graph, Dict[int, List[Tuple]]]:
    """Per-carrier physical graph used for shortest-path routing.

    Nodes: monee nodes whose grid matches the carrier
      - power → Bus on power grid
      - heat  → Junction on water grid
      - gas   → Junction on gas grid
    Edges: passive transport branches in that carrier
      - power → GenericPowerBranch
      - heat  → WaterPipe + HeatExchanger
      - gas   → GasPipe

    The second return is a dict ``edge_branch_ids`` mapping ``(u, v) →
    [branch_id, …]`` so DDaR can credit pipe-branch ids on each edge.
    """
    G = nx.Graph()
    grid_name = {"power": "power", "gas": "gas", "heat": "water"}.get(carrier)
    edge_branches: Dict[Tuple[int, int], List] = defaultdict(list)
    if grid_name is None:
        return G, edge_branches

    if carrier == "power":
        node_iter = list(monee_net.nodes_by_type(mm.Bus))
        edge_classes = (mm.GenericPowerBranch,)
    else:
        node_iter = list(monee_net.nodes_by_type(mm.Junction))
        edge_classes = (mm.GasPipe,) if carrier == "gas" else (mm.WaterPipe, mm.HeatExchanger)

    for n in node_iter:
        if n.grid is not None and n.grid.name == grid_name:
            G.add_node(n.id)

    for cls in edge_classes:
        for b in monee_net.branches_by_type(cls):
            if not getattr(b, "active", True):
                continue
            u, v = b.from_node_id, b.to_node_id
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v)
                edge_branches[(u, v)].append(b.id)
                edge_branches[(v, u)].append(b.id)
    return G, edge_branches


# ─────────────────────────────────────────────────────────────────────────────
# DDaR — Downstream Demand at Risk
# ─────────────────────────────────────────────────────────────────────────────


def compute_ddar_per_component(monee_net) -> pd.DataFrame:
    """Per-component, per-carrier DDaR.

    Returns a tidy DataFrame keyed on the *unified* monee component id
    space:
      - ``carrier``: ``"power" | "heat" | "gas"``
      - ``component_id``: int (node id for nodes; branch id tuple for branches;
        compound id for compounds — stringified)
      - ``component_kind``: ``"node" | "branch" | "compound"``
      - ``ddar_mw``: total demand in MW that flows through this component
        on its carrier's nominal source→load shortest paths
      - ``n_loads``: number of distinct loads contributing

    For each carrier:
      1. build the per-carrier graph
      2. for each load, single-source shortest-path to nearest source
      3. accumulate that load's demand into every node and every edge on
         the path. CPs that act as sources for this carrier also get
         credited (one DDaR row per (CP, carrier))
    """
    rows: List[dict] = []
    for carrier in ("power", "heat", "gas"):
        G, edge_branches = _per_carrier_graph(monee_net, carrier)
        sources = _carrier_sources(monee_net).get(carrier, [])
        # Restrict sources to those present in G (the per-carrier graph).
        sources = [s for s in sources if G.has_node(s)]
        loads = [(n, d, cid, kind) for n, d, cid, kind in _carrier_loads(monee_net).get(carrier, []) if G.has_node(n)]
        if not sources or not loads or G.number_of_nodes() == 0:
            continue

        node_ddar: Dict[int, float] = defaultdict(float)
        edge_ddar: Dict[Tuple[int, int], float] = defaultdict(float)
        node_n_loads: Dict[int, int] = defaultdict(int)
        edge_n_loads: Dict[Tuple[int, int], int] = defaultdict(int)
        # Multi-source shortest path to nearest source.
        try:
            mlen, mpath = nx.multi_source_dijkstra(G, sources=sources)
        except Exception:
            continue

        for load_node, demand, _cid, _kind in loads:
            if load_node not in mpath:
                continue
            path = mpath[load_node]
            # Accumulate into every node on the path.
            for nid in path:
                node_ddar[nid] += demand
                node_n_loads[nid] += 1
            # Accumulate into every edge on the path.
            for a, b in zip(path[:-1], path[1:]):
                key = (a, b) if a < b else (b, a)
                edge_ddar[key] += demand
                edge_n_loads[key] += 1

        # Emit node rows.
        for nid, mw in node_ddar.items():
            rows.append({
                "carrier": carrier,
                "component_id": nid,
                "component_kind": "node",
                "ddar_mw": float(mw),
                "n_loads": int(node_n_loads[nid]),
            })
        # Emit branch rows: each edge may correspond to multiple parallel
        # branches; credit each branch with the same DDaR.
        for (a, b), mw in edge_ddar.items():
            for branch_id in edge_branches.get((a, b), []):
                rows.append({
                    "carrier": carrier,
                    "component_id": str(branch_id),
                    "component_kind": "branch",
                    "ddar_mw": float(mw),
                    "n_loads": int(edge_n_loads[(a, b)]),
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CP substitutability
# ─────────────────────────────────────────────────────────────────────────────


def _cp_rated_capacity(cp, label: str) -> float:
    """Best-effort rated capacity in MW for a CP, re-using the same
    convention as ``cp_metric._cp_throughput_proxy``.
    """
    try:
        if label == "CHP":
            ctrl = cp.model._control_node
            el = abs(float(getattr(ctrl, "el_mw", 0.0) or 0.0))
            heat_w = float(getattr(ctrl, "heat_w", 0.0) or 0.0)
            heat = abs(heat_w) / 1e6
            return el + heat
        if label == "CHPHG":
            ctrl = cp.model._control_node
            el = abs(float(getattr(ctrl, "el_mw", 0.0) or 0.0))
            heat = abs(float(getattr(ctrl, "heat_mw", 0.0) or 0.0))
            return el + heat
        if label in ("PowerToHeat", "PowerToHeatHG", "GasToHeatHG"):
            return abs(float(getattr(cp.model, "heat_energy_mw", 0.0) or 0.0))
        if label == "GasToPower":
            return abs(float(getattr(cp.model, "el_mw", 0.0) or 0.0))
        if label == "PowerToGas":
            m_dot = abs(float(getattr(cp.model, "gas_kgps", 0.0) or 0.0))
            return m_dot * 15.3 * 3.6  # kg/s × HHV × 3.6 → MW
    except Exception:
        return 1.0
    return 1.0


def compute_cp_substitutability(monee_net) -> pd.DataFrame:
    """Per-CP substitutability score.

    For each CP type, the per-CP value is:
      ``substitutability(c) = rated_capacity(c) / Σ rated_capacity over
      all CPs of the same type that share a connected sub-network``

    Range [0, 1]. 1 = sole CP of its type in the whole grid (irreplaceable);
    near-1/N = many alternatives (one of N equal-capacity CPs).

    Returns a DataFrame ``cp_id`` (str) ``cp_type`` ``rated_capacity_mw``
    ``substitutability``. Joins onto df_eval by cp_id.
    """
    rows: List[dict] = []
    compound_specs = [
        (mm.CHP, "CHP"),
        (mm.CHPHG, "CHPHG"),
        (mm.PowerToHeat, "PowerToHeat"),
    ]
    branch_specs = [
        (mm.PowerToGas, "PowerToGas"),
        (mm.GasToPower, "GasToPower"),
        (mm.PowerToHeatHG, "PowerToHeatHG"),
        (mm.GasToHeatHG, "GasToHeatHG"),
    ]

    cap_by_type: Dict[str, float] = defaultdict(float)
    cps_by_type: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for cls, label in compound_specs:
        for cp in monee_net.compounds_by_type(cls):
            cap = _cp_rated_capacity(cp, label)
            cap_by_type[label] += cap
            cps_by_type[label].append((f"compound:{cp.id}", cap))
    for cls, label in branch_specs:
        for b in monee_net.branches_by_type(cls):
            if not getattr(b, "active", True):
                continue
            cap = _cp_rated_capacity(b, label)
            cap_by_type[label] += cap
            cps_by_type[label].append((f"{b.from_node_id}→{b.to_node_id}", cap))

    for label, cps in cps_by_type.items():
        total = max(cap_by_type[label], 1e-9)
        for cp_id, cap in cps:
            rows.append({
                "cp_id": cp_id,
                "cp_type": label,
                "rated_capacity_mw": float(cap),
                "substitutability": float(cap / total),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Source-sink BC and k-shortest-path redundancy
# ─────────────────────────────────────────────────────────────────────────────


def compute_source_sink_bc(monee_net) -> pd.DataFrame:
    """Per-carrier source→load betweenness centrality.

    Built on the per-carrier physical graph; uses
    ``nx.betweenness_centrality_subset`` so only paths from a *source*
    to a *load* are counted. Returns long-form DataFrame ``carrier``
    ``component_id`` ``component_kind`` ``ss_bc``.
    """
    rows: List[dict] = []
    for carrier in ("power", "heat", "gas"):
        G, edge_branches = _per_carrier_graph(monee_net, carrier)
        sources = [s for s in _carrier_sources(monee_net).get(carrier, []) if G.has_node(s)]
        loads = [n for n, _d, _cid, _kind in _carrier_loads(monee_net).get(carrier, []) if G.has_node(n)]
        if not sources or not loads or G.number_of_edges() == 0:
            continue
        try:
            ss = nx.betweenness_centrality_subset(
                G, sources=sources, targets=loads, normalized=True
            )
        except Exception:
            continue
        ss_edges = nx.edge_betweenness_centrality_subset(
            G, sources=sources, targets=loads, normalized=True
        )
        for nid, v in ss.items():
            rows.append({
                "carrier": carrier,
                "component_id": nid,
                "component_kind": "node",
                "ss_bc": float(v),
            })
        for (a, b), v in ss_edges.items():
            for branch_id in edge_branches.get((a, b), []) + edge_branches.get((b, a), []):
                rows.append({
                    "carrier": carrier,
                    "component_id": str(branch_id),
                    "component_kind": "branch",
                    "ss_bc": float(v),
                })
    return pd.DataFrame(rows)


def compute_kshortest_redundancy(monee_net, k: int = 3) -> pd.DataFrame:
    """Per-component fraction of the top-k shortest source→load paths it
    lies on.

    For each (source, load) pair we enumerate up to ``k`` shortest simple
    paths via ``nx.shortest_simple_paths``. For each component, score is
    the average over (source, load) pairs of ``(# paths it's on) / k``.

    A score near 1 means the component is on essentially every alternate
    path — i.e. a bottleneck. A score near 0 means there's always an
    alternate.

    Heavy on dense pairs; bounded by k * |sources| * |loads|.
    """
    rows: List[dict] = []
    for carrier in ("power", "heat", "gas"):
        G, edge_branches = _per_carrier_graph(monee_net, carrier)
        sources = [s for s in _carrier_sources(monee_net).get(carrier, []) if G.has_node(s)]
        loads = [n for n, _d, _cid, _kind in _carrier_loads(monee_net).get(carrier, []) if G.has_node(n)]
        if not sources or not loads or G.number_of_edges() == 0:
            continue

        node_redund: Dict[int, float] = defaultdict(float)
        edge_redund: Dict[Tuple[int, int], float] = defaultdict(float)
        n_pairs = 0
        for s in sources:
            for t in loads:
                try:
                    paths_iter = nx.shortest_simple_paths(G, s, t)
                except Exception:
                    continue
                seen = []
                for path in paths_iter:
                    seen.append(path)
                    if len(seen) >= k:
                        break
                if not seen:
                    continue
                k_eff = max(1, len(seen))
                # Count which nodes / edges appear in how many of the k paths.
                node_count: Dict[int, int] = defaultdict(int)
                edge_count: Dict[Tuple[int, int], int] = defaultdict(int)
                for p in seen:
                    for nid in p:
                        node_count[nid] += 1
                    for a, b in zip(p[:-1], p[1:]):
                        edge_count[(a, b) if a < b else (b, a)] += 1
                n_pairs += 1
                for nid, c in node_count.items():
                    node_redund[nid] += c / k_eff
                for e, c in edge_count.items():
                    edge_redund[e] += c / k_eff

        if n_pairs == 0:
            continue
        for nid, v in node_redund.items():
            rows.append({
                "carrier": carrier,
                "component_id": nid,
                "component_kind": "node",
                "kshortest_redundancy": float(v / n_pairs),
            })
        for (a, b), v in edge_redund.items():
            for branch_id in edge_branches.get((a, b), []) + edge_branches.get((b, a), []):
                rows.append({
                    "carrier": carrier,
                    "component_id": str(branch_id),
                    "component_kind": "branch",
                    "kshortest_redundancy": float(v / n_pairs),
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Min-cut criticality (expensive — opt-in)
# ─────────────────────────────────────────────────────────────────────────────


def compute_min_cut_criticality(monee_net, super_sink: bool = True) -> pd.DataFrame:
    """Per-component reduction in source→load max-flow when removed.

    For each carrier:
      1. Build the carrier graph; capacity = 1 per edge (unweighted).
      2. Add super-source linked to every source, super-sink linked to
         every load (capacity = ∞).
      3. Compute nominal max-flow F0.
      4. For each node and each edge, remove it, recompute max-flow F_c.
         Score = ``(F0 − F_c) / F0`` ∈ [0, 1].

    Cost: O(N · E) max-flow computations per carrier. For 250 nodes and
    500 edges this is ~125k max-flows per carrier, expensive but still
    feasible (single-thread minutes). Off by default in the eval.
    """
    rows: List[dict] = []
    for carrier in ("power", "heat", "gas"):
        G, edge_branches = _per_carrier_graph(monee_net, carrier)
        sources = [s for s in _carrier_sources(monee_net).get(carrier, []) if G.has_node(s)]
        loads = [n for n, _d, _cid, _kind in _carrier_loads(monee_net).get(carrier, []) if G.has_node(n)]
        if not sources or not loads or G.number_of_edges() == 0:
            continue
        # Convert to digraph with super-source/super-sink for max-flow.
        D = G.to_directed()
        for u, v in D.edges:
            D[u][v]["capacity"] = 1
        SS, TT = "_super_source_", "_super_sink_"
        D.add_node(SS)
        D.add_node(TT)
        for s in sources:
            D.add_edge(SS, s, capacity=10**9)
        for t in loads:
            D.add_edge(t, TT, capacity=10**9)
        try:
            F0 = nx.maximum_flow_value(D, SS, TT)
        except Exception:
            continue
        if F0 <= 0:
            continue

        # Node removal.
        for n in list(G.nodes):
            if n in (SS, TT):
                continue
            D2 = D.copy()
            D2.remove_node(n)
            try:
                Fn = nx.maximum_flow_value(D2, SS, TT)
            except Exception:
                Fn = F0
            rows.append({
                "carrier": carrier,
                "component_id": n,
                "component_kind": "node",
                "min_cut_criticality": float(max(0.0, (F0 - Fn) / F0)),
            })
        # Edge removal.
        for u, v in list(G.edges):
            D2 = D.copy()
            if D2.has_edge(u, v):
                D2.remove_edge(u, v)
            if D2.has_edge(v, u):
                D2.remove_edge(v, u)
            try:
                Fe = nx.maximum_flow_value(D2, SS, TT)
            except Exception:
                Fe = F0
            v_score = float(max(0.0, (F0 - Fe) / F0))
            for branch_id in edge_branches.get((u, v), []) + edge_branches.get((v, u), []):
                rows.append({
                    "carrier": carrier,
                    "component_id": str(branch_id),
                    "component_kind": "branch",
                    "min_cut_criticality": v_score,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Carrier-load assignment (richer breakdown for diagnostics)
# ─────────────────────────────────────────────────────────────────────────────


def compute_carrier_load_assignment(monee_net) -> pd.DataFrame:
    """Per-component breakdown of which loads (across carriers) sit
    downstream under nominal source→load routing.

    Output columns (one row per component):
      - ``component_id``, ``component_kind``
      - ``downstream_loads_power``, ``downstream_loads_heat``,
        ``downstream_loads_gas``
      - ``downstream_demand_power``, ``downstream_demand_heat``,
        ``downstream_demand_gas``  (MW per carrier)

    Computed by union-summing the DDaR per-carrier table.
    """
    ddar = compute_ddar_per_component(monee_net)
    if ddar.empty:
        return pd.DataFrame()
    pivot_loads = (
        ddar.pivot_table(
            index=["component_id", "component_kind"],
            columns="carrier", values="n_loads", aggfunc="sum", fill_value=0,
        )
        .add_prefix("downstream_loads_")
        .reset_index()
    )
    pivot_demand = (
        ddar.pivot_table(
            index=["component_id", "component_kind"],
            columns="carrier", values="ddar_mw", aggfunc="sum", fill_value=0.0,
        )
        .add_prefix("downstream_demand_")
        .reset_index()
    )
    return pivot_loads.merge(pivot_demand, on=["component_id", "component_kind"])


# ─────────────────────────────────────────────────────────────────────────────
# Joining onto the matched eval dataframe
# ─────────────────────────────────────────────────────────────────────────────


def _normalise_component_id(cp_id, cp_type) -> Optional[str]:
    """Map a metric-side cp_id to the component-id key used by structural
    metrics (which are keyed on monee node ids and branch ids)."""
    s = str(cp_id)
    try:
        if cp_type in ("CHP", "CHPHG", "PowerToHeat"):
            # compound — structural metrics treat compound as not-on-graph;
            # return None so the merge leaves NaN, then attach_structural_metrics
            # can fall back to the compound's own connected nodes.
            return None
        if cp_type in ("PowerLine", "GasPipe", "WaterPipe", "HeatExchanger"):
            # cp_id is an edge tuple "(a, b, k)"; structural-metric branch
            # rows store it as the str(edge_id). They match directly.
            return s
        # Branch CP cp_id is "from→to"; structural metrics index branch CPs
        # by their actual branch id (a tuple). We can't recover the tuple
        # from "from→to" alone — but the corresponding monee branch id
        # contains the same node pair. Extract numerics for the merge key.
        return s
    except Exception:
        return None


def attach_structural_metrics(
    df_eval: pd.DataFrame,
    monee_net,
    enable_min_cut: bool = False,
    k_shortest: int = 3,
) -> pd.DataFrame:
    """Augment a matched eval dataframe with the structural metrics.

    Adds columns (per row, NaN where unavailable):
      - ``ddar_mw_total``, ``ddar_mw_power``, ``ddar_mw_heat``, ``ddar_mw_gas``
      - ``ss_bc_total``, ``ss_bc_<carrier>`` (best-of-carrier max-aggregation)
      - ``kshortest_redundancy_total``
      - ``substitutability``  (CPs only; 1.0 elsewhere)
      - ``min_cut_criticality_total`` (only if enable_min_cut=True)

    Branch-keyed metrics are aggregated to the same component-id used in
    df_eval (str(edge_tuple) for non-CP branches, "from→to" for branch CPs).
    Compound CPs are left NaN — DDaR / source-sink BC are graph-node
    quantities and compound CPs sit on multiple nodes.
    """
    df = df_eval.copy()

    # 1) DDaR
    ddar = compute_ddar_per_component(monee_net)
    if not ddar.empty:
        # Group by component_id over carriers.
        ddar_total = (
            ddar.groupby("component_id")["ddar_mw"].sum()
            .rename("ddar_mw_total").reset_index()
        )
        ddar_pivot = (
            ddar.pivot_table(
                index="component_id", columns="carrier",
                values="ddar_mw", aggfunc="sum", fill_value=0.0,
            )
            .add_prefix("ddar_mw_")
            .reset_index()
        )
        df["_match_id"] = df["cp_id"].astype(str)
        df = df.merge(ddar_total, left_on="_match_id", right_on="component_id", how="left").drop(columns=["component_id"])
        df = df.merge(ddar_pivot, left_on="_match_id", right_on="component_id", how="left").drop(columns=["component_id"])

    # 2) Source-sink BC
    ss_bc = compute_source_sink_bc(monee_net)
    if not ss_bc.empty:
        ss_total = (
            ss_bc.groupby("component_id")["ss_bc"].max()
            .rename("ss_bc_total").reset_index()
        )
        df = df.merge(ss_total, left_on="_match_id", right_on="component_id", how="left").drop(columns=["component_id"])

    # 3) k-shortest redundancy
    kshort = compute_kshortest_redundancy(monee_net, k=k_shortest)
    if not kshort.empty:
        kshort_total = (
            kshort.groupby("component_id")["kshortest_redundancy"].max()
            .rename("kshortest_redundancy_total").reset_index()
        )
        df = df.merge(kshort_total, left_on="_match_id", right_on="component_id", how="left").drop(columns=["component_id"])

    # 4) Substitutability (CP only)
    sub = compute_cp_substitutability(monee_net)
    if not sub.empty:
        df = df.merge(
            sub[["cp_id", "substitutability", "rated_capacity_mw"]],
            on="cp_id", how="left",
        )

    # 5) Min-cut (heavy, opt-in)
    if enable_min_cut:
        mc = compute_min_cut_criticality(monee_net)
        if not mc.empty:
            mc_total = (
                mc.groupby("component_id")["min_cut_criticality"].max()
                .rename("min_cut_criticality_total").reset_index()
            )
            df = df.merge(mc_total, left_on="_match_id", right_on="component_id", how="left").drop(columns=["component_id"])

    df = df.drop(columns=["_match_id"], errors="ignore")
    # Default substitutability = 1.0 for non-CP rows so the column doesn't
    # disappear from rank-based metrics through dropna.
    if "substitutability" in df.columns:
        df["substitutability"] = df["substitutability"].fillna(1.0)
    return df
