"""Complex-network helpers for the dissertation's MES contribution.

Each section of this module corresponds to one experiment in the dissertation
plan. The functions here are *pure* — they operate on a ``monee.Network`` (and
optionally a precomputed multilayer ``nx.Graph``) and return numbers,
DataFrames, or new graphs. Visualisation and orchestration live in
``dissertation_eval.py``; per-component scoring stays in ``cp_metric.py``.

Sections
--------
E8  build_multilayer_graph, multilayer_centralities
E9  percolation_curve, attack_auc, percolation_for_metric
E10 coupling_strength, criticality_concentration
E11 null_model_configuration, null_model_er, null_model_z_scores
E12 community_partition, bridge_score
E13 spectral_robustness, kirchhoff_index
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import monee.model as mm


# Constant set of carrier-grid names monee uses internally. Kept in one place
# so each section maps the same way.
_CARRIER_GRID_NAME = {"power": "power", "gas": "gas", "heat": "water"}
_GRID_TO_CARRIER = {v: k for k, v in _CARRIER_GRID_NAME.items()}


# ─────────────────────────────────────────────────────────────────────────────
# E8 — Multilayer network framing
# ─────────────────────────────────────────────────────────────────────────────


def build_multilayer_graph(monee_net) -> nx.Graph:
    """Construct the MES as one multilayer ``nx.Graph``.

    Node ids are tuples ``(carrier, original_id)``. Intra-layer edges
    correspond to power lines / gas pipes / water pipes / heat exchangers;
    inter-layer edges correspond to coupling points (compound CPs and branch
    CPs). Each edge carries:
      - ``layer``: ``"power" | "gas" | "heat" | "coupling"``
      - ``cp_type``: monee class name of the component (``"GasPipe"``,
        ``"CHPHG"``, …) — useful when filtering inter-layer edges by which
        kind of CP they are.

    Used by all of E8/E9/E11/E12/E13. Built once per scenario and cached.
    """
    G = nx.Graph()

    # 1) Add nodes per carrier layer.
    for n in monee_net.nodes_by_type(mm.Bus):
        if n.grid is not None and n.grid.name == "power":
            G.add_node(("power", n.id), layer="power")
    for n in monee_net.nodes_by_type(mm.Junction):
        if n.grid is None:
            continue
        carrier = _GRID_TO_CARRIER.get(n.grid.name)
        if carrier is None:
            continue
        G.add_node((carrier, n.id), layer=carrier)

    # 2) Intra-layer edges (passive transport).
    intra_specs: List[Tuple[type, str]] = [
        (mm.GenericPowerBranch, "power"),
        (mm.GasPipe, "gas"),
        (mm.WaterPipe, "heat"),
        (mm.HeatExchanger, "heat"),
    ]
    for cls, carrier in intra_specs:
        for b in monee_net.branches_by_type(cls):
            if not getattr(b, "active", True):
                continue
            u = (carrier, b.from_node_id)
            v = (carrier, b.to_node_id)
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, layer=carrier, cp_type=cls.__name__)

    # 3) Inter-layer (coupling) edges from branch CPs.
    branch_cp_specs: List[Tuple[type, str, str]] = [
        (mm.PowerToGas, "power", "gas"),
        (mm.GasToPower, "gas", "power"),
        (mm.PowerToHeatHG, "power", "heat"),
        (mm.GasToHeatHG, "gas", "heat"),
    ]
    for cls, c_from, c_to in branch_cp_specs:
        for b in monee_net.branches_by_type(cls):
            if not getattr(b, "active", True):
                continue
            u = (c_from, b.from_node_id)
            v = (c_to, b.to_node_id)
            # Branch CPs in monee don't always have endpoints in the obvious
            # carriers — defensively check both orderings.
            if not (G.has_node(u) and G.has_node(v)):
                u, v = (c_to, b.from_node_id), (c_from, b.to_node_id)
            if G.has_node(u) and G.has_node(v):
                G.add_edge(u, v, layer="coupling", cp_type=cls.__name__)

    # 4) Inter-layer edges from compound CPs (CHP/CHPHG/PowerToHeat).
    for cls in (mm.CHP, mm.CHPHG, mm.PowerToHeat):
        for cp in monee_net.compounds_by_type(cls):
            connected: Dict[str, int] = {}
            for key, nid in cp.connected_to.items():
                if "power" in key:
                    connected["power"] = nid
                elif "gas" in key:
                    connected["gas"] = nid
                elif "heat" in key and "return" not in key:
                    connected["heat"] = nid
            carriers = list(connected.keys())
            # Add an inter-layer edge between every pair of connected carriers
            # so the compound shows up as a coupling between all its layers.
            for i in range(len(carriers)):
                for j in range(i + 1, len(carriers)):
                    u = (carriers[i], connected[carriers[i]])
                    v = (carriers[j], connected[carriers[j]])
                    if G.has_node(u) and G.has_node(v):
                        G.add_edge(u, v, layer="coupling", cp_type=cls.__name__)

    return G


def multilayer_centralities(G: nx.Graph) -> pd.DataFrame:
    """Per-node centralities computed on the whole multilayer graph.

    Returns a DataFrame with columns:
      - ``carrier``, ``orig_id``: components of the multilayer node id
      - ``ml_bc``: multilayer betweenness centrality (single computation on
        the supra-graph; treats every node as a candidate source/sink)
      - ``ml_degree``: degree on the supra-graph
      - ``activity``: number of distinct layers in which this node has at
        least one incident edge (range 1..3)
      - ``participation``: multiplex participation coefficient
        ``1 − Σ_layer (k_layer/k_total)²``; 0 = active in one layer only,
        approaches (n_layers−1)/n_layers when activity is even across layers
      - ``inter_layer_degree``: number of incident "coupling"-layer edges

    The eval can join this on ``(carrier, orig_id)`` to attach multilayer
    metrics to per-component CP scores produced by ``cp_metric.py``.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "carrier", "orig_id", "ml_bc", "ml_degree",
            "activity", "participation", "inter_layer_degree",
        ])

    bc = nx.betweenness_centrality(G, normalized=True)

    rows = []
    for u in G.nodes:
        # Per-layer incident-edge counts. For a coupling edge (u, v),
        # contribute to *both* u's own layer and v's layer — that's what
        # makes activity > 1 for nodes with cross-layer neighbours (e.g.
        # the power-side of a CHPHG compound, which has incident power
        # branches AND coupling edges to gas + heat).
        layer_deg: Dict[str, int] = defaultdict(int)
        inter = 0
        for _, v, attrs in G.edges(u, data=True):
            layer = attrs.get("layer", "?")
            if layer == "coupling":
                inter += 1
                layer_deg[u[0]] += 1
                layer_deg[v[0]] += 1
            else:
                layer_deg[layer] += 1
        total = sum(layer_deg.values())
        if total > 0:
            participation = 1.0 - sum((k / total) ** 2 for k in layer_deg.values())
        else:
            participation = 0.0
        rows.append({
            "carrier": u[0],
            "orig_id": u[1],
            "ml_bc": float(bc.get(u, 0.0)),
            "ml_degree": int(G.degree(u)),
            "activity": int(sum(1 for k in layer_deg.values() if k > 0)),
            "participation": float(participation),
            "inter_layer_degree": int(inter),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# E9 — Percolation / robustness curves
# ─────────────────────────────────────────────────────────────────────────────


def percolation_curve(
    G: nx.Graph,
    removal_order: List,
    target: str = "gcc",
) -> np.ndarray:
    """Robustness curve for sequential node removal in a given order.

    ``removal_order`` is a list of node ids (matching ``G.nodes``) sorted by
    descending criticality. We remove them one at a time and record the
    relative size of the giant connected component (``target="gcc"``) or any
    other monotone connectivity statistic.

    Returns array of shape ``(n_steps + 1,)`` where ``R[k]`` is the fraction
    of original nodes that remain in the largest component after removing
    the first ``k`` items of ``removal_order``. ``R[0] == |GCC₀| / |V|`` and
    ``R[n_nodes-1]`` is bounded below by 0.

    Used by E9: lower AUC = more effective attack ordering = better
    criticality metric. Random-removal baseline is computed separately.
    """
    if target != "gcc":
        raise NotImplementedError("only target='gcc' is implemented for now")
    if G.number_of_nodes() == 0:
        return np.array([1.0])

    H = G.copy()
    n0 = H.number_of_nodes()
    R: List[float] = []
    # Start: pre-removal GCC fraction.
    if n0 > 0:
        comps = nx.connected_components(H)
        R.append(max(len(c) for c in comps) / n0)
    else:
        R.append(0.0)

    for node in removal_order:
        if H.has_node(node):
            H.remove_node(node)
        if H.number_of_nodes() == 0:
            R.append(0.0)
            continue
        comps = nx.connected_components(H)
        R.append(max(len(c) for c in comps) / n0)
    return np.asarray(R, dtype=float)


def attack_auc(R: np.ndarray) -> float:
    """Area under the robustness curve, normalised so AUC ∈ [0, 1].

    Lower = more effective attack ordering.
    Computed via trapezoidal integration over fraction-removed in [0, 1].
    """
    R = np.asarray(R, dtype=float)
    if R.size <= 1:
        return float(R[0]) if R.size == 1 else 0.0
    # x = fraction-removed ∈ [0, 1] uniformly spaced
    x = np.linspace(0.0, 1.0, R.size)
    return float(np.trapz(R, x))


def percolation_for_metric(
    G: nx.Graph,
    metric_per_node: Dict,
    rng: Optional[np.random.Generator] = None,
    n_random: int = 10,
) -> Dict[str, np.ndarray | float]:
    """Compute R(f), AUC, and a random-baseline AUC for one metric ranking.

    ``metric_per_node`` maps multilayer node id → criticality value. Nodes
    not present in the dict are appended at the END of the removal order
    (so the metric only governs the first |dict| removals).

    Returns:
      - ``"R"``: the targeted robustness curve
      - ``"AUC_metric"``: AUC of the targeted ordering
      - ``"R_random"``: per-trial random baseline curves stacked, shape (n_random, n+1)
      - ``"AUC_random_mean"``, ``"AUC_random_std"``
      - ``"AUC_z"``: (AUC_random_mean − AUC_metric) / AUC_random_std (positive
        means metric ordering is more effective than random by that many σ)
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    nodes = list(G.nodes)
    # Targeted order: scored nodes desc, then unscored.
    scored = [n for n in nodes if n in metric_per_node]
    scored.sort(key=lambda n: float(metric_per_node[n]), reverse=True)
    unscored = [n for n in nodes if n not in metric_per_node]
    targeted_order = scored + unscored
    R = percolation_curve(G, targeted_order)
    auc = attack_auc(R)

    R_random = np.zeros((n_random, len(R)))
    auc_random = np.zeros(n_random)
    for k in range(n_random):
        order = nodes[:]
        rng.shuffle(order)
        Rk = percolation_curve(G, order)
        # Pad / truncate just in case (always same length here, defensive).
        m = min(len(Rk), R_random.shape[1])
        R_random[k, :m] = Rk[:m]
        auc_random[k] = attack_auc(Rk)
    auc_rand_mean = float(np.mean(auc_random))
    auc_rand_std = float(np.std(auc_random, ddof=1)) if n_random > 1 else 0.0
    z = ((auc_rand_mean - auc) / auc_rand_std) if auc_rand_std > 0 else float("nan")
    return {
        "R": R,
        "AUC_metric": float(auc),
        "R_random": R_random,
        "AUC_random_mean": auc_rand_mean,
        "AUC_random_std": auc_rand_std,
        "AUC_z": float(z),
    }


# ─────────────────────────────────────────────────────────────────────────────
# E10 — Coupling-strength characterisation
# ─────────────────────────────────────────────────────────────────────────────


def coupling_strength(G: nx.Graph) -> Dict[str, float | dict]:
    """Structural summary of how strongly carriers are coupled in the MES.

    Returns scalars meant to be plotted vs MC-ENS in E10:
      - ``n_inter_edges``: total number of inter-layer (coupling) edges
      - ``n_intra_edges_per_layer``: dict carrier → count
      - ``sigma_c[(a,b)]``: coupling density between layer pair, defined as
        ``n_inter(a,b) / sqrt(n_intra(a) · n_intra(b))`` — non-symmetric
        Salton-style normalisation, comparable across grids
      - ``sigma_c_total``: aggregate over all layer pairs
      - ``cp_localization_gini``: Gini of the per-node count of incident
        coupling edges (high = a few CP-rich nodes carry the coupling;
        low = coupling spread evenly across the grid)
    """
    intra: Dict[str, int] = defaultdict(int)
    inter_pair: Dict[Tuple[str, str], int] = defaultdict(int)
    cp_per_node: Dict = defaultdict(int)
    for u, v, attrs in G.edges(data=True):
        layer = attrs.get("layer", "?")
        if layer == "coupling":
            cp_per_node[u] += 1
            cp_per_node[v] += 1
            a, b = u[0], v[0]
            key = tuple(sorted([a, b]))
            inter_pair[key] += 1
        else:
            intra[layer] += 1

    sigma_c: Dict[Tuple[str, str], float] = {}
    for (a, b), n_ab in inter_pair.items():
        denom = math.sqrt(max(intra.get(a, 0), 1) * max(intra.get(b, 0), 1))
        sigma_c[(a, b)] = n_ab / denom if denom > 0 else 0.0
    sigma_total = sum(sigma_c.values())

    counts = np.array([cp_per_node.get(n, 0) for n in G.nodes], dtype=float)
    return {
        "n_inter_edges": int(sum(inter_pair.values())),
        "n_intra_edges_per_layer": dict(intra),
        "sigma_c": {f"{a}-{b}": float(v) for (a, b), v in sigma_c.items()},
        "sigma_c_total": float(sigma_total),
        "cp_localization_gini": float(_gini(counts)),
    }


def criticality_concentration(values: Iterable[float]) -> Dict[str, float]:
    """Concentration measures of any per-component criticality vector.

    Used in E4/E10 to quantify "how concentrated is criticality?". A
    centralized CP layout should give higher Gini than a distributed one.

    Returns:
      - ``gini``: Gini coefficient (0 = even, 1 = single component holds all)
      - ``top1_share``, ``top5_share``: fraction held by top-1, top-5
      - ``entropy``: Shannon entropy of the normalised distribution (in nats)
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = np.maximum(arr, 0.0)
    if arr.size == 0 or arr.sum() <= 0:
        return {"gini": 0.0, "top1_share": 0.0, "top5_share": 0.0, "entropy": 0.0}
    sorted_desc = np.sort(arr)[::-1]
    total = float(arr.sum())
    p = arr / total
    p_pos = p[p > 0]
    return {
        "gini": float(_gini(arr)),
        "top1_share": float(sorted_desc[0] / total),
        "top5_share": float(sorted_desc[: min(5, sorted_desc.size)].sum() / total),
        "entropy": float(-np.sum(p_pos * np.log(p_pos))),
    }


def _gini(arr: np.ndarray) -> float:
    """Plain Gini coefficient on a non-negative array. 0 if degenerate."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    arr = np.maximum(arr, 0.0)
    s = arr.sum()
    if s <= 0:
        return 0.0
    arr_sorted = np.sort(arr)
    n = arr.size
    cum = np.cumsum(arr_sorted)
    return float((2.0 * np.sum((np.arange(1, n + 1)) * arr_sorted) - (n + 1) * cum[-1]) / (n * cum[-1]))


# ─────────────────────────────────────────────────────────────────────────────
# E11 — Null-model comparison
# ─────────────────────────────────────────────────────────────────────────────


def null_model_configuration(
    G: nx.Graph,
    n_swaps_factor: int = 10,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Degree-preserving rewiring of *G* (one null-model draw).

    Edges are kept the same in number; node degrees are preserved exactly.
    The *layer* attribute on each edge is preserved (a rewired edge keeps
    its original layer label) so per-layer edge counts are conserved too.
    Inter-layer rewires therefore only swap between coupling-edge endpoints,
    not across layer boundaries. This is the standard "edge-coloured
    configuration model" used in multilayer null tests.

    ``n_swaps_factor`` * |E| double-edge swaps are attempted; the actual
    number successfully applied is typically >50% of attempts.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()
    if H.number_of_edges() < 2:
        return H

    edges_by_layer: Dict[str, List[tuple]] = defaultdict(list)
    for u, v, a in H.edges(data=True):
        edges_by_layer[a.get("layer", "?")].append((u, v))

    target_swaps = n_swaps_factor * H.number_of_edges()
    for _ in range(target_swaps):
        layer = rng.choice(list(edges_by_layer.keys()))
        edges = edges_by_layer[layer]
        if len(edges) < 2:
            continue
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            continue
        (u1, v1), (u2, v2) = edges[i], edges[j]
        # Avoid creating self-loops or parallel edges (simple graph)
        if u1 == v2 or u2 == v1:
            continue
        if H.has_edge(u1, v2) or H.has_edge(u2, v1):
            continue
        # Apply swap.
        attrs = dict(layer=layer, cp_type=H[u1][v1].get("cp_type"))
        H.remove_edge(u1, v1)
        H.remove_edge(u2, v2)
        H.add_edge(u1, v2, **attrs)
        attrs2 = dict(layer=layer, cp_type=H[u2][v2 if False else v1].get("cp_type", attrs["cp_type"])) if False else attrs
        H.add_edge(u2, v1, **attrs)
        edges[i] = (u1, v2)
        edges[j] = (u2, v1)
    return H


def null_model_er(G: nx.Graph, seed: Optional[int] = None) -> nx.Graph:
    """Erdős–Rényi multilayer null with matched per-layer edge counts.

    Each layer (and the coupling layer) is rewired to a random graph with
    the same number of edges, drawn uniformly without replacement. This is
    the *weakest* null — it doesn't preserve degrees.
    """
    rng = np.random.default_rng(seed)
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    nodes_by_layer: Dict[str, List] = defaultdict(list)
    for n, a in G.nodes(data=True):
        nodes_by_layer[a.get("layer", "?")].append(n)
    edges_by_layer: Dict[str, int] = defaultdict(int)
    for _, _, a in G.edges(data=True):
        edges_by_layer[a.get("layer", "?")] += 1
    for layer, m in edges_by_layer.items():
        if layer == "coupling":
            # Sample inter-layer pairs uniformly across all distinct layer pairs.
            all_nodes = list(G.nodes)
            pairs_drawn = set()
            tries = 0
            while len(pairs_drawn) < m and tries < m * 50:
                u = all_nodes[rng.integers(0, len(all_nodes))]
                v = all_nodes[rng.integers(0, len(all_nodes))]
                if u == v or u[0] == v[0]:
                    tries += 1
                    continue
                key = (u, v) if u < v else (v, u)
                if key in pairs_drawn:
                    tries += 1
                    continue
                pairs_drawn.add(key)
                tries += 1
            for u, v in pairs_drawn:
                H.add_edge(u, v, layer="coupling", cp_type=None)
        else:
            pool = nodes_by_layer.get(layer, [])
            if len(pool) < 2:
                continue
            pairs_drawn = set()
            tries = 0
            while len(pairs_drawn) < m and tries < m * 50:
                u = pool[rng.integers(0, len(pool))]
                v = pool[rng.integers(0, len(pool))]
                if u == v:
                    tries += 1
                    continue
                key = (u, v) if u < v else (v, u)
                if key in pairs_drawn:
                    tries += 1
                    continue
                pairs_drawn.add(key)
                tries += 1
            for u, v in pairs_drawn:
                H.add_edge(u, v, layer=layer, cp_type=None)
    return H


def null_model_z_scores(
    G: nx.Graph,
    stat_fn: Callable[[nx.Graph], float],
    n: int = 200,
    kind: str = "config",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Compare an observed scalar to a null-model ensemble.

    ``stat_fn`` should map a graph to a single float (e.g. average BC, AUC
    of percolation under degree-targeted attack, algebraic connectivity).
    Returns ``{observed, null_mean, null_std, z, p_one_sided}``. The two-sided
    p-value can be derived as ``2 * min(p_one_sided, 1 - p_one_sided)``.
    """
    rng = np.random.default_rng(seed)
    observed = float(stat_fn(G))
    samples = np.empty(n, dtype=float)
    factory = null_model_configuration if kind == "config" else null_model_er
    for k in range(n):
        seed_k = int(rng.integers(0, 2**31 - 1))
        samples[k] = float(stat_fn(factory(G, seed=seed_k)))
    null_mean = float(np.mean(samples))
    null_std = float(np.std(samples, ddof=1)) if n > 1 else 0.0
    z = (observed - null_mean) / null_std if null_std > 0 else float("nan")
    # One-sided p: fraction of nulls at least as extreme on the upper tail.
    p_upper = float(np.mean(samples >= observed)) if samples.size else float("nan")
    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": float(z),
        "p_one_sided_upper": p_upper,
    }


# ─────────────────────────────────────────────────────────────────────────────
# E12 — Community structure & cross-layer bridges
# ─────────────────────────────────────────────────────────────────────────────


def community_partition(G: nx.Graph, seed: Optional[int] = None) -> Dict:
    """Greedy-modularity community partition on the multilayer graph.

    Returns a dict ``node → community_id``. We use ``networkx`` greedy
    modularity (Clauset–Newman–Moore) instead of Louvain so the only
    dependency is networkx itself (Louvain requires ``python-louvain``).
    Output community ids are integers ≥ 0.
    """
    if G.number_of_nodes() == 0:
        return {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
    except Exception:
        # Last-resort fallback: each connected component is one community.
        comms = list(nx.connected_components(G))
    out: Dict = {}
    for cid, comp in enumerate(comms):
        for n in comp:
            out[n] = cid
    return out


def bridge_score(G: nx.Graph, partition: Dict) -> Dict:
    """For each node v, fraction of neighbours that belong to a different
    community. 0 = entirely embedded in own community; 1 = every neighbour
    is in another community. This is the "bridge" signal used in E12.
    """
    out: Dict = {}
    for v in G.nodes:
        nbrs = list(G.neighbors(v))
        if not nbrs:
            out[v] = 0.0
            continue
        own = partition.get(v)
        diff = sum(1 for u in nbrs if partition.get(u) != own)
        out[v] = diff / len(nbrs)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# E13 — Spectral robustness scalars
# ─────────────────────────────────────────────────────────────────────────────


def spectral_robustness(G: nx.Graph) -> Dict[str, float]:
    """Per-layer and supra-graph spectral robustness scalars.

    Computes the algebraic connectivity λ₂ (Fiedler value) of each layer's
    Laplacian and of the supra-Laplacian. λ₂ is the smallest non-zero
    eigenvalue; it scales with how hard it is to disconnect the graph.
    Returns NaN for components with < 2 connected nodes.
    """
    out: Dict[str, float] = {}
    by_layer: Dict[str, nx.Graph] = defaultdict(nx.Graph)
    for u, v, a in G.edges(data=True):
        layer = a.get("layer", "?")
        if layer != "coupling":
            by_layer[layer].add_edge(u, v)

    for layer, H in by_layer.items():
        out[f"lambda2_{layer}"] = _algebraic_connectivity(H)
    out["lambda2_supra"] = _algebraic_connectivity(G)
    return out


def kirchhoff_index(G: nx.Graph) -> float:
    """Effective graph resistance (Kirchhoff index): Σᵢⱼ Rᵢⱼ over all
    distinct pairs. Lower = more robust (easier to move flow between any
    two nodes). Computed from non-zero eigenvalues of the Laplacian:
    ``Kf = N · Σ 1/λᵢ`` for ``λᵢ > 0``.
    Defined per *connected* graph only — falls back to NaN otherwise.
    """
    if G.number_of_nodes() < 2 or not nx.is_connected(G):
        return float("nan")
    L = nx.laplacian_matrix(G).astype(float).toarray()
    try:
        eigvals = np.linalg.eigvalsh(L)
    except Exception:
        return float("nan")
    nonzero = eigvals[eigvals > 1e-9]
    if nonzero.size == 0:
        return float("nan")
    return float(G.number_of_nodes() * np.sum(1.0 / nonzero))


def _algebraic_connectivity(H: nx.Graph) -> float:
    """Smallest non-zero Laplacian eigenvalue, NaN on degenerate input."""
    if H.number_of_nodes() < 2:
        return float("nan")
    if not nx.is_connected(H):
        # Multi-component: λ₂ = 0 by definition. Surfacing 0 is more useful
        # than NaN for downstream comparisons because the graph IS more
        # fragile than a connected one with the same node count.
        return 0.0
    try:
        return float(nx.algebraic_connectivity(H, normalized=False))
    except Exception:
        return float("nan")
