# pip install jax jaxlib networkx pandas numpy

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
import pandas as pd

import monee
import monee.model as mm
import monee.network.mes as mes
from monee.model.core import Node as MNode

jax.config.update("jax_enable_x64", True)


# =============================================================================
# CONFIG
# =============================================================================


@dataclass(frozen=True)
class CPMetricConfig:
    # Linearization reliability
    FLOW_MIN: float = 1e-4  # edges with |flow| < FLOW_MIN excluded from linearization (gas/water pipes)
    EPS_MARGIN: float = 1e-6  # numerical epsilon for margin division
    CLIP_STRESS: float = 1e6  # cap stress to avoid domination by a single edge

    # Aggregation of stress over edges: agg = w_mean*mean + w_max*max
    AGG_MEAN_WEIGHT: float = 0.7
    AGG_MAX_WEIGHT: float = 0.3

    # Carrier weights (calibrate if needed)
    W_POWER: float = 1.0
    W_GAS: float = 1.0
    W_HEAT: float = 1.0

    # Topology scaling: topo_factor = 1 + TOPO_ALPHA*BC_group
    TOPO_ALPHA: float = 1

    # Throughput scaling
    USE_THROUGHPUT_PROXY: bool = True

    # Failure probabilities (replace with real/proxy values for probabilistic claims)
    CP_FAIL_PROB: Optional[Dict[type, float]] = None

    # Robust margins when explicit limits are missing
    USE_PSEUDO_LIMITS: bool = True
    PSEUDO_HEADROOM: float = 0.3  # pseudo_limit = (1+headroom)*max(|flow0|, scale)
    MIN_MARGIN: float = 1e-3  # never allow margin < MIN_MARGIN for stress denom

    # Override per-branch thermal limit (MVA) for all power branches.
    # Use when the model stores an unrealistic max_i_ka (e.g. benchmark placeholder).
    # None = use model value as-is.
    POWER_BRANCH_LIMIT_MVA_OVERRIDE: Optional[float] = None

    # Binding constraint visibility
    BINDING_MARGIN_EPS: float = 1e-6

    # HX scaling relative to typical WaterPipe conductance
    HX_KAPPA: float = 10.0

    # Diagnostics
    RETURN_DEBUG: bool = True


DEFAULT_FAIL_PROB = {
    mm.CHP: 0.1,
    mm.PowerToHeat: 0.1,
    mm.PowerToGas: 0.1,
    mm.GasToPower: 0.1,
}


# =============================================================================
# UTIL: safe attribute reading
# =============================================================================


def _val(x, default=None):
    try:
        return mm.value(x)
    except Exception:
        try:
            return float(x)
        except Exception:
            return default


def _first_attr(obj, names: List[str], default=None):
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            vv = _val(v, default=None)
            if vv is not None:
                return vv
    return default


def _robust_margins(limits: np.ndarray, flows0: np.ndarray, cfg: CPMetricConfig):
    """
    Returns:
      margins: >= MIN_MARGIN (finite)
      has_explicit_limits: boolean mask
      binding_mask: margin <= BINDING_MARGIN_EPS (before MIN_MARGIN clamp)
      pseudo_used_ratio: float
    """
    limits = np.asarray(limits, dtype=float)
    flows0 = np.abs(np.asarray(flows0, dtype=float))

    finite = np.isfinite(limits) & (limits > 0)
    raw_margins = np.empty_like(flows0)

    if flows0.size == 0:
        return np.empty(0), np.empty(0, dtype=bool), np.empty(0, dtype=bool), 0.0

    # explicit margins where possible
    raw_margins[finite] = limits[finite] - flows0[finite]

    missing = ~finite
    pseudo_used_ratio = float(np.mean(missing)) if missing.size else 0.0
    if np.any(missing):
        if not cfg.USE_PSEUDO_LIMITS:
            raw_margins[missing] = np.inf
        else:
            # robust scale so pseudo-limit isn't zero
            scale = np.percentile(flows0[flows0 > 0], 90) if np.any(flows0 > 0) else 1.0
            base = np.maximum(flows0, max(scale, cfg.MIN_MARGIN))
            pseudo_limit = (1.0 + cfg.PSEUDO_HEADROOM) * base
            raw_margins[missing] = pseudo_limit[missing] - flows0[missing]

    binding_mask = raw_margins <= cfg.BINDING_MARGIN_EPS
    margins = np.maximum(raw_margins, cfg.MIN_MARGIN)

    # Ensure finite margins everywhere (defensive)
    margins[~np.isfinite(margins)] = cfg.MIN_MARGIN

    return margins, finite, binding_mask, pseudo_used_ratio


# =============================================================================
# ELECTRICITY (AC) PTDF with distributed balancing
# =============================================================================


def build_ybus_internal(nb, branches):
    Ybus = jnp.zeros((nb, nb), dtype=jnp.complex128)
    for f, t, r, x, gfr, bfr, gto, bto, tap, shift_rad in branches:
        y_series = 1.0 / complex(r, x)
        y_sh_fr = complex(gfr, bfr)
        y_sh_to = complex(gto, bto)
        a = tap if tap != 0.0 else 1.0
        tap_c = a * jnp.exp(1j * shift_rad)

        Ybus = Ybus.at[f, t].add(-y_series / jnp.conj(tap_c))
        Ybus = Ybus.at[t, f].add(-y_series / tap_c)
        Ybus = Ybus.at[f, f].add((y_series / (tap_c * jnp.conj(tap_c))) + y_sh_fr)
        Ybus = Ybus.at[t, t].add(y_series + y_sh_to)
    return Ybus


def power_injections(theta, V, Ybus):
    E = V * jnp.exp(1j * theta)
    current = Ybus @ E
    S = E * jnp.conj(current)
    return S.real, S.imag


def line_active_powers(theta, V, branches):
    P = []
    for f, t, r, x, _gfr, _bfr, _gto, _bto, tap, shift_rad in branches:
        a = tap if tap != 0.0 else 1.0
        y = 1.0 / complex(r, x)
        g = float(y.real)
        b = float(y.imag)
        Vi, Vj = V[f], V[t]
        dth = theta[f] - theta[t] - shift_rad
        Pij = (Vi * Vi / (a * a)) * g - (Vi * Vj / a) * (
            g * jnp.cos(dth) + b * jnp.sin(dth)
        )
        P.append(Pij)
    return jnp.array(P)


def build_reduced_jacobian(theta, V, Ybus, bus_types):
    """
    bus_types: 0=PQ, 1=PV, 2=Slack. (PV detection is best-effort; debug reports pv_count.)
    States: theta for all non-slack; V magnitude for PQ only.
    Equations: P for all non-slack; Q for PQ only.
    """
    is_slack = bus_types == 2
    is_pq = bus_types == 0

    idx_theta = jnp.where(~is_slack)[0]
    idx_Vpq = jnp.where(is_pq)[0]
    idx_P = jnp.where(~is_slack)[0]
    idx_Q = jnp.where(is_pq)[0]

    def P_of(th, Vm):
        return power_injections(th, Vm, Ybus)[0]

    def Q_of(th, Vm):
        return power_injections(th, Vm, Ybus)[1]

    H = jax.jacobian(P_of, argnums=0)(theta, V)
    N = jax.jacobian(P_of, argnums=1)(theta, V)
    M = jax.jacobian(Q_of, argnums=0)(theta, V)
    L = jax.jacobian(Q_of, argnums=1)(theta, V)

    H_red = H[jnp.ix_(idx_P, idx_theta)]
    N_red = N[jnp.ix_(idx_P, idx_Vpq)]
    M_red = M[jnp.ix_(idx_Q, idx_theta)]
    L_red = L[jnp.ix_(idx_Q, idx_Vpq)]

    J_red = jnp.concatenate(
        [
            jnp.concatenate([H_red, N_red], axis=1),
            jnp.concatenate([M_red, L_red], axis=1),
        ],
        axis=0,
    )
    return J_red, idx_theta, idx_Vpq, idx_P, idx_Q


def build_branches_power(monee_net):
    buses = monee_net.nodes_by_type(mm.Bus)
    bus_ids = [n.id for n in buses]
    id_to_local = {nid: i for i, nid in enumerate(bus_ids)}

    branch_tuples = []
    branch_ids = []
    for b in monee_net.branches_by_type(mm.GenericPowerBranch):
        if not (b.active and int(_val(b.model.on_off, 1)) == 1):
            continue
        if b.from_node_id not in id_to_local or b.to_node_id not in id_to_local:
            continue
        br_r = _val(b.model.br_r, 0.0)
        br_x = _val(b.model.br_x, 0.0)
        # Avoid zero impedance (bus-bar / short-circuit branches)
        if abs(br_r) < 1e-9 and abs(br_x) < 1e-9:
            br_x = 1e-6
        branch_tuples.append(
            (
                id_to_local[b.from_node_id],
                id_to_local[b.to_node_id],
                br_r,
                br_x,
                _val(b.model.g_fr, 0.0),
                _val(b.model.b_fr, 0.0),
                _val(b.model.g_to, 0.0),
                _val(b.model.b_to, 0.0),
                _val(b.model.tap, 1.0),
                _val(b.model.shift, 0.0),
            )
        )
        branch_ids.append(b.id)
    return branch_tuples, branch_ids, bus_ids, id_to_local


def build_ybus_power(monee_net):
    branches, branch_ids, bus_ids, id_to_local = build_branches_power(monee_net)
    Ybus = build_ybus_internal(len(bus_ids), branches)
    return branches, branch_ids, Ybus, bus_ids, id_to_local


def _power_operating_point(monee_net, bus_ids):
    bus_map = {n.id: n for n in monee_net.nodes_by_type(mm.Bus)}
    theta = jnp.array(
        [_val(bus_map[nid].model.va_radians, 0.0) for nid in bus_ids], dtype=jnp.float64
    )
    V = jnp.array(
        [_val(bus_map[nid].model.vm_pu, 1.0) for nid in bus_ids], dtype=jnp.float64
    )

    bus_types = []
    pv_count = 0
    slack_count = 0

    for nid in bus_ids:
        node = bus_map[nid]
        if monee_net.has_any_child_of_type(node, mm.ExtPowerGrid):
            bus_types.append(2)
            slack_count += 1
        else:
            # Best-effort PV detection: only counted if a known generator child type is found.
            is_pv = False
            for t in ("Generator", "Gen", "SynchronousGenerator"):
                if hasattr(mm, t):
                    try:
                        if monee_net.has_any_child_of_type(node, getattr(mm, t)):
                            is_pv = True
                            break
                    except Exception:
                        pass
            if is_pv:
                pv_count += 1
            bus_types.append(1 if is_pv else 0)

    return (
        theta,
        V,
        jnp.array(bus_types, dtype=jnp.int32),
        {"pv_count": pv_count, "slack_count": slack_count},
    )


def _distributed_balancing_vector_power(monee_net, bus_ids, id_to_local, s_local):
    """
    +1 at source bus, distributed -1 across all ExtPowerGrid buses if present,
    else distributed across all other buses.
    """
    bus_map = {n.id: n for n in monee_net.nodes_by_type(mm.Bus)}

    slack_locs = []
    for i, nid in enumerate(bus_ids):
        if monee_net.has_any_child_of_type(bus_map[nid], mm.ExtPowerGrid):
            slack_locs.append(i)

    nb = len(bus_ids)
    dP = np.zeros(nb, dtype=float)
    dP[s_local] += 1.0

    if slack_locs:
        bal = [i for i in slack_locs if i != s_local]
    else:
        bal = [i for i in range(nb) if i != s_local]

    if not bal:
        return jnp.array(dP)

    w = np.ones(len(bal), dtype=float)
    w = w / w.sum()
    for bi, wi in zip(bal, w):
        dP[bi] -= wi
    return jnp.array(dP)


def ac_ptdf_distributed(theta, V, Ybus, branches, bus_types, dP, dQ=None):
    nb = theta.shape[0]
    if dQ is None:
        dQ = jnp.zeros(nb)

    J_red, idx_theta, idx_Vpq, idx_P, idx_Q = build_reduced_jacobian(
        theta, V, Ybus, bus_types
    )
    rhs = jnp.concatenate([dP[idx_P], dQ[idx_Q]])

    try:
        dx = jnp.linalg.solve(J_red, rhs)
    except Exception:
        dx = jnp.linalg.lstsq(J_red, rhs, rcond=None)[0]

    n_th = idx_theta.shape[0]
    dtheta = jnp.zeros(nb).at[idx_theta].set(dx[:n_th])
    dV = jnp.zeros(nb).at[idx_Vpq].set(dx[n_th:])

    def P_lines(th, Vm):
        return line_active_powers(th, Vm, branches)

    _, dP_lines = jax.jvp(P_lines, (theta, V), (dtheta, dV))
    return dP_lines


# =============================================================================
# GAS + HEAT PTDF (Laplacian-based) with component correctness
# =============================================================================

_R_SPECIFIC_GAS = 504.5
_WATER_DENSITY = 998.0

_V_MAX_WATER_MPS = 2.5  # typical DH velocity cap
_HX_RESISTANCE_FALLBACK = 1e-3  # only used if there are no WaterPipes to scale from


def _calc_C_squared(diameter_m, length_m, t_k, compressibility):
    return (math.pi**2 * diameter_m**5) / (
        128.0 * length_m * _R_SPECIFIC_GAS * t_k * compressibility
    )


def _darcy_resistance(pipe_model):
    d = _val(getattr(pipe_model, "diameter_m", None), None)
    L = _val(getattr(pipe_model, "length_m", None), None)
    if d is None or L is None or d == 0:
        return None
    A = math.pi * d**2 / 4.0
    friction = getattr(pipe_model, "friction", None)
    f_raw = _val(friction, 0.02)
    f = max(float(f_raw), 1e-6)
    return f * (L / d) / (2.0 * _WATER_DENSITY * A**2)


def _connected_components_from_edges(n, edges):
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        g[v].append(u)
    comps = []
    seen = set()
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        comp = set([i])
        seen.add(i)
        while stack:
            x = stack.pop()
            for y in g[x]:
                if y not in seen:
                    seen.add(y)
                    comp.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps


def _ptdf_laplacian_distributed(B, pipe_data, rhs, cfg: CPMetricConfig):
    """
    Solve B dx = rhs per connected component with gauge-fix.
    If rhs does not sum to 0 within a component, that component is infeasible -> flagged.
    """
    n = B.shape[0]
    if n == 0 or len(pipe_data) == 0:
        return np.zeros(len(pipe_data)), True

    edges = [(fi, ti) for fi, ti, _ in pipe_data]
    comps = _connected_components_from_edges(n, edges)

    dx = np.zeros(n, dtype=float)
    reliable = True

    for comp in comps:
        comp = sorted(comp)
        r = rhs[comp]
        if abs(r.sum()) > 1e-9:
            reliable = False
            continue

        ref = comp[0]
        mask = [i for i in comp if i != ref]
        if not mask:
            continue

        B_red = B[np.ix_(mask, mask)]
        rhs_red = rhs[mask]

        try:
            sol = np.linalg.solve(B_red, rhs_red)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(B_red, rhs_red, rcond=None)[0]

        for i, v in zip(mask, sol):
            dx[i] = v
        dx[ref] = 0.0

    ptdf = np.array([b * (dx[fi] - dx[ti]) for fi, ti, b in pipe_data], dtype=float)
    return ptdf, reliable


def _distributed_balancing_rhs(n: int, s_idx: int, bal_idxs: List[int]):
    rhs = np.zeros(n, dtype=float)
    rhs[s_idx] += 1.0
    bal = [i for i in bal_idxs if i != s_idx]
    if not bal:
        return rhs
    w = np.ones(len(bal), dtype=float)
    w = w / w.sum()
    for bi, wi in zip(bal, w):
        rhs[bi] -= wi
    return rhs


def _gas_grid(monee_net):
    for n in monee_net.nodes_by_type(mm.Junction):
        if n.grid is not None and n.grid.name == "gas":
            return n.grid
    return None


def _water_grid(monee_net):
    for n in monee_net.nodes_by_type(mm.Junction):
        if n.grid is not None and n.grid.name == "water":
            return n.grid
    return None


def _get_gas_hhv(monee_net) -> float:
    try:
        gg = _gas_grid(monee_net)
        if gg is not None and hasattr(gg, "higher_heating_value"):
            return float(gg.higher_heating_value)
    except Exception:
        pass
    return 15.3  # kWh/kg typical NG (≈55 MJ/kg)


def _gas_pipe_max_flow(pipe_model, gas_grid) -> float:
    """
    Weymouth capacity proxy in kg/s using grid p^2 bounds and reference pressure.
    """
    d = _val(getattr(pipe_model, "diameter_m", None), None)
    L = _val(getattr(pipe_model, "length_m", None), None)
    if not d or not L or d <= 0 or L <= 0:
        return np.inf

    t_k = float(getattr(gas_grid, "gas_temperature", 300.0))
    z = float(getattr(gas_grid, "compressibility", 1.0))
    C2 = _calc_C_squared(d, L, t_k, z)

    p_sq_max = float(getattr(gas_grid, "p_squared_pu_max", 1.3))
    p_sq_min = float(getattr(gas_grid, "p_squared_pu_min", 0.7))
    p_ref = float(getattr(gas_grid, "pressure_ref", 1e6))

    if not (p_sq_max > p_sq_min) or p_ref <= 0:
        return np.inf

    delta = (p_sq_max - p_sq_min) * (p_ref**2)
    return math.sqrt(C2 * delta) if delta > 0 else np.inf


def _water_pipe_max_flow(pipe_model) -> float:
    d = _val(getattr(pipe_model, "diameter_m", None), None)
    if not d or d <= 0:
        return np.inf
    A = math.pi * d**2 / 4.0
    return _WATER_DENSITY * _V_MAX_WATER_MPS * A


def build_gas_susceptance(monee_net, cfg: CPMetricConfig):
    junctions = [
        n
        for n in monee_net.nodes_by_type(mm.Junction)
        if n.grid is not None and n.grid.name == "gas"
    ]
    junc_ids = [n.id for n in junctions]
    n = len(junc_ids)
    idx = {nid: i for i, nid in enumerate(junc_ids)}

    gas_grid = _gas_grid(monee_net)
    t_k = (
        float(getattr(gas_grid, "gas_temperature", 300.0))
        if gas_grid is not None
        else 300.0
    )
    z = (
        float(getattr(gas_grid, "compressibility", 1.0))
        if gas_grid is not None
        else 1.0
    )

    B = np.zeros((n, n), dtype=float)
    pipe_data = []
    pipe_ids = []

    for pipe in monee_net.branches_by_type(mm.GasPipe):
        if not (pipe.active and int(_val(pipe.model.on_off, 1)) == 1):
            continue
        fi = idx.get(pipe.from_node_id)
        ti = idx.get(pipe.to_node_id)
        if fi is None or ti is None:
            continue

        m0 = abs(_val(pipe.model.mass_flow, 0.0))
        if m0 < cfg.FLOW_MIN:
            continue

        C2 = _calc_C_squared(
            float(_val(pipe.model.diameter_m, 0.0)),
            float(_val(pipe.model.length_m, 1.0)),
            t_k,
            z,
        )
        b = C2 / (2.0 * m0)

        B[fi, fi] += b
        B[ti, ti] += b
        B[fi, ti] -= b
        B[ti, fi] -= b
        pipe_data.append((fi, ti, b))
        pipe_ids.append(pipe.id)

    return B, pipe_data, pipe_ids, junc_ids


def build_heat_susceptance(monee_net, cfg: CPMetricConfig):
    """
    Water network susceptance with HX coupling.
    - WaterPipes: b = 1/(2*Rm*m0) around operating point.
    - HeatExchanger: added as strong coupling b_hx = HX_KAPPA * median(b_waterpipe).
      If no WaterPipes exist in model/selection, fallback to constant.
    """
    junctions = [
        n
        for n in monee_net.nodes_by_type(mm.Junction)
        if n.grid is not None and n.grid.name == "water"
    ]
    junc_ids = [n.id for n in junctions]
    n = len(junc_ids)
    idx = {nid: i for i, nid in enumerate(junc_ids)}

    B = np.zeros((n, n), dtype=float)
    pipe_data = []
    pipe_ids = []

    def _add_edge(fi, ti, b):
        B[fi, fi] += b
        B[ti, ti] += b
        B[fi, ti] -= b
        B[ti, fi] -= b
        pipe_data.append((fi, ti, b))

    water_b_list = []

    # WaterPipes
    for pipe in monee_net.branches_by_type(mm.WaterPipe):
        if not (pipe.active and int(_val(pipe.model.on_off, 1)) == 1):
            continue
        fi = idx.get(pipe.from_node_id)
        ti = idx.get(pipe.to_node_id)
        if fi is None or ti is None:
            continue

        Rm = _darcy_resistance(pipe.model)
        if Rm is None or Rm <= 0:
            continue

        m0 = abs(_val(pipe.model.mass_flow, 0.0))
        if m0 < cfg.FLOW_MIN:
            continue

        b = 1.0 / (2.0 * Rm * m0)
        _add_edge(fi, ti, b)
        water_b_list.append(b)
        pipe_ids.append(pipe.id)

    # HX conductance scaling: scale to median WaterPipe conductance.
    # Guard: if median is zero (all pipes at flow floor) or no pipes exist,
    # fall back to the fixed resistance constant.
    if water_b_list:
        b_med = float(np.median(water_b_list))
        if b_med > 0.0:
            b_hx = cfg.HX_KAPPA * b_med
        else:
            b_hx = 1.0 / (2.0 * _HX_RESISTANCE_FALLBACK * max(cfg.FLOW_MIN, 1e-6))
    else:
        b_hx = 1.0 / (2.0 * _HX_RESISTANCE_FALLBACK * max(cfg.FLOW_MIN, 1e-6))

    # HeatExchanger branches (hydraulic coupling)
    for hx in monee_net.branches_by_type(mm.HeatExchanger):
        if not hx.active:
            continue
        fi = idx.get(hx.from_node_id)
        ti = idx.get(hx.to_node_id)
        if fi is None or ti is None:
            continue
        _add_edge(fi, ti, b_hx)
        pipe_ids.append(hx.id)

    return B, pipe_data, pipe_ids, junc_ids


# =============================================================================
# LIMITS + FLOWS: canonical definitions (with robust fallback)
# =============================================================================


def _power_branch_limit_and_flow(
    monee_net, branch_id, cfg: "CPMetricConfig" = None
) -> Tuple[float, float]:
    """
    Best-effort:
    - If cfg.POWER_BRANCH_LIMIT_MVA_OVERRIDE is set, use that for all branches.
    - Else try explicit MVA/MW limits on branch model (if exist).
    - Else use current-based limit: max_i_ka [kA] * base_kv [kV] * sqrt(3) * parallel.
    Flow: |p_from_mw| if available, else 0.
    """
    try:
        b = monee_net.branch_by_id(branch_id)
        bm = b.model
    except Exception:
        return np.inf, 0.0

    flow0 = abs(_first_attr(bm, ["p_from_mw", "p_mw", "p_from", "p"], default=0.0))

    # cfg override takes highest priority
    if cfg is not None and cfg.POWER_BRANCH_LIMIT_MVA_OVERRIDE is not None:
        return float(cfg.POWER_BRANCH_LIMIT_MVA_OVERRIDE), float(flow0)

    # Prefer direct thermal/rating fields if present
    direct = _first_attr(
        bm, ["rate_a_mva", "s_max_mva", "thermal_limit_mw", "p_max_mw"], default=None
    )
    if direct is not None and float(direct) > 0:
        return float(direct), float(flow0)

    max_i_ka = _first_attr(bm, ["max_i_ka"], default=None)
    if max_i_ka is None or float(max_i_ka) <= 0:
        return np.inf, float(flow0)

    try:
        from_node = monee_net.node_by_id(b.from_node_id)
        base_kv = _first_attr(from_node.model, ["base_kv", "vn_kv"], default=None)
    except Exception:
        base_kv = None

    if base_kv is None or float(base_kv) <= 0:
        return np.inf, float(flow0)

    parallel = float(_first_attr(bm, ["parallel"], default=1.0) or 1.0)
    limit = float(max_i_ka) * float(base_kv) * math.sqrt(3.0) * parallel  # MVA
    return float(limit), float(flow0)


def _gas_pipe_limit_and_flow(monee_net, pipe_id, gas_grid=None) -> Tuple[float, float]:
    try:
        p = monee_net.branch_by_id(pipe_id)
        pm = p.model
    except Exception:
        return np.inf, 0.0

    flow0 = abs(_first_attr(pm, ["mass_flow", "mass_flow_pos", "m_dot"], default=0.0))

    # Prefer explicit limits if present
    direct = _first_attr(pm, ["mass_flow_max", "m_max", "max_mass_flow"], default=None)
    if direct is not None and float(direct) > 0:
        return float(direct), float(flow0)

    if gas_grid is None:
        return np.inf, float(flow0)

    return float(_gas_pipe_max_flow(pm, gas_grid)), float(flow0)


def _heat_pipe_limit_and_flow(monee_net, pipe_or_hx_id) -> Tuple[float, float]:
    """
    For WaterPipe: velocity-based capacity.
    For HeatExchanger: no geometry -> treat limit as inf so pseudo-limit fallback can handle.
    """
    try:
        p = monee_net.branch_by_id(pipe_or_hx_id)
        pm = p.model
    except Exception:
        return np.inf, 0.0

    flow0 = abs(_first_attr(pm, ["mass_flow", "mass_flow_pos", "m_dot"], default=0.0))

    # Prefer explicit limits if present
    direct = _first_attr(pm, ["mass_flow_max", "m_max", "max_mass_flow"], default=None)
    if direct is not None and float(direct) > 0:
        return float(direct), float(flow0)

    if isinstance(pm, mm.WaterPipe):
        return float(_water_pipe_max_flow(pm)), float(flow0)

    # HX: no good limit -> force pseudo-limits to kick in
    return np.inf, float(flow0)


# =============================================================================
# TOPOLOGY: copied graph; robust edge->branch mapping attempts
# =============================================================================


def compute_physical_topology_metrics(monee_net):
    """
    Returns: (G, bc, deg, debug)
    debug includes topo_mapped_ratio so you know if weights are real or all-ones.
    """
    G0 = monee_net._network_internal
    G = nx.Graph(G0)  # copy

    mapped = 0
    total = 0

    for u, v, data in G.edges(data=True):
        total += 1
        w = 1.0
        br = None

        # Attempt mapping in plausible ways
        # (a) edge data might store branch_id
        bid = data.get("branch_id", None)
        if bid is not None:
            try:
                br = monee_net.branch_by_id(bid)
            except Exception:
                br = None

        # (b) some monee graphs use the edge key itself as a branch id (original approach)
        if br is None:
            try:
                br = monee_net.branch_by_id((u, v))
            except Exception:
                br = None

        if br is not None:
            mapped += 1
            bm = br.model
            if isinstance(bm, mm.GenericPowerBranch):
                x = abs(_val(getattr(bm, "br_x", 0.0), 0.0))
                w = x if x > 0 else 1.0
            elif isinstance(bm, mm.GasPipe):
                d = _val(getattr(bm, "diameter_m", 0.0), 0.0)
                L = _val(getattr(bm, "length_m", 1.0), 1.0)
                w = (L / (d**5)) if d > 0 else 1.0
            elif isinstance(bm, mm.WaterPipe):
                Rm = _darcy_resistance(bm)
                w = Rm if (Rm is not None and Rm > 0) else 1.0

        G.edges[u, v]["weight"] = float(w)

    bc = nx.betweenness_centrality(G, weight="weight")
    deg = dict(G.degree())
    debug = {
        "topo_edge_count": total,
        "topo_mapped_ratio": (mapped / total) if total else 0.0,
    }
    return G, bc, deg, debug


def compute_stress_topology_metrics(monee_net, ctx: "CarrierPTDFContext", cfg: "CPMetricConfig"):
    """
    Betweenness centrality weighted by branch stress = |flow| / margin.
    More stressed (loaded) branches are treated as shorter paths
    (weight = 1 / (stress + eps)), so BC highlights nodes that lie
    on paths through congested branches.

    Returns: (G, bc_nodes, debug)
    """
    # Build stress lookup: branch_id -> loading stress
    stress_by_id = {}

    if "built" in ctx.power:
        for i, bid in enumerate(ctx.power["branch_ids"]):
            margin = float(ctx.power["margins"][i]) if i < len(ctx.power["margins"]) else cfg.MIN_MARGIN
            try:
                flow0 = abs(_first_attr(monee_net.branch_by_id(bid).model,
                                        ["p_from_mw", "p_mw", "p_from", "p"], default=0.0))
            except Exception:
                flow0 = 0.0
            stress_by_id[bid] = float(flow0) / (margin + cfg.EPS_MARGIN)

    if "built" in ctx.gas:
        for i, pid in enumerate(ctx.gas["pipe_ids"]):
            margin = float(ctx.gas["margins"][i]) if i < len(ctx.gas["margins"]) else cfg.MIN_MARGIN
            try:
                flow0 = abs(_first_attr(monee_net.branch_by_id(pid).model,
                                        ["mass_flow", "mass_flow_pos", "m_dot"], default=0.0))
            except Exception:
                flow0 = 0.0
            stress_by_id[pid] = float(flow0) / (margin + cfg.EPS_MARGIN)

    if "built" in ctx.heat:
        for i, pid in enumerate(ctx.heat["pipe_ids"]):
            margin = float(ctx.heat["margins"][i]) if i < len(ctx.heat["margins"]) else cfg.MIN_MARGIN
            try:
                flow0 = abs(_first_attr(monee_net.branch_by_id(pid).model,
                                        ["mass_flow", "mass_flow_pos", "m_dot"], default=0.0))
            except Exception:
                flow0 = 0.0
            stress_by_id[pid] = float(flow0) / (margin + cfg.EPS_MARGIN)

    G0 = monee_net._network_internal
    G = nx.Graph(G0)

    EPS = 1e-9
    mapped = 0
    total = 0

    for u, v, data in G.edges(data=True):
        total += 1
        stress_val = None

        # Try branch_id from edge data
        bid = data.get("branch_id", None)
        if bid is not None and bid in stress_by_id:
            stress_val = stress_by_id[bid]
            mapped += 1

        # Try edge key as branch_id tuple
        if stress_val is None:
            for candidate in [(u, v, 0), (v, u, 0), (u, v), (v, u)]:
                if candidate in stress_by_id:
                    stress_val = stress_by_id[candidate]
                    mapped += 1
                    break

        # Try monee branch lookup
        if stress_val is None:
            for candidate in [(u, v), (v, u)]:
                try:
                    br = monee_net.branch_by_id(candidate)
                    if br.id in stress_by_id:
                        stress_val = stress_by_id[br.id]
                        mapped += 1
                        break
                except Exception:
                    pass

        # weight = 1/(stress+eps): highly loaded edges become "short" (preferred)
        if stress_val is not None and stress_val > 0:
            w = 1.0 / (stress_val + EPS)
        else:
            w = 1.0 / EPS  # unmapped or zero-flow: treat as very short (neutral)
        G.edges[u, v]["stress_weight"] = float(np.clip(w, 0.0, 1e12))

    bc = nx.betweenness_centrality(G, weight="stress_weight")
    debug = {
        "stress_topo_edge_count": total,
        "stress_topo_mapped_ratio": (mapped / total) if total else 0.0,
    }
    return G, bc, debug


def _group_bc(G, compound, weight="weight"):
    try:
        internal_ids = [c.id for c in compound.component_of_type(MNode)]
        return nx.group_betweenness_centrality(G, internal_ids, weight=weight)
    except Exception:
        return 0.0


# =============================================================================
# THROUGHPUT proxy (dimensionless ~ pu on sn_mva)
# =============================================================================


def _system_sn_mva(monee_net) -> float:
    try:
        buses = monee_net.nodes_by_type(mm.Bus)
        if buses and buses[0].grid is not None:
            return float(getattr(buses[0].grid, "sn_mva", 100.0))
    except Exception:
        pass
    return 100.0


def _cp_throughput_proxy(cp_or_branch, label: str, monee_net=None) -> float:
    """
    Returns a dimensionless throughput proxy ~ pu of system base.
    Falls back to 1.0 if fields aren't present.
    """
    sn_mva = _system_sn_mva(monee_net) if monee_net is not None else 100.0

    try:
        if label == "CHP":
            ctrl = cp_or_branch.model._control_node
            el_mw = abs(_val(getattr(ctrl, "el_mw", 0.0), 0.0))
            heat_mw = abs(_val(getattr(ctrl, "heat_w", 0.0), 0.0)) / 1e6
            return max((el_mw + heat_mw) / sn_mva, 1e-6)

        if label == "PowerToHeat":
            ctrl = cp_or_branch.model._control_node
            heat_w = _val(getattr(ctrl, "heat_w", None), None)
            if heat_w is not None:
                return max(abs(float(heat_w)) / (1e6 * sn_mva), 1e-6)
            heat_mw = _val(getattr(cp_or_branch.model, "heat_energy_mw", 0.0), 0.0)
            return max(abs(float(heat_mw)) / sn_mva, 1e-6)

        if label == "GasToPower":
            # Use fixed rated capacity (el_mw), not the solved Pyomo Var (p_to_mw=0 when idle)
            p_mw = abs(_val(getattr(cp_or_branch.model, "el_mw", 0.0), 0.0))
            return max(p_mw / sn_mva, 1e-6)

        if label == "PowerToGas":
            # Use fixed rated capacity (gas_kgps), not the solved Pyomo Var (to_mass_flow=0 when idle)
            m_dot = abs(_val(getattr(cp_or_branch.model, "gas_kgps", 0.0), 0.0))
            hhv = _get_gas_hhv(monee_net) if monee_net is not None else 15.3
            # HHV is stored in kWh/kg; power [MW] = m_dot [kg/s] * HHV [kWh/kg] * 3.6 [MJ/kWh]
            return max((m_dot * hhv * 3.6) / sn_mva, 1e-6)

    except Exception:
        pass

    return 1.0


# =============================================================================
# PTDF contexts + stress
# =============================================================================


def _stress_from_ptdf(ptdf: np.ndarray, margins: np.ndarray, cfg: CPMetricConfig):
    denom = margins + cfg.EPS_MARGIN
    s = np.abs(ptdf) / denom
    s = np.clip(s, 0.0, cfg.CLIP_STRESS)
    if s.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(s.mean()),
        float(s.max()),
        float(cfg.AGG_MEAN_WEIGHT * s.mean() + cfg.AGG_MAX_WEIGHT * s.max()),
    )


class CarrierPTDFContext:
    def __init__(self, cfg: CPMetricConfig):
        self.cfg = cfg
        self.power = {}
        self.gas = {}
        self.heat = {}
        self.debug = {}

    # -------- POWER --------
    def power_prebuild(self, monee_net):
        if "built" in self.power:
            return
        branches, branch_ids, Ybus, bus_ids, id_to_local = build_ybus_power(monee_net)
        theta, V, bus_types, bt_dbg = _power_operating_point(monee_net, bus_ids)

        lf = (
            np.array(
                [
                    _power_branch_limit_and_flow(monee_net, bid, self.cfg)
                    for bid in branch_ids
                ],
                dtype=float,
            )
            if branch_ids
            else np.zeros((0, 2))
        )
        limits = lf[:, 0] if lf.size else np.empty(0)
        flows0 = lf[:, 1] if lf.size else np.empty(0)

        margins, finite_mask, binding_mask, pseudo_ratio = _robust_margins(
            limits, flows0, self.cfg
        )

        self.power.update(
            {
                "branches": branches,
                "branch_ids": branch_ids,
                "Ybus": Ybus,
                "bus_ids": bus_ids,
                "id_to_local": id_to_local,
                "theta": theta,
                "V": V,
                "bus_types": bus_types,
                "margins": margins,
                "binding_mask": binding_mask,
                "ptdf_cache": {},
                "built": True,
            }
        )

        self.debug["power"] = {
            "n_lines": len(branch_ids),
            "pv_count": bt_dbg["pv_count"],
            "slack_count": bt_dbg["slack_count"],
            "finite_limit_ratio": float(np.mean(finite_mask))
            if finite_mask.size
            else 0.0,
            "binding_ratio": float(np.mean(binding_mask)) if binding_mask.size else 0.0,
            "pseudo_used_ratio": pseudo_ratio,
            "margin_min": float(np.min(margins)) if margins.size else None,
            "margin_med": float(np.median(margins)) if margins.size else None,
        }

    def power_ptdf_node(self, monee_net, node_id):
        self.power_prebuild(monee_net)
        cache = self.power["ptdf_cache"]
        if node_id in cache:
            return cache[node_id], True

        id_to_local = self.power["id_to_local"]
        if node_id not in id_to_local:
            z = np.zeros(len(self.power["branches"]), dtype=float)
            cache[node_id] = z
            return z, False

        s_local = id_to_local[node_id]
        dP = _distributed_balancing_vector_power(
            monee_net, self.power["bus_ids"], id_to_local, s_local
        )

        dP_lines = ac_ptdf_distributed(
            self.power["theta"],
            self.power["V"],
            self.power["Ybus"],
            self.power["branches"],
            self.power["bus_types"],
            dP,
            dQ=None,
        )
        out = np.array(dP_lines, dtype=float)
        cache[node_id] = out
        return out, True

    # -------- GAS --------
    def gas_prebuild(self, monee_net):
        if "built" in self.gas:
            return
        B, pipe_data, pipe_ids, junc_ids = build_gas_susceptance(monee_net, self.cfg)

        gg = _gas_grid(monee_net)
        lf = (
            np.array(
                [_gas_pipe_limit_and_flow(monee_net, pid, gg) for pid in pipe_ids],
                dtype=float,
            )
            if pipe_ids
            else np.zeros((0, 2))
        )
        limits = lf[:, 0] if lf.size else np.empty(0)
        flows0 = lf[:, 1] if lf.size else np.empty(0)

        margins, finite_mask, binding_mask, pseudo_ratio = _robust_margins(
            limits, flows0, self.cfg
        )

        self.gas.update(
            {
                "B": B,
                "pipe_data": pipe_data,
                "pipe_ids": pipe_ids,
                "node_ids": junc_ids,
                "margins": margins,
                "binding_mask": binding_mask,
                "ptdf_cache": {},
                "built": True,
            }
        )

        self.debug["gas"] = {
            "n_pipes": len(pipe_ids),
            "finite_limit_ratio": float(np.mean(finite_mask))
            if finite_mask.size
            else 0.0,
            "binding_ratio": float(np.mean(binding_mask)) if binding_mask.size else 0.0,
            "pseudo_used_ratio": pseudo_ratio,
            "margin_min": float(np.min(margins)) if margins.size else None,
            "margin_med": float(np.median(margins)) if margins.size else None,
        }

    def gas_ptdf_node(self, monee_net, node_id):
        self.gas_prebuild(monee_net)
        cache = self.gas["ptdf_cache"]
        if node_id in cache:
            return cache[node_id]

        node_ids = self.gas["node_ids"]
        if node_id not in node_ids:
            z = (np.zeros(len(self.gas["pipe_data"]), dtype=float), False)
            cache[node_id] = z
            return z

        s_idx = node_ids.index(node_id)

        # balancing nodes: prefer ExtHydrGrid, else all other nodes
        bal_idxs = []
        try:
            nodes = [monee_net.node_by_id(nid) for nid in node_ids]
            for i, n in enumerate(nodes):
                if monee_net.has_any_child_of_type(n, mm.ExtHydrGrid):
                    bal_idxs.append(i)
        except Exception:
            bal_idxs = []

        if not bal_idxs:
            bal_idxs = [i for i in range(len(node_ids)) if i != s_idx]

        # Restrict balancing to the same connected component as the source node.
        # Cross-component balancing makes each component's RHS sum non-zero → infeasible.
        edges = [(fi, ti) for fi, ti, _ in self.gas["pipe_data"]]
        comps = _connected_components_from_edges(len(node_ids), edges)
        src_comp = next((c for c in comps if s_idx in c), None)
        if src_comp is not None:
            bal_idxs = [i for i in bal_idxs if i in src_comp]
            if not bal_idxs:
                bal_idxs = [i for i in src_comp if i != s_idx]

        # If source IS the slack (ExtHydrGrid), any injection is absorbed locally → zero PTDF (reliable).
        if s_idx in bal_idxs:
            z = (np.zeros(len(self.gas["pipe_data"]), dtype=float), True)
            cache[node_id] = z
            return z

        rhs = _distributed_balancing_rhs(len(node_ids), s_idx, bal_idxs)
        ptdf, reliable = _ptdf_laplacian_distributed(
            self.gas["B"], self.gas["pipe_data"], rhs, self.cfg
        )
        cache[node_id] = (ptdf, reliable)
        return cache[node_id]

    # -------- HEAT (WATER) --------
    def heat_prebuild(self, monee_net):
        if "built" in self.heat:
            return
        # Build heat susceptance using HX_KAPPA scaling
        B, pipe_data, pipe_ids, junc_ids = build_heat_susceptance(monee_net, self.cfg)

        lf = (
            np.array(
                [_heat_pipe_limit_and_flow(monee_net, pid) for pid in pipe_ids],
                dtype=float,
            )
            if pipe_ids
            else np.zeros((0, 2))
        )
        limits = lf[:, 0] if lf.size else np.empty(0)
        flows0 = lf[:, 1] if lf.size else np.empty(0)

        margins, finite_mask, binding_mask, pseudo_ratio = _robust_margins(
            limits, flows0, self.cfg
        )

        self.heat.update(
            {
                "B": B,
                "pipe_data": pipe_data,
                "pipe_ids": pipe_ids,
                "node_ids": junc_ids,
                "margins": margins,
                "binding_mask": binding_mask,
                "ptdf_cache": {},
                "built": True,
            }
        )

        self.debug["heat"] = {
            "n_edges": len(pipe_ids),
            "finite_limit_ratio": float(np.mean(finite_mask))
            if finite_mask.size
            else 0.0,
            "binding_ratio": float(np.mean(binding_mask)) if binding_mask.size else 0.0,
            "pseudo_used_ratio": pseudo_ratio,
            "margin_min": float(np.min(margins)) if margins.size else None,
            "margin_med": float(np.median(margins)) if margins.size else None,
        }

    def heat_ptdf_node(self, monee_net, node_id):
        self.heat_prebuild(monee_net)
        cache = self.heat["ptdf_cache"]
        if node_id in cache:
            return cache[node_id]

        node_ids = self.heat["node_ids"]
        if node_id not in node_ids:
            z = (np.zeros(len(self.heat["pipe_data"]), dtype=float), False)
            cache[node_id] = z
            return z

        s_idx = node_ids.index(node_id)

        bal_idxs = []
        try:
            nodes = [monee_net.node_by_id(nid) for nid in node_ids]
            for i, n in enumerate(nodes):
                if monee_net.has_any_child_of_type(n, mm.ExtHydrGrid):
                    bal_idxs.append(i)
        except Exception:
            bal_idxs = []

        if not bal_idxs:
            bal_idxs = [i for i in range(len(node_ids)) if i != s_idx]

        # Restrict balancing to the same connected component as the source node.
        # Cross-component balancing makes each component's RHS sum non-zero → infeasible.
        edges = [(fi, ti) for fi, ti, _ in self.heat["pipe_data"]]
        comps = _connected_components_from_edges(len(node_ids), edges)
        src_comp = next((c for c in comps if s_idx in c), None)
        if src_comp is not None:
            bal_idxs = [i for i in bal_idxs if i in src_comp]
            if not bal_idxs:
                bal_idxs = [i for i in src_comp if i != s_idx]

        # If source IS the slack (ExtHydrGrid), any injection is absorbed locally → zero PTDF (reliable).
        if s_idx in bal_idxs:
            z = (np.zeros(len(self.heat["pipe_data"]), dtype=float), True)
            cache[node_id] = z
            return z

        rhs = _distributed_balancing_rhs(len(node_ids), s_idx, bal_idxs)
        ptdf, reliable = _ptdf_laplacian_distributed(
            self.heat["B"], self.heat["pipe_data"], rhs, self.cfg
        )
        cache[node_id] = (ptdf, reliable)
        return cache[node_id]


# =============================================================================
# COUPLING POINT DISCOVERY
# =============================================================================


def _compound_connected_nodes(compound):
    result = {}
    for key, nid in compound.connected_to.items():
        if "power" in key:
            result["power"] = nid
        elif "gas" in key:
            result["gas"] = nid
        elif "heat" in key and "return" not in key:
            result["heat"] = nid
    return result


def _carrier_nodes_for_branch(monee_net, branch):
    carrier_nodes = {}
    for nid in (branch.from_node_id, branch.to_node_id):
        try:
            node = monee_net.node_by_id(nid)
            name = node.grid.name if (node.grid is not None) else None
            if name == "power":
                carrier_nodes["power"] = nid
            elif name == "gas":
                carrier_nodes["gas"] = nid
            elif name == "water":
                carrier_nodes["heat"] = nid
        except Exception:
            pass
    return carrier_nodes


# =============================================================================
# ROW helper
# =============================================================================


def _row_from_detail(
    cp_id,
    cp_type,
    p_fail,
    throughput,
    topo_bc,
    topo_factor,
    total_stress,
    score,
    reliable,
    detail,
):
    row = dict(
        cp_id=cp_id,
        cp_type=cp_type,
        p_fail=float(p_fail),
        throughput=float(throughput),
        topo_bc=float(topo_bc),
        topo_factor=float(topo_factor),
        total_stress=float(total_stress),
        score=float(score),
        reliable=bool(reliable),
    )
    for c in ("power", "gas", "heat"):
        d = detail.get(c, {})
        row[f"{c}_node_id"] = d.get("node_id", None)
        row[f"{c}_reliable"] = d.get("reliable", None)
        row[f"{c}_stress_mean"] = d.get("stress_mean", 0.0)
        row[f"{c}_stress_max"] = d.get("stress_max", 0.0)
        row[f"{c}_stress"] = d.get("stress", 0.0)
    return row


# =============================================================================
# MAIN METRIC
# =============================================================================


def mes_cp_metric(monee_net, cfg: CPMetricConfig = CPMetricConfig()):
    fail_prob = cfg.CP_FAIL_PROB or DEFAULT_FAIL_PROB

    # topology
    G_phys, bc_individual, deg, topo_dbg = compute_physical_topology_metrics(monee_net)

    # PTDF contexts
    ctx = CarrierPTDFContext(cfg)

    rows = []

    # ------------------------
    # Compound CPs
    # ------------------------
    for cp_type, label in [(mm.CHP, "CHP"), (mm.PowerToHeat, "PowerToHeat")]:
        for cp in monee_net.compounds_by_type(cp_type):
            p_fail = float(fail_prob.get(cp_type, 0.1))
            connected = _compound_connected_nodes(cp)
            throughput = (
                _cp_throughput_proxy(cp, label, monee_net)
                if cfg.USE_THROUGHPUT_PROXY
                else 1.0
            )

            bc_group = _group_bc(G_phys, cp)
            topo_factor = (1.0 + cfg.TOPO_ALPHA * float(bc_group))

            carrier_detail = {}
            total_stress = 0.0
            reliable_all = True

            # POWER
            if "power" in connected:
                nid = connected["power"]
                ptdf, ok = ctx.power_ptdf_node(monee_net, nid)
                margins = ctx.power["margins"]
                mean_s, max_s, agg_s = _stress_from_ptdf(ptdf, margins, cfg)
                carrier_detail["power"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s,
                )
                total_stress += cfg.W_POWER * agg_s
                reliable_all = reliable_all and bool(ok)

            # GAS
            if "gas" in connected:
                nid = connected["gas"]
                ptdf, ok = ctx.gas_ptdf_node(monee_net, nid)
                margins = ctx.gas.get("margins", np.zeros_like(ptdf))
                if margins.size != ptdf.size:
                    margins = np.zeros_like(ptdf) + cfg.MIN_MARGIN
                mean_s, max_s, agg_s = _stress_from_ptdf(ptdf, margins, cfg)
                carrier_detail["gas"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s,
                )
                total_stress += cfg.W_GAS * agg_s
                reliable_all = reliable_all and bool(ok)

            # HEAT
            if "heat" in connected:
                nid = connected["heat"]
                ptdf, ok = ctx.heat_ptdf_node(monee_net, nid)
                margins = ctx.heat.get("margins", np.zeros_like(ptdf))
                if margins.size != ptdf.size:
                    margins = np.zeros_like(ptdf) + cfg.MIN_MARGIN
                mean_s, max_s, agg_s = _stress_from_ptdf(ptdf, margins, cfg)
                carrier_detail["heat"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s,
                )
                total_stress += cfg.W_HEAT * agg_s
                reliable_all = reliable_all and bool(ok)

            score = p_fail * throughput * total_stress * topo_factor

            rows.append(
                _row_from_detail(
                    cp_id=cp.id,
                    cp_type=label,
                    p_fail=p_fail,
                    throughput=throughput,
                    topo_bc=bc_group,
                    topo_factor=topo_factor,
                    total_stress=total_stress,
                    score=score,
                    reliable=reliable_all,
                    detail=carrier_detail,
                )
            )

    # ------------------------
    # Branch CPs
    # ------------------------
    for cp_type, label in [
        (mm.PowerToGas, "PowerToGas"),
        (mm.GasToPower, "GasToPower"),
    ]:
        for br in monee_net.branches_by_type(cp_type):
            if not br.active:
                continue

            p_fail = float(fail_prob.get(cp_type, 0.1))
            throughput = (
                _cp_throughput_proxy(br, label, monee_net)
                if cfg.USE_THROUGHPUT_PROXY
                else 1.0
            )

            carrier_nodes = _carrier_nodes_for_branch(monee_net, br)
            bc_avg = float(
                np.mean(
                    [
                        bc_individual.get(n, 0.0)
                        for n in (br.from_node_id, br.to_node_id)
                    ]
                )
            )
            topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)

            carrier_detail = {}
            total_stress = 0.0
            reliable_all = True

            if "power" in carrier_nodes:
                nid = carrier_nodes["power"]
                ptdf, ok = ctx.power_ptdf_node(monee_net, nid)
                margins = ctx.power["margins"]
                mean_s, max_s, agg_s = _stress_from_ptdf(ptdf, margins, cfg)
                carrier_detail["power"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s,
                )
                total_stress += cfg.W_POWER * agg_s
                reliable_all = reliable_all and bool(ok)

            if "gas" in carrier_nodes:
                nid = carrier_nodes["gas"]
                ptdf, ok = ctx.gas_ptdf_node(monee_net, nid)
                margins = ctx.gas.get("margins", np.zeros_like(ptdf))
                if margins.size != ptdf.size:
                    margins = np.zeros_like(ptdf) + cfg.MIN_MARGIN
                mean_s, max_s, agg_s = _stress_from_ptdf(ptdf, margins, cfg)
                carrier_detail["gas"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s,
                )
                total_stress += cfg.W_GAS * agg_s
                reliable_all = reliable_all and bool(ok)

            if "heat" in carrier_nodes:
                nid = carrier_nodes["heat"]
                ptdf, ok = ctx.heat_ptdf_node(monee_net, nid)
                margins = ctx.heat.get("margins", np.zeros_like(ptdf))
                if margins.size != ptdf.size:
                    margins = np.zeros_like(ptdf) + cfg.MIN_MARGIN
                mean_s, max_s, agg_s = _stress_from_ptdf(ptdf, margins, cfg)
                carrier_detail["heat"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s,
                )
                total_stress += cfg.W_HEAT * agg_s
                reliable_all = reliable_all and bool(ok)

            score = p_fail * throughput * total_stress * topo_factor

            rows.append(
                _row_from_detail(
                    cp_id=f"{br.from_node_id}→{br.to_node_id}",
                    cp_type=label,
                    p_fail=p_fail,
                    throughput=throughput,
                    topo_bc=bc_avg,
                    topo_factor=topo_factor,
                    total_stress=total_stress,
                    score=score,
                    reliable=reliable_all,
                    detail=carrier_detail,
                )
            )

    df_scores = (
        pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    )

    if not cfg.RETURN_DEBUG:
        return df_scores

    # Build debug dataframe
    ctx.power_prebuild(monee_net)
    ctx.gas_prebuild(monee_net)
    ctx.heat_prebuild(monee_net)

    debug_rows = []
    for carrier in ("power", "gas", "heat"):
        d = ctx.debug.get(carrier, {})
        debug_rows.append({"carrier": carrier, **d})
    debug_rows.append({"carrier": "topology", **topo_dbg})

    df_debug = pd.DataFrame(debug_rows)
    return df_scores, df_debug


# =============================================================================
# ALL-COMPONENTS METRIC
# =============================================================================


# Default failure probabilities for non-CP branch types
DEFAULT_BRANCH_FAIL_PROB = {
    mm.GenericPowerBranch: 0.05,
    mm.GasPipe: 0.05,
    mm.WaterPipe: 0.05,
    mm.HeatExchanger: 0.05,
}


def _branch_lodf_stress(
    monee_net, ctx: CarrierPTDFContext, carrier: str, from_node_id, to_node_id, cfg: CPMetricConfig
):
    """
    Approximate LODF stress for a branch removal using ptdf_from - ptdf_to.
    Returns (mean_s, max_s, agg_s, reliable).
    """
    if carrier == "power":
        ptdf_from, ok_f = ctx.power_ptdf_node(monee_net, from_node_id)
        ptdf_to, ok_t = ctx.power_ptdf_node(monee_net, to_node_id)
        margins = ctx.power["margins"]
    elif carrier == "gas":
        ptdf_from, ok_f = ctx.gas_ptdf_node(monee_net, from_node_id)
        ptdf_to, ok_t = ctx.gas_ptdf_node(monee_net, to_node_id)
        margins = ctx.gas.get("margins", np.ones(len(ptdf_from)) * cfg.MIN_MARGIN)
    elif carrier == "heat":
        ptdf_from, ok_f = ctx.heat_ptdf_node(monee_net, from_node_id)
        ptdf_to, ok_t = ctx.heat_ptdf_node(monee_net, to_node_id)
        margins = ctx.heat.get("margins", np.ones(len(ptdf_from)) * cfg.MIN_MARGIN)
    else:
        return 0.0, 0.0, 0.0, False

    if ptdf_from.size == 0:
        return 0.0, 0.0, 0.0, True

    if margins.size != ptdf_from.size:
        margins = np.ones(ptdf_from.size) * cfg.MIN_MARGIN

    ptdf_diff = ptdf_from - ptdf_to
    mean_s, max_s, agg_s = _stress_from_ptdf(ptdf_diff, margins, cfg)
    return mean_s, max_s, agg_s, bool(ok_f and ok_t)


def mes_all_components_metric(monee_net, cfg: CPMetricConfig = CPMetricConfig()):
    """
    Score ALL active grid branches (CP and non-CP) using the LODF approximation.

    CPs (CHP, PowerToHeat, GasToPower, PowerToGas) are scored as in
    mes_cp_metric_bulletproof().  Non-CP branches (PowerLine/GenericPowerBranch,
    GasPipe, WaterPipe, HeatExchanger) use the PTDF difference approximation
    ptdf_from - ptdf_to as a proxy for the line-outage distribution factor.

    Returns
    -------
    df_all : pd.DataFrame
        One row per component, sorted by score descending.
        Columns: cp_id, cp_type, is_cp, p_fail, throughput, topo_bc, topo_factor,
                 total_stress, score, reliable, {power,gas,heat}_{node_id,stress,...}
    df_debug : pd.DataFrame  (only if cfg.RETURN_DEBUG)
    """
    # Reuse existing CP scores
    if cfg.RETURN_DEBUG:
        df_cp, df_debug = mes_cp_metric(monee_net, cfg)
    else:
        df_cp = mes_cp_metric(monee_net, cfg)
        df_debug = None

    df_cp = df_cp.copy()
    df_cp["is_cp"] = True

    # Build shared context (already built internally by mes_cp_metric_bulletproof,
    # but we need a fresh one here so we can access it).
    G_phys, bc_individual, _deg, _topo_dbg = compute_physical_topology_metrics(monee_net)
    ctx = CarrierPTDFContext(cfg)
    ctx.power_prebuild(monee_net)
    ctx.gas_prebuild(monee_net)
    ctx.heat_prebuild(monee_net)

    sn_mva = _system_sn_mva(monee_net)

    # CP branch ids to skip (already covered above)
    cp_branch_ids = set()
    for cp_type in (mm.GasToPower, mm.PowerToGas):
        for br in monee_net.branches_by_type(cp_type):
            cp_branch_ids.add(br.id)

    fail_prob_branch = DEFAULT_BRANCH_FAIL_PROB

    rows_non_cp = []

    # ---- Power branches (GenericPowerBranch) ----
    for branch_id in ctx.power["branch_ids"]:
        if branch_id in cp_branch_ids:
            continue
        try:
            br = monee_net.branch_by_id(branch_id)
        except Exception:
            continue

        p_fail = float(fail_prob_branch.get(mm.GenericPowerBranch, 0.05))
        limit, flow0 = _power_branch_limit_and_flow(monee_net, branch_id, cfg)
        throughput = max(float(flow0) / sn_mva, 1e-6) if cfg.USE_THROUGHPUT_PROXY else 1.0

        bc_avg = float(np.mean([
            bc_individual.get(br.from_node_id, 0.0),
            bc_individual.get(br.to_node_id, 0.0),
        ]))
        topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)

        mean_s, max_s, agg_s, reliable = _branch_lodf_stress(
            monee_net, ctx, "power", br.from_node_id, br.to_node_id, cfg
        )
        total_stress = cfg.W_POWER * agg_s
        score = p_fail * throughput * total_stress * topo_factor

        row = dict(
            cp_id=str(branch_id),
            cp_type="PowerLine",
            is_cp=False,
            p_fail=p_fail,
            throughput=throughput,
            topo_bc=bc_avg,
            topo_factor=topo_factor,
            total_stress=total_stress,
            score=score,
            reliable=reliable,
            power_node_id=br.from_node_id,
            power_reliable=reliable,
            power_stress_mean=mean_s,
            power_stress_max=max_s,
            power_stress=agg_s,
            gas_node_id=None, gas_reliable=None,
            gas_stress_mean=0.0, gas_stress_max=0.0, gas_stress=0.0,
            heat_node_id=None, heat_reliable=None,
            heat_stress_mean=0.0, heat_stress_max=0.0, heat_stress=0.0,
        )
        rows_non_cp.append(row)

    # ---- Gas pipes ----
    for pipe_id in ctx.gas["pipe_ids"]:
        if pipe_id in cp_branch_ids:
            continue
        try:
            br = monee_net.branch_by_id(pipe_id)
        except Exception:
            continue

        p_fail = float(fail_prob_branch.get(mm.GasPipe, 0.05))
        gg = _gas_grid(monee_net)
        _limit, flow0 = _gas_pipe_limit_and_flow(monee_net, pipe_id, gg)
        throughput = max(float(flow0) / sn_mva, 1e-6) if cfg.USE_THROUGHPUT_PROXY else 1.0

        bc_avg = float(np.mean([
            bc_individual.get(br.from_node_id, 0.0),
            bc_individual.get(br.to_node_id, 0.0),
        ]))
        topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)

        mean_s, max_s, agg_s, reliable = _branch_lodf_stress(
            monee_net, ctx, "gas", br.from_node_id, br.to_node_id, cfg
        )
        total_stress = cfg.W_GAS * agg_s
        score = p_fail * throughput * total_stress * topo_factor

        row = dict(
            cp_id=str(pipe_id),
            cp_type="GasPipe",
            is_cp=False,
            p_fail=p_fail,
            throughput=throughput,
            topo_bc=bc_avg,
            topo_factor=topo_factor,
            total_stress=total_stress,
            score=score,
            reliable=reliable,
            power_node_id=None, power_reliable=None,
            power_stress_mean=0.0, power_stress_max=0.0, power_stress=0.0,
            gas_node_id=br.from_node_id,
            gas_reliable=reliable,
            gas_stress_mean=mean_s,
            gas_stress_max=max_s,
            gas_stress=agg_s,
            heat_node_id=None, heat_reliable=None,
            heat_stress_mean=0.0, heat_stress_max=0.0, heat_stress=0.0,
        )
        rows_non_cp.append(row)

    # ---- Heat pipes (WaterPipe + HeatExchanger) ----
    for pipe_id in ctx.heat["pipe_ids"]:
        if pipe_id in cp_branch_ids:
            continue
        try:
            br = monee_net.branch_by_id(pipe_id)
        except Exception:
            continue

        bm = br.model
        cp_type_str = "HeatExchanger" if isinstance(bm, mm.HeatExchanger) else "WaterPipe"
        fail_key = mm.HeatExchanger if isinstance(bm, mm.HeatExchanger) else mm.WaterPipe
        p_fail = float(fail_prob_branch.get(fail_key, 0.05))

        _limit, flow0 = _heat_pipe_limit_and_flow(monee_net, pipe_id)
        throughput = max(float(flow0) / sn_mva, 1e-6) if cfg.USE_THROUGHPUT_PROXY else 1.0

        bc_avg = float(np.mean([
            bc_individual.get(br.from_node_id, 0.0),
            bc_individual.get(br.to_node_id, 0.0),
        ]))
        topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)

        mean_s, max_s, agg_s, reliable = _branch_lodf_stress(
            monee_net, ctx, "heat", br.from_node_id, br.to_node_id, cfg
        )
        total_stress = cfg.W_HEAT * agg_s
        score = p_fail * throughput * total_stress * topo_factor

        row = dict(
            cp_id=str(pipe_id),
            cp_type=cp_type_str,
            is_cp=False,
            p_fail=p_fail,
            throughput=throughput,
            topo_bc=bc_avg,
            topo_factor=topo_factor,
            total_stress=total_stress,
            score=score,
            reliable=reliable,
            power_node_id=None, power_reliable=None,
            power_stress_mean=0.0, power_stress_max=0.0, power_stress=0.0,
            gas_node_id=None, gas_reliable=None,
            gas_stress_mean=0.0, gas_stress_max=0.0, gas_stress=0.0,
            heat_node_id=br.from_node_id,
            heat_reliable=reliable,
            heat_stress_mean=mean_s,
            heat_stress_max=max_s,
            heat_stress=agg_s,
        )
        rows_non_cp.append(row)

    df_non_cp = pd.DataFrame(rows_non_cp) if rows_non_cp else pd.DataFrame(
        columns=list(df_cp.columns) + ["is_cp"]
    )

    # Ensure df_cp has same columns as df_non_cp (add is_cp already done above)
    df_all = pd.concat([df_cp, df_non_cp], ignore_index=True, sort=False)
    df_all = df_all.sort_values("score", ascending=False).reset_index(drop=True)

    # ---- Local metric ----
    # Uses only information locally available to the device (1-hop at most):
    #   p_fail             – device failure probability (device parameter)
    #   loading            – |flow| / limit: own utilisation (local measurement)
    #                        For CPs without a meaningful limit: rated throughput proxy.
    #   n_critical_nbrs    – neighbours with degree ≤ 2 (1-hop info only):
    #                        counts neighbours for whom this component is on their
    #                        only or critical path; computable without global routing.
    #   carrier_coupling   – number of distinct energy carriers this component connects
    #                        (observable from own port configuration, no network traversal).
    #
    # local_score = p_fail × loading × (1 + n_critical_nbrs) × carrier_coupling
    #
    # Rationale:
    #   loading           → how much energy actually flows through me right now
    #   n_critical_nbrs   → how many neighbours are stranded if I fail
    #   carrier_coupling  → how many domains my failure disrupts simultaneously

    # Pre-build graph adjacency for 1-hop neighbour degree lookups
    G_local = nx.Graph(monee_net._network_internal)

    def _connected_node_ids(row):
        """All network node IDs directly connected to this component."""
        nids = []
        for col in ("power_node_id", "gas_node_id", "heat_node_id"):
            nid = row.get(col)
            if nid is not None:
                nids.append(nid)
        if not nids:
            cp_type = row.get("cp_type", "")
            cp_id = row.get("cp_id")
            if cp_type in ("CHP", "PowerToHeat"):
                cp_cls = mm.CHP if cp_type == "CHP" else mm.PowerToHeat
                for cp in monee_net.compounds_by_type(cp_cls):
                    if cp.id == cp_id:
                        nids = list(cp.connected_to.values())
                        break
        return nids

    def _n_critical_nbrs(row):
        """Count of 1-hop neighbours whose degree is ≤ 2 (critically dependent)."""
        nids = [n for n in _connected_node_ids(row)
                if n is not None and n == n and G_local.has_node(n)]
        count = 0
        for nid in nids:
            for nbr in G_local.neighbors(nid):
                if nbr != nid and _deg.get(nbr, 1) <= 2:
                    count += 1
        return float(count)

    def _carrier_coupling(row):
        """Number of distinct energy carriers this component connects (1–3)."""
        cp_type = row.get("cp_type", "")
        if cp_type in ("CHP",):
            return 2.0  # electricity + heat
        if cp_type in ("PowerToHeat",):
            return 2.0  # electricity + heat
        if cp_type in ("GasToPower", "PowerToGas"):
            return 2.0  # gas + electricity
        # Single-carrier branches
        return 1.0

    def _loading_local(row):
        """
        |flow| / limit — own utilisation ratio (locally measurable).
        For CPs, falls back to the rated throughput proxy (no meaningful flow limit).
        """
        cp_type = row.get("cp_type", "")
        cp_id = row.get("cp_id")
        throughput = float(row.get("throughput", 1.0))

        if cp_type in ("CHP", "PowerToHeat", "GasToPower", "PowerToGas"):
            return throughput  # rated-capacity proxy; no global state needed

        try:
            import ast
            tup = ast.literal_eval(str(cp_id))
            if cp_type == "PowerLine":
                limit, flow0 = _power_branch_limit_and_flow(monee_net, tup, cfg)
            elif cp_type == "GasPipe":
                gg = _gas_grid(monee_net)
                limit, flow0 = _gas_pipe_limit_and_flow(monee_net, tup, gg)
            else:  # WaterPipe, HeatExchanger
                limit, flow0 = _heat_pipe_limit_and_flow(monee_net, tup)
            if np.isfinite(limit) and limit > 0:
                return float(flow0) / float(limit)
        except Exception:
            pass
        return throughput

    df_all["loading"] = df_all.apply(_loading_local, axis=1)
    df_all["n_critical_nbrs"] = df_all.apply(_n_critical_nbrs, axis=1)
    df_all["carrier_coupling"] = df_all.apply(_carrier_coupling, axis=1)
    df_all["local_score"] = (
        df_all["p_fail"]
        * df_all["loading"]
        * (1.0 + df_all["n_critical_nbrs"])
        * df_all["carrier_coupling"]
    )

    # ---- Self-only metric (zero-hop) ----
    # Uses exclusively the component's own observable state — no network traversal:
    #   p_fail           – device failure probability
    #   loading          – own utilisation |flow| / limit
    #   carrier_coupling – number of carriers connected (readable from own ports)
    # self_score = p_fail × loading × carrier_coupling
    df_all["self_score"] = (
        df_all["p_fail"]
        * df_all["loading"]
        * df_all["carrier_coupling"]
    )

    # ---- Katz centrality (physical graph) ----
    # Katz uses weights as adjacency strengths (A_ij), not distances.
    # Physical "weight" values are resistances (high = harder to traverse),
    # so we invert them to get conductances (high = strong coupling),
    # making the semantics consistent with betweenness centrality.
    G_katz = nx.Graph(G_phys)
    for u, v, data in G_katz.edges(data=True):
        w = data.get("weight", 1.0)
        G_katz.edges[u, v]["katz_weight"] = 1.0 / max(float(w), 1e-12)
    try:
        eigenvalues = np.real(np.linalg.eigvals(nx.to_numpy_array(G_katz, weight="katz_weight")))
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        katz_alpha = 0.85 / spectral_radius if spectral_radius > 0 else 0.1
        katz_individual = nx.katz_centrality(
            G_katz, alpha=katz_alpha, weight="katz_weight", normalized=True
        )
    except Exception:
        katz_individual = {n: 0.0 for n in G_phys.nodes()}

    def _katz_for_row(row):
        cp_type = row.get("cp_type", "")
        cp_id = row.get("cp_id")

        if cp_type in ("CHP", "PowerToHeat"):
            cp_cls = mm.CHP if cp_type == "CHP" else mm.PowerToHeat
            for cp in monee_net.compounds_by_type(cp_cls):
                if cp.id == cp_id:
                    vals = [katz_individual.get(nid, 0.0)
                            for nid in cp.connected_to.values()]
                    return float(np.mean(vals)) if vals else 0.0
            return 0.0

        node_ids = [row.get(col) for col in ("power_node_id", "gas_node_id", "heat_node_id")]
        node_ids = [n for n in node_ids if n is not None and n == n]
        if node_ids:
            return float(np.mean([katz_individual.get(n, 0.0) for n in node_ids]))

        # Branch CPs: cp_id = "from→to"
        try:
            from_id, to_id = str(cp_id).split("→")
            return float(np.mean([
                katz_individual.get(from_id.strip(), 0.0),
                katz_individual.get(to_id.strip(), 0.0),
            ]))
        except Exception:
            return 0.0

    df_all["katz_score"] = df_all.apply(_katz_for_row, axis=1)

    # ---- Closeness vitality (physical graph, same weights as BC and Katz) ----
    # vitality(v) = W(G) - W(G \ v): how much total pairwise distance increases
    # when v is removed. Captures structural indispensability, not just centrality.
    try:
        vitality_individual = nx.closeness_vitality(G_phys, weight="weight")
        # Replace inf (node removal disconnects graph → infinite path distances)
        # with the maximum finite vitality, treating disconnectors as maximally critical.
        finite_vals = [v for v in vitality_individual.values()
                       if np.isfinite(v)]
        max_finite = max(finite_vals) if finite_vals else 0.0
        vitality_individual = {
            n: (max_finite if not np.isfinite(v) else v)
            for n, v in vitality_individual.items()
        }
        # Shift so minimum is 0
        min_v = min(vitality_individual.values()) if vitality_individual else 0.0
        if min_v < 0:
            vitality_individual = {n: v - min_v for n, v in vitality_individual.items()}
    except Exception:
        vitality_individual = {n: 0.0 for n in G_phys.nodes()}

    def _vitality_for_row(row):
        cp_type = row.get("cp_type", "")
        cp_id = row.get("cp_id")

        if cp_type in ("CHP", "PowerToHeat"):
            cp_cls = mm.CHP if cp_type == "CHP" else mm.PowerToHeat
            for cp in monee_net.compounds_by_type(cp_cls):
                if cp.id == cp_id:
                    vals = [vitality_individual.get(nid, 0.0)
                            for nid in cp.connected_to.values()]
                    return float(np.mean(vals)) if vals else 0.0
            return 0.0

        node_ids = [row.get(col) for col in ("power_node_id", "gas_node_id", "heat_node_id")]
        node_ids = [n for n in node_ids if n is not None and n == n]
        if node_ids:
            return float(np.mean([vitality_individual.get(n, 0.0) for n in node_ids]))

        try:
            from_id, to_id = str(cp_id).split("→")
            return float(np.mean([
                vitality_individual.get(from_id.strip(), 0.0),
                vitality_individual.get(to_id.strip(), 0.0),
            ]))
        except Exception:
            return 0.0

    df_all["vitality_score"] = df_all.apply(_vitality_for_row, axis=1)

    # ---- Stress-weighted topology ----
    # BC where edge weight = 1/(loading+eps), so stressed branches are "shorter".
    # Compounds use group_bc; branches/CPs use average endpoint BC.
    try:
        _G_stress, stress_bc_nodes, stress_topo_dbg = compute_stress_topology_metrics(
            monee_net, ctx, cfg
        )

        def _stress_bc_for_row(row):
            cp_type = row.get("cp_type", "")
            cp_id = row.get("cp_id")

            if cp_type in ("CHP", "PowerToHeat"):
                cp_cls = mm.CHP if cp_type == "CHP" else mm.PowerToHeat
                for cp in monee_net.compounds_by_type(cp_cls):
                    if cp.id == cp_id:
                        return float(_group_bc(_G_stress, cp, weight="stress_weight"))
                return 0.0

            # For all branch types, average the endpoint node BCs
            node_ids = []
            for col in ("power_node_id", "gas_node_id", "heat_node_id"):
                nid = row.get(col)
                if nid is not None:
                    node_ids.append(stress_bc_nodes.get(nid, 0.0))
            if node_ids:
                return float(np.mean(node_ids))

            # Fallback: try to parse cp_id as edge tuple (non-CP branches)
            try:
                import ast
                tup = ast.literal_eval(str(cp_id))
                br = monee_net.branch_by_id(tup)
                return float(np.mean([
                    stress_bc_nodes.get(br.from_node_id, 0.0),
                    stress_bc_nodes.get(br.to_node_id, 0.0),
                ]))
            except Exception:
                return 0.0

        df_all["stress_bc"] = df_all.apply(_stress_bc_for_row, axis=1)
        df_all["stress_topo_factor"] = 1.0 + cfg.TOPO_ALPHA * df_all["stress_bc"]
        df_all["stress_score"] = df_all["score"] / df_all["topo_factor"].replace(0, np.nan) * df_all["stress_topo_factor"]
    except Exception as e:
        df_all["stress_bc"] = 0.0
        df_all["stress_topo_factor"] = 1.0
        df_all["stress_score"] = df_all["score"]
        print(f"[warn] stress topology failed: {e}")

    if cfg.RETURN_DEBUG:
        return df_all, df_debug
    return df_all


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    from monee.network import create_balanced_urban_mes_net
    net = create_balanced_urban_mes_net()
    result = monee.run_energy_flow(net, solver=monee.PyomoSolver())
    print(result)
    solved = result.network

    print(
        "\n=== MES CP Criticality (Stress-based, distributed balancing) ==="
    )
    df_scores, df_debug = mes_cp_metric(solved, cfg=CPMetricConfig())

    print("\n--- Scores ---")
    print(df_scores.to_string(index=False))

    print("\n--- Debug / Sanity Report ---")
    print(df_debug.to_string(index=False))
