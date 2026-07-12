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
from monee.model.core import Node as MNode
from monee.model.grid import DEFAULT_GAS_HHV_KWH_PER_KG

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

    # Susceptance build (gas/heat): keep low-flow pipes in the Laplacian instead
    # of dropping them at FLOW_MIN — dropping fragments the graph and produces
    # spurious unreliability flags. The flow floor sets a small absolute m0
    # (kg/s) below which susceptance is computed as if m0 = floor; the cap
    # bounds any single edge's b at K × median(b_finite) to avoid one stiff
    # edge dominating the conditioning of B.
    SUSCEPTANCE_FLOW_FLOOR_KGPS: float = 1e-6
    SUSCEPTANCE_B_RELATIVE_CAP: float = 100.0

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

    # ── Heat / thermal-aware metric ──────────────────────────────────────
    # Heat pipes use the same hydraulic capacity − flow margins as gas
    # (velocity-cap limit via _robust_margins). The earlier "thermal
    # margin" override (m_solved − q_downstream/(cp·ΔT)) was removed: it
    # *increased* with loading — inverted vs. every other carrier — and
    # because monee's HeatExchanger sizes flows at the same 30 K design
    # spread, the two terms nearly cancelled and demand-carrying supply
    # pipes collapsed onto the MIN_MARGIN plateau (heat ranking = noise).
    HEAT_DELTA_T_K: float = 30.0           # supply-return spread (K)
    HEAT_CP_J_PER_KG_K: float = 4186.0     # specific heat of water

    # (B) Slack-distance prefactor for heat: scales heat-stress contribution by
    #     (1 + α · d_thermal) where d_thermal = Σ L·κ on the shortest path
    #     (per-pipe thermal-loss factor) from the component to the nearest
    #     heat injector (ExtHydrGrid + heat-supplying CPs), normalised to
    #     [0, 1] by the farthest node so the multiplier is bounded to
    #     [1, 1+α].
    HEAT_REMOTENESS_ENABLE: bool = True
    HEAT_REMOTENESS_ALPHA: float = 1.0
    HEAT_PIPE_U_W_M2_K: float = 1.5        # external heat-transfer coeff for DH pipe

    # ── Exergy-aware CP edge weights ─────────────────────────────────────
    # Used by ``compute_physical_topology_metrics_exergy_aware``. Carrier
    # quality factors q_k follow Szargut for chemical fuels and the Carnot
    # factor 1 − T0/Ts for thermal carriers (see
    # ``docs/new_edge_weight_theory.tex`` for the full derivation).
    EXERGY_T_AMBIENT_K: float = 293.0
    EXERGY_T_HEAT_SUPPLY_K: float = 363.0   # 90°C — typical district heat
    EXERGY_Q_EL: float = 1.0                # electricity is pure exergy
    EXERGY_Q_GAS: float = 1.0               # cap Szargut β at 1.0 for BC use

    # ── Ablation flags (E2) ──────────────────────────────────────────────
    # Each ABLATE_* fixes one factor of the composite score to 1.0 so the
    # effect of removing it can be measured. The CMRES E2 ablation
    # experiment runs `mes_cp_metric` once per ablation variant and compares
    # ρ vs the full score; effect size is then ρ(full) − ρ(ablated).
    ABLATE_THROUGHPUT: bool = False
    ABLATE_STRESS: bool = False
    ABLATE_TOPO: bool = False
    ABLATE_ADEQUACY: bool = False  # only affects CP rows; non-CPs are unchanged

    # ── CP input-adequacy multiplier ─────────────────────────────────────
    # A CP only delivers into its output sector(s) when its input sector can
    # supply it. We extend CP criticality with a structural conditional term:
    #   score(c) ← score(c) × P(input adequate at c)
    # where P(input adequate) is computed analytically from base failure
    # probabilities along the most-likely path from c's input node to the
    # nearest input-sector slack (ExtPowerGrid for power inputs, ExtHydrGrid
    # for gas inputs). Edge weights are −log(1 − p_fail), so the shortest-path
    # distance d gives P = exp(−d). Exact for radial input grids; a
    # pessimistic lower bound on adequacy when meshed.
    CP_INPUT_ADEQUACY_ENABLE: bool = True

    # Diagnostics
    RETURN_DEBUG: bool = True


DEFAULT_FAIL_PROB = {
    mm.CHP: 0.1,
    mm.CHPHG: 0.1,
    mm.PowerToHeat: 0.1,
    mm.PowerToHeatHG: 0.1,
    mm.GasToHeatHG: 0.1,
    mm.PowerToGas: 0.1,
    mm.GasToPower: 0.1,
}

# CP type labels — kept here so multiple call sites stay in sync. Compound CPs
# instantiate a multi-grid control node + sub-children; branch CPs are
# 2-endpoint multi-grid branches.
COMPOUND_CP_LABELS = ("CHP", "CHPHG", "PowerToHeat")
BRANCH_CP_LABELS = ("PowerToGas", "GasToPower", "PowerToHeatHG", "GasToHeatHG")
ALL_CP_LABELS = COMPOUND_CP_LABELS + BRANCH_CP_LABELS

_COMPOUND_CP_CLASSES = {
    "CHP": mm.CHP,
    "CHPHG": mm.CHPHG,
    "PowerToHeat": mm.PowerToHeat,
}

# Input-side carrier per CP type. Used by the analytical input-adequacy
# multiplier: the CP's criticality is conditioned on its input sector being
# able to feed it.
CP_INPUT_CARRIER = {
    "CHP": "gas",
    "CHPHG": "gas",
    "GasToPower": "gas",
    "GasToHeatHG": "gas",
    "PowerToGas": "power",
    "PowerToHeat": "power",
    "PowerToHeatHG": "power",
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


def _is_finite(v) -> bool:
    """True iff v is a real, finite scalar (not None, NaN, or ±inf)."""
    if v is None:
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _first_attr(obj, names: List[str], default=None):
    """Return the first attribute on *obj* whose value is finite.

    NaN/inf are now treated as missing — solved Pyomo variables can be NaN
    (e.g. unbounded duals after a relaxed solve) and feeding those into
    margins/PTDF cascades non-finite values through every metric.
    """
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            vv = _val(v, default=None)
            if _is_finite(vv):
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
            # One pseudo capacity for the whole group: (1+headroom) × P90 of
            # observed flows (robust scale so it isn't zero). The former
            # per-branch base max(flow, scale) made margins GROW with flow
            # above the 90th percentile — margin 0.3·flow — so the most
            # heavily loaded limit-less branches (all HeatExchangers) scored
            # the LOWEST stress. With a constant limit, margin = limit − flow
            # is monotone decreasing; flows beyond the pseudo capacity clamp
            # to MIN_MARGIN and are flagged binding.
            scale = np.percentile(flows0[flows0 > 0], 90) if np.any(flows0 > 0) else 1.0
            pseudo_limit = (1.0 + cfg.PSEUDO_HEADROOM) * max(scale, cfg.MIN_MARGIN)
            raw_margins[missing] = pseudo_limit - flows0[missing]

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
        br_r = _val(b.model.br_r_pu, 0.0)
        br_x = _val(b.model.br_x_pu, 0.0)
        # Avoid zero impedance (bus-bar / short-circuit branches)
        if abs(br_r) < 1e-9 and abs(br_x) < 1e-9:
            br_x = 1e-6
        branch_tuples.append(
            (
                id_to_local[b.from_node_id],
                id_to_local[b.to_node_id],
                br_r,
                br_x,
                _val(b.model.g_fr_pu, 0.0),
                _val(b.model.b_fr_pu, 0.0),
                _val(b.model.g_to_pu, 0.0),
                _val(b.model.b_to_pu, 0.0),
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
    +1 at source bus, distributed -1 across ExtPowerGrid buses.

    Returns ``(dP_array, ok)``: ok is False when no ExtPowerGrid bus exists
    (or only the source itself is a slack and thus excluded). Callers should
    flag the resulting PTDF unreliable rather than fall back to spreading the
    balance across non-slack buses, which has no physical analog.
    """
    bus_map = {n.id: n for n in monee_net.nodes_by_type(mm.Bus)}

    slack_locs = []
    for i, nid in enumerate(bus_ids):
        if monee_net.has_any_child_of_type(bus_map[nid], mm.ExtPowerGrid):
            slack_locs.append(i)

    nb = len(bus_ids)
    dP = np.zeros(nb, dtype=float)

    bal = [i for i in slack_locs if i != s_local]
    if not bal:
        return jnp.array(dP), False

    dP[s_local] += 1.0
    w = np.ones(len(bal), dtype=float)
    w = w / w.sum()
    for bi, wi in zip(bal, w):
        dP[bi] -= wi
    return jnp.array(dP), True


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


def _calc_C_squared(diameter_m, length_m, t_k, compressibility,
                    r_specific=_R_SPECIFIC_GAS):
    # Friction-free Weymouth constant C² = π²D⁵ / (16 L R T Z), matching the
    # monee simulator (monee.model.phys.nonlinear.gf.calc_C_squared). The
    # per-pipe friction factor λ is applied separately at the susceptance and
    # capacity call sites, so screening stays consistent with the simulator.
    return (math.pi**2 * diameter_m**5) / (
        16.0 * length_m * r_specific * t_k * compressibility
    )


def _gas_friction(pipe_model, default: float = 0.02) -> float:
    """Per-pipe Darcy friction factor at the operating point, matching the
    monee GasPipe ``friction`` Var. Falls back to the monee default (0.02)
    when the value is missing or non-finite (e.g. an unsolved network)."""
    f = _val(getattr(pipe_model, "friction", None), default)
    return float(f) if (_is_finite(f) and float(f) > 0) else float(default)


def _gas_r_specific(gas_grid, default: float = _R_SPECIFIC_GAS) -> float:
    """Specific gas constant R = R_universal / M, derived from the grid to
    match the monee gas formulations (which pass
    ``grid.universal_gas_constant / grid.molar_mass``). Falls back to the
    module default when the grid or its fields are unavailable."""
    try:
        R = float(getattr(gas_grid, "universal_gas_constant"))
        M = float(getattr(gas_grid, "molar_mass"))
        if _is_finite(R) and _is_finite(M) and M > 0:
            return R / M
    except Exception:
        pass
    return float(default)


def _darcy_resistance(pipe_model):
    d = _val(getattr(pipe_model, "diameter_m", None), None)
    L = _val(getattr(pipe_model, "length_m", None), None)
    # Reject non-finite geometry — solved Pyomo Vars can resolve to NaN, and
    # NaN comparisons silently pass `d == 0` then poison every downstream b.
    if not (_is_finite(d) and _is_finite(L)) or d == 0:
        return None
    A = math.pi * d**2 / 4.0
    friction = getattr(pipe_model, "friction", None)
    f_raw = _val(friction, 0.02)
    f = max(float(f_raw) if _is_finite(f_raw) else 0.02, 1e-6)
    Rm = f * (L / d) / (2.0 * _WATER_DENSITY * A**2)
    return Rm if _is_finite(Rm) and Rm > 0 else None


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
        if gg is not None and hasattr(gg, "higher_heating_value_kwh_per_kg"):
            return float(gg.higher_heating_value_kwh_per_kg)
    except Exception:
        pass
    return DEFAULT_GAS_HHV_KWH_PER_KG


def _gas_pipe_max_flow(pipe_model, gas_grid) -> float:
    """
    Weymouth capacity proxy in kg/s using grid p^2 bounds and reference pressure.
    """
    d = _val(getattr(pipe_model, "diameter_m", None), None)
    L = _val(getattr(pipe_model, "length_m", None), None)
    if not d or not L or d <= 0 or L <= 0:
        return np.inf

    t_k = float(getattr(gas_grid, "t_k", 300.0))
    z = float(getattr(gas_grid, "compressibility", 1.0))
    C2 = _calc_C_squared(d, L, t_k, z, _gas_r_specific(gas_grid)) / _gas_friction(pipe_model)

    p_sq_max = float(getattr(gas_grid, "pressure_squared_pu_max", 1.3))
    p_sq_min = float(getattr(gas_grid, "pressure_squared_pu_min", 0.7))
    p_ref = float(getattr(gas_grid, "pressure_ref_pa", 1e6))

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
        float(getattr(gas_grid, "t_k", 300.0))
        if gas_grid is not None
        else 300.0
    )
    z = (
        float(getattr(gas_grid, "compressibility", 1.0))
        if gas_grid is not None
        else 1.0
    )

    raw_pipes: List[Tuple[int, int, float, int]] = []  # (fi, ti, b_raw, pipe_id)

    for pipe in monee_net.branches_by_type(mm.GasPipe):
        if not (pipe.active and int(_val(pipe.model.on_off, 1)) == 1):
            continue
        fi = idx.get(pipe.from_node_id)
        ti = idx.get(pipe.to_node_id)
        if fi is None or ti is None:
            continue

        m0_raw = _val(pipe.model.mass_flow_kgs, 0.0)
        m0 = abs(float(m0_raw)) if _is_finite(m0_raw) else 0.0
        # Floor m0 to keep low-flow pipes in B; b will be capped post-hoc.
        m0_eff = max(m0, cfg.SUSCEPTANCE_FLOW_FLOOR_KGPS)

        d_raw = _val(pipe.model.diameter_m, 0.0)
        L_raw = _val(pipe.model.length_m, 1.0)
        if not (_is_finite(d_raw) and _is_finite(L_raw)):
            continue

        lam = _gas_friction(pipe.model)
        C2 = _calc_C_squared(float(d_raw), float(L_raw), t_k, z, _gas_r_specific(gas_grid))
        b_raw = C2 / (2.0 * lam * m0_eff)
        if not _is_finite(b_raw) or b_raw <= 0:
            continue
        raw_pipes.append((fi, ti, b_raw, pipe.id))

    B = np.zeros((n, n), dtype=float)
    pipe_data: List[Tuple[int, int, float]] = []
    pipe_ids: List[int] = []

    if raw_pipes:
        bs = np.array([p[2] for p in raw_pipes], dtype=float)
        b_med = float(np.median(bs))
        b_cap = (
            cfg.SUSCEPTANCE_B_RELATIVE_CAP * b_med
            if b_med > 0.0 and cfg.SUSCEPTANCE_B_RELATIVE_CAP > 0.0
            else float("inf")
        )
        for fi, ti, b_raw, pid in raw_pipes:
            b = min(b_raw, b_cap)
            B[fi, fi] += b
            B[ti, ti] += b
            B[fi, ti] -= b
            B[ti, fi] -= b
            pipe_data.append((fi, ti, b))
            pipe_ids.append(pid)

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

    raw_water: List[Tuple[int, int, float, int]] = []  # (fi, ti, b_raw, pipe_id)

    for pipe in monee_net.branches_by_type(mm.WaterPipe):
        if not (pipe.active and int(_val(pipe.model.on_off, 1)) == 1):
            continue
        fi = idx.get(pipe.from_node_id)
        ti = idx.get(pipe.to_node_id)
        if fi is None or ti is None:
            continue

        Rm = _darcy_resistance(pipe.model)
        if Rm is None or Rm <= 0 or not _is_finite(Rm):
            continue

        m0_raw = _val(pipe.model.mass_flow_kgs, 0.0)
        m0 = abs(float(m0_raw)) if _is_finite(m0_raw) else 0.0
        m0_eff = max(m0, cfg.SUSCEPTANCE_FLOW_FLOOR_KGPS)

        b_raw = 1.0 / (2.0 * Rm * m0_eff)
        if not _is_finite(b_raw) or b_raw <= 0:
            continue
        raw_water.append((fi, ti, b_raw, pipe.id))

    # Cap WaterPipe b to bound conditioning.
    if raw_water:
        bs = np.array([p[2] for p in raw_water], dtype=float)
        b_med_water = float(np.median(bs))
        b_cap = (
            cfg.SUSCEPTANCE_B_RELATIVE_CAP * b_med_water
            if b_med_water > 0.0 and cfg.SUSCEPTANCE_B_RELATIVE_CAP > 0.0
            else float("inf")
        )
    else:
        b_med_water = 0.0
        b_cap = float("inf")

    B = np.zeros((n, n), dtype=float)
    pipe_data: List[Tuple[int, int, float]] = []
    pipe_ids: List[int] = []

    def _add_edge(fi, ti, b):
        B[fi, fi] += b
        B[ti, ti] += b
        B[fi, ti] -= b
        B[ti, fi] -= b
        pipe_data.append((fi, ti, b))

    water_b_list: List[float] = []
    for fi, ti, b_raw, pid in raw_water:
        b = min(b_raw, b_cap)
        _add_edge(fi, ti, b)
        water_b_list.append(b)
        pipe_ids.append(pid)

    # HX conductance scaling: scale to median (capped) WaterPipe conductance.
    if water_b_list:
        b_med = float(np.median(water_b_list))
        if b_med > 0.0:
            b_hx = cfg.HX_KAPPA * b_med
        else:
            b_hx = 1.0 / (
                2.0
                * _HX_RESISTANCE_FALLBACK
                * max(cfg.SUSCEPTANCE_FLOW_FLOOR_KGPS, 1e-9)
            )
    else:
        b_hx = 1.0 / (
            2.0
            * _HX_RESISTANCE_FALLBACK
            * max(cfg.SUSCEPTANCE_FLOW_FLOOR_KGPS, 1e-9)
        )

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
# HEAT: thermal-aware extensions (A: thermal margin, B: slack distance)
# =============================================================================


def _heat_injector_node_ids(monee_net) -> set:
    """Heat-grid (water) junctions that act as heat sources.

    Includes nodes with ExtHydrGrid attached on the water grid, and nodes where
    a heat-supplying CP (CHP, CHPHG, PowerToHeat, PowerToHeatHG, GasToHeatHG)
    is connected. These are treated as the boundary condition for thermal
    distance / supply-temperature feasibility.
    """
    ids = set()
    try:
        for c in monee_net.childs_by_type(mm.ExtHydrGrid):
            grid = getattr(c, "grid", None)
            if grid is not None and getattr(grid, "name", None) == "water":
                ids.add(c.node_id)
    except Exception:
        pass

    for cp_type in (mm.CHP, mm.CHPHG, mm.PowerToHeat):
        try:
            for cp in monee_net.compounds_by_type(cp_type):
                connected = _compound_connected_nodes(cp)
                if "heat" in connected:
                    ids.add(connected["heat"])
        except Exception:
            pass

    for cp_type in (mm.PowerToHeatHG, mm.GasToHeatHG):
        try:
            for br in monee_net.branches_by_type(cp_type):
                for nid in (br.from_node_id, br.to_node_id):
                    try:
                        n = monee_net.node_by_id(nid)
                        grid = getattr(n, "grid", None)
                        if grid is not None and getattr(grid, "name", None) == "water":
                            ids.add(nid)
                    except Exception:
                        pass
        except Exception:
            pass

    return ids


def _heat_remoteness_per_node(monee_net, junc_ids: List[int], cfg: CPMetricConfig) -> Dict[int, float]:
    """Per-junction thermal-loss distance to the nearest heat injector.

    Edge weight: L · U·π·d / (ṁ·cp) — the dimensionless temperature-decay
    factor along the pipe (so that T_out ≈ T_amb + (T_in − T_amb)·exp(−weight)).
    HeatExchanger edges get weight 0 (no length-wise loss). Returns 0 for
    injector nodes; nodes unreachable from any injector get the max finite
    distance (so they aren't dropped from the score).
    """
    G = nx.Graph()
    G.add_nodes_from(junc_ids)
    junc_set = set(junc_ids)
    cp = float(cfg.HEAT_CP_J_PER_KG_K)
    U = float(cfg.HEAT_PIPE_U_W_M2_K)

    for pipe in monee_net.branches_by_type(mm.WaterPipe):
        if not pipe.active:
            continue
        fi, ti = pipe.from_node_id, pipe.to_node_id
        if fi not in junc_set or ti not in junc_set:
            continue
        d_raw = _val(getattr(pipe.model, "diameter_m", None), None)
        L_raw = _val(getattr(pipe.model, "length_m", None), None)
        m_raw = _first_attr(pipe.model, ["mass_flow_kgs", "mass_flow"], default=0.0)
        if not (_is_finite(d_raw) and _is_finite(L_raw)):
            continue
        d = float(d_raw)
        L = float(L_raw)
        if d <= 0 or L <= 0:
            continue
        m_dot = max(abs(float(m_raw) if _is_finite(m_raw) else 0.0), cfg.FLOW_MIN)
        kappa = (U * math.pi * d) / (m_dot * cp)  # 1/m
        weight = L * kappa
        if G.has_edge(fi, ti):
            if weight < G[fi][ti]["weight"]:
                G[fi][ti]["weight"] = weight
        else:
            G.add_edge(fi, ti, weight=weight)

    for hx in monee_net.branches_by_type(mm.HeatExchanger):
        if not hx.active:
            continue
        fi, ti = hx.from_node_id, hx.to_node_id
        if fi in junc_set and ti in junc_set and not G.has_edge(fi, ti):
            G.add_edge(fi, ti, weight=0.0)

    injectors = [nid for nid in junc_ids if nid in _heat_injector_node_ids(monee_net)]
    if not injectors or not G.edges:
        return {nid: 0.0 for nid in junc_ids}

    try:
        dists = nx.multi_source_dijkstra_path_length(G, sources=injectors, weight="weight")
    except Exception:
        dists = {nid: 0.0 for nid in injectors}

    finite_vals = [v for v in dists.values() if math.isfinite(v)]
    cap = max(finite_vals) if finite_vals else 0.0
    # Normalise to [0, 1] (fraction of the farthest node's distance). The
    # raw Σ L·κ distance is unbounded and dominated by the FLOW_MIN floor:
    # one idle 100 m pipe contributes weight ≈ 100 · (U·π·d)/(FLOW_MIN·cp)
    # ≈ 113, so unnormalised distances inflated remote nodes' heat scores
    # by 2–4 orders of magnitude and drowned the PTDF/margin signal. After
    # normalisation the remoteness multiplier (1 + α·d) is bounded to
    # [1, 1+α] while preserving the remoteness ordering.
    if cap > 0:
        return {
            nid: float(min(dists.get(nid, cap), cap)) / cap for nid in junc_ids
        }
    return {nid: 0.0 for nid in junc_ids}


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

    # Use apparent power S = √(P² + Q²) so the margin (limit − flow) is unit-
    # consistent with the MVA limit. Falling back to |P| alone overestimates
    # margin when Q is non-trivial (e.g. low-PF buses).
    p = _first_attr(bm, ["p_from_mw", "p_mw", "p_from", "p"], default=0.0)
    q = _first_attr(bm, ["q_from_mvar", "q_mvar", "q_from", "q"], default=0.0)
    p_v = float(p) if p is not None and _is_finite(p) else 0.0
    q_v = float(q) if q is not None and _is_finite(q) else 0.0
    flow0 = math.hypot(p_v, q_v)

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

    flow0 = abs(_first_attr(pm, ["mass_flow_kgs", "mass_flow_pos_kgs", "mass_flow", "mass_flow_pos", "m_dot"], default=0.0))

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

    flow0 = abs(_first_attr(pm, ["mass_flow_kgs", "mass_flow_pos_kgs", "mass_flow", "mass_flow_pos", "m_dot"], default=0.0))

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
    """Build the physical-topology graph and its (weighted) betweenness.

    Edge weights model "resistive cost of traversing this branch" so that
    shortest-path BC reflects how hard it is to route through the grid:

      * ``GenericPowerBranch``  → reactance ``|br_x|``
      * ``GasPipe``             → Weymouth proxy ``L / d⁵``
      * ``WaterPipe``           → Darcy resistance (see ``_darcy_resistance``)
      * everything else (CPs, ``GenericTransferBranch``, unresolved edges)
        gets a placeholder

    Raw cross-carrier weights span ~7 orders of magnitude (power ≈ 5e-3,
    heat ≈ 60, gas ≈ 2e4 on the simbench LV grid), which would make almost
    every Dijkstra shortest path stay inside the cheapest carrier and
    starve the other carriers of BC mass. To make the carriers comparable
    we *per-carrier-median-normalise*: ``w' = w / median(w_carrier)`` so
    every carrier's median edge has unit cost, and the relative within-
    carrier ordering is preserved.

    CP / transfer / missing-data edges land at ``w' = 1.0`` after
    normalisation — the same scale as the median edge of every carrier —
    i.e. they look like "an average resistive hop" rather than the
    unit-1 placeholder they had before (which used to be ~200× cheaper
    than a typical power line and ~2e4× cheaper than a typical gas pipe).
    """
    G0 = monee_net._network_internal
    G = nx.Graph(G0)  # copy

    # Pass 1 — classify each edge and collect raw resistive weights.
    # ``None`` for ``raw`` means "no physical weight available; fall back to
    # the post-normalisation median (1.0)".
    #
    # The monee MultiGraph stores the actual branch object on each edge as
    # ``internal_branch``; the old code's ``branch_by_id((u, v))`` lookup
    # silently failed for every edge on simbench grids (the lookup wants
    # ``(u, v, key)``), which collapsed every weight to ``w = 1.0`` and
    # negated the per-carrier resistance differentiation entirely.
    edge_class: dict = {}  # (u, v) -> (carrier_tag, raw_weight_or_None)
    mapped = 0
    total = 0
    for u, v, data in G.edges(data=True):
        total += 1
        br = data.get("internal_branch", None)

        # Fallbacks — keep the old lookup paths as a safety net for
        # custom Network builders that don't attach ``internal_branch``.
        if br is None:
            bid = data.get("branch_id", None)
            if bid is not None:
                try:
                    br = monee_net.branch_by_id(bid)
                except Exception:
                    br = None
        if br is None:
            for cand in ((u, v, 0), (v, u, 0), (u, v), (v, u)):
                try:
                    br = monee_net.branch_by_id(cand)
                    break
                except Exception:
                    continue

        carrier = "cp"  # CPs / GenericTransferBranch / unresolved → placeholder
        raw: Optional[float] = None
        if br is not None:
            mapped += 1
            bm = br.model
            if isinstance(bm, mm.GenericPowerBranch):
                carrier = "power"
                x = abs(_val(getattr(bm, "br_x_pu", 0.0), 0.0))
                raw = x if x > 0 else None
            elif isinstance(bm, mm.GasPipe):
                carrier = "gas"
                d = _val(getattr(bm, "diameter_m", 0.0), 0.0)
                L = _val(getattr(bm, "length_m", 1.0), 1.0)
                raw = (L / (d**5)) if d > 0 else None
            elif isinstance(bm, mm.WaterPipe):
                carrier = "heat"
                Rm = _darcy_resistance(bm)
                raw = Rm if (Rm is not None and Rm > 0) else None
            # else: classifies as ``cp`` (CPs, transfer branches, heat exchangers)
        edge_class[(u, v)] = (carrier, raw)

    # Pass 2 — per-carrier medians of finite raw weights. We deliberately
    # use the median (robust to a few extreme pipe lengths) rather than the
    # mean (which a single 5 km gas trunk pipe could blow up).
    carrier_medians: dict = {}
    for carrier in ("power", "gas", "heat"):
        vals = [w for (c, w) in edge_class.values()
                if c == carrier and w is not None and np.isfinite(w)]
        carrier_medians[carrier] = float(np.median(vals)) if vals else 1.0

    # Pass 3 — assign normalised weights.
    #   carrier-internal edge with a finite raw weight:
    #       w' = raw / median(carrier)
    #   carrier-internal edge whose raw weight was missing (br_x=0, etc.):
    #       w' = 1.0  (the post-normalisation median of every carrier)
    #   CP / transfer / unresolved edge:
    #       w' = 1.0  (equivalent to the resistive median; sits on the
    #                  same scale as the surrounding carrier edges)
    for (u, v), (carrier, raw) in edge_class.items():
        if carrier == "cp" or raw is None:
            w_norm = 1.0
        else:
            denom = carrier_medians.get(carrier, 1.0) or 1.0
            w_norm = raw / denom
        G.edges[u, v]["weight"] = float(w_norm)

    bc = nx.betweenness_centrality(G, weight="weight")
    deg = dict(G.degree())
    debug = {
        "topo_edge_count": total,
        "topo_mapped_ratio": (mapped / total) if total else 0.0,
        "topo_carrier_medians": carrier_medians,
    }
    return G, bc, deg, debug


# ─────────────────────────────────────────────────────────────────────────────
# CP-aware physical-topology metrics
#
# Companion to ``compute_physical_topology_metrics`` that replaces the
# placeholder ``w_cp = 1.0`` on every coupling-point edge with the
# dimensionally-coherent
#
#     w_cp_raw = (2 − η) / Φ_rated      [MW⁻¹]
#
# where ``η`` is the conversion efficiency and ``Φ_rated`` is the rated
# *input* throughput in MW.  The "(2 − η)" form combines the inverse-
# capacity cost (``1/Φ_rated``) and the exergy-loss equivalent resistance
# ((1 − η)/Φ_rated); see the standalone derivation in
# ``docs/cp_edge_weight_theory.tex`` for the full argument.
#
# The function adds ``"cp"`` as a fourth normalisation class alongside
# ``power``, ``gas``, ``heat`` (the existing patch-C split), so CP edges
# end up on a defensible *within-CP* ordering: a big efficient CP is
# cheaper than a small lossy one, and the median CP sits at unit cost on
# the BC graph.
# ─────────────────────────────────────────────────────────────────────────────


_CP_BRANCH_LABEL = {
    mm.PowerToGas:    "PowerToGas",
    mm.GasToPower:    "GasToPower",
    mm.PowerToHeatHG: "PowerToHeatHG",
    mm.GasToHeatHG:   "GasToHeatHG",
}

_CP_COMPOUND_LABEL = {
    mm.CHP:         "CHP",
    mm.CHPHG:       "CHPHG",
    mm.PowerToHeat: "PowerToHeat",
}


def _cp_efficiency(model, label: str) -> float:
    """Total conversion efficiency η ∈ (0, 1].

    Cogeneration CPs (CHP, CHPHG) return the *sum* of electric + heat
    efficiencies — the "useful output / input" ratio across all output
    ports. Single-output CPs return their lone ``efficiency`` attribute.
    Falls back to 0.9 for missing fields (a generous default that errs on
    the side of treating the CP as if it were nearly lossless).
    """
    try:
        if label in ("CHP", "CHPHG"):
            ep = float(getattr(model, "efficiency_power", 0.35))
            eh = float(getattr(model, "efficiency_heat", 0.50))
            eta = ep + eh
            return float(np.clip(eta, 1e-3, 1.0))
        if label in ("PowerToHeat", "PowerToHeatHG", "GasToHeatHG",
                     "PowerToGas", "GasToPower"):
            eta = float(getattr(model, "efficiency", 0.9))
            return float(np.clip(eta, 1e-3, 1.0))
    except Exception:
        pass
    return 0.9


def _cp_rated_capacity_mw(comp, label: str, monee_net) -> float:
    """Rated input throughput in MW. Re-uses ``_cp_throughput_proxy`` (which
    returns the *output* in pu of system base) and back-transforms to
    *input* MW by dividing by efficiency and multiplying by sn_mva.
    """
    sn_mva = _system_sn_mva(monee_net) if monee_net is not None else 100.0
    output_pu = _cp_throughput_proxy(comp, label, monee_net)  # output / sn_mva
    eta = _cp_efficiency(comp.model, label)
    output_mw = float(output_pu) * float(sn_mva)
    if eta <= 0.0:
        return max(output_mw, 1e-6)
    return max(output_mw / eta, 1e-6)


def _build_cp_param_map(monee_net) -> Dict[Tuple, Tuple[float, float, str]]:
    """Pre-compute ``(η, Φ_rated, label)`` for every edge that belongs to a
    coupling point — both branch CPs (direct edges) and the internal
    GenericTransferBranch edges of compound CPs (CHP, CHPHG, PowerToHeat).

    Returned dict is keyed by the *unordered* edge pair ``frozenset({u, v})``
    so we don't have to care about the edge orientation in nx.Graph.
    """
    param_map: Dict[Tuple, Tuple[float, float, str]] = {}

    # Branch CPs — direct edges, one entry each.
    for cls, label in _CP_BRANCH_LABEL.items():
        for br in monee_net.branches_by_type(cls):
            try:
                eta = _cp_efficiency(br.model, label)
                cap = _cp_rated_capacity_mw(br, label, monee_net)
                key = frozenset({br.from_node_id, br.to_node_id})
                param_map[key] = (eta, cap, label)
            except Exception:
                continue

    # Compound CPs — every internal branch (typically GenericTransferBranch
    # connecting the control node to external nodes) inherits the compound's
    # (η, Φ_rated). The compound's `connected_to` dict gives the external
    # node ids per port; combined with its internal nodes this covers all
    # internal edges in the physical graph.
    for cls, label in _CP_COMPOUND_LABEL.items():
        for cp in monee_net.compounds_by_type(cls):
            try:
                eta = _cp_efficiency(cp.model, label)
                cap = _cp_rated_capacity_mw(cp, label, monee_net)
            except Exception:
                continue
            internal_node_ids = {n.id for n in cp.component_of_type(MNode)}
            external_ids = list((cp.connected_to or {}).values())
            # Edge: every (internal, external) pair plus internal cliques.
            for ext in external_ids:
                for intn in internal_node_ids:
                    param_map[frozenset({intn, ext})] = (eta, cap, label)
            for a in internal_node_ids:
                for b in internal_node_ids:
                    if a == b:
                        continue
                    param_map[frozenset({a, b})] = (eta, cap, label)
    return param_map


def compute_physical_topology_metrics_cp_aware(monee_net):
    """Patch-C topology + CP-aware edge weights.

    Like :func:`compute_physical_topology_metrics`, but coupling-point edges
    (branch CPs and compound-internal transfer branches) get a physically
    grounded raw weight ``w_cp = (2 − η)/Φ_rated`` instead of the placeholder
    ``w = 1.0``.

    A new ``cp`` class joins ``power/gas/heat`` in the median-normalisation
    step. The median CP edge ends up at unit cost just like the median of
    every other class, but **within** the CP class big-efficient CPs are
    cheaper to traverse than small-lossy ones, finally giving the BC the
    capacity-sensitivity it was missing.
    """
    G0 = monee_net._network_internal
    G = nx.Graph(G0)
    cp_params = _build_cp_param_map(monee_net)

    edge_class: dict = {}
    mapped = 0
    total = 0
    for u, v, data in G.edges(data=True):
        total += 1
        br = data.get("internal_branch", None)
        if br is None:
            for cand in ((u, v, 0), (v, u, 0), (u, v), (v, u)):
                try:
                    br = monee_net.branch_by_id(cand)
                    break
                except Exception:
                    continue

        # 1) check CP membership first (covers branch CPs AND internal
        #    edges of compound CPs).
        cp_key = frozenset({u, v})
        if cp_key in cp_params:
            mapped += 1
            eta, cap, _label = cp_params[cp_key]
            raw_cp = (2.0 - eta) / max(cap, 1e-6)
            edge_class[(u, v)] = ("cp", raw_cp)
            continue

        # 2) passive carrier branches.
        carrier = "other"  # falls back to median = 1
        raw: Optional[float] = None
        if br is not None:
            mapped += 1
            bm = br.model
            if isinstance(bm, mm.GenericPowerBranch):
                carrier = "power"
                x = abs(_val(getattr(bm, "br_x_pu", 0.0), 0.0))
                raw = x if x > 0 else None
            elif isinstance(bm, mm.GasPipe):
                carrier = "gas"
                d = _val(getattr(bm, "diameter_m", 0.0), 0.0)
                L = _val(getattr(bm, "length_m", 1.0), 1.0)
                raw = (L / (d**5)) if d > 0 else None
            elif isinstance(bm, mm.WaterPipe):
                carrier = "heat"
                Rm = _darcy_resistance(bm)
                raw = Rm if (Rm is not None and Rm > 0) else None
        edge_class[(u, v)] = (carrier, raw)

    class_medians: dict = {}
    for cls in ("power", "gas", "heat", "cp"):
        vals = [w for (c, w) in edge_class.values()
                if c == cls and w is not None and np.isfinite(w)]
        class_medians[cls] = float(np.median(vals)) if vals else 1.0

    for (u, v), (cls, raw) in edge_class.items():
        if cls == "other" or raw is None:
            w_norm = 1.0
        else:
            denom = class_medians.get(cls, 1.0) or 1.0
            w_norm = raw / denom
        G.edges[u, v]["weight"] = float(w_norm)

    bc = nx.betweenness_centrality(G, weight="weight")
    deg = dict(G.degree())
    debug = {
        "topo_edge_count": total,
        "topo_mapped_ratio": (mapped / total) if total else 0.0,
        "topo_class_medians": class_medians,
        "topo_cp_edge_count": sum(1 for (c, _) in edge_class.values() if c == "cp"),
    }
    return G, bc, deg, debug


# ─────────────────────────────────────────────────────────────────────────────
# Exergy-aware physical-topology metrics
#
# Companion to ``compute_physical_topology_metrics_cp_aware`` that replaces
# the energy efficiency η with the *exergetic* efficiency η_ex per the
# corrected derivation in ``docs/new_edge_weight_theory.tex``:
#
#     η_ex = (Σ_j η_j q_j) / q_in        (multi-port, energy-summed)
#     w_cp = (2 − η_ex) / Φ_rated        [MW⁻¹]
#
# where q_k ∈ (0,1] is the carrier exergy quality factor:
#   q_el  = 1            (pure exergy)
#   q_gas = 1            (Szargut β capped at 1 for BC use)
#   q_heat= 1 − T0/Ts    (Carnot factor; depends on the heat-grid Ts)
#
# This penalises low-exergy outputs (district heat) so that e.g. a gas
# boiler — which destroys ~80 % of the input exergy despite ~92 % energy
# efficiency — gets a noticeably higher weight than its energy-only
# variant would suggest.  The classification, normalisation, and
# return-value shapes are otherwise identical to the CP-aware variant.
# ─────────────────────────────────────────────────────────────────────────────


# Carrier of the input port and (output_carrier → energy_efficiency_attr)
# map for the output ports.  Used by ``_cp_exergy_efficiency`` to compute
# the per-carrier exergy contributions.
_CP_IO_SPEC: Dict[str, Tuple[str, List[Tuple[str, str]]]] = {
    "CHP":           ("gas", [("power", "efficiency_power"),
                              ("heat",  "efficiency_heat")]),
    "CHPHG":         ("gas", [("power", "efficiency_power"),
                              ("heat",  "efficiency_heat")]),
    "PowerToHeat":   ("power", [("heat", "efficiency")]),
    "PowerToHeatHG": ("power", [("heat", "efficiency")]),
    "GasToHeatHG":   ("gas",   [("heat", "efficiency")]),
    "PowerToGas":    ("power", [("gas",  "efficiency")]),
    "GasToPower":    ("gas",   [("power", "efficiency")]),
}


def _carrier_quality_factor(carrier: str, monee_net, cfg: CPMetricConfig) -> float:
    """Return q_k ∈ (0,1] for a given carrier.

    Heat uses the Carnot factor 1 − T0/Ts.  Ts is read from the network's
    WaterGrid ``t_ref`` if present (so a 70 °C system gets a different
    factor than a 120 °C system), otherwise from
    ``cfg.EXERGY_T_HEAT_SUPPLY_K``.
    """
    c = (carrier or "").lower()
    if c in ("electricity", "el", "power"):
        return float(cfg.EXERGY_Q_EL)
    if c == "gas":
        return float(cfg.EXERGY_Q_GAS)
    if c in ("heat", "water"):
        T0 = float(cfg.EXERGY_T_AMBIENT_K)
        Ts = float(cfg.EXERGY_T_HEAT_SUPPLY_K)
        if monee_net is not None:
            try:
                for grid in getattr(monee_net, "grids", []) or []:
                    name = (getattr(grid, "name", "") or "").lower()
                    if name in ("heat", "water"):
                        tref = float(_first_attr(grid, ["t_ref_k", "t_ref"], default=Ts))
                        if tref > T0:
                            Ts = tref
                            break
            except Exception:
                pass
        if Ts <= T0:
            return 1e-3
        return float(np.clip(1.0 - T0 / Ts, 1e-3, 1.0))
    return 1.0  # default to "high-quality" for unknown carriers


def _cp_exergy_efficiency(model, label: str,
                          monee_net, cfg: CPMetricConfig) -> float:
    """Aggregate exergetic efficiency η_ex = (Σ η_j q_j) / q_in.

    For multi-output CPs (CHP, CHPHG) the numerator sums over each output
    port's *energy* efficiency weighted by the output carrier's quality.
    For single-output CPs the formula reduces to η · q_out / q_in.
    """
    spec = _CP_IO_SPEC.get(label)
    if spec is None:
        return _cp_efficiency(model, label)  # fall back to energy-η

    in_carrier, outputs = spec
    q_in = _carrier_quality_factor(in_carrier, monee_net, cfg)
    if q_in <= 0:
        return 0.0

    weighted = 0.0
    for out_carrier, attr in outputs:
        try:
            eta_j = float(getattr(model, attr, 0.0))
        except Exception:
            eta_j = 0.0
        eta_j = float(np.clip(eta_j, 0.0, 1.0))
        q_out = _carrier_quality_factor(out_carrier, monee_net, cfg)
        weighted += eta_j * q_out

    eta_ex = weighted / q_in
    # Heat pumps and other carriers where COP > 1 can push η_ex above 1;
    # cap to 1 so the (2 − η_ex) term stays in [1, 2) per Axiom W1 of the
    # exergy theory.
    return float(np.clip(eta_ex, 1e-3, 1.0))


def _build_cp_param_map_exergy(monee_net, cfg: CPMetricConfig) -> Dict[Tuple, Tuple[float, float, str]]:
    """Same shape as ``_build_cp_param_map`` but stores η_ex (not η)."""
    param_map: Dict[Tuple, Tuple[float, float, str]] = {}

    for cls, label in _CP_BRANCH_LABEL.items():
        for br in monee_net.branches_by_type(cls):
            try:
                eta_ex = _cp_exergy_efficiency(br.model, label, monee_net, cfg)
                cap = _cp_rated_capacity_mw(br, label, monee_net)
                key = frozenset({br.from_node_id, br.to_node_id})
                param_map[key] = (eta_ex, cap, label)
            except Exception:
                continue

    for cls, label in _CP_COMPOUND_LABEL.items():
        for cp in monee_net.compounds_by_type(cls):
            try:
                eta_ex = _cp_exergy_efficiency(cp.model, label, monee_net, cfg)
                cap = _cp_rated_capacity_mw(cp, label, monee_net)
            except Exception:
                continue
            internal_node_ids = {n.id for n in cp.component_of_type(MNode)}
            external_ids = list((cp.connected_to or {}).values())
            for ext in external_ids:
                for intn in internal_node_ids:
                    param_map[frozenset({intn, ext})] = (eta_ex, cap, label)
            for a in internal_node_ids:
                for b in internal_node_ids:
                    if a == b:
                        continue
                    param_map[frozenset({a, b})] = (eta_ex, cap, label)
    return param_map


def compute_physical_topology_metrics_exergy_aware(
    monee_net, cfg: Optional[CPMetricConfig] = None,
):
    """Patch-C topology + CP weights based on the exergetic efficiency.

    Identical structure to
    :func:`compute_physical_topology_metrics_cp_aware`, but the CP raw
    weight uses η_ex (multi-port aggregated) instead of the energy
    efficiency η:

      w_cp_raw = (2 − η_ex) / Φ_rated      [MW⁻¹]
                 with η_ex = (Σ η_j q_j) / q_in

    See ``docs/new_edge_weight_theory.tex`` for the derivation and
    ``_carrier_quality_factor`` / ``_cp_exergy_efficiency`` for the
    quality-factor definitions.
    """
    if cfg is None:
        cfg = CPMetricConfig()

    G0 = monee_net._network_internal
    G = nx.Graph(G0)
    cp_params = _build_cp_param_map_exergy(monee_net, cfg)

    edge_class: dict = {}
    mapped = 0
    total = 0
    for u, v, data in G.edges(data=True):
        total += 1
        br = data.get("internal_branch", None)
        if br is None:
            for cand in ((u, v, 0), (v, u, 0), (u, v), (v, u)):
                try:
                    br = monee_net.branch_by_id(cand)
                    break
                except Exception:
                    continue

        cp_key = frozenset({u, v})
        if cp_key in cp_params:
            mapped += 1
            eta_ex, cap, _label = cp_params[cp_key]
            raw_cp = (2.0 - eta_ex) / max(cap, 1e-6)
            edge_class[(u, v)] = ("cp", raw_cp)
            continue

        carrier = "other"
        raw: Optional[float] = None
        if br is not None:
            mapped += 1
            bm = br.model
            if isinstance(bm, mm.GenericPowerBranch):
                carrier = "power"
                x = abs(_val(getattr(bm, "br_x_pu", 0.0), 0.0))
                raw = x if x > 0 else None
            elif isinstance(bm, mm.GasPipe):
                carrier = "gas"
                d = _val(getattr(bm, "diameter_m", 0.0), 0.0)
                L = _val(getattr(bm, "length_m", 1.0), 1.0)
                raw = (L / (d**5)) if d > 0 else None
            elif isinstance(bm, mm.WaterPipe):
                carrier = "heat"
                Rm = _darcy_resistance(bm)
                raw = Rm if (Rm is not None and Rm > 0) else None
        edge_class[(u, v)] = (carrier, raw)

    class_medians: dict = {}
    for cls in ("power", "gas", "heat", "cp"):
        vals = [w for (c, w) in edge_class.values()
                if c == cls and w is not None and np.isfinite(w)]
        class_medians[cls] = float(np.median(vals)) if vals else 1.0

    for (u, v), (cls, raw) in edge_class.items():
        if cls == "other" or raw is None:
            w_norm = 1.0
        else:
            denom = class_medians.get(cls, 1.0) or 1.0
            w_norm = raw / denom
        G.edges[u, v]["weight"] = float(w_norm)

    bc = nx.betweenness_centrality(G, weight="weight")
    deg = dict(G.degree())
    debug = {
        "topo_edge_count": total,
        "topo_mapped_ratio": (mapped / total) if total else 0.0,
        "topo_class_medians": class_medians,
        "topo_cp_edge_count": sum(1 for (c, _) in edge_class.values() if c == "cp"),
        "exergy_q_heat": _carrier_quality_factor("heat", monee_net, cfg),
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
                bm = monee_net.branch_by_id(bid).model
                p0 = abs(_first_attr(bm, ["p_from_mw", "p_mw", "p_from", "p"], default=0.0))
                q0 = abs(_first_attr(bm, ["q_from_mvar", "q_mvar", "q_from", "q"], default=0.0))
                flow0 = math.hypot(p0, q0)
            except Exception:
                flow0 = 0.0
            stress_by_id[bid] = float(flow0) / (margin + cfg.EPS_MARGIN)

    if "built" in ctx.gas:
        for i, pid in enumerate(ctx.gas["pipe_ids"]):
            margin = float(ctx.gas["margins"][i]) if i < len(ctx.gas["margins"]) else cfg.MIN_MARGIN
            try:
                flow0 = abs(_first_attr(monee_net.branch_by_id(pid).model,
                                        ["mass_flow_kgs", "mass_flow_pos_kgs", "mass_flow", "mass_flow_pos", "m_dot"], default=0.0))
            except Exception:
                flow0 = 0.0
            stress_by_id[pid] = float(flow0) / (margin + cfg.EPS_MARGIN)

    if "built" in ctx.heat:
        for i, pid in enumerate(ctx.heat["pipe_ids"]):
            margin = float(ctx.heat["margins"][i]) if i < len(ctx.heat["margins"]) else cfg.MIN_MARGIN
            try:
                flow0 = abs(_first_attr(monee_net.branch_by_id(pid).model,
                                        ["mass_flow_kgs", "mass_flow_pos_kgs", "mass_flow", "mass_flow_pos", "m_dot"], default=0.0))
            except Exception:
                flow0 = 0.0
            stress_by_id[pid] = float(flow0) / (margin + cfg.EPS_MARGIN)

    G0 = monee_net._network_internal
    G = nx.Graph(G0)

    EPS = 1e-9
    mapped = 0
    total = 0

    edge_stress: dict = {}
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
        edge_stress[(u, v)] = stress_val

    # weight = 1/(stress+eps): highly loaded edges become "short" (preferred
    # by weighted shortest paths — nx treats weight as distance). Unmapped
    # edges (no stress data at all — notably every branch-CP / transfer
    # edge, whose ids are never in stress_by_id) get the NEUTRAL median of
    # the mapped weights: the earlier 1/EPS assignment made them the
    # *longest* edges in the graph, i.e. shortest paths never crossed a
    # coupling point and stress_bc degenerated to per-carrier-island BC —
    # the exact opposite of the "conservative, inflates BC" intent.
    mapped_weights = [
        1.0 / (s + EPS) for s in edge_stress.values() if s is not None and s > 0
    ]
    neutral_w = float(np.median(mapped_weights)) if mapped_weights else 1.0
    for (u, v), stress_val in edge_stress.items():
        if stress_val is not None and stress_val > 0:
            w = 1.0 / (stress_val + EPS)
        else:
            w = neutral_w
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


def _apply_ablations(
    cfg: "CPMetricConfig",
    throughput: float,
    total_stress: float,
    topo_factor: float,
    input_adequacy: float,
    is_cp: bool,
) -> float:
    """Compose the composite score under the active ablation flags.

    Used by both CP score sites and the non-CP branch sites in
    ``mes_all_components_metric``. ``ABLATE_ADEQUACY`` is silently ignored on
    non-CP rows (they don't carry a gate), so non-CP scores are unchanged
    when only ABLATE_ADEQUACY is set.
    """
    t = 1.0 if cfg.ABLATE_THROUGHPUT else float(throughput)
    s = 1.0 if cfg.ABLATE_STRESS else float(total_stress)
    o = 1.0 if cfg.ABLATE_TOPO else float(topo_factor)
    if is_cp:
        a = 1.0 if cfg.ABLATE_ADEQUACY else float(input_adequacy)
    else:
        a = 1.0
    return t * s * o * a


def _heat_remoteness_factor(ctx, node_id, cfg: "CPMetricConfig") -> float:
    """(1 + α · d_thermal(node)) — slack-distance prefactor for heat stress.

    ``d_thermal`` is normalised to [0, 1] (see ``_heat_remoteness_per_node``)
    so the factor is bounded to [1, 1+α]. Returns 1.0 when remoteness is
    disabled, the node is unknown, or it sits at a heat injector. Larger
    values penalise components farther from any heat source along thermally
    lossy supply paths.
    """
    if not cfg.HEAT_REMOTENESS_ENABLE:
        return 1.0
    rem = ctx.heat.get("remoteness") if hasattr(ctx, "heat") else None
    if not rem:
        return 1.0
    d = float(rem.get(node_id, 0.0) or 0.0)
    return 1.0 + float(cfg.HEAT_REMOTENESS_ALPHA) * d


def _loading_proxy(flow0: float, limit: float, cfg: "CPMetricConfig") -> float:
    """Throughput proxy as |flow| / limit (utilisation ratio in [0, 1+]).

    Replaces the previous flow0/sn_mva which mixed units across carriers
    (kg/s for gas/heat divided by MVA). With limit and flow0 always in matching
    units (per-carrier helpers guarantee this), the result is dimensionless.
    Falls back to |flow| with a small floor when limit is inf/zero (e.g.
    HeatExchanger without geometry), which still gives meaningful per-component
    relative weighting.
    """
    if not cfg.USE_THROUGHPUT_PROXY:
        return 1.0
    f = abs(float(flow0))
    if np.isfinite(limit) and float(limit) > 0:
        return max(f / float(limit), 1e-6)
    return max(f, 1e-6)


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

    def _safe(x, default=0.0):
        v = _val(x, default)
        return float(v) if _is_finite(v) else float(default)

    try:
        if label == "CHP":
            # Rated capacity, not the solved Pyomo Vars (el_mw/heat_mw on the
            # control node), which the optimizer drives to ~0 when idle and
            # would collapse the score. Rated outputs follow from the gas
            # setpoint exactly as in the CHP conversion equations.
            m = cp_or_branch.model
            hhv = (_get_gas_hhv(monee_net) if monee_net is not None
                   else DEFAULT_GAS_HHV_KWH_PER_KG)
            mdot = abs(_safe(getattr(m, "mass_flow_setpoint_kgs", 0.0)))
            el_mw = abs(_safe(getattr(m, "efficiency_power", 0.0))) * mdot * hhv * 3.6
            heat_mw = abs(_safe(getattr(m, "efficiency_heat", 0.0))) * mdot * hhv * 3.6
            return max((el_mw + heat_mw) / sn_mva, 1e-6)

        if label == "CHPHG":
            # Rated capacity from the gas setpoint (the control-node el_mw/heat_mw
            # are solved Vars that collapse to ~0 when idle), mirroring CHP.
            m = cp_or_branch.model
            hhv = (_get_gas_hhv(monee_net) if monee_net is not None
                   else DEFAULT_GAS_HHV_KWH_PER_KG)
            mdot = abs(_safe(getattr(m, "mass_flow_setpoint_kgs", 0.0)))
            el_mw = abs(_safe(getattr(m, "efficiency_power", 0.0))) * mdot * hhv * 3.6
            heat_mw = abs(_safe(getattr(m, "efficiency_heat", 0.0))) * mdot * hhv * 3.6
            return max((el_mw + heat_mw) / sn_mva, 1e-6)

        if label == "PowerToHeat":
            # Rated heat capacity (fixed constructor arg), consistent with the
            # other CPs; the dispatched heat Var would zero out when idle.
            heat_mw = abs(_safe(getattr(cp_or_branch.model, "heat_energy_mw", 0.0)))
            return max(heat_mw / sn_mva, 1e-6)

        if label in ("PowerToHeatHG", "GasToHeatHG"):
            # 2-endpoint HG branches store heat_energy_mw on the model itself.
            heat_mw = abs(_safe(getattr(cp_or_branch.model, "heat_energy_mw", 0.0)))
            return max(heat_mw / sn_mva, 1e-6)

        if label == "GasToPower":
            # Use fixed rated capacity (el_mw), not the solved Pyomo Var (p_to_mw=0 when idle)
            p_mw = abs(_safe(getattr(cp_or_branch.model, "el_mw", 0.0)))
            return max(p_mw / sn_mva, 1e-6)

        if label == "PowerToGas":
            # Use fixed rated capacity (gas_mass_flow_kgs), not the solved Pyomo Var (to_mass_flow=0 when idle)
            m_dot = abs(_safe(getattr(cp_or_branch.model, "gas_mass_flow_kgs", 0.0)))
            hhv = (_get_gas_hhv(monee_net) if monee_net is not None
                   else DEFAULT_GAS_HHV_KWH_PER_KG)
            # HHV is stored in kWh/kg; power [MW] = m_dot [kg/s] * HHV [kWh/kg] * 3.6 [MJ/kWh]
            return max((m_dot * hhv * 3.6) / sn_mva, 1e-6)

    except Exception:
        pass

    return 1.0


# =============================================================================
# PTDF contexts + stress
# =============================================================================


def _stress_from_ptdf(
    ptdf: np.ndarray,
    margins: np.ndarray,
    cfg: CPMetricConfig,
    unit_factor: float = 1.0,
):
    """Aggregate |PTDF|/margin stress.

    ``unit_factor`` is the carrier's within-carrier normalisation constant
    (the carrier's **median margin**, native units): samples become
    ``|ptdf| · median(margin)/margin`` — dimensionless relative tightness,
    O(1) for a typical branch of *every* carrier. This is what makes the
    per-carrier stresses commensurate before they are composed into
    ``total_stress``: without it the raw 1/margin scales differ by orders
    of magnitude across carriers (injection units ~440×, margin-tightness
    regimes more), the mechanism behind the composite's cross-carrier
    scale bias. A per-MW injection conversion cancels inside the
    median(margin)/margin ratio, so the median alone suffices.
    Within-carrier rankings are unaffected (constant factor); the factor
    also puts CLIP_STRESS on one common scale.
    """
    ptdf = np.asarray(ptdf, dtype=float)
    margins = np.asarray(margins, dtype=float)
    denom = margins + cfg.EPS_MARGIN
    with np.errstate(invalid="ignore", divide="ignore"):
        s = float(unit_factor) * np.abs(ptdf) / denom
    # Drop NaN / ±inf samples instead of letting them poison the aggregate.
    # NaN typically signals an ill-conditioned PTDF (singular Laplacian, NaN
    # in the susceptance matrix, etc.) — better to score the well-defined
    # branches than to return NaN for the whole component.
    s = s[np.isfinite(s)]
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
                # Within-carrier normalisation: median margin, so a branch
                # at typical tightness scores |ptdf| — see _stress_from_ptdf.
                "stress_unit": (
                    float(np.median(margins)) if margins.size else 1.0
                ),
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
        # Cache stores ``(ptdf, reliable)`` tuples — matching gas/heat — so
        # the reliable flag survives across repeat queries. The earlier
        # array-only cache silently flipped unreliable nodes to reliable on
        # the second call.
        if node_id in cache:
            return cache[node_id]

        id_to_local = self.power["id_to_local"]
        if node_id not in id_to_local:
            z = (np.zeros(len(self.power["branches"]), dtype=float), False)
            cache[node_id] = z
            return z

        s_local = id_to_local[node_id]
        dP, ok = _distributed_balancing_vector_power(
            monee_net, self.power["bus_ids"], id_to_local, s_local
        )
        if not ok:
            z = (np.zeros(len(self.power["branches"]), dtype=float), False)
            cache[node_id] = z
            return z

        dP_lines = ac_ptdf_distributed(
            self.power["theta"],
            self.power["V"],
            self.power["Ybus"],
            self.power["branches"],
            self.power["bus_types"],
            dP,
            dQ=None,
        )
        out = (np.array(dP_lines, dtype=float), True)
        cache[node_id] = out
        return out

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
                # Within-carrier normalisation: median margin (kg/s) — see
                # _stress_from_ptdf.
                "stress_unit": (
                    float(np.median(margins)) if margins.size else 1.0
                ),
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

        try:
            cond_b = float(np.linalg.cond(B)) if B.size else float("nan")
        except Exception:
            cond_b = float("nan")

        self.debug["gas"] = {
            "n_pipes": len(pipe_ids),
            "finite_limit_ratio": float(np.mean(finite_mask))
            if finite_mask.size
            else 0.0,
            "binding_ratio": float(np.mean(binding_mask)) if binding_mask.size else 0.0,
            "pseudo_used_ratio": pseudo_ratio,
            "margin_min": float(np.min(margins)) if margins.size else None,
            "margin_med": float(np.median(margins)) if margins.size else None,
            "B_condition": cond_b,
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

        # Balancing nodes: ExtHydrGrid only. No "distribute across all nodes"
        # fallback — that has no physical analog (there is no mass-balance
        # mechanism that absorbs gas equally everywhere). If no slack exists in
        # the source's connected component, flag the PTDF unreliable and
        # return zeros, matching how cross-component requests are handled.
        slack_idxs: List[int] = []
        try:
            nodes = [monee_net.node_by_id(nid) for nid in node_ids]
            for i, n in enumerate(nodes):
                if monee_net.has_any_child_of_type(n, mm.ExtHydrGrid):
                    slack_idxs.append(i)
        except Exception:
            slack_idxs = []

        edges = [(fi, ti) for fi, ti, _ in self.gas["pipe_data"]]
        comps = _connected_components_from_edges(len(node_ids), edges)
        src_comp = next((c for c in comps if s_idx in c), None)
        if src_comp is None:
            z = (np.zeros(len(self.gas["pipe_data"]), dtype=float), False)
            cache[node_id] = z
            return z

        bal_idxs = [i for i in slack_idxs if i in src_comp]
        if not bal_idxs:
            z = (np.zeros(len(self.gas["pipe_data"]), dtype=float), False)
            cache[node_id] = z
            return z

        # If source IS the slack, injection is absorbed locally → zero PTDF (reliable).
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

        # Heat keeps the hydraulic capacity − flow margins from
        # _robust_margins, like every other carrier. (The former thermal-
        # margin override was orientation-inverted — see the config note at
        # HEAT_DELTA_T_K.)

        # (B) Per-junction thermal-loss distance to nearest heat injector.
        # Cached on the context so each heat-bearing component just looks up
        # its node id when computing the heat-side score multiplier.
        if self.cfg.HEAT_REMOTENESS_ENABLE and junc_ids:
            try:
                heat_remoteness = _heat_remoteness_per_node(
                    monee_net, junc_ids, self.cfg
                )
            except Exception:
                heat_remoteness = {nid: 0.0 for nid in junc_ids}
        else:
            heat_remoteness = {nid: 0.0 for nid in junc_ids}

        self.heat.update(
            {
                # Within-carrier normalisation: median margin (kg/s) — see
                # _stress_from_ptdf.
                "stress_unit": (
                    float(np.median(margins)) if margins.size else 1.0
                ),
                "B": B,
                "pipe_data": pipe_data,
                "pipe_ids": pipe_ids,
                "node_ids": junc_ids,
                "margins": margins,
                "binding_mask": binding_mask,
                "ptdf_cache": {},
                "remoteness": heat_remoteness,
                "built": True,
            }
        )

        rem_vals = list(heat_remoteness.values()) if heat_remoteness else []
        try:
            cond_b = float(np.linalg.cond(B)) if B.size else float("nan")
        except Exception:
            cond_b = float("nan")

        self.debug["heat"] = {
            "n_edges": len(pipe_ids),
            "finite_limit_ratio": float(np.mean(finite_mask))
            if finite_mask.size
            else 0.0,
            "binding_ratio": float(np.mean(binding_mask)) if binding_mask.size else 0.0,
            "pseudo_used_ratio": pseudo_ratio,
            "margin_min": float(np.min(margins)) if margins.size else None,
            "margin_med": float(np.median(margins)) if margins.size else None,
            "remoteness_min": float(min(rem_vals)) if rem_vals else None,
            "remoteness_med": float(np.median(rem_vals)) if rem_vals else None,
            "remoteness_max": float(max(rem_vals)) if rem_vals else None,
            "B_condition": cond_b,
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

        # Slack nodes: ExtHydrGrid only. See gas_ptdf_node for rationale —
        # distributing slack across arbitrary nodes is not physical.
        slack_idxs: List[int] = []
        try:
            nodes = [monee_net.node_by_id(nid) for nid in node_ids]
            for i, n in enumerate(nodes):
                if monee_net.has_any_child_of_type(n, mm.ExtHydrGrid):
                    slack_idxs.append(i)
        except Exception:
            slack_idxs = []

        edges = [(fi, ti) for fi, ti, _ in self.heat["pipe_data"]]
        comps = _connected_components_from_edges(len(node_ids), edges)
        src_comp = next((c for c in comps if s_idx in c), None)
        if src_comp is None:
            z = (np.zeros(len(self.heat["pipe_data"]), dtype=float), False)
            cache[node_id] = z
            return z

        bal_idxs = [i for i in slack_idxs if i in src_comp]
        if not bal_idxs:
            # No ExtHydrGrid in this component — fall back to CP heat-feed
            # nodes (CHP / P2H / HG variants). On CP-fed heat islands (the
            # backup-CP grid families) these ARE the physical heat sources,
            # so treating them as balancing slack is sound; without the
            # fallback every heat PTDF on such islands was all-zeros and
            # predicted_heat tied at 0 across the whole island.
            injector_ids = _heat_injector_node_ids(monee_net)
            bal_idxs = [i for i in src_comp if node_ids[i] in injector_ids]
        if not bal_idxs:
            z = (np.zeros(len(self.heat["pipe_data"]), dtype=float), False)
            cache[node_id] = z
            return z

        # If source IS the slack (ExtHydrGrid or CP heat feed), any injection
        # is absorbed locally → zero PTDF (reliable).
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


# ──────────────────────────────────────────────────────────────────────────────
# CP input-adequacy: structural P(input sector can feed this CP)
# ──────────────────────────────────────────────────────────────────────────────


def _carrier_input_adequacy_graph(monee_net, carrier: str):
    """Build a per-carrier graph used for analytical CP input-adequacy.

    Edge weights are −log(1 − p_fail(branch)) using the same
    ``DEFAULT_BRANCH_FAIL_PROB`` table the failure model uses, so the
    shortest-path distance d from any slack to a given node satisfies
    P(path-available) = exp(−d). Nodes are the carrier's grid nodes; the
    second return value is the set of slack node ids on that grid.

    The map ``"power" → Bus``, ``"gas"|"heat" → Junction``, ``"heat" → grid name "water"``
    is enforced here so callers don't have to know monee internals.
    """
    G = nx.Graph()
    grid_name = {"power": "power", "gas": "gas", "heat": "water"}.get(carrier)
    if grid_name is None:
        return G, set()

    if carrier == "power":
        node_iter = monee_net.nodes_by_type(mm.Bus)
        slack_iter = monee_net.childs_by_type(mm.ExtPowerGrid)
    else:
        node_iter = monee_net.nodes_by_type(mm.Junction)
        slack_iter = monee_net.childs_by_type(mm.ExtHydrGrid)

    valid = set()
    for n in node_iter:
        grid = getattr(n, "grid", None)
        if grid is not None and getattr(grid, "name", None) == grid_name:
            G.add_node(n.id)
            valid.add(n.id)

    for b in monee_net.branches:
        if not b.active:
            continue
        bm = b.model
        if carrier == "power":
            if not isinstance(bm, mm.GenericPowerBranch):
                continue
        elif carrier == "gas":
            if not isinstance(bm, mm.GasPipe):
                continue
        else:  # heat (water)
            if not isinstance(bm, (mm.WaterPipe, mm.HeatExchanger)):
                continue
        if b.from_node_id not in valid or b.to_node_id not in valid:
            continue
        p_fail = float(DEFAULT_BRANCH_FAIL_PROB.get(type(bm), 0.05))
        if p_fail >= 1.0:
            weight = float("inf")
        elif p_fail <= 0.0:
            weight = 0.0
        else:
            weight = -math.log(1.0 - p_fail)
        if G.has_edge(b.from_node_id, b.to_node_id):
            # Parallel branches: disconnection requires *all* parallel pipes to
            # fail, so P(disconnect) = Π p_fail and the combined edge weight is
            # −log(1 − Π p_fail). Recover the existing edge's p_fail from its
            # weight, multiply, then re-encode.
            existing_w = float(G[b.from_node_id][b.to_node_id]["weight"])
            if not math.isfinite(existing_w):
                existing_p = 1.0
            else:
                existing_p = 1.0 - math.exp(-existing_w)
            combined_p = max(0.0, min(1.0, existing_p * p_fail))
            if combined_p >= 1.0:
                new_weight = float("inf")
            elif combined_p <= 0.0:
                new_weight = 0.0
            else:
                new_weight = -math.log(1.0 - combined_p)
            G[b.from_node_id][b.to_node_id]["weight"] = new_weight
        else:
            G.add_edge(b.from_node_id, b.to_node_id, weight=weight)

    slack_nodes = set()
    for c in slack_iter:
        grid = getattr(c, "grid", None)
        if grid is not None and getattr(grid, "name", None) == grid_name:
            slack_nodes.add(c.node_id)

    return G, slack_nodes


def _build_input_adequacy_cache(monee_net, cfg: CPMetricConfig):
    """Pre-compute, for each carrier, the per-node adequacy probability
    ``P(adequate at v) = exp(−min-cost path from any slack to v)``.

    Returns ``{carrier → {node_id → P}}`` plus a debug dict.
    """
    cache: Dict[str, Dict[int, float]] = {}
    debug: Dict[str, Dict[str, float]] = {}
    for carrier in ("power", "gas", "heat"):
        G, slacks = _carrier_input_adequacy_graph(monee_net, carrier)
        if not G.number_of_nodes() or not slacks:
            cache[carrier] = {}
            debug[carrier] = {"n_nodes": G.number_of_nodes(),
                              "n_slacks": len(slacks)}
            continue
        # Slacks themselves are P=1 (not failed; even if they were, they're
        # the boundary condition by definition). Multi-source Dijkstra gives
        # min-cost path from any slack to every reachable node.
        try:
            dists = nx.multi_source_dijkstra_path_length(
                G, sources=list(slacks), weight="weight"
            )
        except Exception:
            dists = {}
        adequacy = {nid: math.exp(-float(d)) for nid, d in dists.items()}
        # Unreachable nodes stay absent → caller maps them to 0.0.
        cache[carrier] = adequacy
        if adequacy:
            vals = list(adequacy.values())
            debug[carrier] = {
                "n_nodes": G.number_of_nodes(),
                "n_slacks": len(slacks),
                "n_reachable": len(adequacy),
                "P_min": float(min(vals)),
                "P_med": float(np.median(vals)),
                "P_max": float(max(vals)),
            }
        else:
            debug[carrier] = {"n_nodes": G.number_of_nodes(),
                              "n_slacks": len(slacks),
                              "n_reachable": 0}
    return cache, debug


def _cp_input_adequacy(
    cp,
    cp_label: str,
    monee_net,
    adequacy_cache: Dict[str, Dict[int, float]],
    cfg: CPMetricConfig,
) -> float:
    """P(input adequate) for one CP, looked up from the cached per-node table.

    Returns 1.0 (= no down-weighting) when the input carrier is unknown, the
    cache is empty for that carrier, or the input node sits at a slack.
    Returns 0.0 when the CP's input node exists but is unreachable from any
    slack — i.e. the input sector simply cannot feed it.
    """
    if not cfg.CP_INPUT_ADEQUACY_ENABLE:
        return 1.0
    in_carrier = CP_INPUT_CARRIER.get(cp_label)
    if in_carrier is None:
        return 1.0
    if cp_label in COMPOUND_CP_LABELS:
        nodes = _compound_connected_nodes(cp)
    else:
        nodes = _carrier_nodes_for_branch(monee_net, cp)
    in_node = nodes.get(in_carrier)
    if in_node is None:
        return 1.0
    table = adequacy_cache.get(in_carrier, {})
    if not table:
        return 1.0
    # Reachable but possibly low; missing → 0 (unreachable from any slack).
    return float(table.get(in_node, 0.0))


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
    input_adequacy: float = 1.0,
    # CP-aware (energy-η) variant: w_cp = (2−η)/Φ_rated.
    topo_bc_cp_aware: float = 0.0,
    topo_factor_cp_aware: float = 1.0,
    score_cp_aware: Optional[float] = None,
    # Exergy-aware variant (η_ex = (Σ η_j q_j)/q_in, see
    # docs/new_edge_weight_theory.tex): w_cp = (2 − η_ex)/Φ_rated.
    topo_bc_exergy: float = 0.0,
    topo_factor_exergy: float = 1.0,
    score_exergy: Optional[float] = None,
):
    # score_cp_aware / score_exergy MUST be computed via ``_apply_ablations`` at
    # the call site with the corresponding topo factor. Falling back to
    # ``score * topo_new/topo_old`` was incorrect under ``ABLATE_TOPO=True``,
    # which forces the topo factor inside ``score`` to 1.0 — the ratio rescaled
    # by an inactive divisor.
    row = dict(
        cp_id=cp_id,
        cp_type=cp_type,
        p_fail=float(p_fail),
        throughput=float(throughput),
        topo_bc=float(topo_bc),
        topo_factor=float(topo_factor),
        topo_bc_cp_aware=float(topo_bc_cp_aware),
        topo_factor_cp_aware=float(topo_factor_cp_aware),
        topo_bc_exergy=float(topo_bc_exergy),
        topo_factor_exergy=float(topo_factor_exergy),
        total_stress=float(total_stress),
        score=float(score),
        score_cp_aware=float(score_cp_aware if score_cp_aware is not None else score),
        score_exergy=float(score_exergy if score_exergy is not None else score),
        reliable=bool(reliable),
        input_adequacy=float(input_adequacy),
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

    # topology — three variants, all emitted in parallel so downstream eval
    # can A/B/C-compare them. See docs/cp_edge_weight_theory.tex (CMRES-v1)
    # and docs/new_edge_weight_theory.tex (corrected exergy-aware) for the
    # derivations.
    #   topo_*           : placeholder w_cp=1, per-carrier median norm
    #   topo_*_cp_aware  : energy-η, w_cp = (2−η)/Φ_rated
    #   topo_*_exergy    : exergy-η, w_cp = (2−η_ex)/Φ_rated
    G_phys, bc_individual, deg, topo_dbg = compute_physical_topology_metrics(monee_net)
    G_phys_cpa, bc_individual_cpa, _, topo_dbg_cpa = (
        compute_physical_topology_metrics_cp_aware(monee_net)
    )
    G_phys_ex, bc_individual_ex, _, topo_dbg_ex = (
        compute_physical_topology_metrics_exergy_aware(monee_net, cfg)
    )

    # PTDF contexts
    ctx = CarrierPTDFContext(cfg)

    # Per-node analytical input-adequacy (P(input sector can feed me)) per
    # carrier. Cached once; CPs look up their input node when scoring.
    if cfg.CP_INPUT_ADEQUACY_ENABLE:
        input_adequacy_cache, input_adequacy_dbg = _build_input_adequacy_cache(
            monee_net, cfg
        )
    else:
        input_adequacy_cache, input_adequacy_dbg = {}, {}

    rows = []

    # ------------------------
    # Compound CPs
    # ------------------------
    for cp_type, label in [
        (mm.CHP, "CHP"),
        (mm.CHPHG, "CHPHG"),
        (mm.PowerToHeat, "PowerToHeat"),
    ]:
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
            bc_group_cpa = _group_bc(G_phys_cpa, cp)
            topo_factor_cpa = (1.0 + cfg.TOPO_ALPHA * float(bc_group_cpa))
            bc_group_ex = _group_bc(G_phys_ex, cp)
            topo_factor_ex = (1.0 + cfg.TOPO_ALPHA * float(bc_group_ex))

            carrier_detail = {}
            total_stress = 0.0
            reliable_all = True

            # POWER
            if "power" in connected:
                nid = connected["power"]
                ptdf, ok = ctx.power_ptdf_node(monee_net, nid)
                margins = ctx.power["margins"]
                mean_s, max_s, agg_s = _stress_from_ptdf(
                    ptdf, margins, cfg, ctx.power.get("stress_unit", 1.0)
                )
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
                mean_s, max_s, agg_s = _stress_from_ptdf(
                    ptdf, margins, cfg, ctx.gas.get("stress_unit", 1.0)
                )
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
                mean_s, max_s, agg_s = _stress_from_ptdf(
                    ptdf, margins, cfg, ctx.heat.get("stress_unit", 1.0)
                )
                rem = _heat_remoteness_factor(ctx, nid, cfg)
                carrier_detail["heat"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s * rem,
                )
                total_stress += cfg.W_HEAT * agg_s * rem
                reliable_all = reliable_all and bool(ok)

            input_adequacy = _cp_input_adequacy(
                cp, label, monee_net, input_adequacy_cache, cfg
            )
            score = _apply_ablations(
                cfg, throughput, total_stress, topo_factor, input_adequacy, is_cp=True
            )
            score_cpa = _apply_ablations(
                cfg, throughput, total_stress, topo_factor_cpa, input_adequacy, is_cp=True
            )
            score_ex = _apply_ablations(
                cfg, throughput, total_stress, topo_factor_ex, input_adequacy, is_cp=True
            )

            rows.append(
                _row_from_detail(
                    cp_id=cp.id,
                    cp_type=label,
                    p_fail=p_fail,
                    throughput=throughput,
                    topo_bc=bc_group,
                    topo_factor=topo_factor,
                    topo_bc_cp_aware=bc_group_cpa,
                    topo_factor_cp_aware=topo_factor_cpa,
                    topo_bc_exergy=bc_group_ex,
                    topo_factor_exergy=topo_factor_ex,
                    total_stress=total_stress,
                    score=score,
                    score_cp_aware=score_cpa,
                    score_exergy=score_ex,
                    reliable=reliable_all,
                    detail=carrier_detail,
                    input_adequacy=input_adequacy,
                )
            )

    # ------------------------
    # Branch CPs
    # ------------------------
    for cp_type, label in [
        (mm.PowerToGas, "PowerToGas"),
        (mm.GasToPower, "GasToPower"),
        (mm.PowerToHeatHG, "PowerToHeatHG"),
        (mm.GasToHeatHG, "GasToHeatHG"),
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
            bc_avg_cpa = float(
                np.mean(
                    [
                        bc_individual_cpa.get(n, 0.0)
                        for n in (br.from_node_id, br.to_node_id)
                    ]
                )
            )
            topo_factor_cpa = (1.0 + cfg.TOPO_ALPHA * bc_avg_cpa)
            bc_avg_ex = float(
                np.mean(
                    [
                        bc_individual_ex.get(n, 0.0)
                        for n in (br.from_node_id, br.to_node_id)
                    ]
                )
            )
            topo_factor_ex = (1.0 + cfg.TOPO_ALPHA * bc_avg_ex)

            carrier_detail = {}
            total_stress = 0.0
            reliable_all = True

            if "power" in carrier_nodes:
                nid = carrier_nodes["power"]
                ptdf, ok = ctx.power_ptdf_node(monee_net, nid)
                margins = ctx.power["margins"]
                mean_s, max_s, agg_s = _stress_from_ptdf(
                    ptdf, margins, cfg, ctx.power.get("stress_unit", 1.0)
                )
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
                mean_s, max_s, agg_s = _stress_from_ptdf(
                    ptdf, margins, cfg, ctx.gas.get("stress_unit", 1.0)
                )
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
                mean_s, max_s, agg_s = _stress_from_ptdf(
                    ptdf, margins, cfg, ctx.heat.get("stress_unit", 1.0)
                )
                rem = _heat_remoteness_factor(ctx, nid, cfg)
                carrier_detail["heat"] = dict(
                    node_id=nid,
                    reliable=bool(ok),
                    stress_mean=mean_s,
                    stress_max=max_s,
                    stress=agg_s * rem,
                )
                total_stress += cfg.W_HEAT * agg_s * rem
                reliable_all = reliable_all and bool(ok)

            input_adequacy = _cp_input_adequacy(
                br, label, monee_net, input_adequacy_cache, cfg
            )
            score = _apply_ablations(
                cfg, throughput, total_stress, topo_factor, input_adequacy, is_cp=True
            )
            score_cpa = _apply_ablations(
                cfg, throughput, total_stress, topo_factor_cpa, input_adequacy, is_cp=True
            )
            score_ex = _apply_ablations(
                cfg, throughput, total_stress, topo_factor_ex, input_adequacy, is_cp=True
            )

            rows.append(
                _row_from_detail(
                    cp_id=f"{br.from_node_id}→{br.to_node_id}",
                    cp_type=label,
                    p_fail=p_fail,
                    throughput=throughput,
                    topo_bc=bc_avg,
                    topo_factor=topo_factor,
                    topo_bc_cp_aware=bc_avg_cpa,
                    topo_factor_cp_aware=topo_factor_cpa,
                    topo_bc_exergy=bc_avg_ex,
                    topo_factor_exergy=topo_factor_ex,
                    total_stress=total_stress,
                    score=score,
                    score_cp_aware=score_cpa,
                    score_exergy=score_ex,
                    reliable=reliable_all,
                    detail=carrier_detail,
                    input_adequacy=input_adequacy,
                )
            )

    # When the grid has no CPs (density-0 stems, e.g. simbench_lv_no_backup), the rows
    # list is empty. Constructing the DataFrame normally would yield an
    # empty frame with no columns, and ``sort_values("score")`` would raise
    # ``KeyError: 'score'``. Build the df with the canonical column schema
    # so downstream code (``mes_all_components_metric``,
    # ``cp_metric_vs_actual_impact``) sees a well-typed empty CP table.
    if rows:
        df_scores = (
            pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        )
    else:
        empty_row = _row_from_detail(
            cp_id="", cp_type="", p_fail=0.0, throughput=0.0,
            topo_bc=0.0, topo_factor=0.0, total_stress=0.0,
            score=0.0, reliable=False, detail={}, input_adequacy=1.0,
            topo_bc_cp_aware=0.0, topo_factor_cp_aware=1.0, score_cp_aware=0.0,
            topo_bc_exergy=0.0, topo_factor_exergy=1.0, score_exergy=0.0,
        )
        df_scores = pd.DataFrame(columns=list(empty_row.keys()))

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
    debug_rows.append({"carrier": "topology_cp_aware", **topo_dbg_cpa})
    debug_rows.append({"carrier": "topology_exergy", **topo_dbg_ex})
    for carrier, d in input_adequacy_dbg.items():
        debug_rows.append({"carrier": f"input_adequacy_{carrier}", **d})

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

    This is the first-order proxy Ψ_LODF(·, b) = Ψ(·, u) − Ψ(·, v) for a
    branch b = (u, v).  It has the correct sign and support but diverges
    from the exact generalised LODF of Güler, Gross & Liu (IEEE T-PWRS,
    2007) when the tripped branch is near-binding.  For settings that
    require tighter accuracy, replace this with the rank-one Ψ-update of
    that paper at the cost of one extra solve per branch.

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
    unit = getattr(ctx, carrier).get("stress_unit", 1.0)
    mean_s, max_s, agg_s = _stress_from_ptdf(ptdf_diff, margins, cfg, unit)
    return mean_s, max_s, agg_s, bool(ok_f and ok_t)


def mes_all_components_metric(monee_net, cfg: CPMetricConfig = CPMetricConfig()):
    """
    Score ALL active grid branches (CP and non-CP) using the LODF approximation.

    CPs (CHP, PowerToHeat, GasToPower, PowerToGas) are scored as in
    mes_cp_metric().  Non-CP branches (PowerLine/GenericPowerBranch,
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

    # Build shared context (already built internally by mes_cp_metric, but we
    # need a fresh one here so we can access it).
    G_phys, bc_individual, _deg, _topo_dbg = compute_physical_topology_metrics(monee_net)
    G_phys_cpa, bc_individual_cpa, _, _topo_dbg_cpa = (
        compute_physical_topology_metrics_cp_aware(monee_net)
    )
    G_phys_ex, bc_individual_ex, _, _topo_dbg_ex = (
        compute_physical_topology_metrics_exergy_aware(monee_net, cfg)
    )
    ctx = CarrierPTDFContext(cfg)
    ctx.power_prebuild(monee_net)
    ctx.gas_prebuild(monee_net)
    ctx.heat_prebuild(monee_net)

    # CP branch ids to skip (already covered above)
    cp_branch_ids = set()
    for cp_type in (mm.GasToPower, mm.PowerToGas, mm.PowerToHeatHG, mm.GasToHeatHG):
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
        throughput = _loading_proxy(flow0, limit, cfg)

        bc_avg = float(np.mean([
            bc_individual.get(br.from_node_id, 0.0),
            bc_individual.get(br.to_node_id, 0.0),
        ]))
        topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)
        bc_avg_cpa = float(np.mean([
            bc_individual_cpa.get(br.from_node_id, 0.0),
            bc_individual_cpa.get(br.to_node_id, 0.0),
        ]))
        topo_factor_cpa = (1.0 + cfg.TOPO_ALPHA * bc_avg_cpa)
        bc_avg_ex = float(np.mean([
            bc_individual_ex.get(br.from_node_id, 0.0),
            bc_individual_ex.get(br.to_node_id, 0.0),
        ]))
        topo_factor_ex = (1.0 + cfg.TOPO_ALPHA * bc_avg_ex)

        mean_s, max_s, agg_s, reliable = _branch_lodf_stress(
            monee_net, ctx, "power", br.from_node_id, br.to_node_id, cfg
        )
        total_stress = cfg.W_POWER * agg_s
        score = _apply_ablations(
            cfg, throughput, total_stress, topo_factor, 1.0, is_cp=False
        )
        score_cpa = _apply_ablations(
            cfg, throughput, total_stress, topo_factor_cpa, 1.0, is_cp=False
        )
        score_ex = _apply_ablations(
            cfg, throughput, total_stress, topo_factor_ex, 1.0, is_cp=False
        )

        row = dict(
            cp_id=str(branch_id),
            cp_type="PowerLine",
            is_cp=False,
            p_fail=p_fail,
            throughput=throughput,
            topo_bc=bc_avg,
            topo_factor=topo_factor,
            topo_bc_cp_aware=bc_avg_cpa,
            topo_factor_cp_aware=topo_factor_cpa,
            topo_bc_exergy=bc_avg_ex,
            topo_factor_exergy=topo_factor_ex,
            total_stress=total_stress,
            score=score,
            score_cp_aware=score_cpa,
            score_exergy=score_ex,
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
            input_adequacy=1.0,
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
        limit, flow0 = _gas_pipe_limit_and_flow(monee_net, pipe_id, gg)
        throughput = _loading_proxy(flow0, limit, cfg)

        bc_avg = float(np.mean([
            bc_individual.get(br.from_node_id, 0.0),
            bc_individual.get(br.to_node_id, 0.0),
        ]))
        topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)
        bc_avg_cpa = float(np.mean([
            bc_individual_cpa.get(br.from_node_id, 0.0),
            bc_individual_cpa.get(br.to_node_id, 0.0),
        ]))
        topo_factor_cpa = (1.0 + cfg.TOPO_ALPHA * bc_avg_cpa)
        bc_avg_ex = float(np.mean([
            bc_individual_ex.get(br.from_node_id, 0.0),
            bc_individual_ex.get(br.to_node_id, 0.0),
        ]))
        topo_factor_ex = (1.0 + cfg.TOPO_ALPHA * bc_avg_ex)

        mean_s, max_s, agg_s, reliable = _branch_lodf_stress(
            monee_net, ctx, "gas", br.from_node_id, br.to_node_id, cfg
        )
        total_stress = cfg.W_GAS * agg_s
        score = _apply_ablations(
            cfg, throughput, total_stress, topo_factor, 1.0, is_cp=False
        )
        score_cpa = _apply_ablations(
            cfg, throughput, total_stress, topo_factor_cpa, 1.0, is_cp=False
        )
        score_ex = _apply_ablations(
            cfg, throughput, total_stress, topo_factor_ex, 1.0, is_cp=False
        )

        row = dict(
            cp_id=str(pipe_id),
            cp_type="GasPipe",
            is_cp=False,
            p_fail=p_fail,
            throughput=throughput,
            topo_bc=bc_avg,
            topo_factor=topo_factor,
            topo_bc_cp_aware=bc_avg_cpa,
            topo_factor_cp_aware=topo_factor_cpa,
            topo_bc_exergy=bc_avg_ex,
            topo_factor_exergy=topo_factor_ex,
            total_stress=total_stress,
            score=score,
            score_cp_aware=score_cpa,
            score_exergy=score_ex,
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
            input_adequacy=1.0,
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

        limit, flow0 = _heat_pipe_limit_and_flow(monee_net, pipe_id)
        throughput = _loading_proxy(flow0, limit, cfg)

        bc_avg = float(np.mean([
            bc_individual.get(br.from_node_id, 0.0),
            bc_individual.get(br.to_node_id, 0.0),
        ]))
        topo_factor = (1.0 + cfg.TOPO_ALPHA * bc_avg)
        bc_avg_cpa = float(np.mean([
            bc_individual_cpa.get(br.from_node_id, 0.0),
            bc_individual_cpa.get(br.to_node_id, 0.0),
        ]))
        topo_factor_cpa = (1.0 + cfg.TOPO_ALPHA * bc_avg_cpa)
        bc_avg_ex = float(np.mean([
            bc_individual_ex.get(br.from_node_id, 0.0),
            bc_individual_ex.get(br.to_node_id, 0.0),
        ]))
        topo_factor_ex = (1.0 + cfg.TOPO_ALPHA * bc_avg_ex)

        mean_s, max_s, agg_s, reliable = _branch_lodf_stress(
            monee_net, ctx, "heat", br.from_node_id, br.to_node_id, cfg
        )
        # (B) Slack-distance prefactor on heat stress (averaged over the pipe
        # endpoints, since either endpoint failing splits the supply path).
        rem = 0.5 * (
            _heat_remoteness_factor(ctx, br.from_node_id, cfg)
            + _heat_remoteness_factor(ctx, br.to_node_id, cfg)
        )
        total_stress = cfg.W_HEAT * agg_s * rem
        score = _apply_ablations(
            cfg, throughput, total_stress, topo_factor, 1.0, is_cp=False
        )
        score_cpa = _apply_ablations(
            cfg, throughput, total_stress, topo_factor_cpa, 1.0, is_cp=False
        )
        score_ex = _apply_ablations(
            cfg, throughput, total_stress, topo_factor_ex, 1.0, is_cp=False
        )

        row = dict(
            cp_id=str(pipe_id),
            cp_type=cp_type_str,
            is_cp=False,
            p_fail=p_fail,
            throughput=throughput,
            topo_bc=bc_avg,
            topo_factor=topo_factor,
            topo_bc_cp_aware=bc_avg_cpa,
            topo_factor_cp_aware=topo_factor_cpa,
            topo_bc_exergy=bc_avg_ex,
            topo_factor_exergy=topo_factor_ex,
            total_stress=total_stress,
            score=score,
            score_cp_aware=score_cpa,
            score_exergy=score_ex,
            reliable=reliable,
            power_node_id=None, power_reliable=None,
            power_stress_mean=0.0, power_stress_max=0.0, power_stress=0.0,
            gas_node_id=None, gas_reliable=None,
            gas_stress_mean=0.0, gas_stress_max=0.0, gas_stress=0.0,
            heat_node_id=br.from_node_id,
            heat_reliable=reliable,
            heat_stress_mean=mean_s,
            heat_stress_max=max_s,
            heat_stress=agg_s * rem,
            input_adequacy=1.0,
        )
        rows_non_cp.append(row)

    # ``df_cp`` already carries the ``is_cp`` column (added above), so the
    # empty-non-CP fallback must NOT re-append it — otherwise the resulting
    # frame has two ``is_cp`` columns and downstream ``concat`` produces
    # ambiguous selections.
    df_non_cp = pd.DataFrame(rows_non_cp) if rows_non_cp else pd.DataFrame(
        columns=list(df_cp.columns)
    )

    df_all = pd.concat([df_cp, df_non_cp], ignore_index=True, sort=False)
    df_all = normalize_within_carrier(df_all, cfg)
    df_all = df_all.sort_values("score", ascending=False).reset_index(drop=True)

    # ---- Local metric ----
    # Uses only information locally available to the device (1-hop at most):
    #   loading            - |flow| / limit: own utilisation (local measurement)
    #                        For CPs without a meaningful limit: rated throughput proxy.
    #   n_critical_nbrs    - neighbours with degree ≤ 2 (1-hop info only):
    #                        counts neighbours for whom this component is on their
    #                        only or critical path; computable without global routing.
    #   carrier_coupling   - number of distinct energy carriers this component connects
    #                        (observable from own port configuration, no network traversal).
    #
    # local_score = loading × (1 + n_critical_nbrs) × carrier_coupling
    #
    # Rationale:
    #   loading           → how much energy actually flows through me right now
    #   n_critical_nbrs   → how many neighbours are stranded if I fail
    #   carrier_coupling  → how many domains my failure disrupts simultaneously

    # Pre-build graph adjacency for 1-hop neighbour degree lookups. ``_deg``
    # (from ``compute_physical_topology_metrics`` above) is the degree dict on
    # the same node set, so we reuse it for the critical-neighbour count.
    G_local = nx.Graph(monee_net._network_internal)

    def _connected_node_ids(row):
        """All endpoint network node IDs for this component.

        For CPs we read the ``*_node_id`` columns directly; for non-CP branches
        we additionally parse ``cp_id`` (a stringified ``(u, v, k)`` tuple) and
        add the second endpoint, because the row only stores ``from_node_id``
        on its single carrier column. Without this, centrality scores are
        asymmetric and miss one endpoint.
        """
        nids = []
        for col in ("power_node_id", "gas_node_id", "heat_node_id"):
            nid = row.get(col)
            if nid is not None and nid == nid and nid not in nids:
                nids.append(nid)
        if not bool(row.get("is_cp", False)):
            try:
                import ast
                tup = ast.literal_eval(str(row.get("cp_id", "")))
                br = monee_net.branch_by_id(tup)
                for nid in (br.from_node_id, br.to_node_id):
                    if nid not in nids:
                        nids.append(nid)
            except Exception:
                pass
        if not nids:
            cp_type = row.get("cp_type", "")
            cp_id = row.get("cp_id")
            cp_cls = _COMPOUND_CP_CLASSES.get(cp_type)
            if cp_cls is not None:
                for cp in monee_net.compounds_by_type(cp_cls):
                    if cp.id == cp_id:
                        nids = list(cp.connected_to.values())
                        break
        return nids

    def _n_critical_nbrs(row):
        """Count of distinct 1-hop neighbours whose degree is ≤ 2 (critically
        dependent). We deduplicate across endpoints so a branch (u, v) whose
        two endpoints share neighbours doesn't double-count them.
        """
        nids = [n for n in _connected_node_ids(row)
                if n is not None and n == n and G_local.has_node(n)]
        endpoint_set = set(nids)
        critical_nbrs = set()
        for nid in nids:
            for nbr in G_local.neighbors(nid):
                if nbr in endpoint_set:
                    continue
                if _deg.get(nbr, 1) <= 2:
                    critical_nbrs.add(nbr)
        return float(len(critical_nbrs))

    def _carrier_coupling(row):
        """Number of distinct energy carriers this component connects (1-3)."""
        cp_type = row.get("cp_type", "")
        if cp_type in ("CHP", "CHPHG"):
            return 2.0  # power + heat (matches existing CHP convention)
        if cp_type in ("PowerToHeat", "PowerToHeatHG"):
            return 2.0  # power + heat
        if cp_type == "GasToHeatHG":
            return 2.0  # gas + heat
        if cp_type in ("GasToPower", "PowerToGas"):
            return 2.0  # gas + power
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

        if cp_type in ALL_CP_LABELS:
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
    # CP loading is a rated-capacity proxy (pu), not a [0,1] utilisation like the
    # branch |flow|/limit ratio. Rescale CP rows by the max CP proxy so the
    # local/self scores compare CPs and branches on a common loading axis.
    _is_cp_row = df_all["cp_type"].isin(ALL_CP_LABELS)
    _cp_load_max = df_all.loc[_is_cp_row, "loading"].max() if _is_cp_row.any() else 0.0
    if np.isfinite(_cp_load_max) and _cp_load_max > 0:
        df_all.loc[_is_cp_row, "loading"] = df_all.loc[_is_cp_row, "loading"] / _cp_load_max
    df_all["n_critical_nbrs"] = df_all.apply(_n_critical_nbrs, axis=1)
    df_all["carrier_coupling"] = df_all.apply(_carrier_coupling, axis=1)
    df_all["local_score"] = (
        df_all["loading"]
        * (1.0 + df_all["n_critical_nbrs"])
        * df_all["carrier_coupling"]
    )

    # ---- Self-only metric (zero-hop) ----
    # Uses exclusively the component's own observable state — no network traversal:
    #   loading          - own utilisation |flow| / limit
    #   carrier_coupling - number of carriers connected (readable from own ports)
    # self_score = loading × carrier_coupling
    df_all["self_score"] = (
        df_all["loading"]
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
        node_ids = _connected_node_ids(row)
        if node_ids:
            return float(np.mean([katz_individual.get(n, 0.0) for n in node_ids]))
        return 0.0

    df_all["katz_score"] = df_all.apply(_katz_for_row, axis=1)

    # ---- Closeness vitality (physical graph, same weights as BC and Katz) ----
    # vitality(v) = W(G \ v) - W(G): how much total pairwise distance increases
    # when v is removed (higher = more critical). Captures structural
    # indispensability, not just centrality. nx returns the negated form
    # W(G) - W(G\v), so we flip the sign below.
    try:
        # On a disconnected graph the Wiener index is already infinite, so a
        # single closeness_vitality call would return NaN for every node and
        # collapse the whole metric to zero. Compute per connected component so
        # vitality stays meaningful on the (typically disconnected) MES graph.
        if G_phys.number_of_nodes() and not nx.is_connected(G_phys):
            vitality_individual = {}
            for comp in nx.connected_components(G_phys):
                vitality_individual.update(
                    nx.closeness_vitality(G_phys.subgraph(comp).copy(), weight="weight")
                )
        else:
            vitality_individual = nx.closeness_vitality(G_phys, weight="weight")
        # nx.closeness_vitality returns W(G) − W(G\v): NEGATIVE (down to −inf)
        # for critical nodes, positive for peripheral ones. Negate so the score
        # is oriented "higher = more critical" like every other metric — i.e.
        # W(G\v) − W(G), the increase in total pairwise distance on removal.
        vitality_individual = {n: -v for n, v in vitality_individual.items()}
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
        node_ids = _connected_node_ids(row)
        if node_ids:
            return float(np.mean([vitality_individual.get(n, 0.0) for n in node_ids]))
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

            cp_cls = _COMPOUND_CP_CLASSES.get(cp_type)
            if cp_cls is not None:
                for cp in monee_net.compounds_by_type(cp_cls):
                    if cp.id == cp_id:
                        return float(_group_bc(_G_stress, cp, weight="stress_weight"))
                return 0.0

            # All other rows (branch CPs, non-CP branches): average BC over
            # the resolved endpoint set, which now includes both endpoints of
            # non-CP branches via cp_id parsing.
            nids = _connected_node_ids(row)
            if nids:
                return float(np.mean([stress_bc_nodes.get(n, 0.0) for n in nids]))
            return 0.0

        df_all["stress_bc"] = df_all.apply(_stress_bc_for_row, axis=1)
        df_all["stress_topo_factor"] = 1.0 + cfg.TOPO_ALPHA * df_all["stress_bc"]

        # Recompose stress_score via the same ablation-aware path used for
        # ``score`` instead of rescaling ``score`` by the topo-factor ratio.
        # The latter silently produced ``stress_topo_factor / topo_factor``
        # when ``ABLATE_TOPO=True`` (because the topo factor inside ``score``
        # is forced to 1.0 there), instead of the intended stress topo factor.
        def _stress_score_for_row(row):
            is_cp = bool(row.get("is_cp", False))
            return _apply_ablations(
                cfg,
                float(row.get("throughput", 1.0)),
                   float(row.get("total_stress", 0.0)),
                float(row.get("stress_topo_factor", 1.0)),
                float(row.get("input_adequacy", 1.0)),
                is_cp=is_cp,
            )

        df_all["stress_score"] = df_all.apply(_stress_score_for_row, axis=1)
    except Exception as e:
        df_all["stress_bc"] = 0.0
        df_all["stress_topo_factor"] = 1.0
        df_all["stress_score"] = df_all["score"]
        print(f"[warn] stress topology failed: {e}")

    # Diagnostic: surface any non-finite values still present in metric columns
    # so the caller can trace them back to the originating component (otherwise
    # they propagate silently to ranks / Spearman / NDCG and bomb downstream).
    metric_cols = [
        "score", "total_stress", "throughput", "topo_factor", "topo_bc",
        "stress_bc", "stress_score", "local_score", "self_score",
        "katz_score", "vitality_score", "loading", "n_critical_nbrs",
        "carrier_coupling",
    ]
    bad_mask = df_all[[c for c in metric_cols if c in df_all.columns]].apply(
        lambda s: ~np.isfinite(s.astype(float)), axis=0
    ).any(axis=1)
    if bad_mask.any():
        bad_rows = df_all.loc[bad_mask, ["cp_id", "cp_type"] + [c for c in metric_cols if c in df_all.columns]]
        per_col_bad = {
            c: int((~np.isfinite(df_all[c].astype(float))).sum())
            for c in metric_cols if c in df_all.columns
        }
        per_col_bad = {c: n for c, n in per_col_bad.items() if n > 0}
        print(
            f"[warn] mes_all_components_metric: {int(bad_mask.sum())}/{len(df_all)} "
            f"rows contain non-finite metric values. Per-column NaN/inf counts: {per_col_bad}"
        )
        print(f"[warn] first few offending rows:\n{bad_rows.head(8).to_string(index=False)}")

    # ── Balanced composite (S1+C1+C2+C3) ──────────────────────────────────
    # Adds a parallel ``predicted_score_balanced`` column that applies four
    # corrections on top of the existing composite. See
    # ``attach_balanced_score`` for the formulas. The original
    # ``predicted_score`` is preserved unchanged for A/B comparison.
    df_all = attach_balanced_score(df_all, monee_net, cfg)

    # ── Per-carrier atomic predictors (option 3) ─────────────────────────
    # The single composite ``predicted_score`` necessarily competes with
    # itself across carriers (S1 helped gas but hurt heat in the ablation
    # diagnostics). Emit one prediction per carrier so each can be ranked
    # against its own per-carrier shed without cross-carrier mixing.
    df_all = attach_per_carrier_scores(df_all, cfg)

    if cfg.RETURN_DEBUG:
        return df_all, df_debug
    return df_all


# =============================================================================
# Per-carrier atomic predictors (option 3 — separate predictions per sector)
# =============================================================================


_CARRIER_GROUP_BY_TYPE = {
    "PowerLine": "power", "GenericPowerBranch": "power", "Trafo": "power",
    "GasPipe": "gas",
    "WaterPipe": "heat", "HeatExchanger": "heat",
}


def normalize_within_carrier(
    df_all: pd.DataFrame, cfg: CPMetricConfig
) -> pd.DataFrame:
    """Within-carrier median normalisation of the composite's factors.

    ``throughput`` and the stress columns are divided by their carrier
    group's positive median (power/gas/heat branches via cp_type; CPs form
    their own group), and the composed score columns are rescaled by the
    same divisors, so ``score = throughput · total_stress · topo
    (· adequacy)`` and ``Σ_c predicted_<c> = score`` keep holding exactly.

    Motivation: even on a common per-unit basis the factors live on
    incomparable per-carrier regimes (margin tightness, utilisation), so
    the raw composite ranked essentially by carrier membership (PowerLine
    median score ~2.4 vs GasPipe ~0.0004 on simbench LV). After this pass
    a score reads "criticality relative to a typical component of its
    class" — the cross-carrier comparable quantity. Within-group rankings
    are unchanged (constant positive factor per group); all per-carrier
    stress columns share the row's ``total_stress`` divisor so the
    decomposition identity survives.
    """
    if df_all is None or df_all.empty:
        return df_all

    grp = df_all["cp_type"].map(_CARRIER_GROUP_BY_TYPE) \
        if "cp_type" in df_all.columns else pd.Series(None, index=df_all.index)
    if "is_cp" in df_all.columns:
        grp = grp.where(~df_all["is_cp"].astype(bool), "cp")
    grp = grp.fillna("other")

    def _pos_median(s: pd.Series) -> float:
        v = s.astype(float)
        v = v[np.isfinite(v) & (v > 0)]
        return float(v.median()) if len(v) else 1.0

    ones = pd.Series(1.0, index=df_all.index)
    med_thr = (
        df_all.groupby(grp)["throughput"].transform(_pos_median)
        if "throughput" in df_all.columns else ones
    )
    med_str = (
        df_all.groupby(grp)["total_stress"].transform(_pos_median)
        if "total_stress" in df_all.columns else ones
    )

    if "throughput" in df_all.columns:
        df_all["throughput"] = df_all["throughput"].astype(float) / med_thr
    for col in ("total_stress", "power_stress", "gas_stress", "heat_stress"):
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(float) / med_str

    divisor = ones.copy()
    if not cfg.ABLATE_THROUGHPUT:
        divisor = divisor * med_thr
    if not cfg.ABLATE_STRESS:
        divisor = divisor * med_str
    for col in ("score", "score_cp_aware", "score_exergy"):
        if col in df_all.columns:
            df_all[col] = df_all[col].astype(float) / divisor
    return df_all


def attach_per_carrier_scores(
    df: pd.DataFrame, cfg: CPMetricConfig,
) -> pd.DataFrame:
    """Emit one carrier-specific score per row.

    Each ``predicted_<carrier>`` is the carrier's own contribution to the
    composite ``predicted_score`` — i.e. ``throughput × W_carrier ×
    stress_carrier × topo_factor × input_adequacy``, exactly the per-carrier
    slice of the existing composite. By construction:

      * For a non-CP branch the score is non-zero only on its own carrier
        (the other carriers' stress columns are 0).
      * For a coupling point all carriers it touches receive a positive
        score; the others stay 0.
      * Summing the three ``predicted_<carrier>`` values reproduces the
        original ``score`` column (modulo ablation-flag scalings).

    Three additional column families are emitted in parallel using the
    CP-aware, exergy and balanced topology/multiplier stacks so callers
    can choose which weighting to evaluate per carrier:

      * ``predicted_<carrier>_cp_aware`` — uses ``topo_factor_cp_aware``.
      * ``predicted_<carrier>_exergy``   — uses ``topo_factor_exergy``.
      * ``predicted_<carrier>_balanced`` — applies S1 stress normalisation
        plus the C1/C2/C3 multipliers (per-carrier this time, so cross-
        carrier mixing no longer destroys within-carrier signal).
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    n = len(out)
    thr = out.get("throughput", pd.Series(np.ones(n))).astype(float)
    adq = out.get("input_adequacy", pd.Series(np.ones(n))).astype(float)
    topo = out.get("topo_factor", pd.Series(np.ones(n))).astype(float)
    topo_cpa = out.get("topo_factor_cp_aware", topo).astype(float)
    topo_ex = out.get("topo_factor_exergy", topo).astype(float)
    ext = out.get("ext_headroom_mult", pd.Series(np.ones(n))).astype(float)
    dem = out.get("demand_coupling_mult", pd.Series(np.ones(n))).astype(float)
    sub = out.get("substitutability_mult", pd.Series(np.ones(n))).astype(float)

    carrier_weights = {
        "power": float(cfg.W_POWER),
        "gas":   float(cfg.W_GAS),
        "heat":  float(cfg.W_HEAT),
    }

    # Per-carrier S1 medians (across rows with non-zero stress on that carrier).
    medians: Dict[str, float] = {}
    for carrier in ("power", "gas", "heat"):
        col = f"{carrier}_stress"
        if col not in out.columns:
            medians[carrier] = 1.0
            continue
        v = out[col].astype(float)
        v_pos = v[v > 0]
        medians[carrier] = float(v_pos.median()) if len(v_pos) else 1.0

    zero = np.zeros(n, dtype=float)
    for carrier, w in carrier_weights.items():
        stress_col = f"{carrier}_stress"
        if stress_col not in out.columns:
            out[f"predicted_{carrier}"] = zero.copy()
            out[f"predicted_{carrier}_cp_aware"] = zero.copy()
            out[f"predicted_{carrier}_exergy"] = zero.copy()
            out[f"predicted_{carrier}_balanced"] = zero.copy()
            continue
        stress = out[stress_col].astype(float)
        denom = max(medians[carrier], 1e-12)
        stress_norm = stress / denom

        base = (thr * w * stress * adq).values
        out[f"predicted_{carrier}"] = base * topo.values
        out[f"predicted_{carrier}_cp_aware"] = base * topo_cpa.values
        out[f"predicted_{carrier}_exergy"] = base * topo_ex.values
        # Balanced per-carrier predictor: S1 (median norm) + C1/C2/C3 mults.
        # Stays inside the carrier so cross-carrier mixing can't destroy
        # within-carrier signal (unlike the composite ``predicted_score
        # _balanced``).
        out[f"predicted_{carrier}_balanced"] = (
            thr * w * stress_norm * adq * topo
            * ext * dem * sub
        ).values

    return out


# =============================================================================
# Balanced composite — S1 + C1 + C2 + C3 fixes
# =============================================================================


def _ext_capacity_per_carrier_mw(monee_net) -> Dict[str, float]:
    """Return ``{carrier → max |ext-grid throughput| in MW}``.

    Sums absolute upper/lower bounds across every ExtPowerGrid / ExtHydrGrid
    on each carrier. Falls back to 0 when a grid lacks a finite bound (which
    we read as "unbounded → don't credit it as headroom"; conservative).
    """
    cap: Dict[str, float] = {"power": 0.0, "gas": 0.0, "heat": 0.0}

    def _bound_mag(var):
        try:
            lo = mm.lower(var) if hasattr(var, "min") else None
            hi = mm.upper(var) if hasattr(var, "max") else None
            lo = float(lo) if (lo is not None and np.isfinite(lo)) else 0.0
            hi = float(hi) if (hi is not None and np.isfinite(hi)) else 0.0
            return max(abs(lo), abs(hi))
        except Exception:
            return 0.0

    # ExtPowerGrid → power, p_mw [MW] directly.
    try:
        for c in monee_net.childs_by_type(mm.ExtPowerGrid):
            cap["power"] += _bound_mag(c.model.p_mw)
    except Exception:
        pass

    # ExtHydrGrid → either gas (kg/s × HHV × 3.6 → MW) or heat (water grid).
    try:
        for c in monee_net.childs_by_type(mm.ExtHydrGrid):
            grid = getattr(c, "grid", None)
            if grid is None:
                continue
            mag = _bound_mag(c.model.mass_flow_kgs)
            if hasattr(grid, "higher_heating_value_kwh_per_kg"):
                cap["gas"] += mag * 3.6 * float(grid.higher_heating_value_kwh_per_kg)
            else:
                # Heat slack: convert ṁ to MW via cp · ΔT, using cfg-style
                # defaults so we don't need a config object here.
                cp_water = 4186.0          # J/(kg·K)
                dT = 30.0                  # K (typical supply-return)
                cap["heat"] += mag * cp_water * dT / 1e6
    except Exception:
        pass
    return cap


def _total_demand_per_carrier_mw(monee_net) -> Dict[str, float]:
    """Return ``{carrier → total active load in MW}`` (mirrors the per-
    carrier accounting of ``CascadingModel._max_load_shedding``)."""
    passive_hx = getattr(mm, "PassiveHeatExchangerLoad", mm.HeatExchangerLoad)
    power = 0.0
    heat = 0.0
    gas = 0.0
    for c in monee_net.childs:
        m = c.model
        if not getattr(c, "active", True) or getattr(c, "ignored", False):
            continue
        try:
            if isinstance(m, mm.PowerLoad):
                power += float(mm.upper(m.p_mw) or 0.0)
            elif isinstance(m, mm.HeatLoad):
                heat += float(mm.upper(m.q_mw_heat) or 0.0)
            elif isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
                heat += abs(float(mm.upper(getattr(m, "q_mw_set", m.q_mw)) or 0.0))
            elif isinstance(m, mm.Sink):
                grid = getattr(c, "grid", None)
                if grid is not None and hasattr(grid, "higher_heating_value_kwh_per_kg"):
                    # mass_flow_kgs is stored negative (consumption); abs for demand.
                    gas += abs(float(mm.upper(m.mass_flow_kgs) or 0.0)) * 3.6 \
                        * float(grid.higher_heating_value_kwh_per_kg)
        except Exception:
            continue
    for b in monee_net.branches:
        m = b.model
        if not getattr(b, "active", True) or getattr(b, "ignored", False):
            continue
        if isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
            try:
                heat += abs(float(mm.upper(getattr(m, "q_mw_set", m.q_mw)) or 0.0))
            except Exception:
                continue
    return {"power": power, "heat": heat, "gas": gas}


def _per_carrier_throughput_mw(row, monee_net) -> Dict[str, float]:
    """Best-effort MW throughput per carrier this component is on."""
    out: Dict[str, float] = {}
    try:
        cp_type = str(row.get("cp_type", ""))
        sn_mva = _system_sn_mva(monee_net)
        # CP rows store the throughput proxy in pu of sn_mva; back-transform.
        thr_pu = float(row.get("throughput", 0.0))
        if cp_type in _CP_IO_SPEC:
            mw_total = thr_pu * sn_mva
            in_carrier, outputs = _CP_IO_SPEC[cp_type]
            out[in_carrier] = max(out.get(in_carrier, 0.0), mw_total)
            for out_carrier, _attr in outputs:
                out[out_carrier] = max(out.get(out_carrier, 0.0), mw_total)
            return out
        # Non-CP rows: use loading × limit on the single carrier.
        if cp_type == "PowerLine":
            out["power"] = thr_pu * sn_mva
        elif cp_type == "GasPipe":
            out["gas"] = thr_pu * sn_mva
        elif cp_type in ("WaterPipe", "HeatExchanger"):
            out["heat"] = thr_pu * sn_mva
    except Exception:
        pass
    return out


def attach_balanced_score(
    df: pd.DataFrame, monee_net, cfg: CPMetricConfig,
) -> pd.DataFrame:
    """Compose ``predicted_score_balanced`` from four corrections:

    * **S1** — per-carrier stress median normalisation. Each row's
      per-carrier stress (``power_stress`` / ``gas_stress`` /
      ``heat_stress``) is divided by the median of that carrier's
      non-zero stresses across all components, putting the three
      carriers on a comparable [0, ~few] scale before the W-weighting
      is reapplied.
    * **C1** — ext-grid headroom multiplier. Components on carriers
      where the ext grid can absorb their full throughput get
      down-weighted; under-covered carriers stay full-weight. Stored
      in ``ext_headroom_mult`` ∈ [0.05, 1].
    * **C2** — demand-coupling multiplier. Components are weighted by
      the share of total system demand carried by their carrier(s),
      capturing "this carrier matters more for shed than that one".
      Stored in ``demand_coupling_mult`` ∈ (0, 1].
    * **C3** — substitutability multiplier. Uses
      ``cp_metric_structural.compute_cp_substitutability`` so a CP
      that holds 100 % of its type's capacity stays full-weight while
      a CP that's one of N equal alternatives is reduced toward 1/N.
      Non-CP rows default to 1.0. Stored in ``substitutability_mult``.

    Composite::

        total_stress_balanced
            = W_p · power_stress/median(power_stress)
            + W_g · gas_stress  /median(gas_stress)
            + W_h · heat_stress /median(heat_stress)
        predicted_score_balanced
            = throughput · total_stress_balanced
              · topo_factor · input_adequacy
              · ext_headroom_mult · demand_coupling_mult · substitutability_mult

    All four multipliers are also persisted as separate columns so the
    eval can ablate any one in isolation.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    # ── S1: per-carrier stress median normalisation ──────────────────────
    medians: Dict[str, float] = {}
    for carrier in ("power", "gas", "heat"):
        col = f"{carrier}_stress"
        if col not in out.columns:
            medians[carrier] = 1.0
            continue
        vals = out[col].astype(float)
        vals_pos = vals[vals > 0]
        medians[carrier] = float(vals_pos.median()) if len(vals_pos) else 1.0

    weights = {"power": cfg.W_POWER, "gas": cfg.W_GAS, "heat": cfg.W_HEAT}
    total_stress_balanced = np.zeros(len(out), dtype=float)
    for carrier in ("power", "gas", "heat"):
        col = f"{carrier}_stress"
        if col not in out.columns:
            continue
        denom = max(medians[carrier], 1e-12)
        norm = out[col].astype(float) / denom
        total_stress_balanced += float(weights[carrier]) * norm.values
    out["total_stress_balanced"] = total_stress_balanced
    out["_stress_medians_debug"] = ";".join(
        f"{c}={medians[c]:.4g}" for c in ("power", "gas", "heat")
    )

    # ── C1: ext-grid headroom multiplier ─────────────────────────────────
    ext_cap = _ext_capacity_per_carrier_mw(monee_net)
    demand = _total_demand_per_carrier_mw(monee_net)
    cov: Dict[str, float] = {}
    for c in ("power", "gas", "heat"):
        d = max(demand.get(c, 0.0), 1e-9)
        cov[c] = float(np.clip(ext_cap.get(c, 0.0) / d, 0.0, 1.0))

    # For each row pick the carriers it lives on, then take min coverage
    # (the weakest link decides whether the ext grid can really compensate).
    ext_mult = np.ones(len(out), dtype=float)
    for i, row in out.iterrows():
        carriers = list(_per_carrier_throughput_mw(row, monee_net).keys())
        if not carriers:
            continue
        min_cov = min(cov[c] for c in carriers if c in cov) if carriers else 0.0
        ext_mult[out.index.get_loc(i)] = max(0.05, 1.0 - min_cov)
    out["ext_headroom_mult"] = ext_mult

    # ── C2: demand-coupling multiplier ───────────────────────────────────
    total_d = sum(demand.values()) or 1.0
    demand_share = {c: demand.get(c, 0.0) / total_d for c in ("power", "gas", "heat")}
    dem_mult = np.zeros(len(out), dtype=float)
    for i, row in out.iterrows():
        carriers = list(_per_carrier_throughput_mw(row, monee_net).keys())
        if not carriers:
            dem_mult[out.index.get_loc(i)] = 1.0 / len(demand_share) if demand_share else 1.0
            continue
        # Sum-of-shares for CPs (multi-carrier exposure), capped at 1.
        share = sum(demand_share.get(c, 0.0) for c in carriers)
        dem_mult[out.index.get_loc(i)] = float(np.clip(share, 1e-3, 1.0))
    out["demand_coupling_mult"] = dem_mult

    # ── C3: substitutability multiplier ──────────────────────────────────
    try:
        import cp_metric_structural as cms
        sub_df = cms.compute_cp_substitutability(monee_net)
    except Exception:
        sub_df = pd.DataFrame(columns=["cp_id", "substitutability"])
    if not sub_df.empty:
        sub_map = dict(
            zip(sub_df["cp_id"].astype(str),
                sub_df["substitutability"].astype(float))
        )
    else:
        sub_map = {}
    # compute_cp_substitutability returns ``cp_id`` as ``compound:{id}`` for
    # compounds and ``f"{from}→{to}"`` for branch CPs. mes_*_metric stores
    # cp.id (an int) for compounds in the df, so normalise on the fly.
    def _sub_for_row(row):
        cp_id = str(row.get("cp_id", ""))
        cp_type = str(row.get("cp_type", ""))
        if cp_type in _CP_COMPOUND_LABEL.values() and not cp_id.startswith("compound:"):
            cp_id = f"compound:{cp_id}"
        # Default 1.0: non-CP rows are not penalised (no concept of a same-
        # type alternative for a plain branch).
        return float(sub_map.get(cp_id, 1.0))
    out["substitutability_mult"] = out.apply(_sub_for_row, axis=1).values

    # ── Compose ──────────────────────────────────────────────────────────
    throughput = out.get("throughput", pd.Series(np.ones(len(out)))).astype(float).values
    topo_factor = out.get("topo_factor", pd.Series(np.ones(len(out)))).astype(float).values
    adequacy = out.get("input_adequacy", pd.Series(np.ones(len(out)))).astype(float).values
    out["predicted_score_balanced"] = (
        throughput
        * out["total_stress_balanced"].astype(float).values
        * topo_factor
        * adequacy
        * out["ext_headroom_mult"].astype(float).values
        * out["demand_coupling_mult"].astype(float).values
        * out["substitutability_mult"].astype(float).values
    )
    return out


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
