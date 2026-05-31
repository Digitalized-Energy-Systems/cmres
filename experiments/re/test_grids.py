"""
Test multi-energy system (MES) networks for resilience simulation research.

Three grids of increasing size and coupling diversity:

  Grid 1 – Urban residential district  (20 kV / gas / heat, high CP density)
  Grid 2 – Industrial energy hub        (110 kV / gas only, gas-backup focus)
  Grid 3 – Regional integrated MES      (120 kV / gas / heat, ring, all CP types)

Each ``create_*`` function returns a monee Network ready for energy-flow or
resilience simulation.  Companion ``make_*_timeseries`` functions return a
matching TimeseriesData with sinusoidal demand profiles.

Physical parameters are chosen to match real infrastructure:
  - 20 kV cable:     r = x = 3e-4 Ω/m, max_i_ka = 0.30 kA  → ~10 MVA/line
  - 110 kV OHL:      r = x = 7e-5 Ω/m, max_i_ka = 0.40 kA  → ~76 MVA/line
  - 120 kV cable:    r = x = 3e-4 Ω/m, max_i_ka = 0.30 kA  → ~62 MVA/line
  - Gas pipes:  d = 0.10–0.50 m, Weymouth-friction regime
  - Heat pipes: d = 0.10–0.20 m, Darcy-Weisbach regime

Coupling point parameters:
  - CHP:  η_el = 0.40, η_th = 0.40  (80 % total, typical micro-CHP)
  - G2P:  η    = 0.88               (gas turbine CCGT)
  - P2G:  η    = 0.70               (PEM electrolyser)
  - P2H:  η    = 0.95               (heat pump / resistance heater)
"""

from dataclasses import dataclass

import monee.model as mm
import monee.problem as mp
import numpy as np
import simbench
from monee import (
    PyomoSolver,
    TimeseriesData,
    run_energy_flow_optimization,
)
from monee.io.from_pandapower import from_pandapower_net
from monee.model.formulation import (
    MISOCP_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
)
from monee.network import (
    generate_supply_return_mes_based_on_power_net,
)


@dataclass
class MESContainer:
    """A network bundled with the ext-grid bounds the solver should use.

    ``include_coupling_points`` propagates to both the min-load-shedding
    optimisation (CPs become demand-weighted loads on their input carrier)
    and the performance metric (CP nameplate shed counted alongside
    end-user load shed). Set to True for the ``same_cap`` variants where
    CPs replace primary generation and therefore should themselves be
    treated as input-side demand.
    """

    network: mm.Network
    ext_grid_el_bounds: tuple = (-0.05, 0.05)
    ext_grid_gas_bounds: tuple = (-0.006, 0.006)
    ext_grid_heat_bounds: tuple = (0.0, 6.0)
    include_coupling_points: bool = False

# =============================================================================
# Helpers
# =============================================================================


def _sinusoidal_profile(
    n_steps: int,
    base: float,
    amplitude: float = 0.25,
    noise: float = 0.04,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Bell-shaped daily demand curve with small Gaussian noise.

    Profile peaks near the midpoint (noon) and troughs at the edges (night).
    Clipped to [50 %, 200 %] of base to avoid unphysical values.
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    profile = base * (1.0 + amplitude * np.sin(t - np.pi / 2))
    profile += rng.normal(0, noise * base, n_steps)
    return np.clip(profile, 0.5 * base, 2.0 * base)


def make_urban_district_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Demand profiles for Grid 1 (96 steps ≈ 24 h at 15-min resolution)."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id, "p_mw", _sinusoidal_profile(n_steps, base, amplitude=0.25, rng=rng)
        )
    for c in net.childs_by_type(mm.Sink):
        if c.grid.name == "gas":
            amp = 0.30
            base = float(mm.value(c.model.mass_flow))
            td.add_child_series(
                c.id,
                "mass_flow",
                _sinusoidal_profile(n_steps, base, amplitude=amp, rng=rng),
            )
    return td


def make_industrial_hub_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Flat industrial demand profiles with small daytime variation."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id,
            "p_mw",
            _sinusoidal_profile(n_steps, base, amplitude=0.15, noise=0.02, rng=rng),
        )
    for c in net.childs_by_type(mm.Sink):
        base = float(mm.value(c.model.mass_flow))
        td.add_child_series(
            c.id,
            "mass_flow",
            _sinusoidal_profile(n_steps, base, amplitude=0.20, noise=0.03, rng=rng),
        )
    return td


def make_regional_mes_timeseries(
    net: mm.Network, n_steps: int = 96, seed: int = 0
) -> TimeseriesData:
    """Sinusoidal demand profiles for Grid 3 with per-carrier amplitude tuning."""
    rng = np.random.default_rng(seed)
    td = TimeseriesData()
    for c in net.childs_by_type(mm.PowerLoad):
        base = float(mm.value(c.model.p_mw))
        td.add_child_series(
            c.id, "p_mw", _sinusoidal_profile(n_steps, base, amplitude=0.25, rng=rng)
        )
    for c in net.childs_by_type(mm.Sink):
        if c.grid.name == "gas":
            amp = 0.30
            base = float(mm.value(c.model.mass_flow))
            td.add_child_series(
                c.id,
                "mass_flow",
                _sinusoidal_profile(n_steps, base, amplitude=amp, rng=rng),
            )
    return td


# =============================================================================
# Convenience registry
# =============================================================================

def _balance_demand_for_cp_replacement(net) -> None:
    """For ``cp_capacity_invariant=True`` grids: reduce PowerLoad and gas
    Sink demand uniformly so the primary-side energy supply matches the
    no-CP baseline.

    Reasoning
    ---------
    In ``cp_capacity_invariant=True`` mode (``replace_primary_generation``
    inside ``generate_supply_return_mes_based_on_power_net``), a CP that
    produces output O_k replaces O_k of primary generation on the same
    carrier. The end-user demand on the OUTPUT carrier (heat / gas /
    power) is therefore still served. However, the CP also consumes
    O_k / η on its INPUT carrier — and the primary supply on that input
    side has *not* been reduced. Net effect: the system's primary energy
    consumption goes up by O_k · (1/η − 1) per CP.

    To keep the comparison across density variants fair (every grid sees
    the same primary energy footprint), we subtract each CP's rated
    INPUT draw from end-user demand on that carrier:

        ΔPowerLoad_total = − Σ_{CP draws power} (rated_input_mw)
        ΔSink_gas_total  = − Σ_{CP draws gas}   (rated_input_kgps)

    The reduction is applied uniformly across every demand object on the
    carrier (single scale factor on each `p_mw` / `mass_flow`), preserving
    the spatial distribution of demand. Heat demand is NOT touched (the
    chosen accounting policy targets only the "primary" carriers
    electricity + gas).

    Per-CP rated input draws
    ------------------------
    Indices: P=power, G=gas, H=heat. η_k is the conversion efficiency.

      CHP / CHPHG       I = G        rated_in_MW  = mass_flow_setpoint · 3.6 · HHV
      PowerToHeat       I = P        rated_in_MW  = load_p_mw  (= q_mw / η in the model)
      PowerToHeatHG     I = P        rated_in_MW  = heat_energy_mw / η
      GasToHeatHG       I = G        rated_in_MW  = heat_energy_mw / η
      GasToPower        I = G        rated_in_MW  = el_mw / η
      PowerToGas        I = P        rated_in_MW  = mass_flow_setpoint · 3.6 · HHV / η

    No-op when ``net`` has no CPs (e.g. density=0 or central=True).
    """
    # ``net.grids`` chokes on multi-grid CP control nodes (whose ``.grid`` is
    # a list and therefore unhashable inside the set comprehension in monee).
    # Walk the Junction nodes directly to find the gas grid object — it's the
    # only thing we need from there (``higher_heating_value``).
    gas_grid = None
    for n in net.nodes_by_type(mm.Junction):
        g = n.grid
        if g is not None and not isinstance(g, list) and getattr(g, "name", None) == "gas":
            gas_grid = g
            break
    hhv = float(getattr(gas_grid, "higher_heating_value", 15.3)) if gas_grid else 15.3

    p_in_mw = 0.0  # CP electricity draw [MW]
    g_in_mw = 0.0  # CP gas draw [MW]

    for cp in net.compounds:
        m = cp.model
        cn = type(m).__name__
        if cn in ("CHP", "CHPHG"):
            kgps = abs(float(getattr(m, "mass_flow_setpoint", 0.0) or 0.0))
            g_in_mw += kgps * 3.6 * hhv
        elif cn == "PowerToHeat":
            # load_p_mw = heat_energy_mw / η, set in PowerToHeat.__init__.
            p_in_mw += abs(float(getattr(m, "load_p_mw", 0.0) or 0.0))

    for br in net.branches:
        m = br.model
        cn = type(m).__name__
        if cn == "PowerToHeatHG":
            h = abs(float(getattr(m, "heat_energy_mw", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                p_in_mw += h / eta
        elif cn == "GasToHeatHG":
            h = abs(float(getattr(m, "heat_energy_mw", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                g_in_mw += h / eta
        elif cn == "GasToPower":
            p = abs(float(getattr(m, "el_mw", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                g_in_mw += p / eta
        elif cn == "PowerToGas":
            # mass_flow_setpoint stored on the branch (positive magnitude);
            # the constructor flips its sign onto gas_kgps internally.
            kgps_out = abs(float(getattr(m, "gas_kgps", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                p_in_mw += (kgps_out * 3.6 * hhv) / eta

    # Apply uniform scaling to PowerLoad (positive = consumption).
    if p_in_mw > 0:
        ploads = list(net.childs_by_type(mm.PowerLoad))
        total_pload = sum(float(mm.value(c.model.p_mw) or 0.0) for c in ploads)
        if total_pload > 0:
            if p_in_mw >= total_pload:
                print(
                    f"[same_cap] WARN: CP power draw {p_in_mw:.4f} MW ≥ total "
                    f"PowerLoad {total_pload:.4f} MW; setting PowerLoad to 1e-6 MW each."
                )
                for c in ploads:
                    c.model.p_mw = 1e-6
            else:
                scale = (total_pload - p_in_mw) / total_pload
                for c in ploads:
                    c.model.p_mw = float(mm.value(c.model.p_mw) or 0.0) * scale

    # Apply uniform scaling to gas-grid Sinks (positive = consumption).
    if g_in_mw > 0 and gas_grid is not None:
        sinks = [
            c for c in net.childs_by_type(mm.Sink)
            if c.grid is not None and getattr(c.grid, "name", None) == "gas"
        ]
        total_g_kgps = sum(float(mm.value(c.model.mass_flow) or 0.0) for c in sinks)
        total_g_mw = total_g_kgps * 3.6 * hhv
        if total_g_mw > 0:
            if g_in_mw >= total_g_mw:
                print(
                    f"[same_cap] WARN: CP gas draw {g_in_mw:.4f} MW ≥ total "
                    f"gas Sink {total_g_mw:.4f} MW; setting Sink mass_flow to 1e-9 kg/s each."
                )
                for c in sinks:
                    c.model.mass_flow = 1e-9
            else:
                scale = (total_g_mw - g_in_mw) / total_g_mw
                for c in sinks:
                    c.model.mass_flow = float(mm.value(c.model.mass_flow) or 0.0) * scale

    if p_in_mw > 0 or g_in_mw > 0:
        print(
            f"[same_cap] demand reduction: −{p_in_mw:.4f} MW power, "
            f"−{g_in_mw:.4f} MW gas (≈ −{g_in_mw / (3.6 * hhv):.6f} kg/s)"
        )


# =============================================================================
# Headroom helpers (used by the ``_relaxed`` variants)
# =============================================================================


def _gas_grid_and_hhv(net):
    """Find the gas Grid object and its HHV (kWh/kg) by walking junctions.

    ``net.grids`` chokes on multi-grid CP control nodes whose ``.grid`` is a
    list — same defensive walk as ``_balance_demand_for_cp_replacement``.
    """
    for n in net.nodes_by_type(mm.Junction):
        g = n.grid
        if g is not None and not isinstance(g, list) and getattr(g, "name", None) == "gas":
            return g, float(getattr(g, "higher_heating_value", 15.3))
    return None, 15.3


def _carrier_totals(net) -> dict:
    """Sum demand and in-grid generation for electricity and gas.

    Returns MW for both carriers (gas converted via ``kg/s × 3.6 × HHV``)
    plus the raw kg/s totals needed when writing back to monee Sources /
    Sinks. Headroom logic ignores heat per spec.
    """
    _gas_grid, hhv = _gas_grid_and_hhv(net)

    p_demand = sum(
        float(mm.value(c.model.p_mw) or 0.0)
        for c in net.childs_by_type(mm.PowerLoad)
    )
    # PowerGenerator stores p_mw with a negative sign (load convention); take
    # absolute value for the magnitude.
    p_gen = sum(
        abs(float(mm.value(c.model.p_mw) or 0.0))
        for c in net.childs_by_type(mm.PowerGenerator)
    )

    g_sinks = [
        c for c in net.childs_by_type(mm.Sink)
        if c.grid is not None and getattr(c.grid, "name", None) == "gas"
    ]
    g_demand_kgps = sum(
        float(mm.value(c.model.mass_flow) or 0.0) for c in g_sinks
    )

    g_sources = [
        c for c in net.childs_by_type(mm.Source)
        if c.grid is not None and getattr(c.grid, "name", None) == "gas"
    ]
    # Source.mass_flow stored with a negative sign — take absolute.
    g_gen_kgps = sum(
        abs(float(mm.value(c.model.mass_flow) or 0.0)) for c in g_sources
    )

    return dict(
        p_demand_mw=p_demand,
        p_gen_mw=p_gen,
        g_demand_kgps=g_demand_kgps,
        g_demand_mw=g_demand_kgps * 3.6 * hhv,
        g_gen_kgps=g_gen_kgps,
        g_gen_mw=g_gen_kgps * 3.6 * hhv,
        hhv=hhv,
    )


def _scale_to_abs_total(items, get_val, set_val, target_abs_total) -> bool:
    """Uniformly scale items so Σ|get_val(c)| == target_abs_total.

    Preserves each item's sign. Returns True on success, False if the current
    total is zero (cannot scale up from nothing) or if target is non-positive.
    """
    cur_abs = sum(abs(get_val(c)) for c in items)
    if cur_abs <= 0 or target_abs_total <= 0:
        return False
    factor = target_abs_total / cur_abs
    for c in items:
        set_val(c, get_val(c) * factor)
    return True


def _apply_headroom(net, headroom_frac: float) -> dict:
    """Scale in-grid electricity + gas generation to demand × (1 + headroom/2)
    and return slack budgets sized at demand × (headroom/2). Heat untouched.

    With ``headroom_frac=0.20`` (default for the ``_relaxed`` variants):
      • PowerGenerator total      = 1.10 × Σ PowerLoad
      • Σ Source(gas)             = 1.10 × Σ Sink(gas)        (in MW)
      • slack_el ext-grid bound   = ±0.10 × Σ PowerLoad        (MW)
      • slack_gas ext-grid bound  = ±0.10 × Σ Sink(gas)        (kg/s)

    so the combined supply ceiling (gen + slack) sits at 120 % of demand and
    the 20 % headroom is split 50/50 between gen and slack on each carrier.
    """
    gen_target_ratio = 1.0 + headroom_frac / 2.0   # 1.10 for 20 % headroom
    slack_ratio = headroom_frac / 2.0              # 0.10 for 20 % headroom

    t = _carrier_totals(net)

    # Electricity ──────────────────────────────────────────────────────────
    p_target_mw = t["p_demand_mw"] * gen_target_ratio
    pgens = list(net.childs_by_type(mm.PowerGenerator))
    p_ok = _scale_to_abs_total(
        pgens,
        get_val=lambda c: float(mm.value(c.model.p_mw) or 0.0),
        set_val=lambda c, v: setattr(c.model, "p_mw", v),
        target_abs_total=p_target_mw,
    )
    if not p_ok and t["p_demand_mw"] > 1e-9:
        raise RuntimeError(
            "Cannot scale electricity generation to headroom target: "
            f"current Σ|PowerGenerator.p_mw| = 0 but demand = {t['p_demand_mw']:.4f} MW."
        )

    # Gas ─────────────────────────────────────────────────────────────────
    g_target_mw = t["g_demand_mw"] * gen_target_ratio
    g_target_kgps = g_target_mw / (3.6 * t["hhv"]) if t["hhv"] > 0 else 0.0
    gsources = [
        c for c in net.childs_by_type(mm.Source)
        if c.grid is not None and getattr(c.grid, "name", None) == "gas"
    ]
    g_ok = _scale_to_abs_total(
        gsources,
        get_val=lambda c: float(mm.value(c.model.mass_flow) or 0.0),
        set_val=lambda c, v: setattr(c.model, "mass_flow", v),
        target_abs_total=g_target_kgps,
    )
    if not g_ok and t["g_demand_mw"] > 1e-9:
        raise RuntimeError(
            "Cannot scale gas generation to headroom target: "
            f"current Σ|Source(gas).mass_flow| = 0 but demand = {t['g_demand_mw']:.4f} MW."
        )

    return dict(
        slack_el_mw=t["p_demand_mw"] * slack_ratio,
        slack_gas_kgps=t["g_demand_kgps"] * slack_ratio,
    )


def _validate_headroom_balance(
    net,
    slack_el_mw: float,
    slack_gas_kgps: float,
    headroom_frac: float,
    label: str = "",
    tol: float = 0.01,
) -> None:
    """Verify the carrier totals match the headroom contract:

      gen   ≈ (1 + headroom/2) × demand     (within ``tol`` relative)
      slack ≈ (    headroom/2) × demand     (within ``tol`` relative)

    so the 20 % headroom is *actually* reached and *evenly* split between
    in-grid generation and slack budget. Heat is intentionally not checked.
    Raises ``AssertionError`` with a diagnostic message on mismatch.
    """
    expected_gen = 1.0 + headroom_frac / 2.0
    expected_slack = headroom_frac / 2.0
    t = _carrier_totals(net)
    prefix = f"[headroom:{label}]" if label else "[headroom]"

    def _check(carrier, demand, gen, slack, slack_human):
        if demand <= 1e-9:
            print(f"{prefix} {carrier}: demand=0 — nothing to check")
            return
        gen_r, slack_r = gen / demand, slack / demand
        gen_err = gen_r - expected_gen
        slack_err = slack_r - expected_slack
        print(
            f"{prefix} {carrier}: demand={demand:.4f} MW  "
            f"gen={gen:.4f} MW (×{gen_r:.4f}, target ×{expected_gen:.2f}, Δ={gen_err:+.4f})  "
            f"slack=±{slack_human} (×{slack_r:.4f}, target ×{expected_slack:.2f}, Δ={slack_err:+.4f})"
        )
        assert abs(gen_err) <= tol, (
            f"{prefix} {carrier} gen/demand={gen_r:.4f}, expected ≈{expected_gen:.4f} "
            f"(tol={tol})"
        )
        assert abs(slack_err) <= tol, (
            f"{prefix} {carrier} slack/demand={slack_r:.4f}, expected ≈{expected_slack:.4f} "
            f"(tol={tol})"
        )

    _check(
        "electricity",
        t["p_demand_mw"], t["p_gen_mw"],
        slack_el_mw, f"{slack_el_mw:.4f} MW",
    )
    _check(
        "gas",
        t["g_demand_mw"], t["g_gen_mw"],
        slack_gas_kgps * 3.6 * t["hhv"],
        f"{slack_gas_kgps:.6f} kg/s ≈ {slack_gas_kgps * 3.6 * t['hhv']:.4f} MW",
    )


def create_large_lv_simbench(density, central=False, cp_capacity_invariant=False,
                             headroom_frac=None):
    def create():
        net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
        mn = from_pandapower_net(net)
        mes = generate_supply_return_mes_based_on_power_net(
            mn,
            coupling_density=density,
            centralized=central,
            couplings=("chp", "p2g", "p2h"),
            coupling_kwargs={
                "seed": 1,
                "use_hg_variants": True,
                "chp_p_share": 1.6,
                "p2g_p_share": 1,
                "p2h_p_share": 0.2,
                "cp_size_multiplier": 3.0,
                "replace_primary_generation": cp_capacity_invariant,
            },
            heat_kwargs={
                "node_based_heat_loads": True,
                "node_heat_gen_share": 3.0,
            },
            gas_kwargs={
                "gas_gen_share": 3.0,
                "mesh_seed": 42,
            },
        )

        # ``cp_capacity_invariant=True`` reduces non-CP primary generation by
        # the added CP capacity, but the CPs themselves still draw on their
        # input carrier. Without compensation, total primary energy
        # consumption rises by Σ O_k·(1/η_k − 1). Subtracting the CP input
        # draw from end-user demand on the input carrier keeps the primary
        # supply matched to the no-CP baseline. Applied BEFORE
        # ``apply_formulation`` so the formulations see the adjusted values.
        if cp_capacity_invariant:
            _balance_demand_for_cp_replacement(mes)

        # ``_relaxed`` variants: apply *after* CP rebalancing so the gen and
        # slack budgets are sized against the final post-rebalancing demand.
        slack_overrides = None
        if headroom_frac is not None:
            slack_overrides = _apply_headroom(mes, headroom_frac=headroom_frac)
            _validate_headroom_balance(
                mes,
                slack_el_mw=slack_overrides["slack_el_mw"],
                slack_gas_kgps=slack_overrides["slack_gas_kgps"],
                headroom_frac=headroom_frac,
                label=f"density={density},same_cap={cp_capacity_invariant}",
            )

        mes.apply_formulation(MISOCP_NETWORK_FORMULATION)
        mes.apply_formulation(make_mccormick_dhs_formulation(num_partitions=16))

        if slack_overrides is not None:
            slack_el = slack_overrides["slack_el_mw"]
            slack_gas = slack_overrides["slack_gas_kgps"]
            return MESContainer(
                network=mes,
                ext_grid_el_bounds=(-slack_el, slack_el),
                ext_grid_gas_bounds=(-slack_gas, slack_gas),
                ext_grid_heat_bounds=(-6, 6),
                include_coupling_points=cp_capacity_invariant,
            )

        return MESContainer(
            network=mes,
            ext_grid_el_bounds=(-0.10, 0.10),
            ext_grid_gas_bounds=(-0.007, 0.007),
            ext_grid_heat_bounds=(-6, 6),
            include_coupling_points=cp_capacity_invariant,
        )
    return create

def create_large_lv_simbench_ts(
            net: mm.Network, n_steps: int = 96, seed: int = 0
):
    return TimeseriesData()

ALL_GRIDS = {
    "simbench_lv_no": (create_large_lv_simbench(0), create_large_lv_simbench_ts),
    
    "simbench_lv_low": (create_large_lv_simbench(0.05), create_large_lv_simbench_ts),
    "simbench_lv": (create_large_lv_simbench(0.1), create_large_lv_simbench_ts),
    "simbench_lv_high": (create_large_lv_simbench(0.15), create_large_lv_simbench_ts),
    "simbench_lv_xl": (create_large_lv_simbench(0.2), create_large_lv_simbench_ts),
    "simbench_lv_xxl": (create_large_lv_simbench(0.25), create_large_lv_simbench_ts),
    

    "simbench_lv_low_same_cap": (create_large_lv_simbench(0.05, cp_capacity_invariant=True), create_large_lv_simbench_ts),
    "simbench_lv_same_cap": (create_large_lv_simbench(0.1, cp_capacity_invariant=True), create_large_lv_simbench_ts),
    "simbench_lv_high_same_cap": (create_large_lv_simbench(0.15, cp_capacity_invariant=True), create_large_lv_simbench_ts),
    "simbench_lv_xl_same_cap": (create_large_lv_simbench(0.2, cp_capacity_invariant=True), create_large_lv_simbench_ts),
    "simbench_lv_xxl_same_cap": (create_large_lv_simbench(0.25, cp_capacity_invariant=True), create_large_lv_simbench_ts),

    # ``_relaxed`` variants — same 11 topologies, but each carrier
    # (electricity + gas only; heat untouched) gets 20 % headroom over
    # demand, split evenly: in-grid generation sized at 1.10 × demand and
    # slack ext-grid bound at ±0.10 × demand. See ``_apply_headroom`` /
    # ``_validate_headroom_balance``.
    "simbench_lv_no_relaxed": (create_large_lv_simbench(0, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_low_relaxed": (create_large_lv_simbench(0.05, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_relaxed": (create_large_lv_simbench(0.1, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_high_relaxed": (create_large_lv_simbench(0.15, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_xl_relaxed": (create_large_lv_simbench(0.2, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_xxl_relaxed": (create_large_lv_simbench(0.25, headroom_frac=0.20), create_large_lv_simbench_ts),

    "simbench_lv_low_same_cap_relaxed": (create_large_lv_simbench(0.05, cp_capacity_invariant=True, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_same_cap_relaxed": (create_large_lv_simbench(0.1, cp_capacity_invariant=True, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_high_same_cap_relaxed": (create_large_lv_simbench(0.15, cp_capacity_invariant=True, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_xl_same_cap_relaxed": (create_large_lv_simbench(0.2, cp_capacity_invariant=True, headroom_frac=0.20), create_large_lv_simbench_ts),
    "simbench_lv_xxl_same_cap_relaxed": (create_large_lv_simbench(0.25, cp_capacity_invariant=True, headroom_frac=0.20), create_large_lv_simbench_ts),
}

def print_demands(net: mm.Network) -> None:
    """Print every demand setpoint in MW for the given network.

    Conversions:
      PowerLoad.p_mw           — already MW
      Sink (gas)               — kg/s × hhv [kWh/kg] × 3.6 → MW
      HeatExchangerLoad.q_mw   — MW (q_mw_set stored as −q_mw)
    """
    rows = []

    for c in net.childs:
        m = c.model
        if isinstance(m, mm.PowerLoad):
            rows.append(("electricity", "PowerLoad", c.id, float(mm.value(m.p_mw))))
        elif isinstance(m, mm.HeatLoad):
            rows.append(("heat", "HeatLoad", c.id, float(mm.value(m.q_mw_heat))))
        elif isinstance(m, mm.Sink):
            gname = getattr(c.grid, "name", "?")
            if gname == "gas":
                hhv = getattr(c.grid, "higher_heating_value", 15.3)
                mw = float(mm.value(m.mass_flow)) * 3.6 * hhv
                rows.append(("gas", "Sink", c.id, mw))
            else:
                # water sinks: no MW without ΔT context
                rows.append((gname, "Sink", c.id, float("nan")))

    for b in net.branches:
        m = b.model
        if isinstance(m, mm.HeatExchangerLoad):
            q_set = float(mm.value(m.q_mw_set))
            rows.append(("heat", type(m).__name__, b.id, -q_set))

    by_carrier: dict[str, list] = {}
    for carrier, typ, cid, mw in rows:
        by_carrier.setdefault(carrier, []).append((typ, cid, mw))

    print("\n=== Demands (MW) ===")
    grand = 0.0
    for carrier in sorted(by_carrier):
        items = by_carrier[carrier]
        total = sum(mw for _, _, mw in items if mw == mw)
        grand += total
        print(f"\n[{carrier}]  count={len(items)}  total={total:.4f} MW")
        for typ, cid, mw in sorted(items, key=lambda r: -(r[2] if r[2] == r[2] else 0)):
            mw_str = f"{mw:10.4f} MW" if mw == mw else "       nan"
            print(f"  {typ:24s} id={str(cid):30s}  {mw_str}")
    print(f"\n[grand total]  {grand:.4f} MW")


def solve(
    network,
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
    ext_grid_heat_bounds=(-100, 100),
    include_coupling_points=False,
):
    optimization_problem = mp.create_min_load_shedding_problem(
        bounds_el=(0.9, 1.1),
        bounds_gas=(0.9, 1.1),
        bounds_heat=(0.7, 1.3),
        ext_grid_el_bounds=ext_grid_el_bounds,
        ext_grid_gas_bounds=ext_grid_gas_bounds,
        ext_grid_heat_bounds=ext_grid_heat_bounds,
        include_ext_grids=True,
        include_coupling_points=include_coupling_points,
        check_vm=True,
        check_pressure=True,
        check_temperature=True,
        check_line_loading=True,
        priority_safety_factor=1000.0,
    )

    return run_energy_flow_optimization(
        network,
        solver=PyomoSolver(),
        solver_name="gurobi",
        optimization_problem=optimization_problem,
        exclude_unconnected_nodes=True,
    )


if __name__ == "__main__":
    import monee.model as mm

    print("URBAN")
    print("-------")
    container = create_large_lv_simbench(0.5)()
    net = container.network
    print_demands(net)
    res = solve(
        net,
        ext_grid_el_bounds=container.ext_grid_el_bounds,
        ext_grid_gas_bounds=container.ext_grid_gas_bounds,
        ext_grid_heat_bounds=container.ext_grid_heat_bounds,
        include_coupling_points=container.include_coupling_points,
    )
    print(res.summary())

    # print("Industrial")
    # print("-------")
    # net = create_industrial_hub_net()
    # net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # print(run_energy_flow(net, solver=PyomoSolver()))

    # print("Regional")
    # print("-------")
    # net = create_regional_mes_net()
    # net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # print(run_energy_flow(net, solver=PyomoSolver()))
