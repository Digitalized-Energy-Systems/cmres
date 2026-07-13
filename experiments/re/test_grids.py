"""
Test multi-energy system (MES) networks for resilience simulation research.

All grids share one topology (simbench ``1-LV-rural3--1-no_sw`` + generated
gas / heat layers) and differ in coupling-point density and supply sizing.
Three scenario families demonstrate how CPs affect resilience:

  ``_backup``      - additive CPs, symmetric base headroom h=0.10; the gas
                     side is CP-aware (``donor_gas_cp_margin=0.5``): h_gas is
                     raised above 0.10 wherever needed so the gas surplus
                     stays 50 % above the CP fleet's rated gas draw at every
                     density. Electrical failures can be rescued by CHPs
                     drawing on the gas surplus → "CPs help".
  ``_loadbearing`` - CPs replace primary generation (``cp_capacity_invariant``),
                     symmetric h=0.10 sized CP-aware so the CP fleet is fully
                     fuelable at baseline. Gas-side failures cascade through
                     the load-bearing CHPs → "CPs hurt".
  ``_decoupled``   - decoupled mirror of ``_loadbearing``: the identical
                     seeded fleet (sites, types, rated outputs) realised as
                     independent single-carrier generators with no
                     input-carrier draw (monee ``decoupled_generation``),
                     still replacing primary generation; symmetric h=0.10.
                     ``loadbearing − decoupled`` isolates the cross-carrier
                     dependency, ``decoupled − no-CP`` the pure re-siting
                     effect of moving capacity onto the CP buses.
                     Contract caveats — the subtraction removes the coupling
                     *and* its side effects: input carriers carry no CP draw
                     (effective demand and slack size on D alone), a CHP
                     mirror fails as two independent halves (el + heat, each
                     at the CP base probability; heat via mm.HeatGenerator
                     in FAIL_BASE_PROBABILITY_MAP) instead of one compound,
                     and mirror failures land in single-carrier source rows
                     of the cross-carrier matrix (loadbearing's land in
                     "multi"). The N-1 single-removal sweep targets the full
                     MC-priced surface (incl. generation childs), so the
                     mirror fleet IS swept — its rows carry kind="child".
  ``_control``     - additive CPs, both carriers tight (h=0.04). Negative
                     control: without donor-carrier surplus CPs cannot help.

Headroom ``h`` per carrier sets a fixed absolute net capacity ``h × D`` over
demand (``D`` = end-user demand), split 50/50 between in-grid generation
over-provision and the ext-grid slack bound. In the CP-aware ``_loadbearing``
family, generation additionally covers the rated CP input draw and is credited
with the rated CP output, so the CP fleet is fuelable at baseline while the
net-capacity margin stays sized on the density-invariant ``D`` (not on
``D + CPin``, which grows ~40 % over the density sweep). Heat is never rescaled
— the heat-side ext grid mass flow is hydraulically fixed by the water sinks.

Every baseline (no-fault) grid must solve to ≈ zero end-user load shed; see
``experiments/re/analyze_baseline_shed.py --assert-clean``.

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
    EL_MISOCP_FORMULATION,
    make_heat_convex_milp_formulation,
)
from monee.model.grid import DEFAULT_GAS_HHV_KWH_PER_KG
from monee.network import (
    generate_supply_return_mes_based_on_power_net,
)


@dataclass
class MESContainer:
    """A network bundled with the ext-grid bounds the solver should use.

    ``include_coupling_points`` affects the min-load-shedding *objective*
    only (CPs become demand-weighted loads on their input carrier, which
    resolves degenerate ties toward keeping load-bearing CPs running). The
    performance *metric* always counts end-user shed only — a dead
    load-bearing CP shows up there as downstream unserved load. True for
    the ``_loadbearing`` family where CPs replace primary generation.
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
            base = float(mm.value(c.model.mass_flow_kgs))
            td.add_child_series(
                c.id,
                "mass_flow_kgs",
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
        base = float(mm.value(c.model.mass_flow_kgs))
        td.add_child_series(
            c.id,
            "mass_flow_kgs",
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
            base = float(mm.value(c.model.mass_flow_kgs))
            td.add_child_series(
                c.id,
                "mass_flow_kgs",
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
    io = _cp_io_mw(net)
    p_in_mw, g_in_mw = io["p_in"], io["g_in"]
    gas_grid, hhv = _gas_grid_and_hhv(net)

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
        # Sink.mass_flow_kgs is stored with consumption (negative) sign in
        # monee; take magnitudes for the demand total and preserve each
        # Sink's sign when scaling.
        total_g_kgps = sum(abs(float(mm.value(c.model.mass_flow_kgs) or 0.0)) for c in sinks)
        total_g_mw = total_g_kgps * 3.6 * hhv
        if total_g_mw > 0:
            if g_in_mw >= total_g_mw:
                print(
                    f"[same_cap] WARN: CP gas draw {g_in_mw:.4f} MW ≥ total "
                    f"gas Sink {total_g_mw:.4f} MW; setting Sink mass_flow to 1e-9 kg/s each."
                )
                for c in sinks:
                    c.model.mass_flow_kgs = 1e-9
            else:
                scale = (total_g_mw - g_in_mw) / total_g_mw
                for c in sinks:
                    c.model.mass_flow_kgs = float(mm.value(c.model.mass_flow_kgs) or 0.0) * scale

    if p_in_mw > 0 or g_in_mw > 0:
        print(
            f"[same_cap] demand reduction: −{p_in_mw:.4f} MW power, "
            f"−{g_in_mw:.4f} MW gas (≈ −{g_in_mw / (3.6 * hhv):.6f} kg/s)"
        )


# =============================================================================
# Headroom helpers (per-carrier supply sizing for every grid family)
# =============================================================================


def _gas_grid_and_hhv(net):
    """Find the gas Grid object and its HHV (kWh/kg) by walking junctions.

    ``net.grids`` chokes on multi-grid CP control nodes whose ``.grid`` is a
    list — same defensive walk as ``_balance_demand_for_cp_replacement``.

    Reads ``higher_heating_value_kwh_per_kg`` (the actual GasGrid field; the
    study grids are lgas, HHV ≈ 11.79 kWh/kg). A former ``getattr`` on the
    non-existent ``higher_heating_value`` silently returned a 15.3 fallback
    everywhere, which cancelled in the pure gas-side ratios but inflated the
    CP-aware electricity credits (CHP el output, P2G el draw) by ×1.30 in
    the ``_loadbearing`` sizing.
    """
    for n in net.nodes_by_type(mm.Junction):
        g = n.grid
        if g is not None and not isinstance(g, list) and getattr(g, "name", None) == "gas":
            return g, float(getattr(
                g, "higher_heating_value_kwh_per_kg", DEFAULT_GAS_HHV_KWH_PER_KG
            ))
    return None, DEFAULT_GAS_HHV_KWH_PER_KG


def _cp_io_mw(net) -> dict:
    """Rated CP input draw and output per carrier at regulation 1, in MW.

    Returns ``{"p_in", "g_in", "p_out", "g_out"}``. Heat outputs are not
    tracked (heat supply is never rescaled). Per-CP accounting
    (I = input carrier, O = output carrier, η = conversion efficiency):

      CHP / CHPHG     I=G: mass_flow_setpoint·3.6·HHV   O=P: I·η_power
      PowerToHeat     I=P: load_p_mw (= q_mw/η)
      PowerToHeatHG   I=P: heat_energy_mw / η
      GasToHeatHG     I=G: heat_energy_mw / η
      GasToPower      I=G: el_mw / η                    O=P: el_mw
      PowerToGas      I=P: gas_kgps·3.6·HHV / η         O=G: gas_kgps·3.6·HHV
    """
    _, hhv = _gas_grid_and_hhv(net)
    io = dict(p_in=0.0, g_in=0.0, p_out=0.0, g_out=0.0)

    for cp in net.compounds:
        m = cp.model
        cn = type(m).__name__
        if cn in ("CHP", "CHPHG"):
            kgps = abs(float(getattr(m, "mass_flow_setpoint_kgs", 0.0) or 0.0))
            gas_mw = kgps * 3.6 * hhv
            io["g_in"] += gas_mw
            eta_p = float(getattr(m, "efficiency_power", 0.0) or 0.0)
            io["p_out"] += gas_mw * eta_p
        elif cn == "PowerToHeat":
            # load_p_mw = heat_energy_mw / η, set in PowerToHeat.__init__.
            io["p_in"] += abs(float(getattr(m, "load_p_mw", 0.0) or 0.0))

    for br in net.branches:
        m = br.model
        cn = type(m).__name__
        if cn == "PowerToHeatHG":
            h = abs(float(getattr(m, "heat_energy_mw", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                io["p_in"] += h / eta
        elif cn == "GasToHeatHG":
            h = abs(float(getattr(m, "heat_energy_mw", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                io["g_in"] += h / eta
        elif cn == "GasToPower":
            p = abs(float(getattr(m, "el_mw", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                io["g_in"] += p / eta
                io["p_out"] += p
        elif cn == "PowerToGas":
            # mass_flow_setpoint stored on the branch (positive magnitude);
            # the constructor flips its sign onto gas_mass_flow_kgs internally.
            kgps_out = abs(float(getattr(m, "gas_mass_flow_kgs", 0.0) or 0.0))
            eta = float(getattr(m, "efficiency", 0.0) or 0.0)
            if eta > 1e-6:
                gas_mw = kgps_out * 3.6 * hhv
                io["p_in"] += gas_mw / eta
                io["g_out"] += gas_mw

    return io


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
    # Sink.mass_flow_kgs is stored with a consumption (negative) sign — take
    # absolute for the demand magnitude.
    g_demand_kgps = sum(
        abs(float(mm.value(c.model.mass_flow_kgs) or 0.0)) for c in g_sinks
    )

    g_sources = [
        c for c in net.childs_by_type(mm.Source)
        if c.grid is not None and getattr(c.grid, "name", None) == "gas"
    ]
    # Source.mass_flow_kgs stored with a negative sign — take absolute.
    g_gen_kgps = sum(
        abs(float(mm.value(c.model.mass_flow_kgs) or 0.0)) for c in g_sources
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


def _is_mirror_child(c) -> bool:
    """Decoupled-mirror generation child (monee ``decoupled_generation``)."""
    return str(getattr(c, "name", "") or "").startswith("mirror_cp_")


def _apply_headroom(net, h_el: float, h_gas: float, cp_aware: bool = False) -> dict:
    """Scale in-grid electricity + gas generation and size ext-grid slack
    budgets so each carrier carries a fixed absolute net capacity ``h · D``
    over its demand, split 50/50 between generation over-provision and slack.
    Heat untouched.

    Per carrier with headroom ``h``::

        gen_target = (D + CPin) + (h/2) · D − CPout
        slack      =              (h/2) · D

    where ``D`` is end-user demand and ``CPin`` / ``CPout`` are the rated CP
    input draw / output on that carrier (zero unless ``cp_aware``). Generation
    covers the full effective demand ``D + CPin`` (so the CP fleet is fuelable
    at baseline) plus half the margin; the slack carries the other half, giving

        net capacity = (gen + CPout + slack) − (D + CPin) = h · D

    sized on the density-invariant end-user demand ``D`` — so it stays constant
    across the CP-density sweep. (The ``_loadbearing`` family previously sized
    the margin on ``D + CPin``, so both generation and the fault-time slack
    reserve grew ~40 % over the sweep, a confound for the resilience metric.)

    ``cp_aware=True`` is meant for the ``_loadbearing`` family. For additive CPs
    (off at baseline) use ``cp_aware=False`` (``CPin = CPout = 0``), which
    reduces the above to the plain ``(1 + h/2)·D`` sizing.

    Decoupled-mirror children (``mirror_cp_*``) are plain generator/Source
    children, but they stand in for the fixed-rating CP fleet: like CPs they
    are never rescaled — their output is subtracted from the carrier target
    and only the remaining primary pool is scaled. Uniform scaling would
    silently break the "same capacity as the load-bearing CPs" contract of
    the ``_decoupled`` family.
    """
    t = _carrier_totals(net)
    io = _cp_io_mw(net) if cp_aware else dict(p_in=0.0, g_in=0.0, p_out=0.0, g_out=0.0)

    # Electricity ──────────────────────────────────────────────────────────
    # Margin (net capacity) is sized on the density-invariant end-user demand
    # p_margin_base, NOT on D+CPin, so the *absolute* net capacity h·D_base
    # stays constant across the CP-density sweep. Generation still covers the
    # full effective demand D+CPin (fuelling the CP fleet) plus half the fixed
    # margin; the ext-grid slack carries the other half.
    p_demand_eff = t["p_demand_mw"] + io["p_in"]
    p_margin_base = t["p_demand_mw"]
    p_target_mw = p_demand_eff + (h_el / 2.0) * p_margin_base - io["p_out"]
    if p_demand_eff > 1e-9 and p_target_mw <= 0:
        raise RuntimeError(
            f"Electricity gen target {p_target_mw:.4f} MW ≤ 0: rated CP power "
            f"output ({io['p_out']:.4f} MW) exceeds effective demand + margin "
            f"({p_demand_eff + (h_el / 2.0) * p_margin_base:.4f} MW). "
            f"Reduce CP density/size or raise h_el."
        )
    pgens_all = list(net.childs_by_type(mm.PowerGenerator))
    mirror_p_mw = sum(
        abs(float(mm.value(c.model.p_mw) or 0.0))
        for c in pgens_all if _is_mirror_child(c)
    )
    pgens = [c for c in pgens_all if not _is_mirror_child(c)]
    p_pool_target_mw = p_target_mw - mirror_p_mw
    if p_demand_eff > 1e-9 and p_pool_target_mw <= 0:
        raise RuntimeError(
            f"Electricity primary pool target {p_pool_target_mw:.4f} MW ≤ 0: "
            f"mirror fleet output ({mirror_p_mw:.4f} MW) exceeds the carrier "
            f"target ({p_target_mw:.4f} MW). Reduce CP density/size or raise h_el."
        )
    p_ok = _scale_to_abs_total(
        pgens,
        get_val=lambda c: float(mm.value(c.model.p_mw) or 0.0),
        set_val=lambda c, v: setattr(c.model, "p_mw", v),
        target_abs_total=p_pool_target_mw,
    )
    if not p_ok and p_demand_eff > 1e-9:
        raise RuntimeError(
            "Cannot scale electricity generation to headroom target: "
            f"current Σ|PowerGenerator.p_mw| = 0 but demand = {p_demand_eff:.4f} MW."
        )

    # Gas ─────────────────────────────────────────────────────────────────
    g_demand_eff_mw = t["g_demand_mw"] + io["g_in"]
    g_margin_base_mw = t["g_demand_mw"]
    g_target_mw = g_demand_eff_mw + (h_gas / 2.0) * g_margin_base_mw - io["g_out"]
    if g_demand_eff_mw > 1e-9 and g_target_mw <= 0:
        raise RuntimeError(
            f"Gas gen target {g_target_mw:.4f} MW ≤ 0: rated CP gas output "
            f"({io['g_out']:.4f} MW) exceeds effective demand + margin "
            f"({g_demand_eff_mw + (h_gas / 2.0) * g_margin_base_mw:.4f} MW). "
            f"Reduce CP density/size or raise h_gas."
        )
    g_target_kgps = g_target_mw / (3.6 * t["hhv"]) if t["hhv"] > 0 else 0.0
    gsources_all = [
        c for c in net.childs_by_type(mm.Source)
        if c.grid is not None and getattr(c.grid, "name", None) == "gas"
    ]
    mirror_g_kgps = sum(
        abs(float(mm.value(c.model.mass_flow_kgs) or 0.0))
        for c in gsources_all if _is_mirror_child(c)
    )
    gsources = [c for c in gsources_all if not _is_mirror_child(c)]
    g_pool_target_kgps = g_target_kgps - mirror_g_kgps
    if g_demand_eff_mw > 1e-9 and g_pool_target_kgps <= 0:
        raise RuntimeError(
            f"Gas primary pool target {g_pool_target_kgps:.6f} kg/s ≤ 0: "
            f"mirror fleet output ({mirror_g_kgps:.6f} kg/s) exceeds the carrier "
            f"target ({g_target_kgps:.6f} kg/s). Reduce CP density/size or raise h_gas."
        )
    g_ok = _scale_to_abs_total(
        gsources,
        get_val=lambda c: float(mm.value(c.model.mass_flow_kgs) or 0.0),
        set_val=lambda c, v: setattr(c.model, "mass_flow_kgs", v),
        target_abs_total=g_pool_target_kgps,
    )
    if not g_ok and g_demand_eff_mw > 1e-9:
        raise RuntimeError(
            "Cannot scale gas generation to headroom target: "
            f"current Σ|Source(gas).mass_flow| = 0 but demand = {g_demand_eff_mw:.4f} MW."
        )

    return dict(
        slack_el_mw=(h_el / 2.0) * p_margin_base,
        slack_gas_kgps=(h_gas / 2.0) * g_margin_base_mw / (3.6 * t["hhv"]),
    )


def _validate_headroom_balance(
    net,
    slack_el_mw: float,
    slack_gas_kgps: float,
    h_el: float,
    h_gas: float,
    cp_aware: bool = False,
    label: str = "",
    tol: float = 0.01,
) -> None:
    """Verify the carrier totals match the fixed-net-capacity contract:

      net capacity (gen + CPout + slack) − (demand + CPin) ≈ h · D
      slack                                                ≈ (h/2) · D

    where ``D`` is the density-invariant end-user demand (``margin_base``) and
    CPin/CPout = 0 unless ``cp_aware``. This confirms the absolute net capacity
    is sized on ``D`` — constant across the density sweep — not on ``D + CPin``,
    and that generation still covers the full effective demand (net capacity
    stays ≥ 0, i.e. the CP fleet is fuelable). Heat is intentionally not
    checked. Raises ``AssertionError`` with a diagnostic message on mismatch.
    """
    t = _carrier_totals(net)
    io = _cp_io_mw(net) if cp_aware else dict(p_in=0.0, g_in=0.0, p_out=0.0, g_out=0.0)
    prefix = f"[headroom:{label}]" if label else "[headroom]"

    def _check(carrier, h, demand_eff, margin_base, supply, slack, slack_human):
        if demand_eff <= 1e-9:
            print(f"{prefix} {carrier}: demand=0 — nothing to check")
            return
        overprov = supply - demand_eff
        net_cap = overprov + slack
        net_r = net_cap / margin_base if margin_base > 1e-9 else float("nan")
        slack_r = slack / margin_base if margin_base > 1e-9 else float("nan")
        print(
            f"{prefix} {carrier}: demand_eff={demand_eff:.4f} MW  D_base={margin_base:.4f} MW  "
            f"gen+CPout={supply:.4f} MW  slack=±{slack_human}  "
            f"net_cap={net_cap:.4f} MW (×D_base {net_r:.4f}, target ×{h:.2f}, Δ={net_r - h:+.4f})"
        )
        assert abs(net_r - h) <= tol, (
            f"{prefix} {carrier} net_cap/D_base={net_r:.4f}, expected ≈{h:.4f} (tol={tol})"
        )
        assert abs(slack_r - h / 2.0) <= tol, (
            f"{prefix} {carrier} slack/D_base={slack_r:.4f}, expected ≈{h / 2.0:.4f} (tol={tol})"
        )

    _check(
        "electricity",
        h_el,
        t["p_demand_mw"] + io["p_in"],
        t["p_demand_mw"],
        t["p_gen_mw"] + io["p_out"],
        slack_el_mw, f"{slack_el_mw:.4f} MW",
    )
    _check(
        "gas",
        h_gas,
        t["g_demand_mw"] + io["g_in"],
        t["g_demand_mw"],
        t["g_gen_mw"] + io["g_out"],
        slack_gas_kgps * 3.6 * t["hhv"],
        f"{slack_gas_kgps:.6f} kg/s ≈ {slack_gas_kgps * 3.6 * t['hhv']:.4f} MW",
    )


def create_large_lv_simbench(density, central=False, cp_capacity_invariant=False,
                             h_el=0.20, h_gas=0.20, donor_gas_cp_margin=None,
                             decoupled=False):
    """Factory for one grid variant.

    ``h_el`` / ``h_gas`` are per-carrier headroom fractions (gen sized at
    1 + h/2 × effective demand, ext-grid slack at ±h/2 × effective demand).
    ``cp_capacity_invariant=True`` makes CPs replace primary generation
    (load-bearing) and switches the headroom sizing to CP-aware so the CP
    fleet is fuelable at baseline.

    ``decoupled=True`` builds the decoupled mirror of the load-bearing
    family: the identical seeded fleet realised as independent output-carrier
    generators with no input draw (monee ``decoupled_generation``), still
    replacing primary generation. The mirror children (name prefix
    ``mirror_cp_``) keep their CP-rated output — ``_apply_headroom`` scales
    only the non-mirror primary pool around them. Headroom stays CP-unaware
    (``_cp_io_mw`` is zero here anyway): effective demand is end-user demand
    only, since no input-carrier draw exists to cover.

    ``donor_gas_cp_margin`` (additive ``_backup`` family only) makes the *gas*
    headroom CP-aware: the gas surplus available to ramp the (gas-input) CP
    fleet during a fault is ``h_gas × gas_demand``, while the fleet's rated gas
    draw at regulation 1 is ``Σ CP_gas_in``. With a fixed ``h_gas`` that surplus
    falls behind the fleet as CP density grows (the fleet draw is ≈30 % of gas
    demand already at density 0.1), so ``_backup`` collapses onto ``_control``
    at the higher densities — the gas-rich "donor" can no longer fuel the
    backup CHPs. When set, ``h_gas`` is raised (never lowered) to
    ``Σ CP_gas_in · (1 + margin) / gas_demand`` so the donor surplus stays
    ``(1 + margin)×`` the fleet draw at every density. No effect when there are
    no gas-input CPs (density 0) or for the load-bearing family.
    """
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
                "replace_primary_generation": cp_capacity_invariant or decoupled,
                "decoupled_generation": decoupled,
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
        # if cp_capacity_invariant:
        #     _balance_demand_for_cp_replacement(mes)

        # Donor-carrier sizing for the additive ``_backup`` family: raise the
        # gas headroom so the gas surplus (= h_gas × gas_demand) stays
        # (1 + margin)× the CP fleet's rated gas draw at *this* density.
        # Without this the fixed h_gas is outpaced by the growing CHP fleet and
        # backup converges onto control. See docstring.
        h_gas_eff = h_gas
        if donor_gas_cp_margin is not None:
            io = _cp_io_mw(mes)
            t = _carrier_totals(mes)
            if t["g_demand_mw"] > 1e-9 and io["g_in"] > 0:
                needed_h = io["g_in"] * (1.0 + donor_gas_cp_margin) / t["g_demand_mw"]
                h_gas_eff = max(h_gas, needed_h)
                print(
                    f"[backup:donor_gas] density={density} CP gas draw="
                    f"{io['g_in']:.4f} MW ({100 * io['g_in'] / t['g_demand_mw']:.1f}% of "
                    f"gas demand); h_gas {h_gas:.2f} → {h_gas_eff:.3f} "
                    f"(surplus ≥ draw × {1.0 + donor_gas_cp_margin:.2f})"
                )

        # Headroom is applied *after* CP rebalancing so the gen and slack
        # budgets are sized against the final post-rebalancing demand. In
        # CP-aware mode (load-bearing CPs) the budgets additionally cover
        # the rated CP input draw and credit the rated CP output.
        slack_overrides = _apply_headroom(
            mes, h_el=h_el, h_gas=h_gas_eff, cp_aware=cp_capacity_invariant
        )
        _validate_headroom_balance(
            mes,
            slack_el_mw=slack_overrides["slack_el_mw"],
            slack_gas_kgps=slack_overrides["slack_gas_kgps"],
            h_el=h_el,
            h_gas=h_gas_eff,
            cp_aware=cp_capacity_invariant,
            label=f"density={density},same_cap={cp_capacity_invariant}",
        )

        # Convex MILP/MISOCP set for the Gurobi shed problem: branch-flow
        # MISOCP on electricity + McCormick district heating on the heat
        # pipes (``include_heat_exchangers=False`` keeps the legacy
        # pipes-only behaviour of ``make_mccormick_dhs_formulation``); gas
        # stays on the model-default equations.
        mes.apply_formulation(EL_MISOCP_FORMULATION)
        mes.apply_formulation(
            make_heat_convex_milp_formulation(
                num_partitions=16, include_heat_exchangers=False
            )
        )

        slack_el = slack_overrides["slack_el_mw"]
        slack_gas = slack_overrides["slack_gas_kgps"]
        return MESContainer(
            network=mes,
            ext_grid_el_bounds=(-slack_el, slack_el),
            ext_grid_gas_bounds=(-slack_gas, slack_gas),
            ext_grid_heat_bounds=(-6, 6),
            # The min-shed *objective* must match the performance *metric*,
            # which counts end-user shed only. Adding CP input draw to the
            # objective (the old ``=cp_capacity_invariant``) made the solver
            # shed real end-user load to keep the power-drawing P2G/P2H CPs
            # fed when power was tight — a non-zero baseline shed of 2-15% in
            # the load-bearing family that is a pure objective/metric mismatch
            # (the grid is feasible: see analyze_baseline_shed.py). Keeping it
            # False for every family also makes the objective uniform across
            # backup / loadbearing / control. The CHPs still dispatch, so
            # essential load-bearing CPs are kept alive by the shed penalty.
            include_coupling_points=False,
        )
    return create

def create_large_lv_simbench_ts(
            net: mm.Network, n_steps: int = 96, seed: int = 0
):
    return TimeseriesData()

# Per-family factory shorthands (see module docstring for the semantics).
# "CPs help": tight receiver (el), rich donor (gas); additive CPs. The gas
# headroom is CP-aware (``donor_gas_cp_margin``) so the donor surplus stays
# 50 % above the CP fleet's rated gas draw at every density — a fixed h_gas is
# outpaced by the growing CHP fleet (≈30 % of gas demand already at d=0.1) and
# backup would otherwise collapse onto control at the higher densities.
def _backup(density):
    return create_large_lv_simbench(
        density, h_el=0.10, h_gas=0.10, donor_gas_cp_margin=0.5
    )


# "CPs hurt": CPs replace primary generation; CP-aware symmetric sizing.
def _loadbearing(density):
    return create_large_lv_simbench(
        density, cp_capacity_invariant=True, h_el=0.10, h_gas=0.10
    )


# Decoupled mirror of _loadbearing: same fleet as independent generators,
# no coupling constraint. loadbearing − decoupled = cross-carrier dependency.
def _decoupled(density):
    return create_large_lv_simbench(
        density, decoupled=True, h_el=0.10, h_gas=0.10
    )


# Negative control: no donor surplus anywhere; additive CPs.
def _control(density):
    return create_large_lv_simbench(density, h_el=0.04, h_gas=0.04)


# NOTE: keep ALL_GRIDS a literal dict with one quoted key per line —
# slurm_single_removal_shed.sh extracts the grid list from this block via
# sed, and insertion order defines the SLURM array index → grid mapping.
ALL_GRIDS = {
    "simbench_lv_no_backup": (_backup(0.0), create_large_lv_simbench_ts),
    "simbench_lv_low_backup": (_backup(0.05), create_large_lv_simbench_ts),
    "simbench_lv_mid_backup": (_backup(0.1), create_large_lv_simbench_ts),
    "simbench_lv_high_backup": (_backup(0.15), create_large_lv_simbench_ts),
    "simbench_lv_xl_backup": (_backup(0.2), create_large_lv_simbench_ts),
    "simbench_lv_xxl_backup": (_backup(0.25), create_large_lv_simbench_ts),

    "simbench_lv_no_loadbearing": (_loadbearing(0.0), create_large_lv_simbench_ts),
    "simbench_lv_low_loadbearing": (_loadbearing(0.05), create_large_lv_simbench_ts),
    "simbench_lv_mid_loadbearing": (_loadbearing(0.1), create_large_lv_simbench_ts),
    "simbench_lv_high_loadbearing": (_loadbearing(0.15), create_large_lv_simbench_ts),
    "simbench_lv_xl_loadbearing": (_loadbearing(0.2), create_large_lv_simbench_ts),
    "simbench_lv_xxl_loadbearing": (_loadbearing(0.25), create_large_lv_simbench_ts),

    "simbench_lv_no_control": (_control(0.0), create_large_lv_simbench_ts),
    "simbench_lv_low_control": (_control(0.05), create_large_lv_simbench_ts),
    "simbench_lv_mid_control": (_control(0.1), create_large_lv_simbench_ts),
    "simbench_lv_high_control": (_control(0.15), create_large_lv_simbench_ts),
    "simbench_lv_xl_control": (_control(0.2), create_large_lv_simbench_ts),
    "simbench_lv_xxl_control": (_control(0.25), create_large_lv_simbench_ts),

    "simbench_lv_no_decoupled": (_decoupled(0.0), create_large_lv_simbench_ts),
    "simbench_lv_low_decoupled": (_decoupled(0.05), create_large_lv_simbench_ts),
    "simbench_lv_mid_decoupled": (_decoupled(0.1), create_large_lv_simbench_ts),
    "simbench_lv_high_decoupled": (_decoupled(0.15), create_large_lv_simbench_ts),
    "simbench_lv_xl_decoupled": (_decoupled(0.2), create_large_lv_simbench_ts),
    "simbench_lv_xxl_decoupled": (_decoupled(0.25), create_large_lv_simbench_ts),
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
                hhv = getattr(
                    c.grid, "higher_heating_value_kwh_per_kg",
                    DEFAULT_GAS_HHV_KWH_PER_KG,
                )
                mw = abs(float(mm.value(m.mass_flow_kgs))) * 3.6 * hhv
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
        bounds_vm=(0.9, 1.1),
        bounds_pressure=(0.9, 1.1),
        bounds_t=(0.7, 1.3),
        bounds_ext_el=ext_grid_el_bounds,
        bounds_ext_gas=ext_grid_gas_bounds,
        bounds_ext_heat=ext_grid_heat_bounds,
        include_ext_grids=True,
        include_coupling_points=include_coupling_points,
        check_vm=True,
        check_pressure=True,
        check_t=True,
        check_lp=True,
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
    import sys

    grid_name = sys.argv[1] if len(sys.argv) > 1 else "simbench_lv_xxl_loadbearing"
    print(grid_name)
    print("-------")
    container = ALL_GRIDS[grid_name][0]()
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
