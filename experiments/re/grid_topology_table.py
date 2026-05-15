"""Print one LaTeX table row per grid in :data:`test_grids.ALL_GRIDS`.

Reports per row (in column order):
    1.  scenario id
    2.  # buses                    (electricity nodes)
    3.  # gas junctions            (gas-grid junctions, CP control nodes excluded)
    4.  # heat junctions           (water-grid junctions, idem)
    5.  # branches (electricity)   (GenericPowerBranch / PowerLine / Trafo)
    6.  # branches (gas)           (GasPipe / GasCompressor)
    7.  # branches (heat)          (WaterPipe)
    8.  # CPs                      (size of net.cps)
    9. – 11. rated generation capacity per sector  (electricity / gas / heat) in MW
   12. – 14. rated CP capacity per sector          (electricity / gas / heat) in MW

CP capacity per sector = sum of CP throughput on that carrier side, e.g.
electricity = Σ |p_in| (P2H, P2G) + Σ |p_out| (G2P, CHP). Gas capacities
are converted from kg/s via the gas grid's higher heating value.

Output is the bare ``a & b & c & ... \\\\`` rows — paste straight into your
own ``tabular`` environment.
"""

from __future__ import annotations

from typing import Tuple

import monee.model as mm

from test_grids import ALL_GRIDS


# ─────────────────────────────────────────────────────────────────────────────
# Pretty labels for the scenario column
# ─────────────────────────────────────────────────────────────────────────────


SCENARIO_LABEL = {
    "simbench_lv_no":                       "LV-no",

    "simbench_lv_low":                      "LV-s",
    "simbench_lv":                          "LV-m",
    "simbench_lv_high":                     "LV-l",
    "simbench_lv_xl":                       "LV-xl",
    "simbench_lv_xxl":                      "LV-xxl",

    "simbench_lv_low_same_cap":             "LV-s-eq",
    "simbench_lv_same_cap":                 "LV-m-eq",
    "simbench_lv_high_same_cap":            "LV-l-eq",
    "simbench_lv_xl_same_cap":              "LV-xl-eq",
    "simbench_lv_xxl_same_cap":        "LV-xxl-eq",
}


def _label(name: str) -> str:
    return SCENARIO_LABEL.get(name, name.replace("_", r"\_"))


# ─────────────────────────────────────────────────────────────────────────────
# Counters / accumulators
# ─────────────────────────────────────────────────────────────────────────────


def _count_nodes(net: mm.Network) -> Tuple[int, int, int]:
    n_bus = n_gj = n_hj = 0
    for n in net.nodes:
        m = n.model
        # ``CHPHGControlNode`` inherits from both Bus and Junction —
        # exclude it from the topology counts via exact-type checks.
        t = type(m)
        grid_name = getattr(getattr(n, "grid", None), "name", None)
        if t is mm.Bus and grid_name == "power":
            n_bus += 1
        elif t is mm.Junction and grid_name == "gas":
            n_gj += 1
        elif t is mm.Junction and grid_name == "water":
            n_hj += 1
    return n_bus, n_gj, n_hj


_ELEC_BRANCHES = (mm.GenericPowerBranch, mm.PowerLine, mm.Trafo)
_GAS_BRANCHES  = (mm.GasPipe, mm.GasCompressor)
_HEAT_BRANCHES = (mm.WaterPipe,)


def _count_branches(net: mm.Network) -> Tuple[int, int, int]:
    n_e = n_g = n_h = 0
    for b in net.branches:
        m = b.model
        if isinstance(m, _ELEC_BRANCHES):
            n_e += 1
        elif isinstance(m, _GAS_BRANCHES):
            n_g += 1
        elif isinstance(m, _HEAT_BRANCHES):
            n_h += 1
    return n_e, n_g, n_h


def _gas_hhv(net: mm.Network, default: float = 15.3) -> float:
    for n in net.nodes:
        g = getattr(n, "grid", None)
        if g is not None and getattr(g, "name", "") == "gas":
            return float(getattr(g, "higher_heating_value", default))
    return default


def _generation_capacity(net: mm.Network, hhv: float) -> Tuple[float, float, float]:
    """Per-sector rated generation capacity in MW.

    Excludes CPs (handled separately) and slack/external grids."""
    p_e = p_g = p_h = 0.0
    # Electricity: PowerGenerator (p_mw is internally negated; |·|)
    for c in net.childs_by_type(mm.PowerGenerator):
        try:
            p_e += abs(float(mm.value(c.model.p_mw)))
        except Exception:
            pass
    # Gas: Source on gas grid (mass_flow → MW via HHV; mass_flow internally negated)
    for c in net.childs_by_type(mm.Source):
        if getattr(getattr(c, "grid", None), "name", "") != "gas":
            continue
        try:
            p_g += abs(float(mm.value(c.model.mass_flow))) * 3.6 * hhv
        except Exception:
            pass
    # Heat: dedicated heat generators (q_mw_heat / q_mw stored on the model)
    for cls in (mm.HeatGenerator,
                mm.HeatExchangerGenerator,
                mm.PassiveHeatExchangerGenerator):
        # HeatGenerator is a child; the others are branches.
        try:
            for c in net.childs_by_type(cls):
                v = getattr(c.model, "q_mw_heat", getattr(c.model, "q_mw", None))
                if v is None:
                    continue
                p_h += abs(float(mm.value(v)))
        except Exception:
            pass
        try:
            for b in net.branches_by_type(cls):
                v = getattr(b.model, "q_mw", None)
                if v is None:
                    continue
                p_h += abs(float(mm.value(v)))
        except Exception:
            pass
    return p_e, p_g, p_h


def _demand(net: mm.Network, hhv: float) -> Tuple[float, float, float]:
    """Per-sector total demand in MW.

    Mirrors :func:`_generation_capacity`: electricity from ``PowerLoad``,
    gas from ``Sink`` on the gas grid (kg/s → MW via HHV), heat from
    ``HeatLoad`` and the load-side heat-exchanger branches. Heat-exchanger
    rated load lives in ``q_mw_set`` (stored as ``-q_mw`` by the
    constructor), so we take the absolute value.
    """
    d_e = d_g = d_h = 0.0
    for c in net.childs_by_type(mm.PowerLoad):
        try:
            d_e += abs(float(mm.value(c.model.p_mw)))
        except Exception:
            pass
    for c in net.childs_by_type(mm.Sink):
        if getattr(getattr(c, "grid", None), "name", "") != "gas":
            continue
        try:
            d_g += abs(float(mm.value(c.model.mass_flow))) * 3.6 * hhv
        except Exception:
            pass
    for c in net.childs_by_type(mm.HeatLoad):
        try:
            d_h += abs(float(mm.value(c.model.q_mw_heat)))
        except Exception:
            pass
    for cls in (mm.HeatExchangerLoad, mm.PassiveHeatExchangerLoad):
        try:
            for b in net.branches_by_type(cls):
                v = getattr(b.model, "q_mw_set", None)
                if v is None:
                    continue
                d_h += abs(float(mm.value(v)))
        except Exception:
            pass
    return d_e, d_g, d_h


def _cp_capacity(net: mm.Network, hhv: float) -> Tuple[float, float, float]:
    """Per-sector CP **output** capacity in MW.

    Each CP contributes only on the carrier(s) where it acts as a producer:
      CHP / CHPHG       → electricity + heat (gas is input, not counted)
      P2H / P2HHG       → heat
      P2G               → gas
      G2P               → electricity
      G2H / G2HHG       → heat

    This matches the ``replace_primary_generation`` semantics in
    :func:`generate_supply_return_mes_based_on_power_net`, so
    ``gen + CP_output`` per carrier is invariant for the ``*_same_cap``
    scenarios (modulo carriers whose primary pool is empty, e.g. heat).
    """
    cap_e = cap_g = cap_h = 0.0

    # CHP setpoints live on the *compound*, not on the control node that
    # ``net.cps`` exposes. Walk compounds explicitly.
    for comp in net.compounds:
        m = comp.model
        if not isinstance(m, (mm.CHP, mm.CHPHG)):
            continue
        try:
            mfs = abs(float(getattr(m, "mass_flow_setpoint", 0.0)))
            eta_p = float(getattr(m, "efficiency_power", 0.0))
            eta_h = float(getattr(m, "efficiency_heat", 0.0))
        except Exception:
            continue
        gas_in = mfs * 3.6 * hhv
        cap_e += gas_in * eta_p
        cap_h += gas_in * eta_h

    for cp in net.cps:
        m = cp.model
        if isinstance(m, (mm.CHPControlNode, mm.CHPHGControlNode)):
            continue
        if isinstance(m, (mm.PowerToHeat, mm.PowerToHeatHG)):
            try:
                cap_h += abs(float(getattr(m, "heat_energy_mw", 0.0)))
            except Exception:
                pass
            continue
        if isinstance(m, mm.PowerToGas):
            try:
                kgps = abs(float(mm.value(getattr(m, "gas_kgps", 0.0))))
                cap_g += kgps * 3.6 * hhv
            except Exception:
                pass
            continue
        if isinstance(m, mm.GasToPower):
            try:
                cap_e += abs(float(mm.value(getattr(m, "el_mw", 0.0))))
            except Exception:
                pass
            continue
        if isinstance(m, (mm.GasToHeat, mm.GasToHeatHG)):
            try:
                cap_h += abs(float(getattr(m, "heat_energy_mw", 0.0)))
            except Exception:
                pass
    return cap_e, cap_g, cap_h


# ─────────────────────────────────────────────────────────────────────────────
# Row formatting
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(x: float) -> str:
    """Compact MW formatter — 2 decimals if < 10, 1 decimal if < 100, else int."""
    if x != x:  # NaN
        return "--"
    ax = abs(x)
    if ax < 10:
        return f"{x:.2f}"
    if ax < 100:
        return f"{x:.1f}"
    return f"{x:.0f}"


def row_for(name: str) -> str:
    create_fn, _ = ALL_GRIDS[name]
    container = create_fn()
    net = container.network

    n_bus, n_gj, n_hj = _count_nodes(net)
    nb_e, nb_g, nb_h = _count_branches(net)
    n_cp = len(net.cps)
    hhv = _gas_hhv(net)
    gen_e, gen_g, gen_h = _generation_capacity(net, hhv)
    cp_e, cp_g, cp_h = _cp_capacity(net, hhv)
    dem_e, dem_g, dem_h = _demand(net, hhv)

    cells = [
        _label(name),
        f"{n_bus + n_gj + n_hj}",
        f"{nb_e + nb_g + nb_h}",
        f"{n_cp}",
        _fmt(gen_e), _fmt(gen_g), _fmt(gen_h),
        _fmt(cp_e),  _fmt(cp_g),  _fmt(cp_h),
        _fmt(dem_e), _fmt(dem_g), _fmt(dem_h),
    ]
    return " & ".join(cells) + r" \\"


def main(only=None):
    names = only or list(ALL_GRIDS)
    rows = []
    for name in names:
        rows.append(row_for(name))
    for row in rows:
        print(row)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Emit LaTeX table rows summarising every grid in ALL_GRIDS.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to a subset of scenario ids")
    args = ap.parse_args()
    main(args.only)
