"""Plot single-removal load-shed results.

Reads ``data/out/single_removal_shed/single_removal_shed_<grid>.csv`` for
each grid name passed on the CLI (or all available CSVs if none given) and
writes one HTML/PDF report per grid:

  data/out/single_removal_shed/<grid>_report.html
  data/out/single_removal_shed/<grid>_report/single/*.pdf
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cmres.evaluation.evaluation as ev  # noqa: E402
import pub_style  # noqa: E402  # shared scare-style publication theme for the
                   # bar plots (dark outline, CVD hatch, top legend, compact
                   # horizontal sizing, kept sector colours)

# Re-use the human-readable scenario labels and stable display order from
# cp_cn_evaluation when available. Falls back to the raw grid name so the
# module remains importable without the eval CLI's transitive deps.
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cp_cn_evaluation import SCENARIO_NAME_MAP, pretty_scenario  # noqa: E402
except Exception:  # pragma: no cover
    SCENARIO_NAME_MAP = {}

    def pretty_scenario(name) -> str:
        return "" if name is None else str(name)

DEFAULT_DIR = Path("data/out/single_removal_shed")

# Short per-grid labels (LV-no … LV-xxl) for the compact, side-by-side pooled
# figures; the strategy is carried by the panel/subplot title instead of the bar
# label so the three strategies can sit in one row.
_SIZE_LABEL = {"no": "LV-no", "low": "LV-s", "mid": "LV-m",
               "high": "LV-l", "xl": "LV-xl", "xxl": "LV-xxl"}
# Panel order for the row figure — treatment immediately left of its own control
# (backup ↔ no-reserve, loadbearing ↔ decoupled) so the contrasted pair is
# adjacent. Kept in step with cp_cn_evaluation._RES_FAMILY_ORDER, and
# deliberately not eval_common.FAMILY_ORDER.
_FAMILY_ORDER = {"backup": 0, "control": 1, "loadbearing": 2, "decoupled": 3}


def _short_grid(grid: str) -> str:
    m = re.search(r"lv_([a-z]+)_(?:backup|loadbearing|decoupled|control)$", str(grid))
    if m and m.group(1) in _SIZE_LABEL:
        return _SIZE_LABEL[m.group(1)]
    return pretty_scenario(grid)


#: Named in the title of every shed figure: the RQMC study (cp_cn_evaluation)
#: reports per-carrier shed in MW on the same grids, so an unqualified
#: "mean shed" title does not say which simulation produced it.
SINGLE_REMOVAL_HINT = "single-removal (N−1) experiment"

_CARRIER_COLS = (("electricity", "power_shed"), ("heat", "heat_shed"),
                 ("gas", "gas_shed"))


def _stacked_x_max(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                   grids: List[str], *, mean: bool, suffix: str = "") -> float:
    """Longest stacked (all-carrier) bar over ``grids``.

    Pins the value axis of the per-density × per-strategy shed bars so a
    density's bar is the same length in every family's figure — the per-family
    PDFs are printed side by side and per-figure autoranging silently rescales
    them against each other.
    """
    best = 0.0
    for g in grids:
        sweep, _ = records.get(g, (pd.DataFrame(), None))
        if sweep.empty:
            continue
        total = 0.0
        for _sec, col in _CARRIER_COLS:
            col = f"{col}{suffix}"
            if col in sweep.columns:
                s = float(sweep[col].sum())
                total += s / len(sweep) if mean else s
        best = max(best, total)
    return best if best > 0 else 1.0


# Single-column dissertation typography. Same sizing convention as the
# E16 plots in cmres_eval_plots: figures are rendered at ~750-1200 px
# then scaled down to a ~84 mm column, so every text element is bumped
# enough to stay legible at the printed size.
_SRS_FONT_SIZES = dict(
    base=18,
    title=22,
    axis_title=20,
    axis_tick=18,
    legend=18,
    legend_title=18,
    annotation=16,
    subplot_title=20,
    colorbar=18,
)


def _srs_bump_fonts(fig: go.Figure) -> None:
    """Apply the dissertation-column font sizes to a figure in place.

    Walks every named ``xaxis*``/``yaxis*`` layout block so subplots
    inherit the same axis-title / tick sizes as the primary axes,
    bumps the cmres-template legend / title fonts, and pushes the
    subplot-title annotations from ``make_subplots`` up to the same
    level as the figure title (otherwise auto-generated subplot
    headers get lost beside the bigger axis labels)."""
    sz = _SRS_FONT_SIZES

    # Layout-level font (catches every text element with no explicit override).
    layout = fig.layout
    if layout.font:
        layout.font.size = sz["base"]
    else:
        layout.font = dict(size=sz["base"])

    # Title font (idempotent — only touches the size).
    if layout.title and layout.title.text:
        font = layout.title.font
        if font:
            font.size = sz["title"]
        else:
            layout.title.font = dict(size=sz["title"])

    # Legend body + legend-title fonts.
    if layout.legend:
        if layout.legend.font:
            layout.legend.font.size = sz["legend"]
        else:
            layout.legend.font = dict(size=sz["legend"])
        if layout.legend.title:
            tf = layout.legend.title.font
            if tf:
                tf.size = sz["legend_title"]
            else:
                layout.legend.title.font = dict(size=sz["legend_title"])

    # Bump every axis (xaxis, yaxis, xaxis2, yaxis2, …).
    for ax_attr in dir(layout):
        if not (ax_attr.startswith("xaxis") or ax_attr.startswith("yaxis")):
            continue
        ax = getattr(layout, ax_attr, None)
        if ax is None or not hasattr(ax, "tickfont"):
            continue
        # Title font
        if ax.title:
            if ax.title.font:
                ax.title.font.size = sz["axis_title"]
            else:
                ax.title.font = dict(size=sz["axis_title"])
        # Tick font
        if ax.tickfont:
            ax.tickfont.size = sz["axis_tick"]
        else:
            ax.tickfont = dict(size=sz["axis_tick"])

    # Subplot titles + any in-plot annotation that didn't set its own size.
    for ann in (layout.annotations or ()):
        xref = getattr(ann, "xref", "") or ""
        font = ann.font
        if xref.endswith("paper") and (ann.text or ""):
            # Auto-generated subplot title.
            if font:
                font.size = sz["subplot_title"]
            else:
                ann.font = dict(size=sz["subplot_title"])
        else:
            # In-plot annotation: leave explicit sizes alone, otherwise bump.
            if font and (font.size or 0) < sz["annotation"]:
                font.size = sz["annotation"]


def _srs_finalize(fig: go.Figure, legend: str = "right") -> go.Figure:
    """Apply the cmres style and bump fonts for dissertation 1-column use."""
    fig = ev.apply_cmres_style(fig, legend=legend)
    _srs_bump_fonts(fig)
    return fig


def _srs_bump_all_fonts_by(fig: go.Figure, delta: int) -> None:
    """Increment every text-element font size on *fig* by ``delta`` (pt).

    For one-off per-figure overrides on top of ``_srs_finalize``. Touches
    the layout font, title font, legend body / title font, every
    ``xaxis*``/``yaxis*`` title and tickfont, and every annotation
    (subplot titles + in-plot)."""
    if delta == 0:
        return
    layout = fig.layout

    def _bump(font_holder, default_size: int) -> None:
        """Bump ``font_holder.font.size`` (creating the font dict if absent)."""
        if font_holder is None:
            return
        font = getattr(font_holder, "font", None)
        if font is not None and font.size is not None:
            font.size = font.size + delta
        else:
            # No prior size — assume the base default and bump it.
            setattr(font_holder, "font", dict(size=default_size + delta))

    # Layout-level font.
    if layout.font and layout.font.size is not None:
        layout.font.size = layout.font.size + delta
    else:
        layout.font = dict(size=_SRS_FONT_SIZES["base"] + delta)

    # Title.
    if layout.title and layout.title.text:
        _bump(layout.title, _SRS_FONT_SIZES["title"])

    # Legend body + legend title.
    if layout.legend:
        _bump(layout.legend, _SRS_FONT_SIZES["legend"])
        if layout.legend.title:
            _bump(layout.legend.title, _SRS_FONT_SIZES["legend_title"])

    # Axes.
    for ax_attr in dir(layout):
        if not (ax_attr.startswith("xaxis") or ax_attr.startswith("yaxis")):
            continue
        ax = getattr(layout, ax_attr, None)
        if ax is None or not hasattr(ax, "tickfont"):
            continue
        if ax.title:
            _bump(ax.title, _SRS_FONT_SIZES["axis_title"])
        if ax.tickfont and ax.tickfont.size is not None:
            ax.tickfont.size = ax.tickfont.size + delta
        else:
            ax.tickfont = dict(size=_SRS_FONT_SIZES["axis_tick"] + delta)

    # Annotations (subplot titles + in-plot).
    for ann in (layout.annotations or ()):
        font = ann.font
        if font is not None and font.size is not None:
            font.size = font.size + delta
        else:
            # Pick a sensible default depending on the annotation kind.
            xref = getattr(ann, "xref", "") or ""
            default = (_SRS_FONT_SIZES["subplot_title"]
                       if xref.endswith("paper")
                       else _SRS_FONT_SIZES["annotation"])
            ann.font = dict(size=default + delta)

# Canonical kind→colour pinning so the legend stays consistent across grids:
# a grid without CPs (no ``compound`` / ``branch_cp`` rows) must still leave
# the ``branch`` colour unchanged, otherwise the visual reading flips
# between scenarios. ``_enumerate_targets`` in single_removal_shed.py emits
# exactly these four kinds; any extension MUST add an entry here.
KIND_COLOR_MAP = {
    "branch":    ev.PALETTE_QUAL[0],  # plain branches (PowerLine, GasPipe, …)
    "branch_cp": ev.PALETTE_QUAL[1],  # branch-level CPs (PowerToGas, …)
    "compound":  ev.PALETTE_QUAL[2],  # compound CPs (CHP, PowerToHeat, …)
    # PALETTE_QUAL[3] (#e45756) is reserved for the baseline reference
    # line/marker (see _hist_total / _pooled_total_shed_box) — skip it.
    "child":     ev.PALETTE_QUAL[4],  # generation childs (gens, sources, heat gens)
}
# Stable category order so px doesn't re-order the legend by row count.
KIND_ORDER = ["branch", "branch_cp", "compound", "child"]


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Compare cp_id as string so a numeric pandas dtype doesn't degenerate to
    # an all-False match (and a noisy FutureWarning) on grids whose CP ids
    # happen to look numeric.
    cp_id_str = df["cp_id"].astype(str)
    mask = cp_id_str == "_baseline_"
    baseline = df[mask].iloc[0] if mask.any() else None
    sweep = df[~mask].copy()
    return sweep, baseline


def _hist_total(sweep: pd.DataFrame, grid: str, baseline_total: float) -> go.Figure:
    # Restrict the legend categories to whatever is actually present, but
    # in the canonical order — so colours stay tied to ``kind`` even on
    # grids that have only one or two of the three kinds.
    kinds_present = (
        set(sweep["kind"].dropna().unique()) if "kind" in sweep.columns else set()
    )
    kinds_here = [k for k in KIND_ORDER if k in kinds_present]
    fig = px.histogram(
        sweep,
        x="total_shed",
        color="kind",
        nbins=60,
        color_discrete_map=KIND_COLOR_MAP,
        category_orders={"kind": kinds_here} if kinds_here else None,
        title=f"{pretty_scenario(grid)}: distribution of total shed (MW) per single-component removal",
    )
    # Use a colour outside KIND_COLOR_MAP for the baseline reference line so
    # it's never confused with a category.
    fig.add_vline(
        x=baseline_total,
        line_dash="dash",
        line_color="#e45756",
        annotation_text=f"baseline = {baseline_total:.3g} MW",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Total load shed (MW)",
        yaxis_title="Components (count)",
        height=420, width=900,
    )
    return _srs_finalize(fig, legend="right")


def _top_components(
    sweep: pd.DataFrame, grid: str, top_n: int = 20, conn: bool = False
) -> go.Figure:
    """Top-N components by shed, stacked per carrier.

    ``conn=True`` reads the ``*_shed_conn`` columns (curtailment of loads
    that remain *connected* after the removal — disconnected loads are
    unrecoverable regardless of dispatch, so the connected-only view shows
    the shed the optimisation actually decides about).
    """
    suffix = "_conn" if conn else ""
    top = sweep.sort_values(f"total_shed{suffix}", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure()
    for carrier, sector in (("power_shed", "electricity"),
                            ("heat_shed", "heat"), ("gas_shed", "gas")):
        fig.add_trace(go.Bar(
            x=top[f"{carrier}{suffix}"],
            y=top["cp_id"].astype(str),
            orientation="h",
            name=pub_style.SECTOR_PRETTY[sector],
            marker=pub_style.sector_marker(sector),
        ))
    label = "connected-load shed" if conn else "total shed"
    fig.update_layout(barmode="stack")
    pub_style.apply_theme(
        fig,
        title=f"{pretty_scenario(grid)}: top-{top_n} components by {label} (per-carrier breakdown)",
        height=pub_style.hbar_height(len(top)), width=pub_style.BAR_FIG_WIDTH,
        font_bump=1, legend_top=True,
    )
    fig.update_xaxes(title=("Connected-load shed (MW)" if conn else "Load shed (MW)"))
    fig.update_yaxes(title="Component (cp_id)")
    return fig


def _pareto(sweep: pd.DataFrame, grid: str) -> go.Figure:
    s = sweep.sort_values("total_shed", ascending=False).reset_index(drop=True)
    s["rank"] = np.arange(1, len(s) + 1)
    s["cum_share"] = s["total_shed"].cumsum() / max(s["total_shed"].sum(), 1e-12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["rank"], y=s["total_shed"], mode="lines",
        name="total_shed (MW)", line=dict(color=ev.PALETTE_QUAL[0], width=2),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=s["rank"], y=s["cum_share"], mode="lines",
        name="cumulative share", line=dict(color=ev.PALETTE_QUAL[1], width=2, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        title=f"{pretty_scenario(grid)}: shed Pareto curve (rank vs total_shed and cumulative share)",
        xaxis=dict(title="Component rank (sorted by total_shed desc)", type="log"),
        yaxis=dict(title="Total shed (MW)"),
        yaxis2=dict(title="Cumulative share of total shed", overlaying="y", side="right",
                    range=[0, 1.05], tickformat=".0%"),
        height=440, width=950,
    )
    return _srs_finalize(fig, legend="right")


def _carrier_breakdown(sweep: pd.DataFrame, grid: str) -> go.Figure:
    long = sweep.melt(
        id_vars=["cp_id", "kind"],
        value_vars=["power_shed", "heat_shed", "gas_shed"],
        var_name="carrier", value_name="shed_mw",
    )
    long["carrier"] = long["carrier"].str.replace("_shed", "", regex=False).map(
        {"power": "electricity", "heat": "heat", "gas": "gas"}
    )
    # Log axis demands strictly positive values — same filter as the pooled
    # variant; most branches shed exactly 0 on foreign carriers.
    long = long[long["shed_mw"] > 0]
    fig = px.box(
        long, x="carrier", y="shed_mw", color="carrier", points="outliers",
        color_discrete_map=ev.NETWORK_COLOR_MAP,
        title=f"{pretty_scenario(grid)}: per-carrier shed distribution across all single removals",
    )
    fig.update_layout(yaxis_type="log", yaxis_title="Shed (MW, log)", height=420, width=750)
    return _srs_finalize(fig, legend="right")


def _kind_summary(sweep: pd.DataFrame, grid: str) -> go.Figure:
    g = sweep.groupby("kind")["total_shed"].agg(["count", "mean", "max", "sum"]).reset_index()
    # Stable kind order on the x-axis so reading left→right is consistent
    # across grids (a grid without compounds still leaves the remaining
    # kinds in their canonical slot rather than collapsing left).
    kinds_present = set(g["kind"].dropna().unique())
    kind_order = [k for k in KIND_ORDER if k in kinds_present]
    g["_order"] = g["kind"].map({k: i for i, k in enumerate(kind_order)})
    g = g.sort_values("_order").drop(columns="_order")
    fig = go.Figure(data=[
        go.Bar(name="mean total_shed", x=g["kind"], y=g["mean"],
               marker=pub_style.bar_marker(pub_style.QUAL_PALETTE[0],
                                           pattern_shape=pub_style.PATTERN_SHAPES[0])),
        go.Bar(name="max total_shed", x=g["kind"], y=g["max"],
               marker=pub_style.bar_marker(pub_style.QUAL_PALETTE[1],
                                           pattern_shape=pub_style.PATTERN_SHAPES[1])),
    ])
    # Kept vertical: few (ordinal) component kinds, read left-to-right.
    fig.update_layout(barmode="group")
    pub_style.apply_theme(
        fig, title=f"{pretty_scenario(grid)}: total shed by component kind (mean vs max)",
        height=400, width=pub_style.vbar_width(len(kind_order), 2, base=480),
        font_bump=1, legend_top=True,
    )
    fig.update_xaxes(title="Component kind", categoryorder="array",
                     categoryarray=kind_order)
    fig.update_yaxes(title="Total shed (MW)")
    return fig


def _solve_time(sweep: pd.DataFrame, grid: str) -> go.Figure:
    fig = px.histogram(
        sweep, x="elapsed_s", nbins=40,
        color_discrete_sequence=[ev.PALETTE_QUAL[2]],
        title=f"{pretty_scenario(grid)}: solve-time distribution per single-removal LP",
    )
    fig.update_layout(
        xaxis_title="Per-component solve time (s)",
        yaxis_title="Components (count)",
        height=380, width=750, showlegend=False,
    )
    return _srs_finalize(fig, legend="none")


# ─────────────────────────────────────────────────────────────────────────────
# Pooled (cross-grid) comparison plots
# ─────────────────────────────────────────────────────────────────────────────


def _grid_order(grids: List[str]) -> List[str]:
    """Stable display order: SCENARIO_NAME_MAP order first, then unknown
    grids alphabetically. Keeps the colour-per-grid assignment consistent
    even when only a subset of scenarios is plotted."""
    known = [g for g in SCENARIO_NAME_MAP if g in set(grids)]
    rest = sorted(set(grids) - set(known))
    return known + rest


# Canonical scenario-family helpers live in eval_common so every
# cross-scenario plotter shares the same partitioning logic.
from eval_common import (  # noqa: E402
    family_label as _family_label,
    scenario_family as _scenario_family,
    scenario_stem as _grid_base,
    split_scenarios_by_family as _split_scenarios_by_family,
)

# Per-family line dash / opacity so grids sharing a density stem (and hence
# a hue, see _grid_color_map) stay visually distinguishable across families.
_FAMILY_DASH = {
    "backup": "solid", "loadbearing": "dash", "decoupled": "dashdot",
    "control": "dot",
}
_FAMILY_OPACITY = {
    "backup": 1.0, "loadbearing": 0.7, "decoupled": 0.55, "control": 0.45,
}


def _grid_color_map(grids: List[str]) -> Dict[str, str]:
    """Stable colour per grid, with all family variants of a density stem
    sharing a hue so the eye groups them. Differentiation between family
    members is carried by ``_grid_dash`` (Scatter lines) and
    ``_grid_opacity`` (Bars / Boxes). Palette indexes by *stem* in
    first-seen order so adding families doesn't shift colours.

    Uses the publication ``QUAL_PALETTE`` (blues/purples/teals/…) rather than
    the cmres carrier-anchored palette so a grid-coloured bar never reads as a
    sector bar."""
    palette = list(pub_style.QUAL_PALETTE)
    bases_in_order: List[str] = []
    seen: set = set()
    for g in grids:
        b = _grid_base(g)
        if b not in seen:
            bases_in_order.append(b)
            seen.add(b)
    base_color = {b: palette[i % len(palette)] for i, b in enumerate(bases_in_order)}
    return {g: base_color[_grid_base(g)] for g in grids}


def _grid_dash(g: str) -> str:
    """Line dash style per scenario family. Use on Scatter ``line`` to keep
    same-stem curves distinguishable."""
    return _FAMILY_DASH.get(_scenario_family(g), "solid")


def _grid_opacity(g: str) -> float:
    """Marker / bar opacity per scenario family so boxes / bars sharing a
    hue stay visually distinguishable."""
    return _FAMILY_OPACITY.get(_scenario_family(g), 1.0)


def _pooled_baseline(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                     grids: List[str]) -> go.Figure:
    """Stacked-bar comparison of baseline shed per grid, broken down by carrier.

    Anchors every other pooled figure: the absolute scale of shed depends on
    how much load the grid can already not serve at baseline.
    """
    rows = []
    for g in grids:
        _, baseline = records[g]
        if baseline is None:
            continue
        rows.append({
            "grid": g,
            "electricity": float(baseline.get("power_shed", 0.0)),
            "heat":        float(baseline.get("heat_shed", 0.0)),
            "gas":         float(baseline.get("gas_shed", 0.0)),
            "total":       float(baseline.get("total_shed", 0.0)),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return go.Figure()
    # Order by total baseline shed ascending so the most-resilient grid sits
    # at the bottom of the bar chart.
    df = df.sort_values("total")
    fig = go.Figure()
    for carrier in ("electricity", "heat", "gas"):
        fig.add_trace(go.Bar(
            y=[_short_grid(g) for g in df["grid"]],
            x=df[carrier], name=pub_style.SECTOR_PRETTY[carrier], orientation="h",
            marker=pub_style.sector_marker(carrier),
            hovertemplate=("<b>%{y}</b><br>" + pub_style.SECTOR_PRETTY[carrier]
                           + ": %{x:.4f} MW<extra></extra>"),
        ))
    fig.update_layout(barmode="stack")
    pub_style.apply_theme(
        fig, title="Baseline shed per grid (no fault), stacked by carrier",
        height=pub_style.hbar_height(len(df), 3), width=pub_style.BAR_FIG_WIDTH,
        font_bump=1, legend_top=True,
    )
    fig.update_xaxes(title="Baseline load shed (MW)")
    fig.update_yaxes(title="")
    return fig


def _pooled_total_shed_box(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                           grids: List[str]) -> go.Figure:
    """One box per grid showing the distribution of total_shed across all
    single-component removals. Baseline overlaid as a red marker so the
    "no fault" anchor is visible relative to the spread."""
    fig = go.Figure()
    color = _grid_color_map(grids)
    # Floor/window for the log value-axis: near-zero shed (≈1e-18 on components
    # that cause no curtailment) and a near-zero baseline would otherwise stretch
    # the log axis across ~18 decades and squash the boxes. Clamp to the
    # meaningful (>1e-6 MW) shed window so the distributions stay legible.
    _allv = (pd.concat([records[g][0]["total_shed"]
                        for g in grids if not records[g][0].empty])
             if grids else pd.Series(dtype=float))
    _real = _allv[_allv > 1e-6]
    _floor = float(_real.min()) if len(_real) else 1e-4
    _hi = float(_real.max()) * 1.6 if len(_real) else 1.0
    for g in grids:
        sweep, baseline = records[g]
        if sweep.empty:
            continue
        fig.add_trace(go.Box(
            x=sweep["total_shed"], y=[_short_grid(g)] * len(sweep),
            name=_short_grid(g), marker_color=color[g], orientation="h",
            opacity=_grid_opacity(g),
            boxpoints="outliers", showlegend=False,
            hovertemplate=("<b>" + _short_grid(g)
                           + "</b><br>total_shed = %{x:.4f} MW<extra></extra>"),
        ))
        if baseline is not None:
            fig.add_trace(go.Scatter(
                x=[max(float(baseline["total_shed"]), _floor)], y=[_short_grid(g)],
                mode="markers", marker=dict(color="#e45756", symbol="diamond",
                                            size=10, line=dict(color="#222", width=0.6)),
                name="baseline", legendgroup="baseline",
                showlegend=(g == grids[0]),
                hovertemplate=("<b>" + _short_grid(g)
                               + "</b><br>baseline = %{x:.4f} MW<extra></extra>"),
            ))
    # Horizontal to align with the other pooled bar figures; grid on the
    # category (y) axis, shed on the log value (x) axis.
    fig.update_layout(
        title="Total shed across single-component removals — distribution per grid",
        xaxis_title="Total load shed (MW)", yaxis_title="Grid",
        xaxis_type="log",
        height=max(360, 70 * len(grids) + 180), width=pub_style.BAR_FIG_WIDTH,
    )
    fig.update_xaxes(range=[float(np.log10(_floor * 0.7)), float(np.log10(_hi))])
    fig.update_yaxes(autorange="reversed")
    return _srs_finalize(fig, legend="right")


def _pooled_excess_shed_box(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                            grids: List[str], conn: bool = False) -> go.Figure:
    """One box per grid of (total_shed − baseline_total_shed), restricted to
    components that actually cause extra shed (delta > 0). Makes the "how
    much worse can it get under fault?" comparison directly readable.

    ``conn=True`` uses the ``total_shed_conn`` column instead — the
    curtailment of loads that remain connected after the removal. Since
    disconnected loads are unrecoverable, the connected-only excess is the
    part of the fault response that dispatch (and hence CPs) can influence.
    """
    col = "total_shed_conn" if conn else "total_shed"
    fig = go.Figure()
    color = _grid_color_map(grids)
    n_total = []
    for g in grids:
        sweep, baseline = records[g]
        if sweep.empty or col not in sweep.columns:
            continue
        base_t = float(baseline.get(col, 0.0)) if baseline is not None else 0.0
        delta = sweep[col] - base_t
        positive = delta[delta > 1e-9]
        n_total.append((g, int(len(positive)), int(len(sweep))))
        if positive.empty:
            continue
        fig.add_trace(go.Box(
            x=positive, y=[_short_grid(g)] * len(positive),
            name=_short_grid(g), marker_color=color[g], orientation="h",
            opacity=_grid_opacity(g),
            boxpoints="outliers", showlegend=False,
            hovertemplate=("<b>" + _short_grid(g)
                           + "</b><br>Δ shed = %{x:.4f} MW<extra></extra>"),
        ))
    label = "connected-load shed" if conn else "total_shed"
    fig.update_layout(
        title=(f"Excess shed under fault: {label} − baseline per component"),
        yaxis_title="Grid",
        xaxis_title=("Excess connected shed Δ (MW, log)" if conn
                     else "Excess shed Δ (MW, log)"),
        xaxis_type="log",
        height=max(360, 70 * len(grids) + 180), width=pub_style.BAR_FIG_WIDTH,
    )
    fig.update_yaxes(autorange="reversed")
    # Annotate per-grid n_positive/n_total so the reader doesn't compare
    # boxes of wildly different sample size as if they were equivalent.
    # annotations = [
    #     dict(x=pretty_scenario(g), y=1.02, xref="x", yref="paper",
    #          text=f"{n}/{tot}", showarrow=False,
    #          font=dict(size=_SRS_FONT_SIZES["annotation"], color="#444"))
    #     for g, n, tot in n_total
    # ]
    # fig.update_layout(annotations=annotations)
    return _srs_finalize(fig, legend="right")


def _pooled_pareto(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                   grids: List[str]) -> go.Figure:
    """Pareto cumulative-share curves overlaid: rank (sorted desc by
    total_shed) → cumulative share of total shed mass. Steeper curves =
    more concentrated criticality (a few components dominate)."""
    fig = go.Figure()
    color = _grid_color_map(grids)
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty or sweep["total_shed"].sum() <= 0:
            continue
        s = sweep.sort_values("total_shed", ascending=False).reset_index(drop=True)
        ranks = np.arange(1, len(s) + 1)
        cum_share = s["total_shed"].cumsum() / max(s["total_shed"].sum(), 1e-12)
        fig.add_trace(go.Scatter(
            x=ranks, y=cum_share, mode="lines",
            name=_short_grid(g),
            line=dict(color=color[g], width=2, dash=_grid_dash(g)),
            hovertemplate=("<b>" + _short_grid(g)
                           + "</b><br>rank %{x}<br>cum. share %{y:.1%}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        title="Pareto curves overlaid: how concentrated is shed across components?",
        xaxis=dict(title="Component rank (desc by total_shed)", type="log"),
        yaxis=dict(title="Cumulative share of total shed",
                   range=[0, 1.02], tickformat=".0%"),
        height=480, width=900,
    )
    return _srs_finalize(fig, legend="right")


def _pooled_carrier_box(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                        grids: List[str]) -> go.Figure:
    """Per-carrier shed distribution, grouped boxes — x = carrier, colour =
    grid. Lets you see which grid stresses which sector most under faults."""
    rows = []
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty:
            continue
        long = sweep.melt(
            value_vars=["power_shed", "heat_shed", "gas_shed"],
            var_name="carrier", value_name="shed_mw",
        )
        long["carrier"] = long["carrier"].str.replace("_shed", "", regex=False).map(
            {"power": "electricity", "heat": "heat", "gas": "gas"}
        )
        long["grid"] = _short_grid(g)
        rows.append(long)
    if not rows:
        return go.Figure()
    df = pd.concat(rows, ignore_index=True)
    df = df[df["shed_mw"] > 0]  # log axis demands strictly positive
    fig = px.box(
        df, x="shed_mw", y="carrier", color="grid", points=False, orientation="h",
        color_discrete_map={_short_grid(g): _grid_color_map(grids)[g]
                            for g in grids},
        category_orders={
            "carrier": ["electricity", "heat", "gas"],
            "grid": [_short_grid(g) for g in grids],
        },
        title="Per-carrier shed distribution per grid (zero-shed rows excluded)",
    )
    fig.update_layout(
        xaxis_type="log",
        yaxis_title="Carrier", xaxis_title="Shed (MW, log)",
        height=max(380, 90 * len(grids) + 200), width=pub_style.BAR_FIG_WIDTH,
    )
    fig = _srs_finalize(fig, legend="right")
    # Per-figure-only +2 pt bump on every text element (dissertation-print
    # requirement specific to this carrier-by-grid box plot).
    _srs_bump_all_fonts_by(fig, 2)
    return fig


def _pooled_kind_summary(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                         grids: List[str]) -> go.Figure:
    """Mean total_shed per (grid × kind), grouped bars. Reads the "which
    kind of component drives shed?" question across grids in one view."""
    rows = []
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty or "kind" not in sweep.columns:
            continue
        agg = sweep.groupby("kind")["total_shed"].agg(["mean", "max", "count"]).reset_index()
        agg["grid"] = g
        rows.append(agg)
    if not rows:
        return go.Figure()
    df = pd.concat(rows, ignore_index=True)
    kinds_present = [k for k in KIND_ORDER if k in set(df["kind"])]
    df = df[df["kind"].isin(kinds_present)]
    color = _grid_color_map(grids)
    fig = go.Figure()
    for g in grids:
        sub = df[df["grid"] == g].set_index("kind").reindex(kinds_present).reset_index()
        if sub["mean"].notna().sum() == 0:
            continue
        fig.add_trace(go.Bar(
            name=pretty_scenario(g),
            x=sub["kind"], y=sub["mean"],
            marker=pub_style.bar_marker(color[g]),
            opacity=_grid_opacity(g),
            hovertemplate=("<b>" + pretty_scenario(g)
                           + "</b><br>kind = %{x}<br>mean shed = %{y:.4f} MW"
                           "<br>n = %{customdata[0]}<extra></extra>"),
            customdata=np.c_[sub["count"].fillna(0).astype(int).values],
        ))
    fig.update_layout(barmode="group")
    pub_style.apply_theme(
        fig, title="Mean total shed by component kind, per grid",
        height=460,
        width=min(pub_style._FIG_WIDTH,
                  max(560, 70 * max(1, len(kinds_present) * len(grids)) + 200)),
        font_bump=1, legend_top=True,
    )
    fig.update_xaxes(title="Component kind", categoryorder="array",
                     categoryarray=kinds_present)
    fig.update_yaxes(title="Mean total shed (MW)")
    return fig


def _pooled_mean_shed_by_carrier(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                                 grids: List[str],
                                 x_max: float | None = None) -> go.Figure:
    """Average per-component shed across all grids, grouped bars per carrier.

    For each (grid, carrier) pair, plots the mean of the carrier's shed
    column across all single-component removals. Reads "which sector tends
    to lose load when an arbitrary component drops?" directly per grid.

    The mean is ``sum(shed) / n_runs`` where ``n_runs = len(sweep)`` is the
    number of single-component removals attempted — so every bar here is
    exactly the matching :func:`_pooled_total_shed_by_carrier` bar divided by
    the run count. Dividing by ``len(sweep)`` (not pandas ``.mean()``, which
    silently drops NaN survivors) keeps the denominator equal to the ``n``
    shown on hover even if some removals fail to solve; a failed removal then
    contributes 0 to the numerator (its shed is unknown), so the mean is a
    lower bound when the sweep is not fully clean. On a clean sweep (every
    removal solves) this is identical to ``sweep[col].mean()``.
    """
    carriers = ("electricity", "heat", "gas")
    col_map = {"electricity": "power_shed", "heat": "heat_shed", "gas": "gas_shed"}
    rows = []
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty:
            continue
        n_runs = len(sweep)
        for c in carriers:
            col = col_map[c]
            if col not in sweep.columns:
                continue
            rows.append({
                "grid": g,
                "carrier": c,
                "mean_shed": float(sweep[col].sum() / n_runs),
                "n": int(n_runs),
            })
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for c in carriers:
        sub = df[df["carrier"] == c]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=pub_style.SECTOR_PRETTY[c],
            y=[_short_grid(g) for g in sub["grid"]],
            x=sub["mean_shed"],
            orientation="h",
            marker=pub_style.sector_marker(c),
            hovertemplate=("<b>%{y}</b><br>" + pub_style.SECTOR_PRETTY[c]
                           + ": mean = %{x:.4f} MW"
                           "<br>over %{customdata[0]} removals<extra></extra>"),
            customdata=np.c_[sub["n"].values],
        ))
    # Stacked horizontal: bar length = mean *total* shed per component, so grids
    # stay directly comparable while the segments show the per-sector
    # composition. Horizontal to align with the other pooled bar figures.
    fig.update_layout(barmode="stack")
    pub_style.apply_theme(
        fig,
        title=f"Mean per-component load shed by carrier — {SINGLE_REMOVAL_HINT}",
        height=pub_style.hbar_height(len(grids), 3),
        width=pub_style.BAR_FIG_WIDTH, font_bump=1, legend_top=True,
    )
    # Plotly reverses the legend on stacked bars; on a *horizontal* stack that
    # reads backwards against the row figure's legend, which is not reversed.
    fig.update_layout(legend_traceorder="normal")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Mean shed per component (MW)",
                     range=[0, x_max * 1.06] if x_max else None)
    return fig


def _pooled_total_shed_by_carrier(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                                  grids: List[str], conn: bool = False,
                                  x_max: float | None = None) -> go.Figure:
    """Total shed summed over all single-component removals, grouped bars per
    carrier.

    Mirrors :func:`_pooled_mean_shed_by_carrier` but sums each carrier's shed
    column instead of averaging it, so bar length is the grid's cumulative
    shed mass across the whole sweep rather than the per-component mean.
    Unlike the mean view this scales with the number of components swept, so
    grids with more removals contribute proportionally more.

    ``conn=True`` sums the ``*_shed_conn`` columns — curtailment of loads that
    stay *connected* after the removal, excluding the topologically islanded
    nameplate that no dispatch can recover. This is the only shed dispatch (and
    hence CP capacity) can influence, so the connected view is where a CP-density
    effect is legible: on the study grids the disconnected/islanded mass swamps
    ``total_shed`` and is CP-count-invariant, whereas this recoverable mass
    shrinks as CPs are added.
    """
    carriers = ("electricity", "heat", "gas")
    suffix = "_conn" if conn else ""
    col_map = {"electricity": f"power_shed{suffix}", "heat": f"heat_shed{suffix}",
               "gas": f"gas_shed{suffix}"}
    rows = []
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty:
            continue
        for c in carriers:
            col = col_map[c]
            if col not in sweep.columns:
                continue
            rows.append({
                "grid": g,
                "carrier": c,
                "total_shed": float(sweep[col].sum()),
                "n": int(len(sweep)),
            })
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for c in carriers:
        sub = df[df["carrier"] == c]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=pub_style.SECTOR_PRETTY[c],
            y=[_short_grid(g) for g in sub["grid"]],
            x=sub["total_shed"],
            orientation="h",
            marker=pub_style.sector_marker(c),
            hovertemplate=("<b>%{y}</b><br>" + pub_style.SECTOR_PRETTY[c]
                           + ": total = %{x:.4f} MW"
                           "<br>n = %{customdata[0]}<extra></extra>"),
            customdata=np.c_[sub["n"].values],
        ))
    # Stacked horizontal: bar length = total shed summed over the sweep, so the
    # segments show the per-sector composition while grids stay comparable.
    fig.update_layout(barmode="stack")
    pub_style.apply_theme(
        fig,
        title=(("Total connected-load (recoverable) shed by carrier" if conn
                else "Total load shed by carrier")
               + f" — {SINGLE_REMOVAL_HINT}"),
        height=pub_style.hbar_height(len(grids), 3),
        width=pub_style.BAR_FIG_WIDTH, font_bump=1, legend_top=True,
    )
    fig.update_layout(legend_traceorder="normal")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title=("Total connected-load shed (MW)" if conn
                            else "Total shed (MW)"),
                     range=[0, x_max * 1.06] if x_max else None)
    return fig


def _pooled_mean_shed_by_carrier_row(
        records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        classes: List[Tuple[str, List[str]]]) -> go.Figure:
    """Combined cross-family view: one horizontal stacked-bar panel per scenario
    family (eval_common.FAMILY_ORDER) sharing a single legend, so the
    strategies sit in one row in the dissertation. Mirrors
    :func:`_pooled_mean_shed_by_carrier` but across families."""
    fams = sorted(classes, key=lambda c: _FAMILY_ORDER.get(c[0], 99))
    fig = make_subplots(rows=1, cols=len(fams), shared_xaxes=False,
                        horizontal_spacing=0.06,
                        subplot_titles=[_family_label(f) for f, _ in fams])
    for ci, (_fam, subset) in enumerate(fams, start=1):
        grids = _grid_order(subset)
        y = [_short_grid(g) for g in grids]
        for sec, col in _CARRIER_COLS:
            x = []
            for g in grids:
                sweep, _ = records[g]
                x.append(float(sweep[col].sum() / len(sweep))
                         if (not sweep.empty and col in sweep.columns) else 0.0)
            fig.add_trace(go.Bar(
                y=y, x=x, orientation="h",
                name=pub_style.SECTOR_PRETTY[sec], legendgroup=sec,
                showlegend=(ci == 1), marker=pub_style.sector_marker(sec),
            ), row=1, col=ci)
    fig.update_layout(barmode="stack", bargap=0.25)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Mean shed per component (MW)",
                     row=1, col=(len(fams) + 1) // 2)
    pub_style.apply_theme(
        fig, title=("Mean per-component load shed by carrier, per density and "
                    f"strategy — {SINGLE_REMOVAL_HINT}"),
        width=1180, height=430, legend_top=True, font_bump=8)
    pub_style.clear_subplot_titles(fig)
    # Same bottom-margin correction as the E1 row twin: apply_theme sizes the
    # bottom margin for the unbumped axis typography, so at this bump the
    # x-axis title runs off the canvas unless margin and height grow together.
    fig.update_layout(height=(fig.layout.height or 430) + 44,
                      margin=dict(b=(fig.layout.margin.b or 64) + 44))
    # One value scale across the panels: the row layout exists to compare a
    # density across strategies, which per-panel autoranging defeats.
    all_grids = [g for _f, subset in fams for g in subset]
    fig.update_xaxes(
        range=[0, _stacked_x_max(records, all_grids, mean=True) * 1.06])
    return fig


def _pooled_solve_time(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                       grids: List[str]) -> go.Figure:
    """One box per grid of per-component solve time. Helps spot grids where
    the LP is slower to solve (typically denser CPs / bigger ext-grid bounds).
    """
    fig = go.Figure()
    color = _grid_color_map(grids)
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty or "elapsed_s" not in sweep.columns:
            continue
        fig.add_trace(go.Box(
            x=sweep["elapsed_s"], y=[_short_grid(g)] * len(sweep),
            name=_short_grid(g), marker_color=color[g], orientation="h",
            opacity=_grid_opacity(g),
            boxpoints="outliers", showlegend=False,
            hovertemplate=("<b>" + _short_grid(g)
                           + "</b><br>elapsed = %{x:.2f} s<extra></extra>"),
        ))
    fig.update_layout(
        title="Per-component LP solve time, per grid",
        yaxis_title="Grid", xaxis_title="Elapsed (s)",
        height=max(360, 70 * len(grids) + 180), width=pub_style.BAR_FIG_WIDTH,
    )
    fig.update_yaxes(autorange="reversed")
    return _srs_finalize(fig, legend="right")


def _pooled_top_components(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                           grids: List[str], top_n: int = 10) -> go.Figure:
    """Horizontal grouped bars of the top-N components per grid, labelled by
    cp_id. Each grid contributes ``top_n`` rows; the y-axis is "<grid> ·
    <cp_id>" so the same component appearing in multiple grids stays
    distinguishable."""
    rows = []
    for g in grids:
        sweep, _ = records[g]
        if sweep.empty:
            continue
        top = sweep.sort_values("total_shed", ascending=False).head(top_n).iloc[::-1]
        for _, r in top.iterrows():
            rows.append({
                "grid": g,
                "label": f"{pretty_scenario(g)} · {r['cp_id']}",
                "kind":  r.get("kind", "branch"),
                "total_shed": float(r["total_shed"]),
                "power": float(r.get("power_shed", 0.0)),
                "heat":  float(r.get("heat_shed", 0.0)),
                "gas":   float(r.get("gas_shed", 0.0)),
            })
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for carrier, sector in (("power", "electricity"), ("heat", "heat"),
                            ("gas", "gas")):
        fig.add_trace(go.Bar(
            x=df[carrier], y=df["label"], orientation="h",
            name=pub_style.SECTOR_PRETTY[sector],
            marker=pub_style.sector_marker(sector),
            hovertemplate=("<b>%{y}</b><br>" + pub_style.SECTOR_PRETTY[sector]
                           + ": %{x:.4f} MW<extra></extra>"),
        ))
    fig.update_layout(barmode="stack")
    pub_style.apply_theme(
        fig, title=f"Top-{top_n} components per grid (per-carrier breakdown)",
        height=pub_style.hbar_height(len(df), 3), width=pub_style.BAR_FIG_WIDTH,
        font_bump=1, legend_top=True,
    )
    fig.update_xaxes(title="Load shed (MW)")
    fig.update_yaxes(title="Grid · component")
    return fig


def _emit_pooled_report(
    records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
    grids: List[str],
    out_dir: Path,
    class_label: str = "",
    x_max: Dict[str, float] | None = None,
) -> Path:
    """Build the pooled cross-grid figures for one scenario-family subset and
    write them to ``pooled[_<class>]_report.html`` plus per-figure PDFs
    under ``single/``. ``class_label=""`` keeps the legacy filename so
    runs with only baseline grids stay byte-identical to before.

    ``x_max`` carries the cross-family value-axis maxima (see
    :func:`_stacked_x_max`) so the per-strategy shed bars printed next to
    each other in the chapter share one scale.
    """
    x_max = x_max or {}
    figs: List[go.Figure] = [
        _pooled_baseline(records, grids),
        _pooled_total_shed_box(records, grids),
        _pooled_excess_shed_box(records, grids),
        _pooled_pareto(records, grids),
        _pooled_carrier_box(records, grids),
        _pooled_mean_shed_by_carrier(records, grids, x_max=x_max.get("mean")),
        _pooled_total_shed_by_carrier(records, grids, x_max=x_max.get("total")),
        _pooled_kind_summary(records, grids),
        _pooled_top_components(records, grids, top_n=10),
        _pooled_solve_time(records, grids),
    ]
    titles = [
        "Baseline shed per grid",
        "Total-shed distribution per grid",
        "Excess shed (delta over baseline) per grid",
        "Pareto curves overlaid",
        "Per-carrier shed by grid",
        "Mean per-component shed by sector",
        "Total shed by sector",
        "Mean total shed by kind, per grid",
        "Top-10 components per grid",
        "Solve-time per grid",
    ]
    slug_names = [
        "baseline",
        "total_shed_box",
        "excess_shed_box",
        "pareto_overlay",
        "carrier_box",
        "mean_shed_by_carrier",
        "total_shed_by_carrier",
        "kind_summary",
        "top10_components",
        "solve_time",
    ]
    # Connected-only excess view — needs the ``*_shed_conn`` columns
    # recorded by current single_removal_shed.py runs; skipped silently on
    # legacy CSVs so old result sets still render.
    if any("total_shed_conn" in s.columns for s, _ in records.values()):
        figs.insert(3, _pooled_excess_shed_box(records, grids, conn=True))
        titles.insert(3, "Excess connected-load shed (delta over baseline) per grid")
        slug_names.insert(3, "excess_shed_conn_box")
        # Absolute connected-load (recoverable) shed by carrier, next to its
        # total-shed twin. total_shed is dominated by the topologically islanded
        # nameplate, which no dispatch can recover and is CP-count-invariant;
        # this recoverable slice is the only part CPs move, so it's where the
        # CP-density effect reads directly (it shrinks as CPs are added). The
        # insert(3) above shifted total_shed_by_carrier from index 6 to 7, so
        # the connected twin lands at 8 (immediately after it).
        figs.insert(8, _pooled_total_shed_by_carrier(
            records, grids, conn=True, x_max=x_max.get("total_conn")))
        titles.insert(8, "Total connected-load (recoverable) shed by sector")
        slug_names.insert(8, "total_shed_conn_by_carrier")
    suffix = f"_{class_label}" if class_label else ""
    slugs = [f"pooled{suffix}_{s}" for s in slug_names]
    out_html = out_dir / f"pooled{suffix}_report.html"
    heading = (
        f"single-removal shed — pooled ({class_label}, {len(grids)} grids)"
        if class_label
        else "single-removal shed — pooled across grids"
    )
    ev.write_all_in_one(figs, heading, Path("."), str(out_html),
                        write_single_files=True, titles=titles, slugs=slugs)
    print(f"  -> {out_html}")
    return out_html


def plot_pooled(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                out_dir: Path) -> Path:
    """Pooled cross-grid comparison report.

    ``records`` maps grid name → ``(sweep_df, baseline_row)`` from ``_load``.
    When the record set spans several scenario families the output is
    split into one report per family (``pooled_backup_*``,
    ``pooled_loadbearing_*``, ``pooled_control_*``); otherwise a single ``pooled_*``
    filenames are kept for backwards compatibility. Returns the path of
    the last report written (mostly for the existing single-class call
    sites that ignore the return value).
    """
    grids = _grid_order(list(records))
    classes = _split_scenarios_by_family(grids)

    if len(classes) <= 1:
        # Legacy / single-class run — emit unsuffixed filenames.
        return _emit_pooled_report(records, grids, out_dir, class_label="")

    # Value-axis maxima over *every* family, so the per-strategy figures the
    # chapter prints side by side are comparable bar-for-bar.
    x_max = {
        "mean": _stacked_x_max(records, grids, mean=True),
        "total": _stacked_x_max(records, grids, mean=False),
        "total_conn": _stacked_x_max(records, grids, mean=False, suffix="_conn"),
    }
    # Mixed run — one report per scenario family so each stays compact.
    last_path: Path | None = None
    for class_label, subset in classes:
        sub_records = {g: records[g] for g in subset}
        last_path = _emit_pooled_report(sub_records, subset, out_dir,
                                        class_label=class_label, x_max=x_max)
    # Combined cross-family row used in the dissertation: the three strategies
    # side by side, horizontal stacked bars, one shared legend.
    try:
        row_fig = _pooled_mean_shed_by_carrier_row(records, classes)
        single_dir = out_dir / "single"
        single_dir.mkdir(parents=True, exist_ok=True)
        row_fig.write_image(str(single_dir / "00_pooled_mean_shed_by_carrier_row.pdf"))
    except Exception as e:  # pragma: no cover
        print(f"  mean_shed_by_carrier_row skipped: {e}")
    return last_path  # type: ignore[return-value]


def plot_grid(grid: str, csv_path: Path, out_dir: Path):
    sweep, baseline = _load(csv_path)
    base_total = float(baseline["total_shed"]) if baseline is not None else 0.0

    figs: List[go.Figure] = [
        _hist_total(sweep, grid, base_total),
        _top_components(sweep, grid, top_n=20),
        _pareto(sweep, grid),
        _carrier_breakdown(sweep, grid),
        _kind_summary(sweep, grid),
        _solve_time(sweep, grid),
    ]
    titles = [
        "Distribution of total shed",
        "Top-20 components",
        "Shed Pareto curve",
        "Per-carrier shed distribution",
        "Total shed by component kind",
        "Per-component solve time",
    ]
    slug_names = [
        "hist_total_shed",
        "top20_components",
        "pareto",
        "per_carrier_box",
        "by_kind",
        "solve_time",
    ]
    # Connected-only view (curtailment of still-connected loads, excluding
    # the unrecoverable nameplate of disconnected ones). Needs the
    # ``*_shed_conn`` columns; skipped on legacy CSVs.
    if "total_shed_conn" in sweep.columns:
        figs.insert(2, _top_components(sweep, grid, top_n=20, conn=True))
        titles.insert(2, "Top-20 components by connected-load shed")
        slug_names.insert(2, "top20_components_conn")
    # Namespace slugs by grid so multiple grids in the same `--dir` don't
    # clobber each other's per-figure PDFs under ``single/``.
    slugs = [f"{grid}_{s}" for s in slug_names]
    out_html = out_dir / f"{grid}_report.html"
    ev.write_all_in_one(figs, f"single-removal shed — {pretty_scenario(grid)}", Path("."), str(out_html),
                        write_single_files=True, titles=titles, slugs=slugs)
    print(f"  -> {out_html}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("grids", nargs="*", help="grid names; default = every CSV in --dir")
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help=f"shed CSV directory (default: {DEFAULT_DIR})")
    ap.add_argument("--no-pooled", action="store_true",
                    help="skip the cross-grid pooled report (per-grid only)")
    ap.add_argument("--pooled-only", action="store_true",
                    help="emit only the cross-grid pooled report (skip per-grid)")
    args = ap.parse_args()

    # Skip ``*_shard_<i>_of_<k>.csv`` files — those are per-shard outputs from
    # the slurm run that haven't been merged yet. They share the prefix with
    # the merged CSV and would otherwise generate a noisy report per shard.
    grids = args.grids or [
        p.stem.replace("single_removal_shed_", "")
        for p in sorted(args.dir.glob("single_removal_shed_*.csv"))
        if "_shard_" not in p.stem
    ]
    if not grids:
        print(f"no shed CSVs in {args.dir}", file=sys.stderr)
        return 1
    print(f"plotting {len(grids)} grid(s): {grids}")

    records: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
    for g in grids:
        csv_path = args.dir / f"single_removal_shed_{g}.csv"
        if not csv_path.exists():
            print(f"  skip {g}: missing {csv_path}", file=sys.stderr)
            continue
        sweep, baseline = _load(csv_path)
        records[g] = (sweep, baseline)
        if not args.pooled_only:
            plot_grid(g, csv_path, args.dir)

    if not args.no_pooled and len(records) >= 2:
        plot_pooled(records, args.dir)
    elif not args.no_pooled and len(records) < 2:
        print("  pooled report skipped (need ≥2 grids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
