"""Plot single-removal load-shed results.

Reads ``data/out/single_removal_shed/single_removal_shed_<grid>.csv`` for
each grid name passed on the CLI (or all available CSVs if none given) and
writes one HTML/PDF report per grid:

  data/out/single_removal_shed/<grid>_report.html
  data/out/single_removal_shed/<grid>_report/single/*.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import cmres.evaluation.evaluation as ev  # noqa: E402

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

# Canonical kind→colour pinning so the legend stays consistent across grids:
# a grid without CPs (no ``compound`` / ``branch_cp`` rows) must still leave
# the ``branch`` colour unchanged, otherwise the visual reading flips
# between scenarios. ``_enumerate_targets`` in single_removal_shed.py emits
# exactly these three kinds; any extension MUST add an entry here.
KIND_COLOR_MAP = {
    "branch":    ev.PALETTE_QUAL[0],  # plain branches (PowerLine, GasPipe, …)
    "branch_cp": ev.PALETTE_QUAL[1],  # branch-level CPs (PowerToGas, …)
    "compound":  ev.PALETTE_QUAL[2],  # compound CPs (CHP, PowerToHeat, …)
}
# Stable category order so px doesn't re-order the legend by row count.
KIND_ORDER = ["branch", "branch_cp", "compound"]


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
        title=f"{grid}: distribution of total shed (MW) per single-component removal",
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
    return ev.apply_cmres_style(fig, legend="right")


def _top_components(sweep: pd.DataFrame, grid: str, top_n: int = 20) -> go.Figure:
    top = sweep.sort_values("total_shed", ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure()
    for carrier, color in zip(
        ("power_shed", "heat_shed", "gas_shed"),
        (ev.NETWORK_COLOR_MAP["electricity"], ev.NETWORK_COLOR_MAP["heat"], ev.NETWORK_COLOR_MAP["gas"]),
    ):
        fig.add_trace(go.Bar(
            x=top[carrier],
            y=top["cp_id"].astype(str),
            orientation="h",
            name=carrier.replace("_shed", ""),
            marker_color=color,
        ))
    fig.update_layout(
        barmode="stack",
        title=f"{grid}: top-{top_n} components by total shed (per-carrier breakdown)",
        xaxis_title="Load shed (MW)",
        yaxis_title="Component (cp_id)",
        height=600, width=950,
        margin=dict(l=200),
    )
    return ev.apply_cmres_style(fig, legend="right")


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
        title=f"{grid}: shed Pareto curve (rank vs total_shed and cumulative share)",
        xaxis=dict(title="Component rank (sorted by total_shed desc)", type="log"),
        yaxis=dict(title="Total shed (MW)"),
        yaxis2=dict(title="Cumulative share of total shed", overlaying="y", side="right",
                    range=[0, 1.05], tickformat=".0%"),
        height=440, width=950,
    )
    return ev.apply_cmres_style(fig, legend="right")


def _carrier_breakdown(sweep: pd.DataFrame, grid: str) -> go.Figure:
    long = sweep.melt(
        id_vars=["cp_id", "kind"],
        value_vars=["power_shed", "heat_shed", "gas_shed"],
        var_name="carrier", value_name="shed_mw",
    )
    long["carrier"] = long["carrier"].str.replace("_shed", "", regex=False).map(
        {"power": "electricity", "heat": "heat", "gas": "gas"}
    )
    fig = px.box(
        long, x="carrier", y="shed_mw", color="carrier", points="outliers",
        color_discrete_map=ev.NETWORK_COLOR_MAP,
        title=f"{grid}: per-carrier shed distribution across all single removals",
    )
    fig.update_layout(yaxis_type="log", yaxis_title="Shed (MW, log)", height=420, width=750)
    return ev.apply_cmres_style(fig, legend="right")


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
        go.Bar(name="mean total_shed", x=g["kind"], y=g["mean"], marker_color=ev.PALETTE_QUAL[0]),
        go.Bar(name="max total_shed",  x=g["kind"], y=g["max"],  marker_color=ev.PALETTE_QUAL[1]),
    ])
    fig.update_layout(
        barmode="group",
        title=f"{grid}: total shed by component kind (mean vs max)",
        xaxis_title="Component kind", xaxis=dict(categoryorder="array", categoryarray=kind_order),
        yaxis_title="Total shed (MW)",
        height=400, width=750,
    )
    return ev.apply_cmres_style(fig, legend="right")


def _solve_time(sweep: pd.DataFrame, grid: str) -> go.Figure:
    fig = px.histogram(
        sweep, x="elapsed_s", nbins=40,
        color_discrete_sequence=[ev.PALETTE_QUAL[2]],
        title=f"{grid}: solve-time distribution per single-removal LP",
    )
    fig.update_layout(
        xaxis_title="Per-component solve time (s)",
        yaxis_title="Components (count)",
        height=380, width=750, showlegend=False,
    )
    return ev.apply_cmres_style(fig, legend="none")


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


def _grid_color_map(grids: List[str]) -> Dict[str, str]:
    palette = list(ev.PALETTE_QUAL)
    return {g: palette[i % len(palette)] for i, g in enumerate(grids)}


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
            y=[pretty_scenario(g) for g in df["grid"]],
            x=df[carrier], name=carrier, orientation="h",
            marker_color=ev.NETWORK_COLOR_MAP[carrier],
            hovertemplate=("<b>%{y}</b><br>" + carrier
                           + ": %{x:.4f} MW<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack",
        title="Baseline shed per grid (no fault), stacked by carrier",
        xaxis_title="Baseline load shed (MW)", yaxis_title="",
        height=80 + 36 * max(1, len(df)), width=900,
    )
    return ev.apply_cmres_style(fig, legend="right")


def _pooled_total_shed_box(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                           grids: List[str]) -> go.Figure:
    """One box per grid showing the distribution of total_shed across all
    single-component removals. Baseline overlaid as a red marker so the
    "no fault" anchor is visible relative to the spread."""
    fig = go.Figure()
    color = _grid_color_map(grids)
    for g in grids:
        sweep, baseline = records[g]
        if sweep.empty:
            continue
        fig.add_trace(go.Box(
            y=sweep["total_shed"], x=[pretty_scenario(g)] * len(sweep),
            name=pretty_scenario(g), marker_color=color[g],
            boxpoints="outliers", showlegend=False,
            hovertemplate=("<b>" + pretty_scenario(g)
                           + "</b><br>total_shed = %{y:.4f} MW<extra></extra>"),
        ))
        if baseline is not None:
            fig.add_trace(go.Scatter(
                x=[pretty_scenario(g)], y=[float(baseline["total_shed"])],
                mode="markers", marker=dict(color="#e45756", symbol="diamond",
                                            size=10, line=dict(color="#222", width=0.6)),
                name="baseline", legendgroup="baseline",
                showlegend=(g == grids[0]),
                hovertemplate=("<b>" + pretty_scenario(g)
                               + "</b><br>baseline = %{y:.4f} MW<extra></extra>"),
            ))
    fig.update_layout(
        title="Total shed across single-component removals — distribution per grid",
        xaxis_title="Grid", yaxis_title="Total load shed (MW)",
        yaxis_type="log",
        height=520, width=max(720, 90 * len(grids) + 180),
    )
    return ev.apply_cmres_style(fig, legend="right")


def _pooled_excess_shed_box(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                            grids: List[str]) -> go.Figure:
    """One box per grid of (total_shed − baseline_total_shed), restricted to
    components that actually cause extra shed (delta > 0). Makes the "how
    much worse can it get under fault?" comparison directly readable."""
    fig = go.Figure()
    color = _grid_color_map(grids)
    n_total = []
    for g in grids:
        sweep, baseline = records[g]
        if sweep.empty:
            continue
        base_t = float(baseline["total_shed"]) if baseline is not None else 0.0
        delta = sweep["total_shed"] - base_t
        positive = delta[delta > 1e-9]
        n_total.append((g, int(len(positive)), int(len(sweep))))
        if positive.empty:
            continue
        fig.add_trace(go.Box(
            y=positive, x=[pretty_scenario(g)] * len(positive),
            name=pretty_scenario(g), marker_color=color[g],
            boxpoints="outliers", showlegend=False,
            hovertemplate=("<b>" + pretty_scenario(g)
                           + "</b><br>Δ shed = %{y:.4f} MW<extra></extra>"),
        ))
    fig.update_layout(
        title=("Excess shed under fault: total_shed − baseline per component "
               "(grids with no positive delta are blank)"),
        xaxis_title="Grid", yaxis_title="Excess shed Δ (MW, log)",
        yaxis_type="log",
        height=520, width=max(720, 90 * len(grids) + 180),
    )
    # Annotate per-grid n_positive/n_total so the reader doesn't compare
    # boxes of wildly different sample size as if they were equivalent.
    annotations = [
        dict(x=pretty_scenario(g), y=1.02, xref="x", yref="paper",
             text=f"{n}/{tot}", showarrow=False, font=dict(size=10, color="#444"))
        for g, n, tot in n_total
    ]
    fig.update_layout(annotations=annotations)
    return ev.apply_cmres_style(fig, legend="right")


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
            name=pretty_scenario(g),
            line=dict(color=color[g], width=2),
            hovertemplate=("<b>" + pretty_scenario(g)
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
    return ev.apply_cmres_style(fig, legend="right")


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
        long["grid"] = pretty_scenario(g)
        rows.append(long)
    if not rows:
        return go.Figure()
    df = pd.concat(rows, ignore_index=True)
    df = df[df["shed_mw"] > 0]  # log y demands strictly positive
    fig = px.box(
        df, x="carrier", y="shed_mw", color="grid", points=False,
        color_discrete_map={pretty_scenario(g): _grid_color_map(grids)[g]
                            for g in grids},
        category_orders={
            "carrier": ["electricity", "heat", "gas"],
            "grid": [pretty_scenario(g) for g in grids],
        },
        title="Per-carrier shed distribution per grid (zero-shed rows excluded)",
    )
    fig.update_layout(
        yaxis_type="log",
        xaxis_title="Carrier", yaxis_title="Shed (MW, log)",
        height=520, width=max(820, 100 * len(grids) + 240),
    )
    return ev.apply_cmres_style(fig, legend="right")


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
            marker_color=color[g],
            hovertemplate=("<b>" + pretty_scenario(g)
                           + "</b><br>kind = %{x}<br>mean shed = %{y:.4f} MW"
                           "<br>n = %{customdata[0]}<extra></extra>"),
            customdata=np.c_[sub["count"].fillna(0).astype(int).values],
        ))
    fig.update_layout(
        barmode="group",
        title="Mean total shed by component kind, per grid",
        xaxis=dict(title="Component kind",
                   categoryorder="array", categoryarray=kinds_present),
        yaxis_title="Mean total shed (MW)",
        height=460, width=max(640, 90 * max(1, len(kinds_present) * len(grids)) + 200),
    )
    return ev.apply_cmres_style(fig, legend="right")


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
            y=sweep["elapsed_s"], x=[pretty_scenario(g)] * len(sweep),
            name=pretty_scenario(g), marker_color=color[g],
            boxpoints="outliers", showlegend=False,
            hovertemplate=("<b>" + pretty_scenario(g)
                           + "</b><br>elapsed = %{y:.2f} s<extra></extra>"),
        ))
    fig.update_layout(
        title="Per-component LP solve time, per grid",
        xaxis_title="Grid", yaxis_title="Elapsed (s)",
        height=440, width=max(720, 90 * len(grids) + 180),
    )
    return ev.apply_cmres_style(fig, legend="right")


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
    for carrier, color in zip(
        ("power", "heat", "gas"),
        (ev.NETWORK_COLOR_MAP["electricity"],
         ev.NETWORK_COLOR_MAP["heat"],
         ev.NETWORK_COLOR_MAP["gas"]),
    ):
        fig.add_trace(go.Bar(
            x=df[carrier], y=df["label"], orientation="h",
            name=carrier, marker_color=color,
            hovertemplate=("<b>%{y}</b><br>" + carrier
                           + ": %{x:.4f} MW<extra></extra>"),
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Top-{top_n} components per grid (per-carrier breakdown)",
        xaxis_title="Load shed (MW)", yaxis_title="Grid · component",
        height=80 + 22 * max(1, len(df)), width=950,
        margin=dict(l=240),
    )
    return ev.apply_cmres_style(fig, legend="right")


def plot_pooled(records: Dict[str, Tuple[pd.DataFrame, pd.Series]],
                out_dir: Path) -> Path:
    """Pooled cross-grid comparison report.

    ``records`` maps grid name → ``(sweep_df, baseline_row)`` from ``_load``.
    Emits one HTML with the pooled panels plus per-figure PDFs under
    ``single/`` (slugs are prefixed with ``pooled_``).
    """
    grids = _grid_order(list(records))
    figs: List[go.Figure] = [
        _pooled_baseline(records, grids),
        _pooled_total_shed_box(records, grids),
        _pooled_excess_shed_box(records, grids),
        _pooled_pareto(records, grids),
        _pooled_carrier_box(records, grids),
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
        "Mean total shed by kind, per grid",
        "Top-10 components per grid",
        "Solve-time per grid",
    ]
    slugs = [f"pooled_{s}" for s in (
        "baseline",
        "total_shed_box",
        "excess_shed_box",
        "pareto_overlay",
        "carrier_box",
        "kind_summary",
        "top10_components",
        "solve_time",
    )]
    out_html = out_dir / "pooled_report.html"
    ev.write_all_in_one(figs, "single-removal shed — pooled across grids",
                        Path("."), str(out_html),
                        write_single_files=True, titles=titles, slugs=slugs)
    print(f"  -> {out_html}")
    return out_html


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
    # Namespace slugs by grid so multiple grids in the same `--dir` don't
    # clobber each other's per-figure PDFs under ``single/``.
    slugs = [f"{grid}_{s}" for s in (
        "hist_total_shed",
        "top20_components",
        "pareto",
        "per_carrier_box",
        "by_kind",
        "solve_time",
    )]
    out_html = out_dir / f"{grid}_report.html"
    ev.write_all_in_one(figs, f"single-removal shed — {grid}", Path("."), str(out_html),
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
