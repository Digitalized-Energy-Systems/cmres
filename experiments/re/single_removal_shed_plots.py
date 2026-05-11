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
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import cmres.evaluation.evaluation as ev  # noqa: E402

DEFAULT_DIR = Path("data/out/single_removal_shed")


def _load(csv_path: Path):
    df = pd.read_csv(csv_path)
    baseline = df[df["cp_id"] == "_baseline_"].iloc[0] if (df["cp_id"] == "_baseline_").any() else None
    sweep = df[df["cp_id"] != "_baseline_"].copy()
    sweep["delta_total"] = sweep["total_shed"] - (float(baseline["total_shed"]) if baseline is not None else 0.0)
    sweep["delta_power"] = sweep["power_shed"] - (float(baseline["power_shed"]) if baseline is not None else 0.0)
    sweep["delta_heat"] = sweep["heat_shed"] - (float(baseline["heat_shed"]) if baseline is not None else 0.0)
    sweep["delta_gas"] = sweep["gas_shed"] - (float(baseline["gas_shed"]) if baseline is not None else 0.0)
    return sweep, baseline


def _hist_total(sweep: pd.DataFrame, grid: str, baseline_total: float) -> go.Figure:
    fig = px.histogram(
        sweep,
        x="total_shed",
        color="kind",
        nbins=60,
        color_discrete_sequence=ev.PALETTE_QUAL,
        title=f"{grid}: distribution of total shed (MW) per single-component removal",
    )
    fig.add_vline(
        x=baseline_total,
        line_dash="dash",
        line_color=ev.PALETTE_QUAL[3],
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
    fig = go.Figure(data=[
        go.Bar(name="mean total_shed", x=g["kind"], y=g["mean"], marker_color=ev.PALETTE_QUAL[0]),
        go.Bar(name="max total_shed",  x=g["kind"], y=g["max"],  marker_color=ev.PALETTE_QUAL[1]),
    ])
    fig.update_layout(
        barmode="group",
        title=f"{grid}: total shed by component kind (mean vs max)",
        xaxis_title="Component kind",
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
    args = ap.parse_args()

    grids = args.grids or [
        p.stem.replace("single_removal_shed_", "")
        for p in sorted(args.dir.glob("single_removal_shed_*.csv"))
    ]
    if not grids:
        print(f"no shed CSVs in {args.dir}", file=sys.stderr)
        return 1
    print(f"plotting {len(grids)} grid(s): {grids}")
    for g in grids:
        csv_path = args.dir / f"single_removal_shed_{g}.csv"
        if not csv_path.exists():
            print(f"  skip {g}: missing {csv_path}", file=sys.stderr)
            continue
        plot_grid(g, csv_path, args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
