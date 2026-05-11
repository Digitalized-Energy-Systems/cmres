"""Dissertation-grade Plotly figures for the CMRES evaluation experiments.

Each `plot_eN_*` function reads the CSV(s) emitted by the matching
``experiment_eN_*`` in :mod:`cmres_eval` and writes a self-contained HTML
(and per-figure PDFs via :func:`evaluation.write_all_in_one`).

The plots are intentionally CSV-driven so they can be regenerated without
re-running the (expensive) MC pipeline. Use :func:`plot_all` to discover
every CSV in a CMRES output directory and emit every available figure.

Style conventions (matching the rest of the repo)
-------------------------------------------------
* template = ``plotly_white+publish3``
* carrier palette via :data:`evaluation.NETWORK_COLOR_MAP`
* scenario labels through :func:`cp_cn_evaluation.pretty_scenario`
* one HTML per experiment (via :func:`evaluation.write_all_in_one`) plus
  one PDF per panel under ``single/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cmres.evaluation.evaluation as eval

try:
    from cp_cn_evaluation import pretty_scenario
except Exception:  # pragma: no cover — script may be imported standalone
    def pretty_scenario(name):
        return "" if name is None else str(name)


# ─────────────────────────────────────────────────────────────────────────────
# Shared style — re-export the unified cmres palette/template from
# ``cmres.evaluation.evaluation`` so every figure here matches the rest of the
# eval pipeline. Keep the legacy module-level names (``TEMPLATE``, ``QUAL``,
# ``DIVERGING``, ``SEQUENTIAL``, ``CARRIER_COLORS``) as thin aliases so code
# inside this module reads naturally.
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE = eval.CMRES_TEMPLATE
CARRIER_COLORS = eval.NETWORK_COLOR_MAP | {"power": eval.NETWORK_COLOR_MAP["electricity"]}
DIVERGING = eval.PALETTE_DIVERGING
SEQUENTIAL = eval.PALETTE_SEQUENTIAL
# Extend the 10-colour qualitative palette with Plotly's D3 set for plots that
# need >10 distinct categories.
QUAL = eval.PALETTE_QUAL + px.colors.qualitative.D3


def _layout(**overrides) -> dict:
    out = dict(template=TEMPLATE)
    out.update(overrides)
    return out


def _scenario_order(values: Sequence[str]) -> List[str]:
    """Stable display order: known scenarios first (in NAME_MAP order),
    then unknown alphabetically."""
    try:
        from cp_cn_evaluation import SCENARIO_NAME_MAP
        known = [k for k in SCENARIO_NAME_MAP if k in set(values)]
        rest = sorted(set(values) - set(known))
        return known + rest
    except Exception:
        return sorted(set(values))


def _emit(figs: List[go.Figure], titles: List[str], slugs: List[str],
          out_html: Path, scenario_label: str = "CMRES eval") -> Path:
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    eval.write_all_in_one(
        figs, scenario_label, Path("."), str(out_html),
        titles=titles, slugs=slugs,
    )
    return out_html


# ─────────────────────────────────────────────────────────────────────────────
# E2 — Per-factor ablation of predicted_score
#   CSVs:  E2_ablation_<scenario>.csv
#   Cols : variant, rho, p, ci_lo, ci_hi, n, delta_vs_full
# ─────────────────────────────────────────────────────────────────────────────


_VARIANT_ORDER = ["full", "no_throughput", "no_stress", "no_topo", "no_adequacy"]
_VARIANT_PRETTY = {
    "full":          "Full",
    "no_throughput": "− throughput",
    "no_stress":     "− PTDF stress",
    "no_topo":       "− topology",
    "no_adequacy":   "− input adequacy",
}


def plot_e2_ablation(input_dir: Path, output_dir: Path) -> Optional[Path]:
    """One panel per scenario: ρ ± 95% CI per variant, with delta vs full
    annotated. A pooled summary heatmap shows the per-factor effect across
    scenarios."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = sorted(input_dir.glob("E2_ablation_*.csv"))
    if not files:
        return None

    frames: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        scenario = f.stem.replace("E2_ablation_", "")
        df["scenario"] = scenario
        frames.append(df)
    if not frames:
        return None
    pooled = pd.concat(frames, ignore_index=True)
    pooled["variant_order"] = pooled["variant"].map(
        {v: i for i, v in enumerate(_VARIANT_ORDER)}
    )
    pooled = pooled.sort_values(["scenario", "variant_order"])

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    # Per-scenario bar with CI error bars and Δ-vs-full annotations.
    for scenario in _scenario_order(pooled["scenario"].unique()):
        sub = pooled[pooled["scenario"] == scenario]
        if sub.empty:
            continue
        x_labels = [_VARIANT_PRETTY.get(v, v) for v in sub["variant"]]
        # Colour: full = grey, ablations = red intensity by |delta|.
        colors = []
        for v, d in zip(sub["variant"], sub["delta_vs_full"]):
            if v == "full":
                colors.append("#455a64")
            else:
                # Larger |Δ| means more important factor → deeper red.
                a = float(np.clip(abs(d), 0, 0.5) / 0.5) if np.isfinite(d) else 0.0
                colors.append(f"rgba(211,47,47,{0.30 + 0.65 * a:.3f})")

        err_plus = (sub["ci_hi"] - sub["rho"]).clip(lower=0)
        err_minus = (sub["rho"] - sub["ci_lo"]).clip(lower=0)
        text = [
            "" if v == "full" or not np.isfinite(d) else f"Δ={d:+.3f}"
            for v, d in zip(sub["variant"], sub["delta_vs_full"])
        ]
        fig = go.Figure([
            go.Bar(
                x=x_labels, y=sub["rho"],
                marker=dict(color=colors, line=dict(color="#222", width=0.6)),
                error_y=dict(type="data", symmetric=False,
                             array=err_plus, arrayminus=err_minus,
                             thickness=1.2, width=4, color="#222"),
                text=text, textposition="outside", cliponaxis=False,
                hovertemplate=("<b>%{x}</b><br>ρ = %{y:.3f}"
                               "<br>n = %{customdata[0]}<extra></extra>"),
                customdata=np.c_[sub["n"].values],
                showlegend=False,
            ),
        ])
        fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"))
        fig.update_layout(**_layout(
            title=f"E2 — Ablation impact ({pretty_scenario(scenario)})",
            yaxis=dict(title="Spearman ρ vs MC actual_total",
                       range=[-0.05, 1.05], gridcolor="#e5e5e5"),
            xaxis=dict(title=""),
            height=480, width=820,
        ))
        figs.append(fig)
        titles.append(f"E2 ablation — {pretty_scenario(scenario)}")
        slugs.append(f"e2_ablation_{scenario}")

    # Pooled heatmap of Δ vs full (signed, diverging).
    delta = pooled[pooled["variant"] != "full"].pivot_table(
        index="scenario", columns="variant", values="delta_vs_full",
    )
    delta = delta.reindex(columns=[v for v in _VARIANT_ORDER if v != "full"])
    delta = delta.reindex(index=_scenario_order(delta.index))
    if not delta.empty:
        z = delta.values
        zmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 0.5
        zmax = max(zmax, 0.05)
        text = np.where(np.isfinite(z), np.array([f"{v:+.2f}" for v in z.ravel()],
                                                 dtype=object).reshape(z.shape), "")
        heat = go.Figure(go.Heatmap(
            z=z,
            x=[_VARIANT_PRETTY[v] for v in delta.columns],
            y=[pretty_scenario(s) for s in delta.index],
            colorscale="RdBu", zmid=0, zmin=-zmax, zmax=zmax,
            colorbar=dict(title=dict(text="Δρ vs full", side="right"),
                          thickness=14, len=0.9),
            text=text, texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: Δρ = %{z:+.3f}<extra></extra>",
        ))
        heat.update_layout(**_layout(
            title="E2 — Per-factor ablation effect (Δρ vs full) across scenarios",
            xaxis=dict(title=""), yaxis=dict(title=""),
            height=80 + 36 * len(delta.index), width=860,
        ))
        figs.append(heat)
        titles.append("E2 ablation — pooled Δρ heatmap")
        slugs.append("e2_ablation_pooled_heatmap")

    return _emit(figs, titles, slugs, output_dir / "E2_ablation.html",
                 "E2 — Predicted-score ablation")


# ─────────────────────────────────────────────────────────────────────────────
# E4 — Distributed vs centralized comparison
#   CSVs:  E4_impact_concentration.csv, E4_rho_by_distribution.csv
# ─────────────────────────────────────────────────────────────────────────────


def plot_e4_distribution(input_dir: Path, output_dir: Path) -> Optional[Path]:
    """(a) Concentration metrics per scenario, coloured by distribution.
    (b) Per-metric ρ paired by density across distributions."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    conc_path = input_dir / "E4_impact_concentration.csv"
    rho_path = input_dir / "E4_rho_by_distribution.csv"
    if not conc_path.exists() and not rho_path.exists():
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    if conc_path.exists():
        conc = pd.read_csv(conc_path)
        if not conc.empty:
            metrics = [
                ("actual_gini",       "Gini"),
                ("actual_top1_share", "Top-1 share"),
                ("actual_top5_share", "Top-5 share"),
                ("actual_entropy",    "Entropy (nats)"),
            ]
            metrics = [(c, l) for c, l in metrics if c in conc.columns]
            if metrics:
                fig = make_subplots(
                    rows=1, cols=len(metrics),
                    subplot_titles=[l for _, l in metrics],
                    horizontal_spacing=0.07,
                )
                dists = sorted(conc["distribution"].dropna().unique())
                dist_color = {d: c for d, c in
                              zip(dists, QUAL)}
                for j, (col, label) in enumerate(metrics, start=1):
                    for d in dists:
                        sub = conc[conc["distribution"] == d].sort_values("density")
                        fig.add_trace(
                            go.Scatter(
                                x=sub["density"], y=sub[col],
                                mode="lines+markers",
                                name=d, legendgroup=d,
                                showlegend=(j == 1),
                                marker=dict(size=10, color=dist_color[d],
                                            line=dict(color="#222", width=0.6)),
                                line=dict(color=dist_color[d], width=2),
                                hovertext=[pretty_scenario(s) for s in sub["scenario"]],
                                hovertemplate=("<b>%{hovertext}</b><br>density "
                                               "%{x}<br>" + label + ": %{y:.3f}"
                                               "<extra>" + d + "</extra>"),
                            ),
                            row=1, col=j,
                        )
                    fig.update_xaxes(title_text="CP density", row=1, col=j)
                fig.update_layout(**_layout(
                    title="E4 — Impact concentration vs CP density, by distribution",
                    height=470, width=320 * len(metrics),
                    legend=dict(title="Distribution", orientation="h",
                                y=-0.18, x=0.5, xanchor="center"),
                ))
                figs.append(fig)
                titles.append("E4 concentration vs density")
                slugs.append("e4_concentration_vs_density")

    if rho_path.exists():
        rho = pd.read_csv(rho_path)
        if not rho.empty:
            # Grouped bars: x = metric, group = distribution, facet by density.
            metrics_order = list(rho["metric"].drop_duplicates())
            dists = sorted(rho["distribution"].dropna().unique())
            densities = sorted(rho["density"].dropna().unique())
            dist_color = {d: c for d, c in zip(dists, QUAL)}
            fig = make_subplots(
                rows=1, cols=max(1, len(densities)),
                subplot_titles=[f"density {d:g}" for d in densities],
                shared_yaxes=True, horizontal_spacing=0.06,
            )
            for j, dens in enumerate(densities, start=1):
                for d in dists:
                    sub = rho[(rho["density"] == dens) & (rho["distribution"] == d)]
                    if sub.empty:
                        continue
                    sub = sub.set_index("metric").reindex(metrics_order).reset_index()
                    fig.add_trace(go.Bar(
                        x=sub["metric"], y=sub["rho"],
                        name=d, legendgroup=d,
                        showlegend=(j == 1),
                        marker=dict(color=dist_color[d],
                                    line=dict(color="#222", width=0.5)),
                        hovertemplate=("<b>%{x}</b><br>ρ = %{y:.3f}"
                                       "<extra>" + d + "</extra>"),
                    ), row=1, col=j)
                fig.update_xaxes(tickangle=30, row=1, col=j)
            fig.update_yaxes(title_text="Spearman ρ", range=[-0.1, 1.05], row=1, col=1)
            fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"))
            fig.update_layout(**_layout(
                title="E4 — Per-metric ρ, paired by density × distribution",
                barmode="group",
                height=520, width=max(640, 320 * max(1, len(densities))),
                legend=dict(title="Distribution", orientation="h",
                            y=-0.30, x=0.5, xanchor="center"),
            ))
            figs.append(fig)
            titles.append("E4 ρ by distribution")
            slugs.append("e4_rho_by_distribution")

    if not figs:
        return None
    return _emit(figs, titles, slugs, output_dir / "E4_distribution.html",
                 "E4 — Distributed vs centralized")


# ─────────────────────────────────────────────────────────────────────────────
# E6 — Hyperparameter sensitivity (tornado)
#   CSVs:  E6_sensitivity_<scenario>.csv
# ─────────────────────────────────────────────────────────────────────────────


def plot_e6_sensitivity(input_dir: Path, output_dir: Path) -> Optional[Path]:
    """One tornado per scenario: per-parameter min↔max Δρ vs baseline,
    sorted by absolute swing. Plus a small-multiples view of ρ vs value
    for each parameter."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    files = sorted(input_dir.glob("E6_sensitivity_*.csv"))
    if not files:
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        scenario = f.stem.replace("E6_sensitivity_", "")
        baseline_row = df[df["param"] == "(baseline)"]
        rho_baseline = float(baseline_row["rho"].iloc[0]) if len(baseline_row) else float("nan")
        sweep = df[df["param"] != "(baseline)"].copy()
        if sweep.empty:
            continue
        # For each param: low = min(rho across values), high = max(rho).
        agg = sweep.groupby("param").agg(rho_min=("rho", "min"),
                                         rho_max=("rho", "max"),
                                         val_min=("rho", "idxmin"),
                                         val_max=("rho", "idxmax"))
        agg["delta_low"] = agg["rho_min"] - rho_baseline
        agg["delta_high"] = agg["rho_max"] - rho_baseline
        agg["span"] = agg["rho_max"] - agg["rho_min"]
        agg = agg.sort_values("span", ascending=True)

        # Tornado plot — one bar per param showing the [min, max] range,
        # baseline as a vertical reference line.
        tor = go.Figure()
        tor.add_trace(go.Bar(
            y=agg.index, x=agg["span"], base=agg["rho_min"],
            orientation="h",
            marker=dict(color="rgba(63,81,181,0.85)",
                        line=dict(color="#1a237e", width=0.8)),
            hovertemplate=("<b>%{y}</b><br>min ρ = %{base:.3f}"
                           "<br>max ρ = %{customdata[0]:.3f}"
                           "<br>span = %{x:.3f}<extra></extra>"),
            customdata=np.c_[agg["rho_max"].values],
            showlegend=False,
        ))
        tor.add_vline(x=rho_baseline, line=dict(color="#d32f2f", width=2, dash="dash"),
                      annotation_text=f"baseline ρ = {rho_baseline:.3f}",
                      annotation_position="top right")
        tor.update_layout(**_layout(
            title=f"E6 — Hyperparameter sensitivity tornado ({pretty_scenario(scenario)})",
            xaxis=dict(title="Spearman ρ vs MC actual_total",
                       range=[min(0.0, float(agg["rho_min"].min()) - 0.05),
                              max(1.0, float(agg["rho_max"].max()) + 0.05)],
                       gridcolor="#e5e5e5"),
            yaxis=dict(title=""),
            height=80 + 38 * len(agg), width=900,
        ))
        figs.append(tor)
        titles.append(f"E6 tornado — {pretty_scenario(scenario)}")
        slugs.append(f"e6_tornado_{scenario}")

        # Small multiples: one panel per parameter showing ρ across swept values.
        params = list(agg.index)
        ncols = min(3, len(params))
        nrows = int(np.ceil(len(params) / ncols))
        sm = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=params,
            horizontal_spacing=0.10, vertical_spacing=0.18,
        )
        for i, p in enumerate(params):
            r = i // ncols + 1
            c = i % ncols + 1
            sub = sweep[sweep["param"] == p].copy()
            # Try to keep numeric ordering on x.
            try:
                sub["value_num"] = pd.to_numeric(sub["value"])
                sub = sub.sort_values("value_num")
                xv = sub["value_num"]
            except Exception:
                sub = sub.sort_values("value")
                xv = sub["value"]
            sm.add_trace(go.Scatter(
                x=xv, y=sub["rho"], mode="lines+markers",
                line=dict(color="#3f51b5", width=2),
                marker=dict(size=8, color="#3f51b5",
                            line=dict(color="#1a237e", width=0.6)),
                showlegend=False,
                hovertemplate="<b>" + p + "</b><br>value = %{x}<br>ρ = %{y:.3f}<extra></extra>",
            ), row=r, col=c)
            sm.add_hline(y=rho_baseline, line=dict(color="#d32f2f", width=1, dash="dash"),
                         row=r, col=c)
        sm.update_yaxes(range=[0.0, 1.05])
        sm.update_layout(**_layout(
            title=f"E6 — ρ across swept values ({pretty_scenario(scenario)})",
            height=260 * nrows + 60, width=320 * ncols,
        ))
        figs.append(sm)
        titles.append(f"E6 sweeps — {pretty_scenario(scenario)}")
        slugs.append(f"e6_sweeps_{scenario}")

    if not figs:
        return None
    return _emit(figs, titles, slugs, output_dir / "E6_sensitivity.html",
                 "E6 — Hyperparameter sensitivity")


# ─────────────────────────────────────────────────────────────────────────────
# E7 — MC validity diagnostics
#   CSVs:  E7_rhw_curves.csv, E7_mc_summary.csv
# ─────────────────────────────────────────────────────────────────────────────


def plot_e7_mc_validity(input_dir: Path, output_dir: Path) -> Optional[Path]:
    """RHW(n) per carrier, faceted by scenario; AV variance reduction
    summary as a horizontal bar chart with the threshold of 1× shown."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    rhw_path = input_dir / "E7_rhw_curves.csv"
    sum_path = input_dir / "E7_mc_summary.csv"
    if not rhw_path.exists() and not sum_path.exists():
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    if rhw_path.exists():
        rhw = pd.read_csv(rhw_path)
        if not rhw.empty:
            scenarios = _scenario_order(rhw["scenario"].unique())
            ncols = min(3, len(scenarios))
            nrows = int(np.ceil(len(scenarios) / ncols))
            fig = make_subplots(
                rows=nrows, cols=ncols,
                subplot_titles=[pretty_scenario(s) for s in scenarios],
                horizontal_spacing=0.07, vertical_spacing=0.16,
                shared_xaxes=False,
            )
            for i, s in enumerate(scenarios):
                r = i // ncols + 1
                c = i % ncols + 1
                sub = rhw[rhw["scenario"] == s]
                for k, carrier in enumerate(["power", "heat", "gas"]):
                    cur = sub[sub["carrier"] == carrier].sort_values("n")
                    if cur.empty:
                        continue
                    fig.add_trace(go.Scatter(
                        x=cur["n"], y=cur["rhw"],
                        mode="lines+markers",
                        name=carrier, legendgroup=carrier,
                        showlegend=(i == 0),
                        line=dict(color=CARRIER_COLORS[carrier], width=2),
                        marker=dict(size=6, color=CARRIER_COLORS[carrier],
                                    line=dict(color="#222", width=0.4)),
                        hovertemplate=("<b>" + carrier + "</b><br>n = %{x}"
                                       "<br>RHW = %{y:.4f}<extra></extra>"),
                    ), row=r, col=c)
                fig.add_hline(y=0.05, line=dict(color="#444", width=1, dash="dash"),
                              row=r, col=c)
                fig.update_xaxes(title_text="MC samples n", type="log", row=r, col=c)
                fig.update_yaxes(title_text="Relative half-width" if c == 1 else "",
                                 type="log", row=r, col=c)
            fig.update_layout(**_layout(
                title="E7 — RHW(n) convergence per carrier (target = 0.05, dashed)",
                height=320 * nrows + 60, width=370 * ncols,
                legend=dict(title="Carrier", orientation="h",
                            y=-0.10, x=0.5, xanchor="center"),
            ))
            figs.append(fig)
            titles.append("E7 RHW convergence")
            slugs.append("e7_rhw_convergence")

    if sum_path.exists():
        s = pd.read_csv(sum_path)
        if not s.empty and "AV_reduction_factor" in s.columns:
            s = s.sort_values("AV_reduction_factor", na_position="first")
            colors = ["#388e3c" if (np.isfinite(v) and v > 1.0) else "#d32f2f"
                      for v in s["AV_reduction_factor"]]
            fig = go.Figure(go.Bar(
                y=[pretty_scenario(x) for x in s["scenario"]],
                x=s["AV_reduction_factor"],
                orientation="h",
                marker=dict(color=colors, line=dict(color="#222", width=0.6)),
                text=[f"{v:.2f}×" if np.isfinite(v) else "n/a"
                      for v in s["AV_reduction_factor"]],
                textposition="outside", cliponaxis=False,
                hovertemplate=("<b>%{y}</b><br>AV reduction = %{x:.3f}×"
                               "<br>n_runs = %{customdata[0]}<extra></extra>"),
                customdata=np.c_[s["n_runs"].values] if "n_runs" in s else None,
                showlegend=False,
            ))
            fig.add_vline(x=1.0, line=dict(color="#444", width=1.2, dash="dash"),
                          annotation_text="no reduction", annotation_position="top")
            fig.update_layout(**_layout(
                title="E7 — Antithetic-variates variance-reduction factor",
                xaxis=dict(title="Var(naive) / [2 · Var(pair-mean)]",
                           gridcolor="#e5e5e5"),
                yaxis=dict(title=""),
                height=80 + 36 * len(s), width=820,
            ))
            figs.append(fig)
            titles.append("E7 AV reduction factor")
            slugs.append("e7_av_reduction_factor")

    if not figs:
        return None
    return _emit(figs, titles, slugs, output_dir / "E7_mc_validity.html",
                 "E7 — MC validity diagnostics")


# ─────────────────────────────────────────────────────────────────────────────
# Generic per-metric ρ heatmap+bars used by E8, E15
# ─────────────────────────────────────────────────────────────────────────────


def _per_metric_rho_panels(df: pd.DataFrame, exp_label: str,
                           rho_col: str = "rho") -> List[go.Figure]:
    figs: List[go.Figure] = []

    # 1) heatmap scenario × metric of ρ.
    pivot = df.pivot_table(index="scenario", columns="metric", values=rho_col,
                           aggfunc="mean")
    pivot = pivot.reindex(index=_scenario_order(pivot.index))
    z = pivot.values
    if z.size:
        text = np.where(np.isfinite(z),
                        np.array([f"{v:.2f}" for v in z.ravel()],
                                 dtype=object).reshape(z.shape),
                        "")
        heat = go.Figure(go.Heatmap(
            z=z, x=list(pivot.columns),
            y=[pretty_scenario(s) for s in pivot.index],
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title=dict(text="Spearman ρ", side="right"),
                          thickness=14, len=0.9),
            text=text, texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="<b>%{y}</b><br>%{x}: ρ = %{z:.3f}<extra></extra>",
        ))
        heat.update_layout(**_layout(
            title=f"{exp_label} — Spearman ρ per metric × scenario",
            xaxis=dict(title="Metric", tickangle=30),
            yaxis=dict(title=""),
            height=80 + 36 * max(1, len(pivot.index)),
            width=120 + 90 * max(1, len(pivot.columns)),
        ))
        figs.append(heat)

    # 2) per-metric distribution across scenarios as box+strip.
    if not df.empty:
        # Sort metrics by median ρ so the strongest is left.
        order = (df.groupby("metric")[rho_col].median()
                 .sort_values(ascending=False).index.tolist())
        fig = go.Figure()
        for i, m in enumerate(order):
            sub = df[df["metric"] == m]
            color = QUAL[i % len(QUAL)]
            fig.add_trace(go.Box(
                x=[m] * len(sub), y=sub[rho_col],
                name=m, marker=dict(color=color, size=6,
                                    line=dict(color="#222", width=0.4)),
                line=dict(color=color), fillcolor=color, opacity=0.55,
                boxpoints="all", jitter=0.5, pointpos=0,
                hovertext=[pretty_scenario(s) for s in sub["scenario"]],
                hovertemplate=("<b>%{hovertext}</b><br>" + m
                               + ": ρ = %{y:.3f}<extra></extra>"),
                showlegend=False,
            ))
        fig.add_hline(y=0, line=dict(color="#888", width=1, dash="dot"))
        fig.update_layout(**_layout(
            title=f"{exp_label} — Distribution of ρ across scenarios per metric",
            yaxis=dict(title="Spearman ρ", range=[-1.05, 1.05],
                       gridcolor="#e5e5e5"),
            xaxis=dict(title="", tickangle=30),
            height=520, width=140 + 80 * len(order),
        ))
        figs.append(fig)
    return figs


# ─────────────────────────────────────────────────────────────────────────────
# E8 — Multilayer centralities
# ─────────────────────────────────────────────────────────────────────────────


def plot_e8_multilayer(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E8_multilayer_rho.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    figs = _per_metric_rho_panels(df, "E8")
    titles = ["E8 multilayer ρ heatmap", "E8 multilayer ρ distributions"][:len(figs)]
    slugs = ["e8_multilayer_heatmap", "e8_multilayer_distributions"][:len(figs)]
    return _emit(figs, titles, slugs, Path(output_dir) / "E8_multilayer.html",
                 "E8 — Multilayer centralities")


# ─────────────────────────────────────────────────────────────────────────────
# E9 — Percolation / robustness AUC
# ─────────────────────────────────────────────────────────────────────────────


def plot_e9_percolation(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E9_percolation_auc.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    # Panel 1: AUC_metric vs AUC_random_mean per scenario × metric, with
    # ±std error bars on random and a y=x reference (lower AUC = better).
    metrics = list(df["metric"].drop_duplicates())
    cmap = {m: QUAL[i % len(QUAL)] for i, m in enumerate(metrics)}
    fig = go.Figure()
    lo = float(np.nanmin([df["AUC_metric"].min(), df["AUC_random_mean"].min()]))
    hi = float(np.nanmax([df["AUC_metric"].max(), df["AUC_random_mean"].max()]))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color="#888", dash="dash", width=1.2),
        name="random = metric", showlegend=True,
    ))
    for m in metrics:
        sub = df[df["metric"] == m]
        fig.add_trace(go.Scatter(
            x=sub["AUC_random_mean"], y=sub["AUC_metric"],
            error_x=dict(type="data", array=sub["AUC_random_std"],
                         thickness=1, width=3, color=cmap[m]),
            mode="markers", name=m,
            marker=dict(color=cmap[m], size=11, symbol="circle",
                        line=dict(color="#222", width=0.6)),
            hovertext=[pretty_scenario(s) for s in sub["scenario"]],
            hovertemplate=("<b>%{hovertext}</b><br>" + m
                           + "<br>AUC random = %{x:.3f} ± %{error_x.array:.3f}"
                           "<br>AUC metric = %{y:.3f}"
                           "<br>z = %{customdata[0]:+.2f}<extra></extra>"),
            customdata=np.c_[sub["AUC_z"].values],
        ))
    fig.update_layout(**_layout(
        title="E9 — Targeted-attack AUC vs random baseline (lower = better)",
        xaxis=dict(title="AUC under random removal (mean ± std)",
                   gridcolor="#e5e5e5"),
        yaxis=dict(title="AUC under metric-targeted removal",
                   gridcolor="#e5e5e5"),
        height=600, width=820,
        legend=dict(title="Metric"),
    ))
    figs.append(fig)
    titles.append("E9 percolation AUC vs random")
    slugs.append("e9_auc_vs_random")

    # Panel 2: z-score heatmap (more negative = stronger attack ordering).
    pivot = df.pivot_table(index="scenario", columns="metric", values="AUC_z")
    pivot = pivot.reindex(index=_scenario_order(pivot.index))
    z = pivot.values
    if z.size:
        zmax = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 3.0
        zmax = max(zmax, 1.0)
        text = np.where(np.isfinite(z),
                        np.array([f"{v:+.1f}" for v in z.ravel()],
                                 dtype=object).reshape(z.shape), "")
        heat = go.Figure(go.Heatmap(
            z=z, x=list(pivot.columns),
            y=[pretty_scenario(s) for s in pivot.index],
            colorscale="RdBu", zmid=0, zmin=-zmax, zmax=zmax,
            colorbar=dict(title=dict(text="z-score", side="right"),
                          thickness=14, len=0.9),
            text=text, texttemplate="%{text}",
            hovertemplate="<b>%{y}</b><br>%{x}: z = %{z:+.2f}<extra></extra>",
        ))
        heat.update_layout(**_layout(
            title="E9 — AUC z-score (negative = better than random)",
            xaxis=dict(title="Metric", tickangle=30), yaxis=dict(title=""),
            height=80 + 36 * len(pivot.index),
            width=120 + 90 * len(pivot.columns),
        ))
        figs.append(heat)
        titles.append("E9 AUC z-scores")
        slugs.append("e9_auc_zscore_heatmap")

    return _emit(figs, titles, slugs, Path(output_dir) / "E9_percolation.html",
                 "E9 — Percolation robustness")


# ─────────────────────────────────────────────────────────────────────────────
# E10 — Coupling-strength characterisation
# ─────────────────────────────────────────────────────────────────────────────


def plot_e10_structural(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E10_coupling_strength.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    sigma_cols = [c for c in df.columns if c.startswith("sigma_c[")]
    # Stacked bar: per-scenario σ_c per layer-pair.
    if sigma_cols:
        pretty_pair = {c: c[len("sigma_c["):-1] for c in sigma_cols}
        df_sorted = df.sort_values("sigma_c_total", ascending=True)
        fig = go.Figure()
        palette = px.colors.qualitative.Bold
        for i, c in enumerate(sigma_cols):
            fig.add_trace(go.Bar(
                y=[pretty_scenario(s) for s in df_sorted["scenario"]],
                x=df_sorted[c], orientation="h",
                name=pretty_pair[c],
                marker=dict(color=palette[i % len(palette)],
                            line=dict(color="#222", width=0.3)),
                hovertemplate=("<b>%{y}</b><br>" + pretty_pair[c]
                               + ": σ = %{x:.3f}<extra></extra>"),
            ))
        fig.update_layout(**_layout(
            title="E10 — Coupling strength σ_c per layer pair (stacked)",
            barmode="stack",
            xaxis=dict(title="σ_c", gridcolor="#e5e5e5"),
            yaxis=dict(title=""),
            height=80 + 36 * len(df_sorted), width=900,
            legend=dict(title="Layer pair"),
        ))
        figs.append(fig)
        titles.append("E10 σ_c per layer pair")
        slugs.append("e10_sigma_c_stacked")

    # Mediator scatter: σ_c_total vs MC ENS proxy, coloured by distribution,
    # sized by Gini.
    if "mc_ens_proxy" in df.columns and "sigma_c_total" in df.columns:
        sub = df.dropna(subset=["sigma_c_total", "mc_ens_proxy"])
        if not sub.empty:
            dists = sorted(sub["distribution"].dropna().unique())
            dist_color = {d: QUAL[i % len(QUAL)] for i, d in enumerate(dists)}
            fig = go.Figure()
            for d in dists:
                cur = sub[sub["distribution"] == d]
                gini = cur.get("cp_localization_gini",
                               pd.Series([0.0] * len(cur), index=cur.index))
                size = 12 + 30 * gini.fillna(0.0).clip(0, 1)
                fig.add_trace(go.Scatter(
                    x=cur["sigma_c_total"], y=cur["mc_ens_proxy"],
                    mode="markers+text", name=d,
                    text=[pretty_scenario(s) for s in cur["scenario"]],
                    textposition="top center", textfont=dict(size=10),
                    marker=dict(color=dist_color[d], size=size,
                                opacity=0.85, line=dict(color="#222", width=0.6)),
                    hovertemplate=("<b>%{text}</b><br>σ_c = %{x:.3f}"
                                   "<br>ENS proxy = %{y:.3f}"
                                   "<br>Gini = %{customdata[0]:.3f}"
                                   "<extra>" + d + "</extra>"),
                    customdata=np.c_[gini.values],
                ))
            fig.update_layout(**_layout(
                title="E10 — Total coupling strength vs MC ENS proxy (size = CP-Gini)",
                xaxis=dict(title="σ_c total", gridcolor="#e5e5e5"),
                yaxis=dict(title="Σ |actual_total| over CP rows (MC ENS proxy)",
                           gridcolor="#e5e5e5"),
                height=580, width=820,
                legend=dict(title="Distribution"),
            ))
            figs.append(fig)
            titles.append("E10 σ_c vs MC ENS")
            slugs.append("e10_sigma_vs_ens")

    if not figs:
        return None
    return _emit(figs, titles, slugs, Path(output_dir) / "E10_structural.html",
                 "E10 — Coupling strength")


# ─────────────────────────────────────────────────────────────────────────────
# E11 — Null-model z-scores
# ─────────────────────────────────────────────────────────────────────────────


def plot_e11_null_models(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E11_null_z.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    # One panel per statistic: grouped bar, x = scenario, group = null kind,
    # y = z-score, with shaded |z| > 1.96 region.
    stats = list(df["statistic"].drop_duplicates())
    nulls = sorted(df["null_kind"].dropna().unique())
    null_color = {k: QUAL[i % len(QUAL)] for i, k in enumerate(nulls)}

    fig = make_subplots(
        rows=1, cols=len(stats),
        subplot_titles=stats, shared_yaxes=True, horizontal_spacing=0.05,
    )
    for j, stat in enumerate(stats, start=1):
        sub_stat = df[df["statistic"] == stat]
        scens = _scenario_order(sub_stat["scenario"].unique())
        for k in nulls:
            sub = sub_stat[sub_stat["null_kind"] == k].set_index("scenario")
            sub = sub.reindex(scens)
            fig.add_trace(go.Bar(
                x=[pretty_scenario(s) for s in scens],
                y=sub["z"],
                name=k, legendgroup=k, showlegend=(j == 1),
                marker=dict(color=null_color[k],
                            line=dict(color="#222", width=0.5)),
                hovertemplate=("<b>%{x}</b><br>z = %{y:+.2f}"
                               "<br>obs = %{customdata[0]:.3f}"
                               "<br>null μ = %{customdata[1]:.3f}"
                               " ± %{customdata[2]:.3f}"
                               "<extra>" + stat + " · " + k + "</extra>"),
                customdata=np.c_[sub["observed"].values,
                                 sub["null_mean"].values,
                                 sub["null_std"].values],
            ), row=1, col=j)
        fig.add_hrect(y0=-1.96, y1=1.96, fillcolor="rgba(180,180,180,0.18)",
                      line_width=0, row=1, col=j)
        fig.add_hline(y=0, line=dict(color="#444", width=1), row=1, col=j)
        fig.update_xaxes(tickangle=30, row=1, col=j)
    fig.update_yaxes(title_text="z-score (vs null)", row=1, col=1)
    fig.update_layout(**_layout(
        title="E11 — Observed structural quantities vs null ensembles (|z|>1.96 shaded)",
        barmode="group",
        height=560, width=max(640, 380 * len(stats)),
        legend=dict(title="Null model", orientation="h",
                    y=-0.25, x=0.5, xanchor="center"),
    ))
    figs.append(fig)
    titles.append("E11 null z-scores")
    slugs.append("e11_null_z_scores")

    return _emit(figs, titles, slugs, Path(output_dir) / "E11_null_models.html",
                 "E11 — Null-model z-scores")


# ─────────────────────────────────────────────────────────────────────────────
# E12 — Community structure & cross-layer bridges
# ─────────────────────────────────────────────────────────────────────────────


def plot_e12_community(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E12_bridges.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None

    df = df.assign(scenario_pretty=df["scenario"].map(pretty_scenario))
    df_sorted = df.sort_values("rho_bridge_vs_actual", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["scenario_pretty"], x=df_sorted["rho_bridge_vs_actual"],
        orientation="h", name="ρ(bridge, actual)",
        marker=dict(color="#3949ab", line=dict(color="#1a237e", width=0.5)),
        hovertemplate=("<b>%{y}</b><br>ρ = %{x:+.3f}"
                       "<br>p = %{customdata[0]:.3g}"
                       "<br>n = %{customdata[1]}<extra>raw</extra>"),
        customdata=np.c_[df_sorted["p_bridge_vs_actual"].values,
                         df_sorted["n"].values],
    ))
    fig.add_trace(go.Bar(
        y=df_sorted["scenario_pretty"], x=df_sorted["rho_bridge_vs_actual_given_bc"],
        orientation="h", name="ρ(bridge, actual | BC)",
        marker=dict(color="#d32f2f", line=dict(color="#7f0000", width=0.5)),
        hovertemplate=("<b>%{y}</b><br>ρ = %{x:+.3f}"
                       "<br>p = %{customdata[0]:.3g}<extra>partial</extra>"),
        customdata=np.c_[df_sorted["p_bridge_vs_actual_given_bc"].values],
    ))
    fig.add_vline(x=0, line=dict(color="#444", width=1, dash="dot"))
    fig.update_layout(**_layout(
        title="E12 — Bridge-score correlation with MC actual_total (raw vs BC-residualised)",
        barmode="group",
        xaxis=dict(title="Spearman ρ", range=[-1.05, 1.05], gridcolor="#e5e5e5"),
        yaxis=dict(title=""),
        height=80 + 50 * len(df_sorted), width=900,
        legend=dict(title="Correlation"),
    ))
    figs = [fig]
    titles = ["E12 bridge-score correlations"]
    slugs = ["e12_bridge_correlations"]

    # Communities count as a side bar (small panel).
    if "n_communities" in df.columns:
        fig2 = go.Figure(go.Bar(
            y=df_sorted["scenario_pretty"],
            x=df_sorted["n_communities"], orientation="h",
            marker=dict(color="#00897b", line=dict(color="#004d40", width=0.5)),
            text=df_sorted["n_communities"], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>communities: %{x}<extra></extra>",
            showlegend=False,
        ))
        fig2.update_layout(**_layout(
            title="E12 — Number of detected communities per scenario",
            xaxis=dict(title="# communities", gridcolor="#e5e5e5"),
            yaxis=dict(title=""),
            height=80 + 36 * len(df_sorted), width=820,
        ))
        figs.append(fig2)
        titles.append("E12 communities count")
        slugs.append("e12_communities_count")

    return _emit(figs, titles, slugs, Path(output_dir) / "E12_community.html",
                 "E12 — Community structure")


# ─────────────────────────────────────────────────────────────────────────────
# E13 — Spectral robustness
# ─────────────────────────────────────────────────────────────────────────────


def plot_e13_spectral(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E13_spectral.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None

    lam_cols = [c for c in df.columns if c.startswith("lambda2_")]
    if not lam_cols:
        return None

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    # Grouped bar: λ₂ per layer per scenario.
    scens = _scenario_order(df["scenario"].unique())
    df = df.set_index("scenario").reindex(scens).reset_index()
    fig = go.Figure()
    for c in lam_cols:
        layer = c.replace("lambda2_", "")
        color = CARRIER_COLORS.get(layer, "#5e35b1" if layer == "supra"
                                                  else QUAL[lam_cols.index(c) % len(QUAL)])
        fig.add_trace(go.Bar(
            x=[pretty_scenario(s) for s in df["scenario"]],
            y=df[c], name=layer,
            marker=dict(color=color, line=dict(color="#222", width=0.4)),
            hovertemplate=("<b>%{x}</b><br>" + layer
                           + ": λ₂ = %{y:.4f}<extra></extra>"),
        ))
    fig.update_layout(**_layout(
        title="E13 — Algebraic connectivity λ₂ per layer × scenario",
        barmode="group",
        xaxis=dict(title="", tickangle=30),
        yaxis=dict(title="λ₂", gridcolor="#e5e5e5", type="log"),
        height=540, width=max(640, 90 * len(df) + 320),
        legend=dict(title="Layer"),
    ))
    figs.append(fig)
    titles.append("E13 λ₂ per layer")
    slugs.append("e13_lambda2_per_layer")

    # Kirchhoff index (lower = more robust) on its own bar.
    if "kirchhoff_lcc" in df.columns:
        df_sorted = df.sort_values("kirchhoff_lcc", ascending=True)
        fig2 = go.Figure(go.Bar(
            y=[pretty_scenario(s) for s in df_sorted["scenario"]],
            x=df_sorted["kirchhoff_lcc"], orientation="h",
            marker=dict(color="#5e35b1", line=dict(color="#311b92", width=0.5)),
            hovertemplate="<b>%{y}</b><br>Kirchhoff (LCC): %{x:.3f}<extra></extra>",
            text=[f"{v:.2g}" if np.isfinite(v) else "n/a"
                  for v in df_sorted["kirchhoff_lcc"]],
            textposition="outside", cliponaxis=False,
            showlegend=False,
        ))
        fig2.update_layout(**_layout(
            title="E13 — Kirchhoff index of largest connected component (lower = more robust)",
            xaxis=dict(title="Kirchhoff index", gridcolor="#e5e5e5", type="log"),
            yaxis=dict(title=""),
            height=80 + 36 * len(df_sorted), width=820,
        ))
        figs.append(fig2)
        titles.append("E13 Kirchhoff index")
        slugs.append("e13_kirchhoff")

    return _emit(figs, titles, slugs, Path(output_dir) / "E13_spectral.html",
                 "E13 — Spectral robustness")


# ─────────────────────────────────────────────────────────────────────────────
# E15 — Structural metrics ρ
# ─────────────────────────────────────────────────────────────────────────────


def plot_e15_structural(input_dir: Path, output_dir: Path) -> Optional[Path]:
    p = Path(input_dir) / "E15_structural_rho.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    figs = _per_metric_rho_panels(df, "E15")
    titles = ["E15 structural ρ heatmap", "E15 structural ρ distributions"][:len(figs)]
    slugs = ["e15_structural_heatmap", "e15_structural_distributions"][:len(figs)]
    return _emit(figs, titles, slugs, Path(output_dir) / "E15_structural.html",
                 "E15 — Structural metrics")


# ─────────────────────────────────────────────────────────────────────────────
# E16 — Single-removal-shed validation
# ─────────────────────────────────────────────────────────────────────────────


_E16_SECTORS = (
    ("total_shed", "Total"),
    ("power_shed", "Electricity"),
    ("heat_shed",  "Heat"),
    ("gas_shed",   "Gas"),
)


def _e16_scatter(merged: pd.DataFrame, metric: str, scenario: str) -> go.Figure:
    """One figure per metric: a 2×2 panel of (metric vs sector_shed) for
    {total, electricity, heat, gas}, with markers coloured by ``cp_type``
    (legend shows which branch / compound family each point belongs to).

    Each panel filters to ``sector_shed > 0`` so the log-y axis is
    well-defined, then annotates with Spearman ρ. Skips entirely if the
    x-axis is constant (e.g. ``input_adequacy`` on a CP-free grid would
    crash Kaleido with "axis scaling")."""
    from plotly.subplots import make_subplots
    from scipy.stats import spearmanr

    if metric not in merged.columns or merged[metric].nunique() < 2:
        return go.Figure()

    cp_types = sorted(merged.get("cp_type", pd.Series(dtype=str)).dropna().unique())
    color_map = {t: QUAL[i % len(QUAL)] for i, t in enumerate(cp_types)}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[label for _, label in _E16_SECTORS],
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.10, vertical_spacing=0.16,
    )

    legend_seen: set = set()
    any_data = False
    for idx, (sector_col, sector_label) in enumerate(_E16_SECTORS):
        row, col = idx // 2 + 1, idx % 2 + 1
        # Plotly's first axis is "x"/"y" (no "1"); only later subplots get
        # numeric suffixes. Use that convention for `xref` / `yref` strings.
        ax_suffix = "" if idx == 0 else str(idx + 1)
        if sector_col not in merged.columns:
            continue
        df = merged[merged[metric].notna() & (merged[sector_col] > 0)].copy()
        if df.empty:
            fig.add_annotation(
                xref=f"x{ax_suffix} domain", yref=f"y{ax_suffix} domain",
                x=0.5, y=0.5, showarrow=False,
                text=f"no rows with {sector_col} > 0",
                font=dict(color="#888"),
                row=row, col=col,
            )
            continue
        any_data = True
        # One trace per cp_type so the legend lists branch families.
        for cp_type in cp_types:
            sub = df[df["cp_type"] == cp_type] if "cp_type" in df.columns else df
            if sub.empty:
                continue
            show_in_legend = cp_type not in legend_seen
            legend_seen.add(cp_type)
            fig.add_trace(
                go.Scatter(
                    x=sub[metric], y=sub[sector_col],
                    mode="markers", name=cp_type,
                    legendgroup=cp_type, showlegend=show_in_legend,
                    marker=dict(
                        color=color_map.get(cp_type, "#888"),
                        size=7, opacity=0.85,
                        line=dict(color="#222", width=0.4),
                    ),
                    hovertemplate=(
                        f"<b>{cp_type}</b><br>"
                        + (("cp_id=%{customdata[0]}<br>") if "cp_id" in sub.columns else "")
                        + f"{metric}=%{{x:.4g}}<br>{sector_col}=%{{y:.4g}}<extra></extra>"
                    ),
                    customdata=sub[["cp_id"]].values if "cp_id" in sub.columns else None,
                ),
                row=row, col=col,
            )
        # ρ annotation in the panel.
        if len(df) >= 3 and df[metric].nunique() >= 2 and df[sector_col].nunique() >= 2:
            rho, p = spearmanr(df[metric], df[sector_col])
            fig.add_annotation(
                xref=f"x{ax_suffix} domain", yref=f"y{ax_suffix} domain",
                x=0.02, y=0.98, xanchor="left", yanchor="top",
                showarrow=False, align="left",
                text=f"ρ = {rho:+.3f}<br>p = {p:.2e}<br>n = {len(df)}",
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor="#888", borderwidth=1,
                font=dict(size=10),
                row=row, col=col,
            )
        fig.update_yaxes(type="log", row=row, col=col, title_text=f"{sector_col} (MW, log)")
        fig.update_xaxes(row=row, col=col, title_text=metric if row == 2 else "")

    if not any_data:
        return go.Figure()

    fig.update_layout(**_layout(
        title=f"E16 — {pretty_scenario(scenario)}: {metric} vs analytical shed (per sector)",
        height=720, width=1080,
        legend=dict(title="cp_type"),
    ))
    return fig


def _e16_top_overlap(merged: pd.DataFrame, scenario: str,
                     metrics: List[str], ks: Sequence[int] = (5, 10, 20, 50)) -> go.Figure:
    """Top-K overlap (metric-rank ∩ shed-rank) / K versus K, one line per metric.
    Captures how well each metric reproduces the analytical top-K critical set."""
    rows: List[dict] = []
    shed_rank = merged.sort_values("total_shed", ascending=False)
    join_col = "_join_cp_id" if "_join_cp_id" in merged.columns else "cp_id"
    for m in metrics:
        if m not in merged.columns or merged[m].notna().sum() == 0:
            continue
        metric_rank = merged.sort_values(m, ascending=False)
        for k in ks:
            if k > len(merged):
                continue
            top_metric = set(metric_rank.head(k)[join_col])
            top_shed = set(shed_rank.head(k)[join_col])
            rows.append({"metric": m, "k": k,
                         "overlap": len(top_metric & top_shed) / k})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = px.line(
        df, x="k", y="overlap", color="metric", markers=True,
        color_discrete_sequence=QUAL,
    )
    fig.update_layout(**_layout(
        title=f"E16 — {pretty_scenario(scenario)}: top-K overlap (metric vs analytical shed)",
        xaxis_title="K", yaxis_title="|top_K(metric) ∩ top_K(shed)| / K",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=420, width=860, legend=dict(title="Metric"),
    ))
    return fig


def plot_e16_single_removal(input_dir: Path, output_dir: Path) -> Optional[Path]:
    """Per-metric ρ vs analytical shed AND vs MC, with the ceiling
    (ρ_shed_vs_mc) drawn as a hatched reference for each scenario.

    When per-scenario ``E16_<scenario>_merged.csv`` files are present
    (emitted by the refactored ``experiment_e16_single_removal_validation``),
    appends per-metric scatter panels and a top-K overlap curve per scenario.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    metric_path = input_dir / "E16_metric_vs_shed.csv"
    ceil_path = input_dir / "E16_shed_vs_mc_ceiling.csv"
    if not metric_path.exists():
        return None
    df = pd.read_csv(metric_path)
    if df.empty:
        return None
    ceil = pd.read_csv(ceil_path) if ceil_path.exists() else pd.DataFrame()

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    # ρ vs analytical shed — heatmap when multiple scenarios are present,
    # horizontal bar when there is only one (a 1×N heatmap crashes Kaleido
    # with "axis scaling" because the y-axis is degenerate). Also drop
    # columns / rows that are entirely NaN — Kaleido fails the same way when
    # an axis has no finite range to scale.
    pivot_shed = df.pivot_table(index="scenario", columns="metric", values="rho_vs_shed")
    pivot_shed = pivot_shed.reindex(index=_scenario_order(pivot_shed.index))
    pivot_shed = pivot_shed.dropna(axis=1, how="all").dropna(axis=0, how="all")
    z = pivot_shed.values
    if z.size:
        if pivot_shed.shape[0] >= 2:
            text = np.where(np.isfinite(z),
                            np.array([f"{v:.2f}" for v in z.ravel()],
                                     dtype=object).reshape(z.shape), "")
            heat = go.Figure(go.Heatmap(
                z=z, x=list(pivot_shed.columns),
                y=[pretty_scenario(s) for s in pivot_shed.index],
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title=dict(text="ρ vs shed", side="right"),
                              thickness=14, len=0.9),
                text=text, texttemplate="%{text}",
                hovertemplate="<b>%{y}</b><br>%{x}: ρ = %{z:.3f}<extra></extra>",
            ))
            heat.update_layout(**_layout(
                title="E16 — Spearman ρ between metric and analytical single-removal shed",
                xaxis=dict(title="Metric", tickangle=30), yaxis=dict(title=""),
                height=80 + 36 * len(pivot_shed.index),
                width=120 + 90 * len(pivot_shed.columns),
            ))
            figs.append(heat)
            titles.append("E16 ρ vs analytical shed")
            slugs.append("e16_rho_vs_shed_heatmap")
        else:
            scenario = pivot_shed.index[0]
            sub = df[df["scenario"] == scenario]
            # Order metrics by ρ_vs_total so the chart reads left→right by
            # overall agreement with shed; sectors get their own colour.
            order = sub.sort_values("rho_vs_total_shed",
                                    na_position="last")["metric"].tolist()
            sector_color = {
                "total": "#444444",
                "power": eval.NETWORK_COLOR_MAP["electricity"],
                "heat":  eval.NETWORK_COLOR_MAP["heat"],
                "gas":   eval.NETWORK_COLOR_MAP["gas"],
            }
            sector_pretty = {
                "total": "Total", "power": "Electricity",
                "heat": "Heat", "gas": "Gas",
            }
            bar = go.Figure()
            for tag in ("total", "power", "heat", "gas"):
                rho_col = f"rho_vs_{tag}_shed"
                hi_col = f"ci_hi_{tag}_shed"
                lo_col = f"ci_lo_{tag}_shed"
                n_col = f"n_{tag}" if tag != "total" else "n"
                if rho_col not in sub.columns:
                    continue
                sub_t = sub.set_index("metric").reindex(order).reset_index()
                rho = sub_t[rho_col].astype(float)
                hi = sub_t.get(hi_col, pd.Series(dtype=float))
                lo = sub_t.get(lo_col, pd.Series(dtype=float))
                err_hi = (hi - rho).clip(lower=0) if not hi.empty else None
                err_lo = (rho - lo).clip(lower=0) if not lo.empty else None
                n_arr = sub_t[n_col] if n_col in sub_t.columns else sub_t.get("n")
                bar.add_trace(go.Bar(
                    name=sector_pretty[tag],
                    x=sub_t["metric"], y=rho,
                    marker=dict(color=sector_color[tag],
                                line=dict(color="#222", width=0.4)),
                    error_y=dict(
                        type="data", symmetric=False,
                        array=err_hi, arrayminus=err_lo,
                        thickness=1.2, width=3,
                    ) if err_hi is not None else None,
                    customdata=np.c_[n_arr.values] if n_arr is not None else None,
                    hovertemplate=(
                        f"<b>{sector_pretty[tag]}</b><br>"
                        "metric = %{x}<br>ρ = %{y:+.3f}<br>n = %{customdata[0]}<extra></extra>"
                    ),
                ))
            bar.add_hline(y=0, line=dict(color="#444", width=1, dash="dot"))
            bar.update_layout(**_layout(
                title=f"E16 — {pretty_scenario(scenario)}: Spearman ρ vs analytical shed, per sector",
                barmode="group",
                xaxis=dict(title="Metric", tickangle=-25),
                yaxis=dict(title="Spearman ρ", range=[-1.10, 1.10],
                           gridcolor="#e5e5e5"),
                height=460, width=120 + 110 * max(len(order), 1),
                legend=dict(title="Sector"),
            ))
            figs.append(bar)
            titles.append("E16 ρ vs analytical shed — per sector")
            slugs.append("e16_rho_vs_shed_bar_per_sector")

    # ρ vs shed vs ρ vs MC scatter — does shed-quality predict MC-quality?
    fig = go.Figure()
    metrics = list(df["metric"].drop_duplicates())
    cmap = {m: QUAL[i % len(QUAL)] for i, m in enumerate(metrics)}
    for m in metrics:
        sub = df[df["metric"] == m]
        fig.add_trace(go.Scatter(
            x=sub["rho_vs_shed"], y=sub["rho_vs_mc"],
            mode="markers", name=m,
            marker=dict(color=cmap[m], size=11,
                        line=dict(color="#222", width=0.5)),
            hovertext=[pretty_scenario(s) for s in sub["scenario"]],
            hovertemplate=("<b>%{hovertext}</b><br>" + m
                           + "<br>ρ vs shed = %{x:+.3f}"
                           "<br>ρ vs MC = %{y:+.3f}<extra></extra>"),
        ))
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                  line=dict(color="#888", dash="dash", width=1.2))
    fig.add_hline(y=0, line=dict(color="#bbb", width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color="#bbb", width=1, dash="dot"))
    fig.update_layout(**_layout(
        title="E16 — ρ vs analytical shed (x) vs ρ vs MC actual (y)",
        xaxis=dict(title="Spearman ρ vs shed", range=[-1.05, 1.05],
                   gridcolor="#e5e5e5"),
        yaxis=dict(title="Spearman ρ vs MC actual", range=[-1.05, 1.05],
                   gridcolor="#e5e5e5"),
        height=620, width=720,
        legend=dict(title="Metric"),
    ))
    figs.append(fig)
    titles.append("E16 metric ρ — shed vs MC")
    slugs.append("e16_rho_shed_vs_mc")

    # Ceiling: how well does the analytical shed itself predict MC?
    if not ceil.empty:
        ceil = ceil.sort_values("rho_shed_vs_mc", ascending=True)
        err_hi = (ceil["ci_hi"] - ceil["rho_shed_vs_mc"]).clip(lower=0)
        err_lo = (ceil["rho_shed_vs_mc"] - ceil["ci_lo"]).clip(lower=0)
        fig2 = go.Figure(go.Bar(
            y=[pretty_scenario(s) for s in ceil["scenario"]],
            x=ceil["rho_shed_vs_mc"], orientation="h",
            error_x=dict(type="data", symmetric=False,
                         array=err_hi, arrayminus=err_lo,
                         thickness=1.2, width=4, color="#222"),
            marker=dict(color="#00897b", line=dict(color="#004d40", width=0.5)),
            text=[f"ρ={v:+.2f}" for v in ceil["rho_shed_vs_mc"]],
            textposition="outside", cliponaxis=False,
            hovertemplate=("<b>%{y}</b><br>ρ = %{x:+.3f}"
                           "<br>n = %{customdata[0]}<extra></extra>"),
            customdata=np.c_[ceil["n"].values],
            showlegend=False,
        ))
        fig2.add_vline(x=0, line=dict(color="#444", width=1, dash="dot"))
        fig2.update_layout(**_layout(
            title="E16 — Ceiling: ρ between analytical shed and MC actual_total",
            xaxis=dict(title="Spearman ρ", range=[-1.05, 1.10],
                       gridcolor="#e5e5e5"),
            yaxis=dict(title=""),
            height=80 + 36 * len(ceil), width=820,
        ))
        figs.append(fig2)
        titles.append("E16 ceiling ρ")
        slugs.append("e16_ceiling_rho")

    # ── Per-scenario raw scatter + top-K overlap (only if the
    #    refactored experiment emitted merged CSVs) ────────────────────────
    merged_files = sorted(input_dir.glob("E16_*_merged.csv"))
    available_metrics = [m for m in df["metric"].drop_duplicates() if m]
    for mf in merged_files:
        scenario = mf.stem.removeprefix("E16_").removesuffix("_merged")
        try:
            merged = pd.read_csv(mf)
        except Exception as e:
            print(f"[E16] failed to read {mf.name}: {e}")
            continue
        if merged.empty:
            continue
        present_metrics = [m for m in available_metrics if m in merged.columns]
        for m in present_metrics:
            sc_fig = _e16_scatter(merged, m, scenario)
            if sc_fig.data:
                figs.append(sc_fig)
                titles.append(f"E16 {scenario} — {m} vs total_shed")
                slugs.append(f"e16_{scenario}_scatter_{m}")
        ov_fig = _e16_top_overlap(merged, scenario, present_metrics)
        if ov_fig.data:
            figs.append(ov_fig)
            titles.append(f"E16 {scenario} — top-K overlap")
            slugs.append(f"e16_{scenario}_top_k_overlap")

    return _emit(figs, titles, slugs, output_dir / "E16_single_removal.html",
                 "E16 — Single-removal-shed validation")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level discovery
# ─────────────────────────────────────────────────────────────────────────────


PLOTTERS = {
    "E2":  plot_e2_ablation,
    "E4":  plot_e4_distribution,
    "E6":  plot_e6_sensitivity,
    "E7":  plot_e7_mc_validity,
    "E8":  plot_e8_multilayer,
    "E9":  plot_e9_percolation,
    "E10": plot_e10_structural,
    "E11": plot_e11_null_models,
    "E12": plot_e12_community,
    "E13": plot_e13_spectral,
    "E15": plot_e15_structural,
    "E16": plot_e16_single_removal,
}


def plot_all(input_dir: Path, output_dir: Optional[Path] = None,
             experiments: Optional[List[str]] = None) -> Dict[str, Optional[Path]]:
    """Generate every available plot from CSVs in ``input_dir``.

    ``output_dir`` defaults to ``input_dir`` (HTMLs are written next to the
    CSVs). ``experiments`` restricts the run to a subset (e.g. ``["E2","E9"]``).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else input_dir
    chosen = experiments or list(PLOTTERS)
    results: Dict[str, Optional[Path]] = {}
    for key in chosen:
        fn = PLOTTERS.get(key)
        if fn is None:
            continue
        try:
            results[key] = fn(input_dir, output_dir)
            if results[key] is not None:
                print(f"[{key}] wrote {results[key]}")
            else:
                print(f"[{key}] no input CSV found, skipped")
        except Exception as e:
            print(f"[{key}] FAILED: {type(e).__name__}: {e}")
            results[key] = None
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Render dissertation-grade Plotly figures from CMRES eval CSVs.")
    ap.add_argument("input_dir", type=Path,
                    help="directory containing E*_*.csv files")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (defaults to input_dir)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these experiment IDs (e.g. E2 E9 E16)")
    args = ap.parse_args()
    plot_all(args.input_dir, args.out, args.only)
