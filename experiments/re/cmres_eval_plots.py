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

from eval_common import split_scenarios_by_family  # noqa: E402  # canonical
                                                  # scenario families
                                                  # partition, used by every
                                                  # cross-scenario plotter.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cmres.evaluation.evaluation as eval
import pub_style  # shared scare-style publication theme (bar outline, CVD
                  # hatch, top legend, compact sizing)

# Lazy proxy: ``cp_cn_evaluation`` imports back from this module
# (``SECTOR_COLORS`` etc.), so a top-level ``from cp_cn_evaluation import
# pretty_scenario`` hits a partial module when ``cmres_eval_plots`` is
# imported first, silently binds the no-op fallback, and every plot ends
# up showing raw grid keys.  Resolving lazily at call time avoids that —
# by the time any plotter runs, both modules are fully loaded.
def pretty_scenario(name):
    if name is None:
        return ""
    try:
        from cp_cn_evaluation import pretty_scenario as _real
    except Exception:  # pragma: no cover — module imported standalone
        return str(name)
    return _real(name)


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
# Extend the 10-colour qualitative palette with two of Plotly's preset
# palettes so plots that key colour off scenarios / metrics have enough
# distinct hues for the full 22-grid set (11 baselines × 2 stress levels)
# without modulo collisions in the first ~28 categories.
QUAL = eval.PALETTE_QUAL + list(px.colors.qualitative.D3) + list(px.colors.qualitative.Set2)


def _layout(**overrides) -> dict:
    out = dict(template=TEMPLATE)
    out.update(overrides)
    return out


# Single-column dissertation typography. A 1-column figure is roughly
# 84 mm / 3.3 in wide on the printed page, so plotly figures rendered at
# ~800-1200 px get scaled down by ~3-4×. The base cmres template uses
# 12-14 pt fonts, which become 3-4 pt on the page — unreadable. Bumping
# every text element to 18-22 pt keeps the smallest tick label at
# ≳ 5-6 pt after scaling, which is the conventional minimum for printed
# dissertation figures.
_E16_FONT_SIZES = dict(
    base=18,        # default text everywhere
    title=22,       # figure title
    axis_title=20,  # x/y axis titles
    axis_tick=18,   # x/y tick labels
    legend=18,      # legend body
    legend_title=18,
    annotation=16,  # in-plot ρ/p stat boxes
    subplot_title=20,
    colorbar=18,
)


def _e16_layout(**overrides) -> dict:
    """``_layout`` variant with 1-column-dissertation-sized fonts.

    Use for every E16 figure so the published version stays legible when
    scaled down to a single dissertation column. Sizes can be overridden
    per-figure by passing standard layout kwargs (e.g.
    ``font=dict(size=24)``).
    """
    font_size = _E16_FONT_SIZES["base"]
    base = dict(
        template=TEMPLATE,
        font=dict(size=font_size),
        title=dict(font=dict(size=_E16_FONT_SIZES["title"])),
        legend=dict(
            font=dict(size=_E16_FONT_SIZES["legend"]),
            title=dict(font=dict(size=_E16_FONT_SIZES["legend_title"])),
        ),
    )
    # Recursive merge for the dict-of-dicts so callers can pass
    # ``legend=dict(title="…")`` without dropping the font sizes above.
    # Special-case Plotly shorthands ``title="…"`` and
    # ``xaxis_title=…``/``yaxis_title=…`` so the title-font override
    # above can still attach.
    overrides = dict(overrides)
    if "title" in overrides and isinstance(overrides["title"], str):
        overrides["title"] = {"text": overrides["title"]}
    for shorthand_key in list(overrides.keys()):
        if shorthand_key.endswith("_title") and not shorthand_key.startswith("legend"):
            ax_name = shorthand_key[: -len("_title")]  # "xaxis", "yaxis", "xaxis2", ...
            title_val = overrides.pop(shorthand_key)
            title_dict = (
                {"text": title_val} if isinstance(title_val, str) else dict(title_val)
            )
            existing = overrides.get(ax_name)
            if isinstance(existing, dict):
                merged_ax = dict(existing)
                merged_ax.setdefault("title", title_dict)
                if isinstance(merged_ax["title"], str):
                    merged_ax["title"] = {"text": merged_ax["title"]}
                overrides[ax_name] = merged_ax
            else:
                overrides[ax_name] = {"title": title_dict}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merged = dict(base[k])
            for vk, vv in v.items():
                if isinstance(vv, dict) and isinstance(merged.get(vk), dict):
                    sub = dict(merged[vk])
                    sub.update(vv)
                    merged[vk] = sub
                else:
                    merged[vk] = vv
            base[k] = merged
        else:
            base[k] = v
    # Axis titles / tick fonts: apply a default to every named axis the
    # caller mentioned; the template's per-axis defaults can't carry font
    # sizes through ``update_layout`` reliably.
    for ax_key in list(base.keys()):
        if not (ax_key.startswith("xaxis") or ax_key.startswith("yaxis")):
            continue
        ax_val = base[ax_key]
        if not isinstance(ax_val, dict):
            continue
        ax_val = dict(ax_val)
        ax_val.setdefault(
            "title",
            {"font": {"size": _E16_FONT_SIZES["axis_title"]}},
        )
        if isinstance(ax_val["title"], str):
            ax_val["title"] = {
                "text": ax_val["title"],
                "font": {"size": _E16_FONT_SIZES["axis_title"]},
            }
        elif isinstance(ax_val["title"], dict):
            t = dict(ax_val["title"])
            tf = dict(t.get("font") or {})
            tf.setdefault("size", _E16_FONT_SIZES["axis_title"])
            t["font"] = tf
            ax_val["title"] = t
        tf = dict(ax_val.get("tickfont") or {})
        tf.setdefault("size", _E16_FONT_SIZES["axis_tick"])
        ax_val["tickfont"] = tf
        base[ax_key] = ax_val
    return base


def _e16_bump_subplot_titles(fig: "go.Figure") -> None:
    """Bump the auto-created ``subplot_titles`` annotation font sizes."""
    for ann in (fig.layout.annotations or ()):
        # Subplot titles are annotations with ``xref`` ending in 'paper'
        # (added by ``make_subplots``). Don't touch in-plot annotations
        # (which use 'x domain' / 'y domain').
        xref = getattr(ann, "xref", "") or ""
        if xref.endswith("paper"):
            font = dict(ann.font.to_plotly_json()) if ann.font else {}
            font["size"] = _E16_FONT_SIZES["subplot_title"]
            ann.font = font


# ─────────────────────────────────────────────────────────────────────────────
# Canonical metric labels — same display strings the cp_cn_evaluation
# bar / scatter / NDCG figures use (from ``eval_common.CORE_METRIC_LABELS``).
# E16 plots historically rendered raw column names like ``"predicted_score"``
# while cp_cn rendered ``"PTDF stress + phys. BC"`` — route every E16
# axis / legend label through :func:`metric_label` so the two report
# pipelines stay in sync.
# ─────────────────────────────────────────────────────────────────────────────


def metric_label(col_or_iter):
    """Translate a metric column name to the canonical display label.

    Accepts a single string or any iterable of strings. Returns either a
    string or a list, matching the input shape. Unknown metric names fall
    through unchanged so this is safe to apply blanket-style to any axis
    that may contain non-CORE metrics.
    """
    try:
        from eval_common import CORE_METRIC_LABELS as _LBL
    except Exception:
        _LBL = {}
    if isinstance(col_or_iter, str):
        return _LBL.get(col_or_iter, col_or_iter)
    return [_LBL.get(c, c) for c in col_or_iter]


# ─────────────────────────────────────────────────────────────────────────────
# Shared sector palette + bar style — single source of truth for legend
# colours / labels / Bar styling used by both the E16 plots and the
# cp_cn cross-carrier / ρ figures. Keep all sector-bearing plots
# referencing these names so the dissertation legend stays consistent.
# ─────────────────────────────────────────────────────────────────────────────

#: Canonical sector ↔ colour map. ``power`` / ``heat`` / ``gas`` reuse the
#: cmres ``NETWORK_COLOR_MAP`` so plain carrier figures match the sector
#: figures pixel-perfect.
SECTOR_COLORS: Dict[str, str] = {
    "total": "#444444",
    "multi": "#7e57c2",  # purple — distinct from any single-carrier hue
    "power": eval.NETWORK_COLOR_MAP["electricity"],
    "heat":  eval.NETWORK_COLOR_MAP["heat"],
    "gas":   eval.NETWORK_COLOR_MAP["gas"],
}

#: Sector ↔ display label. Used as the trace ``name=`` (and therefore as
#: the legend entry text) on every sector-bearing bar. The qualifier on
#: ``"Multi (CPs)"`` is the only non-obvious entry — the plain carrier
#: bars don't need a partition suffix.
SECTOR_PRETTY: Dict[str, str] = {
    "total": "Total",
    "multi": "Multi (CPs)",
    "power": "Electricity",
    "heat":  "Heat",
    "gas":   "Gas",
}

#: Canonical sector display order. Anything that builds a sector legend
#: should iterate this list (skipping keys it doesn't carry).
SECTOR_ORDER: List[str] = ["total", "multi", "power", "heat", "gas"]


def outlined_marker(color: str) -> dict:
    """``marker=`` with the canonical 0.4 px ``#222`` outline. Use for any
    Bar that should match the E16 style but isn't sector- or carrier-keyed
    (per-metric ρ bars, ranking-accuracy bars, etc.)."""
    return dict(color=color, line=dict(color="#222", width=0.4))


def sector_marker(sector_key: str, *, alpha: Optional[float] = None) -> dict:
    """Return the ``marker=`` kwargs for one sector bar (color + outline)."""
    return outlined_marker(SECTOR_COLORS.get(sector_key, "#888888"))


def carrier_marker(carrier_name: str) -> dict:
    """``marker=`` for plain ``electricity``/``heat``/``gas`` bars — same
    outline + line styling as :func:`sector_marker` but coloured straight
    from ``NETWORK_COLOR_MAP`` (so it's safe to pass ``"electricity"`` /
    ``"heat"`` / ``"gas"`` directly without translating to a sector key).
    """
    return outlined_marker(eval.NETWORK_COLOR_MAP.get(carrier_name, "#888888"))


def bar_error_kwargs(
    *,
    err_hi=None, err_lo=None,
    axis: str = "y",
) -> Optional[dict]:
    """Build a uniform ``error_x`` / ``error_y`` dict matching the E16
    bar styling. ``err_hi`` / ``err_lo`` may be None when only one side
    is available; the other defaults to zero. Returns ``None`` when no
    error data has been supplied so callers can use ``error_y=...`` /
    ``error_x=...`` directly without branching."""
    if err_hi is None and err_lo is None:
        return None
    if err_hi is None:
        err_hi = [0.0] * len(err_lo)
    if err_lo is None:
        err_lo = [0.0] * len(err_hi)
    return dict(
        type="data", symmetric=False,
        array=list(err_hi), arrayminus=list(err_lo),
        thickness=1.2, width=3,
    )


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

    # Pooled heatmap of Δ vs full (signed, diverging) — split by stress
    # family when several scenario families are present so a
    # 22-row matrix doesn't overflow the printed page.
    def _emit_delta_heatmap(sub_pooled, class_label):
        delta = sub_pooled[sub_pooled["variant"] != "full"].pivot_table(
            index="scenario", columns="variant", values="delta_vs_full",
        )
        delta = delta.reindex(columns=[v for v in _VARIANT_ORDER if v != "full"])
        delta = delta.reindex(index=_scenario_order(delta.index))
        if delta.empty:
            return
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
        title_suffix = f" ({class_label})" if class_label else ""
        slug_suffix = f"_{class_label}" if class_label else ""
        heat.update_layout(**_layout(
            title=(
                "E2 — Per-factor ablation effect (Δρ vs full) across scenarios"
                + title_suffix
            ),
            xaxis=dict(title=""), yaxis=dict(title=""),
            height=80 + 36 * len(delta.index), width=860,
        ))
        figs.append(heat)
        titles.append(f"E2 ablation — pooled Δρ heatmap{title_suffix}")
        slugs.append(f"e2_ablation_pooled_heatmap{slug_suffix}")

    classes = split_scenarios_by_family(pooled["scenario"].drop_duplicates())
    if len(classes) <= 1:
        _emit_delta_heatmap(pooled, class_label="")
    else:
        for cl, scens in classes:
            _emit_delta_heatmap(pooled[pooled["scenario"].isin(scens)], class_label=cl)

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

    def _slug(base, class_label):
        return f"{base}_{class_label}" if class_label else base

    def _title_suffix(class_label):
        return f" ({class_label})" if class_label else ""

    def _emit_rhw(sub_rhw, class_label):
        scenarios = _scenario_order(sub_rhw["scenario"].unique())
        if not scenarios:
            return
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
            sub = sub_rhw[sub_rhw["scenario"] == s]
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
            title=(
                "E7 — RHW(n) convergence per carrier (target = 0.05, dashed)"
                + _title_suffix(class_label)
            ),
            height=320 * nrows + 60, width=370 * ncols,
            legend=dict(title="Carrier", orientation="h",
                        y=-0.10, x=0.5, xanchor="center"),
        ))
        figs.append(fig)
        titles.append(f"E7 RHW convergence{_title_suffix(class_label)}")
        slugs.append(_slug("e7_rhw_convergence", class_label))

    def _emit_av(sub_s, class_label):
        if sub_s.empty:
            return
        sub_s = sub_s.sort_values("AV_reduction_factor", na_position="first")
        colors = ["#388e3c" if (np.isfinite(v) and v > 1.0) else "#d32f2f"
                  for v in sub_s["AV_reduction_factor"]]
        fig = go.Figure(go.Bar(
            y=[pretty_scenario(x) for x in sub_s["scenario"]],
            x=sub_s["AV_reduction_factor"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="#222", width=0.6)),
            text=[f"{v:.2f}×" if np.isfinite(v) else "n/a"
                  for v in sub_s["AV_reduction_factor"]],
            textposition="outside", cliponaxis=False,
            hovertemplate=("<b>%{y}</b><br>AV reduction = %{x:.3f}×"
                           "<br>n_runs = %{customdata[0]}<extra></extra>"),
            customdata=np.c_[sub_s["n_runs"].values] if "n_runs" in sub_s else None,
            showlegend=False,
        ))
        fig.add_vline(x=1.0, line=dict(color="#444", width=1.2, dash="dash"),
                      annotation_text="no reduction", annotation_position="top")
        fig.update_layout(**_layout(
            title=(
                "E7 — Antithetic-variates variance-reduction factor"
                + _title_suffix(class_label)
            ),
            xaxis=dict(title="Var(naive) / [2 · Var(pair-mean)]",
                       gridcolor="#e5e5e5"),
            yaxis=dict(title=""),
            height=80 + 36 * len(sub_s), width=820,
        ))
        figs.append(fig)
        titles.append(f"E7 AV reduction factor{_title_suffix(class_label)}")
        slugs.append(_slug("e7_av_reduction_factor", class_label))

    if rhw_path.exists():
        rhw = pd.read_csv(rhw_path)
        if not rhw.empty:
            classes = split_scenarios_by_family(rhw["scenario"].drop_duplicates())
            if len(classes) <= 1:
                _emit_rhw(rhw, class_label="")
            else:
                for cl, scens in classes:
                    _emit_rhw(rhw[rhw["scenario"].isin(scens)], class_label=cl)

    if sum_path.exists():
        s = pd.read_csv(sum_path)
        if not s.empty and "AV_reduction_factor" in s.columns:
            classes = split_scenarios_by_family(s["scenario"].drop_duplicates())
            if len(classes) <= 1:
                _emit_av(s, class_label="")
            else:
                for cl, scens in classes:
                    _emit_av(s[s["scenario"].isin(scens)], class_label=cl)

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
    # y = z-score, with shaded |z| > 1.96 region. Split by scenario family
    # when several families are present so the per-stat
    # panel widths stay readable.
    stats = list(df["statistic"].drop_duplicates())
    nulls = sorted(df["null_kind"].dropna().unique())
    null_color = {k: QUAL[i % len(QUAL)] for i, k in enumerate(nulls)}

    def _emit_null_z(sub_df, class_label):
        fig = make_subplots(
            rows=1, cols=len(stats),
            subplot_titles=stats, shared_yaxes=True, horizontal_spacing=0.05,
        )
        for j, stat in enumerate(stats, start=1):
            sub_stat = sub_df[sub_df["statistic"] == stat]
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
        title_suffix = f" ({class_label})" if class_label else ""
        slug_suffix = f"_{class_label}" if class_label else ""
        fig.update_layout(**_layout(
            title=(
                "E11 — Observed structural quantities vs null ensembles "
                "(|z|>1.96 shaded)" + title_suffix
            ),
            barmode="group",
            height=560, width=max(640, 380 * len(stats)),
            legend=dict(title="Null model", orientation="h",
                        y=-0.25, x=0.5, xanchor="center"),
        ))
        figs.append(fig)
        titles.append(f"E11 null z-scores{title_suffix}")
        slugs.append(f"e11_null_z_scores{slug_suffix}")

    classes = split_scenarios_by_family(df["scenario"].drop_duplicates())
    if len(classes) <= 1:
        _emit_null_z(df, class_label="")
    else:
        for cl, scens in classes:
            _emit_null_z(df[df["scenario"].isin(scens)], class_label=cl)

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
                        + f"{metric_label(metric)}=%{{x:.4g}}<br>{sector_col}=%{{y:.4g}}<extra></extra>"
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
                font=dict(size=_E16_FONT_SIZES["annotation"]),
                row=row, col=col,
            )
        fig.update_yaxes(type="log", row=row, col=col, title_text=f"{sector_col} (MW, log)")
        fig.update_xaxes(
            row=row, col=col,
            title_text=metric_label(metric) if row == 2 else "",
        )

    if not any_data:
        return go.Figure()

    fig.update_layout(**_e16_layout(
        title=f"{pretty_scenario(scenario)}: {metric_label(metric)} vs analytical shed (per sector)",
        height=720, width=1080,
        legend=dict(title="cp_type"),
    ))
    # Bump axis-title and tick fonts on every sub-axis (update_layout above
    # only set fonts on the ``xaxis``/``yaxis`` keys it received).
    fig.update_xaxes(
        title_font=dict(size=_E16_FONT_SIZES["axis_title"]),
        tickfont=dict(size=_E16_FONT_SIZES["axis_tick"]),
    )
    fig.update_yaxes(
        title_font=dict(size=_E16_FONT_SIZES["axis_title"]),
        tickfont=dict(size=_E16_FONT_SIZES["axis_tick"]),
    )
    _e16_bump_subplot_titles(fig)
    return fig


def _e16_sector_hbar(metric_labels: List[str], traces: list, *,
                     title: str) -> go.Figure:
    """Render a horizontal grouped per-sector ρ bar in the shared scare style.

    ``metric_labels`` is the bottom→top y-category order (highest ρ on top).
    ``traces`` is a list of ``(tag, rho_array, err_hi, err_lo, n_arr)`` where
    ``tag`` is a sector key (``total`` / ``multi`` / ``power`` / ``heat`` /
    ``gas``). Sector hue is kept; a CVD hatch is layered on top; legend rides
    on top horizontally and the figure height tracks the metric count so the
    bars stay slim instead of ballooning.

    Horizontal because the metric names ("PTDF stress + phys. BC", …) are long
    — down the y-axis they read cleanly without the −25° tick rotation the old
    vertical layout needed, and the figure no longer grows to ~1400 px wide.
    """
    bar = go.Figure()
    any_trace = False
    for tag, rho, err_hi, err_lo, n_arr in traces:
        if rho is None or not rho.notna().any():
            continue
        any_trace = True
        err_x = (
            dict(type="data", symmetric=False,
                 array=list(err_hi), arrayminus=list(err_lo),
                 thickness=1.2, width=4, color=pub_style.MUTED_COLOR)
            if err_hi is not None else None
        )
        bar.add_trace(go.Bar(
            name=pub_style.SECTOR_PRETTY.get(tag, tag),
            y=metric_labels, x=rho, orientation="h",
            marker=pub_style.sector_marker(tag),
            error_x=err_x,
            customdata=np.c_[n_arr.values] if n_arr is not None else None,
            hovertemplate=(
                f"<b>{pub_style.SECTOR_PRETTY.get(tag, tag)}</b><br>"
                "metric = %{y}<br>ρ = %{x:+.3f}<br>n = %{customdata[0]}<extra></extra>"
            ),
        ))
    if not any_trace:
        return go.Figure()
    bar.add_vline(x=0, line=dict(color="#444", width=1, dash="dot"))
    bar.update_layout(barmode="group")
    n_series = len(bar.data)
    pub_style.apply_theme(
        bar, title=title,
        height=pub_style.hbar_height(len(metric_labels), n_series),
        width=pub_style.BAR_FIG_WIDTH, font_bump=1, legend_top=True,
    )
    bar.update_xaxes(title="Spearman ρ", range=[-1.10, 1.10])
    bar.update_yaxes(title="")
    return bar


def _e16_rho_per_sector_bar(df: pd.DataFrame, scenario: str) -> go.Figure:
    """Grouped bar chart for one scenario: per metric, one bar per
    *sector slice* (total / multi / electricity / heat / gas), with sector
    legend and CI error bars.

    The slices are partitioned by component kind so the per-carrier signal
    isn't distorted by coupling-point rows:
      total       — every matched row (the headline number).
      multi       — CP rows only (compound + branch_cp), vs total_shed —
                    how well the metric ranks coupling components.
      power/heat/gas — non-CP rows only, vs same-sector shed — clean
                    within-carrier ranking on plain branches/pipes.

    This is the canonical per-scenario per-sector correlation view."""
    sub = df[df["scenario"] == scenario]
    if sub.empty:
        return go.Figure()
    order = sub.sort_values("rho_vs_total_shed",
                            na_position="last")["metric"].tolist()
    sub_t = sub.set_index("metric").reindex(order).reset_index()
    metric_labels = metric_label(list(sub_t["metric"]))
    traces = []
    for tag in SECTOR_ORDER:
        rho_col = f"rho_vs_{tag}_shed"
        if rho_col not in sub.columns:
            continue
        rho = sub_t[rho_col].astype(float)
        if not rho.notna().any():
            continue
        hi = sub_t.get(f"ci_hi_{tag}_shed", pd.Series(dtype=float))
        lo = sub_t.get(f"ci_lo_{tag}_shed", pd.Series(dtype=float))
        err_hi = (hi - rho).clip(lower=0) if not hi.empty else None
        err_lo = (rho - lo).clip(lower=0) if not lo.empty else None
        n_col = f"n_{tag}" if tag != "total" else "n"
        n_arr = sub_t[n_col] if n_col in sub_t.columns else sub_t.get("n")
        traces.append((tag, rho, err_hi, err_lo, n_arr))
    return _e16_sector_hbar(
        metric_labels, traces,
        title=f"{pretty_scenario(scenario)}: Spearman ρ vs analytical shed, per sector",
    )


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
            rows.append({"metric": metric_label(m), "k": k,
                         "overlap": len(top_metric & top_shed) / k})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = px.line(
        df, x="k", y="overlap", color="metric", markers=True,
        color_discrete_sequence=QUAL,
    )
    fig.update_layout(**_e16_layout(
        title=f"{pretty_scenario(scenario)}: top-K overlap (metric vs analytical shed)",
        xaxis_title="K", yaxis_title="|top_K(metric) ∩ top_K(shed)| / K",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=420, width=860, legend=dict(title="Metric"),
    ))
    return fig


def _e16_rho_per_sector_bar_aggregated(
    merged_files: Sequence[Path],
) -> go.Figure:
    """Pooled per-sector ρ bar chart: ρ recomputed on rows concatenated
    across every scenario's ``E16_<scenario>_merged.csv``.

    Same sector partitioning as ``_e16_rho_per_sector_bar`` (total / multi /
    power / heat / gas) so the headline numbers are directly comparable to
    the per-scenario plots — but every metric × sector gets a single ρ
    computed on the union of all scenarios' rows rather than one ρ per
    scenario. Gives the strongest statistical view of which metrics rank
    components well across the whole study.
    """
    try:
        from eval_common import spearman_with_ci as _spearman_with_ci
        from eval_common import MC_FAILED_EPS as _MC_FAILED_EPS
    except Exception:
        # Plot module shouldn't hard-fail if eval_common changes shape.
        from scipy.stats import spearmanr as _spearmanr

        def _spearman_with_ci(a, b):  # type: ignore[no-redef]
            r = _spearmanr(a, b)
            return float(r.statistic), float(r.pvalue), float("nan"), float("nan")

        _MC_FAILED_EPS = 1e-6  # type: ignore[assignment]

    if not merged_files:
        return go.Figure()

    frames: List[pd.DataFrame] = []
    for mf in merged_files:
        try:
            d = pd.read_csv(mf)
        except Exception as e:
            print(f"[E16:aggregated] failed to read {mf.name}: {e}")
            continue
        if d.empty:
            continue
        d = d.copy()
        d["_scenario"] = mf.stem.removeprefix("E16_").removesuffix("_merged")
        frames.append(d)
    if not frames:
        return go.Figure()
    pooled = pd.concat(frames, ignore_index=True)

    # Same partitioning as cmres_eval._E16 sector_specs.
    sector_specs = [
        ("total", "total_shed", None),
        ("multi", "total_shed", "cp"),
        ("power", "power_shed", "branch"),
        ("heat",  "heat_shed",  "branch"),
        ("gas",   "gas_shed",   "branch"),
    ]

    # Candidate metric columns: any non-meta column with float-like values
    # present on every scenario. Exclude the known-shed / id / metadata
    # columns so we don't accidentally rank against keys.
    META = {
        "scenario", "metric", "cp_id", "cp_type", "kind",
        "total_shed", "power_shed", "heat_shed", "gas_shed",
        "_join_cp_id", "_scenario",
    }
    candidate_metrics = [
        c for c in pooled.columns
        if c not in META and pd.api.types.is_numeric_dtype(pooled[c])
    ]
    # Cosmetic: keep only the same set of metrics the per-scenario bar
    # already reports on, i.e. those that produced ρ rows in the metric
    # CSV. To avoid plumbing that list around, derive it from any metric
    # that has at least one non-NaN value across the pooled set AND varies.
    candidate_metrics = [
        m for m in candidate_metrics
        if pooled[m].notna().sum() >= 3 and pooled[m].nunique() >= 2
        and m not in {"actual_total", "rho_vs_shed"}
    ]
    if not candidate_metrics:
        return go.Figure()

    # ρ table: rows = metric, cols = sector tag.
    rho_records: List[dict] = []
    for m in candidate_metrics:
        row: dict = {"metric": m}
        for tag, col, kind_filter in sector_specs:
            sub = pooled[pooled[m].notna()]
            if col not in sub.columns:
                row[f"rho_{tag}"] = float("nan")
                row[f"n_{tag}"] = 0
                continue
            if kind_filter == "branch" and "kind" in sub.columns:
                sub = sub[sub["kind"] == "branch"]
            elif kind_filter == "cp" and "kind" in sub.columns:
                sub = sub[sub["kind"] != "branch"]
            if tag != "total":
                sub = sub[sub[col].notna() & (sub[col] > 0)]
            else:
                sub = sub[sub[col].notna()]
            if len(sub) < 3 or sub[m].nunique() < 2 or sub[col].nunique() < 2:
                row[f"rho_{tag}"] = float("nan")
                row[f"ci_lo_{tag}"] = float("nan")
                row[f"ci_hi_{tag}"] = float("nan")
                row[f"n_{tag}"] = int(len(sub))
                continue
            rho, _p, lo, hi = _spearman_with_ci(sub[m], sub[col])
            row[f"rho_{tag}"] = float(rho)
            row[f"ci_lo_{tag}"] = float(lo)
            row[f"ci_hi_{tag}"] = float(hi)
            row[f"n_{tag}"] = int(len(sub))
        rho_records.append(row)
    rho_df = pd.DataFrame(rho_records)
    if rho_df.empty:
        return go.Figure()
    # Order metrics by the headline (total) ρ, NaN-last, matching the
    # per-scenario bar so the two figures read in the same direction.
    # Ascending so the highest-ρ metric lands on top of the horizontal bar.
    rho_df = rho_df.sort_values("rho_total", ascending=True, na_position="first")
    metric_labels = metric_label(rho_df["metric"].tolist())

    traces = []
    for tag, _col, _kind in sector_specs:
        rho_col = f"rho_{tag}"
        if rho_col not in rho_df.columns:
            continue
        rho = rho_df[rho_col].astype(float)
        if not rho.notna().any():
            continue
        hi = rho_df.get(f"ci_hi_{tag}", pd.Series(dtype=float))
        lo = rho_df.get(f"ci_lo_{tag}", pd.Series(dtype=float))
        err_hi = (hi - rho).clip(lower=0) if not hi.empty else None
        err_lo = (rho - lo).clip(lower=0) if not lo.empty else None
        n_arr = rho_df[f"n_{tag}"]
        traces.append((tag, rho, err_hi, err_lo, n_arr))
    n_total_row = int(rho_df["n_total"].max()) if "n_total" in rho_df.columns else 0
    return _e16_sector_hbar(
        metric_labels, traces,
        title=(
            f"Pooled across all scenarios (n={n_total_row}): "
            "Spearman ρ vs analytical shed, per sector"
        ),
    )


_E16_RANK_METRICS = [
    "predicted_score", "predicted_score_cp_aware", "predicted_score_balanced",
    "predicted_stress", "topo_bc", "stress_bc", "katz_score", "vitality_score",
    "local_score", "self_score",
]


def _e16_ranking_per_sector_aggregated(
    merged_files: Sequence[Path], k: int = 10,
) -> go.Figure:
    """Pooled per-sector *top-of-ranking* accuracy: Kendall τ, NDCG@k and
    precision@k recomputed on rows concatenated across every scenario's
    ``E16_<scenario>_merged.csv``, using the same sector partitioning as
    ``_e16_rho_per_sector_bar_aggregated`` (total / multi / power / heat / gas).

    Three horizontal grouped panels (one per measure), bars grouped by sector.
    Complements the per-sector ρ bar: ρ measures full-list monotonicity, these
    measure how well the few most critical components surface — which is what a
    planner screening a handful of coupling points actually cares about. Pooling
    these across sectors collapses the signal, so they are resolved per sector.
    """
    from plotly.subplots import make_subplots
    from scipy.stats import kendalltau as _kendalltau
    try:
        from eval_common import ndcg as _ndcg
    except Exception:  # pragma: no cover
        _ndcg = None

    if not merged_files:
        return go.Figure()
    frames: List[pd.DataFrame] = []
    for mf in merged_files:
        try:
            d = pd.read_csv(mf)
        except Exception:
            continue
        if not d.empty:
            frames.append(d)
    if not frames:
        return go.Figure()
    pooled = pd.concat(frames, ignore_index=True)

    sector_specs = [
        ("total", "total_shed", None),
        ("power", "power_shed", "branch"),
        ("heat",  "heat_shed",  "branch"),
        ("gas",   "gas_shed",   "branch"),
        ("multi", "total_shed", "cp"),
    ]
    metrics = [m for m in _E16_RANK_METRICS
               if m in pooled.columns and pooled[m].notna().sum() >= 3
               and pooled[m].nunique() >= 2]
    if not metrics:
        return go.Figure()

    def _prec_at_k(scores, gains, kk):
        s = np.asarray(scores, float)
        g = np.asarray(gains, float)
        kk = min(kk, len(s))
        if kk < 1:
            return float("nan")
        return len(set(np.argsort(-s)[:kk]) & set(np.argsort(-g)[:kk])) / kk

    def _ndcg_at_k(scores, gains, kk):
        if _ndcg is not None:
            return float(_ndcg(np.clip(np.asarray(gains, float), 0, None),
                               np.asarray(scores, float), k=kk))
        g = np.clip(np.asarray(gains, float), 0, None)
        order = np.argsort(-np.asarray(scores, float))[:kk]
        disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
        dcg = (g[order] * disc).sum()
        ig = np.sort(g)[::-1][:kk]
        idcg = (ig * disc[:len(ig)]).sum()
        return dcg / idcg if idcg > 0 else float("nan")

    # res[(metric, tag)] = (kendall, ndcg, precision)
    res: dict = {}
    n_total = 0
    for m in metrics:
        for tag, col, kind_filter in sector_specs:
            sub = pooled[pooled[m].notna()]
            if col not in sub.columns:
                res[(m, tag)] = (float("nan"),) * 3
                continue
            if kind_filter == "branch" and "kind" in sub.columns:
                sub = sub[sub["kind"] == "branch"]
            elif kind_filter == "cp" and "kind" in sub.columns:
                sub = sub[sub["kind"] != "branch"]
            if tag != "total":
                sub = sub[sub[col].notna() & (sub[col] > 0)]
            else:
                sub = sub[sub[col].notna()]
                n_total = max(n_total, len(sub))
            if len(sub) < 3 or sub[m].nunique() < 2 or sub[col].nunique() < 2:
                res[(m, tag)] = (float("nan"),) * 3
                continue
            x, ref = sub[m].values, sub[col].values
            res[(m, tag)] = (float(_kendalltau(x, ref).correlation),
                             _ndcg_at_k(x, ref, k), _prec_at_k(x, ref, k))

    metric_labels = metric_label(metrics)
    panels = [(0, "Kendall τ", [-0.5, 1.0]),
              (1, f"NDCG@{k}", [0.0, 1.0]),
              (2, f"Precision@{k}", [0.0, 1.0])]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[t for _, t, _ in panels])
    for ci, (idx, _t, _rng) in enumerate(panels, start=1):
        for tag, _col, _kf in sector_specs:
            vals = [res.get((m, tag), (float("nan"),) * 3)[idx] for m in metrics]
            fig.add_trace(go.Bar(
                y=metric_labels, x=vals, orientation="h",
                name=pub_style.SECTOR_PRETTY.get(tag, tag), legendgroup=tag,
                showlegend=(ci == 1),
                marker=pub_style.bar_marker(pub_style.SECTOR_COLOR.get(tag, "#888888")),
            ), row=1, col=ci)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.04)
    for ci, (_idx, _t, rng) in enumerate(panels, start=1):
        fig.update_xaxes(range=rng, row=1, col=ci)
    pub_style.apply_theme(
        fig,
        title=(f"Pooled across scenarios (n={n_total}): per-sector ranking "
               f"accuracy vs analytical shed"),
        width=1180, height=620, legend_top=True)
    fig.update_yaxes(autorange="reversed")
    return fig


def _e16_per_sector_heatmap(df: pd.DataFrame) -> go.Figure:
    """Extended E16 heatmap with per-sector y-axis resolution.

    Each scenario expands into 5 rows (total / multi / electricity / heat /
    gas). Cells = Spearman ρ for that (scenario, sector, metric) combination.
    Horizontal separators are drawn between scenario blocks so the
    grouped structure is visually obvious.

    Input is the E16_metric_vs_shed.csv frame; the per-sector ρ values
    live in ``rho_vs_<tag>_shed`` columns.
    """
    if df is None or df.empty:
        return go.Figure()

    sector_tags = ["total", "multi", "power", "heat", "gas"]
    sector_short = {
        "total": "Total",
        "multi": "Multi",
        "power": "Electricity",
        "heat":  "Heat",
        "gas":   "Gas",
    }

    scenarios = _scenario_order(df["scenario"].unique())
    metrics = list(df["metric"].drop_duplicates())
    if not scenarios or not metrics:
        return go.Figure()

    # Build (scenario, sector) row index. Sectors in fixed display order
    # so every scenario block has the same internal layout.
    row_labels: List[str] = []
    row_keys: List[tuple] = []
    for s in scenarios:
        for tag in sector_tags:
            row_labels.append(f"{pretty_scenario(s)} — {sector_short[tag]}")
            row_keys.append((s, tag))

    # Vectorised build of the (rows × metrics) ρ matrix: one pivot per
    # sector column (5), each indexed/reindexed to the canonical row /
    # column order. Replaces a triple-nested Python loop over
    # ``df_idx.loc[(s, m), col]`` that did ~scenarios × sectors × metrics
    # individual MultiIndex lookups.
    sector_rho_cols = [f"rho_vs_{t}_shed" for t in sector_tags]
    sector_rho_cols = [c for c in sector_rho_cols if c in df.columns]
    z = np.full((len(row_labels), len(metrics)), np.nan, dtype=float)
    # ``scenario`` ordered against the resolved ``scenarios`` list so
    # ``row_keys`` and the matrix rows stay aligned 1-to-1.
    for col_name in sector_rho_cols:
        tag = col_name[len("rho_vs_"):-len("_shed")]
        pivot = (
            df.pivot_table(
                index="scenario", columns="metric",
                values=col_name, aggfunc="first",
            )
            .reindex(index=scenarios, columns=metrics)
        )
        # Place pivot rows into the corresponding scenario block.
        for scen_i, s in enumerate(scenarios):
            try:
                row_i = scen_i * len(sector_tags) + sector_tags.index(tag)
            except ValueError:
                continue
            row_vals = pivot.loc[s].to_numpy(dtype=float, na_value=np.nan)
            z[row_i, :] = row_vals

    # Drop fully-NaN columns / rows so Kaleido can scale the axes.
    finite_cols = ~np.all(np.isnan(z), axis=0)
    z = z[:, finite_cols]
    metrics = [m for m, keep in zip(metrics, finite_cols) if keep]
    finite_rows = ~np.all(np.isnan(z), axis=1)
    z = z[finite_rows, :]
    row_labels = [r for r, keep in zip(row_labels, finite_rows) if keep]
    row_keys = [k for k, keep in zip(row_keys, finite_rows) if keep]
    if z.size == 0:
        return go.Figure()

    # +2 pt over the base E16 typography on every text element of this
    # figure (per dissertation request — this heatmap is the headline E16
    # figure and needs a touch more presence than the surrounding plots).
    ann_size = _E16_FONT_SIZES["annotation"] + 2
    cbar_size = _E16_FONT_SIZES["colorbar"] + 2

    # Translate metric column names to canonical display labels for the
    # x-axis (and the per-cell annotation x positions below) so this
    # heatmap reads identically to the cp_cn pairwise-ρ heatmap.
    metric_labels = metric_label(metrics)

    heat = go.Figure(go.Heatmap(
        z=z, x=metric_labels, y=row_labels,
        # Match the cp_cn pairwise-ρ heatmap convention: ``RdBu_r`` with
        # ``reversescale=False`` puts red on positive ρ and blue on
        # negative ρ (the corr_fig in pooled_metric_comparison uses the
        # same setting — the two heatmaps now read the same).
        colorscale="RdBu_r", reversescale=False,
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(
                text="Spearman ρ", side="right",
                font=dict(size=cbar_size),
            ),
            tickfont=dict(size=cbar_size),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            thickness=14, len=0.9,
        ),
        xgap=1, ygap=1,
        hovertemplate="<b>%{y}</b><br>%{x}: ρ = %{z:.3f}<extra></extra>",
    ))

    # Cell annotations with adaptive text colour — same logic as the
    # cp_cn pairwise heatmap: white text on saturated cells (|ρ| > 0.55)
    # for legibility, black on light cells. Using a layout-level
    # annotation list (instead of texttemplate) gives us per-cell colour
    # control plus the +2 pt size.
    annotations = []
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            if not np.isfinite(v):
                continue
            text_color = "white" if abs(v) > 0.55 else "black"
            annotations.append(dict(
                x=metric_labels[j], y=row_labels[i],
                text=f"{v:.2f}",
                xref="x", yref="y", showarrow=False,
                font=dict(size=ann_size, color=text_color),
            ))

    # Scenario block separators as layout shapes so the per-sector rows
    # that belong together read as a group.
    shapes = []
    if row_keys:
        prev_scenario = row_keys[0][0]
        for i, (s, _tag) in enumerate(row_keys[1:], start=1):
            if s != prev_scenario:
                shapes.append(dict(
                    type="line", xref="paper",
                    x0=0, x1=1,
                    yref="y", y0=i - 0.5, y1=i - 0.5,
                    line=dict(color="#333", width=1.4),
                ))
                prev_scenario = s

    heat.update_layout(**_e16_layout(
        title=(
            "Spearman ρ between metric and analytical shed, "
            "per scenario × sector"
        ),
        # tickangle matches the cp_cn pairwise heatmap (-35 vs the
        # previous +30) so the two figures share an axis orientation.
        xaxis=dict(title="Metric", tickangle=-35, automargin=True),
        yaxis=dict(
            title="Scenario — Sector",
            autorange="reversed",
            automargin=True,
            # Gridlines between every category for per-sector resolution
            # (in addition to the heavier scenario-boundary separators).
            showgrid=True, gridcolor="#eeeeee", gridwidth=1,
        ),
        # +8 px per row over the previous 28 → 36 px row pitch so the
        # extended figure has more vertical breathing room.
        height=max(420, 100 + 36 * len(row_labels)),
        width=240 + 110 * max(len(metrics), 1),
        font=dict(size=_E16_FONT_SIZES["base"] + 2),
        title_font=dict(size=_E16_FONT_SIZES["title"] + 2),
        annotations=annotations,
        shapes=shapes,
    ))
    # Bump axis title / tick fonts +2 too (``_e16_layout`` set them at
    # the base sizes; plotly's layout-level ``font`` doesn't propagate
    # to ``xaxis.title.font`` or ``tickfont`` automatically).
    heat.update_xaxes(
        title_font=dict(size=_E16_FONT_SIZES["axis_title"] + 2),
        tickfont=dict(size=_E16_FONT_SIZES["axis_tick"] + 2),
    )
    heat.update_yaxes(
        title_font=dict(size=_E16_FONT_SIZES["axis_title"] + 2),
        tickfont=dict(size=_E16_FONT_SIZES["axis_tick"] + 2),
    )
    return heat


def plot_e16_single_removal(input_dir: Path, output_dir: Path) -> Optional[Path]:
    """Per-metric ρ vs analytical shed AND vs MC, with the ceiling
    (ρ_shed_vs_mc) drawn as a hatched reference for each scenario.

    When per-scenario ``E16_<scenario>_merged.csv`` files are present
    (emitted by the refactored ``experiment_e16_single_removal_validation``),
    appends per-metric scatter panels and a top-K overlap curve per scenario.

    When the input mixes scenario families the cross-scenario
    figures (heatmap, pooled bar, shed-vs-MC scatter, ceiling) are emitted
    once per family so each fits a handful of grids on its axis — single-family
    runs keep the original layout and slugs.
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
    pooled_files_all = sorted(input_dir.glob("E16_*_merged.csv"))

    figs: List[go.Figure] = []
    titles: List[str] = []
    slugs: List[str] = []

    def _slug(base: str, class_label: str) -> str:
        return f"{base}_{class_label}" if class_label else base

    def _title_suffix(class_label: str) -> str:
        return f" ({class_label})" if class_label else ""

    def _emit_cross_scenario(sub_df, sub_pooled_files, sub_ceil,
                             class_label: str) -> "pd.Index | None":
        """Append the cross-scenario E16 figures for one family
        subset. Returns the pivot_shed.index used for the heatmap so the
        per-scenario per-sector bar loop later can reuse the same order."""

        # ρ vs analytical shed — extended per-sector heatmap whose y-axis
        # lists (scenario, sector) pairs so every combination is visible in
        # one matrix. Falls back to the legacy total-only heatmap when there
        # is only one scenario (a 1×N matrix crashes Kaleido on axis scaling).
        pivot_shed = sub_df.pivot_table(index="scenario", columns="metric",
                                        values="rho_vs_shed")
        pivot_shed = pivot_shed.reindex(index=_scenario_order(pivot_shed.index))
        pivot_shed = pivot_shed.dropna(axis=1, how="all").dropna(axis=0, how="all")
        z = pivot_shed.values
        if z.size:
            if pivot_shed.shape[0] >= 2:
                heat = _e16_per_sector_heatmap(sub_df)
                if heat.data:
                    figs.append(heat)
                    titles.append(
                        f"ρ vs analytical shed — per scenario × sector"
                        f"{_title_suffix(class_label)}"
                    )
                    slugs.append(
                        _slug("e16_rho_vs_shed_heatmap_per_sector", class_label)
                    )
            else:
                # Single-scenario fallback: legacy total-only heatmap.
                text = np.where(np.isfinite(z),
                                np.array([f"{v:.2f}" for v in z.ravel()],
                                         dtype=object).reshape(z.shape), "")
                heat = go.Figure(go.Heatmap(
                    z=z, x=metric_label(list(pivot_shed.columns)),
                    y=[pretty_scenario(s) for s in pivot_shed.index],
                    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                    colorbar=dict(
                        title=dict(
                            text="ρ vs shed", side="right",
                            font=dict(size=_E16_FONT_SIZES["colorbar"]),
                        ),
                        tickfont=dict(size=_E16_FONT_SIZES["colorbar"]),
                        thickness=14, len=0.9,
                    ),
                    text=text, texttemplate="%{text}",
                    textfont=dict(size=_E16_FONT_SIZES["annotation"]),
                    hovertemplate="<b>%{y}</b><br>%{x}: ρ = %{z:.3f}<extra></extra>",
                ))
                heat.update_layout(**_e16_layout(
                    title=(
                        "Spearman ρ between metric and analytical single-removal shed"
                        + _title_suffix(class_label)
                    ),
                    xaxis=dict(title="Metric", tickangle=30), yaxis=dict(title=""),
                    height=max(320, 80 + 64 * len(pivot_shed.index)),
                    width=180 + 110 * len(pivot_shed.columns),
                ))
                figs.append(heat)
                titles.append(f"ρ vs analytical shed{_title_suffix(class_label)}")
                slugs.append(_slug("e16_rho_vs_shed_heatmap", class_label))

        # Aggregated (pooled across this class's scenarios) per-sector ρ bar.
        if sub_pooled_files:
            agg_bar = _e16_rho_per_sector_bar_aggregated(sub_pooled_files)
            if agg_bar.data:
                figs.append(agg_bar)
                titles.append(
                    f"ρ vs analytical shed — pooled across scenarios"
                    f"{_title_suffix(class_label)}"
                )
                slugs.append(
                    _slug("e16_rho_vs_shed_per_sector_pooled", class_label)
                )
            # Per-sector top-of-ranking accuracy (Kendall τ / NDCG@k /
            # precision@k), the deployment-relevant complement to the ρ bar.
            rank_bar = _e16_ranking_per_sector_aggregated(sub_pooled_files)
            if rank_bar.data:
                figs.append(rank_bar)
                titles.append(
                    f"ranking accuracy per sector — pooled across scenarios"
                    f"{_title_suffix(class_label)}"
                )
                slugs.append(
                    _slug("e16_ranking_per_sector_pooled", class_label)
                )

        # ρ vs shed vs ρ vs MC scatter — does shed-quality predict MC-quality?
        fig = go.Figure()
        metrics = list(sub_df["metric"].drop_duplicates())
        cmap = {m: QUAL[i % len(QUAL)] for i, m in enumerate(metrics)}
        for m in metrics:
            sub = sub_df[sub_df["metric"] == m]
            m_label = metric_label(m)
            fig.add_trace(go.Scatter(
                x=sub["rho_vs_shed"], y=sub["rho_vs_mc"],
                mode="markers", name=m_label,
                marker=dict(color=cmap[m], size=11,
                            line=dict(color="#222", width=0.5)),
                hovertext=[pretty_scenario(s) for s in sub["scenario"]],
                hovertemplate=("<b>%{hovertext}</b><br>" + m_label
                               + "<br>ρ vs shed = %{x:+.3f}"
                               "<br>ρ vs MC = %{y:+.3f}<extra></extra>"),
            ))
        fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1,
                      line=dict(color="#888", dash="dash", width=1.2))
        fig.add_hline(y=0, line=dict(color="#bbb", width=1, dash="dot"))
        fig.add_vline(x=0, line=dict(color="#bbb", width=1, dash="dot"))
        fig.update_layout(**_e16_layout(
            title=(
                "ρ vs analytical shed (x) vs ρ vs MC actual (y)"
                + _title_suffix(class_label)
            ),
            xaxis=dict(title="Spearman ρ vs shed", range=[-1.05, 1.05],
                       gridcolor="#e5e5e5"),
            yaxis=dict(title="Spearman ρ vs MC actual", range=[-1.05, 1.05],
                       gridcolor="#e5e5e5"),
            height=620, width=720,
            legend=dict(title="Metric"),
        ))
        figs.append(fig)
        titles.append(f"Metric ρ — shed vs MC{_title_suffix(class_label)}")
        slugs.append(_slug("e16_rho_shed_vs_mc", class_label))

        # Ceiling: how well does the analytical shed itself predict MC?
        if sub_ceil is not None and not sub_ceil.empty:
            c = sub_ceil.sort_values("rho_shed_vs_mc", ascending=True)
            err_hi = (c["ci_hi"] - c["rho_shed_vs_mc"]).clip(lower=0)
            err_lo = (c["rho_shed_vs_mc"] - c["ci_lo"]).clip(lower=0)
            fig2 = go.Figure(go.Bar(
                y=[pretty_scenario(s) for s in c["scenario"]],
                x=c["rho_shed_vs_mc"], orientation="h",
                error_x=dict(type="data", symmetric=False,
                             array=err_hi, arrayminus=err_lo,
                             thickness=1.2, width=4, color=pub_style.MUTED_COLOR),
                marker=pub_style.bar_marker("#00897b"),
                text=[f"ρ={v:+.2f}" for v in c["rho_shed_vs_mc"]],
                textposition="outside", cliponaxis=False,
                hovertemplate=("<b>%{y}</b><br>ρ = %{x:+.3f}"
                               "<br>n = %{customdata[0]}<extra></extra>"),
                customdata=np.c_[c["n"].values],
                showlegend=False,
            ))
            fig2.add_vline(x=0, line=dict(color="#444", width=1, dash="dot"))
            pub_style.apply_theme(
                fig2,
                title=(
                    "ρ between N-1 shed and MC"
                    + _title_suffix(class_label)
                ),
                height=pub_style.hbar_height(len(c)),
                width=pub_style.BAR_FIG_WIDTH, font_bump=1, no_legend=True,
            )
            fig2.update_xaxes(title="Spearman ρ", range=[-1.05, 1.10])
            fig2.update_yaxes(title="")
            figs.append(fig2)
            titles.append(f"Ceiling ρ{_title_suffix(class_label)}")
            slugs.append(_slug("e16_ceiling_rho", class_label))

        return pivot_shed.index if z.size else None

    # Cross-scenario figures: one set per scenario family (or one combined set
    # if only baselines are present, preserving legacy slug names).
    all_scenarios = list(df["scenario"].drop_duplicates())
    classes = split_scenarios_by_family(all_scenarios)
    if len(classes) <= 1:
        _emit_cross_scenario(df, pooled_files_all, ceil, class_label="")
        scenario_order_for_bars = (
            df.pivot_table(index="scenario", columns="metric", values="rho_vs_shed")
              .pipe(lambda p: p.reindex(index=_scenario_order(p.index)))
              .dropna(axis=1, how="all").dropna(axis=0, how="all").index
        )
    else:
        seen_orders: List = []
        for class_label, scens in classes:
            sub_df = df[df["scenario"].isin(scens)]
            sub_pooled = [
                f for f in pooled_files_all
                if f.stem.removeprefix("E16_").removesuffix("_merged") in set(scens)
            ]
            sub_ceil = (
                ceil[ceil["scenario"].isin(scens)] if not ceil.empty else ceil
            )
            sub_order = _emit_cross_scenario(sub_df, sub_pooled, sub_ceil,
                                             class_label=class_label)
            if sub_order is not None:
                seen_orders.append(sub_order)
        # Concatenate the per-class orders so the per-scenario bars below
        # still iterate every scenario in a stable, class-grouped order.
        if seen_orders:
            scenario_order_for_bars = pd.Index(np.concatenate([o.values for o in seen_orders]))
        else:
            scenario_order_for_bars = pd.Index(df["scenario"].drop_duplicates())

    # Per-scenario per-sector grouped bar — emitted for *every* scenario so
    # the per-sector ρ view is always available, regardless of family.
    # Order matches the cross-scenario figures above (baseline block first,
    # per-family blocks in mixed runs).
    for scenario in scenario_order_for_bars:
        bar = _e16_rho_per_sector_bar(df, scenario)
        if bar.data:
            figs.append(bar)
            titles.append(f"{pretty_scenario(scenario)} — ρ vs shed, per sector")
            slugs.append(f"e16_{scenario}_rho_vs_shed_per_sector")

    # ── Per-scenario raw scatter + top-K overlap (only if the
    #    refactored experiment emitted merged CSVs) ────────────────────────
    available_metrics = [m for m in df["metric"].drop_duplicates() if m]
    for mf in pooled_files_all:
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
                titles.append(f"{pretty_scenario(scenario)} — {m} vs total_shed")
                slugs.append(f"e16_{scenario}_scatter_{m}")
        ov_fig = _e16_top_overlap(merged, scenario, present_metrics)
        if ov_fig.data:
            figs.append(ov_fig)
            titles.append(f"{pretty_scenario(scenario)} — top-K overlap")
            slugs.append(f"e16_{scenario}_top_k_overlap")

    return _emit(figs, titles, slugs, output_dir / "E16_single_removal.html",
                 "Single-removal-shed validation")


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
