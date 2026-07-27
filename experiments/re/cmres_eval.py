"""CMRES evaluation experiments.

Each function below corresponds to one experiment in the CMRES evaluation
plan. They all consume the same inputs as the existing
``cp_cn_evaluation.evaluate`` pipeline (per-network dataframes plus a solved
monee Network) so they can be called as one extra block at the end of
``evaluate()`` without touching the existing pipeline.

Layout
------
``run_cmres_block(...)`` is the entry point — it collects per-scenario
artefacts and produces, for each experiment, one ``Path`` to the HTML output
or the dataframe (so callers can decide what to emit further).

Mapping to existing eval functions
----------------------------------
E1  cp_metric_vs_actual_impact       (per-network ρ + ranking battery)
E5  cp_only_metric_comparison        (per-CP-type heatmap)
E1/E5 pooled equivalents:
    pooled_metric_comparison, cp_only_pooled_metric_comparison

The functions below add the experiments NOT covered by the existing eval:

E2  experiment_e2_ablation        — per-factor ablation of predicted_score
E3  experiment_e3_density          — ρ vs CP density across scenarios
E4  experiment_e4_distribution     — distributed vs centralized comparison
E6  experiment_e6_sensitivity      — hyperparameter sweep (α, weights, etc.)
E7  experiment_e7_mc_validity      — RHW(n) curves, AV variance reduction
E8  experiment_e8_multilayer       — multilayer centralities → df_scores
E9  experiment_e9_percolation      — robustness curves + AUC per metric
E10 experiment_e10_structural      — coupling-strength scalars per scenario
E11 experiment_e11_null_models     — z-scores of structural quantities
E12 experiment_e12_community       — community + bridge_score
E13 experiment_e13_spectral        — λ₂, Kirchhoff per scenario / per layer
E15 experiment_e15_structural      — DDaR, source-sink BC, k-shortest, substitutability
E16 experiment_e16_single_removal  — single-removal-shed validation (slurm-fed CSV)

Shared logic (matched-df construction, statistical helpers, id matching) is
imported from ``eval_common`` so the same code paths run here and in
``cp_cn_evaluation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import scipy.stats as _stats  # used by E12 partial-correlation residual test

import cmres.evaluation.evaluation as eval

import cp_metric_complex as cmc
import eval_common as _ec
from cp_metric import (
    CPMetricConfig,
    mes_all_components_metric,
)


# ─────────────────────────────────────────────────────────────────────────────
# Container shuttling per-scenario artefacts between experiments
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScenarioArtefacts:
    """Bundle of per-scenario inputs that each experiment may consume.

    ``df_eval``: matched (predicted vs actual) dataframe from
        ``cp_metric_vs_actual_impact`` — the *unfiltered* one.
    ``monee_net``: the post-solve network used by cp_metric.
    ``mc_npz_path``: optional path to the scenario's mc_result.npz; used by
        E7 for the convergence trace and antithetic effectiveness.
    ``label``: scenario name (e.g. ``"simbench_lv_mid_backup"``).
    ``density``: optional CP density for E3 (caller passes through from
        ``ALL_GRIDS``); ``None`` if unknown.
    ``distribution``: ``"distributed" | "centralized" | None`` for E4.
    ``multilayer_G``: cached multilayer graph (built lazily by E8 and reused
        by E9, E11, E12, E13).
    ``impact_df_nt``: optional per-scenario MC impact slice (the input
        ``build_matched_df`` consumed). E16 uses it for the full-surface
        ceiling — shed rows without a metric score (kind ``"child"``) can
        only be joined to MC actuals through it, not through df_eval.
    """

    label: str
    df_eval: pd.DataFrame
    monee_net: object
    mc_npz_path: Optional[Path] = None
    density: Optional[float] = None
    distribution: Optional[str] = None
    multilayer_G: Optional[nx.Graph] = field(default=None, repr=False)
    impact_df_nt: Optional[pd.DataFrame] = field(default=None, repr=False)

    def get_multilayer_graph(self) -> nx.Graph:
        if self.multilayer_G is None:
            self.multilayer_G = cmc.build_multilayer_graph(self.monee_net)
        return self.multilayer_G


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (re-exported from eval_common; aliased to keep the
# underscore-prefixed call-site convention used throughout this module)
# ─────────────────────────────────────────────────────────────────────────────


_spearman_with_ci = _ec.spearman_with_ci
_wilcoxon = _ec.wilcoxon
_holm_correct = _ec.holm_correct
_extract_orig_id = _ec.extract_orig_id


# ─────────────────────────────────────────────────────────────────────────────
# E2 — Per-factor ablation of predicted_score
# ─────────────────────────────────────────────────────────────────────────────


_ABLATION_VARIANTS: Dict[str, dict] = {
    "full":             {},
    "no_throughput":    {"ABLATE_THROUGHPUT": True},
    "no_stress":        {"ABLATE_STRESS": True},
    "no_topo":          {"ABLATE_TOPO": True},
    "no_adequacy":      {"ABLATE_ADEQUACY": True},
}


def experiment_e2_ablation(
    monee_net,
    impact_df_nt,
    network_type: str,
    output_dir: Path,
    eps: float = _ec.MC_FAILED_EPS,
    df_eval_full: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run mes_all_components_metric once per ablation variant and report
    Spearman ρ vs MC actual_total. Effect of removing factor X is
    ``ρ(full) − ρ(no_X)``.

    ``df_eval_full`` is the matched dataframe from
    ``cp_metric_vs_actual_impact`` for this scenario. When provided, the
    ``"full"`` variant is read from it instead of re-running
    ``mes_all_components_metric`` — this saves the expensive PTDF computation
    that has already been done.

    Returns a tidy DataFrame ``(variant, ρ, ρ_lo, ρ_hi, p, delta_vs_full,
    n)`` ready for plotting.
    """

    def _rho_for_matched(df_matched: pd.DataFrame):
        df = df_matched[df_matched["actual_total"].astype(float) > eps]
        if len(df) < 3:
            return (float("nan"), float("nan"), float("nan"), float("nan"), len(df))
        rho, p, lo, hi = _spearman_with_ci(df["predicted_score"], df["actual_total"])
        return rho, p, lo, hi, len(df)

    rows = []
    rho_full: Optional[float] = None
    for variant, kwargs in _ABLATION_VARIANTS.items():
        if variant == "full" and df_eval_full is not None and len(df_eval_full) > 0:
            # Reuse the already-computed matched df from the per-scenario run.
            df_matched = df_eval_full
        else:
            cfg = CPMetricConfig(**kwargs)
            df_scores, _ = mes_all_components_metric(monee_net, cfg=cfg)
            df_matched = _ec.build_matched_df(df_scores, impact_df_nt)
        rho, p, lo, hi, n = _rho_for_matched(df_matched)
        if variant == "full":
            rho_full = rho
        rows.append({
            "variant": variant,
            "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
            "n": n,
            "delta_vs_full": (
                rho - rho_full
                if (rho_full is not None and np.isfinite(rho) and np.isfinite(rho_full))
                else float("nan")
            ),
        })
    out_df = pd.DataFrame(rows)
    output_dir.mkdir(exist_ok=True, parents=True)
    out_df.to_csv(output_dir / f"E2_ablation_{network_type}.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E3 — ρ vs CP density across scenarios
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e3_density(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """For each metric × scenario, compute ρ vs MC actual_total restricted
    to MC-sampled CPs, then emit one figure with ρ vs density (one line per
    metric). If multiple distributions are present at the same density, the
    figure colours them differently so the centralized vs distributed gap
    is visible at d=0.5.
    """
    if metrics is None:
        metrics = [
            "predicted_score",   # full (with ablations off)
            "predicted_stress",  # PTDF stress only
            "topo_factor",       # 1 + α·BC alone
            "input_adequacy",
            "local_score",
            "self_score",
            "katz_score",
            "vitality_score",
        ]
    rows: List[dict] = []
    for art in artefacts:
        if art.density is None:
            continue
        df = art.df_eval.copy()
        df = df[df["actual_total"].astype(float) > 0]
        if len(df) < 3:
            continue
        for m in metrics:
            if m not in df.columns:
                continue
            rho, p, lo, hi = _spearman_with_ci(df[m], df["actual_total"])
            rows.append({
                "scenario": art.label,
                "density": art.density,
                "distribution": art.distribution or "distributed",
                "metric": m,
                "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                "n": len(df),
            })
    out_df = pd.DataFrame(rows)
    output_dir.mkdir(exist_ok=True, parents=True)
    out_df.to_csv(output_dir / "E3_rho_vs_density.csv", index=False)

    # Plot: ρ vs density per metric, coloured by metric, faceted on
    # distribution. Bootstrap CI as error bars.
    if not out_df.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        cmap = {m: c for m, c in zip(metrics, eval.PALETTE_QUAL)}
        for m in metrics:
            sub = out_df[out_df["metric"] == m].sort_values("density")
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["density"], y=sub["rho"],
                error_y=dict(
                    type="data", symmetric=False,
                    array=(sub["ci_hi"] - sub["rho"]).clip(lower=0),
                    arrayminus=(sub["rho"] - sub["ci_lo"]).clip(lower=0),
                ),
                mode="lines+markers", name=m, marker=dict(color=cmap.get(m), size=8),
                hovertext=sub["scenario"],
            ))
        fig.update_layout(
            template=eval.CMRES_TEMPLATE,
            height=520, width=900,
            xaxis=dict(title="CP density"),
            yaxis=dict(title="Spearman ρ vs MC actual_total", range=[-1.05, 1.05]),
            legend=dict(title="Metric"),
        )
        fig.update_layout(**eval.LEGEND_RIGHT)
        eval.write_all_in_one(
            [fig], "Figure", Path("."),
            str(output_dir / "E3_rho_vs_density.html"),
            titles=["E3: Spearman ρ vs CP density per metric"],
            slugs=["e3_rho_vs_density"],
        )
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E4 — Distributed vs centralized at matched density
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e4_distribution(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Pair scenarios at matched density across distribution variants.
    Compute (a) impact concentration (Gini, top-1, top-5 share, entropy)
    per scenario, (b) per-metric ρ comparison.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # (a) Concentration of impact per scenario.
    conc_rows: List[dict] = []
    for art in artefacts:
        df = art.df_eval
        if df.empty:
            continue
        actual = df["actual_total"].astype(float).abs().values
        c = cmc.criticality_concentration(actual)
        conc_rows.append({
            "scenario": art.label,
            "density": art.density,
            "distribution": art.distribution or "distributed",
            **{f"actual_{k}": v for k, v in c.items()},
        })
    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(out_dir / "E4_impact_concentration.csv", index=False)

    # (b) Per-metric ρ table, paired by density.
    rho_rows: List[dict] = []
    metrics = [c for c in [
        "predicted_score", "predicted_stress", "topo_factor",
        "input_adequacy", "local_score", "self_score",
    ] if all(c in a.df_eval.columns for a in artefacts)]
    for art in artefacts:
        df = art.df_eval[art.df_eval["actual_total"] > 0]
        if len(df) < 3:
            continue
        for m in metrics:
            rho, p, lo, hi = _spearman_with_ci(df[m], df["actual_total"])
            rho_rows.append({
                "scenario": art.label,
                "density": art.density,
                "distribution": art.distribution or "distributed",
                "metric": m, "rho": rho, "n": len(df),
            })
    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(out_dir / "E4_rho_by_distribution.csv", index=False)
    return {"concentration": conc_df, "rho": rho_df}


# ─────────────────────────────────────────────────────────────────────────────
# E6 — Hyperparameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e6_sensitivity(
    monee_net,
    impact_df_nt,
    network_type: str,
    output_dir: Path,
    sweeps: Optional[Dict[str, List]] = None,
    base_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    """Sweep hyperparameters and record ρ vs MC actual_total. Default sweep
    covers TOPO_ALPHA, the carrier weights, HEAT_REMOTENESS_ALPHA,
    HEAT_DELTA_T_K, SUSCEPTANCE_B_RELATIVE_CAP. Each parameter is varied
    one-at-a-time around the default config; the result is a tornado-style
    table where each row is one (parameter, value, ρ).
    """
    base_kwargs = base_kwargs or {}
    if sweeps is None:
        sweeps = {
            "TOPO_ALPHA":              [0.0, 0.5, 1.0, 2.0, 5.0],
            "W_POWER":                 [0.0, 0.5, 1.0, 2.0],
            "W_GAS":                   [0.0, 0.5, 1.0, 2.0],
            "W_HEAT":                  [0.0, 0.5, 1.0, 2.0],
            "HEAT_REMOTENESS_ALPHA":   [0.0, 0.5, 1.0, 2.0, 5.0],
            "HEAT_DELTA_T_K":          [10.0, 20.0, 30.0, 50.0],
            "SUSCEPTANCE_B_RELATIVE_CAP": [10.0, 50.0, 100.0, 500.0],
        }

    def _rho_for(cfg: CPMetricConfig) -> float:
        df_scores, _ = mes_all_components_metric(monee_net, cfg=cfg)
        df_matched = _ec.build_matched_df(df_scores, impact_df_nt)
        df_matched = df_matched[df_matched["actual_total"].astype(float) > _ec.MC_FAILED_EPS]
        if len(df_matched) < 3:
            return float("nan")
        rho, _, _, _ = _spearman_with_ci(df_matched["predicted_score"], df_matched["actual_total"])
        return rho

    rho_baseline = _rho_for(CPMetricConfig(**base_kwargs))
    rows: List[dict] = []
    rows.append({"param": "(baseline)", "value": "default", "rho": rho_baseline,
                 "delta_vs_baseline": 0.0})
    for param, values in sweeps.items():
        for v in values:
            kw = dict(base_kwargs)
            kw[param] = v
            rho = _rho_for(CPMetricConfig(**kw))
            rows.append({
                "param": param, "value": v, "rho": rho,
                "delta_vs_baseline": (rho - rho_baseline)
                if (np.isfinite(rho) and np.isfinite(rho_baseline)) else float("nan"),
            })
    out_df = pd.DataFrame(rows)
    output_dir.mkdir(exist_ok=True, parents=True)
    out_df.to_csv(output_dir / f"E6_sensitivity_{network_type}.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E7 — MC validity diagnostics
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e7_mc_validity(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
) -> pd.DataFrame:
    """Read the convergence trace stored by MCEngine in mc_result.npz and
    emit (a) RHW(n) per carrier per scenario, (b) the antithetic
    variance-reduction factor estimated from per_run pairs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    summary: List[dict] = []
    rhw_rows: List[dict] = []
    for art in artefacts:
        if art.mc_npz_path is None or not art.mc_npz_path.exists():
            continue
        with np.load(art.mc_npz_path) as data:
            keys = set(data.files)
            conv = data["convergence"] if "convergence" in keys else np.empty((0, 8))
            per_run = data["per_run"] if "per_run" in keys else np.empty((0, 3))
            n_runs_arr = data["n_runs"] if "n_runs" in keys else np.array([0])
            n_runs = int(np.asarray(n_runs_arr).reshape(-1)[0])
        # RHW vs n.
        if conv.size:
            for row in conv:
                n_now = int(row[0])
                mean_now = row[1:4]
                rhw_now = row[4:7]
                ess_now = float(row[7])
                for k, name in enumerate(["power", "heat", "gas"]):
                    rhw_rows.append({
                        "scenario": art.label, "n": n_now, "carrier": name,
                        "mean": float(mean_now[k]), "rhw": float(rhw_now[k]),
                        "ess": ess_now,
                    })
        # Antithetic variance reduction: assumes paired (Y, Y_av) layout.
        # If per_run is shaped (N, 3) with even N and antithetic enabled, then
        # rows 2k and 2k+1 are a pair.
        var_y = float("nan")
        var_pair = float("nan")
        var_reduction = float("nan")
        if per_run.size and per_run.shape[0] >= 4 and per_run.shape[0] % 2 == 0:
            # per-carrier independent variance estimate
            Y = per_run[0::2]   # (N/2, 3)
            Y_av = per_run[1::2]  # (N/2, 3)
            var_y_vec = np.var(per_run, axis=0, ddof=1)        # naive Var(Y)
            var_pair_vec = np.var((Y + Y_av) / 2.0, axis=0, ddof=1)  # Var of pair-mean
            var_y = float(np.nanmean(var_y_vec))
            var_pair = float(np.nanmean(var_pair_vec))
            if var_pair > 0:
                var_reduction = float(var_y / (2 * var_pair))
        summary.append({
            "scenario": art.label,
            "n_runs": n_runs,
            "var_naive": var_y,
            "var_antithetic_pair_mean": var_pair,
            "AV_reduction_factor": var_reduction,
        })
    summary_df = pd.DataFrame(summary)
    rhw_df = pd.DataFrame(rhw_rows)
    summary_df.to_csv(out_dir / "E7_mc_summary.csv", index=False)
    rhw_df.to_csv(out_dir / "E7_rhw_curves.csv", index=False)
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# E8 — Multilayer centralities → df_scores augmentation
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e8_multilayer(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
) -> pd.DataFrame:
    """Compute multilayer centralities per scenario and join them onto
    each artefact's ``df_eval`` so downstream eval can compare ρ for
    multilayer vs single-layer centralities.

    The join is by the ``carrier`` of the CP's "input" node mapped onto
    the multilayer (carrier, orig_id) tuple. Non-CP rows are joined by
    matching (carrier, orig_id) for their primary carrier endpoint.

    Returns a tidy DataFrame ``(scenario, metric, rho, n)`` for the new
    multilayer metrics so the evaluation can include them in the
    main-results table.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    rows: List[dict] = []
    for art in artefacts:
        G = art.get_multilayer_graph()
        cents_df = cmc.multilayer_centralities(G)
        if cents_df.empty:
            continue
        # Aggregate (carrier, orig_id) → max over its participating layers
        # so each component gets a single ml_bc / participation / ml_degree
        # row regardless of how many layers it touches.
        agg = (
            cents_df.groupby("orig_id", as_index=False)
            .agg({"ml_bc": "max", "ml_degree": "max",
                  "activity": "max", "participation": "max",
                  "inter_layer_degree": "max"})
        )
        # Join onto df_eval. df_eval has cp_id strings — extract numeric id
        # for compounds and "from" id for branch CPs. Best-effort join.
        df = art.df_eval.copy()
        df["_join_id"] = df["cp_id"].astype(str).map(_extract_orig_id)
        merged = df.merge(agg, left_on="_join_id", right_on="orig_id", how="left")
        # Update artefact in place so later experiments (E9 ranking) see it.
        art.df_eval = merged

        valid = merged[merged["actual_total"] > 0]
        for m in ["ml_bc", "ml_degree", "activity", "participation", "inter_layer_degree"]:
            if m not in valid.columns or valid[m].notna().sum() < 3:
                continue
            sub = valid[valid[m].notna()]
            rho, p, lo, hi = _spearman_with_ci(sub[m], sub["actual_total"])
            rows.append({
                "scenario": art.label, "metric": m, "rho": rho,
                "p": p, "ci_lo": lo, "ci_hi": hi, "n": int(len(sub)),
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E8_multilayer_rho.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E9 — Percolation / robustness curves per metric
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e9_percolation(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
    metrics: Optional[List[str]] = None,
    n_random: int = 20,
) -> pd.DataFrame:
    """For each scenario and each metric, compute the targeted-attack AUC
    on the multilayer graph and compare to a random-removal baseline.

    Lower AUC = more effective attack ordering. The z-score of (random
    AUC − targeted AUC) / std is the effect size.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if metrics is None:
        metrics = ["predicted_score", "predicted_stress", "topo_factor",
                   "ml_bc", "input_adequacy", "katz_score", "vitality_score"]

    rng = np.random.default_rng(0)
    rows: List[dict] = []
    for art in artefacts:
        G = art.get_multilayer_graph()
        if G.number_of_nodes() == 0:
            continue
        # Map cp_id → multilayer node tuple. We use a *primary* carrier:
        # for compound CPs, prefer the connected_to entry; for branch CPs,
        # use the from-node side.
        df = art.df_eval.copy()
        # No leading underscore: itertuples() renames underscore-leading
        # columns to positional fields (_orig_id → _N), so r._orig_id
        # raises AttributeError on the first row.
        df["orig_id_ml"] = df["cp_id"].astype(str).map(_extract_orig_id)
        # Build a per-metric ranking dict {ml_node_id: score}.
        for m in metrics:
            if m not in df.columns:
                continue
            scoring = df[df[m].notna()].copy()
            ranking: Dict = {}
            for r in scoring.itertuples(index=False):
                if r.orig_id_ml is None:
                    continue
                # Try every carrier — whichever exists in G.
                for carrier in ("power", "gas", "heat"):
                    nid = (carrier, int(r.orig_id_ml))
                    if G.has_node(nid):
                        prev = ranking.get(nid)
                        if prev is None or float(getattr(r, m)) > prev:
                            ranking[nid] = float(getattr(r, m))
                        break
            if not ranking:
                continue
            res = cmc.percolation_for_metric(G, ranking, rng=rng, n_random=n_random)
            rows.append({
                "scenario": art.label, "metric": m,
                "AUC_metric": res["AUC_metric"],
                "AUC_random_mean": res["AUC_random_mean"],
                "AUC_random_std": res["AUC_random_std"],
                "AUC_z": res["AUC_z"],
                "n_ranked": len(ranking),
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E9_percolation_auc.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E10 — Coupling-strength characterisation
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e10_structural(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
) -> pd.DataFrame:
    """Per-scenario structural summary: σ_c per layer pair, total σ_c,
    CP-localization Gini. The scenario-level MC ENS is included so a
    follow-up regression can test the "structural mediator" hypothesis
    (RQ10).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    rows: List[dict] = []
    for art in artefacts:
        G = art.get_multilayer_graph()
        cs = cmc.coupling_strength(G)
        # Gather scenario-level MC ENS as the sum of |actual_total| over CP
        # rows (proxy for total expected impact attributable to CPs in MC).
        df = art.df_eval
        ens_proxy = float(df["actual_total"].astype(float).abs().sum()) if "actual_total" in df else float("nan")
        rows.append({
            "scenario": art.label,
            "density": art.density,
            "distribution": art.distribution or "distributed",
            "n_inter_edges": cs["n_inter_edges"],
            "sigma_c_total": cs["sigma_c_total"],
            "cp_localization_gini": cs["cp_localization_gini"],
            "mc_ens_proxy": ens_proxy,
            **{f"sigma_c[{k}]": v for k, v in cs["sigma_c"].items()},
            **{f"intra_edges_{k}": v for k, v in cs["n_intra_edges_per_layer"].items()},
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E10_coupling_strength.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E11 — Null-model z-scores
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e11_null_models(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
    n_nulls: int = 100,
    kinds: Optional[List[str]] = None,
) -> pd.DataFrame:
    """For each scenario, compare three structural quantities of the
    observed multilayer graph against degree-preserving and ER null
    ensembles:
      1. Average BC
      2. Algebraic connectivity λ₂ of the supra-Laplacian
      3. Targeted-attack AUC under degree ordering
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    kinds = kinds or ["config", "er"]

    def _avg_bc(G: nx.Graph) -> float:
        if G.number_of_nodes() == 0:
            return float("nan")
        bc = nx.betweenness_centrality(G, normalized=True)
        return float(np.mean(list(bc.values()))) if bc else float("nan")

    def _supra_lambda2(G: nx.Graph) -> float:
        return cmc._algebraic_connectivity(G)

    def _attack_auc_by_degree(G: nx.Graph) -> float:
        if G.number_of_nodes() == 0:
            return float("nan")
        order = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)
        return cmc.attack_auc(cmc.percolation_curve(G, order))

    stats: Dict[str, Callable] = {
        "avg_bc": _avg_bc,
        "lambda2_supra": _supra_lambda2,
        "AUC_degree_attack": _attack_auc_by_degree,
    }

    rows: List[dict] = []
    for art in artefacts:
        G = art.get_multilayer_graph()
        for kind in kinds:
            for name, fn in stats.items():
                try:
                    z = cmc.null_model_z_scores(G, fn, n=n_nulls, kind=kind, seed=42)
                except Exception as e:
                    print(
                        f"[E11:{art.label}] null_model_z_scores({name}, {kind}) "
                        f"failed: {type(e).__name__}: {e}"
                    )
                    z = {"observed": float("nan"), "null_mean": float("nan"),
                         "null_std": float("nan"), "z": float("nan"),
                         "p_one_sided_upper": float("nan")}
                rows.append({
                    "scenario": art.label, "null_kind": kind,
                    "statistic": name, **z,
                })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E11_null_z.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E12 — Community structure & cross-layer bridges
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e12_community(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
) -> pd.DataFrame:
    """Compute community partition + bridge_score per multilayer node,
    then test whether bridge_score adds explanatory power for MC
    actual_total beyond BC alone.

    Reports per-scenario partial correlations:
      - ρ(bridge_score, actual_total)
      - ρ(bridge_score, actual_total | BC) via residualisation
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    rows: List[dict] = []
    for art in artefacts:
        G = art.get_multilayer_graph()
        if G.number_of_nodes() == 0:
            continue
        partition = cmc.community_partition(G, seed=42)
        bridge = cmc.bridge_score(G, partition)
        df = art.df_eval.copy()
        # No leading underscore — see the E9 comment (itertuples renames).
        df["orig_id_ml"] = df["cp_id"].astype(str).map(_extract_orig_id)
        # Map to multilayer node and look up bridge / community.
        comp_bridge: List[float] = []
        comp_actual: List[float] = []
        comp_bc: List[float] = []
        for r in df.itertuples(index=False):
            # ``orig_id_ml`` is the integer node id parsed from cp_id; a falsy
            # check would drop the legitimate id 0. Skip only when parsing
            # actually failed.
            if r.orig_id_ml is None:
                continue
            for carrier in ("power", "gas", "heat"):
                nid = (carrier, int(r.orig_id_ml))
                if nid in bridge:
                    comp_bridge.append(float(bridge[nid]))
                    comp_actual.append(float(r.actual_total))
                    comp_bc.append(float(getattr(r, "topo_bc", float("nan"))))
                    break
        if len(comp_bridge) < 5:
            continue
        rho_raw, p_raw, _, _ = _spearman_with_ci(comp_bridge, comp_actual)
        # Residualise against BC: regress actual on BC, take residuals,
        # correlate residuals with bridge.
        bridge_arr = np.asarray(comp_bridge)
        actual_arr = np.asarray(comp_actual)
        bc_arr = np.asarray(comp_bc)
        mask = np.isfinite(bridge_arr) & np.isfinite(actual_arr) & np.isfinite(bc_arr)
        bridge_arr, actual_arr, bc_arr = bridge_arr[mask], actual_arr[mask], bc_arr[mask]
        rho_partial, p_partial = float("nan"), float("nan")
        if bridge_arr.size >= 5 and np.std(bc_arr) > 0:
            slope = np.cov(actual_arr, bc_arr, ddof=1)[0, 1] / np.var(bc_arr, ddof=1)
            intercept = np.mean(actual_arr) - slope * np.mean(bc_arr)
            actual_resid = actual_arr - (slope * bc_arr + intercept)
            try:
                res = _stats.spearmanr(bridge_arr, actual_resid)
                rho_partial, p_partial = float(res.statistic), float(res.pvalue)
            except Exception:
                pass
        rows.append({
            "scenario": art.label,
            "n": int(bridge_arr.size),
            "n_communities": int(len(set(partition.values()))) if partition else 0,
            "rho_bridge_vs_actual": rho_raw,
            "p_bridge_vs_actual": p_raw,
            "rho_bridge_vs_actual_given_bc": rho_partial,
            "p_bridge_vs_actual_given_bc": p_partial,
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E12_bridges.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E13 — Spectral robustness
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e13_spectral(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
) -> pd.DataFrame:
    """Per-scenario λ₂ per layer + supra, plus Kirchhoff index of the
    largest connected component of the supra-graph.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    rows: List[dict] = []
    for art in artefacts:
        G = art.get_multilayer_graph()
        spectral = cmc.spectral_robustness(G)
        # Kirchhoff is undefined if graph is disconnected. Take the largest CC.
        if G.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G), key=len)
            sub = G.subgraph(largest_cc).copy()
            kirchhoff = cmc.kirchhoff_index(sub)
        else:
            kirchhoff = float("nan")
        rows.append({
            "scenario": art.label,
            "density": art.density,
            "distribution": art.distribution or "distributed",
            "kirchhoff_lcc": kirchhoff,
            **{k: float(v) for k, v in spectral.items()},
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E13_spectral.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E15 — Structural metrics (DDaR, source-sink BC, k-shortest, substitutability)
# ─────────────────────────────────────────────────────────────────────────────


def experiment_e15_structural(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
    enable_min_cut: bool = False,
    k_shortest: int = 3,
) -> pd.DataFrame:
    """Attach structural metrics to each scenario's df_eval and report
    Spearman ρ vs MC actual_total per metric per scenario.

    The structural metrics (ddar_mw_total, ss_bc_total,
    kshortest_redundancy_total, substitutability, optionally
    min_cut_criticality_total) are joined onto ``art.df_eval`` so that
    downstream functions (E2 ablation, E9 percolation) can also see them.

    ``enable_min_cut=True`` adds the expensive min-cut criticality —
    feasible on simbench LV but takes a few minutes per scenario.
    """
    import cp_metric_structural as cms

    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    rows: List[dict] = []
    metrics = [
        "ddar_mw_total",
        "ss_bc_total",
        "kshortest_redundancy_total",
        "substitutability",
        "rated_capacity_mw",
    ]
    if enable_min_cut:
        metrics.append("min_cut_criticality_total")

    for art in artefacts:
        try:
            augmented = cms.attach_structural_metrics(
                art.df_eval, art.monee_net,
                enable_min_cut=enable_min_cut, k_shortest=k_shortest,
            )
        except Exception as e:
            print(f"[E15:{art.label}] attach_structural_metrics failed: {type(e).__name__}: {e}")
            continue
        # Update the artefact in place so E2/E9/E12 see the new columns.
        art.df_eval = augmented
        valid = augmented[augmented["actual_total"].astype(float) > _ec.MC_FAILED_EPS]
        for m in metrics:
            if m not in valid.columns:
                continue
            sub = valid[valid[m].notna()]
            if len(sub) < 3:
                continue
            rho, p, lo, hi = _spearman_with_ci(sub[m], sub["actual_total"])
            rows.append({
                "scenario": art.label,
                "metric": m,
                "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                "n": int(len(sub)),
            })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E15_structural_rho.csv", index=False)
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# E16 — Single-removal-shed validation (analytical ground truth)
# ─────────────────────────────────────────────────────────────────────────────


# Compound CP types whose row in df_eval carries a bare integer id
# (because ``mes_cp_metric`` stores ``cp_id=cp.id`` for compounds and
# ``build_matched_df`` keeps it as ``str(int)``).  The shed CSV uses
# ``"compound:{int}"`` (``_enumerate_targets`` in single_removal_shed.py),
# so without prefixing here every compound CP silently drops out of the
# join.
_E16_COMPOUND_TYPES = ("CHP", "CHPHG", "PowerToHeat")


def _e16_join_id(cp_id, cp_type) -> str:
    """Normalise a df_eval cp_id to the shed CSV's cp_id format."""
    cp_id_str = str(cp_id)
    if cp_type in _E16_COMPOUND_TYPES and not cp_id_str.startswith("compound:"):
        # ``str(int)`` for compound rows → "5"; shed uses "compound:5".
        return f"compound:{cp_id_str}"
    return cp_id_str


#: Shed columns carried through the E16 join. The ``_conn`` twins hold the
#: curtailment of loads that stay *connected* after the removal — the part
#: dispatch (and therefore CP capacity) can influence, as opposed to the
#: topologically islanded nameplate that dominates the plain columns. They
#: are absent from legacy shed CSVs, so every consumer must tolerate that.
_E16_SHED_COLS = ["total_shed", "power_shed", "heat_shed", "gas_shed"]
E16_CONN_SUFFIX = "_conn"
_E16_SHED_COLS_CONN = [c + E16_CONN_SUFFIX for c in _E16_SHED_COLS]


def _load_shed_csv(shed_csv: Path) -> pd.DataFrame:
    """Load a single-removal shed CSV, baseline-subtracted.

    Per-carrier sheds are baseline-subtracted: every solved shed includes
    the no-fault baseline (nonzero on several grids — up to 0.055 MW power
    in the loadbearing family, above SHED_EPS), so without subtraction a
    component that never touches a carrier still "affects" it at ≈baseline
    and every branch enters that carrier's slice tied at the offset. Min-
    shed is monotone (removal can only increase shed), so negatives after
    subtraction are solver noise — clipped to 0.
    """
    shed = pd.read_csv(shed_csv)
    shed_cols = _E16_SHED_COLS + _E16_SHED_COLS_CONN
    base_rows = shed[shed["cp_id"] == "_baseline_"]
    if not base_rows.empty:
        for col in shed_cols:
            if col not in shed.columns:
                continue
            base_val = float(base_rows.iloc[0][col] or 0.0)
            if base_val > 0:
                shed[col] = (shed[col] - base_val).clip(lower=0)
    shed = shed[shed["cp_id"] != "_baseline_"].copy()
    shed["cp_id"] = shed["cp_id"].astype(str)
    return shed


def _shed_impact_id(cp_id, kind, impact_ids, branch_lookup) -> Optional[str]:
    """Map a shed-CSV row to the metrics/impact id space (``child:3``,
    ``compound:5``, ``branch:(5, 134, 0)``). Kind-based, unlike
    ``eval_common.match_impact_id`` which keys off the metric-side cp_type;
    shed rows carry ``kind`` instead. Returns ``None`` when unmatched."""
    s = str(cp_id)
    if kind in ("child", "compound"):
        return s if s in impact_ids else None
    if kind == "branch":
        candidate = f"branch:{s}"
        if candidate in impact_ids:
            return candidate
        parts = [p.strip() for p in s.strip("()").split(",")]
        return branch_lookup.get((parts[0], parts[1])) if len(parts) >= 2 else None
    if kind == "branch_cp":
        try:
            from_id, to_id = s.split("→")
        except ValueError:
            return None
        return branch_lookup.get((from_id.strip(), to_id.strip()))
    return None


def _e16_merge_one(art: "ScenarioArtefacts", shed_csv: Path) -> pd.DataFrame:
    """Return the per-scenario merged dataframe (or empty if nothing joins).

    Inner join on df_eval: shed rows without a metric score (kind
    ``"child"``) drop out here by design — they are consumed by the
    full-surface ceiling instead (see the caller).
    """
    shed = _load_shed_csv(shed_csv)
    df = art.df_eval.copy()
    df["_join_cp_id"] = df.apply(
        lambda r: _e16_join_id(r["cp_id"], r.get("cp_type", "")), axis=1
    )
    keep = ["cp_id", "kind"] + [
        c for c in _E16_SHED_COLS + _E16_SHED_COLS_CONN if c in shed.columns
    ]
    return df.merge(
        shed[keep], left_on="_join_cp_id", right_on="cp_id", how="inner",
        suffixes=("", "_shed"),
    )


def experiment_e16_single_removal_validation(
    artefacts: List[ScenarioArtefacts],
    output_dir: Path,
    shed_dir: Optional[Path] = None,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare each metric to the deterministic single-removal load shed.

    For every scenario whose
    ``shed_dir/single_removal_shed_<scenario>.csv`` exists, join the shed
    table onto the matched df_eval and compute Spearman ρ between every
    candidate metric and the analytical shed (the structural ceiling) AND
    between the analytical shed and the MC actual_total (an upper bound
    on what any topology metric could achieve on this grid).

    The expected layout is the one written by ``single_removal_shed.py``:

        shed_dir/single_removal_shed_<scenario>.csv

    Each row has ``cp_id, kind, total_shed`` (and per-carrier shed). The
    join normalises the metric-side cp_id for compound CPs (``"5"`` →
    ``"compound:5"``) so compounds aren't silently dropped.

    Outputs (in ``output_dir``):
      - ``E16_metric_vs_shed.csv``           : ρ per (scenario, metric)
      - ``E16_shed_vs_mc_ceiling.csv``       : ρ shed vs MC actual_total,
        metric-scored components only (branches + CPs — the rows the
        ranking analysis sees)
      - ``E16_shed_vs_mc_ceiling_full.csv``  : same ρ over the FULL swept
        surface incl. score-less child targets (generation outages), joined
        straight onto the MC actuals; needs ``art.impact_df_nt``
      - ``E16_<scenario>_merged.csv``        : raw joined dataframe per scenario

    Plotting is handled by ``cmres_eval_plots.plot_e16_single_removal``,
    which consumes these CSVs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if shed_dir is None:
        shed_dir = out_dir.parent / "single_removal_shed"
    # The canonical 10-metric set, kept in sync with the cp_cn_evaluation
    # figures. The carrier-matched per-sector predictors (predicted_power /
    # _heat / _gas) are deliberately excluded: each is defined only on its own
    # carrier's slice, so on the per-sector figures it is one bar plus four
    # blanks and cannot be compared row-wise against a general predictor.
    # Override with the ``metrics`` argument when probing a one-off subset.
    metrics = metrics or list(_ec.CORE_METRIC_COLS)

    rows: List[dict] = []
    ceiling_rows: List[dict] = []
    full_ceiling_rows: List[dict] = []
    cp_pool_frames: List[pd.DataFrame] = []
    for art in artefacts:
        shed_csv = Path(shed_dir) / f"single_removal_shed_{art.label}.csv"
        if not shed_csv.exists():
            print(f"[E16:{art.label}] no shed CSV at {shed_csv}; skipping")
            continue
        merged = _e16_merge_one(art, shed_csv)
        if merged.empty:
            print(f"[E16:{art.label}] empty join; skipping")
            continue

        merged.to_csv(out_dir / f"E16_{art.label}_merged.csv", index=False)

        # Ceiling: how well does the analytical ground truth itself match MC?
        if "actual_total" in merged.columns:
            cmask = merged["actual_total"].astype(float) > _ec.MC_FAILED_EPS
            if cmask.sum() >= 3:
                rho, p, lo, hi = _spearman_with_ci(
                    merged.loc[cmask, "total_shed"],
                    merged.loc[cmask, "actual_total"],
                )
                ceil_row = {
                    "scenario": art.label,
                    "rho_shed_vs_mc": rho, "p_shed_vs_mc": p,
                    "ci_lo": lo, "ci_hi": hi,
                    "n": int(cmask.sum()),
                }
                # Connected-load twin: the same ceiling measured against the
                # recoverable shed only. The plain number is dominated by the
                # islanded nameplate, which is CP-invariant and therefore
                # inflates the agreement for reasons unrelated to the metric.
                conn_col = "total_shed" + E16_CONN_SUFFIX
                if conn_col in merged.columns:
                    cmask_c = cmask & merged[conn_col].notna()
                    if (cmask_c.sum() >= 3
                            and merged.loc[cmask_c, conn_col].nunique() >= 2):
                        rho_c, p_c, lo_c, hi_c = _spearman_with_ci(
                            merged.loc[cmask_c, conn_col],
                            merged.loc[cmask_c, "actual_total"],
                        )
                        ceil_row.update({
                            "rho_shed_conn_vs_mc": rho_c,
                            "p_shed_conn_vs_mc": p_c,
                            "ci_lo_conn": lo_c, "ci_hi_conn": hi_c,
                            "n_conn": int(cmask_c.sum()),
                        })
                ceiling_rows.append(ceil_row)

        # Full-surface ceiling: the merge above is an inner join on df_eval,
        # so score-less child targets (generation outages) never reach it.
        # Join the shed table straight onto the MC actuals instead — this is
        # the honest "what any predictor could achieve over the whole
        # MC-priced failure surface" number.
        if art.impact_df_nt is not None:
            shed_all = _load_shed_csv(shed_csv)
            impact_ids, blk, total_lookup, _pc = _ec.build_actual_lookups(
                art.impact_df_nt
            )
            shed_all["_impact_id"] = [
                _shed_impact_id(cid, k, impact_ids, blk)
                for cid, k in zip(shed_all["cp_id"], shed_all["kind"])
            ]
            m_full = shed_all.dropna(subset=["_impact_id"]).copy()
            m_full["actual_total"] = m_full["_impact_id"].map(total_lookup)
            fmask = (
                m_full["actual_total"].astype(float) > _ec.MC_FAILED_EPS
            ) & m_full["total_shed"].notna()
            if fmask.sum() >= 3:
                rho, p, lo, hi = _spearman_with_ci(
                    m_full.loc[fmask, "total_shed"],
                    m_full.loc[fmask, "actual_total"],
                )
                full_ceiling_rows.append({
                    "scenario": art.label,
                    "rho_shed_vs_mc": rho, "p_shed_vs_mc": p,
                    "ci_lo": lo, "ci_hi": hi,
                    "n": int(fmask.sum()),
                    "n_child": int(
                        (m_full.loc[fmask, "kind"] == "child").sum()
                    ),
                })

        # Per-metric ρ — sliced into five groups so the per-sector signal on
        # plain branches isn't distorted by coupling-point rows (which carry
        # huge composite stress but small single-removal shed):
        #   total      — every matched row, vs total_shed
        #   multi      — CP rows only (kind != "branch"), vs total_shed —
        #                tells us how well the metric ranks coupling
        #                components, which are inherently multi-sector
        #   power/heat/gas — non-CP rows (kind == "branch") only, vs the
        #                same-sector shed, so the per-carrier number is
        #                a clean within-carrier ranking score
        # Each per-carrier ρ is conditioned on that carrier's shed being
        # non-zero — otherwise a component that never affects (say) heat
        # would drag every metric's heat-ρ toward zero.
        def _sector_specs(suffix: str):
            return [
                ("total", "total_shed" + suffix, None),
                ("multi", "total_shed" + suffix, "cp"),
                ("power", "power_shed" + suffix, "branch"),
                ("heat",  "heat_shed" + suffix,  "branch"),
                ("gas",   "gas_shed" + suffix,   "branch"),
            ]

        # Both shed views: the plain columns (which include the islanded
        # nameplate) and, when the sweep recorded them, the connected-load
        # columns. Result keys are suffixed the same way so a plot can pick
        # a view by column name.
        views = [""]
        if all(c + E16_CONN_SUFFIX in merged.columns
               for c in ("total_shed", "power_shed", "heat_shed", "gas_shed")):
            views.append(E16_CONN_SUFFIX)

        # Carrier-rank pooled frame: within-carrier percentile ranks pooled
        # across carriers. The de-confounded "overall" reference — pooling
        # *raw* sheds across carriers flips ρ negative (Simpson: CP rows +
        # cross-carrier scale mixing) even when every within-carrier ρ is
        # positive, so the raw-pooled ``total`` column must not be read as
        # an overall quality score. This one can.
        branch_mask = (
            merged["kind"] == "branch" if "kind" in merged.columns
            else pd.Series(True, index=merged.index)
        )
        _member = [_ec.carrier_member_mask(merged, t)
                   for t in _ec.CARRIER_TAG_ORDER]
        ranked_pools = {
            sfx: _ec.carrier_rank_pooled(
                merged,
                [f"power_shed{sfx}", f"heat_shed{sfx}", f"gas_shed{sfx}"],
                branch_mask,
                member_masks=_member,
            )
            for sfx in views
        }

        # Cross-scenario CP pool: CP rows with within-scenario percentile
        # ranks of both references, so the tiny per-scenario CP population
        # (n≈14) can be aggregated into one usable sample downstream.
        cp_rows = merged[~branch_mask] if "kind" in merged.columns else merged.iloc[0:0]
        if len(cp_rows) >= 3:
            keep_cols = [c for c in metrics if c in cp_rows.columns]
            # Metric columns are percentile-ranked within the scenario too:
            # raw scores are not on a common scale across scenarios, and a
            # between-scenario level shift would contaminate the pooled ρ.
            pool = cp_rows[keep_cols].rank(pct=True)
            pool["scenario"] = art.label
            if "total_shed" in cp_rows.columns:
                pool["shed_rank"] = cp_rows["total_shed"].rank(pct=True)
            if "actual_total" in cp_rows.columns:
                pool["mc_rank"] = cp_rows["actual_total"].rank(pct=True)
            cp_pool_frames.append(pool)

        for m in metrics:
            if m not in merged.columns or merged[m].notna().sum() < 3:
                continue
            sub_base = merged[merged[m].notna()]
            if len(sub_base) < 3:
                continue
            row: dict = {"scenario": art.label, "metric": m, "n": int(len(sub_base))}
            for sfx in views:
                for tag, col, kind_filter in _sector_specs(sfx):
                    rho_key = f"rho_vs_{tag}_shed{sfx}"
                    p_key = f"p_vs_{tag}_shed{sfx}"
                    lo_key, hi_key = f"ci_lo_{tag}_shed{sfx}", f"ci_hi_{tag}_shed{sfx}"
                    n_key = f"n_{tag}{sfx}"
                    if col not in sub_base.columns:
                        row[rho_key] = row[p_key] = row[lo_key] = row[hi_key] = float("nan")
                        row[n_key] = 0
                        continue
                    sub = sub_base
                    if kind_filter == "branch" and "kind" in sub.columns:
                        sub = sub[sub["kind"] == "branch"]
                    elif kind_filter == "cp" and "kind" in sub.columns:
                        sub = sub[sub["kind"] != "branch"]
                    # Per-carrier slices select on component *membership*,
                    # not on the outcome being non-zero — see
                    # eval_common.CARRIER_MEMBER_TYPES. Values are clipped at
                    # 0 (sub-MIPGap negatives are solver noise) instead of
                    # thresholded, so the analytical and RQMC slices are the
                    # same population and their difference is interpretable.
                    if tag in _ec.CARRIER_MEMBER_TYPES:
                        sub = sub[_ec.carrier_member_mask(sub, tag)
                                  & sub[col].notna()]
                        sub = sub.assign(**{col: sub[col].clip(lower=0.0)})
                    else:
                        sub = sub[sub[col].notna()]
                    if len(sub) < 3 or sub[m].nunique() < 2 or sub[col].nunique() < 2:
                        row[rho_key] = row[p_key] = row[lo_key] = row[hi_key] = float("nan")
                        row[n_key] = int(len(sub))
                        continue
                    rho, p, lo, hi = _spearman_with_ci(sub[m], sub[col])
                    row[rho_key] = rho
                    row[p_key] = p
                    row[lo_key] = lo
                    row[hi_key] = hi
                    row[n_key] = int(len(sub))
                    # Incremental ranking value beyond the 0-hop local-demand
                    # baseline: partial ρ | self_score. A metric can post a high
                    # raw per-carrier ρ purely by proxying demand mass — this
                    # column is what structure/physics adds on top.
                    if (kind_filter == "branch" and m != "self_score"
                            and "self_score" in sub.columns and len(sub) >= 4):
                        prho, pp = _ec.partial_spearman(
                            sub[m], sub[col], sub["self_score"]
                        )
                        row[f"partial_rho_vs_{tag}_shed{sfx}"] = prho
                        row[f"partial_p_vs_{tag}_shed{sfx}"] = pp

                # Carrier-rank pooled slice (tag "ranked"): the legitimate
                # overall number, same column naming scheme as the raw sectors.
                ranked_pool = ranked_pools[sfx]
                rsub = (ranked_pool[ranked_pool[m].notna()]
                        if m in ranked_pool.columns else ranked_pool.iloc[0:0])
                if len(rsub) >= 3 and rsub[m].nunique() >= 2:
                    rho, p, lo, hi = _spearman_with_ci(rsub[m], rsub["_ranked_ref"])
                    row[f"rho_vs_ranked_shed{sfx}"] = rho
                    row[f"p_vs_ranked_shed{sfx}"] = p
                    row[f"ci_lo_ranked_shed{sfx}"] = lo
                    row[f"ci_hi_ranked_shed{sfx}"] = hi
                else:
                    row[f"rho_vs_ranked_shed{sfx}"] = float("nan")
                    row[f"p_vs_ranked_shed{sfx}"] = float("nan")
                    row[f"ci_lo_ranked_shed{sfx}"] = float("nan")
                    row[f"ci_hi_ranked_shed{sfx}"] = float("nan")
                row[f"n_ranked{sfx}"] = int(len(rsub))
                # Back-compat aliases so existing plot code keeps working.
                row[f"rho_vs_shed{sfx}"] = row[f"rho_vs_total_shed{sfx}"]
                row[f"p_vs_shed{sfx}"] = row[f"p_vs_total_shed{sfx}"]
                row[f"ci_lo_shed{sfx}"] = row[f"ci_lo_total_shed{sfx}"]
                row[f"ci_hi_shed{sfx}"] = row[f"ci_hi_total_shed{sfx}"]
            # Also compare to MC for cross-reference. ``actual_total`` may not
            # exist if df_eval came from a non-MC source — guard either way.
            if "actual_total" in sub_base.columns:
                rho_mc, p_mc, lo_mc, hi_mc = _spearman_with_ci(
                    sub_base[m], sub_base["actual_total"]
                )
            else:
                rho_mc = p_mc = lo_mc = hi_mc = float("nan")
            row["rho_vs_mc"] = rho_mc
            row["p_vs_mc"] = p_mc
            row["ci_lo_mc"] = lo_mc
            row["ci_hi_mc"] = hi_mc
            rows.append(row)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_dir / "E16_metric_vs_shed.csv", index=False)
    if ceiling_rows:
        ceil_df = pd.DataFrame(ceiling_rows)
        ceil_df.to_csv(out_dir / "E16_shed_vs_mc_ceiling.csv", index=False)
    if full_ceiling_rows:
        pd.DataFrame(full_ceiling_rows).to_csv(
            out_dir / "E16_shed_vs_mc_ceiling_full.csv", index=False
        )

    # Cross-scenario pooled CP evaluation. Per scenario the CP slice is
    # n≈14 — too small for stable rank statistics (P@10 random baseline
    # 0.71, τ variance huge). Pooling within-scenario percentile ranks
    # across scenarios gives one usable n≈200 sample per family.
    if cp_pool_frames:
        cp_all = pd.concat(cp_pool_frames, ignore_index=True)
        cp_all["family"] = cp_all["scenario"].map(_ec.scenario_family)
        cp_rows_out: List[dict] = []
        for fam, fam_df in cp_all.groupby("family"):
            for m in metrics:
                if m not in fam_df.columns:
                    continue
                for ref, ref_tag in [("shed_rank", "shed"), ("mc_rank", "mc")]:
                    if ref not in fam_df.columns:
                        continue
                    s = fam_df[[m, ref]].dropna()
                    if len(s) < 5 or s[m].nunique() < 2:
                        continue
                    rho, p, lo, hi = _spearman_with_ci(s[m], s[ref])
                    k = _ec.default_ndcg_k(len(s))
                    cp_rows_out.append({
                        "family": fam, "metric": m, "ref": ref_tag,
                        "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi,
                        "rndcg_at_k": _ec.random_normalized_ndcg(
                            s[ref].values, s[m].values, k=k),
                        "rprec_at_k": _ec.rprecision_at_k(
                            s[ref].values, s[m].values, k),
                        "k": k, "n": int(len(s)),
                        "n_scenarios": int(fam_df["scenario"].nunique()),
                    })
        if cp_rows_out:
            pd.DataFrame(cp_rows_out).to_csv(
                out_dir / "E16_cp_pooled.csv", index=False
            )
    return out_df


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner
# ─────────────────────────────────────────────────────────────────────────────


def run_cmres_block(
    artefacts: List[ScenarioArtefacts],
    impact_df: pd.DataFrame,
    output_dir: Path,
    enabled: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the full CMRES evaluation battery.

    ``enabled`` selects a subset of experiment IDs (e.g. ``["E2", "E8", "E9"]``).
    Default = all of E2, E3, E4, E6, E7, E8, E9, E10, E11, E12, E13.
    """
    enabled = enabled or [
        "E2", "E3", "E4", "E6", "E7",
        "E8", "E9", "E10", "E11", "E12", "E13",
        "E15", "E16",
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    results: Dict[str, pd.DataFrame] = {}

    if "E16" in enabled:
        # E16 reads the slurm-produced single_removal_shed CSVs from a
        # parallel directory; if they don't exist, the experiment skips
        # cleanly with a per-scenario warning.
        try:
            results["E16"] = experiment_e16_single_removal_validation(
                artefacts, output_dir,
            )
        except Exception as e:
            print(f"[E16] FAILED: {type(e).__name__}: {e}")

    # E8 and E15 must run before E2/E9/E12/E16 because they augment
    # df_eval with new columns (ml_bc, ddar_mw_total, ss_bc_total,
    # substitutability, …). Order: E8 → E15 → everything else.
    if "E8" in enabled:
        try:
            results["E8"] = experiment_e8_multilayer(artefacts, output_dir)
        except Exception as e:
            print(f"[E8] FAILED: {type(e).__name__}: {e}")

    if "E15" in enabled:
        try:
            results["E15"] = experiment_e15_structural(artefacts, output_dir)
        except Exception as e:
            print(f"[E15] FAILED: {type(e).__name__}: {e}")

    # if "E2" in enabled:
    #     for art in artefacts:
    #         scenario_imp = impact_df[impact_df["network_type"] == art.label] \
    #             if "network_type" in impact_df.columns else impact_df
    #         try:
    #             # Pass the matched df from E1/E5 so the "full" variant doesn't
    #             # rerun mes_all_components_metric (the slow part).
    #             experiment_e2_ablation(
    #                 art.monee_net, scenario_imp, art.label, output_dir,
    #                 df_eval_full=art.df_eval,
    #             )
    #         except Exception as e:
    #             print(f"[E2:{art.label}] FAILED: {type(e).__name__}: {e}")

    if "E3" in enabled:
        try:
            results["E3"] = experiment_e3_density(artefacts, output_dir)
        except Exception as e:
            print(f"[E3] FAILED: {type(e).__name__}: {e}")

    if "E4" in enabled:
        try:
            results.update({f"E4_{k}": v
                           for k, v in experiment_e4_distribution(artefacts, output_dir).items()})
        except Exception as e:
            print(f"[E4] FAILED: {type(e).__name__}: {e}")

    # if "E6" in enabled:
    #     for art in artefacts:
    #         scenario_imp = impact_df[impact_df["network_type"] == art.label] \
    #             if "network_type" in impact_df.columns else impact_df
    #         try:
    #             experiment_e6_sensitivity(art.monee_net, scenario_imp, art.label, output_dir)
    #         except Exception as e:
    #             print(f"[E6:{art.label}] FAILED: {type(e).__name__}: {e}")

    if "E7" in enabled:
        try:
            results["E7"] = experiment_e7_mc_validity(artefacts, output_dir)
        except Exception as e:
            print(f"[E7] FAILED: {type(e).__name__}: {e}")

    if "E9" in enabled:
        try:
            results["E9"] = experiment_e9_percolation(artefacts, output_dir)
        except Exception as e:
            print(f"[E9] FAILED: {type(e).__name__}: {e}")

    if "E10" in enabled:
        try:
            results["E10"] = experiment_e10_structural(artefacts, output_dir)
        except Exception as e:
            print(f"[E10] FAILED: {type(e).__name__}: {e}")

    if "E11" in enabled:
        try:
            results["E11"] = experiment_e11_null_models(artefacts, output_dir)
        except Exception as e:
            print(f"[E11] FAILED: {type(e).__name__}: {e}")

    if "E12" in enabled:
        try:
            results["E12"] = experiment_e12_community(artefacts, output_dir)
        except Exception as e:
            print(f"[E12] FAILED: {type(e).__name__}: {e}")

    if "E13" in enabled:
        try:
            results["E13"] = experiment_e13_spectral(artefacts, output_dir)
        except Exception as e:
            print(f"[E13] FAILED: {type(e).__name__}: {e}")

    try:
        import cmres_eval_plots
        cmres_eval_plots.plot_all(output_dir, output_dir, experiments=enabled)
    except Exception as e:
        print(f"[plots] FAILED: {type(e).__name__}: {e}")

    return results
