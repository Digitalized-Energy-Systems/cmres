"""Shared helpers for ``cp_cn_evaluation`` and ``cmres_eval``.

Holds the constants, statistical primitives, and the matched-df builder that
both modules need so the same logic doesn't live in three places.

Sections
--------
1. Constants:           MC_FAILED_EPS, CP_TYPE_SET, _COMPOUND_CP_TYPES, …
2. Statistical helpers: spearman_with_ci, wilcoxon, holm_correct,
                        bootstrap_ci, ndcg, precision_at_k
3. Matching helpers:    extract_orig_id, build_branch_lookup, match_impact_id
4. Matched-df builder:  build_actual_lookups, build_matched_df
5. Derived columns:     derive_metric_columns

Stability of the bootstrap CI: ``bootstrap_ci`` accepts an ``rng`` argument so
callers preserve reproducibility.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import scipy.stats as _stats


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────


# Components whose actual total impact is below this threshold are treated as
# "never sampled by the MC failure model" and excluded from rank-based
# metrics. Including them inflates NDCG (saturated by ties at zero) and
# biases P@k toward random.
MC_FAILED_EPS = 1e-6

# Noise floor for the analytical single-removal shed (MW). The min-shed
# solves run at MIPGap 1e-4 on MW-scale demand, so sheds below ~1e-4 MW are
# solver tolerance, not physics: in the 2026-06 run 64% of "power-affected"
# rows sat below 0.1 kW. Conditioning per-carrier slices on ``> 0`` admits
# those rows and inflates any predictor that merely knows the component's
# carrier (matched power predictor: ρ 0.89 with them, 0.66 without —
# the noise rows are free concordant pairs at the bottom of the ranking).
# 1e-4 vs 1e-3 is insensitive; keep well above MIPGap × demand scale.
SHED_EPS = 1e-4

# Coupling-point cp_type labels emitted by mes_*_metric.
CP_TYPE_SET = frozenset({
    "CHP", "CHPHG",
    "PowerToHeat", "PowerToHeatHG",
    "PowerToGas", "GasToPower",
    "GasToHeatHG",
})

# Compound vs branch-CP labels (used by match_impact_id to pick the right
# id-rendering convention).
_COMPOUND_CP_TYPES = ("CHP", "CHPHG", "PowerToHeat")
_NON_CP_BRANCH_TYPES = ("PowerLine", "GasPipe", "WaterPipe", "HeatExchanger")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario-family split — shared by every cross-scenario plotter so the
# ``_backup`` / ``_loadbearing`` / ``_control`` partitioning logic lives in
# exactly one place (see test_grids.py module docstring for the family
# semantics: backup = additive CPs + rich gas donor, loadbearing = CPs
# replace primary generation, control = additive CPs + no donor surplus).
# ─────────────────────────────────────────────────────────────────────────────

FAMILY_ORDER = ("backup", "loadbearing", "control")


def scenario_family(scenario) -> str:
    """Family of a scenario key (``backup`` / ``loadbearing`` / ``control``).

    Unknown keys (legacy names, ad-hoc grids) map to ``"other"`` instead of
    raising so older result CSVs can still be plotted.
    """
    s = str(scenario)
    for fam in FAMILY_ORDER:
        if s.endswith(f"_{fam}"):
            return fam
    return "other"


def scenario_stem(scenario) -> str:
    """Scenario key with the family suffix stripped — the density stem
    (e.g. ``simbench_lv_xxl``), used as palette / order key so the same
    density shares a hue across families."""
    s = str(scenario)
    fam = scenario_family(s)
    return s[: -(len(fam) + 1)] if fam != "other" else s


def split_scenarios_by_family(scenarios) -> List[Tuple[str, List[str]]]:
    """Partition scenarios into ``[(family, [scenarios...]), ...]`` in
    ``FAMILY_ORDER`` (+ trailing ``"other"`` group if any).

    Empty classes are dropped so a single-family run collapses to one
    group — letting callers detect "no split needed" via ``len(result) <= 1``
    and keep their original (unsuffixed) figure slugs.

    Order is stable: scenarios within each class keep their input order, which
    matters for figures whose colour mapping or axis ticks read in source
    order (e.g. ``_scenario_order`` already returns the canonical sequence).
    """
    groups = {fam: [] for fam in FAMILY_ORDER + ("other",)}
    for s in scenarios:
        groups[scenario_family(s)].append(s)
    return [(fam, members) for fam, members in groups.items() if members]


# Canonical 10-metric line-up used by both ``cp_cn_evaluation`` (scatter /
# correlation / NDCG figures) and ``cmres_eval.experiment_e16`` (per-sector
# Spearman ρ vs analytical shed). Single source of truth so the two
# pipelines stay in sync. Order = how the metrics will appear on plots
# whose category axis isn't explicitly sorted.
#
# Category coverage:
#   • 3 composite predictors (main / CP-aware / balanced)
#   • 1 bare PTDF stress
#   • 3 centrality-based references (phys. BC / stress BC / Katz)
#   • 1 closeness vitality
#   • 2 simple local baselines (1-hop / 0-hop)
CORE_METRICS: List[Tuple[str, str]] = [
    ("predicted_score",          "PTDF stress + phys. BC"),
    ("predicted_score_cp_aware", "CP-aware composite"),
    ("predicted_score_balanced", "Balanced composite"),
    ("predicted_stress",         "PTDF stress only"),
    ("topo_bc",                  "Phys. BC only"),
    ("stress_bc",                "Stress BC only"),
    ("katz_score",               "Katz BC only"),
    ("vitality_score",           "Closeness vitality"),
    ("local_score",              "1-hop local"),
    ("self_score",               "0-hop self"),
]
CORE_METRIC_COLS: List[str] = [c for c, _ in CORE_METRICS]

# Carrier-matched per-sector predictors. The composite mixes incomparable
# per-carrier score scales (PowerLine median 0.78 vs GasPipe 0.00), which
# costs ~0.22 ρ on the power slice (0.67 composite vs 0.89 matched) — so the
# E16 per-sector evaluation additionally tests each carrier's own predictor
# against its own shed. Kept out of CORE_METRICS: the cp_cn pooled figures
# have no per-carrier slicing where a matched predictor would be meaningful.
MATCHED_METRICS: List[Tuple[str, str]] = [
    ("predicted_power", "Matched predictor (power)"),
    ("predicted_heat",  "Matched predictor (heat)"),
    ("predicted_gas",   "Matched predictor (gas)"),
]
MATCHED_METRIC_COLS: List[str] = [c for c, _ in MATCHED_METRICS]

CORE_METRIC_LABELS: Dict[str, str] = {
    c: lab for c, lab in CORE_METRICS + MATCHED_METRICS
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────


def spearman_with_ci(a, b, alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """Spearman ρ + Fisher-z 95 % CI.

    Returns ``(rho, p, ci_lo, ci_hi)``. NaN-safe: pairs with non-finite
    entries are dropped before the rank computation. CI is NaN if ``n ≤ 3``
    or ``rho`` is non-finite.
    """
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    a_arr, b_arr = a_arr[mask], b_arr[mask]
    n = a_arr.size
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    res = _stats.spearmanr(a_arr, b_arr)
    rho, p = float(res.statistic), float(res.pvalue)
    if n > 3 and np.isfinite(rho):
        # Clamp away from ±1 so atanh stays finite under perfect correlation.
        z = math.atanh(max(-0.999999, min(0.999999, rho)))
        se = 1.0 / math.sqrt(n - 3)
        zc = float(_stats.norm.ppf(1.0 - alpha / 2))
        ci_lo = math.tanh(z - zc * se)
        ci_hi = math.tanh(z + zc * se)
    else:
        ci_lo, ci_hi = float("nan"), float("nan")
    return rho, p, ci_lo, ci_hi


def wilcoxon(a, b) -> Tuple[float, float]:
    """Wilcoxon signed-rank test on paired arrays. Returns ``(W, p)`` or
    ``(nan, nan)`` when the test is undefined (n < 5, all-equal pairs).
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    mask = np.isfinite(arr_a) & np.isfinite(arr_b)
    arr_a, arr_b = arr_a[mask], arr_b[mask]
    if arr_a.size < 5 or np.allclose(arr_a, arr_b):
        return float("nan"), float("nan")
    try:
        res = _stats.wilcoxon(arr_a, arr_b, zero_method="wilcox", correction=False)
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return float("nan"), float("nan")


def holm_correct(pvals: List[float]) -> List[float]:
    """Holm step-down p-value adjustment, NaN-preserving."""
    arr = np.asarray(pvals, dtype=float)
    finite_idx = np.where(np.isfinite(arr))[0]
    if finite_idx.size == 0:
        return list(arr)
    finite_p = arr[finite_idx]
    order = np.argsort(finite_p)
    n = finite_p.size
    adj = np.empty(n)
    running_max = 0.0
    for rank, k in enumerate(order):
        v = finite_p[k] * (n - rank)
        running_max = max(running_max, v)
        adj[k] = min(1.0, running_max)
    out = arr.copy()
    out[finite_idx] = adj
    return list(out)


def bootstrap_ci(
    stat_fn: Callable,
    actual_arr,
    pred_arr,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for any scalar statistic of two paired arrays.

    ``stat_fn(actual, predicted) → float``. Returns ``(ci_lo, ci_hi)``.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(actual_arr)
    p = np.asarray(pred_arr)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan")
    boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[k] = stat_fn(a[idx], p[idx])
    return (
        float(np.percentile(boot, 100 * alpha / 2)),
        float(np.percentile(boot, 100 * (1 - alpha / 2))),
    )


def _discounts(cutoff: int) -> np.ndarray:
    """``1/log2(i+2)`` for i = 0..cutoff-1. Cached per-call (cheap to recompute)."""
    return 1.0 / np.log2(np.arange(int(cutoff)) + 2.0)


def _idcg(gains: np.ndarray, cutoff: int) -> float:
    """Ideal DCG: sort ``gains`` descending, take top ``cutoff``, weight by discounts."""
    if cutoff <= 0 or gains.size == 0:
        return 0.0
    top = np.partition(gains, -cutoff)[-cutoff:] if cutoff < gains.size else gains
    top = np.sort(top)[::-1]
    return float(np.sum(top * _discounts(cutoff)))


def ndcg(actual_vals, predicted_scores, k: Optional[int] = None) -> float:
    """Normalised Discounted Cumulative Gain (optionally @k).

    Relevance = ``max(actual, 0)`` so components with zero/negative impact
    get relevance 0 — they are not "relevant" to identify as critical.
    Using ``actual − min()`` would inflate all scores when the range is
    wide and arbitrarily reward metrics that rank the least-harmful
    component last.

    ``k`` cuts the discounted sum to the top ``k`` positions of each
    ranking (predicted **and** ideal). When ``k`` is ``None`` (default)
    the full list is used — same behaviour as the original definition,
    kept for backwards compatibility. Use a small ``k`` (e.g. ``10``)
    to avoid the saturation that heavy-tailed ``actual`` distributions
    induce on full-list NDCG: a single top-item hit can dominate the
    score, making the metric near-1 even for predictors that disagree
    completely on the rest of the ranking.
    """
    actual_arr = np.asarray(actual_vals, dtype=float)
    if actual_arr.size == 0:
        return 0.0
    n = len(actual_arr)
    cutoff = n if k is None else min(int(k), n)
    if cutoff <= 0:
        return 0.0
    # argsort(-x), not argsort(x)[::-1]: both sort descending but break ties
    # differently, and ndcg_batch (bootstrap CIs) uses the former — keep the
    # point estimate on the same tie-breaking or tied metrics get CIs that
    # exclude their own point value.
    pred_order = np.argsort(-np.asarray(predicted_scores, dtype=float))[:cutoff]
    gains = np.maximum(actual_arr, 0.0)
    discounts = _discounts(cutoff)
    dcg = float(np.sum(gains[pred_order] * discounts))
    idcg = _idcg(gains, cutoff)
    return float(dcg / idcg) if idcg > 0 else 0.0


def ndcg_batch(
    actual_batch: np.ndarray,
    pred_batch: np.ndarray,
    k: Optional[int] = None,
) -> np.ndarray:
    """Vectorised NDCG over a leading batch axis.

    ``actual_batch`` / ``pred_batch`` have shape ``(B, n)``; returns a
    ``(B,)`` array of NDCG values. ~50× faster than calling ``ndcg`` in
    a Python loop because the argsorts and gathers run as single C
    operations across the whole batch — the workhorse of the vectorised
    bootstrap below.
    """
    a = np.asarray(actual_batch, dtype=float)
    p = np.asarray(pred_batch, dtype=float)
    if a.ndim != 2 or p.ndim != 2 or a.shape != p.shape:
        raise ValueError(
            f"ndcg_batch expects matching (B,n) shapes, got {a.shape} / {p.shape}"
        )
    B, n = a.shape
    cutoff = n if k is None else min(int(k), n)
    if cutoff <= 0 or n == 0:
        return np.zeros(B, dtype=float)
    gains = np.maximum(a, 0.0)
    discounts = _discounts(cutoff)
    # argsort in descending order: negate to reuse default ascending sort.
    pred_idx = np.argsort(-p, axis=1)[:, :cutoff]
    ideal_idx = np.argsort(-a, axis=1)[:, :cutoff]
    dcg = (np.take_along_axis(gains, pred_idx, axis=1) * discounts).sum(axis=1)
    idcg = (np.take_along_axis(gains, ideal_idx, axis=1) * discounts).sum(axis=1)
    out = np.zeros(B, dtype=float)
    mask = idcg > 0
    out[mask] = dcg[mask] / idcg[mask]
    return out


def random_normalized_ndcg(
    actual_vals,
    predicted_scores,
    k: Optional[int] = None,
    n_random: int = 200,  # ignored, kept for API compat
    rng=None,             # ignored, kept for API compat
) -> float:
    """Random-baseline-normalised NDCG (rNDCG).

    ``rNDCG = (NDCG − E[NDCG_random]) / (1 − E[NDCG_random])``

    - ``0`` means the predicted ranking is no better than a uniform random
      permutation of the same items;
    - ``1`` is the ideal ranking;
    - negative values mean the predicted ranking is *worse* than random
      (anti-correlated with ``actual``).

    The random-permutation expectation has a **closed form**: for a
    uniform random permutation of ``n`` items, every item lands in each
    of the top-``k`` slots with probability ``1/n``, so the expected
    gain at every slot equals ``mean(gains)`` and

        E[DCG@k] = mean(gains) × Σ_{i=0..k-1} 1/log2(i+2)

    which we divide by IDCG@k to get ``E[NDCG_random]``. Replaces the
    earlier Monte-Carlo estimate (which dominated bootstrap-CI cost) —
    same numerics to within MC noise, but O(n + k) per call instead of
    O(n_random × n log n).
    """
    actual_arr = np.asarray(actual_vals, dtype=float)
    n = actual_arr.size
    if n == 0:
        return 0.0
    cutoff = n if k is None else min(int(k), n)
    if cutoff <= 0:
        return 0.0
    gains = np.maximum(actual_arr, 0.0)
    idcg = _idcg(gains, cutoff)
    if idcg <= 0:
        return 0.0
    mean_gain = float(gains.mean())
    e_random = (mean_gain * float(_discounts(cutoff).sum())) / idcg
    obs = ndcg(actual_arr, predicted_scores, k=k)
    denom = 1.0 - e_random
    if denom <= 0:
        return 0.0
    return float((obs - e_random) / denom)


def bootstrap_ndcg_ci(
    actual_arr,
    pred_arr,
    *,
    k: Optional[int] = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    normalize_random: bool = False,
) -> Tuple[float, float]:
    """Vectorised percentile-bootstrap CI for NDCG (or rNDCG when
    ``normalize_random=True``).

    Generates the full ``(n_boot, n)`` resample-index matrix in one C
    call, gathers actual/predicted values into a batch, and runs
    :func:`ndcg_batch` to score all bootstrap samples at once — replaces
    a Python loop that called ``ndcg`` 1000× per CI. For
    ``normalize_random=True`` the per-sample random baseline is computed
    in closed form (no inner MC), so an rNDCG bootstrap is now the same
    cost as a plain NDCG bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    a = np.asarray(actual_arr, dtype=float)
    p = np.asarray(pred_arr, dtype=float)
    n = a.size
    if n == 0:
        return float("nan"), float("nan")
    cutoff = n if k is None else min(int(k), n)
    if cutoff <= 0:
        return float("nan"), float("nan")

    idx = rng.integers(0, n, size=(int(n_boot), n))
    a_b = a[idx]                # (B, n)
    p_b = p[idx]                # (B, n)
    raw = ndcg_batch(a_b, p_b, k=cutoff)

    if not normalize_random:
        return (
            float(np.percentile(raw, 100 * alpha / 2)),
            float(np.percentile(raw, 100 * (1 - alpha / 2))),
        )

    # Closed-form rNDCG per bootstrap sample: each row b uses the
    # resampled gains' mean and own IDCG.
    gains_b = np.maximum(a_b, 0.0)
    discounts = _discounts(cutoff)
    sum_discounts = float(discounts.sum())
    # Per-row IDCG: top-k of gains_b.
    if cutoff < n:
        top_idx = np.argpartition(-gains_b, cutoff, axis=1)[:, :cutoff]
        top_b = np.take_along_axis(gains_b, top_idx, axis=1)
    else:
        top_b = gains_b
    # Sort each row's top-k descending so the discount weighting is correct.
    top_b = -np.sort(-top_b, axis=1)
    idcg_b = (top_b * discounts).sum(axis=1)
    mean_gain_b = gains_b.mean(axis=1)
    safe = idcg_b > 0
    e_rand = np.zeros_like(idcg_b)
    e_rand[safe] = (mean_gain_b[safe] * sum_discounts) / idcg_b[safe]
    denom = 1.0 - e_rand
    rndcg = np.zeros_like(raw)
    keep = denom > 0
    rndcg[keep] = (raw[keep] - e_rand[keep]) / denom[keep]
    return (
        float(np.percentile(rndcg, 100 * alpha / 2)),
        float(np.percentile(rndcg, 100 * (1 - alpha / 2))),
    )


def default_ndcg_k(n: int) -> int:
    """Heuristic default for ``k`` in NDCG@k.

    Returns ``max(5, min(20, ceil(n × 0.2)))`` — the top quintile, capped
    to a 5..20 range. Below 5 the discount weighting barely matters; above
    20 saturation on heavy-tailed relevance becomes visible again.
    """
    if n <= 0:
        return 0
    import math
    return max(5, min(20, int(math.ceil(n * 0.2))))


def precision_at_k(actual_vals, predicted_scores, k: int) -> float:
    """Fraction of true top-k components that appear in the predicted top-k.

    ``k`` is clamped to the list length so a perfect predictor on a short
    list (e.g. the n=14 CP-only slice at k=20) still scores 1.0 instead of
    being silently capped at ``n/k``.
    """
    actual_arr = np.asarray(actual_vals)
    if actual_arr.size == 0:
        return 0.0
    k = min(int(k), actual_arr.size)
    if k <= 0:
        return 0.0
    actual_top = set(np.argsort(actual_arr)[-k:])
    pred_top = set(np.argsort(predicted_scores)[-k:])
    return len(actual_top & pred_top) / k


def rprecision_at_k(actual_vals, predicted_scores, k: int) -> float:
    """Random-baseline-normalised precision@k (rP@k), analogous to rNDCG.

    ``rP@k = (P@k − k/n) / (1 − k/n)`` — 0 means no better than picking k
    items at random, 1 is perfect, negative is worse than random. Plain P@k
    is incomparable across slice sizes: its random baseline is k/n, i.e.
    0.03 on a 384-row branch slice but 0.71 on a 14-row CP slice, so the
    same raw value means "excellent" on one and "below chance" on the other.
    Returns NaN when k ≥ n (P@k is then 1 by construction).
    """
    actual_arr = np.asarray(actual_vals)
    n = actual_arr.size
    if n == 0 or k <= 0:
        return float("nan")
    k = min(int(k), n)
    baseline = k / n
    if baseline >= 1.0:
        return float("nan")
    p = precision_at_k(actual_arr, predicted_scores, k)
    return float((p - baseline) / (1.0 - baseline))


def partial_spearman(x, y, control) -> Tuple[float, float]:
    """Spearman ρ between ``x`` and ``y`` after removing the (rank-linear)
    contribution of ``control`` from both — the incremental ranking signal
    of ``x`` beyond the control variable.

    Used to quantify what a structural metric knows beyond the 0-hop local
    demand baseline (``self_score``): a metric can post a high raw ρ purely
    by proxying demand mass. Returns ``(rho, p)``; NaN-safe.
    """
    arrs = [np.asarray(v, dtype=float) for v in (x, y, control)]
    mask = np.logical_and.reduce([np.isfinite(a) for a in arrs])
    x_a, y_a, c_a = (a[mask] for a in arrs)
    if x_a.size < 4:
        return float("nan"), float("nan")
    rx, ry, rc = (_stats.rankdata(a) for a in (x_a, y_a, c_a))
    if np.ptp(rc) == 0:
        res = _stats.spearmanr(rx, ry)
        return float(res.statistic), float(res.pvalue)
    ex = rx - np.polyval(np.polyfit(rc, rx, 1), rc)
    ey = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
    # Degenerate residuals (variable ≈ fully explained by the control) are
    # float noise at ~1e-14 — ranking that noise produces spurious ±1.
    tol = 1e-8 * x_a.size
    if np.max(np.abs(ex)) < tol or np.max(np.abs(ey)) < tol:
        return float("nan"), float("nan")
    res = _stats.spearmanr(ex, ey)
    return float(res.statistic), float(res.pvalue)


def carrier_rank_pooled(
    df,
    shed_cols: Sequence[str],
    branch_mask,
    ref_col: str = "_ranked_ref",
    min_shed: float = SHED_EPS,
):
    """Long frame for the carrier-rank pooled evaluation.

    For each carrier shed column: take branch rows with shed above the
    noise floor (``min_shed``, default :data:`SHED_EPS`) and replace the
    shed by its percentile rank *within that carrier*, then concatenate.
    Correlating a metric against ``ref_col`` on the result gives a pooled
    "overall" ρ that is immune to the Simpson flip of raw cross-carrier
    pooling (raw scales differ by orders of magnitude between carriers, so
    pooled raw ρ came out negative in 15/15 scenarios while every
    within-carrier ρ was positive). Components affecting several carriers
    appear once per carrier — intentional: each appearance is one ranking
    task. Returns a new frame with ``ref_col`` added.
    """
    parts = []
    for col in shed_cols:
        if col not in df.columns:
            continue
        sub = df[branch_mask & df[col].notna() & (df[col] > min_shed)]
        if len(sub) < 3:
            continue
        part = sub.copy()
        part[ref_col] = sub[col].rank(pct=True)
        parts.append(part)
    if not parts:
        out = df.iloc[0:0].copy()
        out[ref_col] = []
        return out
    return pd.concat(parts, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Matching helpers
# ─────────────────────────────────────────────────────────────────────────────


def extract_orig_id(cp_id: str):
    """Pull the integer 'orig_id' out of a cp_id string.

    Examples:
      ``"compound:5"``        → 5
      ``"branch:(5, 134, 0)"`` → 5
      ``"5→134"``             → 5
      Returns ``None`` on failure.
    """
    s = str(cp_id)
    try:
        if s.startswith("compound:"):
            return int(s[len("compound:"):])
        if s.startswith("branch:"):
            inner = s[len("branch:"):].strip("()")
            return int(inner.split(",")[0].strip())
        if "→" in s:
            return int(s.split("→")[0].strip())
        return int(s)
    except Exception:
        return None


def build_branch_lookup(impact_ids: Iterable[str]) -> Dict[Tuple[str, str], str]:
    """Map ``(a_str, b_str) → impact_id`` for every branch impact id, both
    directions. Used by ``match_impact_id`` so a metric-side cp_id of
    ``"(9, 2, 0)"`` matches an impact_df id of ``"branch:(2, 9, 0)"``.
    """
    branch_lookup: Dict[Tuple[str, str], str] = {}
    for iid in impact_ids:
        if not isinstance(iid, str) or not iid.startswith("branch:"):
            continue
        inner = iid[len("branch:"):].strip("()")
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
            branch_lookup.setdefault((a, b), iid)
            branch_lookup.setdefault((b, a), iid)
    return branch_lookup


def match_impact_id(
    cp_id,
    cp_type: str,
    impact_ids: Set[str],
    branch_lookup: Optional[Dict[Tuple[str, str], str]] = None,
) -> Optional[str]:
    """Find the impact_df id string for one ``mes_*_metric`` row.

    Routing by cp_type:
      - Compound CPs (``CHP``, ``CHPHG``, ``PowerToHeat``): ``"compound:{id}"``.
      - Non-CP branches (``PowerLine``, ``GasPipe``, ``WaterPipe``,
        ``HeatExchanger``): ``"branch:{cp_id}"`` direct, or fuzzy via
        branch_lookup if the tuple-order differs.
      - Branch CPs (``GasToPower``, ``PowerToGas``, ``PowerToHeatHG``,
        ``GasToHeatHG``): cp_id is rendered as ``"from→to"``.
    """
    if cp_type in _COMPOUND_CP_TYPES:
        candidate = f"compound:{cp_id}"
        return candidate if candidate in impact_ids else None

    if branch_lookup is None:
        branch_lookup = build_branch_lookup(impact_ids)

    if cp_type in _NON_CP_BRANCH_TYPES:
        candidate = f"branch:{cp_id}"
        if candidate in impact_ids:
            return candidate
        inner = str(cp_id).strip("()")
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) >= 2:
            return branch_lookup.get((parts[0], parts[1]))
        return None

    # Branch CP — id is "from→to"
    try:
        from_id, to_id = str(cp_id).split("→")
        from_id, to_id = from_id.strip(), to_id.strip()
    except ValueError:
        return None
    return branch_lookup.get((from_id, to_id))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Matched-df builder
# ─────────────────────────────────────────────────────────────────────────────


def harm_from_impact(impact: pd.Series) -> pd.Series:
    """Convert the signed MC impact (``out_mean − in_mean``, NEGATIVE for a
    harmful component) into a non-negative harm value: ``max(0, −impact)``.

    Not ``abs()``: the impact for a harmless component (and for the two
    carriers a single-carrier component never touches) is a noisy
    difference centred on 0, and ``abs()`` folds that noise upward into a
    fake positive "actual impact" ∝ the sampling SD — components at the
    bottom of the true ranking then get ranked by their noise magnitude.
    Clamping keeps the harmful (negative) side and zeroes the
    background-was-higher side instead of counting it as criticality.
    """
    return (-impact).clip(lower=0)


def build_actual_lookups(impact_df_nt: pd.DataFrame):
    """Build the lookup tables used to attach MC ground-truth values to a
    score row.

    Returns a 4-tuple ``(impact_ids, branch_lookup, total_lookup,
    per_carrier_lookup)``:
      - ``impact_ids``: set of all id strings in ``impact_df_nt``.
      - ``branch_lookup``: see ``build_branch_lookup``.
      - ``total_lookup``: ``id → Σ harm`` over carriers (see
        :func:`harm_from_impact` for the sign handling).
      - ``per_carrier_lookup``: ``carrier → {id → harm}`` for per-carrier
        slicing in the eval (electricity / heat / gas).
    """
    impact_abs = impact_df_nt.copy()
    impact_abs["impact"] = harm_from_impact(impact_abs["impact"])
    # min_count=1: a never-MC-sampled component (all-NaN impacts) stays NaN
    # instead of becoming a fake measured 0 that ties with true zeros.
    actual_total = (
        impact_abs.groupby("id")["impact"].sum(min_count=1)
        .reset_index()
        .rename(columns={"impact": "actual_total"})
    )
    actual_per_carrier = (
        impact_abs.pivot_table(index="id", columns="carrier", values="impact", aggfunc="sum")
        .reset_index()
    )
    impact_ids = set(actual_total["id"].astype(str))
    branch_lookup = build_branch_lookup(impact_ids)
    total_lookup = dict(zip(actual_total["id"].astype(str), actual_total["actual_total"]))
    per_carrier_lookup = {
        col: dict(zip(actual_per_carrier["id"].astype(str), actual_per_carrier[col]))
        for col in actual_per_carrier.columns
        if col != "id"
    }
    return impact_ids, branch_lookup, total_lookup, per_carrier_lookup


def build_matched_df(
    df_scores: pd.DataFrame,
    impact_df_nt: pd.DataFrame,
) -> pd.DataFrame:
    """Join ``mes_*_metric`` scores against MC actuals into a tidy
    matched dataframe.

    Each output row corresponds to one component for which a metric score
    exists AND an MC impact row exists. Columns:
      - identification: ``cp_id``, ``cp_type``
      - composite/intermediate metrics: ``predicted_score``,
        ``predicted_stress``, ``topo_factor``, ``topo_bc``, ``stress_bc``,
        ``stress_score``, ``local_score``, ``self_score``, ``katz_score``,
        ``vitality_score``, ``input_adequacy``
      - per-carrier predictions: ``predicted_power_stress`` /
        ``predicted_gas_stress`` / ``predicted_heat_stress``
      - actuals: ``actual_total``, ``actual_electricity``, ``actual_gas``,
        ``actual_heat``

    This is the *unfiltered* matched df — callers apply their own
    ``actual_total > MC_FAILED_EPS`` filter when they want only MC-sampled
    components. Returns an empty DataFrame if nothing matches.
    """
    impact_ids, branch_lookup, total_lookup, per_carrier_lookup = (
        build_actual_lookups(impact_df_nt)
    )

    rows: List[dict] = []
    for score_row in df_scores.itertuples(index=False):
        impact_id = match_impact_id(
            score_row.cp_id, score_row.cp_type, impact_ids, branch_lookup
        )
        if impact_id is None:
            continue
        actual_total_val = total_lookup.get(impact_id)
        if actual_total_val is None:
            continue
        score_dict = score_row._asdict()
        # Default to 1.0 so non-CP rows (which don't have an input-adequacy
        # gate and end up with NaN in the column after pd.concat in
        # mes_all_components_metric) don't accidentally land at 0.
        adq_raw = score_dict.get("input_adequacy", 1.0)
        try:
            adq = float(adq_raw)
            if not (adq == adq) or adq in (float("inf"), float("-inf")):
                adq = 1.0
        except (TypeError, ValueError):
            adq = 1.0

        entry = {
            "cp_id": str(score_row.cp_id),
            "cp_type": score_row.cp_type,
            "predicted_score": score_row.score,
            "predicted_stress": score_row.total_stress,
            "topo_factor": score_row.topo_factor,
            "topo_bc": score_row.topo_bc,
            # CP-aware variant: BC computed with w_cp = (2−η)/Φ_rated and a
            # ``cp`` normalisation class alongside power/gas/heat (see the
            # standalone derivation in docs/cp_edge_weight_theory.tex).
            "topo_factor_cp_aware": score_dict.get(
                "topo_factor_cp_aware", score_row.topo_factor),
            "topo_bc_cp_aware": score_dict.get("topo_bc_cp_aware", score_row.topo_bc),
            "predicted_score_cp_aware": score_dict.get("score_cp_aware", score_row.score),
            # Exergy-aware variant: w_cp = (2−η_ex)/Φ_rated with carrier
            # quality factors q_k (electricity = 1, gas ≈ 1, heat = Carnot).
            # See docs/new_edge_weight_theory.tex.
            "topo_factor_exergy": score_dict.get(
                "topo_factor_exergy", score_row.topo_factor),
            "topo_bc_exergy": score_dict.get("topo_bc_exergy", score_row.topo_bc),
            "predicted_score_exergy": score_dict.get("score_exergy", score_row.score),
            # Balanced composite — S1 (per-carrier stress norm) + C1 (ext-
            # grid headroom) + C2 (demand-coupling) + C3 (substitutability).
            # Computed by ``cp_metric.attach_balanced_score``.
            "predicted_score_balanced": score_dict.get(
                "predicted_score_balanced", score_row.score),
            "total_stress_balanced": score_dict.get(
                "total_stress_balanced", score_row.total_stress),
            "ext_headroom_mult": score_dict.get("ext_headroom_mult", 1.0),
            "demand_coupling_mult": score_dict.get("demand_coupling_mult", 1.0),
            "substitutability_mult": score_dict.get("substitutability_mult", 1.0),
            # Per-carrier atomic predictors (option 3): one score per
            # carrier × per topology variant, so each can be ranked against
            # its own carrier's shed without cross-carrier mixing. See
            # cp_metric.attach_per_carrier_scores.
            "predicted_power":          score_dict.get("predicted_power", 0.0),
            "predicted_gas":            score_dict.get("predicted_gas", 0.0),
            "predicted_heat":           score_dict.get("predicted_heat", 0.0),
            "predicted_power_cp_aware": score_dict.get("predicted_power_cp_aware", 0.0),
            "predicted_gas_cp_aware":   score_dict.get("predicted_gas_cp_aware", 0.0),
            "predicted_heat_cp_aware":  score_dict.get("predicted_heat_cp_aware", 0.0),
            "predicted_power_exergy":   score_dict.get("predicted_power_exergy", 0.0),
            "predicted_gas_exergy":     score_dict.get("predicted_gas_exergy", 0.0),
            "predicted_heat_exergy":    score_dict.get("predicted_heat_exergy", 0.0),
            "predicted_power_balanced": score_dict.get("predicted_power_balanced", 0.0),
            "predicted_gas_balanced":   score_dict.get("predicted_gas_balanced", 0.0),
            "predicted_heat_balanced":  score_dict.get("predicted_heat_balanced", 0.0),
            "stress_bc": score_dict.get("stress_bc", 0.0),
            "stress_score": score_dict.get("stress_score", score_row.score),
            "local_score": score_dict.get("local_score", score_row.score),
            "self_score": score_dict.get("self_score", score_row.score),
            "katz_score": score_dict.get("katz_score", 0.0),
            "vitality_score": score_dict.get("vitality_score", 0.0),
            "input_adequacy": adq,
            "actual_total": actual_total_val,
        }
        for metric_col, carrier_col in [
            ("power_stress", "electricity"),
            ("gas_stress", "gas"),
            ("heat_stress", "heat"),
        ]:
            entry[f"predicted_{metric_col}"] = score_dict.get(metric_col, 0.0)
            carrier_map = per_carrier_lookup.get(carrier_col)
            entry[f"actual_{carrier_col}"] = (
                carrier_map.get(impact_id, 0.0) if carrier_map is not None else 0.0
            )
        rows.append(entry)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Derived columns
# ─────────────────────────────────────────────────────────────────────────────


def derive_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the derived metric columns that downstream views consume:

      - ``score_no_topo``      : pure PTDF stress (= ``predicted_stress``)
      - ``score_topo_only``    : raw BC (= ``topo_bc``)
      - ``score_no_adequacy``  : full predicted_score with the input-adequacy
                                  gate divided out (NaN when adequacy is 0,
                                  i.e. unreachable input nodes)

    The function is idempotent — already-present columns are kept.
    Always returns a copy so the caller's frame is never mutated; the
    previous implementation only copied on the first branch, then mutated
    the caller's frame in later branches.
    """
    df = df.copy()
    if "score_no_topo" not in df.columns and "predicted_stress" in df.columns:
        df["score_no_topo"] = df["predicted_stress"]
    if "score_topo_only" not in df.columns and "topo_bc" in df.columns:
        df["score_topo_only"] = df["topo_bc"]
    if "score_no_adequacy" not in df.columns and "predicted_score" in df.columns:
        if "input_adequacy" in df.columns:
            adq = df["input_adequacy"].where(df["input_adequacy"] > 0, float("nan"))
            df["score_no_adequacy"] = df["predicted_score"] / adq
        else:
            df["score_no_adequacy"] = df["predicted_score"]
    if "input_adequacy" not in df.columns:
        df["input_adequacy"] = 1.0
    return df
