"""RQMC Monte Carlo engine for MES resilience simulation.

Variance-reduction stack
------------------------
1. Randomised Quasi-Monte Carlo (RQMC)
   Owen-scrambled Sobol sequences. All random inputs for one simulation
   run are pre-generated as a single Sobol point, reshaped to
   (n_components, T_incident, N_DIM) with N_DIM=2:
     dim 0 → N(1, 0.1) probability multiplier (via norm.ppf)
     dim 1 → Bernoulli trigger (direct uniform comparison)

2. Antithetic Variates (AV)
   Every Sobol point u is paired with its antithetic twin (1 − u). The
   FailureScenario carries the raw uniform[0,1] inputs, so the reflection
   is exact for both the Gaussian multiplier and the Bernoulli trigger.

3. Self-Normalised Importance Sampling (SNIS)
   The WeightedAccumulator implements the SNIS estimator. For plain RQMC
   (all log_weight = 0) it reduces to unweighted Welford. IS log-weights
   are stored in FailureScenario and accumulated for downstream IS
   extensions.

Stopping criterion — sequential relative CI
   The simulation stops when the 95 % CI relative half-width for each
   carrier's mean performance loss satisfies:

       (CI half-width) / max(|mean|, eps)  ≤  rel_tol

   for all three carriers simultaneously, after at least ``min_runs`` samples.
   The CI is computed by Welford over the union of normal + antithetic
   samples; with negative AV correlation this estimator is conservative
   for the AV-paired estimator's true variance, so convergence is slightly
   slower than the AV reduction would allow.

References
----------
Sobol' (1967); Owen (1995, 1997, 2013); Joe & Kuo (2008);
Hammersley & Morton (1956); Chan, Golub & LeVeque (1983);
Kish (1965); Welford (1962); McKay, Beckman & Conover (1979) (LHS
fallback when d > 21 201).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, ClassVar, List, Optional, Tuple

import numpy as np
import scipy.stats as stats
import scipy.stats.qmc as qmc

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Component Registry  —  canonical ordering of potentially-failing components
# ─────────────────────────────────────────────────────────────────────────────


class ComponentRegistry:
    """Canonical, stable index → component mapping for a Network.

    The registry iterates components in the same order as
    ``SimpleResilienceModel.generate_failures``, assigning each a unique
    integer index.  That index slices into the FailureScenario uniform array,
    ensuring each RQMC Sobol dimension is always tied to the same component.

    Iteration order
    ---------------
    1. ``net.nodes``     – independent nodes only
    2. ``net.branches``  – independent branches, plus HeatExchanger regardless
    3. ``net.childs``    – independent children only
    4. ``net.compounds`` – all compounds

    Identity
    --------
    Components are keyed by ``(kind, comp.id)`` where ``kind`` is one of
    ``"node" | "branch" | "child" | "compound"``.  This survives
    ``Network.copy()`` (which produces fresh Python objects with new
    ``id(comp)`` but identical ``comp.id``) and avoids collisions between the
    four categories, whose ``comp.id`` spaces overlap.

    Parameters
    ----------
    net : monee.Network
    """

    def __init__(self, net):
        import monee.model as mm

        self._mm = mm
        self._components: list = []
        self._id_map: dict = {}

        for node in net.nodes:
            if node.independent:
                self._register(node, "node")
        for branch in net.branches:
            if branch.independent or type(branch.model) is mm.HeatExchanger:
                self._register(branch, "branch")
        for child in net.childs:
            if child.independent:
                self._register(child, "child")
        for compound in net.compounds:
            self._register(compound, "compound")

    def _register(self, comp, kind: str) -> None:
        idx = len(self._components)
        self._id_map[(kind, comp.id)] = idx
        self._components.append(comp)

    @staticmethod
    def _kind_of(comp) -> str:
        cls = type(comp).__name__
        if cls == "Node":
            return "node"
        if cls == "Branch":
            return "branch"
        if cls == "Child":
            return "child"
        if cls == "Compound":
            return "compound"
        return cls.lower()

    def index_of(self, comp) -> Optional[int]:
        """Return the registry index for *comp*, or ``None`` if not registered."""
        return self._id_map.get((self._kind_of(comp), comp.id))

    @property
    def n_components(self) -> int:
        """Number of registered components."""
        return len(self._components)

    @property
    def components(self) -> list:
        return list(self._components)


# ─────────────────────────────────────────────────────────────────────────────
# Failure Scenario  —  pre-sampled uniform[0,1] inputs for one MC run
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FailureScenario:
    """Pre-sampled uniform[0,1] inputs for one resilience simulation run.

    ``uniforms`` has shape ``(n_components, T_incident, N_DIM)`` where
    N_DIM = 2.

    Dimension layout
    ----------------
    0 : failure probability multiplier  →  N(1, 0.1)  via ``norm.ppf``
    1 : failure Bernoulli trigger       →  compare U directly with fail_prob

    Repair-time and damage-multiplier dimensions were removed: repair is now
    a deterministic function of ``(component, severity)`` so its variance
    no longer consumes Sobol budget.

    Antithetic twin
    ---------------
    ``make_antithetic()`` returns a reflected scenario (1 − U).  When paired
    with the original, both inputs are maximally negatively correlated.
    For ``log_weight != 0`` (IS mode), the weight is inherited unchanged.
    """

    N_DIM: ClassVar[int] = 2

    uniforms: np.ndarray  # shape (n_components, T_incident, N_DIM)
    log_weight: float = 0.0  # log IS weight; 0.0 → plain RQMC (no reweighting)

    def make_antithetic(self) -> "FailureScenario":
        """Return the antithetic twin: reflect all uniforms about 0.5."""
        return FailureScenario(
            uniforms=1.0 - self.uniforms,
            log_weight=self.log_weight,
        )


# ─────────────────────────────────────────────────────────────────────────────
# RQMC Sampler  —  Owen-scrambled Sobol sequence
# ─────────────────────────────────────────────────────────────────────────────


class RQMCSampler:
    """Generate FailureScenarios from an Owen-scrambled Sobol sequence.

    For a network with *n_c* components, *T* incident timesteps, and
    ``FailureScenario.N_DIM`` (=2) random inputs per (component, timestep),
    the Sobol engine operates in d = n_c × T × N_DIM dimensions. All
    scenarios are pre-generated in one batch (Sobol sequences should be
    consumed as a block for best uniformity).

    If d > 21 201 (scipy Sobol limit), the sampler falls back to a randomised
    Latin Hypercube design, which still provides better marginal coverage than
    pure pseudo-random.

    Parameters
    ----------
    n_components : int
        From ``ComponentRegistry.n_components``.
    T_incident : int
        Number of incident timesteps (must match ``SimpleResilienceModel``).
    n_scenarios : int
        Total scenarios to pre-generate.  Should be ``≥ max_runs // 2``
        when antithetic_variates=True (one Sobol point → two runs).
    base_seed : int | None
        Scrambling seed.  ``None`` → non-reproducible.
    """

    SOBOL_MAX_DIM: ClassVar[int] = 21_201

    def __init__(
        self,
        n_components: int,
        T_incident: int,
        n_scenarios: int,
        base_seed: Optional[int] = None,
    ):
        self._n_components = n_components
        self._T_incident = T_incident
        self._d = n_components * T_incident * FailureScenario.N_DIM
        self._idx = 0

        if n_components == 0:
            # Edge case: empty network
            self._raw = np.empty((n_scenarios, 0))
            return

        if self._d <= self.SOBOL_MAX_DIM:
            engine = qmc.Sobol(d=self._d, scramble=True, seed=base_seed)
        else:
            engine = qmc.LatinHypercube(d=self._d, seed=base_seed)

        self._raw: np.ndarray = engine.random(n_scenarios)  # (n_scenarios, d)

    def next_scenario(self) -> FailureScenario:
        """Return the next pre-generated scenario."""
        if self._idx >= len(self._raw):
            raise StopIteration("RQMCSampler exhausted; increase n_scenarios.")
        row = self._raw[self._idx]
        self._idx += 1
        if self._n_components == 0:
            u = np.empty((0, self._T_incident, FailureScenario.N_DIM))
        else:
            u = row.reshape(self._n_components, self._T_incident, FailureScenario.N_DIM)
        return FailureScenario(uniforms=u.copy())

    def reset(self) -> None:
        """Rewind the sampler to the first scenario."""
        self._idx = 0

    @property
    def n_scenarios(self) -> int:
        return len(self._raw)

    @property
    def d(self) -> int:
        """Sobol dimensionality."""
        return self._d


# ─────────────────────────────────────────────────────────────────────────────
# Weighted online statistics  —  supports both plain MC and IS
# ─────────────────────────────────────────────────────────────────────────────


class WeightedAccumulator:
    """Online weighted mean and variance (Chan et al. 1983).

    Supports vector-valued samples (one slot per energy carrier) and
    importance-sampling weights via ``log_weight``.  For unit weights
    (``log_weight = 0`` for all samples), the estimator reduces to ordinary
    Welford.

    Variance estimate uses the Bessel-corrected weighted formula::

        Var ≈ S / (W − W₂/W)

    where W = Σwᵢ, W₂ = Σwᵢ², S = Σwᵢ(xᵢ − μ̂)².

    The effective sample size (ESS) follows Kish (1965)::

        ESS = W² / W₂
    """

    CARRIER_NAMES: ClassVar[List[str]] = ["power", "heat", "gas"]

    def __init__(self, n_carriers: int = 3):
        self.n = 0
        self._sum_w = 0.0
        self._sum_w2 = 0.0
        self._mean = np.zeros(n_carriers)
        self._S = np.zeros(n_carriers)  # Σwᵢ(xᵢ − μ̂)²

    def update(self, x, log_weight: float = 0.0) -> None:
        """Incorporate one new weighted sample vector."""
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            import warnings

            warnings.warn(
                f"WeightedAccumulator.update: non-finite sample {x} skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        w = math.exp(log_weight)
        self.n += 1
        self._sum_w += w
        self._sum_w2 += w * w
        # Chan et al. (1983) parallel weighted update
        old_mean = self._mean.copy()
        self._mean += (w / self._sum_w) * (x - old_mean)
        self._S += w * (x - old_mean) * (x - self._mean)

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def ess(self) -> float:
        """Kish (1965) effective sample size."""
        if self._sum_w2 == 0.0:
            return 0.0
        return (self._sum_w**2) / self._sum_w2

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2 or self._sum_w == 0.0:
            return np.full_like(self._mean, np.inf)
        denom = self._sum_w - self._sum_w2 / self._sum_w
        if denom <= 0.0:
            return np.full_like(self._mean, np.inf)
        return self._S / denom

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Two-sided (1−alpha) CI; Normal for ESS≥30, Student-t otherwise."""
        if self.n < 2:
            inf = np.full_like(self._mean, np.inf)
            return -inf, inf
        eff = max(self.ess, 1.0)
        se = np.sqrt(self.variance / eff)
        if eff >= 30:
            z = stats.norm.ppf(1.0 - alpha / 2.0)
        else:
            z = stats.t.ppf(1.0 - alpha / 2.0, df=max(eff - 1.0, 1.0))
        hw = z * se
        return self._mean - hw, self._mean + hw

    def relative_half_width(self, eps: float = 1e-8) -> np.ndarray:
        """95 % CI half-width / |mean| per carrier."""
        if self.n < 2:
            return np.full_like(self._mean, np.inf)
        eff = max(self.ess, 1.0)
        se = np.sqrt(self.variance / eff)
        if eff >= 30:
            z = stats.norm.ppf(0.975)
        else:
            z = stats.t.ppf(0.975, df=max(eff - 1.0, 1.0))
        hw = z * se
        return hw / (np.abs(self._mean) + eps)


# Backward-compatible alias for code that imports WelfordAccumulator directly.
WelfordAccumulator = WeightedAccumulator


# ─────────────────────────────────────────────────────────────────────────────
# Convergence criterion  —  sequential relative CI
# ─────────────────────────────────────────────────────────────────────────────


class CIStoppingCriterion:
    """Stop when the 95 % CI relative half-width ≤ rel_tol for all carriers.

    Uses ``WeightedAccumulator`` so IS runs (non-unit weights) are handled
    correctly via the ESS-based CI width.

    Parameters
    ----------
    rel_tol    : fractional convergence tolerance (default 0.05 = 5 %)
    min_runs   : minimum completed runs before convergence is checked
    n_carriers : number of tracked carriers (default 3: power, heat, gas)
    """

    def __init__(
        self,
        rel_tol: float = 0.05,
        min_runs: int = 200,
        n_carriers: int = 3,
    ):
        self.rel_tol = rel_tol
        self.min_runs = min_runs
        self._acc = WeightedAccumulator(n_carriers)

    def update(self, carrier_performances, log_weight: float = 0.0) -> None:
        self._acc.update(carrier_performances, log_weight)

    def should_stop(self) -> Tuple[bool, dict]:
        if self._acc.n < self.min_runs:
            return False, self._diagnostics()
        rhw = self._acc.relative_half_width()
        converged = bool(np.all(rhw <= self.rel_tol))
        return converged, self._diagnostics()

    def _diagnostics(self) -> dict:
        lo, hi = self._acc.confidence_interval()
        return {
            "n": self._acc.n,
            "ess": self._acc.ess,
            "mean": self._acc.mean,
            "std": self._acc.std,
            "ci_lower": lo,
            "ci_upper": hi,
            "rel_half_width": self._acc.relative_half_width(),
        }

    @property
    def stats(self) -> dict:
        return self._diagnostics()


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MCResult:
    """Summary statistics returned by ``MCEngine.run()``."""

    n_runs: int
    converged: bool

    # Per-carrier statistics (index 0=power, 1=heat, 2=gas)
    mean: np.ndarray
    std: np.ndarray
    ci_lower: np.ndarray  # 95 % CI lower bound
    ci_upper: np.ndarray  # 95 % CI upper bound
    rel_half_width: np.ndarray  # CI half-width / |mean|
    ess: float  # effective sample size (Kish 1965)

    # Per-run raw carrier vectors, shape (n_runs, 3)
    per_run: np.ndarray

    # E7: convergence trace. Each entry is a snapshot at a checkpoint:
    # (n_runs, mean[3], rel_half_width[3], ess). Populated per
    # CHECKPOINT_EVERY runs by the engine so downstream eval can plot
    # RHW(n) and Var-Reduction-Factor without re-running the simulation.
    convergence: Optional[np.ndarray] = None  # shape (n_checkpoints, 1+3+3+1)

    CARRIER_NAMES: List[str] = field(
        default_factory=lambda: ["power", "heat", "gas"], repr=False
    )

    def summary(self) -> str:
        hdr = (
            f"MCResult  n={self.n_runs}  ess={self.ess:.1f}"
            f"  converged={self.converged}\n"
            f"{'Carrier':<8} {'Mean':>10} {'Std':>10} "
            f"{'CI 95% lo':>12} {'CI 95% hi':>12} {'RHW':>8}\n" + "-" * 62
        )
        rows = "\n".join(
            f"{name:<8} {self.mean[i]:>10.4f} {self.std[i]:>10.4f} "
            f"{self.ci_lower[i]:>12.4f} {self.ci_upper[i]:>12.4f} "
            f"{self.rel_half_width[i]:>8.4f}"
            for i, name in enumerate(self.CARRIER_NAMES)
        )
        return hdr + "\n" + rows


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo engine
# ─────────────────────────────────────────────────────────────────────────────


class MCEngine:
    """RQMC Monte Carlo engine for MES resilience simulation.

    Always RQMC: pass a pre-built ``RQMCSampler`` via the ``sampler``
    parameter. ``run_func`` must accept a single ``FailureScenario`` and
    return an array-like of length ``n_carriers``::

        run_func(scenario: FailureScenario) -> array-like

    The engine generates Sobol scenarios, optionally pairs each with its
    antithetic twin (1 − u), and evaluates both.

    Parameters
    ----------
    rel_tol : float
        Convergence tolerance for the relative CI half-width (default 0.05).
    max_runs : int
        Hard upper bound on simulation runs (default 2000).
    min_runs : int
        Minimum runs before convergence is checked (default 200).
    antithetic_variates : bool
        Pair each Sobol point with its reflected antithetic twin.
    sampler : RQMCSampler
        Pre-configured RQMC sampler. Required.
    n_carriers : int
        Number of energy carriers (default 3: power, heat, gas).
    """

    def __init__(
        self,
        rel_tol: float = 0.05,
        max_runs: int = 2000,
        min_runs: int = 200,
        antithetic_variates: bool = True,
        sampler: Optional["RQMCSampler"] = None,
        n_carriers: int = 3,
    ):
        if sampler is None:
            raise ValueError("MCEngine requires an RQMCSampler.")
        self.rel_tol = rel_tol
        self.max_runs = max_runs
        self.min_runs = min_runs
        self.antithetic_variates = antithetic_variates
        self._sampler = sampler
        self.n_carriers = n_carriers

    # E7: how often to snapshot mean/RHW for the convergence trace stored
    # in MCResult.convergence. 100 keeps the trace small (~160 rows for
    # max_runs=2¹⁴) while still producing usable RHW(n) curves.
    CHECKPOINT_EVERY: ClassVar[int] = 100

    def run(self, run_func: Callable) -> MCResult:
        stopping = CIStoppingCriterion(
            rel_tol=self.rel_tol,
            min_runs=self.min_runs,
            n_carriers=self.n_carriers,
        )
        per_run: List[np.ndarray] = []
        # E7: convergence trace — recorded every CHECKPOINT_EVERY iterations
        # so the eval can plot RHW(n) without re-running the MC.
        trace: List[np.ndarray] = []
        converged = False
        i = 0
        # Pre-seed diagnostics so the post-loop warning has a valid `diag`
        # even when the loop never executes (max_runs=0, or sampler exhausted
        # on the first call).
        _, diag = stopping.should_stop()

        while i < self.max_runs:
            try:
                scenario = self._sampler.next_scenario()
            except StopIteration:
                log.warning(
                    "RQMCSampler exhausted after %d runs "
                    "(increase n_scenarios or max_runs).",
                    i,
                )
                break

            # ── Normal run ──────────────────────────────────────────────────
            carrier_perf = np.asarray(run_func(scenario), dtype=float)
            stopping.update(carrier_perf, scenario.log_weight)
            per_run.append(carrier_perf)
            i += 1

            # ── Antithetic run (same Sobol point, reflected) ─────────────────
            if self.antithetic_variates and i < self.max_runs:
                av = scenario.make_antithetic()
                carrier_perf_av = np.asarray(run_func(av), dtype=float)
                stopping.update(carrier_perf_av, av.log_weight)
                per_run.append(carrier_perf_av)
                i += 1

            # ── Convergence check ───────────────────────────────────────────
            converged, diag = stopping.should_stop()

            # E7 checkpoint: row layout is
            # [n, mean[0..2], rhw[0..2], ess]  (length = 1 + 3 + 3 + 1 = 8).
            if i % self.CHECKPOINT_EVERY == 0:
                trace.append(
                    np.concatenate(
                        [
                            np.array([float(i)]),
                            np.asarray(diag["mean"], dtype=float),
                            np.asarray(diag["rel_half_width"], dtype=float),
                            np.array([float(diag.get("ess", float(i)))]),
                        ]
                    )
                )
                rhw_str = " ".join(f"{v:.3f}" for v in diag["rel_half_width"])
                mean_str = " ".join(f"{v:.3f}" for v in diag["mean"])
                ess = diag.get("ess", float(i))
                log.info(
                    "n=%5d  ess=%6.1f  mean=[%s]  RHW=[%s]", i, ess, mean_str, rhw_str
                )

            if converged:
                log.info("Converged after %d runs (RHW ≤ %s)", i, self.rel_tol)
                break

        if not converged:
            log.warning(
                "Reached max_runs=%d without convergence.  RHW=%s",
                self.max_runs,
                diag["rel_half_width"],
            )

        _, final_diag = stopping.should_stop()
        # Always record the final state as the last checkpoint so the trace
        # is well-defined even if the loop converged between checkpoints.
        trace.append(
            np.concatenate(
                [
                    np.array([float(i)]),
                    np.asarray(final_diag["mean"], dtype=float),
                    np.asarray(final_diag["rel_half_width"], dtype=float),
                    np.array([float(final_diag.get("ess", float(i)))]),
                ]
            )
        )
        return MCResult(
            n_runs=i,
            converged=converged,
            mean=final_diag["mean"],
            std=final_diag["std"],
            ci_lower=final_diag["ci_lower"],
            ci_upper=final_diag["ci_upper"],
            rel_half_width=final_diag["rel_half_width"],
            ess=final_diag.get("ess", float(i)),
            per_run=np.array(per_run),
            convergence=np.asarray(trace) if trace else None,
        )
