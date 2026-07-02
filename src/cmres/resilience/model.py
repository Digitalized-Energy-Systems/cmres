import logging

import monee.model as mm
import numpy as np
import scipy.stats as stats
from monee import Network

import cmres.data.observer as observer
import cmres.omef.solver.monee as ms
from cmres.resilience.core import (
    Effect,
    Failure,
    ResilienceModel,
    StepModel,
)
from cmres.resilience.metric import (
    CascadingResilienceMetric,
    GeneralResiliencePerformanceMetric,
)

log = logging.getLogger(__name__)


FAIL_BASE_PROBABILITY_MAP = {
    mm.Source: 0.1,
    mm.Sink: 0.0,
    mm.PowerToGas: 0.1,
    mm.PowerToHeat: 0.1,
    mm.CHP: 0.1,
    mm.CHPHG: 0.1,
    mm.PowerToHeatHG: 0.1,
    mm.GasToHeatHG: 0.1,
    mm.GasToPower: 0.1,
    mm.PowerGenerator: 0.1,
    mm.HeatExchanger: 0.1,
    mm.HeatExchangerGenerator: 0.1,
    mm.HeatExchangerLoad: 0.1,
    mm.GenericPowerBranch: 0.1,
    mm.PowerLine: 0.1,
    mm.Trafo: 0.00,
    mm.WaterPipe: 0.1,
    mm.GasPipe: 0.1,
    mm.Bus: 0.00,
    mm.Junction: 0.00,
}


# Discrete probability per timestep: CDF difference with bell centred at midpoint of
# incident window, scale = window/4 so ±2σ covers the incident span.  Sums to ≈1 over
# the incident window and auto-scales when incident_timesteps changes.
def FAILURE_TIME_MODEL(incident_time_steps, time):
    return stats.norm.cdf(
        time + 0.5,
        loc=incident_time_steps / 2,
        scale=max(incident_time_steps / 4.0, 1.0),
    ) - stats.norm.cdf(
        time - 0.5,
        loc=incident_time_steps / 2,
        scale=max(incident_time_steps / 4.0, 1.0),
    )


# Numeric guard: clip uniform inputs away from 0/1 before norm.ppf
_U_EPS = 1e-8


def _ppf_normal(u, loc, scale):
    """Inverse-CDF of N(loc, scale) at clipped u ∈ (0,1)."""
    return float(stats.norm.ppf(np.clip(u, _U_EPS, 1.0 - _U_EPS), loc=loc, scale=scale))


class SimpleResilienceModel(ResilienceModel):
    def __init__(
        self,
        incident_shift=5,
        incident_timesteps=10,
        base_fail_probability_map=FAIL_BASE_PROBABILITY_MAP,
        time_model=FAILURE_TIME_MODEL,
        base_fail=0.3,
    ) -> None:
        self._base_fail = base_fail
        self._incident_shift = incident_shift
        self._incident_timesteps = incident_timesteps
        self._base_fail_probability_map = base_fail_probability_map
        self._time_model = lambda time: time_model(
            self._incident_timesteps + incident_shift, time
        )
        # Zero-truncation diagnostics. ``generate_failures`` always returns
        # at least one Failure: if no component triggers spontaneously the
        # highest-probability candidate is forced. The MC estimator that
        # consumes these scenarios is therefore conditional on "≥1 failure"
        # and biased upward by ~1/(1 − P(all-zero)). Tracking the fraction
        # of scenarios that needed the forced-failure path lets downstream
        # reporting surface this bias instead of hiding it.
        self.scenario_count: int = 0
        self.forced_failure_count: int = 0

    # ── Scenario-aware failure generation ─────────────────────────────────────

    def _eval_failure_scenario(self, comp, t_idx, time, registry, scenario):
        """Compute (fail_prob, triggered) from pre-sampled uniform inputs.

        Uses the FailureScenario's uniform[0,1] values for component *comp*
        at incident timestep *t_idx*, transforming each dimension:
          dim 0 → N(1, 0.1) probability multiplier (via norm.ppf)
          dim 1 → Bernoulli trigger (direct uniform comparison)

        Returns
        -------
        fail_prob : float
        triggered : bool
        """
        model_type = type(comp.model)
        if model_type not in self._base_fail_probability_map:
            return 0.0, False

        base_prob = self._base_fail_probability_map[model_type]
        cidx = registry.index_of(comp)
        if cidx is None:
            raise ValueError(
                f"Component {comp!r} (kind={type(comp).__name__}, id={comp.id}) "
                "not found in ComponentRegistry — registry is out of sync with "
                "the network."
            )
        if cidx >= scenario.uniforms.shape[0]:
            raise ValueError(
                f"Registry index {cidx} exceeds scenario.uniforms component "
                f"axis ({scenario.uniforms.shape[0]}). Scenario was built for "
                "a smaller registry."
            )
        if t_idx < 0 or t_idx >= scenario.uniforms.shape[1]:
            raise ValueError(
                f"t_idx={t_idx} out of scenario timestep range "
                f"[0, {scenario.uniforms.shape[1]})"
            )

        u = scenario.uniforms[cidx, t_idx]  # shape (N_DIM,)

        # dim 0: probability multiplier ~ N(1, 0.1), clipped to [0, 1]
        prob_mult = max(0.0, _ppf_normal(u[0], loc=1.0, scale=0.1))
        fail_prob = min(
            1.0, (base_prob * prob_mult * self._time_model(time) * self._base_fail)
        )
        # dim 1: Bernoulli trigger — no model, direct uniform comparison
        triggered = float(u[1]) < fail_prob
        return fail_prob, triggered

    def generate_failures(self, net: Network, registry=None, scenario=None):
        """Generate failure list for the network.

        Parameters
        ----------
        net : Network
        registry : ComponentRegistry
            Canonical component ordering built from the same *net*. Required.
        scenario : FailureScenario
            Pre-sampled uniform inputs from an RQMC sampler. Required. All
            stochastic failure decisions come from the scenario's Sobol-
            stratified uniforms.

        Notes
        -----
        At least one failure is always returned.  If no component triggers
        spontaneously, the component/timestep with the highest computed
        failure probability is forced to fail.  This keeps every MC scenario
        eventful while preserving the relative ranking of failure likelihoods.

        Statistical note: the forced-failure rule is a zero-truncation.
        The returned estimator is therefore the expectation *conditional on*
        at least one failure, which biases the unconditional mean upward
        by a factor ≲ 1/(1 − P(all-zero)).  For the priors used here,
        P(all-zero) ≪ 1 per scenario and the bias is numerically negligible.
        Remove the guaranteed-failure branch below to recover an unbiased
        estimator of the unconditional mean.
        """
        if registry is None or scenario is None:
            raise ValueError(
                "generate_failures requires both registry and scenario "
                "(legacy global-RNG mode has been removed)."
            )
        failures = []
        # Track components that already have an unresolved failure so that a
        # single component is not faulted twice before its repair time is set.
        # Keyed by (kind, comp.id) — int comp.id spaces overlap across nodes,
        # children and compounds (e.g. compound id 0..16 collides with child id
        # 0..16 in the simbench grids), and a flat set of ints would silently
        # skip a compound trigger whenever an earlier node or child with the
        # same int id had already failed in this scenario.
        already_failed: set = set()
        # Fallback for the guaranteed-failure mechanism: (fail_prob, comp, time)
        best_candidate = None

        for i in range(self._incident_timesteps):
            time = i + self._incident_shift

            for node in net.nodes:
                key = ("node", node.id)
                if not node.independent or key in already_failed:
                    continue
                fail_prob, triggered = self._eval_failure_scenario(
                    node, i, time, registry, scenario
                )
                if triggered:
                    failures.append(Failure(time, node, fail_prob, Effect.DEAD))
                    already_failed.add(key)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, node, time)

            for branch in net.branches:
                key = ("branch", branch.id)
                # HeatExchanger is failable despite independent=False — the
                # MC registry (mc.py) allocates Sobol dimensions for it and
                # FAIL_BASE_PROBABILITY_MAP prices it at 0.1; skipping it
                # here made those dimensions dead weight and left every HX
                # with an unmeasured (NaN) impact.
                if (
                    not branch.independent
                    and type(branch.model) is not mm.HeatExchanger
                ) or key in already_failed:
                    continue
                fail_prob, triggered = self._eval_failure_scenario(
                    branch, i, time, registry, scenario
                )
                if triggered:
                    failures.append(Failure(time, branch, fail_prob, Effect.DEAD))
                    already_failed.add(key)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, branch, time)

            for child in net.childs:
                key = ("child", child.id)
                if not child.independent or key in already_failed:
                    continue
                fail_prob, triggered = self._eval_failure_scenario(
                    child, i, time, registry, scenario
                )
                if triggered:
                    failures.append(Failure(time, child, fail_prob, Effect.DEAD))
                    already_failed.add(key)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, child, time)

            for compound in net.compounds:
                key = ("compound", compound.id)
                if not compound.independent or key in already_failed:
                    continue
                fail_prob, triggered = self._eval_failure_scenario(
                    compound, i, time, registry, scenario
                )
                if triggered:
                    failures.append(Failure(time, compound, fail_prob, Effect.DEAD))
                    already_failed.add(key)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, compound, time)

        # Guarantee at least one failure per scenario.
        self.scenario_count += 1
        if not failures and best_candidate is not None:
            fp, comp, time = best_candidate
            log.debug(
                "No spontaneous failures; forcing failure on %s at t=%d (p=%.4f).",
                comp,
                time,
                fp,
            )
            failures.append(Failure(time, comp, fp, Effect.DEAD))
            self.forced_failure_count += 1

        return failures


# Substrings that indicate a Pyomo solve aborted on a resource limit
# (time, iteration, …) and returned a *witness* incumbent rather than a
# converged solution. We pattern-match the stringified
# ``solver_status`` / ``termination_condition`` instead of importing
# pyomo enums so this module stays decoupled from monee's solver
# backend.
_ABORT_INDICATORS = (
    "limit",      # ``maxTimeLimit``, ``maxIterations``, ``minStepLength`` …
    "aborted",    # ``SolverStatus.aborted``
    "interrupt",  # ``userInterrupt``
)


def _solver_aborted_with_witness(sresult) -> bool:
    """Return True if a ``SolverResult`` with ``success=True`` actually
    came from a non-converged abort (time-limit, iteration-limit, etc.).

    Used by :class:`CascadingModel` to drop such samples from the MC
    estimator — the witness incumbent's load shed is not comparable to
    the converged samples and would otherwise pull the mean toward
    whatever Gurobi happened to have when it cut off.
    """
    tc = getattr(sresult, "termination_condition", None) or ""
    status = getattr(sresult, "solver_status", None) or ""
    combined = (str(tc) + " " + str(status)).lower()
    return any(tag in combined for tag in _ABORT_INDICATORS)


class CascadingModel(StepModel):
    """Per-step performance evaluator with infeasibility-fallback shedding.

    Despite the name, no cascading repair/failure logic is currently active —
    the prior ``process_network_state`` machinery has been removed. The
    model still exposes ``performance`` per step via the observer; the
    ``performance_after_cascade`` channel is no longer emitted because it
    used to be a copy of ``performance``.
    """

    def __init__(
        self,
        performance_accuracy=100,
        ext_grid_el_bounds=ms.DEFAULT_EXT_GRID_EL_BOUNDS,
        ext_grid_gas_bounds=ms.DEFAULT_EXT_GRID_GAS_BOUNDS,
        ext_grid_heat_bounds=ms.DEFAULT_EXT_GRID_HEAT_BOUNDS,
        include_coupling_points=False,
    ) -> None:
        self._cascading_metric = CascadingResilienceMetric()
        self._performance_metric = GeneralResiliencePerformanceMetric()
        self._iteration_number_omef = performance_accuracy
        self._faults = None
        self._last_performance = None
        self._ext_grid_el_bounds = ext_grid_el_bounds
        self._ext_grid_gas_bounds = ext_grid_gas_bounds
        self._ext_grid_heat_bounds = ext_grid_heat_bounds
        self._include_coupling_points = include_coupling_points

    def calc_performance(self, network: Network, without_load=False):
        log.info("Solving network for performance calculation")
        result = ms.solve(
            network,
            ext_grid_el_bounds=self._ext_grid_el_bounds,
            ext_grid_gas_bounds=self._ext_grid_gas_bounds,
            ext_grid_heat_bounds=self._ext_grid_heat_bounds,
            include_coupling_points=self._include_coupling_points,
        )
        log.info("Network solve complete")
        # The performance metric always counts end-user shed only:
        # ``include_coupling_points`` steers the shed *objective* above
        # (resolving degenerate ties toward running load-bearing CPs), but a
        # curtailed CP's service loss is already captured as downstream
        # unserved load — counting its input draw too would double-count and
        # make the additive and load-bearing families incomparable.
        return (
            self._performance_metric.calc(
                result.network,
                include_coupling_points=False,
            ),
            result,
        )

    def fault_delta_exists(self, step):
        for fault in self._faults:
            if fault.start_time == step:
                return True
        return False

    def _max_load_shedding(self, net):
        """Return (power_MW, heat_MW, gas_MW) assuming all active load is shed.

        Mirrors the load-types accounted for by ``GeneralResiliencePerformanceMetric``
        in ``monee.problem.metric`` so the solver-failure fallback and the
        successful-solve metric measure the same thing. Specifically:
          - power: PowerLoad.p_mw
          - heat:  HeatLoad.q_mw_heat  +  HeatExchangerLoad.q_mw  +  PassiveHeatExchangerLoad.q_mw
          - gas:   |Sink.mass_flow_kgs| × 3.6 × HHV  (kg/s × kWh/kg → MW)

        Earlier this counted only HeatExchangerLoad, so on simbench LV grids
        (which use HeatLoad children) the fallback reported heat=0 even when
        all heat load was effectively shed.
        """
        passive_hx = getattr(mm, "PassiveHeatExchangerLoad", mm.HeatExchangerLoad)

        power = sum(
            mm.upper(c.model.p_mw)
            for c in net.childs
            if isinstance(c.model, mm.PowerLoad) and c.active and not c.ignored
        )
        heat = 0.0
        for c in net.childs:
            m = c.model
            if not c.active or c.ignored:
                continue
            if isinstance(m, mm.HeatLoad):
                heat += mm.upper(m.q_mw_heat)
            elif isinstance(m, (mm.HeatExchangerLoad, passive_hx)):
                heat += mm.upper(m.q_mw)
        for b in net.branches:
            if not b.active or b.ignored:
                continue
            if isinstance(b.model, (mm.HeatExchangerLoad, passive_hx)):
                heat += mm.upper(b.model.q_mw)
        gas = sum(
            abs(mm.upper(c.model.mass_flow_kgs)) * 3.6 * c.grid.higher_heating_value_kwh_per_kg
            for c in net.childs
            if isinstance(c.model, mm.Sink)
            and hasattr(c.grid, "higher_heating_value_kwh_per_kg")
            and c.active
            and not c.ignored
        )

        return (power, heat, gas)

    def step(self, net, step, step_state, step_result, base_net):

        log.debug("Starting step %s", step)

        if self.fault_delta_exists(step) or self._last_performance is None:
            sresult = None
            try:
                performance, sresult = self.calc_performance(net)
            except Exception:
                active_faults = [
                    str(f) for f in (self._faults or []) if f.start_time <= step
                ]
                log.exception(
                    "calc_performance raised at step=%d; falling back to "
                    "max-load-shedding. Active faults so far: %s",
                    step,
                    active_faults,
                )
                observer.gather(
                    "infeasibility",
                    {
                        "step": step,
                        "kind": "exception",
                        "n_active_faults": len(active_faults),
                        "active_faults": " | ".join(active_faults),
                        "report": "",
                    },
                )
                performance = self._max_load_shedding(net)
            else:
                if not getattr(sresult, "success", True):
                    active_faults = [
                        str(f) for f in (self._faults or []) if f.start_time <= step
                    ]
                    report = ""
                    rep_obj = getattr(sresult, "infeasibility_report", None)
                    if rep_obj is not None:
                        try:
                            report = rep_obj.summary(max_items=20)
                        except Exception:
                            report = repr(rep_obj)
                    log.warning(
                        "Solver returned infeasible at step=%d "
                        "(active faults=%d); falling back to max-load-shedding.",
                        step,
                        len(active_faults),
                    )
                    observer.gather(
                        "infeasibility",
                        {
                            "step": step,
                            "kind": "infeasible",
                            "n_active_faults": len(active_faults),
                            "active_faults": " | ".join(active_faults),
                            "report": report,
                        },
                    )
                    performance = self._max_load_shedding(net)
                elif _solver_aborted_with_witness(sresult):
                    # Gurobi (or any Pyomo solver) hit its TimeLimit and
                    # returned a non-converged witness incumbent. The
                    # witness's shed value is not comparable to the
                    # converged samples — average it in and the MC mean
                    # quietly drifts toward whatever Gurobi happened to
                    # have at the cutoff. Poison the run with NaN
                    # performance so the MC accumulator drops the sample
                    # via its existing non-finite skip path (and the
                    # ``n_skipped_nonfinite`` counter on ``MCResult``
                    # surfaces the count in mc_summary.txt / mc_result.npz).
                    active_faults = [
                        str(f) for f in (self._faults or []) if f.start_time <= step
                    ]
                    tc = getattr(sresult, "termination_condition", None)
                    status = getattr(sresult, "solver_status", None)
                    log.warning(
                        "Solver aborted (status=%s, termination=%s) at step=%d "
                        "(active faults=%d); dropping this MC sample (NaN "
                        "performance).",
                        status, tc, step, len(active_faults),
                    )
                    observer.gather(
                        "infeasibility",
                        {
                            "step": step,
                            "kind": "time_limit",
                            "n_active_faults": len(active_faults),
                            "active_faults": " | ".join(active_faults),
                            "report": (
                                f"solver_status={status}; "
                                f"termination_condition={tc}"
                            ),
                        },
                    )
                    performance = (float("nan"), float("nan"), float("nan"))
            self._last_performance = performance
        else:
            performance = self._last_performance

        observer.gather("performance", performance)

        self._cascading_metric.gather(net, step, performance=performance)

    def calc_metric(self):
        return self._cascading_metric.calc()
