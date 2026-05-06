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
from cmres.resilience.fault import name_of
from cmres.resilience.metric import (
    CascadingResilienceMetric,
    GeneralResiliencePerformanceMetric,
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Antithetic-variate fallbacks (used in legacy / no-scenario mode only)
# ─────────────────────────────────────────────────────────────────────────────


# Antithetic counterpart of the default normal-draw failure model.
# For X ~ N(μ, σ):  antithetic = 2μ − X  (reflects about the mean).
def ANTITHETIC_FAILURE_PROBABILITY_MODEL(base_prob):
    return base_prob * (2.0 - np.random.normal(1, scale=0.1))


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


def FAILURE_PROBABILITY_MODEL(base_prob):
    return max(0.0, min(1.0, base_prob * np.random.normal(1, scale=0.1)))


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


def FAILURE_SPATIAL_MODEL(coords):
    return 1


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
        fail_probability_model=FAILURE_PROBABILITY_MODEL,
        time_model=FAILURE_TIME_MODEL,
        spatial_model=FAILURE_SPATIAL_MODEL,
        base_fail=0.3,
        antithetic: bool = False,
    ) -> None:
        self._base_fail = base_fail
        self._incident_shift = incident_shift
        self._incident_timesteps = incident_timesteps
        self._base_fail_probability_map = base_fail_probability_map
        self._antithetic = antithetic
        # Antithetic mode: replace the default normal-draw model with its
        # reflection (2μ − X) so that paired runs are negatively correlated.
        if antithetic and fail_probability_model is FAILURE_PROBABILITY_MODEL:
            self._fail_probability_model = ANTITHETIC_FAILURE_PROBABILITY_MODEL
        else:
            self._fail_probability_model = fail_probability_model
        self._time_model = lambda time: time_model(
            self._incident_timesteps + incident_shift, time
        )
        self._spatial_model = lambda coords: (
            spatial_model(coords) if coords is not None else 1
        )

    def calc_fail(self, network: Network, component, time):
        model_type = type(component.model)
        if model_type not in self._base_fail_probability_map:
            return 0
        base_failure_probability = self._base_fail_probability_map[model_type]
        return (
            self._fail_probability_model(base_failure_probability)
            * self._time_model(time)
            * self._base_fail
        )

    def _bernoulli(self, p: float) -> bool:
        """Bernoulli trial respecting the antithetic-variates flag.

        Normal run:     fail if U  < p   (U ~ Uniform(0,1))
        Antithetic run: fail if 1−U < p, i.e. U > 1−p

        Uses numpy's RNG throughout so that antithetic pairing (via np.random.seed)
        is consistent with the normal-draw probability multiplier in the same stream.
        """
        u = np.random.random()
        return (1.0 - u) < p if self._antithetic else u < p

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
        if cidx is None or cidx >= scenario.uniforms.shape[0]:
            # Component not in registry — fall back to legacy RNG
            fp = self.calc_fail(None, comp, time)
            return fp, self._bernoulli(fp)

        if t_idx < 0 or t_idx >= scenario.uniforms.shape[1]:
            # Timestep out of scenario range — fall back to legacy RNG
            fp = self.calc_fail(None, comp, time)
            return fp, self._bernoulli(fp)

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
        registry : ComponentRegistry | None
            Canonical component ordering built from the same *net*.  Required
            when *scenario* is provided.
        scenario : FailureScenario | None
            Pre-sampled uniform inputs.  When provided (alongside *registry*),
            all random draws come from the scenario's Sobol-stratified uniforms
            rather than the global RNG — enabling true RQMC variance reduction.
            When ``None``, falls back to the legacy global-RNG path.

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
        use_scenario = registry is not None and scenario is not None
        failures = []
        # Track components that already have an unresolved failure so that a
        # single component is not faulted twice before its repair time is set.
        already_failed: set = set()
        # Fallback for the guaranteed-failure mechanism: (fail_prob, comp, time)
        best_candidate = None

        for i in range(self._incident_timesteps):
            time = i + self._incident_shift

            for node in net.nodes:
                if not node.independent or node.id in already_failed:
                    continue
                if use_scenario:
                    fail_prob, triggered = self._eval_failure_scenario(
                        node, i, time, registry, scenario
                    )
                else:
                    fail_prob = self.calc_fail(net, node, time)
                    triggered = self._bernoulli(fail_prob)
                if triggered:
                    failures.append(Failure(time, node, fail_prob, Effect.DEAD))
                    already_failed.add(node.id)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, node, time)

            for branch in net.branches:
                if not branch.independent or branch.id in already_failed:
                    continue
                if use_scenario:
                    fail_prob, triggered = self._eval_failure_scenario(
                        branch, i, time, registry, scenario
                    )
                else:
                    fail_prob = self.calc_fail(net, branch, time)
                    triggered = self._bernoulli(fail_prob)
                if triggered:
                    failures.append(Failure(time, branch, fail_prob, Effect.DEAD))
                    already_failed.add(branch.id)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, branch, time)

            for child in net.childs:
                if not child.independent or child.id in already_failed:
                    continue
                if use_scenario:
                    fail_prob, triggered = self._eval_failure_scenario(
                        child, i, time, registry, scenario
                    )
                else:
                    fail_prob = self.calc_fail(net, child, time)
                    triggered = self._bernoulli(fail_prob)
                if triggered:
                    failures.append(Failure(time, child, fail_prob, Effect.DEAD))
                    already_failed.add(child.id)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, child, time)

            for compound in net.compounds:
                if not compound.independent or compound.id in already_failed:
                    continue
                if use_scenario:
                    fail_prob, triggered = self._eval_failure_scenario(
                        compound, i, time, registry, scenario
                    )
                else:
                    fail_prob = self.calc_fail(net, compound, time)
                    triggered = self._bernoulli(fail_prob)
                if triggered:
                    failures.append(Failure(time, compound, fail_prob, Effect.DEAD))
                    already_failed.add(compound.id)
                elif fail_prob > 0 and (
                    best_candidate is None or fail_prob > best_candidate[0]
                ):
                    best_candidate = (fail_prob, compound, time)

        # Guarantee at least one failure per scenario.
        if not failures and best_candidate is not None:
            fp, comp, time = best_candidate
            log.debug(
                "No spontaneous failures; forcing failure on %s at t=%d (p=%.4f).",
                comp,
                time,
                fp,
            )
            failures.append(Failure(time, comp, fp, Effect.DEAD))

        return failures


def to_failure_probability(relative_violation, ramp=0, steepness=1, exponent=1.5):
    return steepness * ((relative_violation - ramp) * 100) ** exponent / 100


def deactivate_node(network: Network, node):
    for child_id in node.child_ids:
        child = network.child_by_id(child_id)
        network.deactivate(child)


def activate_node(network: Network, node):
    for child_id in node.child_ids:
        child = network.child_by_id(child_id)
        network.activate(child)


def calc_relative_violation(component, attribute, target, rel_allowed_diff):
    return max(
        (
            abs((mm.value(getattr(component.model, attribute)) - target))
            - rel_allowed_diff * target
        )
        / (target * rel_allowed_diff),
        0,
    )


class CascadingModel(StepModel):
    def __init__(
        self,
        performance_accuracy=100,
        ext_grid_el_bounds=ms.DEFAULT_EXT_GRID_EL_BOUNDS,
        ext_grid_gas_bounds=ms.DEFAULT_EXT_GRID_GAS_BOUNDS,
        ext_grid_heat_bounds=ms.DEFAULT_EXT_GRID_HEAT_BOUNDS,
    ) -> None:
        self._cascading_metric = CascadingResilienceMetric()
        self._performance_metric = GeneralResiliencePerformanceMetric()
        self._current_failures = []
        self._iteration_number_omef = performance_accuracy
        self._faults = None
        self._last_performance = None
        self._ext_grid_el_bounds = ext_grid_el_bounds
        self._ext_grid_gas_bounds = ext_grid_gas_bounds
        self._ext_grid_heat_bounds = ext_grid_heat_bounds

    def calc_performance(self, network: Network, without_load=False):
        log.info("Solving network for performance calculation")
        # print(network)
        result = ms.solve(
            network,
            ext_grid_el_bounds=self._ext_grid_el_bounds,
            ext_grid_gas_bounds=self._ext_grid_gas_bounds,
            ext_grid_heat_bounds=self._ext_grid_heat_bounds,
        )
        log.info("Network solve complete")
        return self._performance_metric.calc(result.network), result

    def check_repairs(self, network, bound_tuple, step, network_name):
        attribute, target, allowed_diff = bound_tuple
        for i in range(len(self._current_failures) - 1, -1, -1):
            node, failure = self._current_failures[i]
            if node.grid.name != network_name or attribute not in node.values_as_dict():
                continue
            if step >= failure["step"] + failure["min_duration"]:
                relative_violation = to_failure_probability(
                    calc_relative_violation(node, attribute, target, allowed_diff)
                )
                if relative_violation < np.random.random():
                    activate_node(network, node)
                    del self._current_failures[i]
                    observer.gather(
                        "cascading repair",
                        {
                            "step": step,
                            "node": name_of(node),
                            "probability": relative_violation,
                            "min_duration": -1,
                            "type": "repair",
                        },
                    )

    def process_node(self, network: Network, node, bound_tuple, step):
        attribute, target, allowed_diff = bound_tuple
        relative_violation = to_failure_probability(
            calc_relative_violation(node, attribute, target, allowed_diff)
        )
        if relative_violation > np.random.random():
            min_duration = int(relative_violation * np.random.random() * 10)
            deactivate_node(network, node)
            failure_description = {
                "step": step,
                "node": name_of(node),
                "probability": relative_violation,
                "min_duration": min_duration,
                "type": "failure",
            }
            self._current_failures.append((node, failure_description))

            observer.gather(
                "cascading failure",
                failure_description,
            )

    def process_network_state(self, network: Network, step):
        self.check_repairs(network, ms.BOUND_GAS, step, "gas")
        self.check_repairs(network, ms.BOUND_HEAT, step, "heat")
        self.check_repairs(network, ms.BOUND_EL, step, "power")

        for node in network.nodes:
            if not node.independent:
                continue
            if node.grid.name == "gas":
                self.process_node(network, node, ms.BOUND_GAS, step)
            if node.grid.name == "heat":
                self.process_node(network, node, ms.BOUND_HEAT, step)
            if node.grid.name == "power":
                self.process_node(network, node, ms.BOUND_EL, step)

        for branch in network.branches_by_type(mm.GenericPowerBranch):
            if not branch.independent:
                continue
            self.process_node(network, branch, ms.BOUND_LP, step)

    def fault_delta_exists(self, step):
        for fault in self._faults:
            if fault.start_time == step:
                return True
        return False

    @staticmethod
    def _max_load_shedding(net):
        """Return (power_MW, heat_MW, gas_MW) assuming all active load is shed."""
        power = sum(
            mm.upper(c.model.p_mw)
            for c in net.childs
            if isinstance(c.model, mm.PowerLoad) and c.active and not c.ignored
        )
        heat = sum(
            mm.upper(c.model.q_mw)
            for c in net.childs + net.branches
            if isinstance(c.model, mm.HeatExchangerLoad) and c.active and not c.ignored
        )
        gas = sum(
            mm.upper(c.model.mass_flow) * 3.6 * c.grid.higher_heating_value
            for c in net.childs
            if isinstance(c.model, mm.Sink)
            and hasattr(c.grid, "higher_heating_value")
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
            self._last_performance = performance
        else:
            performance = self._last_performance

        observer.gather("performance", performance)

        """
        self.process_network_state(net, step)

        # calc performance and base performance
        performance_after_cascade, _ = self.calc_performance(net)
        """
        observer.gather("performance_after_cascade", performance)

        self._cascading_metric.gather(
            net,
            step,
            performance=performance,
            performance_after_cascade=performance,
        )

    def calc_metric(self):
        return self._cascading_metric.calc()
