import logging
import traceback
from typing import List

import monee.model as mm
import numpy as np
import scipy.stats as stats
from monee import Network

import cmres.data.observer as observer
import cmres.omef.solver.monee as ms
from cmres.resilience.core import (
    Effect,
    Failure,
    RepairModel,
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


# Antithetic counterparts of the default normal-draw models.
# For X ~ N(μ, σ):  antithetic = 2μ − X  (reflects about the mean).
def ANTITHETIC_FAILURE_PROBABILITY_MODEL(base_prob):
    return base_prob * (2.0 - np.random.normal(1, scale=0.1))


def ANTITHETIC_DMG_COEFF_VARIANCE_MODEL(dmg_coeff):
    return dmg_coeff * (2.0 - np.random.normal(1, scale=0.1))


FAIL_BASE_PROBABILITY_MAP = {
    mm.Source: 0.1,
    mm.Sink: 0.0,
    mm.PowerToGas: 0.1,
    mm.PowerToHeat: 0.1,
    mm.CHP: 0.1,
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
        heat_impact=0.1,
        gas_impact=0.1,
        power_impact=5,
        mes_impact=1,
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
        self._heat_impact = heat_impact
        self._gas_impact = gas_impact
        self._power_impact = power_impact
        self._mes_impact = mes_impact
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

    def _read_impact(self, component):
        if (
            not hasattr(component, "grid")
            or component.grid is None
            or type(component.grid) is dict
        ):
            return self._mes_impact

        if component.grid.name == "water":
            return self._heat_impact
        elif component.grid.name == "power":
            return self._power_impact
        elif component.grid.name == "gas":
            return self._gas_impact

        return self._mes_impact

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
                    failures.append(Failure(time, node, fail_prob, Effect.DEAD, -1))
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
                    failures.append(Failure(time, branch, fail_prob, Effect.DEAD, -1))
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
                    failures.append(Failure(time, child, fail_prob, Effect.DEAD, -1))
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
                    failures.append(Failure(time, compound, fail_prob, Effect.DEAD, -1))
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
            failures.append(Failure(time, comp, fp, Effect.DEAD, -1))

        return failures


DMG_COEFF_FUNC_MAP = {
    mm.Source: lambda model: mm.upper(model.mass_flow),
    mm.Sink: lambda model: mm.upper(model.mass_flow),
    mm.PowerToGas: lambda model: mm.upper(model.to_mass_flow),
    mm.PowerToHeat: lambda model: mm.upper(model.heat_energy_w),
    mm.CHP: lambda model: mm.upper(model.mass_flow),
    mm.GasToPower: lambda model: mm.upper(model.from_mass_flow),
    mm.PowerGenerator: lambda model: mm.upper(model.p_mw),
    mm.HeatExchanger: lambda model: mm.upper(model.q_w),
    mm.HeatExchangerGenerator: lambda model: mm.upper(model.q_w),
    mm.HeatExchangerLoad: lambda model: mm.upper(model.q_w),
    mm.GenericPowerBranch: lambda model: model.br_r,
    mm.PowerLine: lambda model: model.br_r,
    mm.Trafo: lambda _: 1,
    mm.WaterPipe: lambda model: model.length_m,
    mm.GasPipe: lambda model: model.length_m,
    mm.Bus: lambda _: 1,
    mm.Junction: lambda _: 1,
}


def DMG_COEFF_VARIANCE_MODEL(dmg_coeff):
    return dmg_coeff * np.random.normal(1, scale=0.1)


class SimpleRepairModel(RepairModel):
    def __init__(
        self,
        delay_for_repair=10,
        dmg_coeff_func_map=DMG_COEFF_FUNC_MAP,
        dmg_coeff_variance_model=DMG_COEFF_VARIANCE_MODEL,
        incident_timesteps=10,
        incident_shift=0,
        antithetic: bool = False,
    ) -> None:
        self._dmg_coeff_func_map = dmg_coeff_func_map
        self._antithetic = antithetic
        # Antithetic mode: replace normal-draw model with its reflection.
        if antithetic and dmg_coeff_variance_model is DMG_COEFF_VARIANCE_MODEL:
            self._dmg_coeff_variance_model = ANTITHETIC_DMG_COEFF_VARIANCE_MODEL
        else:
            self._dmg_coeff_variance_model = dmg_coeff_variance_model
        self._delay_for_repair = delay_for_repair
        self._incident_timesteps = incident_timesteps
        self._incident_shift = incident_shift

    def generate_repairs(
        self,
        _,
        failures: List[Failure],
        registry=None,
        scenario=None,
    ):
        """Assign repaired_time to each failure.

        Parameters
        ----------
        registry : ComponentRegistry | None
        scenario : FailureScenario | None
            When provided, repair time and damage coefficient are derived from
            pre-sampled uniforms (dims 2 and 3) rather than the global RNG.
        """
        use_scenario = registry is not None and scenario is not None

        for failure in failures:
            f: Failure = failure
            if type(f.component.model) not in self._dmg_coeff_func_map:
                raise Exception(
                    f"There is no dmg coeff defined for {type(f.component.model)}!"
                )
            raw_dmg_coeff = self._dmg_coeff_func_map[type(f.component.model)](
                f.component.model
            )

            if use_scenario:
                cidx = registry.index_of(f.component)
                # Map failure simulation time back to incident array index:
                # f.time = i + incident_shift  →  t_idx = i = f.time − incident_shift
                raw_t_idx = f.time - self._incident_shift
                t_max = scenario.uniforms.shape[1] - 1
                t_idx = max(0, min(raw_t_idx, t_max))
                if t_idx != raw_t_idx:
                    import warnings

                    warnings.warn(
                        f"SimpleRepairModel: repair t_idx={raw_t_idx} clamped to "
                        f"[0, {t_max}]; scenario shape may be too small for "
                        f"the configured incident window.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                # Clamp: if cidx is valid, use scenario; else fall through to RNG
                if cidx is not None and cidx < scenario.uniforms.shape[0]:
                    u = scenario.uniforms[cidx, t_idx]  # shape (N_DIM,)

                    # dim 2: repair base time ~ N(5, 5)
                    base_time = _ppf_normal(u[2], loc=5.0, scale=5.0)
                    # dim 3: damage coefficient multiplier ~ N(1, 0.1), clipped ≥ 0
                    dmg_mult = max(0.0, _ppf_normal(u[3], loc=1.0, scale=0.1))

                    dmg = dmg_mult * raw_dmg_coeff * f.severity
                    time_needed = max(
                        base_time + self._delay_for_repair + dmg / 10,
                        self._incident_timesteps,
                    )
                    f.repaired_time = int(f.time + time_needed)
                    continue

            # ── Legacy RNG path ──────────────────────────────────────────────
            dmg = self._dmg_coeff_variance_model(raw_dmg_coeff) * f.severity
            base_time = np.random.normal(5, 5)
            if self._antithetic:
                base_time = 10.0 - base_time
            time_needed = max(
                base_time + self._delay_for_repair + dmg / 10,
                self._incident_timesteps,
            )
            f.repaired_time = int(f.time + time_needed)

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
    def __init__(self, performance_accuracy=100) -> None:
        self._cascading_metric = CascadingResilienceMetric()
        self._performance_metric = GeneralResiliencePerformanceMetric()
        self._current_failures = []
        self._iteration_number_omef = performance_accuracy
        self._faults = None
        self._last_performance = None

    def calc_performance(self, network: Network, without_load=False):
        log.debug("Solving network for performance calculation")
        # print(network)
        result = ms.solve(network)
        log.debug("Network solve complete")
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
            if fault.start_time == step or fault.stop_time == step:
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
            mm.upper(c.model.q_w) / 1e6
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
            try:
                performance, _ = self.calc_performance(net)
            except Exception:
                print(traceback.format_exc())

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
