"""
Implementation of MGBM stopping criteria.

@article{marti2016stopping,
  title={A stopping criterion for multi-objective optimization evolutionary algorithms},
  author={Mart{'\i}, Luis and Garc{'\i}a, Jes{\'u}s and Berlanga, Antonio and Molina, Jos{'e} M},
  journal={Information Sciences},
  volume={367},
  pages={700--718},
  year={2016},
  publisher={Elsevier}
}
"""

import copy

from interface import implements, Interface
from overrides import overrides
from pykalman import KalmanFilter
import numpy as np


class EvidenceGatherer(Interface):
    """
    Searches for evidences to stop an algorithm. For this
    evidences it uses indicators of ProgressIndicator implementations.
    """

    def gather_evidence(self, indicator_values):
        """
        Calculates a specific evidence based on the indicator_values.

        :param indicator_values: list of indicator values
        :returns: calculated evidence that the algorithm should stop (or not)
        """
        pass


class ProgressIndicator(Interface):
    """
    Calculates a measure of progress of a population-based algorithm
    """

    def compute_indicator(self, population):
        """
        Calculates the indicator.

        :param populaion: population of the current iteration
        :returns: measure
        """
        pass


class StopDecision(Interface):
    """
    Decides if a given situation is met with the help of evidences.
    """

    def decide_stop(self, evidences):
        """
        :param evidences: evidences of the current progress
        :returns: true if the decision decided to stop the progress, false otherwise
        """
        pass


class StopCriterion:
    """
    StopCriterion consisting of indicators gatherers and decisions. Implementing
    the stop decision pipeline.
    """

    def __init__(self, indicators, gatherers, decisions):
        """
        :param indicators: list of ProgressIndicator
        :param gatheres: list of EvidenceGatherer
        :param decisions: list of StopDecision
        """
        self.indicators = indicators
        self.gatherers = gatherers
        self.decisions = decisions

    def stop(self, performances):
        """
        Calculated the indicator-values feed them into the gatherers and feed their result
        to the stop decisions. If all stop decision vote for stop (return True) this method
        will result true too, otherwise its false.

        :param performances: performances
        :returns: True if all decisions voting for true, false otherwise
        """
        indicator_values = [
            indicator.compute_indicator(performances) for indicator in self.indicators
        ]
        evidences = [
            gatherer.gather_evidence(indicator_values) for gatherer in self.gatherers
        ]
        decisions = [decision.decide_stop(evidences) for decision in self.decisions]

        return all(decisions)


class PerformanceVarianceIndicator(implements(ProgressIndicator)):
    """
    Variance over last
    """

    def __init__(self, backsight):
        self.__backsight = backsight

    @overrides
    def compute_indicator(self, population):
        return np.var(population[0 : min(self.__backsight, len(population))]) / 10


class IterationCountIndicator(implements(ProgressIndicator)):
    """
    Count iteration number
    """

    def __init__(self):
        self.__count = -1

    @overrides
    def compute_indicator(self, population):
        self.__count += 1
        return self.__count

    def increase_count(self, inc):
        self.__count += inc


class KalmanEvidenceGatherer(implements(EvidenceGatherer)):
    """
    Implementation of a kalman based evidence gathering. Based on the
    indicator-value-history it will predict how stable the values tend to
    become 0.
    """

    def __init__(self, r):
        """
        :param r: noise for the kalman filter
        """
        self.__r = r
        self.__train = [1]

    def new_filter(self):
        """
        Create new freshly initialized kalman filter.

        :returns: KalmanFilter
        """
        return KalmanFilter(
            transition_matrices=[1.0],
            observation_matrices=None,
            transition_covariance=[0.0],
            observation_covariance=[self.__r],
            transition_offsets=[0.0],
            observation_offsets=None,
            initial_state_mean=None,
            initial_state_covariance=None,
            random_state=None,
            em_vars=[
                "transition_covariance",
                "observation_covariance",
                "initial_state_mean",
                "initial_state_covariance",
            ],
            n_dim_state=None,
            n_dim_obs=None,
        )

    @overrides
    def gather_evidence(self, indicator_values):
        self.__train += indicator_values
        return self.new_filter().em(self.__train).smooth(indicator_values)[0].max()


class PassThroughGatherer(implements(EvidenceGatherer)):
    """
    Just returns the first indicator value
    """

    @overrides
    def gather_evidence(self, indicator_values):
        return indicator_values[0]


class ThresholdDecision(implements(StopDecision)):
    """
    Threshold Decision for a StopCriterion. This class will check if all evidences are
    below a given threshold.
    """

    def __init__(self, threshold, compare_reverse=False):
        """
        :param threshold: threshold for the evidences
        """
        self.threshold = threshold
        self.__reverse = compare_reverse

    @overrides
    def decide_stop(self, evidences):
        return all(
            [
                evidence > self.threshold
                if self.__reverse
                else evidence < self.threshold
                for evidence in evidences
            ]
        )


def createMGBMStoppingCriterion(threshold, R, backsight) -> StopCriterion:
    """
    Create the stopping criterion MGBM for multiobjective optimization.

    :param threshold: threshold of the kalman output when the criteria will signal stop
    :param R: noise for the kalman-filter
    """
    decision = ThresholdDecision(threshold)
    gatherer = KalmanEvidenceGatherer(R)
    indicator = PerformanceVarianceIndicator(backsight)

    return StopCriterion([indicator], [gatherer], [decision])


def createDefaultStopping(given_stopping, iteration_count):
    """
    Create stopping-criterion based on number of iterations.

    :param iteration_count: number of iterations
    """
    if given_stopping is not None:
        return given_stopping

    decision = ThresholdDecision(iteration_count, True)
    gatherer = PassThroughGatherer()
    indicator = IterationCountIndicator()

    return StopCriterion([indicator], [gatherer], [decision])
