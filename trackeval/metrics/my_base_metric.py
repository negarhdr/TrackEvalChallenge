from abc import ABC, abstractmethod
import numpy as np

class AbstractMetric(ABC):
    """
    Abstract class for a metric that can be evaluated on a dataset.
    """

    @abstractmethod
    def eval_sequence(self, data):
        """
        Evaluates the metric for a single sequence.
        
        :param data: Dictionary containing the data for the sequence.
        :return: Dictionary with the metric result (e.g., {'MaxSim': score}).
        """
        pass

    @abstractmethod
    def combine_per_sequence(self, frame_results, method="average"):
        """
        Combines the per-frame results to compute a sequence-level metric.
        
        :param frame_results: List of metric results for each frame in the sequence.
        :param method: The method to combine the results ('max', 'average', 'sum').
        :return: A combined metric result for the entire sequence.
        """
        pass

