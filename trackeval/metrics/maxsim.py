""" 
maxsim.py 

Run Test Example:
    python -m trackeval.metrics.maxsim -m doctest -v 
"""

import numpy as np
from ._base_metric import _BaseMetric
from .. import _timing
from .. import utils


class MaxSim(_BaseMetric):
    """
    Class to calculate the Maximum Similarity (MaxSim) metric between groundtruth and detections.
    """

    @staticmethod
    def get_default_config():
        """Default class config values"""
        default_config = {'COMBINE_METHOD': 'max',  # The aggregation method for combining the results within and between sequences. Default: 'max'
            'PRINT_CONFIG': True,  # Whether to print the config information on init. Default: False.
        }
        return default_config

    def __init__(self, config=None):
        """
        Initialize the MaxSim metric with the specified combination method.
        :param combine_method: Aggregation method ('max', 'average', 'sum').
        """
        super().__init__()
        fields = ['MaxSim']
        self.fields = self.summary_fields = self.integer_fields = self.float_fields = fields

        # Configuration options:
        self.config = utils.init_config(config, self.get_default_config(), self.get_name())
        # Set the combination method (max, average, sum)
        self.combine_method  = self.config['COMBINE_METHOD']
    
    @_timing.time
    def eval_sequence(self, data):
        """
        Calculate MaxSim for a single sequence.
        :param data: Dictionary containing 'num_tracker_dets', 'num_gt_dets', and 'similarity_scores'.
        :return: Aggregated maximum similarity for each sequence.

        Example:
        >>> data = {
        ...     'num_tracker_dets': 2,
        ...     'num_gt_dets': 2,
        ...     'similarity_scores': [np.array([[0.1, 0.5], [0.7, 0.9]]), np.array([[0.4]])]
        ... }
        >>> metric = MaxSim({'COMBINE_METHOD': 'max', 'PRINT_CONFIG': False})
        >>> metric.eval_sequence(data)  # doctest: +ELLIPSIS
        {'MaxSim': 0.9}
        >>> data['similarity_scores'] = [np.array([])]  # No similarity scores
        >>> metric.eval_sequence(data)
        {'MaxSim': 0.0}
        """

        # Initialise results
        seq_res = {}

        for field in self.fields:
            # Return 0 if no ground truth or detections are present
            if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
                seq_res[field] = 0.0
            else:
                # List to store max similarity per frame
                frame_results = []
                # Loop through each timestep
                for similarity_matrix in data['similarity_scores']:
                    # Ensure non-empty matrix
                    if similarity_matrix.size > 0:  
                        frame_results.append(np.max(similarity_matrix))
                    else:
                        frame_results.append(0.0)  
                # Return the aggregated result using the specified combination method
                seq_res[field] = self.combine_per_sequence(frame_results)
            
        return seq_res
    
    def combine_per_sequence(self, frame_results):
        """
        Combine per-frame results to calculate sequence-level metric.
        :param frame_results: List of maximum similarity scores per frame.
        :return Sequence-level result.
        """
        return self._combine_sim(frame_results)
        
    def combine_sequences(self, all_res):
        """
        Combines metrics across all sequences
        :param all_res: List of maximum similarity scores per sequence.
        :return: Combined result for all sequences.
        """
        res = {}
        for field in self.fields:
            res[field] = self._combine_sim([all_res[k][field] for k in all_res.keys()])
            return res
    
    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        pass

    def combine_classes_det_averaged(self, all_res):
        pass
    
    def _combine_sim(self, res):
        """
        Combine the similarity results within or between sequences based on a combination method in ["max", "sum", "average"]
        :param res: List of maximum similarity scores
        :return: Combined result based on a combination method
        """
        if self.combine_method == "max":
            return max(res) #{'MaxSim': max(res)}
        elif self.combine_method == "average":
            return sum(res) / len(res) #{'MaxSim': sum(res) / len(res)}
        elif self.combine_method == "sum":
            return sum(res) #{'MaxSim': sum(res)}
        else:
            raise ValueError(f"Unknown method: {self.combine_method}")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
