import numpy as np
from .my_base_metric import AbstractMetric

class MaxSim(AbstractMetric):
    """Class to calculate the Maximum Similarity (MaxSim) metric between GT and detections."""

    def __init__(self, combine_method="average"):
        """
        Initialize the MaxSim metric with the specified combination method.
        :param combine_method: Aggregation method ('max', 'average', 'sum').
        """
        super().__init__()
        self.combine_method = combine_method  # Set the combination method (max, average, sum)

    @classmethod
    def get_name(cls):
        return cls.__name__
    
    def eval_sequence(self, data):
        """Calculate MaxSim for a single sequence."""
        # Return 0 if no GT or detections are present
        if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
            return {'MaxSim': 0.0}
        
        # List to store max similarity per frame
        frame_results = []
        # Loop through each timestep
        for similarity_matrix in data['similarity_scores']:
            if similarity_matrix.size > 0:  # Ensure non-empty matrix
                frame_results.append(np.max(similarity_matrix))
            else:
                frame_results.append(0.0)  # No matches for this frame

        # Return the aggregated result using the specified combination method
        return self.combine_per_sequence(frame_results)
    
    def combine_per_sequence(self, frame_results):
        """
        Combine per-frame results to calculate sequence-level metric.
        :param frame_results: List of maximum similarity scores per frame.
        :return: Sequence-level result.
        """
        if self.combine_method == "max":
            return {'MaxSim': max(frame_results)}
        elif self.combine_method == "average":
            return {'MaxSim': sum(frame_results) / len(frame_results)}
        elif self.combine_method == "sum":
            return {'MaxSim': sum(frame_results)}
        else:
            raise ValueError(f"Unknown method: {self.combine_method}")

if __name__ == "__main__":
    # Example dataset, structured as sequences with similarity_scores and counts
    example_data = [
        {
            "similarity_scores": [
                np.array([[0.5, 0.6], [0.3, 0.8]]),  # Frame 1
                np.array([[0.2, 0.9], [0.4, 0.7]]),  # Frame 2
                np.array([[0.1, 0.4], [0.3, 0.5]])   # Frame 3
            ],
            "num_gt_dets": 2,  # Correct number of ground truth detections
            "num_tracker_dets": 2  # Correct number of tracker detections
        },
        {
            "similarity_scores": [
                np.array([[0.7, 0.8], [0.6, 0.9]]),  # Frame 1
                np.array([[0.5, 0.4], [0.3, 0.6]])   # Frame 2
            ],
            "num_gt_dets": 2,  # Correct number of ground truth detections
            "num_tracker_dets": 2  # Correct number of tracker detections
        },
        # Edge case: no detections
        {
            "similarity_scores": [],
            "num_gt_dets": 0,  # No ground truth detections
            "num_tracker_dets": 0  # No tracker detections
        }
    ]

    # Initialize MaxSim metric with 'average' as the combination method
    max_sim_metric = MaxSim(combine_method="average")

    # Loop over the example data to evaluate the MaxSim for each sequence
    for i, data in enumerate(example_data):
        print(f"Evaluating sequence {i + 1}...")
        
        # Calculate per-frame results using eval_sequence
        sequence_result = max_sim_metric.eval_sequence(data)
        print("Per sequence results:", sequence_result)
        