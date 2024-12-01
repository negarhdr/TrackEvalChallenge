
""" 
run_kitti.py 

Run Example:
    python scripts/run_kitti.py --USE_PARALLEL False --METRICS CLEAR --TRACKERS_TO_EVAL CIWT
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trackeval  
from scripts.base_runner import BaseRunner

class KittiRunner(BaseRunner):
    def __init__(self):
        """
        Initialize KittiRunner with specific default configurations.
        """
        default_eval_config = trackeval.Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = False
        default_dataset_config = trackeval.datasets.Kitti2DBox.get_default_dataset_config()
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'MaxSim']}
        super().__init__(default_eval_config, default_dataset_config, default_metrics_config)

if __name__ == '__main__':
    runner = KittiRunner()
    runner.run(
        dataset_cls = trackeval.datasets.Kitti2DBox,
        metric_classes = [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.MaxSim]
    )