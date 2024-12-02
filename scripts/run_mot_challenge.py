""" 
run_mot_challenge.py 

Run Example:
    python scripts/run_mot_challenge.py --USE_PARALLEL False --METRICS CLEAR --TRACKERS_TO_EVAL MPNTrack
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from scripts.base_runner import BaseRunner
from trackeval import metrics, datasets, Evaluator


class MotChallengeRunner(BaseRunner):
    def __init__(self):
        """
        Initialize MotChallengeRunner with specific default configurations.
        """
        default_eval_config = Evaluator.get_default_eval_config()
        default_eval_config['DISPLAY_LESS_PROGRESS'] = False
        default_dataset_config = datasets.MotChallenge2DBox.get_default_dataset_config()
        default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE', 'MaxSim'], 'THRESHOLD': 0.5}
        super().__init__(default_eval_config, default_dataset_config, default_metrics_config)

    def parse_arguments(self, default_config):
        """
        Override parse_arguments to handle SEQ_INFO special case for MOT Challenge.
        """
        parser = argparse.ArgumentParser()
        for setting, value in default_config.items():
            arg_type = list if isinstance(value, list) or value is None else type(value)
            parser.add_argument(f"--{setting}", nargs='+' if arg_type == list else None, type=str)
        args = vars(parser.parse_args())

        for setting, value in args.items():
            if value is not None:
                # Check boolean type 
                if isinstance(default_config[setting], bool):
                    if value.lower() == 'true':
                        default_config[setting] = True
                    elif value.lower() == 'false':
                        default_config[setting] = False
                    else:
                        raise Exception('Command line parameter ' + setting + ' must be True or False')
                # Check integer type
                elif isinstance(default_config[setting], int):
                    default_config[setting] = int(value)
                # Check None type
                elif not value:
                    default_config[setting] = None
                # Special handling for SEQ_INFO
                elif setting == 'SEQ_INFO':  
                    default_config[setting] = dict(zip(value, [None] * len(value)))
                # Other
                else:
                    default_config[setting] = value 
        return default_config

if __name__ == '__main__':
    runner = MotChallengeRunner()
    runner.run(
        dataset_cls=datasets.MotChallenge2DBox,
        metric_classes=[metrics.HOTA, metrics.CLEAR, metrics.Identity, metrics.VACE, metrics.MaxSim]
    )