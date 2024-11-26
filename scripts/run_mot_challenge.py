""" 
run_mot_challenge.py 

Run Example:
    python scripts/run_mot_challenge.py --USE_PARALLEL False --METRICS CLEAR --TRACKERS_TO_EVAL MPNTrack
"""

import sys
import os
import argparse
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

def parse_arguments(default_config):
    """
    Parse command-line arguments and update the default config.
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

def filter_config(config, keys):
    """
    Filter the "config" dictionary (merged configs) by a set of "keys".
    """
    return {k: config[k] for k in keys if k in config}  # instead of checking config keys one by one in keys, I say collect all the keys and check if they are in config 
    #return {k: v for k, v in config.items() if k in keys}

if __name__ == '__main__':
    freeze_support()

    # Load default configurations
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'VACE'], 'THRESHOLD': 0.5}

    # Merge all configs and parse arguments
    merged_config = {**default_eval_config, **default_dataset_config, **default_metrics_config}
    config = parse_arguments(merged_config)

    # Split configs
    eval_config = filter_config(config, default_eval_config.keys())
    dataset_config = filter_config(config, default_dataset_config.keys())
    metrics_config = filter_config(config, default_metrics_config.keys())

    # Initialize evaluator, dataset, and metrics
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        metric(metrics_config)
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]
        if metric.get_name() in metrics_config['METRICS']
    ]
    if not metrics_list:
        raise Exception('No metrics selected for evaluation')

    # Run evaluation
    evaluator.evaluate(dataset_list, metrics_list)
