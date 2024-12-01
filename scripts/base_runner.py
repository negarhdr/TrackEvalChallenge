# scripts/base_runner.py

""" 
base_runner.py 

Run Example:
    python scripts/base_runner.py 
"""

import trackeval 

class BaseRunner:
    def __init__(self, default_eval_config, default_dataset_config, default_metrics_config):
        """
        Initialize the BaseRunner with default configurations.
        """
        # Merge and parse configs
        self.default_eval_config = default_eval_config
        self.default_dataset_config = default_dataset_config
        self.default_metrics_config = default_metrics_config
        self.merged_config = {**default_eval_config, **default_dataset_config, **default_metrics_config}
        self.config = self.parse_arguments(self.merged_config)

    def parse_arguments(self, default_config):
        """
        Parse command-line arguments and update the default config.
        """
        import argparse
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
                        raise Exception(f'Command line parameter {setting} must be True or False')
                # Check integer type
                elif isinstance(default_config[setting], int):
                    default_config[setting] = int(value)
                # Check None type
                elif not value:
                    default_config[setting] = None
                # Default handling
                else:
                    default_config[setting] = value
        return default_config

    def filter_config(self, config, keys):
        """
        Filter the "config" dictionary (merged configs) by a set of "keys".
        """
        return {k: config[k] for k in keys if k in config}
    
    def run(self, dataset_cls, metric_classes):
        """
        Run the evaluation process.
        """
        # Split configs
        eval_config = self.filter_config(self.config, self.default_eval_config.keys())
        dataset_config = self.filter_config(self.config, self.default_dataset_config.keys())
        metrics_config = self.filter_config(self.config, self.default_metrics_config.keys())

        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [dataset_cls(dataset_config)]
        metrics_list = [
            metric() for metric in metric_classes if metric.get_name() in metrics_config['METRICS']
        ]
        if not metrics_list:
            raise Exception('No metrics selected for evaluation')

        evaluator.evaluate(dataset_list, metrics_list)

