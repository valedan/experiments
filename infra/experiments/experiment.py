from itertools import product
from pathlib import Path

import yaml
from tqdm import tqdm

from infra.experiments.run import Run
from infra.utils.utils import flatten, unflatten


class Experiment:
    """Manages an experiment, which is essentially a collection of training runs. Requires a params file that defines a hyperparam sweep, which will be parsed into individual run configs."""

    def __init__(self, params: dict | str | Path, experiment_dir: str | Path):
        self.experiment_dir = Path(experiment_dir)
        self.log_file = Path(
            f"{self.experiment_dir}/logs.txt"
        )  # file to log experiment progress, info, errors
        # but run dir should also contain config and state for that run why does experiment need to duplicate?
        self.runs_file = Path(
            f"{self.experiment_dir}/runs.jsonl"
        )  # contains config and state of every run
        self.runs_dir = Path(f"{self.experiment_dir}/runs")  # contains individual run dirs
        self.runs = []  # run objects

        # Create experiment dirs and files
        # TODO: support loading existing experiment dir
        self.runs_dir.mkdir(parents=True, exist_ok=False)
        self.log_file.touch()
        self.runs_file.touch()
        self.params = params if isinstance(params, dict) else yaml.safe_load(open(params))

    def start(self):
        raise NotImplementedError()


class GridSearchExperiment(Experiment):
    """Takes experiment params and creates a separate run config for every unique combination of param values"""

    def __init__(self, params: dict | str | Path, experiment_dir: str | Path):
        super().__init__(params, experiment_dir)

        flat_params = flatten(self.params)
        flat_params = {k: v for k, v in flat_params.items() if v is not None}

        for k, v in flat_params.items():
            # values must all be lists for product() to work (not iterable because strings would be split)
            if not isinstance(v, list):
                flat_params[k] = [v]

        # get all unique combinations of values
        config_values = list(product(*flat_params.values()))

        configs = []

        for values in config_values:
            config = dict(zip(flat_params.keys(), values))

            configs.append(unflatten(config))

        # create runs
        for run_id, config in enumerate(configs, 1):
            run_dir = Path(f"{self.experiment_dir}/runs/{run_id:04d}")
            run = Run(config, run_dir, run_id)
            self.runs.append(run)

    def start(self):
        runs = [run for run in self.runs if run.state in ["new"]]

        # TODO: handle failed runs
        for run in tqdm(runs, desc="Conducting experiment", unit="run"):
            run.start()

        return len(runs)
