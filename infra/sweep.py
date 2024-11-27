from __future__ import annotations
import time
from pathlib import Path
import json
import yaml
from tqdm import tqdm
from infra.loaders import get_loaders
from infra.run import Run
from itertools import product
from infra.run import RunConfig, OptimizerConfig, LoggerConfig, TokenizerConfig
from infra.loaders import LoaderConfig
from infra.utils import flatten, unflatten
from typing import Any


def create_run_configs(params: dict[str, Any]) -> list[RunConfig]:
    """Takes sweep params and creates a separate run config for every unique combination of param values"""
    flat_params = flatten(params)
    flat_params = {k: v for k, v in flat_params.items() if v is not None}
    paired_names = flat_params.pop("paired.names", None)

    for k, v in flat_params.items():
        # values must all be lists for product() to work (not iterable because strings would be split)
        if not isinstance(v, list):
            flat_params[k] = [v]

    # get all unique combinations of values
    config_values = list(product(*flat_params.values()))

    configs = []

    for values in config_values:
        config = dict(zip(flat_params.keys(), values))

        if paired_names:
            # allows specifying sets of values without taking every combination of them
            paired_values = config.pop("paired.values")
            config |= dict(zip(paired_names, paired_values))

        grouped_params = unflatten(config)

        # TODO: abstract this grouped param stuff
        grouped_params["loader"] = LoaderConfig(**grouped_params["loader"])

        if "optimizer" in grouped_params:
            grouped_params["optimizer"] = OptimizerConfig(**grouped_params["optimizer"])

        if "logger" in grouped_params:
            grouped_params["logger"] = LoggerConfig(**grouped_params["logger"])

        if "tokenizer" in grouped_params:
            grouped_params["tokenizer"] = TokenizerConfig(**grouped_params["tokenizer"])
        configs.append(RunConfig(**grouped_params))

    return configs


class Sweep:
    """Manages a sweep, which is essentially a collection of training runs. Requires a params file that defines a hyperparam sweep, which will be parsed into individual run configs."""

    def __init__(self, params_path: Path, sweep_dir: Path | None = None):
        self.runs = []
        params_path = Path(params_path)

        if sweep_dir:
            self.sweep_dir = Path(sweep_dir)
        else:
            # assume first 3 chars of the params file is the sweep ID
            self.sweep_dir = params_path.parent / params_path.name[:3]

        self.create_sweep_files()

        with open(params_path) as f:
            params = yaml.safe_load(f)

        run_configs = create_run_configs(params)
        for idx, config in enumerate(run_configs):
            run_id = idx + 1
            run_dir = Path(f"{self.sweep_dir}/runs/{run_id:04d}")
            run = Run(
                config,
                run_dir,
                id=run_id,
                state="new",
            )
            self.runs.append(run)

    def create_sweep_files(self):
        self.sweep_dir.mkdir(parents=True, exist_ok=False)
        Path(f"{self.sweep_dir}/runs").mkdir()
        self.log_file = Path(f"{self.sweep_dir}/logs.txt")
        self.runs_file = Path(f"{self.sweep_dir}/runs.jsonl")
        self.log_file.touch()
        self.runs_file.touch()

    def log_run_states(self):
        """Clear the runs file and write the current state of all runs to it"""
        with open(self.runs_file, "w") as f:
            for run in self.runs:
                f.write(json.dumps(run.to_dict()) + "\n")

    def start(self):
        runs = [run for run in self.runs if run.state in ["new"]]
        print(f"Starting {len(runs)} runs.")

        with tqdm(total=len(runs), desc="Conducting sweep", unit="run") as sweep_bar:
            for run in runs:
                run.start()
                # TODO: can tqdm just iterate over runs?
                self.log_run_states()
                sweep_bar.update(1)

        print("Sweep completed")

        return len(runs)
