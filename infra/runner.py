from __future__ import annotations
from dataclasses import dataclass
from infra.configs import RunConfig, create_configs
from pathlib import Path
import json
import yaml
import torch as t
from tqdm import tqdm
from infra.data import mnist_train_loaders
from infra.logger import DataLogger
from infra.models import MLP
from infra.trainers import train_classifier


@dataclass
class Run:
    config: RunConfig
    id: int
    state: str

    def dict(self):
        return {"id": self.id, "state": self.state, **self.config.model_dump()}


class Sweep:
    """Manages a sweep. Takes a run_func, and a list of values for each param. it will create a run for each combination of params. takes an arg indicating how many runs can go in parallel, as well as a results dir."""

    def __init__(self, exp_file: Path):
        self.exp_file = exp_file
        self.create_sweep_files()
        self.runs = []

        with open(self.exp_file) as f:
            params = yaml.safe_load(f)
            self.exp_type = params["exp"]["type"]

        configs = create_configs(params)

        self.create_runs(configs)

    def create_sweep_files(self):
        self.data_dir = (
            self.exp_file.parent / self.exp_file.name[:3]
        )  # first 3 chars of the exp file is the ID
        self.data_dir.mkdir(parents=False, exist_ok=False)
        Path(f"{self.data_dir}/runs").mkdir()
        self.log_file = Path(f"{self.data_dir}/logs.txt")
        self.runs_file = Path(f"{self.data_dir}/runs.jsonl")
        self.log_file.touch()
        self.runs_file.touch()

    def create_runs(self, configs: list[RunConfig]):
        run_id = 1
        for config in configs:
            run = Run(config, run_id, "new")
            self.runs.append(run)
            # should the run object handle persisting its state to the runs file?
            with open(self.runs_file, "a") as f:
                f.write(json.dumps(run.dict()) + "\n")
            run_id += 1

    def start(self):
        runs = [run for run in self.runs if run.state in ["new"]]

        loader_groups = {}
        for run in runs:
            loader_groups.setdefault(run.config.loader, []).append(run)

        print(f"Starting {len(runs)} runs.")

        with tqdm(total=len(runs), desc="Conducting sweep", unit="run") as sweep_bar:
            run_idx = 1
            for loader_config, runs_for_loader in loader_groups.items():
                train_loader, val_loader = mnist_train_loaders(
                    **loader_config.model_dump()
                )
                for run in runs_for_loader:
                    with tqdm(
                        total=(len(train_loader.dataset) * run.config.trainer.epochs),
                        desc=f"Run {run_idx}/{len(runs)}",
                        unit="samples",
                        leave=False,
                    ) as run_bar:


                        data_file = Path(f"{self.data_dir}/runs/{run.id:04d}.jsonl")
                        logger = DataLogger(
                            id=run.id,
                            data_file=data_file,
                            after_log=run_bar.update,
                        )
                        model = MLP(**run.config.model.model_dump())

                        train_classifier(
                            logger,
                            model,
                            run.config.trainer,
                            train_loader,
                            val_loader,
                            device="cuda" if t.cuda.is_available() else "cpu",
                        )
                        run_idx += 1

                        sweep_bar.update(1)

        print("Sweep completed")

        return len(runs)
