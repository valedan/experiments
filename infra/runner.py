from pathlib import Path
import yaml
import os
import torch as t
import time
import random
from tqdm import tqdm
import json
import traceback
from infra.data import mnist_train_loaders
from infra.logger import DataLogger
from itertools import product
from concurrent import futures
from typing import Any, Callable
import multiprocessing
from multiprocessing import Value, Process, Manager, Queue
from tqdm.contrib.concurrent import process_map

multiprocessing.set_start_method("spawn", force=True)


class Run:
    """Manages a single run. It should take a run function and hyperparams, create an id, log files, and logger, and do the run. It can be called async. It also provides an api to fetch the run data."""

    def __init__(
        self,
        trainer: Callable,
        trainer_params: dict[str, Any],
        loader_params: dict[str, Any],
        data_dir: Path,
        run_id: int,
        name: str | None = None,
    ):
        self.trainer = trainer
        self.trainer_params = trainer_params
        self.loader_params = loader_params
        self.data_dir = data_dir
        self.run_id = run_id
        self.name = name
        self.state = "new"
        self.train_loader, self.val_loader = mnist_train_loaders(**loader_params)

    def num_train_samples(self):
        return (
            len(self.train_loader)
            * self.loader_params["train_batch"]
            * self.trainer_params["epochs"]
        )

    # TODO: Change this to __call__
    def start(self):
        data_file = f"{self.data_dir}/runs/{self.run_id:04d}{'_' + self.name if self.name else None}.jsonl"
        log_file = f"{self.data_dir}/logs.txt"
        progress_file = f"{self.data_dir}/progress.txt"
        logger = DataLogger(
            str(self.run_id), data_file, log_file, progress_file, print_logs=False
        )
        logger.log(f"Starting run {self.run_id}")
        try:
            self.trainer(
                logger,
                self.trainer_params,
                self.train_loader,
                self.val_loader,
                device=t.device("cuda" if t.cuda.is_available() else "cpu"),
            )
        except Exception as e:
            logger.log(f"{e.__class__.__name__}: {str(e)}\n{traceback.format_exc()}")
            self.state = "error"
        else:
            self.state = "finished"
            logger.log(f"Finished run {self.run_id}")


def update_progress_bar(progress_file, total_samples):
    with tqdm(total=total_samples, desc="Conducting sweep", unit="samples") as pbar:
        while pbar.n < total_samples:
            new_count = 0
            with open(progress_file, "r") as file:
                for line in file:
                    new_count += int(line.strip())
            pbar.update(new_count - pbar.n)
            time.sleep(0.1)  # Wait a bit before checking again


class Sweep:
    """Manages a sweep. Takes a run_func, and a list of values for each param. it will create a run for each combination of params. takes an arg indicating how many runs can go in parallel, as well as a results dir."""

    def __init__(
        self,
        trainer: Callable,
        exp_dir: str,
        exp_name: str,
        max_workers: int = 10,
        verbose: bool = False,
    ):
        self.trainer = trainer
        self.max_workers = max_workers
        self.verbose = verbose
        self.exp_dir = exp_dir  # exp_dir is where the params files live - the data dirs are subdirs here
        self.exp_name = exp_name
        self.data_dir = Path(f"{exp_dir}/{exp_name}")
        self.data_dir.mkdir(parents=False, exist_ok=False)
        Path(f"{self.data_dir}/runs").mkdir()
        self.exp_file = self.find_exp_file()
        if not self.exp_file:
            raise ValueError(f"No exp file found - {self.exp_file}")
        with open(self.exp_file) as f:
            self.params = yaml.safe_load(f)

        self.runs = []

        # TODO: clean up
        log_file = Path(f"{self.data_dir}/logs.txt")
        progress_file = Path(f"{self.data_dir}/progress.txt")
        log_file.touch()
        progress_file.touch()
        self.create_runs()

    def find_exp_file(self):
        for filename in os.listdir(self.exp_dir):
            if filename.startswith(self.exp_name) and filename.endswith(".yml"):
                return os.path.join(self.exp_dir, filename)
        return None

    def format_run_name(self, trainer_config, loader_config):
        # name should include all non-singular params

        # TODO: this is awful, tidy up

        variable_trainer_params = []
        for param, param_values in {
            **self.params["trainer"],
        }.items():
            if len(param_values) > 1:
                variable_trainer_params.append(param)

        variable_loader_params = []
        for param, param_values in {
            **self.params["loader"],
        }.items():
            if len(param_values) > 1:
                variable_loader_params.append(param)

        name_segments = []

        for param in variable_trainer_params:
            name_segments.append(f"{param}={trainer_config[param]}")

        for param in variable_loader_params:
            name_segments.append(f"{param}={loader_config[param]}")

        if len(name_segments) == 0:
            return "run"
        else:
            return "_".join(name_segments)

    def create_runs(self):
        run_id = 1
        trainer_configs = list(product(*self.params["trainer"].values()))
        loader_configs = list(product(*self.params["loader"].values()))
        # shuffle b/c i tend to define params like depth in order from smallest to biggest, so this smooths out resource usage for the sweep
        random.shuffle(trainer_configs)
        for loader_config_values in loader_configs:
            loader_config = dict(
                zip(self.params["loader"].keys(), loader_config_values)
            )
            for trainer_config_values in trainer_configs:
                trainer_config = dict(
                    zip(self.params["trainer"].keys(), trainer_config_values)
                )
                name = self.format_run_name(trainer_config, loader_config)
                run = Run(
                    self.trainer,
                    trainer_config,
                    loader_config,
                    self.data_dir,
                    run_id,
                    name=name,
                )
                self.runs.append(run)
                runs_file = Path(f"{self.data_dir}/runs.jsonl")
                if not runs_file.exists():
                    runs_file.parent.mkdir(parents=True, exist_ok=True)
                    runs_file.touch()
                with open(runs_file, "a") as f:
                    run_info = {"run_id": run_id, "run_name": name}
                    run_info |= trainer_config
                    run_info |= loader_config
                    f.write(json.dumps(run_info) + "\n")

                run_id += 1

    def start_run(self, run):
        try:
            return run.start()
        except Exception as e:
            error_msg = f"Error in run {run.run_id}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print to console
            # If you have a logging system set up, you might want to log it as well
            # logger.error(error_msg)
            raise  # Re-raise the exception so the future knows it failed

    def start(self):
        runs = [run for run in self.runs if run.state in ["new"]]
        total_samples = sum([run.num_train_samples() for run in runs])
        # TODO: this is duplicated
        progress_file = f"{self.data_dir}/progress.txt"
        # Start the progress bar updater in a separate process
        prog = Process(
            target=update_progress_bar,
            args=(progress_file, total_samples),
        )
        prog.start()
        print(
            f"Starting {len(runs)} runs with {self.max_workers} workers and {total_samples} total samples."
        )
        with futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures_list = [executor.submit(self.start_run, run) for run in runs]
            for future in futures.as_completed(futures_list):
                try:
                    future.result()
                except Exception as e:
                    print(f"Run failed with error: {str(e)}")

        prog.terminate()  # need to kill because total_samples is currently overestimate due to not accounting for smaller final batch
        prog.join()
        print("Sweep completed")

        return len(runs)
