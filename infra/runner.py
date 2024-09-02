from infra.configs import ConfigLoader, RunConfig
from pathlib import Path
import yaml
import torch as t
import time
from tqdm import tqdm
import traceback
from infra.data import mnist_train_loaders
from infra.logger import DataLogger
from concurrent import futures
import multiprocessing
from multiprocessing import Process
from infra.models import MLP
from infra.trainers import train_classifier

multiprocessing.set_start_method("spawn", force=True)


class Run:
    """Manages a single run. It should take a run function and hyperparams, create an id, log files, and logger, and do the run. It can be called async. It also provides an api to fetch the run data."""

    def __init__(
        self, config: RunConfig, data_file: Path, progress_file: Path, log_file: Path
    ):
        self.config = config
        self.data_file = data_file
        self.progress_file = progress_file
        self.log_file = log_file
        self.state = "new"
        if self.config.exp.type == "mnist":
            self.train_loader, self.val_loader = mnist_train_loaders(
                **self.config.loader.model_dump()
            )
            self.trainer = train_classifier
        else:
            raise ValueError()

    def num_train_samples(self):
        return len(self.train_loader.dataset) * self.config.trainer.epochs

    def start(self):
        logger = DataLogger(
            str(self.config.run_id),
            self.data_file,
            self.log_file,
            self.progress_file,
            print_logs=False,
        )
        model = MLP(**self.config.model.model_dump())
        logger.log(f"Starting run {self.config.run_id}")
        try:
            self.trainer(
                logger,
                model,
                self.config.trainer,
                self.train_loader,
                self.val_loader,
                device=t.device("cuda" if t.cuda.is_available() else "cpu"),
            )
        except Exception as e:
            logger.log(f"{e.__class__.__name__}: {str(e)}\n{traceback.format_exc()}")
            self.state = "error"
        else:
            self.state = "finished"
            logger.log(f"Finished run {self.config.run_id}")


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
        exp_file: Path,
        max_workers: int = 10,
        verbose: bool = False,
    ):
        self.exp_file = exp_file
        self.max_workers = max_workers
        self.verbose = verbose
        self.create_data_dir()
        self.runs = {}

        with open(self.exp_file) as f:
            params = yaml.safe_load(f)
            self.exp_type = params["exp"]["type"]
            self.configs = ConfigLoader(params)

        self.create_runs()

    def create_data_dir(self):
        self.data_dir = (
            self.exp_file.parent / self.exp_file.name[:4]
        )  # first 3 chars of the exp file is the ID
        self.data_dir.mkdir(parents=False, exist_ok=False)
        Path(f"{self.data_dir}/runs").mkdir()
        self.progress_file = Path(f"{self.data_dir}/progress.txt")
        self.log_file = Path(f"{self.data_dir}/logs.txt")
        self.runs_file = Path(f"{self.data_dir}/runs.jsonl")
        self.progress_file.touch()
        self.log_file.touch()
        self.runs_file.touch()

    def create_runs(self):
        run_id = 1
        for config in self.configs.shuffled():
            config.run_id = run_id
            run_data_file = Path(f"{self.data_dir}/runs/{run_id:04d}.jsonl")
            run = Run(config, run_data_file, self.progress_file, self.log_file)
            self.runs[run_id] = run
            with open(self.runs_file, "a") as f:
                f.write(config.model_dump_json() + "\n")
            run_id += 1

    def start_run(self, run):
        try:
            run.start()
            return run.config.run_id
        except Exception as e:
            error_msg = f"Error in run {run.run_id}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # logger.error(error_msg)
            raise

    def start(self):
        runs = [run for run in self.runs.values() if run.state in ["new"]]
        total_samples = sum([run.num_train_samples() for run in runs])
        # Start the progress bar updater in a separate process
        prog = Process(
            target=update_progress_bar,
            args=(self.progress_file, total_samples),
        )
        prog.start()
        print(
            f"Starting {len(runs)} runs with {self.max_workers} workers and {total_samples} total samples."
        )
        with futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures_list = [executor.submit(self.start_run, run) for run in runs]
            for future in futures.as_completed(futures_list):
                try:
                    error = future.exception()
                    if error:
                        print(f"Run failed with error: {error}")
                    future.result()
                except Exception as e:
                    print(f"Run failed with error: {str(e)}")

        prog.terminate()
        prog.join()
        print("Sweep completed")

        return len(runs)
