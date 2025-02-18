import json
from pathlib import Path

import infra.components.metrics as metrics
import infra.experiments.configs as configs
from infra.components.logger import DataLogger
from infra.tasks.supervised_training import SupervisedTrainer
from infra.utils.utils import create_profiler, initialize_device


class Run:
    def __init__(
        self,
        config: dict,
        run_dir: Path | str,
        id: str | int | None = None,
    ):
        self.id = str(id) if id is not None else None  # used by sweeps, not needed for single runs
        self.config = config

        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(exist_ok=False)

        self.data_file = self.run_dir / "run_data.jsonl"
        self.data_file.touch()

        self.state_file = self.run_dir / "state.txt"
        self.state = "new"

        self.config_file = self.run_dir / "config.json"
        with open(self.config_file, "w") as f:
            f.write(json.dumps(self.config))

    @classmethod
    def load(cls, run_dir: Path | str):
        pass

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        with open(self.state_file, "w") as f:
            f.write(value)

    async def start(self):
        self.state = "running"

        try:
            task = build_task_from_config(self.config, self)
            await task.run()
            self.state = "finished"

        finally:
            if self.state != "finished":
                self.state = "failed"


def build_task_from_config(config: dict, run: Run) -> SupervisedTrainer:
    """Creates and configures a task based on a config dictionary.

    This lives in experiments/ rather than tasks/ to keep the idea of configs contained to experiments

    Args:
        config: A dictionary containing task configuration, typically loaded from a params file.
              See module docstring for schema details.

    Returns:
        A task object, fully configured and ready to run, eg an instance of SupervisedTrainer
    """
    config["task"] = config.get("task") or "supervised_training"  # set the default
    task_args = {}

    device = initialize_device(config.get("device")) if config.get("device") else None

    if device:
        task_args["device"] = device

    if config.get("enable_profiling"):
        task_args["profiler"] = create_profiler(run.run_dir)

    if config.get("model"):
        model = configs.build_model_from_config(config["model"]).to(device)
        task_args["model"] = model
        if config.get("optimizer"):
            task_args["optimizer"] = configs.build_optimizer_from_config(config, model)

    if config.get("loader"):
        train_loader, val_loader, _ = configs.build_loaders_from_config(
            config["dataset"], config["loader"]
        )
        task_args["train_loader"] = train_loader
        task_args["val_loader"] = val_loader

    if config.get("logger"):
        config["logger"]["metrics"] = [
            vars(metrics)[metric_name] for metric_name in config["logger"]["metrics"]
        ]
        task_args["logger"] = DataLogger(run.data_file, run.id, **config["logger"])

    if config.get("criterion"):
        task_args["criterion"] = configs.build_criterion_from_config(config).to(device)

    match config["task"]:
        case "supervised_training":
            task = SupervisedTrainer(
                checkpoint_dir=run.run_dir / "checkpoints", **task_args, **config["trainer"]
            )
        case _:
            raise ValueError("Unknown task")

    return task
