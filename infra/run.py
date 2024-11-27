from dataclasses import dataclass, field, asdict
from contextlib import nullcontext
import torch as t
from typing import Literal
from torch import GradScaler, OutOfMemoryError, nn
from torch.utils.data import DataLoader
from infra.logger import DataLogger
from pathlib import Path
from tqdm import tqdm
from infra import metrics
from infra.models import get_model
from infra.loaders import LoaderConfig, get_loaders, TokenizerConfig


@dataclass
class OptimizerConfig:
    name: Literal["sgd", "adam"] = "adam"
    learning_rate: float = 0.001


@dataclass
class LoggerConfig:
    flush_interval: int = 1000
    aggregation_interval: int = 1
    metrics: list[str] = field(default_factory=lambda: ["loss"])


@dataclass
class RunConfig:
    model: dict | None = None
    loader: LoaderConfig | None = (
        None  # TODO: loaders need a total refactor - should not need to code a new func per dataset in general
    )
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    tokenizer: TokenizerConfig | None = None
    val_interval: int = 1000
    criterion: Literal["cross_entropy"] = "cross_entropy"
    epochs: int = 1


criterions = {"cross_entropy": nn.CrossEntropyLoss}
optimizers = {"sgd": t.optim.SGD, "adam": t.optim.Adam}


class Run:
    def __init__(
        self,
        config: RunConfig,
        run_dir: Path | str,
        id: str | int | None = None,
        state: str = "new",
        profile: bool = False,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.log_path = self.run_dir / "training_logs.jsonl"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.id = str(id) if id is not None else None
        self.state = state
        self.profile = profile
        self.step = 0
        self.epoch = 0

        if self.run_dir.exists():
            raise RuntimeError(f"run dir already exists: {self.run_dir.absolute()}")

        self.run_dir.mkdir()
        self.checkpoint_dir.mkdir()

    def to_dict(self):
        return {"id": self.id, "state": self.state, **asdict(self.config)}

    def create_logger(self, logger_id=None, flush_interval=None):
        chosen_metrics = [vars(metrics)[metric_name] for metric_name in self.config.logger.metrics]
        if flush_interval is None:
            flush_interval = self.config.logger.flush_interval

        # TODO: just pass logger config directly
        return DataLogger(
            data_path=self.log_path,
            id=logger_id,
            flush_interval=flush_interval,
            aggregation_interval=self.config.logger.aggregation_interval,
            metrics=chosen_metrics,
        )

    def create_optimizer(self, model):
        optimizer = optimizers[self.config.optimizer.name](
            model.parameters(), lr=self.config.optimizer.learning_rate, fused=True
        )
        return optimizer

    def init_model(self, model=None):
        if not model:
            if not self.config.model:
                raise ValueError("No model config provided")
            model = get_model(self.config.model)

        return model.to(self.device)

    def create_criterion(self):
        # TODO: ignore_index needed for ignoring padding token in nlp tasks, need to generalize
        return criterions[self.config.criterion](ignore_index=0).to(self.device)

    def init_device(self):
        if t.cuda.is_available():
            self.device = "cuda"
            t.cuda.empty_cache()
        else:
            self.device = "cpu"

    def save_model_checkpoint(self, model, filename=None):
        if filename is None:
            filename = self.step

        t.save(model.state_dict(), self.checkpoint_dir / f"{filename}.pt")

    def evaluate_batch(self, batch, model, criterion, logger):
        """used for both train and validation batches"""
        with t.autocast(device_type=str(self.device), dtype=t.float16):
            samples, labels = batch
            samples = samples.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = model(samples)
            # treat all dims but the last as batch dims
            logits = logits.flatten(end_dim=-2)

            loss = criterion(logits, labels)

        logger.log(
            logits=logits,
            labels=labels,
            loss=loss,
            step=self.step,
            epoch=self.epoch,
            run_id=self.id,
        )

        return loss, logits, labels

    def start(
        self,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        model=None,
    ):
        # t.cuda.memory._record_memory_history()
        self.init_device()

        if train_loader is None:
            train_loader, val_loader, _ = get_loaders(
                self.config.loader, tokenizer_config=self.config.tokenizer
            )

        model = self.init_model(model)
        criterion = self.create_criterion()
        optimizer = self.create_optimizer(model)
        scaler = GradScaler()

        with tqdm(
            total=(len(train_loader.dataset) * self.config.epochs),
            desc="Training Run",
            unit="samples",
            leave=False,
        ) as run_bar, self.profiler(), self.create_logger("train") as train_logger:
            for _ in range(self.config.epochs):
                try:
                    self.epoch += 1
                    for batch in train_loader:
                        self.step += 1

                        model.train()
                        optimizer.zero_grad()
                        loss, _, _ = self.evaluate_batch(batch, model, criterion, train_logger)

                        scaler.scale(loss).backward()
                        # TODO: look into using clip_grad_norm here to deal with exploding gradients
                        scaler.step(optimizer)
                        scaler.update()
                        run_bar.update(len(batch[0]))

                        if val_loader and self.step % self.config.val_interval == 0:
                            model.eval()
                            # 0 disables automatic flushing because we just want it to happen after the whole epoch. could cause perf issues if val dataset is extremely large
                            with self.create_logger("val", 0) as val_logger, t.no_grad():
                                for batch in val_loader:
                                    self.evaluate_batch(batch, model, criterion, val_logger)
                except OutOfMemoryError as e:
                    # t.cuda.memory._dump_snapshot("my_snapshot.pickle")
                    breakpoint()

        self.save_model_checkpoint(model, "final")

    def profiler(self):
        """pytorch profiler to use when a Run is in profile mode"""

        if not self.profile:
            return nullcontext()

        return t.profiler.profile(
            activities=[
                t.profiler.ProfilerActivity.CPU,
                t.profiler.ProfilerActivity.CUDA,
            ],
            schedule=t.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=t.profiler.tensorboard_trace_handler("./log/pytorch_profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
