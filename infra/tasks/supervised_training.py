from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from torch import GradScaler, nn
from torch.profiler import record_function
from torch.utils.data import DataLoader

from infra.components.logger import DataLogger
from infra.utils.utils import initialize_device


class SupervisedTrainer:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        model: nn.Module,
        logger: DataLogger,
        train_loader: DataLoader,
        val_interval: int = 1000,
        epochs: int = 1,
        device: torch.device | None = None,
        optimizer: Optimizer | None = None,
        criterion: nn.Module | None = None,
        val_loader: DataLoader | None = None,
        profiler: torch.profiler.profile | None = None,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.logger = logger
        self.train_loader = train_loader

        self.val_interval = val_interval
        self.epochs = epochs
        self.device = device or initialize_device()
        self.model = model.to(self.device)
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.criterion.to(self.device)
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.profiler = profiler or nullcontext()

        self.epoch = 0
        self.step = 0

        self.checkpoint_dir.mkdir()

    def save_model_checkpoint(self, model, filename=None):
        if filename is None:
            filename = self.step

        torch.save(model.state_dict(), self.checkpoint_dir / f"{filename}.pt")

    def evaluate_batch(self, batch, split):
        """used for both train and validation batches"""
        with torch.autocast(device_type=str(self.device), dtype=torch.float16):
            samples, labels = batch
            samples = samples.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(samples)
            # treat all dims but the last as batch dims
            logits = logits.flatten(end_dim=-2)
            loss = self.criterion(logits, labels)

        self.logger.log(
            logits=logits,
            labels=labels,
            loss=loss,
            step=self.step,
            epoch=self.epoch,
            split=split,
        )

        return loss, logits, labels

    async def run(self):
        scaler = GradScaler()

        with (
            tqdm(
                total=(len(self.train_loader.dataset) * self.epochs),
                desc="Training Run",
                unit="samples",
                leave=False,
            ) as run_bar,
            self.profiler as prof,
            self.logger,
        ):
            for epoch in range(self.epochs):
                self.epoch = epoch
                for batch in self.train_loader:
                    self.step += 1

                    self.model.train()
                    self.optimizer.zero_grad()

                    with record_function("forward"):
                        loss, _, _ = self.evaluate_batch(batch, "train")

                    with record_function("backward"):
                        scaler.scale(loss).backward()

                    # TODO: look into using clip_grad_norm here to deal with exploding gradients
                    with record_function("optimizer"):
                        scaler.step(self.optimizer)

                    scaler.update()
                    run_bar.update(len(batch[0]))

                    if self.val_loader and self.step % self.val_interval == 0:
                        print("val")
                        with record_function("validation"):
                            self.logger.flush()  # flush any training logs so they are not combined with val for metrics
                            self.model.eval()
                            with torch.no_grad():
                                for batch in self.val_loader:
                                    self.evaluate_batch(batch, "val")
                            self.logger.flush()
                    if prof:
                        prof.step()

        if prof:
            prof.export_memory_timeline(f"{self.checkpoint_dir}/memory_timeline.html")

        self.save_model_checkpoint(self.model, "final")
