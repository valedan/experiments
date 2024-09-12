from collections import defaultdict
from typing import Callable
from datetime import datetime
import json
from pathlib import Path
import torch as t


def multiclass_accuracy(logits, labels):
    assert len(logits) == len(labels)
    preds = logits.argmax(dim=1)
    correct = preds == labels
    accuracy = correct.sum().item() / len(logits)
    return accuracy


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DataLogger:
    """A logger class to use for experiments. Each instance can have an associated data file to write new data to, and a log file to write logs to. Uses jsonl for data and txt for logs. Can have an identifier to associate to a run/sweep."""

    def __init__(
        self,
        data_file: Path | None = None,
        id: str | None = None,
        after_log: Callable | None = None,
        write_n_train_logs: int = 1,
    ):
        self.logs = defaultdict(list)
        self.data_file = data_file if data_file else None
        self.id = id
        self.after_log = after_log
        self.write_n_train_logs = write_n_train_logs

    def log(self, logits, labels, loss, step, epoch, split):
        self.logs[epoch].append(
            {
                "logits": logits,
                "labels": labels,
                "loss": loss,
                "step": step,
                "split": split,
                "created_at": now(),
            }
        )

    def flush(self):
        """Should be called at the end of an epoch"""
        total_train_samples = 0
        for epoch, logs in self.logs.items():
            total_train_loss = t.tensor([0.0]).to("cuda")
            train_accuracies = []
            total_val_loss = t.tensor([0.0]).to("cuda")
            val_accuracies = []
            data = []
            for idx, log in enumerate(logs):
                if log["split"] == "train":
                    # TODO: DataLogger should take a list of metrics to calculate
                    total_train_samples += len(log['labels'])
                    total_train_loss += log["loss"]
                    train_accuracy = multiclass_accuracy(log["logits"], log["labels"])
                    train_accuracies.append(train_accuracy)
                    if idx % self.write_n_train_logs == 0:
                        data.append(
                            {
                                "batch_train_loss": log["loss"].item(),
                                "batch_train_accuracy": train_accuracy,
                                "step": log["step"],
                                "epoch": epoch,
                            }
                        )
                elif log["split"] == "val":
                    total_val_loss += log["loss"]
                    val_accuracy = multiclass_accuracy(log["logits"], log["labels"])
                    val_accuracies.append(val_accuracy)
            datum = {"epoch": epoch}
            if len(train_accuracies) > 0:
                datum |= {
                    "epoch_train_loss": total_train_loss.item() / len(train_accuracies),
                    "epoch_train_accuracy": sum(train_accuracies)
                    / len(train_accuracies),
                }
            if len(val_accuracies) > 0:
                datum |= {
                    "epoch_val_loss": total_val_loss.item() / len(val_accuracies),
                    "epoch_val_accuracy": sum(val_accuracies) / len(val_accuracies),
                }
            data.append(datum)

        with self.data_file.open("a") as f:
            for datum in data:
                f.write(json.dumps({"id": self.id, **datum}) + "\n")
        if self.after_log:
            self.after_log(total_train_samples)

        self.logs = defaultdict(list)
