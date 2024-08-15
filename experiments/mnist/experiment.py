# %%


from tqdm import tqdm
from multiprocessing import Queue, Value
from infra.data import mnist_train_set
from infra.logger import DataLogger
from datasets import load_dataset
from typing import Any
import torch as t
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from infra.models import MLP
from infra.runner import Sweep
from infra.metrics import multiclass_accuracy

plt.switch_backend("agg")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
# print(f"Process {os.getpid()} - Using device: {device}")


def train_classifier(
    logger: DataLogger,
    params: dict[str, Any],
    run_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    assert train_loader and val_loader
    model = MLP(28 * 28, 10, params["depth"], params["width"])
    model = model.to(device)
    step = 0
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = t.optim.SGD(model.parameters(), params["learning_rate"])

    for epoch in range(1, params["epochs"] + 1):
        logger.log(f"Starting epoch {epoch}/{params['epochs']}")
        total_train_loss = 0
        train_accuracies = []
        for batch_idx, (train_images, labels) in enumerate(train_loader):
            logger.log(train_images.shape)
            train_images = train_images.to(device)
            labels = labels.to(device)
            is_final_batch = batch_idx + 1 == len(train_loader)
            # TODO: put this data wrangling into the dataloader or dataset
            # TODO: normalize images so that pixels are [0, 1] not [0, 255] and see what the impact is
            flat_images = train_images.view(train_images.shape[0], -1)
            # flat_images.to(device)

            optimizer.zero_grad()
            logits = model(flat_images)
            loss = criterion(logits, labels)
            train_accuracy = multiclass_accuracy(logits, labels)
            total_train_loss += loss
            train_accuracies.append(train_accuracy)
            loss.backward()
            optimizer.step()
            datum = {
                "epoch": epoch,
                "step": step,
                "batch_train_loss": loss.item(),
                "batch_train_accuracy": train_accuracy,
            }
            if is_final_batch:
                datum |= {
                    "epoch_train_loss": total_train_loss.item() / len(train_loader),
                    "epoch_train_accuracy": sum(train_accuracies)
                    / len(train_accuracies),
                }

            if step % params["val_interval"] == 0:
                total_val_loss = 0
                # the accuracy will be slightly off if the last batch has fewer items, but does not seem worth the complexity to fix
                val_accuracies = []
                with t.no_grad():
                    for i, (val_images, labels) in enumerate(val_loader):
                        val_images = val_images.to(device)
                        labels = labels.to(device)
                        flat_images = val_images.view(val_images.shape[0], -1)
                        logits = model(flat_images)
                        val_loss = criterion(logits, labels)
                        total_val_loss += val_loss.item()
                        val_accuracies.append(multiclass_accuracy(logits, labels))

                total_val_loss = total_val_loss / len(val_loader)
                val_accuracy = sum(val_accuracies) / len(val_accuracies)
                datum["epoch_val_loss"] = total_val_loss
                datum["epoch_val_accuracy"] = val_accuracy
                logger.log(
                    f"Batch {batch_idx} - Train Loss {loss.item():.4f} - Val Loss {total_val_loss:.4f}"
                )

            logger.add(**datum)
            logger.log_progress(len(train_images))
            step += 1


params = {
    "trainer": {
        # "width": [10, 50, 200, 800, 1600],
        # "depth": [1, 2, 3, 4, 5],
        "width": [10, 20, 30, 40],
        "depth": [1, 2, 3, 4, 5],
        "epochs": [1],
        "val_interval": [1],
        "learning_rate": [0.001],
    },
    "loader": {
        # "train_frac": [0.01, 0.05, 0.2],
        "train_frac": [0.1],
        "val_frac": [0.05],
        "train_batch": [64],
        "val_batch": [64],
    },
}

test_params = {
    "trainer": {
        "width": 10,
        "depth": 2,
        "epochs": 5,
        "val_interval": 1,
        "learning_rate": 0.001,
    },
    "loader": {
        "train_frac": 0.01,
        "val_frac": 0.01,
        "train_batch": 128,
        "val_batch": 128,
    },
}


def formatter(params):
    return f"d{params['depth']}w{params['width']}"


# sweep_dir = "results/exp1_width_depth_frac"
sweep_dir = "results/test"


def main():
    sweep = Sweep(
        train_classifier,
        params,
        sweep_dir,
        run_name_formatter=formatter,
        max_workers=20,
    )
    sweep.start()
    # train_classifier(DataLogger("test", "test.jsonl", "test.txt"), test_params, "test")


if __name__ == "__main__":
    main()
