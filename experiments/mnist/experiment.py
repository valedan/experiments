# %%

import asyncio
from infra.logger import DataLogger
from datasets import load_dataset
from typing import Any
import torch as t
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

from infra.models import MLP
from infra.runner import Sweep

plt.switch_backend("agg")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def multiclass_accuracy(logits, labels):
    assert len(logits) == len(labels)
    preds = logits.argmax(dim=1)
    correct = preds == labels
    accuracy = correct.sum().item() / len(logits)
    return accuracy


def create_mnist_loaders(
    train_frac=0, val_frac=0, test_frac=0, train_batch=1, val_batch=1, test_batch=1
):
    train_loader, val_loader, test_loader = None, None, None
    assert all(frac <= 1 for frac in (train_frac, val_frac, test_frac))
    if train_frac or val_frac:
        train_set = load_dataset("mnist", split="train").with_format(
            type="torch", columns=["image", "label"]
        )
        train_set, val_set, _ = data.random_split(
            train_set, [train_frac, val_frac, 1 - train_frac - val_frac]
        )
        if train_frac:
            train_loader = data.DataLoader(
                train_set, batch_size=train_batch, shuffle=True, num_workers=4
            )
        if val_frac:
            val_loader = data.DataLoader(
                val_set, batch_size=val_batch, shuffle=False, num_workers=4
            )

    if test_frac:
        test_set = load_dataset("mnist", split="test").with_format(
            type="torch", columns=["image", "label"]
        )
        test_set, _ = data.random_split(test_set, [test_frac, 1 - test_frac])
        test_loader = data.DataLoader(
            test_set, batch_size=test_batch, shuffle=False, num_workers=4
        )

    return train_loader, val_loader, test_loader


async def train_classifier(
    logger: DataLogger,
    params: dict[str, Any],
):
    train_loader, val_loader, _ = create_mnist_loaders(
        params["train_frac"],
        params["val_frac"],
        0,
        params["train_batch"],
        params["val_batch"],
    )
    assert train_loader and val_loader
    logger.log(
        f"Train set: {len(train_loader.dataset)}, Val set: {len(val_loader.dataset)}"
    )
    model = MLP(28 * 28, 10, params["depth"], params["width"])
    model = model.to(device)
    step = 0
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = t.optim.SGD(model.parameters(), params["learning_rate"])

    for epoch in range(1, params["epochs"] + 1):
        logger.log(f"Starting epoch {epoch}/{params['epochs']}")
        for batch_idx, batch in enumerate(train_loader):
            # TODO: put this data wrangling into the dataloader or dataset
            images, labels = (
                batch["image"].to(t.float32).to(device),
                batch["label"].to(device),
            )
            # TODO: normalize images so that pixels are [0, 1] not [0, 255] and see what the impact is
            flat_images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            logits = model(flat_images)
            loss = criterion(logits, labels)
            train_accuracy = multiclass_accuracy(logits, labels)
            loss.backward()
            optimizer.step()
            datum = {
                "epoch": epoch,
                "step": step,
                "train_loss": loss.item(),
                "train_accuracy": train_accuracy,
            }
            if step % params["val_interval"] == 0:
                total_loss = 0
                val_batches = 0
                # the accuracy will be slightly off if the last batch has fewer items, but does not seem worth the complexity to fix
                accuracies = []
                with t.no_grad():
                    for i, val_batch in enumerate(val_loader):
                        images, labels = (
                            val_batch["image"].to(t.float32).to(device),
                            val_batch["label"].to(device),
                        )
                        flat_images = images.view(images.shape[0], -1)
                        logits = model(flat_images)
                        val_loss = criterion(logits, labels)
                        total_loss += val_loss.item()
                        accuracies.append(multiclass_accuracy(logits, labels))
                        val_batches += 1

                total_val_loss = total_loss / val_batches
                val_accuracy = sum(accuracies) / len(accuracies)
                datum["val_loss"] = total_val_loss
                datum["val_accuracy"] = val_accuracy
                logger.log(
                    f"Batch {batch_idx} - Train Loss {loss.item():.4f} - Val Loss {total_val_loss:.4f}"
                )

                logger.add(**datum)
            step += 1


params = {
    "width": range(10, 100, 30),
    "depth": [2, 4],
    "epochs": [5],
    "val_interval": [1],
    "learning_rate": [0.001],
    "train_frac": [0.01],
    "val_frac": [0.01],
    "train_batch": [64],
    "val_batch": [64],
}


def main():
    sweep = Sweep(
        train_classifier,
        params,
        "test_sweep",
        run_name_formatter=lambda params: f"d{params['depth']}w{params['width']}",
    )
    asyncio.run(sweep.start())


if __name__ == "__main__":
    main()
