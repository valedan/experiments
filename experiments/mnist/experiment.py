# %%

from infra.logger import DataLogger
from datasets import load_dataset
import torch as t
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

from infra.models import MLP

plt.switch_backend("agg")

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def multiclass_accuracy(logits, labels):
    assert len(logits) == len(labels)
    preds = logits.argmax(dim=1)
    correct = preds == labels
    accuracy = correct.sum().item() / len(logits)
    return accuracy


def train_classifier(
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    logger: DataLogger,
    n_epochs: int = 1,
    val_interval: int = 100,
    learning_rate: float = 0.001,
):
    step = 0
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = t.optim.SGD(model.parameters(), learning_rate)

    for epoch in range(1, n_epochs + 1):
        logger.log(f"Starting epoch {epoch}/{n_epochs}")
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
            if step % val_interval == 0:
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


train_set = load_dataset("mnist", split="train").with_format(
    type="torch", columns=["image", "label"]
)  # 60000
test_set = load_dataset("mnist", split="test").with_format(
    type="torch", columns=["image", "label"]
)  # 10000
train_set, _ = data.random_split(train_set, [1000, 59000])
test_set, _ = data.random_split(test_set, [1000, 9000])

train_loader = data.DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)
val_loader = data.DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=4)

for width in range(10, 100, 30):
    # width = 1000
    depth = 5
    epochs = 10
    val_interval = 1
    id = f"d{depth}_w{width}"
    data_file = f"test_results/{id}.jsonl"
    log_file = f"test_results/{id}.txt"
    logger = DataLogger(id, data_file, log_file)
    model = MLP(28 * 28, 10, depth, width)
    model = model.to(device)
    train_classifier(model, train_loader, val_loader, logger, epochs, val_interval)
