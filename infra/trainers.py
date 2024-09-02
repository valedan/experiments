from infra.configs import TrainerParams
from infra.logger import DataLogger
import torch as t
from torch import nn
from torch.utils.data import DataLoader
from infra.metrics import multiclass_accuracy


def train_classifier(
    logger: DataLogger,
    model: nn.Module,
    params: TrainerParams,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device="cpu",
):
    assert train_loader and val_loader
    model = model.to(device)
    step = 0
    criterion = nn.CrossEntropyLoss().to(device)
    if params.optimizer == "sgd":
        assert params.learning_rate is not None
        optimizer = t.optim.SGD(model.parameters(), params.learning_rate)
    elif params.optimizer == "adam":
        optimizer = t.optim.Adam(model.parameters())


    for epoch in range(1, params.epochs + 1):
        total_train_loss = 0
        train_accuracies = []
        for batch_idx, (train_images, labels) in enumerate(train_loader):
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

            if step % params.val_interval == 0:
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
                # TODO: add verbose flag
                # logger.log(
                #     f"Batch {batch_idx} - Train Loss {loss.item():.4f} - Val Loss {total_val_loss:.4f}"
                # )

            logger.add(**datum)
            logger.log_progress(len(train_images))
            step += 1
