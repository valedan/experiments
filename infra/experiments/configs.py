"""
Handles the translation from experiment config files to runnable tasks.

Config Dictionary Schema:
------------------------
RunConfig: {
    # Required fields marked with *

    *task: "supervised_training" (only task type currently supported)
    enable_profiling: bool
    device: str
    trainer: {
        val_interval: int,
        epochs: int,
    }
    model: {
        *architecture: "mlp" | "transformer",
        use_dnn: bool,

        # For MLP architecture:
        *width: int,
        *depth: int,
        *input_dim: int,
        *output_dim: int,

        # For Transformer architecture:
        *d_model: int,
        *nhead: int,
        *num_decoder_layers: int,
        *dim_feedforward: int,
        *vocab_size: int,
        *context_size: int,
    }
    dataset: {
        *name: str,
        train_frac: float,
        val_frac: float,
        test_frac: float,
        preload: bool
    },
    loader: {
        train_batch: int,
        val_batch: int,
        test_batch: int,
        num_workers: int,
        persistent_workers: bool,
        pin_memory: bool,
        prefetch_factor: int,
    }
    logger: {
        flush_interval: int,
        aggregation_interval: int,
        metrics: list[str] (names of functions in metrics.py)
    }
    tokenizer: {  # Required if loader.dataset is "tinystories"
        vocab_size: int,
        max_train_items: int
    }
    criterion: {
        *name: "cross_entropy",
        ignore_index: int
    }
    optimizer: {
        *name: "sgd" | "adam",
        learning_rate: float
    }
}

All fields are optional unless marked with *. Component-level defaults will be used for missing optional fields.

Why is all the task and component config stored as part of the run config? Couldn't the run be initialized
with the task it needs to run, and just the run-specific config? No, because the components can be
resource-intensive once initialized (particularly models), so tasks (which rely on components) should only
be initialized once the run is started, not when the run is initialized. An experiment can initialize
dozens of runs and then start them sequentially, so each run must contain all config needed to construct
the task.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from infra.components import models
from infra.components.loaders import create_dataloaders, create_torchvision_datasets

torchvision_datasets = ["mnist", "cifar10"]


def build_loaders_from_config(
    dataset_config: dict, loader_config: dict
) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    """Returns 3 dataloaders - train, validation, and test"""
    dataset = dataset_config.pop("name")

    if dataset in torchvision_datasets:
        splits = create_torchvision_datasets(dataset, **dataset_config)
    else:
        # TODO: hf datasets
        raise ValueError("dataset not implemented")

    loaders = create_dataloaders(
        *splits, preload=dataset_config.get("preload", False), **loader_config
    )
    return loaders


def build_model_from_config(config: dict) -> nn.Module:
    """Creates a model instance from a config dictionary"""
    architecture = config.pop("architecture")

    match architecture:
        case "mlp":
            return models.MLP(**config)
        case "transformer":
            return models.Transformer(**config)
        case _:
            raise ValueError(f"Unknown architecture {architecture}")


def build_criterion_from_config(config: dict) -> nn.Module:
    criterions = {"cross_entropy": nn.CrossEntropyLoss}
    criterion = criterions[config["criterion"]["name"]](
        ignore_index=config["criterion"].get("ignore_index")
    )
    return criterion


def build_optimizer_from_config(config: dict, model: nn.Module) -> nn.Module:
    optimizers = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
    optimizer = optimizers[config["optimizer"]["name"]](
        model.parameters(), lr=config["optimizer"]["learning_rate"], fused=True
    )
    return optimizer
