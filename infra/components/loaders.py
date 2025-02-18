from collections.abc import Iterable
from typing import Any

import datasets as hf_datasets
import torch as t
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from torchvision import datasets as torch_datasets
from torchvision import transforms as torch_transforms

DATA_DIR = "/home/dan/.data"


def preload_dataset_to_gpu(dataset: Subset, transform):
    if len(dataset) == 0:
        return dataset

    data = t.stack([transform(dataset.dataset[i][0]) for i in dataset.indices]).to("cuda")
    targets = t.tensor([dataset.dataset[i][1] for i in dataset.indices]).to("cuda")

    preloaded_dataset = TensorDataset(data, targets)
    return preloaded_dataset


def split_dataset(dataset: Dataset, fractions: Iterable[float]):
    # generally we'll want train/val splits to be consistent
    generator = t.Generator().manual_seed(42)

    # need to explicitly set a remainder, without it, random_split will add any remainder to the splits if the sum of fractions is below 1
    remainder = round(1 - sum(fractions), 3)
    splits = random_split(dataset, [*fractions, remainder], generator=generator)
    return splits


def create_dataloaders(
    train_set: Dataset | None = None,
    val_set: Dataset | None = None,
    test_set: Dataset | None = None,
    train_batch: int = 128,
    val_batch: int = 1024,
    test_batch: int = 1024,
    preload: bool = False,
    num_workers: int = 4,
    persistent_workers: bool = True,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
):
    loader_config = {}
    if not preload:
        loader_config = {
            "num_workers": num_workers,
            "persistent_workers": persistent_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor,
        }

    train_loader, val_loader, test_loader = None, None, None
    if train_set:
        train_loader = DataLoader(train_set, batch_size=train_batch, shuffle=True, **loader_config)

    if val_set:
        val_loader = DataLoader(val_set, batch_size=val_batch, shuffle=False, **loader_config)

    if test_set:
        test_loader = DataLoader(test_set, batch_size=test_batch, shuffle=False, **loader_config)

    return train_loader, val_loader, test_loader


def create_torchvision_datasets(
    dataset_name: str,
    transforms: Iterable[Any] | None = None,
    train_frac: float = 1.0,
    val_frac: float = 1.0,
    test_frac: float = 0.0,
    preload: bool = False,
):
    # at some point we will need to add additional transforms for certain datasets, eg imagenet
    transforms = transforms or []
    composed_transform = torch_transforms.Compose([*transforms, torch_transforms.ToTensor()])

    dataset_kwargs = {
        "transform": composed_transform if not preload else None,
        "root": DATA_DIR,
        "download": True,
    }

    # need to handle each dataset initialization separately because torchvision datasets differ in their constructor apis
    match dataset_name:
        case "mnist":
            train_set = torch_datasets.MNIST(train=True, **dataset_kwargs)
            test_set = torch_datasets.MNIST(train=False, **dataset_kwargs)
        case "cifar10":
            train_set = torch_datasets.CIFAR10(train=True, **dataset_kwargs)
            test_set = torch_datasets.CIFAR10(train=False, **dataset_kwargs)
        case _:
            raise ValueError("unsupported dataset")

    train_set, val_set, _ = split_dataset(train_set, [train_frac, val_frac])
    test_set, _ = split_dataset(test_set, [test_frac])

    if preload:
        train_set = preload_dataset_to_gpu(train_set, composed_transform)
        val_set = preload_dataset_to_gpu(val_set, composed_transform)
        test_set = preload_dataset_to_gpu(test_set, composed_transform)

    return train_set, val_set, test_set


# class TinyStoriesDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset["text"]

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         return item

#     # TODO: check claude claim that consistent seq len across batches matters for gpu perf


# def tinystories_loaders(
#     train_frac: float = 1.0,
#     val_frac: float = 1.0,
#     test_frac: float = 0.0,
#     train_batch: int = 128,
#     val_batch: int = 1024,
#     test_batch: int = 1024,
#     preload: bool = False,
#     vocab_size: int = 2048,
#     max_train_items: int = 1000,
# ):
#     # TODO: this eats 50gb ram
#     # TODO: need to apply train_frac
#     train_set = TinyStoriesDataset(
#         hf_datasets.load_dataset("roneneldan/TinyStories", split="train").with_format("torch")
#     )
#     tokenizer = Tokenizer.train(train_set[: max_train_items or -1], vocab_size)
#     val_set = TinyStoriesDataset(
#         hf_datasets.load_dataset("roneneldan/TinyStories", split="validation").with_format("torch")
#     )
#     # TODO: merge this with split_dataset
#     test_set = Subset(val_set, range(20000, 21990))
#     val_set = Subset(val_set, range(20000))

#     def collate_fn(data):
#         tokens = t.tensor(tokenizer.encode_batch(data))
#         # TODO: this is wrong
#         labels = tokens.view(-1)

#         return tokens, labels

#     loader_config = {
#         "num_workers": 6,
#         "persistent_workers": True,
#         "pin_memory": True,
#         "prefetch_factor": 2,
#     }

#     train_loader = DataLoader(
#         train_set,
#         batch_size=train_batch,
#         shuffle=True,
#         collate_fn=collate_fn,
#         **loader_config,
#     )
#     val_loader = DataLoader(
#         val_set, batch_size=val_batch, shuffle=False, collate_fn=collate_fn, **loader_config
#     )
#     test_loader = DataLoader(
#         test_set,
#         batch_size=test_batch,
#         shuffle=False,
#         collate_fn=collate_fn,
#         **loader_config,
#     )

#     return train_loader, val_loader, test_loader
