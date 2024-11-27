from torchvision import datasets as torch_datasets, transforms
import datasets as hf_datasets
import torch as t
from torch.utils.data import TensorDataset, random_split, DataLoader, Subset, Dataset
from dataclasses import dataclass

from infra.tokens import Tokenizer

DATA_DIR = "/home/dan/.data"
# TODO: separate datasets and loaders. need to handle each dataset separately but can convert to a common Dataset format. then just need a couple generic dataloaders.


@dataclass
class TokenizerConfig:
    vocab_size: int
    max_train_items: int = 1000


@dataclass(frozen=True)
class LoaderConfig:
    dataset: str
    train_frac: float = 1.0
    val_frac: float = 1.0
    test_frac: float = 0.0
    train_batch: int = 128
    val_batch: int = 1024
    test_batch: int = 1024
    preload: bool = False


def get_loaders(
    config: LoaderConfig, tokenizer_config: TokenizerConfig | None = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns 3 dataloaders - train, validation, and test"""

    match config.dataset:
        case "mnist":
            return mnist_loaders(config)
        case "cifar":
            return cifar_loaders(config)
        case "imagenet":
            return imagenet_loaders(config)
        case "tinystories":
            assert tokenizer_config
            return tinystories_loaders(config, tokenizer_config=tokenizer_config)
        case _:
            raise ValueError("No valid dataset specified in loader config.")


def preload_dataset(dataset: Subset, transform, device):
    if len(dataset) == 0:
        return dataset

    data = t.stack([transform(dataset.dataset[i][0]) for i in dataset.indices]).to(device)
    targets = t.tensor([dataset.dataset[i][1] for i in dataset.indices]).to(device)

    preloaded_dataset = TensorDataset(data, targets)
    return preloaded_dataset


# TODO just use Subset.indices
def split_dataset(dataset: Dataset, fractions: list[float]):
    print(fractions)
    generator = t.Generator().manual_seed(42)
    # need to explicitly set a remainder, without it, random_split will add any remainder to the splits if the sum of fractions is below 1
    remainder = round(1 - sum(fractions), 3)
    splits = random_split(dataset, [*fractions, remainder], generator=generator)
    return splits


def mnist_loaders(config: LoaderConfig):
    # TODO: Normalize all the vision loaders
    transform = transforms.Compose([transforms.ToTensor()])
    dynamic_transforms = transform if not config.preload else None

    train_set = torch_datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=dynamic_transforms
    )
    test_set = torch_datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=dynamic_transforms
    )

    train_set, _ = split_dataset(train_set, [config.train_frac])
    val_set, test_set, _ = split_dataset(test_set, [config.val_frac, config.test_frac])

    if config.preload:
        train_set = preload_dataset(train_set, transform, "cuda")
        val_set = preload_dataset(val_set, transform, "cuda")
        test_set = preload_dataset(test_set, transform, "cuda")
        loader_config = {}
    else:
        loader_config = {"num_workers": 4, "persistent_workers": True, "pin_memory": True}

    train_loader = DataLoader(
        train_set, batch_size=config.train_batch, shuffle=True, **loader_config
    )
    val_loader = DataLoader(val_set, batch_size=config.val_batch, shuffle=False, **loader_config)
    test_loader = DataLoader(test_set, batch_size=config.test_batch, shuffle=False, **loader_config)

    return train_loader, val_loader, test_loader


# TODO: reduce duplication with mnist
def cifar_loaders(config: LoaderConfig):
    transform = transforms.Compose([transforms.ToTensor()])
    dynamic_transforms = transform if not config.preload else None

    train_set = torch_datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=dynamic_transforms
    )
    test_set = torch_datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=dynamic_transforms
    )

    train_set, _ = split_dataset(train_set, [config.train_frac])
    val_set, test_set, _ = split_dataset(test_set, [config.val_frac, config.test_frac])

    if config.preload:
        train_set = preload_dataset(train_set, transform, "cuda")
        val_set = preload_dataset(val_set, transform, "cuda")
        test_set = preload_dataset(test_set, transform, "cuda")
        loader_config = {}
    else:
        loader_config = {"num_workers": 4, "persistent_workers": True, "pin_memory": True}

    train_loader = DataLoader(
        train_set, batch_size=config.train_batch, shuffle=True, **loader_config
    )
    val_loader = DataLoader(val_set, batch_size=config.val_batch, shuffle=False, **loader_config)
    test_loader = DataLoader(test_set, batch_size=config.test_batch, shuffle=False, **loader_config)

    return train_loader, val_loader, test_loader


def imagenet_loaders(config: LoaderConfig):
    ops = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    train_set = torch_datasets.ImageFolder(f"{DATA_DIR}/imagenet-1k-parsed/train", ops)
    val_set = torch_datasets.ImageFolder(f"{DATA_DIR}/imagenet-1k-parsed/val", ops)
    # TODO: merge this with split_dataset
    test_set = Subset(val_set, range(40000, 50000))
    val_set = Subset(val_set, range(40000))

    loader_config = {
        "num_workers": 12,
        "persistent_workers": True,
        "pin_memory": True,
        "prefetch_factor": 4,
    }

    train_loader = DataLoader(
        train_set, batch_size=config.train_batch, shuffle=False, **loader_config
    )
    val_loader = DataLoader(val_set, batch_size=config.val_batch, shuffle=False, **loader_config)
    test_loader = DataLoader(test_set, batch_size=config.test_batch, shuffle=False, **loader_config)

    return train_loader, val_loader, test_loader


class TinyStoriesDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset["text"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item

    # TODO: check claude claim that consistent seq len across batches matters for gpu perf


def tinystories_loaders(config: LoaderConfig, tokenizer_config: TokenizerConfig):
    #TODO: this eats 50gb ram
    # TODO: need to apply train_frac
    train_set = TinyStoriesDataset(
        hf_datasets.load_dataset("roneneldan/TinyStories", split="train").with_format("torch")
    )
    tokenizer = Tokenizer.train(
        train_set[: tokenizer_config.max_train_items or -1], tokenizer_config.vocab_size
    )
    val_set = TinyStoriesDataset(
        hf_datasets.load_dataset("roneneldan/TinyStories", split="validation").with_format("torch")
    )
    # TODO: merge this with split_dataset
    test_set = Subset(val_set, range(20000, 21990))
    val_set = Subset(val_set, range(20000))

    def collate_fn(data):
        tokens = t.tensor(tokenizer.encode_batch(data))
        # TODO: this is wrong
        labels = tokens.view(-1)

        return tokens, labels

    loader_config = {
        "num_workers": 6,
        "persistent_workers": True,
        "pin_memory": True,
        "prefetch_factor": 2,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=config.train_batch,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_config,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.val_batch, shuffle=False, collate_fn=collate_fn, **loader_config
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config.test_batch,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_config,
    )

    return train_loader, val_loader, test_loader
