from torchvision import datasets, transforms
import torch as t
from torch.utils.data import TensorDataset, random_split, DataLoader


def mnist_train_set(train_frac=1, val_frac=0):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]
    )
    train_set = datasets.MNIST(root="./.data", train=True, download=True)
    # TODO: is this actually sent to cuda??
    data = t.stack([transform(img) for img, _ in train_set]).to('cuda')
    targets = t.tensor(train_set.targets).to('cuda')
    dataset = TensorDataset(data, targets)

    generator = t.Generator().manual_seed(42)
    #TODO: do the split before the preprocessing
    train_set, val_set, _ = random_split(
        dataset, [train_frac, val_frac, 1 - train_frac - val_frac], generator=generator
    )

    return train_set, val_set


def mnist_train_loaders(
    train_frac=1, val_frac=0, train_batch=64, val_batch=64, train_set=None, val_set=None
):
    num_workers = 4
    if train_set is None and val_set is None:
        train_set, val_set = mnist_train_set(train_frac, val_frac)
    if train_frac and train_set:
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch,
            shuffle=True,
            # num_workers=num_workers,
            # persistent_workers=True,
            # pin_memory=True,
        )
    if val_frac and val_set:
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch,
            shuffle=False,
            # num_workers=num_workers,
            # persistent_workers=True,
            # pin_memory=True,
        )
    return train_loader, val_loader
