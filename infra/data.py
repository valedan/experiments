from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader


def mnist_train_set(train_frac=1, val_frac=0):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_set, val_set, _ = random_split(
        train_set, [train_frac, val_frac, 1 - train_frac - val_frac]
    )

    return train_set, val_set


def mnist_train_loaders(
    train_frac=1, val_frac=0, train_batch=64, val_batch=64, train_set=None, val_set=None
):
    if train_set is None and val_set is None:
        train_set, val_set = mnist_train_set(train_frac, val_frac)
    if train_frac and train_set:
        train_loader = DataLoader(
            train_set,
            batch_size=train_batch,
            shuffle=True,
            num_workers=1,
            persistent_workers=True,
        )
    if val_frac and val_set:
        val_loader = DataLoader(
            val_set,
            batch_size=val_batch,
            shuffle=False,
            num_workers=1,
            persistent_workers=True,
        )
    return train_loader, val_loader
