from infra.loaders import TokenizerConfig, create_loaders_from_config, LoaderConfig
from datasets import load_dataset
from infra import tokens

base_loader_params = {
    "train_frac": 0.1,
    "val_frac": 0.1,
    "test_frac": 0.1,
    "train_batch": 10,
    "test_batch": 10,
    "val_batch": 10,
}


def get_loader_batches(dataset, loader_args=None, n_expected_loaders=3):
    loader_args = loader_args or {}
    config = LoaderConfig(
        **base_loader_params,
        dataset=dataset,
    )

    loaders = create_loaders_from_config(config, **loader_args)

    if n_expected_loaders:
        assert len(loaders) == n_expected_loaders

    for loader in loaders:
        yield next(iter(loader))


def test_mnist():
    for batch, labels in get_loader_batches("mnist"):
        assert batch.shape == (10, 1, 28, 28)
        assert labels.shape == (10,)


def test_cifar():
    for batch, labels in get_loader_batches("cifar"):
        assert batch.shape == (10, 3, 32, 32)
        assert labels.shape == (10,)


def test_imagenet():
    for batch, labels in get_loader_batches("imagenet"):
        assert batch.shape == (10, 3, 224, 224)
        assert labels.shape == (10,)


def test_tinystories():
    for batch, labels in get_loader_batches(
        "tinystories", {"tokenizer_config": TokenizerConfig(vocab_size=100)}
    ):
        assert labels.shape == (10 * batch.shape[-1],)
