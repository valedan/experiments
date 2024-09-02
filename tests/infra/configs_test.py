import yaml
from infra.configs import ConfigLoader, RunConfig

complex_config = """
exp:
  type: 'mnist'
model:
  width: [10, 20]
  depth: [1, 2, 3]
trainer:
  val_interval: [1]
  optimizer: ['adam']
  epochs:
    derive_from: "model.depth"
    operation: "multiply"
    value: 10
    offset: 10
loader:
  train_frac: [0.01]
  val_frac: 0.05
  train_batch: [64]
  val_batch: 64
"""


def test_complex_config():
    configs = ConfigLoader(yaml.safe_load(complex_config))
    assert isinstance(configs[0], RunConfig)
    assert len(configs) == 6
    assert set([c.trainer.epochs for c in configs]) == set([20, 30, 40])
    assert set([c.model.width for c in configs]) == set([10, 20])
    assert set([c.model.depth for c in configs]) == set([1, 2, 3])
    assert configs[0].loader.val_batch == 64
    assert configs[0].trainer.optimizer == "adam"
    assert configs[0].model.input_dim == 28 * 28
