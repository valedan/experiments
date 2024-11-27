import yaml
from infra.sweep import create_run_configs

simple_config = """
model:
  architecture: 'mlp'
  width: [10, 20]
  depth: 2
  input_dim: 768
  output_dim: 10
loader:
  dataset: 'mnist'
"""
def test_simple_config():
    configs = create_run_configs(yaml.safe_load(simple_config))
    assert len(configs) == 2
    assert set([c.model['width'] for c in configs]) == set([10, 20])

complex_config = """
model:
  architecture: 'mlp'
  width: [10, 20]
  input_dim: 768
  output_dim: 10
val_interval: [1]
epochs: 2
criterion: 'cross_entropy'
optimizer:
  name: ['adam']
loader:
  dataset: 'mnist'
  train_frac: [0.01]
  val_frac: 0.05
  train_batch: [64]
  val_batch: 64
paired:
  names: ['model.depth', 'epochs']
  values: [[1, 10], [2, 20], [3, 30]]
"""


def test_complex_config():
    configs = create_run_configs(yaml.safe_load(complex_config))
    assert len(configs) == 6
    assert set([c.epochs for c in configs]) == set([10, 20, 30])
    assert set([c.model['width'] for c in configs]) == set([10, 20])
    assert set([c.model['depth'] for c in configs]) == set([1, 2, 3])
    assert configs[0].loader.val_batch == 64
    assert configs[0].loader.train_batch == 64
    assert configs[0].optimizer.name == "adam"
