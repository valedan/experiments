from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from infra.experiments.run import Run
from infra.experiments.experiment import Experiment

# %%
# # sweep api

# params = {
#     'epochs': [10],
#     'model': {
#         'architecture': 'mlp',
#         'input_dim': 784,
#         'output_dim': 10,
#         'depth': [2, 4],
#         'width': [1000, 2000]
#     },
#     'loader': {
#         'dataset': 'zalando-datasets/fashion_mnist',
#     }
# }

# sweep = Sweep(params, './notebooks/scratch/simple_sweep')


# result = await sweep.start()

# if result.failed:
#     raise ValueError('some runs failed')

# df = sweep.read_runs_data()

# groups = df.groupby(['width, depth'])

# fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 14))

# for idx, group in enumerate(groups):
#     ax = axes[idx]
#     ax.set_title(
#         f"Width {group.iloc[0]['width']}, Depth {group.iloc[0]['depth']}"
#     )
#     ax.plot(group["step"], group["batch_train_loss"], label="Train Loss (Batch)")
#     ax.plot(group["step"], group["epoch_val_loss"], label="Val Loss")

# plt.show()

# %% run api

params = {
    "trainer": {
        "epochs": 10,
    },
    "model": {
        "architecture": "mlp",
        "input_dim": 784,
        "output_dim": 10,
        "depth": 2,
        "width": 1000,
    },
    "loader": {
        # 'dataset': 'zalando-datasets/fashion_mnist',
        "dataset": "mnist"
    },
    "logger": {
        'metrics': ['loss', 'accuracy']
    }
}
dir =  "./notebooks/scratch/simple_run"

shutil.rmtree(dir)
run = Run(params,dir)

await run.start()

df = run.read_data()


plt.figure(figsize=(10, 6))
plt.plot(df["step"], df["batch_train_loss"], label="Train Loss (Batch)")
plt.plot(df["step"], df["epoch_val_loss"], label="Val Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.show()
