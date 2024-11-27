import torch as t
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd

# import mnist_001
from infra import data, configs, models, trainers, logger, analysis

loader_config = configs.LoaderConfig(
    train_frac=1, val_frac=0.5, test_frac=0.5, train_batch=512, val_batch=10000, preload=True
)
train_loader, val_loader, test_loader = data.mnist_loaders(loader_config, 'cuda')
epochs = 20

data_file_mlp = Path("./notebooks/scratch/cnns_1.json")
data_file_lenet = Path("./notebooks/scratch/cnns_2.json")
data_file_lenet_modern = Path("./notebooks/scratch/cnns_3.json")
# model = models.MLP(input_dim=28*28, output_dim=10, depth=2, width=2000)
# model = models.LeNet()
model = models.LeNetModern()

with tqdm(
    total=(len(train_loader.dataset) * epochs), desc="Run", unit="samples"
) as run_bar:

    log = logger.DataLogger(
        data_path=data_file_lenet_modern,
        after_log=run_bar.update,
        overwrite=True
    )

    trainers.train_classifier(
        logger=log,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer="adam",
        val_interval=100,
        epochs=epochs,
        device='cuda'
    )

# %%
palette = analysis.palette
df_1 = analysis.read_run(data_file_lenet, ["train_loss", "train_accuracy"])
df_2 = analysis.read_run(data_file_lenet_modern, ["train_loss", "train_accuracy"])
fig, ax = plt.subplots()
mask = df_1["val_loss"].notna()
ax.plot(df_1["step"], df_1["train_loss"], c=palette(0), ls='-', alpha=0.5)
ax.plot(df_1.loc[mask, "step"], df_1.loc[mask, "val_loss"], c=palette(0), alpha=1)
ax.plot(df_1.loc[mask, "step"], df_1.loc[mask, "val_accuracy"], c=palette(0), alpha=1)

mask = df_2["val_loss"].notna()
ax.plot(df_2["step"], df_2["train_loss"], c=palette(2), ls='-', alpha=0.5)
ax.plot(df_2.loc[mask, "step"], df_2.loc[mask, "val_loss"], c=palette(2), alpha=1)
ax.plot(df_2.loc[mask, "step"], df_2.loc[mask, "val_accuracy"], c=palette(2), alpha=1)

#%%

df_1["val_accuracy"].dropna()
df_2["val_accuracy"].dropna()
