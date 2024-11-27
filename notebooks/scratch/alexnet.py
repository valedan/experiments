import torch as t
from datasets import load_dataset
from torch import nn
import einops as ein
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import pandas as pd

from infra import data, configs, models, trainers, logger, analysis

loader_config = configs.LoaderConfig(
    train_frac=0.1, val_frac=0.1, test_frac=0.5, train_batch=512, val_batch=512, preload=False
)
train_loader, val_loader, test_loader = data.imagenet_loaders(loader_config)
#%%
# im = train_loader.dataset[0][0]
# print(im.shape)
# plt.imshow(ein.rearrange(im, "c h w -> h w c"))

# len(train_loader.dataset)
#%%
epochs = 20

# data_file_mlp = Path("./notebooks/scratch/imagenet_1.json")
# data_file_lenet = Path("./notebooks/scratch/imagenet_2.json")
# data_file_custom = Path("./notebooks/scratch/imagenet_3.json")
data_file_alexnet = Path("./notebooks/scratch/imagenet_alexnet.json")
# mlp = models.MLP(input_dim=32*32*3, output_dim=10, depth=2, width=4000)
# lenet = models.LeNetModern()
alexnet = models.AlexNet()

with tqdm(
        total=(len(train_loader.dataset) * epochs), desc="Run", unit="samples"
) as run_bar:

    log = logger.DataLogger(
        data_path=data_file_alexnet,
        after_log=run_bar.update,
        overwrite=True
    )

    trainers.train_classifier(
        logger=log,
        model=alexnet,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer="adam",
        val_interval=100,
        epochs=epochs,
        device='cuda'
    )

# %%
# palette = analysis.palette
# df_1 = analysis.read_run(data_file_alexnet, ["train_loss", "train_accuracy"])
# # df_2 = analysis.read_run(data_file_lenet, ["train_loss", "train_accuracy"])
# fig, ax = plt.subplots()
# mask = df_1["val_loss"].notna()
# ax.plot(df_1["step"], df_1["train_loss"], c=palette(0), ls='-', alpha=0.5)
# ax.plot(df_1.loc[mask, "step"], df_1.loc[mask, "val_loss"], c=palette(0), alpha=1)
# ax.plot(df_1.loc[mask, "step"], df_1.loc[mask, "val_accuracy"], c=palette(0), alpha=1)

# # mask = df_2["val_loss"].notna()
# # ax.plot(df_2["step"], df_2["train_loss"], c=palette(2), ls='-', alpha=0.5)
# # ax.plot(df_2.loc[mask, "step"], df_2.loc[mask, "val_loss"], c=palette(2), alpha=1)
# # ax.plot(df_2.loc[mask, "step"], df_2.loc[mask, "val_accuracy"], c=palette(2), alpha=1)

# #%%

# df_1["val_accuracy"].dropna()
# # df_2["val_accuracy"].dropna()
