
# %%

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from infra.analysis import get_sweep_data, initialize_plots_dir

exp_name = "mnist/005"
plots_dir = initialize_plots_dir(exp_name)
df = get_sweep_data(exp_name)

lrs = df.groupby("learning_rate")
for lr, df in lrs:
    w_count = len(df["width"].unique())
    d_count = len(df["depth"].unique())
    df = df.groupby(["width", "depth"])

    fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

    frac = 0.01
    axes = axes.flatten()
    fig.suptitle(f"Loss - Train frac {frac}, LR {lr}")
    if lr == 0.1:
        axes[0].set_ylim(0, 2.5)
    else:
        axes[0].set_ylim(0, 6)

    for idx, (run_id, run_data) in enumerate(df):
        ax = axes[idx]
        ax.set_title(
            f"Width {run_data.iloc[0]['width']}, Depth {run_data.iloc[0]['depth']}"
        )
        ax.plot(run_data["step"], run_data["batch_train_loss"], label="Train Loss (Batch)")
        ax.plot(run_data["step"], run_data["epoch_val_loss"], label="Val Loss")

    plt.savefig(plots_dir / f"loss_{frac}_{lr}.png")
    plt.show()
