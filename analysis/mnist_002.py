# %%

import pandas as pd

import matplotlib.pyplot as plt
from pathlib import Path

plots_dir = Path("/home/dan/Dropbox/projects/exploration/analysis/002")
plots_dir.mkdir(exist_ok=True)
exp_dir = Path("/home/dan/Dropbox/projects/exploration/experiments/mnist/002")
runs_dir = exp_dir / "runs"
pd.options.display.max_rows = 10

params_df = pd.read_json(exp_dir / "runs.jsonl", lines=True)
params_df["train_frac"] = params_df["run_name"].str[-4:]
params_df["val_frac"] = 0.05
params_df["train_batch"] = 64
params_df["val_batch"] = 64


def read_run(filename):
    return pd.read_json(filename, lines=True)


def read_runs(dir):
    run_files = list(dir.glob("*.jsonl"))
    df = pd.DataFrame()
    for f in run_files:
        new_data = read_run(f)

        df = pd.concat([df, new_data])
    return df


df = read_runs(runs_dir)
df = pd.merge(df, params_df, left_on="id", right_on="run_id")
df = df.drop(columns=["run_name", "created_at", "run_id"])
# frac_dfs = {frac: group for frac, group in df.groupby("train_frac")}

# %%
# for frac, df in frac_dfs.items():
w_count = len(df["width"].unique())
d_count = len(df["depth"].unique())
df = df.groupby(["width", "depth"])

fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

frac = 0.01
axes = axes.flatten()
fig.suptitle(f"Loss - Train frac {frac}")

for idx, (run_id, run_data) in enumerate(df):
    ax = axes[idx]
    ax.set_title(
        f"Width {run_data.iloc[0]['width']}, Depth {run_data.iloc[0]['depth']}"
    )
    ax.plot(run_data["step"], run_data["batch_train_loss"], label="Train Loss (Batch)")
    ax.plot(run_data["step"], run_data["epoch_val_loss"], label="Val Loss")

plt.savefig(plots_dir / f"loss_{frac}.png")
plt.show()
