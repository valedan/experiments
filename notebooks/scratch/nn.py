# %%

import matplotlib.pyplot as plt
from copy import copy, deepcopy
from pathlib import Path
from infra.loaders import LoaderParams
from infra.run import Run, RunConfig
from infra import analysis

nn_config = RunConfig(
    model={
        "architecture": "mlp",
        "input_dim": 28 * 28,
        "output_dim": 10,
        "width": 1000,
        "depth": 2,
        "use_dnn": False
    },
    loader=LoaderParams(dataset="mnist", train_frac=0.5, val_frac=0.05, preload=True),
    epochs=50,
    val_interval=1000000,
    train_log_interval=10,
)
dnn_config = deepcopy(nn_config)
dnn_config.model['use_dnn'] = True

nn_file = Path("./notebooks/scratch/nn_test.jsonl")
dnn_file = Path("./notebooks/scratch/dnn_test.jsonl")


# Run(config=nn_config, state="new").start(data_file=nn_file, overwrite=True)
Run(config=dnn_config, state="new").start(data_file=dnn_file, overwrite=True)
# %%

palette = analysis.palette
nn_df = analysis.read_run(nn_file, ["train_loss", "train_accuracy"])
dnn_df = analysis.read_run(dnn_file, ["train_loss", "train_accuracy"])
fig, ax = plt.subplots()
ax.plot(nn_df["step"], nn_df["train_loss"], c=palette(0), ls='-', alpha=0.5)
ax.plot(dnn_df["step"], dnn_df["train_loss"], c=palette(2), ls='-', alpha=0.5)
# %%
print(nn_df.iloc[-1]['created_at'] - nn_df.iloc[0]['created_at'])
print(dnn_df.iloc[-1]['created_at'] - dnn_df.iloc[0]['created_at'])
