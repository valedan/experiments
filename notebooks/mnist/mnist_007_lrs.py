
# %%

from scipy.signal import savgol_filter
from itertools import product
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from infra.analysis import get_sweep_data, get_sweep_data_for_experiments, initialize_plots_dir

exp_name = "mnist/007"
plots_dir = initialize_plots_dir(exp_name)
df = get_sweep_data_for_experiments(['mnist/005', 'mnist/006', 'mnist/007', 'mnist/008'])
# %%
df.loc[df['optimizer'].isna(), 'optimizer'] = 'sgd'
# drop the old high lrs
df = df[~((df['learning_rate'] >= 0.5) & (df['exp_name'] != 'mnist/008'))]
# df.loc[df['learning_rate'].isna(), 'learning_rate'] = 1e-3

# %%
# df['effective_step'] = df['step'] * df['learning_rate']
models = df.groupby(['width', 'depth'])
w_count = len(df["width"].unique())
d_count = len(df["depth"].unique())

palette = mpl.colormaps['tab20']
#%%
fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

axes = axes.flatten()
fig.suptitle(f"Loss - All LRs all Models")
axes[0].set_ylim(0, 3)
for model_idx, (_, model_df) in enumerate(models):
    lrs = model_df.groupby(['optimizer', 'learning_rate'], dropna=False)
    ax = axes[model_idx]
    # fig, ax = plt.subplots(figsize=(14,14))
    # ax.set_ylim(0,3)
    # ax.set_xlim(0, 1000)
    for lr_idx, ((optim, lr), lr_df) in enumerate(lrs):
        label = f"{optim}_{lr}"
        color_train = palette(lr_idx * 2 / 20)
        color_val = palette((lr_idx * 2 + 1) / 20)
        ax.set_title(
            f"Width {lr_df.iloc[0]['width']}, Depth {lr_df.iloc[0]['depth']}"
        )
        ax.plot(lr_df["step"], lr_df["batch_train_loss"], label=f"Train Loss - {label}", c=color_train, ls='-' )
        ax.plot(lr_df["step"], lr_df["epoch_val_loss"], label=f"Val Loss - {label}", c=color_val, ls=':')

# plt.legend()
plt.savefig(plots_dir / f"loss_all_lrs_all_models.png")
plt.show()
#%%
df2 = df[df['learning_rate'] > 0.001]
df2 = df2[df2['learning_rate'] < 0.5]
df2 = df2[df2['optimizer'] == 'sgd']
print(df2['learning_rate'].unique())
window_length = 51
poly_order = 3
df2['smoothed_train_loss'] = savgol_filter(df2['batch_train_loss'], window_length, poly_order)
df2['smoothed_val_loss'] = savgol_filter(df2['epoch_val_loss'], window_length, poly_order)
df2['smoothed_train_accuracy'] = savgol_filter(df2['batch_train_accuracy'], window_length, poly_order)
df2['smoothed_val_accuracy'] = savgol_filter(df2['epoch_val_accuracy'], window_length, poly_order)
models2 = df2.groupby(['width', 'depth'])
# fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

# axes = axes.flatten()
# fig.suptitle(f"Loss - All LRs all Models")
# axes[0].set_ylim(0, 3)
for model_idx, (_, model_df) in enumerate(models2):
    lrs = model_df.groupby(['optimizer', 'learning_rate'], dropna=False)
    # ax = axes[model_idx]
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_ylim(0,3)
    ax.set_xlim(0, 10000)
    for lr_idx, ((optim, lr), lr_df) in enumerate(lrs):
        label = f"{optim}_{lr}"
        color_train = palette(lr_idx * 2 / 20)
        color_val = palette((lr_idx * 2 + 1) / 20)
        ax.set_title(
            f"Width {lr_df.iloc[0]['width']}, Depth {lr_df.iloc[0]['depth']}"
        )
        ax.plot(lr_df["step"], lr_df["smoothed_train_loss"], label=f"Train Loss - {label}", c=color_train, ls='-' )
        ax.plot(lr_df["step"], lr_df["smoothed_val_loss"], label=f"Val Loss - {label}", c=color_val, ls='--')
        ax.plot(lr_df["step"], lr_df["batch_train_loss"], c=color_train, ls='-', alpha=0.25)
        ax.plot(lr_df["step"], lr_df["epoch_val_loss"], c=color_val, ls=':', alpha=0.25)
        # ax.plot(lr_df["step"], lr_df["smoothed_train_accuracy"], label=f"Train Acc - {label}", c=color_train, ls='-' )
        # ax.plot(lr_df["step"], lr_df["smoothed_val_accuracy"], label=f"Val Acc - {label}", c=color_val, ls='--')
        # ax.plot(lr_df["step"], lr_df["batch_train_accuracy"], c=color_train, ls='-', alpha=0.25)
        # ax.plot(lr_df["step"], lr_df["epoch_val_accuracy"], c=color_val, ls='--', alpha=0.25)

    plt.legend()
    plt.savefig(plots_dir / f"loss_reasonable_lrs_{model_idx}.png")
    plt.show()

# %%
#
df3 = df[df['learning_rate'] > 0.025]
df3 = df3[df3['learning_rate'] < 0.5]
df3 = df3[df3['optimizer'] == 'sgd']
print(df3['learning_rate'].unique())
window_length = 51
poly_order = 3
df3['smoothed_train_loss'] = savgol_filter(df3['batch_train_loss'], window_length, poly_order)
df3['smoothed_val_loss'] = savgol_filter(df3['epoch_val_loss'], window_length, poly_order)
models3 = df3.groupby(['width', 'depth'])
# fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

# axes = axes.flatten()
# fig.suptitle(f"Loss - All LRs all Models")
# axes[0].set_ylim(0, 3)
for model_idx, (_, model_df) in enumerate(models3):
    lrs = model_df.groupby(['optimizer', 'learning_rate'], dropna=False)
    # ax = axes[model_idx]
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_ylim(0,3)
    ax.set_xlim(0, 2000)
    for lr_idx, ((optim, lr), lr_df) in enumerate(lrs):
        label = f"{optim}_{lr}"
        color_train = palette(lr_idx * 2 / 20)
        color_val = palette((lr_idx * 2 + 1) / 20)
        ax.set_title(
            f"Width {lr_df.iloc[0]['width']}, Depth {lr_df.iloc[0]['depth']}"
        )
        ax.plot(lr_df["step"], lr_df["smoothed_train_loss"], label=f"Train Loss - {label}", c=color_train, ls='-' )
        ax.plot(lr_df["step"], lr_df["smoothed_val_loss"], label=f"Val Loss - {label}", c=color_val, ls='--')
        ax.plot(lr_df["step"], lr_df["batch_train_loss"], c=color_train, ls='-', alpha=0.25)
        ax.plot(lr_df["step"], lr_df["epoch_val_loss"], c=color_val, ls=':', alpha=0.25)

    plt.legend()
    plt.savefig(plots_dir / f"loss_mid_lrs_{model_idx}.png")
    plt.show()
# %%

df4 = df[df['learning_rate'] > 0.1]
df4 = df4[df4['optimizer'] == 'sgd']
print(df4['learning_rate'].unique())
window_length = 51
poly_order = 3
df4['smoothed_train_loss'] = savgol_filter(df4['batch_train_loss'], window_length, poly_order)
df4['smoothed_val_loss'] = savgol_filter(df4['epoch_val_loss'], window_length, poly_order)
models4 = df4.groupby(['width', 'depth'])
# fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

# axes = axes.flatten()
# fig.suptitle(f"Loss - All LRs all Models")
# axes[0].set_ylim(0, 3)
for model_idx, (_, model_df) in enumerate(models4):
    lrs = model_df.groupby(['optimizer', 'learning_rate'], dropna=False)
    # ax = axes[model_idx]
    fig, ax = plt.subplots(figsize=(14,14))
    ax.set_ylim(0,5)
    ax.set_xlim(0, 2000)
    for lr_idx, ((optim, lr), lr_df) in enumerate(lrs):
        label = f"{optim}_{lr}"
        color_train = palette(lr_idx * 2 / 20)
        color_val = palette((lr_idx * 2 + 1) / 20)
        ax.set_title(
            f"Width {lr_df.iloc[0]['width']}, Depth {lr_df.iloc[0]['depth']}"
        )
        ax.plot(lr_df["step"], lr_df["smoothed_train_loss"], label=f"Train Loss - {label}", c=color_train, ls='-' )
        ax.plot(lr_df["step"], lr_df["smoothed_val_loss"], label=f"Val Loss - {label}", c=color_val, ls='--')
        ax.plot(lr_df["step"], lr_df["batch_train_loss"], c=color_train, ls='-', alpha=0.25)
        ax.plot(lr_df["step"], lr_df["epoch_val_loss"], c=color_val, ls=':', alpha=0.25)

    plt.legend()
    plt.savefig(plots_dir / f"loss_big_lrs_{model_idx}.png")
    plt.show()
#%%

dfa = df[((df['optimizer'] == 'adam') | (df['learning_rate'] == 0.1))]
print(dfa.head())
window_length = 51
poly_order = 3
dfa['smoothed_train_loss'] = savgol_filter(dfa['batch_train_loss'], window_length, poly_order)
dfa['smoothed_val_loss'] = savgol_filter(dfa['epoch_val_loss'], window_length, poly_order)
dfa['smoothed_train_accuracy'] = savgol_filter(dfa['batch_train_accuracy'], window_length, poly_order)
dfa['smoothed_val_accuracy'] = savgol_filter(dfa['epoch_val_accuracy'], window_length, poly_order)
modelsa = dfa.groupby(['width', 'depth'])
fig, axes = plt.subplots(w_count, d_count, sharex=True, sharey=True, figsize=(14, 14))

axes = axes.flatten()
fig.suptitle(f"Loss - Adam vs best sgd lr")
axes[0].set_ylim(0, 1)
for model_idx, (_, model_df) in enumerate(modelsa):
    adam = model_df[model_df['optimizer'] == 'adam']
    sgd = model_df[model_df['optimizer'] != 'adam']
    ax = axes[model_idx]
    # fig, ax = plt.subplots(figsize=(14,14))
    # ax.set_ylim(0,3)
    # ax.set_xlim(0, 10000)
    label = "adam"
    color_train = palette(0)
    color_val = palette(1)
    sgd_color_train = palette(2)
    sgd_color_val = palette(3)
    ax.set_title(
        f"Width {model_df.iloc[0]['width']}, Depth {model_df.iloc[0]['depth']}"
    )
    # ax.plot(adam["step"], adam["smoothed_train_loss"], label=f"Train Loss - adam", c=color_train, ls='-' )
    # ax.plot(adam["step"], adam["smoothed_val_loss"], label=f"Val Loss - adam", c=color_val, ls='--')
    # ax.plot(adam["step"], adam["batch_train_loss"], c=color_train, ls='-', alpha=0.25)
    # ax.plot(adam["step"], adam["epoch_val_loss"], c=color_val, ls='--', alpha=0.25)
    # ax.plot(sgd["step"], sgd["smoothed_train_loss"], label=f"Train Loss - sgd", c=sgd_color_train, ls='-' )
    # ax.plot(sgd["step"], sgd["smoothed_val_loss"], label=f"Val Loss - sgd", c=sgd_color_val, ls='--')
    # ax.plot(sgd["step"], sgd["batch_train_loss"], c=sgd_color_train, ls='-', alpha=0.25)
    # ax.plot(sgd["step"], sgd["epoch_val_loss"], c=sgd_color_val, ls='--', alpha=0.25)
    ax.plot(adam["step"], adam["smoothed_train_accuracy"], label=f"Train accuracy - adam", c=color_train, ls='-' )
    ax.plot(adam["step"], adam["smoothed_val_accuracy"], label=f"Val accuracy - adam", c=color_val, ls='--')
    ax.plot(adam["step"], adam["batch_train_accuracy"], c=color_train, ls='-', alpha=0.25)
    ax.plot(adam["step"], adam["epoch_val_accuracy"], c=color_val, ls='--', alpha=0.25)
    ax.plot(sgd["step"], sgd["smoothed_train_accuracy"], label=f"Train accuracy - sgd", c=sgd_color_train, ls='-' )
    ax.plot(sgd["step"], sgd["smoothed_val_accuracy"], label=f"Val accuracy - sgd", c=sgd_color_val, ls='--')
    ax.plot(sgd["step"], sgd["batch_train_accuracy"], c=sgd_color_train, ls='-', alpha=0.25)
    ax.plot(sgd["step"], sgd["epoch_val_accuracy"], c=sgd_color_val, ls='--', alpha=0.25)

plt.legend()
plt.savefig(plots_dir / f"loss_adam_vs_sgd.png")
plt.show()
