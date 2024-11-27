
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from infra.analysis import get_sweep_data, initialize_plots_dir, metrics

palette = mpl.colormaps['tab20']
exp_name = "mnist/014"
plots_dir = initialize_plots_dir(exp_name)
df = get_sweep_data(exp_name, cols_to_smooth=metrics.keys())

print(df.head())
param_counts = df.groupby('exp.total_params')

for param_count, param_df in param_counts:
    fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(14,14))
    fig.suptitle(f"{param_count} params")
    axes = axes.flatten()
    axes[0].set_ylim(0,3)
    depths = param_df.groupby('model.depth')
    for idx, (depth, ddf) in enumerate(depths):
        #TODO log scales
        ax = axes[idx]
        width = ddf.iloc[0]['model.width']
        assert len(ddf['model.width'].unique()) == 1
        ax.set_title(f"Depth {depth}, Width {width}")
        ax.plot(ddf["step"], ddf["batch_train_loss"], c=palette(0), ls='-', alpha=0.25)
        ax.plot(ddf["step"], ddf["smoothed_batch_train_loss"],  c=palette(0), ls='--')
        # ax.plot(ddf["step"], ddf["epoch_val_loss"], c=palette(2), ls='--', alpha=0.25)
        ax.plot(ddf["step"], ddf["smoothed_epoch_val_loss"],  c=palette(2), ls='--')
        ax.plot(ddf["step"], ddf["epoch_val_accuracy"], c=palette(4), ls='-', )
        ax.plot(ddf["step"], ddf["smoothed_batch_train_accuracy"], c=palette(6), ls='-', alpha=0.5)
        # ax.plot(ddf['step'], ddf['batch_train_loss'])
        # ax.plot(ddf['step'], ddf['epoch_val_loss'], ls='--')
    plt.show()




#%%

df[(df['total_params'] == 5000) & (df['depth'] == 2)][['step', 'smoothed_batch_train_loss', 'batch_train_loss', 'run_id']].head(30)
