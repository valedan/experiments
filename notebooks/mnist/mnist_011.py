from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from infra.analysis import get_sweep_data, initialize_plots_dir

exp_name = "mnist/011"
plots_dir = initialize_plots_dir(exp_name)
df = get_sweep_data(exp_name)

param_counts = df.groupby('total_params')

palette = mpl.colormaps['tab20']

for param_count, param_df in param_counts:
    fig, axes = plt.subplots(5, 4, sharex=True, sharey=True, figsize=(14,14))
    fig.suptitle(f"{param_count} params")
    axes = axes.flatten()
    axes[0].set_ylim(0,3)
    axes[0].set_xlim(0,1000)
    depths = param_df.groupby('depth')
    for idx, (depth, ddf) in enumerate(depths):
        ax = axes[idx]
        width = ddf.iloc[0]['width']
        assert len(ddf['width'].unique()) == 1
        ax.set_title(f"Depth {depth}, Width {width}")
        ax.plot(ddf['step'], ddf['batch_train_loss'])
        ax.plot(ddf['step'], ddf['epoch_val_loss'], ls='--')


    plt.show()
