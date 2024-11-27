from scipy.signal import savgol_filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from infra.analysis import get_sweep_data, initialize_plots_dir

exp_name = "tests/001"
plots_dir = initialize_plots_dir(exp_name)
df = get_sweep_data(exp_name)
depths = df.groupby('model.depth')

palette = mpl.colormaps['tab20']

for depth, ddf in depths:
    fig, axes = plt.subplots(5, 3, sharex=True, sharey=True, figsize=(14,14))
    fig.suptitle(f"Depth {depth}")
    axes = axes.flatten()
    axes[0].set_ylim(0,3)
    widths = ddf.groupby('model.width')
    for idx, (width, wdf) in enumerate(widths):
        ax = axes[idx]
        ax.set_title(f"Width {width}")
        ax.plot(wdf['step'], wdf['batch_train_loss'])
        ax.plot(wdf['step'], wdf['epoch_val_loss'], ls='--')


    plt.show()
