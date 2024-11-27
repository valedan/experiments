from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from infra.sweep import Sweep
from infra.analysis import SweepData

palette = mpl.colormaps['tab20']
def test_sweep_and_analysis():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    exp_dir = Path(__file__).parent / "test_sweeps" / timestamp
    sweep_file = Path(__file__).parent / 'test_config.yml'
    sweep = Sweep(sweep_file, sweep_dir=exp_dir)
    run_count = sweep.start()

    assert run_count == 2

    sweep_data = SweepData(exp_dir)
    depths = sweep_data.data.groupby('model.depth')

    assert len(depths) == 2

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14,14))
    axes = axes.flatten()
    axes[0].set_ylim(0,3)
    for idx, (depth, ddf) in enumerate(depths):
        train = ddf[ddf['logger_id'] == 'train']
        val = ddf[ddf['logger_id'] == 'val']
        ax = axes[idx]
        width = ddf.iloc[0]['model.width']
        ax.set_title(f"Depth {depth}, Width {width}")
        ax.plot(train["step"], train["loss_mean"], c=palette(0), ls='-', alpha=0.25)
        ax.plot(train["step"], train["smoothed_loss_mean"],  c=palette(0), ls='--')
        # ax.plot(ddf["step"], ddf["epoch_val_loss"], c=palette(2), ls='--', alpha=0.25)
        ax.plot(val["step"], val["smoothed_loss_mean"],  c=palette(2), ls='--')
        ax.plot(val["step"], val["accuracy_mean"], c=palette(4), ls='-', )
        ax.plot(train["step"], train["smoothed_accuracy_mean"], c=palette(6), ls='-', alpha=0.5)
        # ax.plot(ddf['step'], ddf['batch_train_loss'])
        # ax.plot(ddf['step'], ddf['epoch_val_loss'], ls='--')
        plt.savefig(sweep_data.results_dir / "test.png")
