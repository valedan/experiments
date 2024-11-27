from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from infra.sweep import Sweep
from infra.analysis import SweepData

palette = mpl.colormaps['tab20']
def test_transformer_training():
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    exp_dir = Path(__file__).parent / "test_sweeps" / f"transformer_{timestamp}"
    sweep_file = Path(__file__).parent / 'transformer_config.yml'
    sweep = Sweep(sweep_file, sweep_dir=exp_dir)
    sweep.start()

    sweep_data = SweepData(exp_dir)

    train = sweep_data.data[sweep_data.data['logger_id'] == 'train']
    val = sweep_data.data[sweep_data.data['logger_id'] == 'val']
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Transformer training")
    ax.plot(train["step"], train["loss_mean"], c=palette(0), ls='-', alpha=0.25)
    ax.plot(train["step"], train["smoothed_loss_mean"],  c=palette(0), ls='--')
    ax.plot(train["step"], train["smoothed_accuracy_mean"], c=palette(6), ls='-', alpha=0.5)
    ax.plot(val["step"], val["smoothed_loss_mean"],  c=palette(2), ls='--')
    ax.plot(val["step"], val["accuracy_mean"], c=palette(4), ls='-', )
    plt.savefig(sweep_data.results_dir / "test.png")
    plt.show()
