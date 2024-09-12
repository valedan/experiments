import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from infra.analysis import (
    get_sweep_data,
    initialize_plots_dir,
    metrics,
    smoothed_metrics,
)

all_metrics = {**metrics, **smoothed_metrics}


def create_summary(df):
    summary = []
    for run_id, run_df in df.groupby("id"):
        run_summary = pd.Series({"run_id": run_id})
        for metric_name, metric_info in all_metrics.items():
            ascending = metric_info["better"] == "lower"
            best = run_df.sort_values(metric_name, ascending=ascending).iloc[0]
            run_summary[f"best_{metric_name}_step"] = best["step"]
            run_summary[f"best_{metric_name}"] = best[metric_name]

        summary.append(run_summary)
    return pd.DataFrame(summary)


palette = mpl.colormaps["tab20"]
exp_name = "tests/001"
plots_dir = initialize_plots_dir(exp_name)
df = get_sweep_data(exp_name, cols_to_smooth=metrics.keys())
summary = create_summary(df)
summary
# %%

param_counts = df.groupby("total_params")

metrics_to_plot = [
    "batch_train_loss",
    "smoothed_batch_train_loss",
    "epoch_val_loss",
    "smoothed_epoch_val_loss",
    "batch_train_accuracy",
    "smoothed_batch_train_accuracy",
    "epoch_val_accuracy",
    "smoothed_epoch_val_accuracy",
]
for param_count, param_df in param_counts:
    fig, axes = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(15, 15))
    fig.suptitle(f"{param_count} params")
    axes = axes.flatten()
    axes[0].set_ylim(0, 3)
    depths = param_df.groupby("depth")
    for idx, (depth, ddf) in enumerate(depths):
        run_summary = summary.loc[summary["run_id"] == ddf["run_id"].iloc[0]]
        ax = axes[idx]
        width = ddf.iloc[0]["width"]
        ax.set_title(f"Depth {depth}, Width {width}")
        for metric in metrics_to_plot:
            ax.plot(
                ddf["step"],
                ddf[metric],
                color=all_metrics[metric]["color"],
                ls=all_metrics[metric]["linestyle"],
                alpha=all_metrics[metric]["alpha"],
            )
            ax.plot(
                run_summary[f"best_{metric}_step"].iloc[0],
                run_summary[f"best_{metric}"].iloc[0],
                "o",
                color=all_metrics[metric]["color"],
                alpha=all_metrics[metric]["alpha"],
            )
    plt.show()


# %%

palette = mpl.colormaps["viridis"]
metrics_to_plot = [
    "smoothed_batch_train_loss",
    "smoothed_batch_train_accuracy"
]
for param_count, param_df in param_counts:
    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title(f"{param_count} params - {metric}")
        depths = param_df.groupby("depth")
        colors = palette(np.linspace(0, 1, len(depths)))
        if "loss" in metric:
            ax.set_ylim(0, 2.5)
        else:
            ax.set_ylim(0, 1.0)
        for idx, (depth, ddf) in enumerate(depths):
            run_summary = summary.loc[summary["run_id"] == ddf["run_id"].iloc[0]]
            width = ddf.iloc[0]["width"]
            color = colors[idx]
            line, = ax.plot(
                ddf["step"],
                ddf[metric],
                color=color,
                ls="-",
                alpha=1.0,
                label=f"d{depth}_w{width}",
            )
            #last 30 points are removed in smoothed metrics
            ann_y = line.get_ydata()[-31]
            ax.annotate(f"d{depth}_w{width}", xy=(ax.get_xticks()[-2]+150, ann_y), color=color)
            ax.plot(
                run_summary[f"best_{metric}_step"].iloc[0],
                run_summary[f"best_{metric}"].iloc[0],
                "o",
                color=color,
                alpha=0.5,
            )
        ax.legend()
        plt.show()
