from scipy.signal import savgol_filter
import json
import numpy as np
import matplotlib as mpl
from pathlib import Path
import pandas as pd
import shutil

base_exp_dir = Path("/home/dan/Dropbox/projects/exploration/experiments")
base_plots_dir = Path("/home/dan/Dropbox/projects/exploration/analysis")

palette = mpl.colormaps["tab20"]
metrics = {
    "batch_train_loss": {
        "better": "lower",
        "color": palette(0),
        "linestyle": "-",
        "alpha": 0.25,
    },
    "epoch_val_loss": {
        "better": "lower",
        "color": palette(2),
        "linestyle": "-",
        "alpha": 0.25,
    },
    "batch_train_accuracy": {
        "better": "higher",
        "color": palette(4),
        "linestyle": "-",
        "alpha": 0.25,
    },
    "epoch_val_accuracy": {
        "better": "higher",
        "color": palette(6),
        "linestyle": "-",
        "alpha": 0.25,
    },
}

smoothed_metrics = {
    f"smoothed_{k}": {**v, "alpha": 1.0, "linestyle": "--"} for k, v in metrics.items()
}


def initialize_plots_dir(exp_name, refresh=False):
    plots_dir = base_plots_dir / exp_name
    if refresh and plots_dir.exists():
        shutil.rmtree(str(plots_dir))

    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def read_run(filename):
    return pd.read_json(filename, lines=True)


def get_sweep_data(exp_name, cols_to_smooth=None):
    params_list = []
    with open(base_exp_dir / exp_name / "runs.jsonl") as f:
        # TODO: tidy up
        for line in f.readlines():
            params = json.loads(line)
            flat_params = {}
            for group_name, group in params.items():
                if isinstance(group, dict):
                    for param_name, param in group.items():
                        flat_params[f"{group_name}.{param_name}"] = param
                else:
                    flat_params[group_name] = group
            params_list.append(flat_params)
    params_df = pd.DataFrame(params_list)

    runs_dir = base_exp_dir / exp_name / "runs"
    run_files = list(runs_dir.glob("*.jsonl"))
    df = pd.DataFrame()
    for f in run_files:
        new_data = read_run(f)
        df = pd.concat([df, new_data])

    df = pd.merge(df, params_df, left_on="id", right_on="id")
    df = df.drop(columns=["created_at"])
    if cols_to_smooth:
        for col in cols_to_smooth:
            assert col in df.columns
            df[f"smoothed_{col}"] = savgol_filter(df[col], 51, 3)
            for _, run_df in df.groupby("id"):
                # remove early and late values because savgol goes to crazy values at the edges
                start_indices = run_df.index[run_df["step"] < 30]
                end_indices = run_df.index[
                    run_df["step"] >= (run_df["step"].max() - 29)
                ]

                df.loc[start_indices, f"smoothed_{col}"] = np.nan
                df.loc[end_indices, f"smoothed_{col}"] = np.nan
    return df


def get_sweep_data_for_experiments(exp_names, cols_to_smooth=None):
    dfs = []
    for exp in exp_names:
        df = get_sweep_data(exp, cols_to_smooth)
        df["exp_name"] = exp
        dfs.append(df)
    return pd.concat(dfs)
