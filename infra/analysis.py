from pathlib import Path
import pandas as pd
import shutil

base_exp_dir = Path("/home/dan/Dropbox/projects/exploration/experiments")
base_plots_dir = Path("/home/dan/Dropbox/projects/exploration/analysis")

def initialize_plots_dir(exp_name, refresh=False):
    plots_dir = base_plots_dir / exp_name
    if refresh and plots_dir.exists():
        shutil.rmtree(str(plots_dir))

    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def read_run(filename):
    return pd.read_json(filename, lines=True)


def get_sweep_data(exp_name, cols_to_smooth=["batch_train_loss", ""]):
    params_df = pd.read_json(base_exp_dir / exp_name / "runs.jsonl", lines=True)
    runs_dir = base_exp_dir / exp_name / "runs"
    run_files = list(runs_dir.glob("*.jsonl"))
    df = pd.DataFrame()
    for f in run_files:
        new_data = read_run(f)
        df = pd.concat([df, new_data])

    df = pd.merge(df, params_df, left_on="id", right_on="run_id")
    df = df.drop(columns=["created_at"])
    return df

def get_sweep_data_for_experiments(exp_names):
    dfs = []
    for exp in exp_names:
        df = get_sweep_data(exp)
        df['exp_name'] = exp
        dfs.append(df)
    return pd.concat(dfs)
