from collections.abc import Iterable
from scipy.signal import savgol_filter
import json
import numpy as np
import matplotlib as mpl
from pathlib import Path
import pandas as pd
import shutil

from infra.utils import flatten

palette = mpl.colormaps["tab20"]
metrics = {
    "loss_mean": {
        "better": "lower",
        "color": palette(0),
        "linestyle": "-",
        "alpha": 0.25,
    },
    "accuracy_mean": {
        "better": "higher",
        "color": palette(2),
        "linestyle": "-",
        "alpha": 0.25,
    },
}

# TODO: delete?
smoothed_metrics = {
    f"smoothed_{k}": {**v, "alpha": 1.0, "linestyle": "--"} for k, v in metrics.items()
}


class SweepData:
    def __init__(
        self,
        sweep_dirs: Path | Iterable[Path],
        results_dir: Path | None = None,
        cols_to_smooth: Iterable[str] | None = tuple(metrics),
    ):
        self.cols_to_smooth = tuple(cols_to_smooth or ())

        if isinstance(sweep_dirs, Iterable):
            self.sweep_dirs = tuple(Path(dir) for dir in sweep_dirs)
        else:
            self.sweep_dirs = (Path(sweep_dirs),)
        assert all(dir.is_dir() for dir in self.sweep_dirs)

        if results_dir is None:
            self.results_dir = self.sweep_dirs[0] / "results"
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.data = self.ingest_sweep_data()

    def read_run(self, filename):
        df = pd.read_json(filename, lines=True, dtype={'run_id': str})

        for col in self.cols_to_smooth:
            df[f"smoothed_{col}"] = savgol_filter(df[col], 51, 3)
            # remove early and late values because savgol goes to crazy values at the edges
            df.loc[~df["step"].between(30, df["step"].max() - 30), f"smoothed_{col}"] = np.nan

        return df

    def ingest_sweep_data(self):
        dfs = []
        for sweep_dir in self.sweep_dirs:
            with open(sweep_dir / "runs.jsonl") as f:
                run_configs_df = pd.DataFrame(flatten(json.loads(line)) for line in f.readlines())

            run_files = list((sweep_dir / "runs").glob("*/training_logs.jsonl"))

            df = pd.concat(self.read_run(f) for f in run_files)
            df = pd.merge(df, run_configs_df, left_on="run_id", right_on="id")
            df["sweep_name"] = sweep_dir.name
            dfs.append(df)
        return pd.concat(dfs)

    def clear_results_dir(self):
        shutil.rmtree(str(self.results_dir))
        self.results_dir.mkdir(parents=True, exist_ok=True)
