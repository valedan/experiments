from typing import Callable, Iterable
from datetime import datetime
import json
from pathlib import Path


def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


class DataLogger:
    """Handles data logging for a training run. Computing metrics usually requires syncing gpu tensors to cpu, which hurts performance if done too often, so we buffer metric calculations and file writes.

    Log training steps with log(). There are flushed to the data_path based on flush_interval. In general, data sent to log() is not directly logged to the file - instead it is used by the specified metric functions to compute aggregate statistics over the flush interval. For example, the accuracy metric relies on logits so these must be logged, but only the accuracy stats are written to the data file, not the raw logits. Field names in fields_to_log are written directly to the log file based on their value in the most recent log entry prior to flushing.
    """

    def __init__(
        self,
        data_path: Path | str,
        id: str | None = None,
        flush_interval: int = 1000,
        aggregation_interval: int = 1,
        metrics: Iterable[Callable] | None = None,
    ):
        self.id = id
        self.buffer = []
        self.data_path = Path(data_path)
        self.flush_interval = flush_interval
        self.aggregation_interval = aggregation_interval
        self.metrics = list(metrics or [])
        self.fields_to_log = ["step", "time", "epoch", "run_id"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.flush()
        return False

    def log(self, **kwargs):
        self.buffer.append({**kwargs, "time": timestamp()})

        if self.flush_interval and len(self.buffer) >= self.flush_interval:
            self.flush()


    def flush(self):
        if not self.buffer:
            return None

        for idx in range(0, len(self.buffer), self.aggregation_interval):
            batch = self.buffer[idx:idx+self.aggregation_interval]

            summary = {
                "logger_id": self.id,
                "n_steps": len(batch),
            }
            summary |= {
                field: batch[-1][field]
                for field in self.fields_to_log
                if field in batch[-1]
            }

            field_data_map = {}
            keys = set(key for log in batch for key in log)
            for key in keys:
                field_data_map[key] = [log[key] for log in batch if key in log]

            for metric in self.metrics:
                summary |= metric(field_data_map)

            with self.data_path.open("a") as f:
                f.write(json.dumps(summary) + "\n")

        self.buffer = []
