from datetime import datetime
import json
from pathlib import Path


class DataLogger:
    """A logger class to use for experiments. Each instance can have an associated data file to write new data to, and a log file to write logs to. Uses jsonl for data and txt for logs. Can have an identifier to associate to a run/sweep."""

    def __init__(
        self,
        id: str | None = None,
        data_file: str | None = None,
        log_file: str | None = None,
        id_key: str = "id",
        print_logs: bool = True,
    ):
        self.id = id
        self.id_key = id_key
        self.data_file = Path(data_file) if data_file else None
        self.log_file = Path(log_file) if log_file else None
        self.print_logs = print_logs
        self.init_files()

    def init_files(self):
        if self.data_file:
            if self.data_file.suffix != ".jsonl":
                raise ValueError("Data file must be jsonl")
            if not self.data_file.exists():
                self.data_file.parent.mkdir(parents=True, exist_ok=True)
                self.data_file.touch()

        if self.log_file:
            if self.log_file.suffix != ".txt":
                raise ValueError("Log file must be txt")
            if not self.log_file.exists():
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                self.log_file.touch()

    @staticmethod
    def _now():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add(self, **kwargs):
        if not self.data_file:
            raise ValueError("No data file registered")

        data = {**kwargs, "id": self.id, "created_at": self._now()}
        with self.data_file.open("a") as f:
            f.write(json.dumps(data) + "\n")

    def log(self, log: str):
        if not self.log_file:
            raise ValueError("No log file registered")

        log = f"{self._now()} - {log}"
        if self.print_logs:
            print(log)
        with self.log_file.open("a") as f:
            f.write(log + "\n")
