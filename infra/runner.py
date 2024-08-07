import random
import traceback
from infra.logger import DataLogger
import asyncio
from itertools import product
from typing import Any, Callable


class Run:
    """Manages a single run. It should take a run function and hyperparams, create an id, log files, and logger, and do the run. It can be called async. It also provides an api to fetch the run data."""

    def __init__(
        self,
        run_func: Callable,
        params: dict[str, Any],
        data_dir: str,
        id: str | None = None,
        name: str | None = None,
    ):
        self.run_func = run_func
        self.params = params
        self.data_dir = data_dir
        if id:
            self.id = id
        else:
            self.id = self.create_id()
        self.name = name
        self.state = "new"
        data_file = f"{data_dir}/{id}{'_' + name if name else None}.jsonl"
        log_file = f"{data_dir}/{id}{'_' + name if name else None}.txt"
        self.logger = DataLogger(self.id, data_file, log_file)

    @staticmethod
    def create_id():
        length = 8
        hex_chars = "0123456789ABCDEF"
        return "".join(random.choice(hex_chars) for _ in range(length))

    async def start(self, semaphore):
        self.logger.log(f"Starting run {self.id}")
        async with semaphore:
            try:
                await self.run_func(self.logger, self.params)
            except Exception as e:
                self.logger.log(
                    f"{e.__class__.__name__}: {str(e)}\n{traceback.format_exc()}"
                )
                self.state = "error"
            else:
                self.state = "finished"
                self.logger.log(f"Finished run {self.id}")


class Sweep:
    """Manages a sweep. Takes a run_func, and a list of values for each param. it will create a run for each combination of params. takes an arg indicating how many runs can go in parallel, as well as a results dir."""

    def __init__(
        self,
        run_func: Callable,
        params: dict[str, list[Any]],
        data_dir: str,
        max_parallel: int = 10,
        run_name_formatter=None,
    ):
        self.run_func = run_func
        self.params = params
        self.data_dir = data_dir
        self.max_parallel = max_parallel
        self.run_name_formatter = run_name_formatter
        self.runs = []
        self.create_runs()

    def create_runs(self):
        id = 1
        keys = self.params.keys()
        values = self.params.values()
        run_configs = product(*values)  # make anki
        for config_values in run_configs:
            config = dict(zip(keys, config_values))
            name = self.run_name_formatter(config) if self.run_name_formatter else None
            run = Run(self.run_func, config, self.data_dir, str(id), name=name)
            self.runs.append(run)
            id += 1

    async def start(self):
        semaphore = asyncio.Semaphore(self.max_parallel)
        tasks = [run.start(semaphore) for run in self.runs if run.state in ["new"]]
        print(len(tasks))
        print(self.max_parallel)
        await asyncio.gather(*tasks)
