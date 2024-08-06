import random


class Run:
    """Manages a single run. It should take a run function and hyperparams, create an id, log files, and logger, and do the run. It can be called async. It also provides an api to fetch the run data."""

    def __init__(self, run_func, **params):
        self.run_func = run_func
        self.params = params
        self.id = self.create_id()

    @staticmethod
    def create_id():
        length = 8
        hex_chars = "0123456789ABCDEF"
        return "".join(random.choice(hex_chars) for _ in range(length))

    async def start(self):
        return self.run_func(**self.params)


class Sweep:
    """Manages a sweep. Takes a run_func, and a list of values for each param. it will create a run for each combination of params. takes an arg indicating how many runs can go in parallel, as well as a results dir."""

    def __init__():
