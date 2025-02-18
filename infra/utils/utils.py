from collections import abc

import torch

def initialize_device(device = None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if device == 'cuda':
        torch.cuda.empty_cache()

    return device

def flatten(input: abc.Mapping, sep=".", base=[]):
    flat = {}
    for name, param in input.items():
        if isinstance(param, abc.Mapping):
            flat |= flatten(param, sep=sep, base=[*base, name])
        else:
            flat_name = sep.join([*base, name])
            flat[flat_name] = param
    return flat


def unflatten(input: abc.Mapping, sep="."):
    unflat = {}
    for name, param in input.items():
        parts = name.split(sep)
        working_dict = unflat
        for i, part in enumerate(parts):
            islast = i + 1 == len(parts)
            if islast:
                working_dict[part] = param
            else:
                working_dict = working_dict.setdefault(part, {})

    return unflat


def create_profiler(log_dir):
    """Create a pytorch profiler with reasonable defaults

    Args:
     - log_dir: Path to the dir that any log files should be written to

    Returns: An instantiated profiler ready to use as a context manager
    """

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{log_dir}/pytorch_profiler"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
    )
