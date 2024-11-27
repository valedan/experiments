from collections import abc


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
