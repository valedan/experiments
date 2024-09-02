from itertools import product
import random
from typing import Literal
import numpy as np
import math
from pydantic import BaseModel, Field, field_validator

default_params = {"mnist": {"model.input_dim": 28 * 28, "model.output_dim": 10}}


class ExperimentParams(BaseModel):
    type: Literal["mnist"]


class MLPParams(BaseModel):
    width: int
    depth: int
    input_dim: int
    output_dim: int


class TrainerParams(BaseModel):
    optimizer: Literal["sgd", "adam"]
    learning_rate: float | None = None
    val_interval: int
    epochs: int

    @field_validator('epochs', mode='before')
    @classmethod
    def round_epochs(cls, v):
        if isinstance(v, float):
            return round(v)
        return v


class LoaderParams(BaseModel):
    train_frac: float = Field(ge=0, le=1)
    val_frac: float = Field(ge=0, le=1)
    train_batch: int
    val_batch: int


class RunConfig(BaseModel):
    run_id: int | None = None
    exp: ExperimentParams
    model: MLPParams
    trainer: TrainerParams
    loader: LoaderParams


def calculate_width(config):
    a = config['model.depth']
    b = config['model.input_dim'] + config['model.depth'] + config['model.output_dim'] + 1
    c = config['model.output_dim'] - config['exp.total_params']
    roots = np.roots([a, b, c])
    width = max(int(round(root)) for root in roots)
    return width


class ConfigLoader:
    def __init__(self, params: dict):
        self.configs: list[RunConfig] = self.create_configs(params)

    def __iter__(self):
        return iter(self.configs)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, index):
        # TODO make slices return a ConfigLoader, not a list. Need constructor to accept a list instead of yaml
        return self.configs[index]

    def shuffled(self):
        return iter(random.sample(self.configs, len(self.configs)))

    def flatten_params(self, params):
        flat_params = {}
        for group_name, group in params.items():
            for param_name, param in group.items():
                flat_params[f"{group_name}.{param_name}"] = param
        return flat_params

    def unflatten_params(self, flat_params):
        grouped_params = {}
        for name, param in flat_params.items():
            group, param_name = name.split(".")
            grouped_params.setdefault(group, {})[param_name] = param

        return grouped_params

    def create_configs(self, params):
        flat_params = self.flatten_params(params)
        variable_params = {}
        static_params = {}
        derived_params = {}
        for name, param in flat_params.items():
            if isinstance(param, dict) and "derive_from" in param.keys():
                derived_params[name] = param
            elif isinstance(param, list):
                variable_params[name] = param
            else:
                static_params[name] = param

        config_values = list(product(*variable_params.values()))
        configs = []

        for values in config_values:
            config = dict(zip(variable_params.keys(), values))
            config |= static_params
            config = self.add_default_params(config)
            config = self.add_derived_params(config, derived_params)
            if config is None:
                print(f"Invalid config {config}")
            else:
                config = self.unflatten_params(config)
                configs.append(RunConfig(**config))

        return configs

    def add_default_params(self, config):
        config |= default_params.get(config["exp.type"], {})

        return config

    def add_derived_params(self, config, derived_params):
        for param_name, derive_rule in derived_params.items():
            if derive_rule["derive_from"] == "auto":
                if param_name == "model.width":
                    value = calculate_width(config)
            else:
                origin_param = config[derive_rule["derive_from"]]
                if derive_rule["operation"] == "multiply":
                    value = origin_param * derive_rule["value"]
                if derive_rule["operation"] == "divide":
                    value = origin_param / derive_rule["value"]
                if derive_rule["operation"] == "reverse divide":
                    value = derive_rule["value"] / origin_param
                if derive_rule["operation"] == "root":
                    value = math.pow(origin_param, 1 / derive_rule["value"])
                if "offset" in derive_rule.keys():
                    value += derive_rule["offset"]

                if "min" in derive_rule.keys():
                    if value < derive_rule["min"]:
                        return None
            config[param_name] = value

        return config
