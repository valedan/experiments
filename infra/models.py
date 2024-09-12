from torch import nn
import numpy as np


class MLP(nn.Module):
    # TODO: add batch norm? claude suggested for faster training
    def __init__(self, input_dim, output_dim, depth, width):
        super().__init__()
        self.depth = depth
        self.width = width
        self.input = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.output = nn.Linear(width, output_dim)

    def forward(self, X):
        # todont: use sequential. maybe inherit from it and don't use a forward at all?
        X = self.input(X)
        for layer in self.layers:
            X = nn.functional.relu(layer(X))

        X = self.output(X)
        return X
