from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, depth, width):
        super().__init__()
        self.depth = depth
        self.width = width
        self.input = nn.Linear(input_dim, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.output = nn.Linear(width, output_dim)

    def forward(self, X):
        X = self.input(X)
        for layer in self.layers:
            X = layer(X)

        X = self.output(X)
        return X
