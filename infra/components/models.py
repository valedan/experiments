from infra.components import dnn
from torch import nn
import torch as t




class MLP(nn.Module):
    def __init__(
        self, width: int, depth: int, input_dim: int, output_dim: int, use_dnn: bool = True
    ):
        super().__init__()
        print(f"Use dnn: {use_dnn}")
        self.nn = dnn if use_dnn else nn
        self.depth = depth
        self.width = width
        self.input = self.nn.Linear(input_dim, width)
        self.layers = self.nn.ModuleList([self.nn.Linear(width, width) for _ in range(depth)])
        self.output = self.nn.Linear(width, output_dim)

    def forward(self, X: t.Tensor):
        X = X.view(X.size(0), -1)
        X = self.input(X)
        for layer in self.layers:
            X = self.nn.functional.relu(layer(X))

        X = self.output(X)
        return X


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding="same"),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 120, 5, padding=0),
            nn.Sigmoid(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10),
        )

    def forward(self, X: t.Tensor):
        X = self.model(X)
        return X


class LeNetModern(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 120, 5, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, X: t.Tensor):
        X = self.model(X)
        return X


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
        )

    def forward(self, X: t.Tensor):
        X = self.model(X)
        return X


class PositionalEncoding(nn.Module):
    def __init__(self, context_size, d_model, use_dnn=True):
        super().__init__()
        self.nn = dnn if use_dnn else nn
        self.encodings = self.nn.Embedding(context_size, d_model)

    def forward(self, X):
        return X + self.encodings(X.shape[1])


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        vocab_size: int,
        context_size: int,
        use_dnn: bool=True,
    ):
        super().__init__()
        print(f"Use dnn: {use_dnn}")
        self.nn = dnn if use_dnn else nn
        self.embeddings = self.nn.Embedding(vocab_size, d_model)
        self.encodings = PositionalEncoding(context_size, d_model)
        self.decoder = self.nn.Sequential(
            *[
                self.nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward)
                for _ in range(num_decoder_layers)
            ]
        )
        self.output = self.nn.Linear(d_model, vocab_size)

    def forward(self, X):
        X = self.embeddings(X)
        X = self.encodings(X)
        X = self.decoder(X)
        X = self.output(X)
        return X
