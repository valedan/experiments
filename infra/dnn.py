import torch as t
import math
from torch import nn

functional = nn.functional
Module = nn.Module
ModuleList = nn.ModuleList
Parameter = nn.Parameter
Sequential = nn.Sequential


# def softmax():
#     pass


# functional.softmax = softmax


def initialize_params_uniform(size, bound, device=None, dtype=None):
    tensor = t.rand(size, device=device, dtype=dtype) * (2 * bound) - bound
    return tensor


class ReLU(Module):
    def forward(self, X):
        X[X < 0] = 0
        return X


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        bound = 1 / math.sqrt(in_features)
        self.weight = Parameter(
            initialize_params_uniform((out_features, in_features), bound, device, dtype)
        )
        if bias:
            self.bias = Parameter(initialize_params_uniform(out_features, bound, device, dtype))

    def forward(self, X):
        if self.bias is None:
            return X @ self.weight.T
        else:
            return X @ self.weight.T + self.bias


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embeddings = Parameter(t.Tensor(num_embeddings, embedding_dim))
        # TODO
        nn.init.normal_(self.embeddings, std=0.2)

    def forward(self, X):
        return self.embeddings[X]


class ResidualBlock(nn.Module):
    def __init__(self, block: Sequential):
        super().__init__()
        self.block = block

    def forward(self, X):
        return X + self.block(X)


class LayerNorm(Module):
    def __init__(self, size):
        super().__init__()
        self.weight = Parameter(t.ones(size))
        self.bias = Parameter(t.zeros(size))

    def forward(self, X):
        std = X.std(dim=-1)
        mean = X.mean(dim=-1)
        norm = (X - mean.unsqueeze(-1)) / std.unsqueeze(-1)
        return norm * self.weight + self.bias


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, kdim=None, vdim=None):
        super().__init__()
        if kdim is None:
            kdim = round(embed_dim / num_heads)

        if vdim is None:
            vdim = round(embed_dim / num_heads)

        self.scale = t.tensor((math.sqrt(kdim)))
        # TODO: change these to linear?
        # TODO: use xavier init (implement that)
        self.W_Q = Parameter(t.randn(num_heads, embed_dim, kdim))
        self.W_K = Parameter(t.randn(num_heads, embed_dim, kdim))
        self.W_V = Parameter(t.randn(num_heads, embed_dim, vdim))

        # TODO: do not use biases by default
        self.B_Q = Parameter(t.randn(num_heads, kdim))
        self.B_K = Parameter(t.randn(num_heads, kdim))
        self.B_V = Parameter(t.randn(num_heads, vdim))
        self.output = Linear(num_heads * vdim, embed_dim)

    def forward(self, X):
        if X.ndim == 2:
            # add missing batch dim
            X = X.unsqueeze(0)

        Q = (X.unsqueeze(1) @ self.W_Q) + self.B_Q.unsqueeze(1)
        K = (X.unsqueeze(1) @ self.W_K) + self.B_K.unsqueeze(1)
        V = (X.unsqueeze(1) @ self.W_V) + self.B_V.unsqueeze(1)
        qk = Q @ K.mT
        # TODO add attention mask for padding tokens to avoid wasted compute - i think the tokenizer provides this to the model along with the batch
        mask = t.triu(qk, diagonal=1)
        mask[mask != 0] = float("-inf")
        # TODO
        attention = nn.functional.softmax((qk + mask) / self.scale, dim=-1) @ V
        concat = attention.permute(0, 2, 1, 3).flatten(start_dim=-2)
        X = self.output(concat)

        return X


class TransformerDecoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.attention_block = ResidualBlock(
            Sequential(LayerNorm(d_model), MultiheadAttention(d_model, nhead))
        )
        self.mlp_block = ResidualBlock(
            Sequential(
                LayerNorm(d_model),
                Linear(d_model, dim_feedforward),
                # TODO: use gelu or swiglu
                ReLU(),
                Linear(dim_feedforward, d_model),
            )
        )

    def forward(self, X):
        X = self.attention_block(X)
        X = self.mlp_block(X)
        return X
