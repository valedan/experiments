import torch
from torch import nn
from infra import dnn


def test_layer_norm():
    d_model = 512
    seq_len = 10
    batch_size = 16
    x = torch.rand(batch_size, seq_len, d_model)
    layer_norm = dnn.LayerNorm(d_model)
    y = layer_norm(x)
    mean = y.mean(-1)
    std = y.std(-1)
    assert mean.allclose(torch.zeros_like(mean), atol=1e-6)
    assert std.allclose(torch.ones_like(std), atol=1e-6)

def test_multihead_attention():
    d_model = 512
    seq_len = 10
    n_heads = 4
    batch_size = 16
    x = torch.rand(batch_size, seq_len, d_model)
    attention = dnn.MultiheadAttention(d_model, n_heads)
    y = attention(x)
    assert y.shape == (batch_size, seq_len, d_model)
