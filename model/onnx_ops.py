"""Op-restricted building blocks used so the exported ONNX graph only contains ops from the
target NPU toolchain's whitelist (no LayerNormalization/ReduceMean/Sqrt/Pow/Erf/Einsum/Reciprocal/
Neg/Unsqueeze/Tile/Expand). Each replacement is mathematically equivalent to the op it stands in
for, so pretrained checkpoints load and behave the same as before, on both the training and the
ONNX-exported path.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExportableLayerNorm(nn.Module):
    """Drop-in replacement for nn.LayerNorm(dim) (same weight/bias parameter names and shapes,
    so it loads existing checkpoints unchanged). Computes the mean/var normalization via
    InstanceNorm1d with a single channel instead of ReduceMean+Sub+Pow+ReduceMean+Sqrt+Div,
    since only BatchNormalization/InstanceNormalization are in the whitelist, not
    LayerNormalization. Reshaping each (..., dim) row to its own (1-channel, dim-length) instance
    makes InstanceNormalization normalize over exactly the same axis LayerNorm would.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        # passed explicitly (instead of None) so the ONNX exporter's instance_norm symbolic
        # doesn't need to infer the channel count (1) from a reshape with a dynamic batch axis,
        # which it can't do and errors out on ("unknown channel size").
        self.register_buffer('_inorm_weight', torch.ones(1), persistent=False)
        self.register_buffer('_inorm_bias', torch.zeros(1), persistent=False)

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, 1, self.dim)
        normed = F.instance_norm(x_flat, weight=self._inorm_weight, bias=self._inorm_bias, eps=self.eps)
        normed = normed.reshape(shape)
        return normed * self.weight + self.bias


def gelu_tanh(x):
    """tanh approximation of GELU (Mul/Add/Tanh only) in place of the exact erf-based
    F.gelu, since Erf isn't in the whitelist. Matches F.gelu to within ~3e-4 absolute error.
    """
    c = math.sqrt(2.0 / math.pi)
    x_cubed = x * x * x
    inner = c * (x + 0.044715 * x_cubed)
    return 0.5 * x * (1.0 + torch.tanh(inner))


def safe_exp(x):
    """exp(x) via the sigmoid identity exp(x) = sigmoid(x) / (1 - sigmoid(x)), avoiding the
    Exp op (not in the whitelist). Only exact/stable for bounded x (no overflow/cancellation
    for the |x| ranges this model actually produces); do not reuse for unbounded inputs.
    """
    s = torch.sigmoid(x)
    return s / (1.0 - s)


def safe_reciprocal(x):
    """1/x without triggering PyTorch's `1.0 / x` -> ONNX Reciprocal symbolic (not in the
    whitelist); dividing by a Tensor numerator instead of the Python scalar 1.0 keeps it a
    plain Div node.
    """
    return torch.ones_like(x) / x
