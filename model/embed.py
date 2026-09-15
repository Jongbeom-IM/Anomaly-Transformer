import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # Klepsydra's ONNX importer only recognizes Transpose as a channel-first/channel-last
        # no-op around 4D (NCHW<->NHWC) ops; the 3D perm=[0,2,1] that x.permute(0,2,1) /
        # .transpose(1,2) would emit around a Conv1d is rejected ("Transpose given unsupported
        # permutation"). Reshaping to a dummy 4D NHWC tensor and running the same weights
        # through Conv2d (kernel (3,1)) produces the identical result via the perm=[0,3,1,2] /
        # [0,2,3,1] pattern Klepsydra does support.
        b, l, c = x.shape
        x = x.reshape(b, l, 1, c)
        left = x[:, -1:, :, :]
        right = x[:, :1, :, :]
        x = torch.cat([left, x, right], dim=1)
        x = x.permute(0, 3, 1, 2)
        weight = self.tokenConv.weight.unsqueeze(-1)
        x = F.conv2d(x, weight)
        x = x.permute(0, 2, 3, 1).reshape(b, l, -1)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
