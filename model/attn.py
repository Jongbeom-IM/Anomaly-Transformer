import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os

from .onnx_ops import ExportableLayerNorm, matmul_transb, safe_exp, safe_reciprocal


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                distances[i][j] = abs(i - j)
        self.register_buffer('distances', distances, persistent=False)
        # Set to True (by pth_to_onnx.py, only around the export call) to switch to
        # _forward_export below. Kept off by default since it unrolls per head and assumes
        # batch size 1, which would be both wrong and slow for ordinary (batched) training.
        self.export_mode = False

    def forward(self, queries, keys, values, sigma, attn_mask):
        if self.export_mode:
            return self._forward_export(queries, keys, values, sigma, attn_mask)
        return self._forward_train(queries, keys, values, sigma, attn_mask)

    def _forward_train(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # torch.einsum("blhe,bshe->bhls", ...) exported literally as an ONNX Einsum node,
        # which the target op whitelist doesn't include; permute + matmul is equivalent and
        # only uses Transpose/MatMul.
        q = queries.permute(0, 2, 1, 3)  # B H L E
        k = keys.permute(0, 2, 1, 3)  # B H S E
        scores = torch.matmul(q, k.transpose(-1, -2))  # B H L S
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        # torch.pow(3, sigma) - 1 == exp(sigma * ln(3)) - 1; safe_exp avoids the Pow/Exp ops
        # (sigma is in (0, ~1) here, so the argument stays small and the sigmoid-based exp
        # trick is numerically safe, see onnx_ops.safe_exp).
        sigma = safe_exp(sigma * math.log(3)) - 1
        # sigma.unsqueeze(-1).repeat(1, 1, 1, window_size) / distances.unsqueeze(0).unsqueeze(0)
        # .repeat(...) both used Unsqueeze+Tile/Expand (not in the whitelist); reshape to add the
        # broadcast dim instead, and let the elementwise ops below broadcast naturally (Mul/Div
        # support broadcasting directly, no explicit tiling needed for the math itself).
        sigma = sigma.reshape(sigma.shape[0], sigma.shape[1], sigma.shape[2], 1)  # B H L 1
        prior_dist = self.distances.reshape(1, 1, window_size, window_size)  # 1 1 L L
        dist_sq = prior_dist * prior_dist
        sigma_sq = sigma * sigma
        exponent = (0.0 - dist_sq) / (2.0 * sigma_sq)  # broadcasts to B H L L
        prior = safe_reciprocal(math.sqrt(2 * math.pi) * sigma) * safe_exp(exponent)  # B H L L
        # broadcast sigma up to the same B H L L shape the caller expects, via a genuine Mul
        # (not Tile/Expand): multiplying by a same-shape ones tensor forces the broadcast.
        sigma = sigma * torch.ones_like(prior_dist)

        series = self.dropout(torch.softmax(attn, dim=-1))
        # torch.einsum("bhls,bshd->blhd", ...) rewritten the same way as the scores einsum above.
        v = values.permute(0, 2, 1, 3)  # B H S D
        V = torch.matmul(series, v).permute(0, 2, 1, 3)  # B L H D

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)

    def _forward_export(self, queries, keys, values, sigma, attn_mask):
        # Splitting Q/K/V into per-head slices via reshape-to-4D + permute(0,2,1,3) (as
        # _forward_train does) emits a Transpose whose permutation Klepsydra's ONNX importer
        # rejects ("Transpose given unsupported permutation") -- it only recognizes Transpose
        # as a channel-first/channel-last no-op around 4D (NCHW<->NHWC) ops, not this
        # head-to-front reordering. Since export is always batch size 1 (--static_batch), each
        # head can instead be sliced out (Slice, not integer indexing -- Gather isn't in
        # Klepsydra's supported op list either) and processed as a plain 2D matrix, with
        # matmul_transb giving Q@K^T as a single Gemm(transB=1) instead of Transpose+MatMul.
        # The four outputs are then reassembled per head with Reshape+Concat, matching the
        # B,H,L,S / B,L,H,D layouts _forward_train produces, without a single Transpose node.
        assert not self.mask_flag, 'masked attention export is not implemented (unused: the ' \
            'only call site in AnomalyTransformer.py passes mask_flag=False)'
        _, L, H, E = queries.shape
        _, S, _, Dv = values.shape
        scale = self.scale or 1. / sqrt(E)
        window_size = self.distances.shape[-1]

        V_parts, series_parts, prior_parts, sigma_parts = [], [], [], []
        for h in range(H):
            q_h = queries[:, :, h:h + 1, :].reshape(L, E)
            k_h = keys[:, :, h:h + 1, :].reshape(S, E)
            v_h = values[:, :, h:h + 1, :].reshape(S, Dv)
            sigma_h = sigma[:, :, h:h + 1].reshape(L, 1)

            scores_h = matmul_transb(q_h, k_h) * scale  # L S
            series_h = self.dropout(torch.softmax(scores_h, dim=-1))  # L S

            sigma_h = torch.sigmoid(sigma_h * 5) + 1e-5
            sigma_h = safe_exp(sigma_h * math.log(3)) - 1  # L 1
            dist_sq = self.distances * self.distances  # L L
            sigma_sq = sigma_h * sigma_h  # L 1, broadcasts against L L below
            exponent = (0.0 - dist_sq) / (2.0 * sigma_sq)
            prior_h = safe_reciprocal(math.sqrt(2 * math.pi) * sigma_h) * safe_exp(exponent)  # L L
            sigma_h = sigma_h * torch.ones_like(self.distances)  # broadcast to L L

            v_out_h = torch.matmul(series_h, v_h)  # L Dv

            V_parts.append(v_out_h.reshape(1, L, 1, Dv))
            series_parts.append(series_h.reshape(1, 1, L, S))
            prior_parts.append(prior_h.reshape(1, 1, L, window_size))
            sigma_parts.append(sigma_h.reshape(1, 1, L, window_size))

        V = torch.cat(V_parts, dim=2)  # 1 L H Dv
        series = torch.cat(series_parts, dim=1)  # 1 H L S
        prior = torch.cat(prior_parts, dim=1)  # 1 H L L
        sigma_out = torch.cat(sigma_parts, dim=1)  # 1 H L L

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma_out)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = ExportableLayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma
