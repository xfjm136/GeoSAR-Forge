"""Decomposition-aware TCN + GRU + Self-Attention quantile forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .common import TemporalConvEncoder, config_to_dict as _config_to_dict, sort_quantiles


class _TemporalSelfAttention(nn.Module):
    """Lightweight temporal self-attention with pre-norm."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        return x + attn_out


@dataclass
class DecompTCNGRUConfig:
    input_dim: int
    static_dim: int
    long_term_indices: tuple[int, ...]
    short_term_indices: tuple[int, ...]
    hidden_dim: int = 96
    gru_layers: int = 2
    dropout_rate: float = 0.15
    horizon: int = 3
    n_attn_heads: int = 4


class DecompTCNGRUQuantileForecaster(nn.Module):
    def __init__(self, config: DecompTCNGRUConfig) -> None:
        super().__init__()
        self.config = config
        self.long_proj = nn.Linear(len(config.long_term_indices), config.hidden_dim)
        self.long_tcn = TemporalConvEncoder(config.hidden_dim, dropout=config.dropout_rate)
        self.long_attn = _TemporalSelfAttention(config.hidden_dim, config.n_attn_heads, config.dropout_rate)
        self.short_proj = nn.Linear(len(config.short_term_indices), config.hidden_dim)
        self.short_gru = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.gru_layers,
            dropout=config.dropout_rate if config.gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.static_mlp = nn.Sequential(
            nn.Linear(config.static_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.horizon * 3),
        )

    def forward(
        self,
        seq: torch.Tensor,
        static: torch.Tensor,
        *,
        neighbor_seq: torch.Tensor | None = None,
        neighbor_static: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        long_x = self.long_proj(seq[:, :, list(self.config.long_term_indices)])
        long_feat = self.long_tcn(long_x.transpose(1, 2)).transpose(1, 2)
        long_feat = self.long_attn(long_feat)[:, -1, :]

        short_x = self.short_proj(seq[:, :, list(self.config.short_term_indices)])
        short_feat, _ = self.short_gru(short_x)
        short_feat = short_feat[:, -1, :]

        static_feat = self.static_mlp(static)
        gate = self.gate(torch.cat([long_feat, short_feat, static_feat], dim=-1))
        fused = gate * long_feat + (1.0 - gate) * short_feat
        out = self.head(torch.cat([fused, static_feat], dim=-1))
        out = out.view(seq.shape[0], self.config.horizon, 3)
        out = sort_quantiles(out)
        if return_attention:
            return out, {"fusion_gate_mean": gate.detach()}
        return out


def config_to_dict(config: DecompTCNGRUConfig) -> dict[str, Any]:
    return _config_to_dict(config)
