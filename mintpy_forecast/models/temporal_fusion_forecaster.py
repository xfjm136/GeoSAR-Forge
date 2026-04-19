"""Temporal Fusion Transformer forecaster for InSAR displacement prediction.

Architecture:
- Linear projection + continuous time positional encoding
- Multi-head self-attention for temporal dependencies
- Gated fusion with static features
- Optional neighbor context via cross-attention
- Quantile output (p10/p50/p90)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch
from torch import nn

from .common import sort_quantiles


@dataclass
class TemporalFusionConfig:
    input_dim: int
    static_dim: int
    hidden_dim: int = 96
    n_heads: int = 4
    n_attn_layers: int = 2
    dropout_rate: float = 0.12
    horizon: int = 3
    edge_dim: int = 4
    use_neighbor_context: bool = True


class _ContinuousTimeEncoding(nn.Module):
    """Positional encoding from actual day offsets for irregular SAR revisit times."""

    def __init__(self, dim: int, n_frequencies: int = 16) -> None:
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_frequencies) * 0.02)
        self.proj = nn.Linear(n_frequencies * 2, dim)

    def forward(self, day_offsets: torch.Tensor) -> torch.Tensor:
        t = day_offsets.unsqueeze(-1)
        angles = t * self.freqs.unsqueeze(0).unsqueeze(0)
        pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(pe)


class _TemporalSelfAttention(nn.Module):
    """Pre-norm multi-head self-attention with feed-forward."""

    def __init__(self, dim: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class _NeighborContextModule(nn.Module):
    """Lightweight neighbor attention for spatial context."""

    def __init__(self, input_dim: int, static_dim: int, edge_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.center_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(input_dim + static_dim + edge_dim, hidden_dim)
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        center_feat: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_static: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nb_pooled = neighbor_seq.mean(dim=1)
        nb_input = torch.cat([nb_pooled, neighbor_static, edge_features], dim=-1)
        nb_embed = self.neighbor_proj(nb_input)
        center_expand = self.center_proj(center_feat).unsqueeze(1).expand_as(nb_embed)
        score_in = torch.cat([center_expand, nb_embed], dim=-1)
        scores = self.attn_score(score_in).squeeze(-1)
        scores = scores.masked_fill(~neighbor_mask, float("-inf"))
        no_neighbor = ~neighbor_mask.any(dim=1)
        if torch.any(no_neighbor):
            scores[no_neighbor] = 0.0
        weights = torch.softmax(scores, dim=-1)
        if torch.any(no_neighbor):
            weights[no_neighbor] = 0.0
        aggregated = (weights.unsqueeze(-1) * nb_embed).sum(dim=1)
        gate = self.gate(torch.cat([center_feat, aggregated], dim=-1))
        out = center_feat + self.dropout(gate * aggregated)
        return out, weights


class TemporalFusionForecaster(nn.Module):
    """Temporal Fusion Transformer for InSAR displacement forecasting.

    Key improvements over DecompTCNGRU:
    1. Multi-head self-attention captures arbitrary temporal dependencies
    2. Continuous time encoding handles irregular SAR revisit times
    3. Learned temporal gating for adaptive pooling
    """

    def __init__(self, config: TemporalFusionConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(config.input_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )
        self.time_encoding = _ContinuousTimeEncoding(d)
        self.attn_layers = nn.ModuleList([
            _TemporalSelfAttention(d, config.n_heads, config.dropout_rate)
            for _ in range(config.n_attn_layers)
        ])
        self.temporal_gate = nn.Sequential(
            nn.Linear(d, d),
            nn.Sigmoid(),
        )
        self.static_proj = nn.Sequential(
            nn.Linear(config.static_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        if config.use_neighbor_context:
            self.neighbor_ctx = _NeighborContextModule(
                config.input_dim, config.static_dim, config.edge_dim, d, config.dropout_rate
            )
        else:
            self.neighbor_ctx = None

        self.head = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(d, config.horizon * 3),
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
        B, T, C = seq.shape
        x = self.input_proj(seq)

        day_gap_idx = 2
        day_offsets = seq[:, :, day_gap_idx].cumsum(dim=1)
        x = x + self.time_encoding(day_offsets)

        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        gate = self.temporal_gate(x)
        temporal_feat = (gate * x).mean(dim=1)
        static_feat = self.static_proj(static)

        if self.neighbor_ctx is not None and neighbor_seq is not None:
            temporal_feat, nb_weights = self.neighbor_ctx(
                temporal_feat, neighbor_seq, neighbor_static, edge_features, neighbor_mask
            )
        else:
            nb_weights = None

        fused = torch.cat([temporal_feat, static_feat], dim=-1)
        out = self.head(fused)
        out = out.view(B, self.config.horizon, 3)
        out = sort_quantiles(out)

        if return_attention:
            aux = {}
            if nb_weights is not None:
                aux["neighbor_attention_mean"] = nb_weights.detach()
            return out, aux
        return out


def config_to_dict(config: TemporalFusionConfig) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        return asdict(config)
    return dict(config)
