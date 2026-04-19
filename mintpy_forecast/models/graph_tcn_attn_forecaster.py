"""Lightweight graph-context TCN forecaster for hazard mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .common import TemporalAttentionPool, TemporalConvEncoder, config_to_dict as _config_to_dict, sort_quantiles


@dataclass
class GraphTCNAttnConfig:
    input_dim: int
    static_dim: int
    edge_dim: int
    hidden_dim: int = 96
    dropout_rate: float = 0.15
    horizon: int = 3
    message_passing_steps: int = 2


class _NeighborAggregator(nn.Module):
    def __init__(self, input_dim: int, static_dim: int, edge_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.center_proj = nn.Linear(input_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(input_dim + static_dim + edge_dim, hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        center_seq: torch.Tensor,
        neighbor_seq: torch.Tensor,
        neighbor_static: torch.Tensor,
        edge_features: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        center_embed = self.center_proj(center_seq)
        static_expand = neighbor_static.unsqueeze(1).expand(-1, neighbor_seq.shape[1], -1, -1)
        edge_expand = edge_features.unsqueeze(1).expand(-1, neighbor_seq.shape[1], -1, -1)
        neighbor_input = torch.cat([neighbor_seq, static_expand, edge_expand], dim=-1)
        neighbor_embed = self.neighbor_proj(neighbor_input)
        score_in = torch.cat([center_embed.unsqueeze(2).expand_as(neighbor_embed), neighbor_embed], dim=-1)
        score = self.score(score_in).squeeze(-1)
        score = score.masked_fill(~neighbor_mask.unsqueeze(1), float("-inf"))
        no_neighbor = ~neighbor_mask.any(dim=1)
        if torch.any(no_neighbor):
            score[no_neighbor] = 0.0
        weight = torch.softmax(score, dim=-1)
        if torch.any(no_neighbor):
            weight[no_neighbor] = 0.0
        aggregated = torch.sum(weight.unsqueeze(-1) * neighbor_embed, dim=2)
        out = self.out(torch.cat([center_embed, aggregated], dim=-1))
        return center_embed + out, weight


class GraphTCNAttnQuantileForecaster(nn.Module):
    def __init__(self, config: GraphTCNAttnConfig) -> None:
        super().__init__()
        self.config = config
        self.aggregators = nn.ModuleList(
            [
                _NeighborAggregator(
                    input_dim=config.input_dim if idx == 0 else config.hidden_dim,
                    static_dim=config.static_dim,
                    edge_dim=config.edge_dim,
                    hidden_dim=config.hidden_dim,
                    dropout=config.dropout_rate,
                )
                for idx in range(config.message_passing_steps)
            ]
        )
        self.neighbor_seq_proj = nn.ModuleList(
            [nn.Identity()] + [nn.Linear(config.input_dim, config.hidden_dim) for _ in range(max(config.message_passing_steps - 1, 0))]
        )
        self.tcn = TemporalConvEncoder(config.hidden_dim, dropout=config.dropout_rate)
        self.temporal_pool = TemporalAttentionPool(config.hidden_dim)
        self.static_mlp = nn.Sequential(
            nn.Linear(config.static_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
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
        if neighbor_seq is None or neighbor_static is None or edge_features is None or neighbor_mask is None:
            raise ValueError("hazard 模型需要 neighbor_seq / neighbor_static / edge_features / neighbor_mask")

        center = seq
        attn_weights: list[torch.Tensor] = []
        current_neighbor_seq = neighbor_seq
        for idx, aggregator in enumerate(self.aggregators):
            if idx > 0:
                current_neighbor_seq = self.neighbor_seq_proj[idx](neighbor_seq)
            center, weight = aggregator(center, current_neighbor_seq, neighbor_static, edge_features, neighbor_mask)
            attn_weights.append(weight)

        encoded = self.tcn(center.transpose(1, 2)).transpose(1, 2)
        temporal_feat, temporal_weight = self.temporal_pool(encoded)
        static_feat = self.static_mlp(static)
        out = self.head(torch.cat([temporal_feat, static_feat], dim=-1))
        out = out.view(seq.shape[0], self.config.horizon, 3)
        out = sort_quantiles(out)
        if return_attention:
            mean_attn = torch.stack(attn_weights, dim=0).mean(dim=0).mean(dim=1)
            return out, {
                "neighbor_attention_mean": mean_attn.detach(),
                "temporal_attention": temporal_weight.detach(),
            }
        return out


def config_to_dict(config: GraphTCNAttnConfig) -> dict[str, Any]:
    return _config_to_dict(config)
