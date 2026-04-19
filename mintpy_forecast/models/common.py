"""Shared forecasting model blocks."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.conv1(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        y = self.act(self.norm1(y))
        y = self.dropout(y)
        y = self.conv2(y)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        y = self.dropout(self.act(self.norm2(y)))
        return residual + y


class TemporalConvEncoder(nn.Module):
    def __init__(self, channels: int, *, dropout: float, dilations: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ResidualTemporalBlock(channels, dilation=d, dropout=dropout) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(weight.unsqueeze(-1) * x, dim=1)
        return pooled, weight


def sort_quantiles(out: torch.Tensor) -> torch.Tensor:
    return torch.sort(out, dim=-1).values


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"Unsupported config type: {type(config)!r}")
