from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Low-rank adapter for an existing Linear layer.

    The wrapped pretrained weight and bias are frozen. Only lora_A and lora_B
    are trainable, and the adapter can be merged into a plain nn.Linear layer.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for LoRA")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ) -> "LoRALinear":
        return cls(linear=linear, rank=rank, alpha=alpha, dropout=dropout)

    def adapter_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.weight + self.adapter_weight().to(dtype=self.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        adapter = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + adapter * self.scaling

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        merged.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            merged.bias.data.copy_(self.bias)
        return merged
