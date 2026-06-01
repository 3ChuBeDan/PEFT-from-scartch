from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """Weight-decomposed low-rank adapter for an existing Linear layer.

    For a PyTorch Linear weight W with shape [out_features, in_features],
    DoRA keeps W0 frozen, learns a low-rank direction update BA, and learns one
    magnitude scalar per output row:

        W = m * normalize(W0 + BA, dim=1)
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for DoRA")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        self.scaling = self.alpha / self.rank
        self.eps = eps
        self.dropout = nn.Dropout(dropout)

        w0 = linear.weight.detach().clone()
        self.register_buffer("weight_base", w0)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.magnitude = nn.Parameter(torch.linalg.vector_norm(w0, dim=1, keepdim=True))
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
    ) -> "DoRALinear":
        return cls(linear=linear, rank=rank, alpha=alpha, dropout=dropout)

    def direction_update(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        direction = self.weight_base + self.direction_update().to(dtype=self.weight_base.dtype)
        norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(self.eps)
        return self.magnitude.to(dtype=direction.dtype) * direction / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.merged_weight(), self.bias)

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight_base.device,
            dtype=self.weight_base.dtype,
        )
        merged.weight.data.copy_(self.merged_weight())
        if self.bias is not None:
            merged.bias.data.copy_(self.bias)
        return merged
