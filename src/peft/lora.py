from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Low-rank adapter for an existing Linear layer.

    The wrapped pretrained weight and bias are frozen. Only lora_A and lora_B
    are trainable, and the adapter can be merged into a plain nn.Linear layer.

    Features:
    - merge / unmerge: merge adapter into the base weight and restore.
    - enable / disable: temporarily turn off the adapter.
    - save / load adapter: persist only the trainable parameters.
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

        # Store original weight and bias
        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        # Low-rank trainable parameters
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # State flags
        self.adapter_enabled = True
        self.merged = False

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

    def merge_adapter(self) -> LoRALinear:
        """Merge the adapter into the base weight (in-place).

        The module behaves like a plain nn.Linear while merged.
        """
        if self.merged:
            return self  # already merged

        merged_w = self.merged_weight()
        self.weight.data.copy_(merged_w)
        self.merged = True
        return self

    def unmerge_adapter(self) -> LoRALinear:
        """Restore the original base weight, keeping the adapter trainable."""
        if not self.merged:
            return self

        # Subtract the current adapter contribution from the merged weight
        adapter = self.adapter_weight().to(dtype=self.weight.dtype)
        self.weight.data.sub_(adapter)
        self.merged = False
        return self

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    @contextmanager
    def disable_adapter_context(self):
        """Context manager that temporarily disables the adapter."""
        prev = self.adapter_enabled
        self.disable_adapter()
        try:
            yield
        finally:
            self.adapter_enabled = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            # Merged mode: weight already includes adapter
            return F.linear(x, self.weight, self.bias)

        base = F.linear(x, self.weight, self.bias)
        if not self.adapter_enabled:
            return base

        adapter = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base + adapter * self.scaling

    def save_adapter(self, path: str) -> None:
        """Save only the trainable adapter parameters to a file."""
        state = {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
            "alpha": self.alpha,
            "rank": self.rank,
            "dropout": self.dropout.p if self.dropout.p > 0 else 0.0,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }
        torch.save(state, path)

    def load_adapter(self, path: str) -> None:
        """Load adapter parameters from a file.

        The base weight and structure must already match.
        """
        state = torch.load(path, map_location=self.weight.device)
        if state["in_features"] != self.in_features or state["out_features"] != self.out_features:
            raise ValueError("Adapter dimensions do not match the current layer.")
        self.lora_A.data.copy_(state["lora_A"])
        self.lora_B.data.copy_(state["lora_B"])
        self.alpha = state["alpha"]
        self.rank = state["rank"]
        self.scaling = self.alpha / self.rank

    def merge(self) -> nn.Linear:
        """Permanently merge adapter and return a plain nn.Linear."""
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