from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class DoRALinear(nn.Module):
    """Weight-decomposed low-rank adapter for an existing Linear layer.

    For a PyTorch Linear weight W with shape [out_features, in_features],
    DoRA keeps W0 frozen, learns a low-rank direction update BA, and learns one
    magnitude scalar per input column:

        W = m * normalize(W0 + BA, dim=0)

    Features:
    - merge: merge adapter into the base weight or export a plain Linear.
    - enable / disable: temporarily turn off the adapter.
    - save / load adapter: persist only the trainable parameters.
    - magnitude initialisation: 'weight_norm' (default) or 'ones'.
    - use_detached_gradient: whether to detach the norm from the computational
        graph during forward (default True). This saves VRAM (~24.4% on LLaMA-7B)
        with negligible accuracy loss (~0.2%).
    - Dropout is NOT supported (raises error if attempted).
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        eps: float = 1e-6,
        init_magnitude: str = "weight_norm",
        use_detached_gradient: bool = True,
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
        self.use_detached_gradient = use_detached_gradient

        w0 = linear.weight.detach().clone()
        self.register_buffer("weight_base", w0)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        # Magnitude initialisation
        if init_magnitude == "weight_norm":
            mag_init = torch.linalg.vector_norm(w0, dim=0, keepdim=True)
        elif init_magnitude == "ones":
            mag_init = torch.ones(1, self.in_features, device=w0.device, dtype=w0.dtype)
        else:
            raise ValueError(f"Unknown init_magnitude: {init_magnitude}")

        self.magnitude = nn.Parameter(mag_init)
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # State flags
        self.adapter_enabled = True
        self.merged = False
        self._init_magnitude = init_magnitude

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        alpha: float | None = None,
        init_magnitude: str = "weight_norm",
        use_detached_gradient: bool = True,
    ) -> "DoRALinear":
        return cls(
            linear=linear,
            rank=rank,
            alpha=alpha,
            init_magnitude=init_magnitude,
            use_detached_gradient=use_detached_gradient,
        )

    def direction_update(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        direction = self.weight_base + self.direction_update().to(dtype=self.weight_base.dtype)
        # Column-wise norm
        norm = torch.linalg.vector_norm(direction, dim=0, keepdim=True)
        # Optionally detach the norm to save VRAM with negligible accuracy loss
        if self.use_detached_gradient:
            norm = norm.detach()
        norm = norm.clamp_min(self.eps)
        return self.magnitude.to(dtype=direction.dtype) * direction / norm

    def merge_adapter(self) -> DoRALinear:
        """Merge the DoRA adapter into weight_base (in-place).

        The module behaves like a plain nn.Linear while merged.
        """
        if self.merged:
            return self

        merged_w = self.merged_weight()
        self.weight_base.data.copy_(merged_w)
        self.merged = True
        return self

    def unmerge_adapter(self) -> DoRALinear:
        """Restore the original weight_base, keeping the adapter trainable."""
        if not self.merged:
            return self
        # Unmerge is not straightforward because magnitude is learnt.
        # A proper implementation would store a backup before merging.
        raise NotImplementedError(
            "DoRA unmerge is not supported yet. Use merge() to create a plain Linear, "
            "or re-initialise the adapter."
        )

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
            return F.linear(x, self.weight_base, self.bias)

        if not self.adapter_enabled:
            # Use the frozen base weight without any update
            return F.linear(x, self.weight_base, self.bias)

        return F.linear(x, self.merged_weight(), self.bias)

    def save_adapter(self, path: str) -> None:
        """Save only the trainable adapter parameters to a file."""
        state = {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
            "magnitude": self.magnitude.data,
            "alpha": self.alpha,
            "rank": self.rank,
            "eps": self.eps,
            "init_magnitude": self._init_magnitude,
            "use_detached_gradient": self.use_detached_gradient,
            "in_features": self.in_features,
            "out_features": self.out_features,
        }
        torch.save(state, path)

    def load_adapter(self, path: str) -> None:
        """Load adapter parameters from a file.

        The base weight and structure must already match.
        """
        state = torch.load(path, map_location=self.weight_base.device)
        if state["in_features"] != self.in_features or state["out_features"] != self.out_features:
            raise ValueError("Adapter dimensions do not match the current layer.")
        self.lora_A.data.copy_(state["lora_A"])
        self.lora_B.data.copy_(state["lora_B"])
        self.magnitude.data.copy_(state["magnitude"])
        self.alpha = state["alpha"]
        self.rank = state["rank"]
        self.scaling = self.alpha / self.rank
        # Restore behaviour flags (if present)
        if "use_detached_gradient" in state:
            self.use_detached_gradient = state["use_detached_gradient"]

    def merge(self) -> nn.Linear:
        """Permanently merge adapter and return a plain nn.Linear."""
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
