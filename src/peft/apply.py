from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from torch import nn

from .dora import DoRALinear
from .lora import LoRALinear


@dataclass(frozen=True)
class ParameterCounts:
    total: int
    trainable: int

    @property
    def trainable_percent(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0


def count_parameters(model: nn.Module) -> ParameterCounts:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParameterCounts(total=total, trainable=trainable)


def _replace_child(parent: nn.Module, name: str, new_module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(name)] = new_module
    else:
        setattr(parent, name, new_module)


def _target_match(module_name: str, target_modules: Iterable[str]) -> bool:
    leaf = module_name.rsplit(".", 1)[-1]
    return any(leaf == target or module_name.endswith(target) for target in target_modules)


def apply_peft(
    model: nn.Module,
    method: str,
    rank: int,
    alpha: float | None,
    dropout: float,
    target_modules: Iterable[str],
    init_magnitude: str = "weight_norm",
    use_detached_gradient: bool = True,
) -> list[str]:
    """Replace matching Linear modules with LoRA or DoRA wrappers.

    Args:
        model: The model to modify.
        method: 'lora' or 'dora'.
        rank: Rank of low-rank update.
        alpha: Scaling factor.
        dropout: Dropout rate (only for LoRA; DoRA will raise an error if >0).
        target_modules: Names of modules to replace (e.g., ['q_proj', 'v_proj']).
        init_magnitude: Only for DoRA, 'weight_norm' or 'ones'.
        use_detached_gradient: Only for DoRA, whether to detach norm (default True).

    Returns:
        Fully-qualified names of replaced modules.
    """
    method = method.lower()
    if method not in {"lora", "dora"}:
        raise ValueError("method must be 'lora' or 'dora'")

    # DoRA does not support dropout
    if method == "dora" and dropout != 0.0:
        raise ValueError("DoRA does not support dropout; set dropout=0.0.")

    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        if not _target_match(module_name, target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, child_name = (
            module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        )
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module_name, module))

    replaced_names: list[str] = []
    for parent, child_name, module_name, linear in replacements:
        if method == "lora":
            wrapper = LoRALinear.from_linear(linear, rank, alpha, dropout)
        else:  # dora
            wrapper = DoRALinear.from_linear(
                linear,
                rank,
                alpha,
                init_magnitude,
                use_detached_gradient,
            )
        _replace_child(parent, child_name, wrapper)
        replaced_names.append(module_name)

    if not replaced_names:
        targets = ", ".join(target_modules)
        raise ValueError(f"No nn.Linear modules matched target_modules=[{targets}]")
    return replaced_names


def get_adapter_modules(model: nn.Module) -> dict[str, nn.Module]:
    """Return all LoRA/DoRA adapter modules currently in the model."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (LoRALinear, DoRALinear))
    }


def save_adapters(model: nn.Module, path: str) -> None:
    """Save all adapter weights from a model into a single file."""
    adapters = get_adapter_modules(model)
    state = {}
    for name, mod in adapters.items():
        if isinstance(mod, LoRALinear):
            state[name] = {
                "lora_A": mod.lora_A.data,
                "lora_B": mod.lora_B.data,
                "alpha": mod.alpha,
                "rank": mod.rank,
                "dropout": mod.dropout.p if mod.dropout.p > 0 else 0.0,
            }
        elif isinstance(mod, DoRALinear):
            state[name] = {
                "lora_A": mod.lora_A.data,
                "lora_B": mod.lora_B.data,
                "magnitude": mod.magnitude.data,
                "alpha": mod.alpha,
                "rank": mod.rank,
                "eps": mod.eps,
                "init_magnitude": mod._init_magnitude,
                "use_detached_gradient": mod.use_detached_gradient,
            }
    torch.save(state, path)


def load_adapters(model: nn.Module, path: str) -> None:
    """Load all adapter weights from a file into the model's adapter modules."""
    adapters = get_adapter_modules(model)
    state = torch.load(path, map_location="cpu")
    for name, params in state.items():
        if name not in adapters:
            raise KeyError(f"Adapter module '{name}' not found in model.")
        mod = adapters[name]
        if isinstance(mod, LoRALinear):
            mod.lora_A.data.copy_(params["lora_A"])
            mod.lora_B.data.copy_(params["lora_B"])
            mod.alpha = params["alpha"]
            mod.rank = params["rank"]
            mod.scaling = mod.alpha / mod.rank
        elif isinstance(mod, DoRALinear):
            mod.lora_A.data.copy_(params["lora_A"])
            mod.lora_B.data.copy_(params["lora_B"])
            mod.magnitude.data.copy_(params["magnitude"])
            mod.alpha = params["alpha"]
            mod.rank = params["rank"]
            mod.scaling = mod.alpha / mod.rank
            # Restore behaviour flags
            if "use_detached_gradient" in params:
                mod.use_detached_gradient = params["use_detached_gradient"]
