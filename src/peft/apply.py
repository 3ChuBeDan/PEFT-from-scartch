from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

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
) -> list[str]:
    """Replace matching Linear modules with LoRA or DoRA wrappers.

    Returns the fully-qualified names of replaced modules.
    """

    method = method.lower()
    if method not in {"lora", "dora"}:
        raise ValueError("method must be 'lora' or 'dora'")

    replacements: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for module_name, module in model.named_modules():
        if not _target_match(module_name, target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        parent = model.get_submodule(parent_name) if parent_name else model
        replacements.append((parent, child_name, module_name, module))

    wrapper_cls = LoRALinear if method == "lora" else DoRALinear
    replaced_names: list[str] = []
    for parent, child_name, module_name, linear in replacements:
        _replace_child(parent, child_name, wrapper_cls.from_linear(linear, rank, alpha, dropout))
        replaced_names.append(module_name)

    if not replaced_names:
        targets = ", ".join(target_modules)
        raise ValueError(f"No nn.Linear modules matched target_modules=[{targets}]")
    return replaced_names
