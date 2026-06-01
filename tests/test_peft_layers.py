from __future__ import annotations

import torch
from torch import nn

from src.peft import DoRALinear, LoRALinear, apply_peft, count_parameters


def test_lora_initial_output_matches_linear() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = LoRALinear.from_linear(linear, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(7, 5)

    torch.testing.assert_close(wrapped(x), linear(x))


def test_dora_initial_output_matches_linear() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = DoRALinear.from_linear(linear, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(7, 5)

    torch.testing.assert_close(wrapped(x), linear(x), atol=1e-6, rtol=1e-6)


def test_only_adapter_parameters_are_trainable_in_wrappers() -> None:
    linear = nn.Linear(5, 3)
    lora = LoRALinear.from_linear(linear, rank=2)
    dora = DoRALinear.from_linear(linear, rank=2)

    assert {name for name, p in lora.named_parameters() if p.requires_grad} == {"lora_A", "lora_B"}
    assert {name for name, p in dora.named_parameters() if p.requires_grad} == {
        "magnitude",
        "lora_A",
        "lora_B",
    }


def test_merge_preserves_lora_logits() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = LoRALinear.from_linear(linear, rank=2, alpha=4)
    wrapped.lora_B.data.normal_(mean=0.0, std=0.02)
    x = torch.randn(7, 5)

    merged = wrapped.merge()
    torch.testing.assert_close(wrapped(x), merged(x), atol=1e-6, rtol=1e-6)


def test_merge_preserves_dora_logits() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = DoRALinear.from_linear(linear, rank=2, alpha=4)
    wrapped.lora_B.data.normal_(mean=0.0, std=0.02)
    wrapped.magnitude.data.mul_(1.01)
    x = torch.randn(7, 5)

    merged = wrapped.merge()
    torch.testing.assert_close(wrapped(x), merged(x), atol=1e-6, rtol=1e-6)


def test_apply_peft_replaces_target_linear_and_counts_trainable_params() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    for param in model.parameters():
        param.requires_grad = False

    replaced = apply_peft(
        model=model,
        method="dora",
        rank=2,
        alpha=4,
        dropout=0.0,
        target_modules=["0"],
    )
    counts = count_parameters(model)

    assert replaced == ["0"]
    assert isinstance(model[0], DoRALinear)
    assert counts.trainable == model[0].magnitude.numel() + model[0].lora_A.numel() + model[0].lora_B.numel()
