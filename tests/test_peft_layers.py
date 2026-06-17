from __future__ import annotations

import pytest
import torch
from torch import nn

from src.peft import (
    DoRALinear,
    LoRALinear,
    apply_peft,
    count_parameters,
    save_adapters,
    load_adapters,
)


def test_lora_initial_output_matches_linear() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = LoRALinear.from_linear(linear, rank=2, alpha=4, dropout=0.0)
    x = torch.randn(7, 5)

    torch.testing.assert_close(wrapped(x), linear(x))


def test_dora_initial_output_matches_linear() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = DoRALinear.from_linear(linear, rank=2, alpha=4)
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



def test_dora_disable_adapter_produces_base_output() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    wrapped = DoRALinear.from_linear(linear, rank=2, alpha=4)
    x = torch.randn(7, 5)

    out_enabled = wrapped(x)
    wrapped.disable_adapter()
    out_disabled = wrapped(x)
    wrapped.enable_adapter()
    out_reenabled = wrapped(x)

    torch.testing.assert_close(out_disabled, linear(x))
    torch.testing.assert_close(out_reenabled, out_enabled)


def test_dora_magnitude_initialization() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)

    dora_wn = DoRALinear.from_linear(linear, rank=2, alpha=4, init_magnitude="weight_norm")
    dora_ones = DoRALinear.from_linear(linear, rank=2, alpha=4, init_magnitude="ones")

    expected_norm = torch.linalg.vector_norm(linear.weight, dim=0, keepdim=True)
    torch.testing.assert_close(dora_wn.magnitude, expected_norm)
    torch.testing.assert_close(dora_ones.magnitude, torch.ones(1, 5))


def test_save_and_load_adapter_preserves_output(tmp_path) -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 3))
    apply_peft(model, method="dora", rank=2, alpha=4, dropout=0.0, target_modules=["0"])
    x = torch.randn(7, 5)
    out_before = model(x).detach().clone()

    path = tmp_path / "adapter.pt"
    save_adapters(model, path)

    new_model = nn.Sequential(nn.Linear(5, 3))
    apply_peft(new_model, method="dora", rank=2, alpha=4, dropout=0.0, target_modules=["0"])
    load_adapters(new_model, path)

    out_after = new_model(x).detach().clone()
    torch.testing.assert_close(out_after, out_before)


def test_apply_peft_raises_on_dora_dropout() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(ValueError, match="DoRA does not support dropout"):
        apply_peft(model, method="dora", rank=2, alpha=4, dropout=0.1, target_modules=["0"])


def test_lora_unmerge_restores_original_weight() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 3)
    original_weight = linear.weight.data.clone()
    wrapped = LoRALinear.from_linear(linear, rank=2, alpha=4)
    wrapped.lora_B.data.normal_(mean=0.0, std=0.02)
    wrapped.merge_adapter()
    wrapped.unmerge_adapter()
    torch.testing.assert_close(wrapped.weight, original_weight)