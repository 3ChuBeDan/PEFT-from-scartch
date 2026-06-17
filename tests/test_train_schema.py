from __future__ import annotations

import argparse

from torch import nn

from train import (
    RESULT_COLUMNS,
    configure_method,
    load_named_or_local_dataset,
)


def test_benchmark_result_schema_contains_required_columns() -> None:
    required = {
        "method",
        "rank",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "trainable_percent",
        "train_time_sec",
        "peak_vram_mb",
        "checkpoint_size_mb",
    }

    assert required.issubset(set(RESULT_COLUMNS))


def test_uit_vion_alias_loads_prepared_local_csv(tmp_path, monkeypatch) -> None:
    data_dir = tmp_path / "data" / "uit_vion"
    data_dir.mkdir(parents=True)
    dataset_path = data_dir / "dataset.csv"
    dataset_path.write_text(
        "text,label,split\n"
        "Tin cong nghe,TECHNOLOGY,train\n"
        "Tin the thao,SPORT,validation\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("train.DEFAULT_UIT_VION_PATH", dataset_path)

    dataset = load_named_or_local_dataset("uit-vion")

    assert set(dataset.keys()) == {"train", "validation"}
    assert dataset["train"][0]["text"] == "Tin cong nghe"


def test_configure_method_records_effective_zero_dropout_for_dora() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    args = argparse.Namespace(
        method="dora",
        rank=2,
        alpha=4,
        dropout=0.05,
        target_modules="0",
        init_magnitude="weight_norm",
        no_detach=False,
    )

    replaced = configure_method(args, model)

    assert replaced == ["0"]
    assert args.dropout == 0.0
