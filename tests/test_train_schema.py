from __future__ import annotations

from train import RESULT_COLUMNS


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