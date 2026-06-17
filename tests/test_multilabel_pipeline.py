from __future__ import annotations

import numpy as np
from datasets import Dataset, DatasetDict

from train import encode_multilabels, make_multilabel_compute_metrics, normalize_label_list


def test_normalize_label_list_accepts_json_and_delimited_strings() -> None:
    assert normalize_label_list(["G10", "E2", "E2"]) == ["E2", "G10"]
    assert normalize_label_list("G10|E2") == ["E2", "G10"]
    assert normalize_label_list('["G10", "E2"]') == ["E2", "G10"]


def test_encode_multilabels_builds_multi_hot_vectors() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"text": "a", "labels": ["E2", "G10"]},
                    {"text": "b", "labels": ["S6"]},
                ]
            ),
            "validation": Dataset.from_list([{"text": "c", "labels": ["G10"]}]),
        }
    )

    encoded, num_labels, label_map = encode_multilabels(dataset, "labels", label_map_path=None)

    assert num_labels == 3
    assert label_map == {"E2": 0, "G10": 1, "S6": 2}
    assert encoded["train"][0]["labels"] == [1.0, 1.0, 0.0]
    assert encoded["train"][1]["labels"] == [0.0, 0.0, 1.0]


def test_multilabel_metrics_use_sigmoid_threshold() -> None:
    logits = np.array([[5.0, -5.0, 5.0], [-5.0, 5.0, -5.0]])
    labels = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    metrics = make_multilabel_compute_metrics(0.5)((logits, labels))

    assert metrics["accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["hamming_loss"] == 0.0