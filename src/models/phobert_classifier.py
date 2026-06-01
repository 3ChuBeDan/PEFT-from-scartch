from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def build_phobert_classifier(
    model_name: str = "vinai/phobert-base-v2",
    num_labels: int = 3,
):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
