from __future__ import annotations

from transformers import AutoModelForSequenceClassification


def build_phobert_classifier(
    model_name: str = "vinai/phobert-base-v2",
    num_labels: int = 3,
    problem_type: str | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=problem_type,
        id2label=id2label,
        label2id=label2id,
    )
