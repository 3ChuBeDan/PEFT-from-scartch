from __future__ import annotations

import csv
import json
import subprocess
import sys


def test_prepare_uit_vion_creates_stratified_dataset_and_summary(tmp_path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "dataset.csv"
    rows = ["title,topic"]
    for label in ["TECHNOLOGY", "SPORT", "LAW"]:
        for idx in range(10):
            rows.append(f"{label} title {idx},{label}")
    input_path.write_text("\n".join(rows), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_uit_vion.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            "42",
        ],
        check=True,
    )

    with output_path.open(encoding="utf-8", newline="") as handle:
        prepared = list(csv.DictReader(handle))
    summary = json.loads(output_path.with_name("summary.json").read_text(encoding="utf-8"))

    assert set(prepared[0]) == {"text", "label", "split"}
    assert len(prepared) == 30
    assert summary["num_labels"] == 3
    assert summary["split_counts"] == {"test": 3, "train": 24, "validation": 3}
