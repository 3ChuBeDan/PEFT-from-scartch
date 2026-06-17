from __future__ import annotations

import argparse
import csv
import json
import random
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from sklearn.model_selection import train_test_split


TEXT_COLUMNS = ("text", "title", "sentence", "content", "headline")
LABEL_COLUMNS = ("label", "topic", "category", "target", "class")
SPLIT_COLUMNS = ("split", "set", "subset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare UIT-ViON or a compatible news-topic dataset as text,label,split CSV."
    )
    parser.add_argument("--input", required=True, help="Input ZIP, directory, CSV/TSV or JSONL file.")
    parser.add_argument("--output", default="data/uit_vion/dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Optional stratified cap per label for quick subset experiments.",
    )
    return parser.parse_args()


def find_column(fieldnames: Iterable[str], candidates: tuple[str, ...], kind: str) -> str:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(f"Could not infer {kind} column from columns={list(fieldnames)}")


def read_csv_like(path: Path) -> list[dict[str, Any]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def candidate_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".csv", ".tsv", ".jsonl"}:
        return [root]
    if root.is_dir():
        return [
            path
            for suffix in ("*.csv", "*.tsv", "*.jsonl")
            for path in root.rglob(suffix)
            if path.is_file()
        ]
    return []


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(input_path) as archive:
                archive.extractall(tmpdir)
            return load_rows(Path(tmpdir))

    files = candidate_files(input_path)
    if not files:
        raise ValueError(f"No CSV/TSV/JSONL files found under {input_path}")

    rows: list[dict[str, Any]] = []
    for path in sorted(files):
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl(path))
        else:
            rows.extend(read_csv_like(path))
    return rows


def normalize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str | None]:
    if not rows:
        raise ValueError("Input dataset is empty")

    fieldnames = rows[0].keys()
    text_column = find_column(fieldnames, TEXT_COLUMNS, "text")
    label_column = find_column(fieldnames, LABEL_COLUMNS, "label")
    split_column = None
    for candidate in SPLIT_COLUMNS:
        try:
            split_column = find_column(fieldnames, (candidate,), "split")
            break
        except ValueError:
            continue

    normalized = []
    for row in rows:
        text = str(row.get(text_column, "")).strip()
        label = str(row.get(label_column, "")).strip()
        if not text or not label:
            continue
        item = {"text": text, "label": label}
        if split_column is not None:
            split = str(row.get(split_column, "")).strip().lower()
            if split in {"dev", "val"}:
                split = "validation"
            if split:
                item["split"] = split
        normalized.append(item)
    return normalized, split_column


def cap_per_label(rows: list[dict[str, str]], max_per_label: int | None, seed: int) -> list[dict[str, str]]:
    if max_per_label is None:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    capped = []
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        capped.extend(label_rows[:max_per_label])
    rng.shuffle(capped)
    return capped


def add_stratified_splits(
    rows: list[dict[str, str]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> list[dict[str, str]]:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/validation/test ratios must sum to 1.0")

    labels = [row["label"] for row in rows]
    train_rows, temp_rows = train_test_split(
        rows,
        train_size=train_ratio,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [row["label"] for row in temp_rows]
    validation_share = validation_ratio / (validation_ratio + test_ratio)
    validation_rows, test_rows = train_test_split(
        temp_rows,
        train_size=validation_share,
        random_state=seed,
        stratify=temp_labels,
    )

    output = []
    for split_name, split_rows in (
        ("train", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        for row in split_rows:
            output.append({**row, "split": split_name})
    return output


def write_dataset(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, str]], output_path: Path) -> None:
    label_counts = Counter(row["label"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    labels_by_split: dict[str, dict[str, int]] = {}
    for split_name in sorted(split_counts):
        labels_by_split[split_name] = dict(
            sorted(Counter(row["label"] for row in rows if row["split"] == split_name).items())
        )
    summary = {
        "num_records": len(rows),
        "num_labels": len(label_counts),
        "label_counts": dict(sorted(label_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "labels_by_split": labels_by_split,
    }
    summary_path = output_path.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, split_column = normalize_rows(load_rows(Path(args.input)))
    rows = cap_per_label(rows, args.max_per_label, args.seed)
    if split_column is None or any("split" not in row for row in rows):
        rows = add_stratified_splits(
            rows,
            args.seed,
            args.train_ratio,
            args.validation_ratio,
            args.test_ratio,
        )
    output_path = Path(args.output)
    write_dataset(rows, output_path)
    write_summary(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Wrote summary to {output_path.with_name('summary.json')}")


if __name__ == "__main__":
    main()
