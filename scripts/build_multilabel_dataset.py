from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-label JSONL dataset from extracted weak labels")
    parser.add_argument("--input-dir", default="extracted_data/extracted_labels_both")
    parser.add_argument("--output-dir", default="data/multilabel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--keep-empty-text", action="store_true")
    return parser.parse_args()


def labels_from_entry(entry: dict[str, Any]) -> list[str]:
    labels = entry.get("hard_labels") or entry.get("themes") or []
    if not labels and entry.get("top_label"):
        labels = [entry["top_label"]]
    return sorted({str(label).strip() for label in labels if str(label).strip()})


def metadata_from_path(path: Path) -> dict[str, str]:
    parts = path.parts
    year = path.parent.name
    ticker = path.parent.parent.name if len(parts) >= 2 else ""
    output_group = path.parent.parent.parent.name if len(parts) >= 3 else ""
    return {
        "ticker": ticker,
        "year": year,
        "output_group": output_group,
        "source_file": str(path),
    }


def iter_records(input_dir: Path, keep_empty_text: bool) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[tuple[str, tuple[str, ...]]] = set()

    for path in sorted(input_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            skipped["invalid_json"] += 1
            continue

        metadata = metadata_from_path(path)
        for entry in payload.get("extracted", []):
            text = str(entry.get("text") or "").strip()
            if not keep_empty_text and (not text or text == "---"):
                skipped["empty_text"] += 1
                continue

            labels = labels_from_entry(entry)
            if not labels:
                skipped["no_labels"] += 1
                continue

            dedupe_key = (text, tuple(labels))
            if dedupe_key in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(dedupe_key)

            records.append(
                {
                    "text": text,
                    "labels": labels,
                    **metadata,
                }
            )

    return records, skipped


def assign_splits(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> list[dict[str, Any]]:
    total_ratio = train_ratio + validation_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("At least one split ratio must be positive")

    train_cutoff = train_ratio / total_ratio
    validation_cutoff = (train_ratio + validation_ratio) / total_ratio

    doc_keys = sorted({(record["ticker"], record["year"], record["source_file"]) for record in records})
    rng = random.Random(seed)
    rng.shuffle(doc_keys)

    split_by_doc: dict[tuple[str, str, str], str] = {}
    for idx, key in enumerate(doc_keys):
        fraction = idx / max(1, len(doc_keys))
        if fraction < train_cutoff:
            split = "train"
        elif fraction < validation_cutoff:
            split = "validation"
        else:
            split = "test"
        split_by_doc[key] = split

    with_splits = []
    for record in records:
        doc_key = (record["ticker"], record["year"], record["source_file"])
        with_splits.append({**record, "split": split_by_doc[doc_key]})
    return with_splits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]], skipped: Counter[str], output_dir: Path) -> dict[str, Any]:
    split_counts = Counter(record["split"] for record in records)
    label_counts = Counter(label for record in records for label in record["labels"])
    labels_by_split = {
        split: Counter(label for record in records if record["split"] == split for label in record["labels"])
        for split in ["train", "validation", "test"]
    }
    summary = {
        "num_records": len(records),
        "split_counts": dict(split_counts),
        "num_labels": len(label_counts),
        "label_counts": dict(sorted(label_counts.items())),
        "multi_label_records": sum(1 for record in records if len(record["labels"]) > 1),
        "skipped": dict(skipped),
        "labels_by_split": {
            split: dict(sorted(counts.items()))
            for split, counts in labels_by_split.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = iter_records(input_dir, keep_empty_text=args.keep_empty_text)
    records = assign_splits(
        records,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    labels = sorted({label for record in records for label in record["labels"]})
    label_map = {label: idx for idx, label in enumerate(labels)}
    (output_dir / "label_map.json").write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")

    write_jsonl(output_dir / "dataset.jsonl", records)
    for split in ["train", "validation", "test"]:
        write_jsonl(output_dir / f"{split}.jsonl", [record for record in records if record["split"] == split])

    summary = summarize(records, skipped, output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
