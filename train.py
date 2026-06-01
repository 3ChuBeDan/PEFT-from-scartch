from __future__ import annotations

import argparse
import csv
import json
import os
import random
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
from datasets import DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.models import build_phobert_classifier
from src.peft import apply_peft, count_parameters


def require_runtime_dependencies() -> None:
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: accelerate>=1.1.0 is required by transformers.Trainer. "
            "Install dependencies in the active notebook/kernel environment with: "
            f"{sys.executable} -m pip install -r requirements.txt"
        ) from exc


DEFAULT_MODEL = "vinai/phobert-base-v2"
DEFAULT_TARGET_MODULES = "query,value"
RESULT_COLUMNS = [
    "run_id",
    "method",
    "model_name",
    "dataset",
    "rank",
    "alpha",
    "dropout",
    "seed",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "trainable_params",
    "total_params",
    "trainable_percent",
    "train_time_sec",
    "peak_vram_mb",
    "checkpoint_size_mb",
    "output_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhoBERT FT/LoRA/DoRA benchmark trainer")
    parser.add_argument("--method", choices=["ft", "lora", "dora"], required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default="uit-vsfc")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--results-file", default="results/benchmark_results.csv")
    parser.add_argument("--target-modules", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_dataset_name(name: str) -> str:
    return name.strip().lower()


def load_uit_vsfc_or_csv(dataset_arg: str) -> DatasetDict:
    dataset_key = normalize_dataset_name(dataset_arg)
    if dataset_key == "uit-vsfc":
        try:
            return load_dataset("uitnlp/vietnamese_students_feedback", trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                "Could not load UIT-VSFC from Hugging Face. Pass a local CSV path "
                "with columns text,label,split via --dataset."
            ) from exc

    dataset_path = Path(dataset_arg)
    if dataset_path.suffix.lower() != ".csv":
        raise ValueError("--dataset must be 'uit-vsfc' or a local CSV path")
    raw = load_dataset("csv", data_files=str(dataset_path))["train"]
    if "split" not in raw.column_names:
        raise ValueError("Local CSV fallback must include a split column")
    splits = {}
    for split_name in sorted(set(raw["split"])):
        split_data = raw.filter(lambda row, split_name=split_name: row["split"] == split_name)
        splits[str(split_name)] = split_data
    return DatasetDict(splits)


def infer_text_column(column_names: list[str]) -> str:
    for candidate in ["text", "sentence", "review", "comment", "content"]:
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not infer text column from columns={column_names}")


def infer_label_column(column_names: list[str]) -> str:
    for candidate in ["label", "sentiment", "labels", "target"]:
        if candidate in column_names:
            return candidate
    raise ValueError(f"Could not infer label column from columns={column_names}")


def encode_labels(dataset: DatasetDict, label_column: str) -> tuple[DatasetDict, int, dict[str, int]]:
    labels = []
    for split in dataset.values():
        labels.extend(split[label_column])
    unique = sorted(set(labels))
    if all(isinstance(label, (int, np.integer)) for label in unique):
        label_map = {str(label): int(label) for label in unique}

        def map_int_label(row: dict[str, Any]) -> dict[str, int]:
            return {"labels": int(row[label_column])}

        return dataset.map(map_int_label), len(unique), label_map

    label_map = {str(label): idx for idx, label in enumerate(unique)}

    def map_str_label(row: dict[str, Any]) -> dict[str, int]:
        return {"labels": label_map[str(row[label_column])]}

    return dataset.map(map_str_label), len(label_map), label_map


def prepare_dataset(
    dataset_arg: str,
    tokenizer,
    max_length: int,
    max_train_samples: int | None,
    max_eval_samples: int | None,
) -> tuple[DatasetDict, int, dict[str, int]]:
    dataset = load_uit_vsfc_or_csv(dataset_arg)
    if "validation" not in dataset and "dev" in dataset:
        dataset["validation"] = dataset["dev"]
    if "validation" not in dataset and "test" in dataset:
        dataset["validation"] = dataset["test"]
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        dataset = DatasetDict(train=split["train"], validation=split["test"])

    text_column = infer_text_column(dataset["train"].column_names)
    label_column = infer_label_column(dataset["train"].column_names)
    dataset, num_labels, label_map = encode_labels(dataset, label_column)

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            max_length=max_length,
        )

    keep_columns = {"input_ids", "attention_mask", "labels"}
    tokenized = dataset.map(tokenize, batched=True)
    remove_columns = [col for col in tokenized["train"].column_names if col not in keep_columns]
    tokenized = tokenized.remove_columns(remove_columns)

    if max_train_samples is not None:
        tokenized["train"] = tokenized["train"].select(range(min(max_train_samples, len(tokenized["train"]))))
    if max_eval_samples is not None:
        tokenized["validation"] = tokenized["validation"].select(
            range(min(max_eval_samples, len(tokenized["validation"])))
        )
    return tokenized, num_labels, label_map


def freeze_all_but_classifier(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True


def configure_method(args: argparse.Namespace, model: torch.nn.Module) -> list[str]:
    if args.method == "ft":
        for param in model.parameters():
            param.requires_grad = True
        return []

    freeze_all_but_classifier(model)
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    return apply_peft(
        model=model,
        method=args.method,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=target_modules,
    )


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
    }


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return total / (1024 * 1024)


def save_trainable_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if any(name.startswith(param_name) for param_name, p in model.named_parameters() if p.requires_grad)
    }
    torch.save(trainable_state, path / "adapter_model.pt")


def append_results(results_file: Path, row: dict[str, Any]) -> None:
    results_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = results_file.exists()
    with results_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def main() -> None:
    require_runtime_dependencies()
    args = parse_args()
    set_seed(args.seed)
    if args.alpha is None and args.method in {"lora", "dora"}:
        args.alpha = 2 * args.rank
    if args.learning_rate is None:
        args.learning_rate = 2e-5 if args.method == "ft" else 2e-4

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    dataset, num_labels, label_map = prepare_dataset(
        args.dataset,
        tokenizer,
        args.max_length,
        args.max_train_samples,
        args.max_eval_samples,
    )
    model = build_phobert_classifier(args.model_name, num_labels=num_labels)
    replaced_modules = configure_method(args, model)
    counts = count_parameters(model)

    rank_part = f"_r{args.rank}" if args.method in {"lora", "dora"} else ""
    run_id = f"{args.method}{rank_part}_seed{args.seed}_{int(time.time())}"
    output_dir = Path(args.output_dir) / run_id
    save_strategy = "epoch" if args.method == "ft" else "no"
    load_best_model = args.method == "ft"
    training_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "save_strategy": save_strategy,
        "load_best_model_at_end": load_best_model,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": [],
        "seed": args.seed,
    }
    arg_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in arg_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": compute_metrics,
    }
    trainer_arg_names = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_arg_names:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    train_time = time.perf_counter() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    if args.method == "ft":
        trainer.save_model(str(output_dir / "checkpoint_final"))
    else:
        save_trainable_checkpoint(model, output_dir / "adapter_checkpoint")

    checkpoint_size_mb = directory_size_mb(output_dir)
    row = {
        "run_id": run_id,
        "method": args.method,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "rank": "" if args.method == "ft" else args.rank,
        "alpha": "" if args.method == "ft" else args.alpha,
        "dropout": "" if args.method == "ft" else args.dropout,
        "seed": args.seed,
        "accuracy": metrics.get("eval_accuracy"),
        "macro_f1": metrics.get("eval_macro_f1"),
        "weighted_f1": metrics.get("eval_weighted_f1"),
        "trainable_params": counts.trainable,
        "total_params": counts.total,
        "trainable_percent": counts.trainable_percent,
        "train_time_sec": train_time,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_size_mb": checkpoint_size_mb,
        "output_dir": str(output_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, ensure_ascii=False)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "num_labels": num_labels,
                "label_map": label_map,
                "replaced_modules": replaced_modules,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    append_results(Path(args.results_file), row)

    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
