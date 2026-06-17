# PEFT-from-scratch

Implement PEFT methods from scratch with PyTorch, focused on reproducing
DoRA-style fine-tuning for PhoBERT sentiment classification.

## What is implemented

- Full fine‑tuning baseline for `vinai/phobert-base-v2`.
- LoRA wrapper for `torch.nn.Linear`, without Hugging Face PEFT.
  - Merge/unmerge adapter, enable/disable adapter, save/load adapter weights.
- DoRA wrapper for `torch.nn.Linear`, using weight decomposition:
  `W = m * normalize(W0 + BA)`.
  - Column‑wise magnitude and direction decomposition.
  - Optional detached gradient (saves ~24% VRAM, accuracy drop ~0.2%).
  - Configurable magnitude initialisation (`weight_norm` or `ones`).
  - Merge to a plain linear layer, enable/disable, save/load adapter.
- Benchmark CLI for FT, LoRA and DoRA on UIT‑VSFC, prepared UIT‑ViON or a local CSV fallback.
- Notebook for result aggregation and plots.

## Install

```bash
pip install -r requirements.txt
```

## Train

**Full fine‑tuning**
```bash
python train.py --method ft --dataset uit-vsfc --seed 42
```

**LoRA**
```bash
python train.py --method lora --rank 8 --alpha 16 --dropout 0.05 --seed 42
```

**DoRA** (dropout is not supported, the script will automatically set it to 0)
```bash
python train.py --method dora --rank 8 --alpha 16 --seed 42
```

**DoRA with custom initialisation and detached gradient disabled**
```bash
python train.py --method dora --rank 8 --alpha 16 --init-magnitude ones --no-detach
```

**Smoke test** (small subset)
```bash
python train.py --method dora --rank 8 --alpha 16 --epochs 1 --max-train-samples 64 --max-eval-samples 64
```

Every run writes `metrics.json`, `config.json`, a checkpoint under `outputs/`,
and appends one row to `results/benchmark_results_2.csv`.

## DoRA‑specific options

| Flag | Default | Description |
|------|---------|-------------|
| `--init-magnitude` | `weight_norm` | How to initialise the magnitude vector. Choices: `weight_norm` (use the column‑wise norm of the pretrained weight) or `ones`. |
| `--no-detach` | `False` | If set, the column‑wise norm used in the forward pass will **not** be detached from the computational graph (i.e., full gradient). The default behaviour saves VRAM with negligible accuracy cost. |

## Local CSV fallback

If the public UIT‑VSFC dataset cannot be downloaded, pass a CSV path:

```bash
python train.py --method dora --dataset data/uit_vsfc.csv
```

The CSV must contain:

- `text`: input sentence
- `label`: sentiment label
- `split`: `train`, `validation` or `test`

## UIT-ViON preparation

UIT-ViON is not bundled in this repository. Download `data.zip` from the
UIT-ViON repository, then normalize it to the local CSV schema:

```bash
python scripts/prepare_uit_vion.py \
  --input path/to/data.zip \
  --output data/uit_vion/dataset.csv
```

For a quick stratified subset before full benchmarking:

```bash
python scripts/prepare_uit_vion.py \
  --input path/to/data.zip \
  --output data/uit_vion/subset.csv \
  --max-per-label 1000
```

Train on the prepared default file:

```bash
python train.py --method lora --dataset uit-vion --rank 8 --alpha 16 --seed 42
```

## Multi‑label extracted dataset

Build a multi‑label dataset from extracted weak labels:

```bash
python scripts/build_multilabel_dataset.py --input-dir extracted_data/extracted_labels_both --output-dir data/multilabel
```

This writes `dataset.jsonl`, split‑specific JSONL files, `label_map.json`, and
`summary.json`. Labels are preserved exactly as extracted.

Train PhoBERT with multi‑label BCE loss:

```bash
python train.py --method dora --dataset data/multilabel/dataset.jsonl --task-type multi_label --label-map data/multilabel/label_map.json --rank 8 --alpha 16
```

## Python API (minimal example)

```python
from transformers import AutoModel
from src.peft import apply_peft, save_adapters, load_adapters

model = AutoModel.from_pretrained("vinai/phobert-base-v2")

# Apply DoRA to specific modules
apply_peft(model, method="dora", rank=8, alpha=16, dropout=0.0,
           target_modules=["query", "value"],
           init_magnitude="weight_norm", use_detached_gradient=True)

# Train the model ...

# Save only the adapter weights (small checkpoint)
save_adapters(model, "checkpoints/dora_adapter.pt")

# Later, load the adapter into a fresh base model
base_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
apply_peft(base_model, method="dora", rank=8, alpha=16, dropout=0.0,
           target_modules=["query", "value"])
load_adapters(base_model, "checkpoints/dora_adapter.pt")

# Merge the adapter into the base weights (convert to plain nn.Linear)
for module in base_model.modules():
    if hasattr(module, "merge_adapter"):
        module.merge_adapter()
```

## Benchmark notebook

Open `notebooks/phobert_dora_benchmark.ipynb` to run the planned benchmark
commands, aggregate `results/benchmark_results_2.csv`, plot metrics and verify
that DoRA merge preserves logits.

## Tests

```bash
pytest
```
