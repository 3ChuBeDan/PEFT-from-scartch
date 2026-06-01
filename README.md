# PEFT-from-scartch

Implement PEFT methods from scratch with PyTorch, focused on reproducing
DoRA-style fine-tuning for PhoBERT sentiment classification.

## What is implemented

- Full fine-tuning baseline for `vinai/phobert-base-v2`.
- LoRA wrapper for `torch.nn.Linear`, without Hugging Face PEFT.
- DoRA wrapper for `torch.nn.Linear`, using weight decomposition:
  `W = m * normalize(W0 + BA)`.
- Benchmark CLI for FT, LoRA and DoRA on UIT-VSFC or a local CSV fallback.
- Notebook for result aggregation and plots.

## Install

```bash
pip install -r requirements.txt
```

## Train

Full fine-tuning:

```bash
python train.py --method ft --dataset uit-vsfc --seed 42
```

LoRA:

```bash
python train.py --method lora --rank 8 --alpha 16 --dropout 0.05 --seed 42
```

DoRA:

```bash
python train.py --method dora --rank 8 --alpha 16 --dropout 0.05 --seed 42
```

Smoke test with a small subset:

```bash
python train.py --method dora --rank 8 --alpha 16 --epochs 1 --max-train-samples 64 --max-eval-samples 64
```

Every run writes `metrics.json`, `config.json`, a checkpoint under `outputs/`,
and appends one row to `results/benchmark_results.csv`.

## Local CSV fallback

If the public UIT-VSFC dataset cannot be downloaded, pass a CSV path:

```bash
python train.py --method dora --dataset data/uit_vsfc.csv
```

The CSV must contain:

- `text`: input sentence
- `label`: sentiment label
- `split`: `train`, `validation` or `test`

## Benchmark notebook

Open `notebooks/phobert_dora_benchmark.ipynb` to run the planned benchmark
commands, aggregate `results/benchmark_results.csv`, plot metrics and verify
that DoRA merge preserves logits.

## Tests

```bash
pytest
```
