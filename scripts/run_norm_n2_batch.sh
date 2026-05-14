#!/bin/bash
# N=2 normalize ON vs OFF sweep across (model, task) pairs.
# Hyperparameters:
#   pop=2 batch=16 iters=500 train=128 val=200 val_every=5
#   sigma=1e-3 lr_on=lr_off=1e-4 reward=accuracy seeds=42,43,44
# max_new_tokens auto-resolved per task by run_norm_n2.py registry.

set -e
cd ~/stat4830

PAIRS=(
  "opt-1.3b               cb"
  "qwen2.5-3b-instruct    mnli"
  "llama2-7b              trec"
  "opt-1.3b               trec"
  "llama3.2-1b            copa"
  "opt-1.3b               wic"
  "qwen2.5-3b-instruct    wic"
)

for pair in "${PAIRS[@]}"; do
  read -r model task <<<"$pair"
  echo "[$(date)] === $model × $task ==="
  uv run python -m src.scripts.adhoc.run_norm_n2 \
    --model "$model" \
    --task  "$task" \
    --seeds 42 43 44 \
    --pop 2 \
    --batch-size 16 \
    --num-iters 500 \
    --train-size 128 \
    --val-size 200 \
    --val-every 5 \
    --sigma 1e-3 \
    --lr-on 1e-4 \
    --lr-off 1e-4 \
    --reward accuracy
done

echo "[$(date)] === All pairs complete ==="
