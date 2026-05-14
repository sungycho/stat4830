#!/bin/bash
# N=2 normalize ON vs OFF — math500 generation task.
# Same hyperparameters as run_norm_n2_batch.sh:
#   pop=2 batch=16 iters=500 train=128 val=200 val_every=5
#   sigma=1e-3 lr_on=lr_off=1e-4 reward=accuracy seeds=42,43,44
# max_new_tokens=256 (auto-resolved by registry for math500).
# WARNING: 256-token generations × N=2 antithetic × batch=16 are heavy —
# expect ~hours per run, especially for qwen2.5-7b-instruct.

set -e
cd ~/stat4830

PAIRS=(
  "opt-2.7b              math500"
  "qwen2.5-7b-instruct   math500"
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

echo "[$(date)] === math500 pairs complete ==="
