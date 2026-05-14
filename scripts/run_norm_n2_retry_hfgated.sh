#!/bin/bash
# Re-run the (model, task) pairs that failed in the original batch because
# HF_TOKEN wasn't exported (llama2 / llama3.2 are gated on HF Hub).
# Same hyperparameters as scripts/run_norm_n2_batch.sh.

set -e
cd ~/stat4830

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN is not set. Export it first, e.g.:" >&2
  echo "  export HF_TOKEN=hf_xxx" >&2
  exit 1
fi

PAIRS=(
  "llama2-7b     trec"
  "llama3.2-1b   copa"
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

echo "[$(date)] === Retry pairs complete ==="
