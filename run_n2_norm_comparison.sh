#!/bin/bash
set -e
cd ~/stat4830

# N=2, normalization ON vs OFF, seed=43, matched to existing N=2 curve (~8320 train_fwd)

echo "[$(date)] === N=2 normalize=ON seed=43 (130 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 200 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n2_norm_on_s43

echo "[$(date)] === N=2 normalize=ON seed=44 (130 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 200 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n2_norm_on_s44

echo "[$(date)] === N=2 normalize=OFF seed=43 (130 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 200 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --no-normalize \
  --out-dir results/gsm8k_n2_norm_off_s43

echo "[$(date)] === N=2 normalize=OFF seed=44 (130 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 200 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --no-normalize \
  --out-dir results/gsm8k_n2_norm_off_s44

echo "[$(date)] === DONE ==="
