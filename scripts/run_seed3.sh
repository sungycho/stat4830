#!/bin/bash
set -e
cd ~/stat4830

echo "[$(date)] === SEED 3 (seed=44) ==="

echo "[$(date)] N=16 (30 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 16 --num-iters 30 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n16_s3

echo "[$(date)] N=8 (60 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 8 --num-iters 60 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n8_s3

echo "[$(date)] N=4 (120 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 4 --num-iters 120 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n4_s3

echo "[$(date)] N=2 (240 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 240 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n2_s3

echo "[$(date)] N=1 (480 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 1 --num-iters 480 --batch-size 16 --val-every 20 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n1_s3

echo "[$(date)] === SEED 3 DONE ==="
