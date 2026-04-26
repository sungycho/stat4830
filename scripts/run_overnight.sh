#!/bin/bash
set -e
cd /workspace/stat4830

echo "[$(date)] === PHASE 1: RESUME TO 15,360 TRAIN FWD ==="

echo "[$(date)] N=8 resume (30 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 8 --num-iters 30 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 42 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n8_resume \
  --resume-from results/gsm8k_n8_20260413_021840/latest.pt

echo "[$(date)] N=4 resume (60 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 4 --num-iters 60 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 42 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n4_resume \
  --resume-from results/gsm8k_n4_20260414_025348/latest.pt

echo "[$(date)] N=2 resume (110 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 110 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 42 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n2_resume \
  --resume-from results/gsm8k_n2_20260414_010958/latest.pt

echo "[$(date)] N=1 resume (320 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 1 --num-iters 320 --batch-size 16 --val-every 20 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 42 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n1_resume \
  --resume-from results/gsm8k_n1_20260414_001549/latest.pt

echo "[$(date)] === PHASE 2: SEED 2 (seed=43) ==="

echo "[$(date)] Seed 2 N=16 (30 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 16 --num-iters 30 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n16_s2

echo "[$(date)] Seed 2 N=8 (60 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 8 --num-iters 60 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n8_s2

echo "[$(date)] Seed 2 N=4 (120 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 4 --num-iters 120 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n4_s2

echo "[$(date)] Seed 2 N=2 (240 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 240 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n2_s2

echo "[$(date)] Seed 2 N=1 (480 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 1 --num-iters 480 --batch-size 16 --val-every 20 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n1_s2

echo "[$(date)] === PHASE 3: SEED 3 (seed=44) ==="

echo "[$(date)] Seed 3 N=16 (30 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 16 --num-iters 30 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n16_s3

echo "[$(date)] Seed 3 N=8 (60 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 8 --num-iters 60 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n8_s3

echo "[$(date)] Seed 3 N=4 (120 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 4 --num-iters 120 --batch-size 16 --val-every 5 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n4_s3

echo "[$(date)] Seed 3 N=2 (240 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 240 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n2_s3

echo "[$(date)] Seed 3 N=1 (480 iters)..."
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 1 --num-iters 480 --batch-size 16 --val-every 20 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --out-dir results/gsm8k_n1_s3

echo "[$(date)] === ALL DONE ==="
