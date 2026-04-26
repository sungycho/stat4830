#!/bin/bash
set -e
cd ~/stat4830

# Resume each of the 4 runs for 70 more iters (130+70=200 total → 12,800 train_fwd)

echo "[$(date)] === N=2 normalize=ON seed=43 resume (70 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 70 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --resume-from results/gsm8k_n2_norm_on_s43/latest.pt \
  --out-dir results/gsm8k_n2_norm_on_s43_r2

echo "[$(date)] === N=2 normalize=ON seed=44 resume (70 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 70 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --resume-from results/gsm8k_n2_norm_on_s44/latest.pt \
  --out-dir results/gsm8k_n2_norm_on_s44_r2

echo "[$(date)] === N=2 normalize=OFF seed=43 resume (70 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 70 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 43 --max-new-tokens 256 --early-stop-delta 0.0 \
  --no-normalize \
  --resume-from results/gsm8k_n2_norm_off_s43/latest.pt \
  --out-dir results/gsm8k_n2_norm_off_s43_r2

echo "[$(date)] === N=2 normalize=OFF seed=44 resume (70 iters) ==="
uv run python -m src.scripts.train_es \
  --task gsm8k --model Qwen/Qwen2.5-1.5B-Instruct \
  --population-size 2 --num-iters 70 --batch-size 16 --val-every 10 \
  --sigma 0.001 --lr 0.001 --train-size 128 --val-size 200 \
  --seed 44 --max-new-tokens 256 --early-stop-delta 0.0 \
  --no-normalize \
  --resume-from results/gsm8k_n2_norm_off_s44/latest.pt \
  --out-dir results/gsm8k_n2_norm_off_s44_r2

echo "[$(date)] === ALL DONE ==="
