#!/usr/bin/env bash
# run_rho_probe_malignant.sh
#
# Runs the degeneracy probe on two malignant-regime (model, task) pairs:
#   1. LLaMA-3.2-1B-Instruct  / MNLI
#   2. Qwen2.5-1.5B-Instruct  / WIC
#
# Theory predicts: incoherent failure mode (reasoning errors, each example
# fails differently) → rho close to 1 → P(A=0) close to 1.
#
# Results are saved to results/degen_probe/ and compared at the end.
# Usage: bash run_rho_probe_malignant.sh [--K 200] [--batch-size 16]

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via CLI flags)
# ---------------------------------------------------------------------------
K=200
BATCH_SIZE=16
SIGMA=0.001
MAX_NEW_TOKENS=32      # base LLaMA may prepend preamble before label word; 4 causes parse failures
NUM_WORKERS=4
PROBE_SIZE=200
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --K)           K="$2";          shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --sigma)       SIGMA="$2";      shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

OUT_DIR="results/degen_probe"
mkdir -p "$OUT_DIR"

OUT_LLAMA="$OUT_DIR/rho_malignant_llama1b_mnli.json"
OUT_QWEN="$OUT_DIR/rho_malignant_qwen15b_wic.json"

echo "============================================================"
echo "  ρ-probe: malignant regime comparison"
echo "  K=$K  B=$BATCH_SIZE  sigma=$SIGMA  seed=$SEED"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Run 1: LLaMA-3.2-1B-Instruct / MNLI
# ---------------------------------------------------------------------------
echo ">>> [1/2] LLaMA-3.2-1B on MNLI"
echo "    Expected: high rho (reasoning NLI; 1B model struggles per-example)"
echo ""

uv run python -m src.scripts.probe_degeneracy \
    --task         mnli \
    --model        meta-llama/Llama-3.2-1B \
    --sigma        "$SIGMA" \
    --K            "$K" \
    --batch-size   "$BATCH_SIZE" \
    --probe-size   "$PROBE_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-workers  "$NUM_WORKERS" \
    --seed         "$SEED" \
    --output       "$OUT_LLAMA"

echo ""
echo ">>> [1/2] Done. Results: $OUT_LLAMA"
echo ""

# ---------------------------------------------------------------------------
# Run 2: Qwen2.5-1.5B-Instruct / WIC
# ---------------------------------------------------------------------------
echo ">>> [2/2] Qwen2.5-1.5B-Instruct on WIC"
echo "    Expected: high rho (subtle semantic disambiguation; per-example failure)"
echo ""

uv run python -m src.scripts.probe_degeneracy \
    --task         wic \
    --model        Qwen/Qwen2.5-1.5B-Instruct \
    --sigma        "$SIGMA" \
    --K            "$K" \
    --batch-size   "$BATCH_SIZE" \
    --probe-size   "$PROBE_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-workers  "$NUM_WORKERS" \
    --seed         "$SEED" \
    --output       "$OUT_QWEN"

echo ""
echo ">>> [2/2] Done. Results: $OUT_QWEN"
echo ""

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Comparing results..."
echo "============================================================"
echo ""

uv run python -m src.scripts.compare_rho_probes \
    --file-a  "$OUT_LLAMA"  --label-a "LLaMA-1B / MNLI" \
    --file-b  "$OUT_QWEN"   --label-b "Qwen-1.5B / WIC"
