# ES Experiment Protocol

Standard operating procedure for every new (model, task) combination.
Follow steps in order. Do not skip baseline check.

---

## Step 0 — Baseline Gate

Run base model on val set. **Only proceed if baseline accuracy is below 0.75.**
If the model already does well, ES has nothing to improve and results are uninterpretable.

```bash
uv run python -c "
from src.tasks import get_task
from src.backends.factory import create_backend
from src.utils.seeds import set_seeds
set_seeds(42)
task = get_task('TASK')
_, val = task.load_data(1, 200, 42)
from collections import Counter
dist = Counter(e['label'] for e in val)
print('Label dist:', dict(dist))
majority = max(dist.values()) / sum(dist.values())
print(f'Majority-class baseline: {majority:.3f}')
"
```

Then run one forward pass to check accuracy:
```bash
uv run python -m src.scripts.run_temperature_sweep \
  --model MODEL \
  --task TASK \
  --val-size 200 \
  --max-new-tokens MAX_TOK \
  --temperatures 1.0 \
  --k-values 1 \
  --batch-size 16 \
  --out results/baseline/TASK_MODEL
```

**Gate**: if `majority=0.0` at K=1 (i.e. correct_rate ≈ 0), model can't do the task at all → skip ES, too hard.
**Gate**: if correct_rate > 0.75, task is too easy → skip ES, ceiling too close.

---

## Step 1 — Hyperparameter Sweep (sigma × lr)

Find the best (sigma, lr) at small N and T to avoid burning GPU budget.

```bash
for SIGMA in 1e-4 1e-3 1e-2; do
  for LR in 1e-5 1e-4 1e-3; do
    uv run python -m src.scripts.train_es \
      --task TASK \
      --model MODEL \
      --population-size 8 \
      --num-iters 10 \
      --train-size 64 \
      --val-size 200 \
      --batch-size 16 \
      --max-new-tokens MAX_TOK \
      --sigma $SIGMA \
      --lr $LR \
      --val-every 2 \
      --seed 42 \
      --out-dir results/hparam_sweep/TASK_MODEL/s${SIGMA}_lr${LR}
  done
done
```

**Pick**: the (sigma, lr) pair with highest final val_acc. If multiple tie, prefer smaller sigma (more stable).

**Rule of thumb priors** (from our runs so far):
- sigma: `1e-3` usually works; go lower (`1e-4`) for larger models
- lr: `1e-4` is conservative safe default

---

## Step 2 — Population Size Sweep (fixed forward-pass budget)

Fix total forward passes = **N × T = 1024** (adjust if GPU is slow).
Sweep N ∈ {2, 4, 8, 16, 32, 64}, set T = 1024/N accordingly.

```bash
for N in 2 4 8 16 32 64; do
  T=$((1024 / N))
  uv run python -m src.scripts.train_es \
    --task TASK \
    --model MODEL \
    --population-size $N \
    --num-iters $T \
    --train-size 64 \
    --val-size 200 \
    --batch-size 16 \
    --max-new-tokens MAX_TOK \
    --sigma BEST_SIGMA \
    --lr BEST_LR \
    --val-every $((T / 5)) \
    --seed 42 \
    --out-dir results/pop_sweep/TASK_MODEL/N${N}_T${T}
done
```

**What to look for**:
- Peak val_acc as a function of N at fixed budget
- Whether N=2 collapses (normalization instability)
- Whether large N (32, 64) also collapses or flattens

---

## Step 3 — Top-K Sweep (at best N from Step 2)

```bash
for K in 0 2 4 8; do
  uv run python -m src.scripts.train_es \
    --task TASK \
    --model MODEL \
    --population-size BEST_N \
    --num-iters 50 \
    --train-size 64 \
    --val-size 200 \
    --batch-size 16 \
    --max-new-tokens MAX_TOK \
    --sigma BEST_SIGMA \
    --lr BEST_LR \
    --top-k $K \
    --val-every 5 \
    --seed 42 \
    --out-dir results/topk_sweep/TASK_MODEL/K${K}
done
```

K=0 means use all N (no filtering). K=2 keeps only the top 2 by |advantage|.

---

## Step 4 — Full Run with Decomposition Tracking

Only run if Steps 1-3 show non-trivial improvement over baseline.

```bash
# First get base model categories
uv run python -m src.scripts.analyze_failures \
  --model MODEL \
  --val-size 200 \
  --max-new-tokens MAX_TOK \
  --out results/decomp/TASK_MODEL/base.json

# Full ES run with decomposition tracking
uv run python -m src.scripts.train_es \
  --task TASK \
  --model MODEL \
  --population-size BEST_N \
  --num-iters 100 \
  --train-size 64 \
  --val-size 200 \
  --batch-size 16 \
  --max-new-tokens MAX_TOK \
  --sigma BEST_SIGMA \
  --lr BEST_LR \
  --top-k BEST_K \
  --val-every 5 \
  --track-decomposition \
  --base-json results/decomp/TASK_MODEL/base.json \
  --seed 42 \
  --out-dir results/full_run/TASK_MODEL
```

---

## Quick Reference — MAX_TOK by task

| Task      | MAX_TOK | Notes                        |
|-----------|---------|------------------------------|
| boolq     | 8       | yes/no answer                |
| rte       | 8       | yes/no answer                |
| sst2      | 5       | positive/negative            |
| sst5      | 5       | very negative … very positive|
| trec      | 5       | single category word         |
| countdown | 64      | arithmetic steps             |

## Quick Reference — model shorthand

| Shorthand  | Full model ID                   |
|------------|---------------------------------|
| opt350m    | facebook/opt-350m               |
| opt1b      | facebook/opt-1.3b               |
| qwen3b     | Qwen/Qwen2.5-3B-Instruct        |

---

## Checklist per run

- [ ] Baseline gate passed (0.25 < base_acc < 0.75)
- [ ] Label distribution checked (no severe class imbalance → majority-class hacking risk)
- [ ] Hyperparameter sweep done
- [ ] Results saved to `results/` with descriptive subdir name
- [ ] N sweep run at fixed forward-pass budget
- [ ] Top-K sweep run at best N
- [ ] Decomposition tracking enabled on final run
