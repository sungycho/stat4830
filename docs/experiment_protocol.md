# ES Experiment Protocol

Standard operating procedure for every new (model, task) combination.
Follow steps in order. Do not skip baseline check.

---

## Setup — sync-remote

One-time setup to sync local code to the remote GPU server without manual `scp`.

### 1. Add your remote host to SSH config

Edit `~/.ssh/config` on your local machine:

```
Host gpu
    HostName REMOTE_IP_OR_HOSTNAME
    User REMOTE_USER
    IdentityFile ~/.ssh/id_rsa   # or your key path
```

Test: `ssh gpu` should log in without a password prompt.

### 2. Define the sync-remote alias

Add to your local `~/.zshrc` (or `~/.bashrc`):

```bash
alias sync-remote='rsync -avz --exclude=".venv/" --exclude="__pycache__/" --exclude="*.pyc" --exclude=".git/" ~/project/stat4830/ gpu:~/project/stat4830/'
```

Then reload: `source ~/.zshrc`

### 3. Usage

```bash
sync-remote          # push local → remote (run after every code change)
```

To pull results back (JSON + PNG only, skip large .pt files):

```bash
rsync -avz --include="*/" --include="*.json" --include="*.png" --exclude="*" \
  gpu:~/project/stat4830/results/ ~/project/stat4830/results/
```

**Always delete `.pt` checkpoints on remote before pulling** (they are large and not needed locally):

```bash
ssh gpu "find ~/project/stat4830/results/ -name '*.pt' -delete"
```

---

## Naming Conventions

All results directories and plot titles must follow these formats:

- **Directory**: `results/{exp_type}/{TASK}_{MODEL}/` e.g. `results/calibration/trec_qwen3b/`
- **Model shorthand**: see Quick Reference table below
- **Plot title**: `{MODEL}_{TASK}_{exp_name} (N×T={budget})` e.g. `Qwen-3B_TREC_pop_scaling (N×T=1024)`

---

## Step 0 — Baseline Gate

Run base model on val set. **Only proceed if `0.1 < base_acc < 0.75`.**
- Below 0.1: model can't do the task at all → ES has nothing to build on
- Above 0.75: task too easy → ceiling too close, ES gain uninterpretable

```bash
uv run python -c "
from src.tasks import get_task
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

Then check base model accuracy:
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

---

## Step 1 — Hyperparameter Calibration (sigma × lr)

Start with the quick version. Only escalate to full if results are ambiguous (top-2 combos within 0.01 of each other, or best combo at the grid boundary).

### Quick (3×3 = 9 combos, ~5 min)

Coarse grid using powers-of-10 values. Enough to identify the right order of magnitude.

```bash
uv run python -m src.scripts.grid_search \
  --config configs/hparam_calibration.json \
  --task TASK \
  --model MODEL \
  --device cuda \
  --out-dir results/calibration/TASK_MODEL/quick
```

`configs/hparam_calibration.json` sweeps sigma ∈ {1e-4, 1e-3, 1e-2} × lr ∈ {1e-5, 1e-4, 1e-3} with N=8, T=5, val_size=50, n_seeds=1.

### Full (4×4 = 16 combos, ~30 min)

Finer grid adding intermediate values (3×10ⁿ). Use when quick results are ambiguous or best combo lands at grid edge.

```bash
uv run python -m src.scripts.run_experiment \
  --block calibration \
  --task TASK \
  --model MODEL \
  --max-new-tokens MAX_TOK \
  --val-size 200 \
  --n-seeds 3 \
  --device cuda \
  --out-dir results/calibration/TASK_MODEL/full
```

Default calibration block sweeps sigma ∈ {3e-4, 1e-3, 3e-3, 1e-2} × lr ∈ {1e-4, 3e-4, 1e-3, 3e-3} with N=8, T=20, val_size=200, n_seeds=3.

Ranked summary printed at end and saved to `summary.json`.

**Pick**: highest `mean_best_val`. If tie, prefer smaller sigma (more stable).

**Rule of thumb priors**:
- sigma: `1e-3` usually works; go lower (`1e-4`) for larger models
- lr: `1e-4` conservative safe default
- If best combo is at a grid boundary → run full calibration with extended range

---

## Step 2 — Population Size Sweep (fixed forward-pass budget)

Uses the `pop_scaling` block: N ∈ {1, 4, 8, 16, 32} at fixed budget N×T=const.

```bash
uv run python -m src.scripts.run_experiment \
  --block pop_scaling \
  --task TASK \
  --model MODEL \
  --max-new-tokens MAX_TOK \
  --val-size 200 \
  --n-seeds 1 \
  --device cuda \
  --best-sigma BEST_SIGMA \
  --best-lr BEST_LR \
  --out-dir results/pop_scaling/TASK_MODEL
```

**What to look for**:
- Inverted-U in val_acc vs N → confirms N* exists
- N=1,2 collapse (normalization instability)
- Large N (32+) flattening

---

## Step 3 — Top-K Sweep (at best N from Step 2)

Uses the `top_k` block: K ∈ {0, 4, 8} at N=16.

```bash
uv run python -m src.scripts.run_experiment \
  --block top_k \
  --task TASK \
  --model MODEL \
  --max-new-tokens MAX_TOK \
  --val-size 200 \
  --n-seeds 1 \
  --device cuda \
  --best-sigma BEST_SIGMA \
  --best-lr BEST_LR \
  --out-dir results/top_k/TASK_MODEL
```

---

## Step 4 — Transfer Results and Plot

### 4a — Delete .pt checkpoints on remote (save space)

```bash
find results/ -name "*.pt" -delete
```

### 4b — Rsync JSON and PNG files to local machine

Run this **on your local machine**:
```bash
rsync -avz --include="*/" --include="*.json" --include="*.png" --exclude="*" \
  gpu:~/project/stat4830/results/TASK_MODEL/ \
  ~/project/stat4830/results/TASK_MODEL/
```

### 4c — Plot calibration heatmap

```bash
uv run python -m src.scripts.plot_results \
  --exp-dir results/calibration/TASK_MODEL \
  --title "MODEL_SHORT_TASK_calibration"
```

### 4d — Plot population scaling curves

```bash
uv run python -m src.scripts.plot_pop_scaling_curves \
  --exp-dir results/pop_scaling/TASK_MODEL \
  --title "MODEL_SHORT_TASK_pop_scaling (N×T=BUDGET)"
```

**Plot title format**: always `{MODEL_SHORT}_{TASK}_{exp_name} (N×T={budget})`.

---

## Quick Reference — tasks

| Task name | HuggingFace ID                    | Task type           | Labels / output                         | MAX_TOK |
|-----------|-----------------------------------|---------------------|-----------------------------------------|---------|
| sst2      | nyu-mll/glue (sst2)               | sentiment           | positive / negative                     | 5       |
| boolq     | aps/super_glue (boolq)            | reading comp. Q&A   | yes / no                                | 8       |
| rte       | aps/super_glue (rte)              | NLI (2-class)       | yes / no                                | 8       |
| cb        | aps/super_glue (cb)               | NLI (3-class)       | entailment / neutral / contradiction    | 8       |
| wsc       | aps/super_glue (wsc)              | coreference         | yes / no                                | 8       |
| wic       | aps/super_glue (wic)              | word sense          | yes / no                                | 8       |
| copa      | aps/super_glue (copa)             | causal reasoning    | choice 1 / choice 2                     | 8       |
| mnli      | nyu-mll/glue (mnli)               | NLI (3-class)       | entailment / neutral / contradiction    | 8       |
| squad     | rajpurkar/squad                   | extractive Q&A      | free-form span                          | 32      |
| drop      | ucinlp/drop                       | discrete reasoning  | free-form number/span                   | 32      |
| record    | aps/super_glue (record)           | reading comp.       | entity span                             | 32      |
| gsm8k     | openai/gsm8k                      | math word problem   | integer answer                          | 32      |
| math500   | hendrycks/competition_mathematics | competition math    | expression / number                     | 64      |
| trec      | CogComp/trec                      | question type       | 6-class label                           | 5       |
| sst5      | sst (default)                     | fine sentiment      | 5-class label                           | 5       |
| countdown | (custom)                          | arithmetic          | equation steps                          | 64      |

## Quick Reference — models

### OPT family

| Shorthand  | HuggingFace ID       | Plot label | Params |
|------------|----------------------|------------|--------|
| opt350m    | facebook/opt-350m    | OPT-350M   | 350M   |
| opt1b      | facebook/opt-1.3b    | OPT-1.3B   | 1.3B   |
| opt2b      | facebook/opt-2.7b    | OPT-2.7B   | 2.7B   |
| opt13b     | facebook/opt-13b     | OPT-13B    | 13B    |
| opt66b     | facebook/opt-66b     | OPT-66B    | 66B    |

### Llama family

| Shorthand    | HuggingFace ID                   | Plot label    | Params |
|--------------|----------------------------------|---------------|--------|
| llama2-7b    | meta-llama/Llama-2-7b-hf         | Llama-2-7B    | 7B     |
| llama2-13b   | meta-llama/Llama-2-13b-hf        | Llama-2-13B   | 13B    |
| llama2-70b   | meta-llama/Llama-2-70b-hf        | Llama-2-70B   | 70B    |
| llama3-8b    | meta-llama/Meta-Llama-3-8B       | Llama-3-8B    | 8B     |
| llama3.1-70b | meta-llama/Llama-3.1-70B         | Llama-3.1-70B | 70B    |

### Encoder / small models

| Shorthand  | HuggingFace ID                      | Plot label    | Params |
|------------|-------------------------------------|---------------|--------|
| roberta-b  | FacebookAI/roberta-base             | RoBERTa-base  | 125M   |
| roberta-l  | FacebookAI/roberta-large            | RoBERTa-large | 355M   |
| distilbert | distilbert/distilbert-base-uncased  | DistilBERT    | 66M    |
| gpt2-xl    | openai-community/gpt2-xl            | GPT-2-XL      | 1.5B   |
| phi2       | microsoft/phi-2                     | Phi-2         | 2.7B   |

### Qwen2.5 family

| Shorthand     | HuggingFace ID                          | Plot label        | Params |
|---------------|-----------------------------------------|-------------------|--------|
| qwen1.5b      | Qwen/Qwen2.5-1.5B-Instruct              | Qwen2.5-1.5B      | 1.5B   |
| qwen3b        | Qwen/Qwen2.5-3B-Instruct                | Qwen2.5-3B        | 3B     |
| qwen7b        | Qwen/Qwen2.5-7B-Instruct                | Qwen2.5-7B        | 7B     |
| qwen1.5b-math | Qwen/Qwen2.5-Math-1.5B-Instruct         | Qwen2.5-Math-1.5B | 1.5B   |
| qwen7b-math   | Qwen/Qwen2.5-Math-7B-Instruct           | Qwen2.5-Math-7B   | 7B     |

---

## Checklist per run

- [ ] Baseline gate passed (0.1 < base_acc < 0.75)
- [ ] Label distribution checked (no severe class imbalance → majority-class hacking risk)
- [ ] Quick calibration (3×3) done; escalate to full (4×4) if results ambiguous
- [ ] Best sigma/lr recorded
- [ ] Results saved under `results/{exp_type}/TASK_MODEL/`
- [ ] Pop scaling sweep run at fixed forward-pass budget
- [ ] Top-K sweep run at best N
- [ ] `.pt` files deleted on remote before transfer
- [ ] JSON + PNG files rsynced to local
- [ ] All plots titled `MODEL_SHORT_TASK_exp_name (N×T=budget)`
