# Plan EXP-002: TrivialAugmentWide + RandomErasing on Width-2x Baseline
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md

The hypothesis (from brainstorm-002 § Chosen Idea): adding TrivialAugmentWide and RandomErasing to the width-2x ResNet-20's training transforms will raise best_test_acc from 92.29% to 92.8-93.5% by increasing effective training-set diversity and closing the generalization gap that the wider model exposes. The improvement bar is >= 92.39% (baseline 92.29% + 0.1pp).

## Milestones

### Milestone 1: Code changes implemented on experiment branch

- [ ] Create experiment branch `autoresearch/exp-002` from `autoresearch/dev`
- [ ] Add `transforms.TrivialAugmentWide()` to `train_tf` between `RandomHorizontalFlip()` and `ToTensor()`
- [ ] Add `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` to `train_tf` after `Normalize()`
- [ ] Run `uv run ruff check train.py && uv run ruff format --check train.py` — confirm exit code 0

### Milestone 2: Experiment runs to the time-budget cap

- [ ] Confirm GPU 0 is idle via `nvidia-smi`
- [ ] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm process completes with exit code 0 and summary block (`---`) present in run.log

### Milestone 3: Verification protocol completes with a definitive verdict

- [ ] Run the three verification conditions in order (short-circuit on first failure)
- [ ] Record verdict and informational metrics

## Code Changes

**`train.py` lines 124-131 (train_tf transforms pipeline)**: Add two transforms to the existing `transforms.Compose` list. `TrivialAugmentWide()` goes after `RandomHorizontalFlip()` and before `ToTensor()` because it operates on PIL images. `RandomErasing(p=0.25, scale=(0.02, 0.2))` goes after `Normalize()` because it operates on tensors. The resulting pipeline order is:

```python
train_tf = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ]
)
```

No other changes to train.py. The architecture (WIDTH_MULT=2), schedule (wall-clock-fractional LambdaLR), optimizer (SGD lr=0.1, momentum=0.9, WD=1e-4), batch size (128), seed (42), and all other settings remain at EXP-001 values.

## Configuration Changes

- **Train augmentation pipeline**: `[RandomCrop, RandomHorizontalFlip, ToTensor, Normalize]` → `[RandomCrop, RandomHorizontalFlip, TrivialAugmentWide, ToTensor, Normalize, RandomErasing]`
  - TrivialAugmentWide: zero-hyperparameter augmentation, picks one of 14 PIL-level operations per image with random magnitude. Available in torchvision 0.13+ (project pins 0.24.1).
  - RandomErasing(p=0.25, scale=(0.02, 0.2)): erases a random rectangular region with 25% probability per image, area fraction in [2%, 20%]. Standard Cutout-equivalent for tensor-space.

No hyperparameter changes. No new dependencies.

## Execution Environment

- **Method**: Local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- **Resources**: One NVIDIA H20 GPU (98 GB VRAM). Expected peak VRAM ~600 MB (unchanged from EXP-001 — augmentation adds no GPU memory overhead).
- **Estimated runtime**: ~340-400s total (300s training + eval overhead + startup). Per-step time may increase slightly from PIL-side augmentation ops (~12-13ms vs EXP-001's 11ms), yielding ~60-65 epochs in the budget.
- **Log output**: `run.log` at project root, capturing all stdout/stderr. TASK.md: "do NOT use tee."
- **Tool skill**: None (local run).

## Abort Criteria

- No output in `run.log` for 2 minutes after launch → hang (kill process)
- Python traceback in `run.log` → code error (read traceback, fix, retry)
- Total wall-clock > 600s without summary block → timeout (kill process)
- Loss values go to `nan` or `inf` → numerical divergence (kill process)
- LR transitions not observed at pct_done ~50% and ~75% → schedule misconfiguration (post-hoc check)

## Verification Protocol

### Verification Procedure

Baseline: 92.29% (from experiment-indices/maximize-cifar10-test-accuracy.tsv). Improvement threshold: 92.39% (baseline + 0.1pp).

**Condition 1 — best_test_acc > 92.39%**:
```bash
best_acc=$(grep "^best_test_acc:" run.log | head -n 1 | awk '{print $2}' | tr -d '%')
awk "BEGIN {exit ($best_acc > 92.39) ? 0 : 1}"
```

**Condition 2 — Summary block complete (script finished without crash)**:
```bash
grep -c "^---$" run.log  # must be 1
# all 10 metric lines present
for f in best_test_acc final_test_acc final_test_loss training_seconds total_seconds startup_seconds peak_vram_mb num_epochs num_steps num_params; do grep -c "^${f}:" run.log; done
```

**Condition 3 — Validation runs at most once per epoch**:
```bash
eval_count=$(grep -c "eval ep" run.log)
num_epochs=$(grep "^num_epochs:" run.log | awk '{print $2}')
[ "$eval_count" -le "$num_epochs" ]
```

Short-circuit: on first FAIL, remaining conditions are skipped.

### Informational Metrics (Optional)

- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- total_seconds: `grep "^total_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- num_steps: `grep "^num_steps:" run.log | awk '{print $2}'`
- num_params: `grep "^num_params:" run.log | awk '{print $2}'` (should be identical to EXP-001: 1,073,962)
