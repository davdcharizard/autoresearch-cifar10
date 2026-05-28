# Plan EXP-017: Mixup α=0.2 Replacing RandomErasing
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-017.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Remove `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` from the augmentation pipeline (line 131 of train.py)
- [ ] Add `MIXUP_ALPHA = 0.2` hyperparameter constant in the hyperparameters section
- [ ] Add batch-level mixup logic in the training loop: sample λ from Beta(α, α), permute batch indices, mix inputs
- [ ] Replace hard-target `F.cross_entropy(outputs, targets, label_smoothing=0.2)` with soft-target cross-entropy that incorporates both mixup label mixing and label smoothing
- [ ] Verify no syntax errors via `python -c "import ast; ast.parse(open('train.py').read())"`

### Milestone 2: Experiment run completes
- [ ] Run `uv run train.py > run.log 2>&1` and confirm output contains full summary block
- [ ] Confirm ~98 epochs complete within 300s budget (throughput unchanged)

### Milestone 3: Verification
- [ ] Extract `best_test_acc` from run.log and compare against 95.67% threshold
- [ ] Confirm full summary block present (best_test_acc, final_test_acc, training_seconds, etc.)

## Code Changes

- **train.py (hyperparameters section, ~line 25)**: Add `MIXUP_ALPHA = 0.2` constant.

- **train.py (augmentation pipeline, lines 124-133)**: Remove the `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` line. The remaining pipeline is: RandomCrop(32, padding=4) → RandomHorizontalFlip → TrivialAugmentWide → ToTensor → Normalize. This frees regularization budget for the mixup replacement.

- **train.py (training loop, lines 210-220)**: After moving inputs/targets to GPU, add mixup logic:
  1. Sample λ from Beta(MIXUP_ALPHA, MIXUP_ALPHA) — a single scalar per batch for simplicity
  2. Clamp λ = max(λ, 1-λ) so the first sample always dominates (standard mixup practice)
  3. Generate a random permutation of batch indices
  4. Mix inputs: `inputs = λ * inputs + (1-λ) * inputs[perm]`
  5. Convert integer targets to one-hot, apply label smoothing (0.2), then mix: `mixed_targets = λ * targets_smooth + (1-λ) * targets_smooth[perm]`

- **train.py (loss computation, ~line 220)**: Replace `F.cross_entropy(outputs, targets, label_smoothing=0.2)` with manual soft-target cross-entropy: `loss = -torch.sum(mixed_targets * F.log_softmax(outputs, dim=1)) / inputs.size(0)`. This correctly handles the mixed soft labels. The manual loss naturally incorporates label smoothing since it's baked into the mixed targets.

## Configuration Changes
- `MIXUP_ALPHA`: N/A → 0.2 (mild interpolation; α=0.2 produces λ values concentrated near 0 and 1, median ~0.85, providing regularization without the excessive interpolation that caused EXP-010's α=1.0 failure)
- RandomErasing: p=0.25, scale=(0.02, 0.2) → removed (replaced by mixup; removing instead of stacking avoids the over-regularization that caused EXP-010's failure)

## Execution Environment
- Method: local command `uv run train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~310-320s total (300s training + ~10-15s startup/eval)
- Log output: stdout/stderr captured to `run.log` in project root via shell redirection
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 60 seconds → likely crash or hang
- Loss becomes NaN/inf within first 5 epochs → mixup or loss computation bug
- Per-step time increases significantly (>20ms/step sustained) → unexpected overhead from mixup (unlikely — mixup is a few tensor ops)
- Training budget exhausted with fewer than 80 epochs → unexpected throughput regression

## Verification Protocol

### Verification Procedure

Baseline: 95.57% (EXP-015, commit 626e9d1). Improvement threshold: 95.67%.

**Condition 1: best_test_acc > baseline + 0.1pp = 95.67%**
```bash
grep "^best_test_acc:" run.log | awk '{print $2}'
```
Pass if the extracted value (as a float) is strictly > 95.67. Fail otherwise.

**Condition 2: Full summary block present**
```bash
grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:\|^num_steps:" run.log
```
Pass if count is 6 (all six summary fields present). Fail otherwise.

**Condition 3: Validation runs at most once per epoch**
```bash
grep -c "eval ep" run.log
```
Compare against the `num_epochs` value from the summary block. Pass if eval count ≤ num_epochs. Fail otherwise.

### Informational Metrics (Optional)
- `final_test_acc`: `grep "^final_test_acc:" run.log | awk '{print $2}'`
- `final_test_loss`: `grep "^final_test_loss:" run.log | awk '{print $2}'`
- `num_epochs`: `grep "^num_epochs:" run.log | awk '{print $2}'`
- `training_seconds`: `grep "^training_seconds:" run.log | awk '{print $2}'`
- `peak_vram_mb`: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
