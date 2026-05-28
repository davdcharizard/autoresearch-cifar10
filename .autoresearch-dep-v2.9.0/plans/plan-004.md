# Plan EXP-004: Nesterov Momentum + Label Smoothing 0.1
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md

Hypothesis: adding nesterov=True to SGD and label_smoothing=0.1 to cross_entropy will raise best_test_acc from 93.33% to 93.5-93.8%. Threshold: >= 93.43%.

## Milestones

### Milestone 1: Code changes implemented
- [ ] Create branch `autoresearch/exp-004` from `autoresearch/dev`
- [ ] Add `nesterov=True` to SGD optimizer call
- [ ] Add `label_smoothing=0.1` to F.cross_entropy call
- [ ] Ruff check pass

### Milestone 2: Experiment runs to completion
- [ ] GPU 0 idle, launch, confirm exit code 0

### Milestone 3: Verification
- [ ] Three conditions pass

## Code Changes

**`train.py` line 149-151 (SGD constructor)**: Add `nesterov=True` keyword argument.

**`train.py` line 203 (cross_entropy call)**: Change `F.cross_entropy(outputs, targets)` to `F.cross_entropy(outputs, targets, label_smoothing=0.1)`.

## Configuration Changes
- **Nesterov**: False → True (WRN paper standard)
- **Label smoothing**: 0 → 0.1 (Inception-v3 standard, regularization)

## Execution Environment
- **Method**: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- **Resources**: One H20 GPU, ~600 MB VRAM
- **Estimated runtime**: ~355s
- **Log output**: run.log at project root

## Abort Criteria
- No output 2min → hang; traceback → code error; >600s → timeout; nan/inf → divergence

## Verification Protocol

### Verification Procedure
Baseline: 93.33%. Threshold: 93.43%.
- Condition 1: best_test_acc > 93.43%
- Condition 2: Summary block complete
- Condition 3: eval_count <= num_epochs

### Informational Metrics
- training_seconds, total_seconds, peak_vram_mb, num_epochs, num_steps, num_params
