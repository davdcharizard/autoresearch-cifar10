# Plan EXP-003: Weight Decay 5e-4 on Width-2x Augmented Baseline
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md

The hypothesis: changing WEIGHT_DECAY from 1e-4 to 5e-4 on the width-2x augmented baseline will raise best_test_acc from 92.92% to 93.1-93.5% by strengthening L2 regularization to match the WRN paper's recipe for wider CIFAR-10 models.

## Milestones

### Milestone 1: Code change implemented on experiment branch
- [ ] Create branch `autoresearch/exp-003` from `autoresearch/dev`
- [ ] Change `WEIGHT_DECAY = 1e-4` to `WEIGHT_DECAY = 5e-4` in train.py hyperparameters block
- [ ] Run ruff check and format check — confirm pass

### Milestone 2: Experiment runs to time-budget cap
- [ ] Confirm GPU 0 idle
- [ ] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [ ] Confirm exit code 0 and summary block present

### Milestone 3: Verification completes
- [ ] Run three verification conditions
- [ ] Record verdict and metrics

## Code Changes

**`train.py` line 24**: Change `WEIGHT_DECAY = 1e-4` to `WEIGHT_DECAY = 5e-4`. One constant. No other changes.

## Configuration Changes

- **WEIGHT_DECAY**: 1e-4 → 5e-4 (WRN paper's standard for wider CIFAR-10 models; 5x increase to strengthen L2 regularization on 1.07M-param model)

## Execution Environment

- **Method**: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- **Resources**: One NVIDIA H20 GPU, ~600 MB peak VRAM expected
- **Estimated runtime**: ~350s total (300s training + overhead)
- **Log output**: `run.log` at project root
- **Tool skill**: None

## Abort Criteria

- No output for 2 minutes → hang
- Python traceback → code error
- Wall-clock > 600s without summary → timeout
- Loss nan/inf → divergence

## Verification Protocol

### Verification Procedure

Baseline: 92.92%. Threshold: 93.02% (baseline + 0.1pp).

**Condition 1**: `best_test_acc > 93.02%`
**Condition 2**: Summary block complete (--- + 10 metric lines)
**Condition 3**: eval_count <= num_epochs

### Informational Metrics
- training_seconds, total_seconds, peak_vram_mb, num_epochs, num_steps, num_params
