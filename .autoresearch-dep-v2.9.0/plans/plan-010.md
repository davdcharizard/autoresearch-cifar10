# Plan EXP-010: CutMix Batch Augmentation
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-010.md

## Milestones

### Milestone 1: CutMix implementation in train.py
- [x] Add `import numpy as np` (needed for `np.random.beta`)
- [x] Add `CUTMIX_ALPHA = 1.0` hyperparameter constant
- [x] Add `rand_bbox` helper function that computes a random bounding box given image dimensions and λ
- [x] Insert CutMix logic in the training loop: after `inputs`/`targets` are moved to GPU, before `optimizer.zero_grad()` — generate λ from Beta(α, α), compute bbox, shuffle indices, blend images, store adjusted λ
- [x] Replace `loss = F.cross_entropy(outputs, targets)` with mixed-label loss: `loss = lam * F.cross_entropy(outputs, targets_a) + (1 - lam) * F.cross_entropy(outputs, targets_b)`
- [x] Syntax check: `uv run python -c "import train"`

### Milestone 2: Run experiment
- [x] Execute `uv run train.py > run.log 2>&1` locally
- [x] Confirm log file is being written and training is progressing
- [x] Wait for completion (~5-7 minutes total including startup)

### Milestone 3: Verification
- [x] Check `best_test_acc` from run.log against threshold 95.49%
- [x] Verify summary block completeness (10 fields)
- [x] Verify eval count <= num_epochs

## Code Changes
- **train.py**: Add CutMix batch augmentation. Specifically:
  1. Add `import numpy as np` at the top (numpy is already a transitive dependency via torchvision).
  2. Add `CUTMIX_ALPHA = 1.0` in the hyperparameters section.
  3. Add a `rand_bbox(size, lam)` function that takes the batch tensor shape `(B, C, H, W)` and λ, computes the cut ratio as `sqrt(1 - lam)`, draws a random center point, clips the box to image bounds, and returns `(bbx1, bby1, bbx2, bby2)`.
  4. In the training loop, between `targets = targets.to(device, ...)` and `optimizer.zero_grad()`, insert CutMix logic:
     - Draw `lam` from `np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)`
     - Generate shuffled indices: `rand_index = torch.randperm(inputs.size(0), device=device)`
     - Compute bbox via `rand_bbox(inputs.size(), lam)`
     - Blend: `inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]`
     - Adjust λ to reflect actual pixel ratio: `lam = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2))`
     - Store `targets_a = targets` and `targets_b = targets[rand_index]`
  5. Replace the loss line inside the AMP autocast block with mixed-label loss:
     `loss = lam * F.cross_entropy(outputs, targets_a) + (1 - lam) * F.cross_entropy(outputs, targets_b)`

  Rationale: CutMix provides cross-sample regularization orthogonal to the existing per-sample augmentation pipeline. The implementation follows the original Yun et al. 2019 paper. The adjusted λ (step 4, last sub-step) corrects for boundary clipping — standard practice from the paper.

## Configuration Changes
- `CUTMIX_ALPHA`: new, set to `1.0` (uniform λ distribution, standard setting from the CutMix paper)
- No other hyperparameters change. LR, WD, batch size, schedule, warmup all preserved from EXP-009.

## Execution Environment
- Method: local command `uv run train.py > run.log 2>&1`
- Resources: single H20 GPU (~96 GB VRAM), expect ~865 MB peak VRAM (same as EXP-009 — CutMix is in-place on existing tensors)
- Estimated runtime: ~5-7 minutes total (300s training + ~110s startup/eval overhead). CutMix adds negligible per-step overhead (tensor indexing + one extra CE forward).
- Log output: `run.log` in project root, stdout+stderr redirected
- Tool skill: none (local execution)

## Abort Criteria
- No output in run.log after 120 seconds from launch (startup should take ~1-2s)
- Loss becomes NaN or inf (check via `grep -i "nan\|inf" run.log`)
- OOM error in run.log
- Training accuracy stays below 50% after 20 epochs (CutMix shouldn't prevent basic learning with α=1.0)

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 95.49%** (baseline 95.39% + 0.1pp)
```bash
grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'
```
Pass if the extracted value is strictly greater than 95.49. Fail otherwise.

**Condition 2: Summary block complete (10 fields)**
```bash
grep -c "^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:" run.log
```
Pass if the count equals 10. Fail otherwise.

**Condition 3: Eval count <= num_epochs**
```bash
eval_count=$(grep -c "eval ep" run.log)
num_epochs=$(grep "^num_epochs:" run.log | awk '{print $2}')
```
Pass if eval_count <= num_epochs. Fail otherwise.

Timeout for each condition: 10 seconds (all are grep/awk on a local log file).

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- num_steps: `grep "^num_steps:" run.log | awk '{print $2}'`
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- final_test_loss: `grep "^final_test_loss:" run.log | awk '{print $2}'`
