# Plan EXP-005: AMP (torch.cuda.amp) with GradScaler
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md

Hypothesis: AMP will increase throughput 1.5-2x, fitting ~100+ epochs (vs 69) in 300s, raising best_test_acc from 93.33% to 93.8-94.5%. Threshold: >= 93.43%.

## Milestones

### Milestone 1: Code changes implemented
- [ ] Create branch `autoresearch/exp-005` from `autoresearch/dev`
- [ ] Add channels_last conversion after model creation
- [ ] Add GradScaler creation
- [ ] Wrap forward+loss in autocast context
- [ ] Replace loss.backward() with scaler.scale(loss).backward()
- [ ] Replace optimizer.step() with scaler.step(optimizer) + scaler.update()
- [ ] Convert input tensors to channels_last in the training loop
- [ ] Ruff check pass

### Milestone 2: Experiment runs to completion
### Milestone 3: Verification completes

## Code Changes

**train.py** — 6 modifications:

1. **After model creation (line 147)**: Add `model = model.to(memory_format=torch.channels_last)`
2. **After optimizer (line 152)**: Add `scaler = torch.amp.GradScaler('cuda')`
3. **Input conversion (line 200)**: Add `memory_format=torch.channels_last` to inputs.to() call
4. **Forward+loss (lines 204-205)**: Wrap in `with torch.amp.autocast('cuda', dtype=torch.float16):`
5. **Backward (line 206)**: `loss.backward()` → `scaler.scale(loss).backward()`
6. **Optimizer step (line 207)**: `optimizer.step()` → `scaler.step(optimizer)` + add `scaler.update()` after scheduler.step()

## Configuration Changes
- AMP autocast: off → FP16 (torch.amp.autocast)
- GradScaler: none → enabled (torch.amp.GradScaler)
- Memory format: default (NCHW) → channels_last (NHWC)

## Execution Environment
- **Method**: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- **Resources**: One H20 GPU, ~600 MB VRAM expected (may decrease slightly with FP16)
- **Estimated runtime**: ~320-360s total (300s training + overhead). Per-step time expected ~6-8ms.
- **Log output**: run.log

## Abort Criteria
- No output 2min → hang; traceback → code error; >600s → timeout; nan/inf → divergence
- If scaler detects inf gradients repeatedly, it skips steps — if loss plateaus or NaN, abort

## Verification Protocol

### Verification Procedure
Baseline: 93.33%. Threshold: 93.43%.
- Condition 1: best_test_acc > 93.43%
- Condition 2: Summary block complete
- Condition 3: eval_count <= num_epochs

### Informational Metrics
- training_seconds, total_seconds, peak_vram_mb, num_epochs, num_steps, num_params
