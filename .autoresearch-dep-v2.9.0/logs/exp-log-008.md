# EXP-008: torch.compile with warmup pass

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-008.md
- **Plan**: plans/plan-008.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-008
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Added `model = torch.compile(model)` immediately after the `.to(device, memory_format=torch.channels_last)` call (line ~150). Then inserted a warmup block before `t_start_training` that creates dummy input/target tensors, runs a forward+backward pass under AMP autocast, resets optimizer state, and cleans up CUDA memory. Two bugs were fixed: (1) `torch.randn` doesn't accept `memory_format` kwarg — fixed by chaining `.to(memory_format=...)`, (2) `scaler.update()` asserts if `scaler.step()` wasn't called — fixed by using plain `loss.backward()` instead of `scaler.scale(loss).backward()` for warmup.

### Surprises & Discoveries

- `torch.randn()` does not accept `memory_format` as a keyword argument — must create tensor first then convert with `.to(memory_format=...)`.
- `GradScaler.update()` requires a prior `GradScaler.step()` call (asserts `len(found_infs) > 0`). For warmup-only passes, use plain `loss.backward()` without the scaler.
- Per-step time with torch.compile is ~9-10ms — no improvement over baseline 9ms. The H20 GPU may already be well-utilized for this model size, or the Inductor backend doesn't find significant fusion opportunities for this small ResNet with AMP.

### Decisions

Used `torch.compile(model)` with default mode (no explicit `mode=` argument). For warmup, used plain `loss.backward()` instead of scaler-wrapped backward to avoid scaler state issues. This still triggers compilation of the forward and backward graphs.

## Experimental Adjustments

- **No adjustments made**: Per-step time is ~9-10ms, identical to baseline. torch.compile provides no measurable throughput improvement on H20 for this model. The run will complete with ~83 epochs, same as EXP-007.

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27 16:55
- **Ended**: 2026-05-27 17:02

Description:
- Running torch.compile + warmup experiment locally via `uv run train.py > run.log 2>&1`. Warmup compilation took ~15s. Training now running at ~9-10ms/step — similar to baseline, suggesting torch.compile provides minimal speedup on this model/GPU. Two code fixes were needed before successful launch: (1) torch.randn doesn't accept memory_format kwarg, (2) scaler.update() requires prior scaler.step().

Observations:
- Warmup compilation succeeded (source: run.log "Warmup compilation done")
- Per-step time ~9-10ms at steps 50-350, similar to EXP-007 baseline of 9ms (source: run.log step lines)
- TensorFloat32 warning from Inductor suggests model may not be leveraging TF32 matmul (source: run.log Inductor warning)

Key Metrics:
- best_test_acc: 94.75%
- final_test_acc: 94.75%
- final_test_loss: 0.1552
- training_seconds: 300.0
- total_seconds: 411.8
- startup_seconds: 10.6
- peak_vram_mb: 484.1
- num_epochs: 82
- num_steps: 31835
- num_params: 4,286,026

## Verification Results

### Conditions Checked

- **Condition 1 (best_test_acc > 94.92%)**: **FAIL** — best_test_acc = 94.75%, which is <= 94.92% threshold. Source: run.log summary block.
- **Condition 2 (summary block complete)**: **PASS** — 4/4 required fields present (best_test_acc, final_test_acc, training_seconds, num_epochs). Source: run.log summary block.
- **Condition 3 (eval_count <= num_epochs)**: **PASS** — eval_count = 82, num_epochs = 82. Source: run.log.

### Informational Metrics

- training_seconds: 300.0
- peak_vram_mb: 484.1
- final_test_acc: 94.75%
- final_test_loss: 0.1552
- num_epochs: 82
- num_steps: 31835
- num_params: 4,286,026

## Errors & Dead Ends

### 2026-05-27 — torch.randn memory_format kwarg not supported
- Error: `TypeError: randn() received an invalid combination of arguments - got (int, int, int, int, memory_format=torch.memory_format, device=torch.device)`
- Root cause: `torch.randn` does not accept `memory_format` directly; must create tensor then `.to(memory_format=...)`
- Source: run.log (Run 1 attempt 1)
- Do NOT retry: never pass `memory_format` to `torch.randn`; use `.to(memory_format=...)` after creation

### 2026-05-27 — scaler.update() without prior scaler.step()
- Error: `AssertionError: No inf checks were recorded prior to update.`
- Root cause: Warmup called `scaler.scale(loss).backward()` + `optimizer.zero_grad()` + `scaler.update()` without `scaler.step(optimizer)`. The scaler requires a step before update.
- Source: run.log (Run 1 attempt 2)
- Do NOT retry: for warmup passes, use plain `loss.backward()` instead of `scaler.scale(loss).backward()` — the model forward graph compilation is the same regardless of loss scaling

## Human Notes

> 
