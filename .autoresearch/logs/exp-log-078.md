# EXP-078: Pre-Activation BasicBlock

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-078.md
- **Plan**: plans/plan-078.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-078
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: no-improvement

## Implementation Notes

### Summary

EXP-078 implements the approved pre-activation BasicBlock topology in `train.py` only. `BasicBlock` now normalizes and activates its input before `conv1`, normalizes and activates again before `conv2`, adds the unchanged option-A shortcut, and returns the residual sum without a final block-level ReLU. `ResNet` now applies a final BatchNorm/ReLU before global average pooling, and startup output includes `Block topology: pre-activation BasicBlock`.

### Surprises & Discoveries

No implementation surprises. The pre-activation layout requires `bn1 = nn.BatchNorm2d(in_channels)` rather than `out_channels`; this was already anticipated in the plan. The final BN means parameter count should change slightly from the current 822,790 anchor.

### Decisions

Kept the shortcut computed from the original input `x`, preserving the existing option-A stride slicing and zero-channel padding exactly. Kept the stem as the current `conv1 -> bn1 -> relu` path and applied the final pre-activation BN/ReLU only after `layer3`, which matches the planned narrow topology test without changing transforms, optimizer, schedule, or CutMix settings.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 26119 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: running
- **Started**: 2026-06-09 22:51 UTC
- **Ended**: 2026-06-09 22:58 UTC

Description:
- This run tests whether changing residual block topology to pre-activation improves the current probabilistic CutMix anchor. It preserves stage widths, depth, option-A shortcuts, CutMix alpha/probability, label smoothing, optimizer, LR schedule, transforms, batch size, compile/channels-last, seed, and evaluation harness. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found GPU0 idle and GPU1 partially active; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected pre-activation variant: `Device: cuda`, `ResNet-20 | params: 822,846`, `Block topology: pre-activation BasicBlock`, and unchanged `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05` (source: run.log L1-L4).
- The parameter count changed by +56 versus the 822,790 anchor, matching the planned small BN-topology effect.
- Early training was healthy with no error signatures; pre-drop best reached 88.82% at epoch 51.
- The first LR drop was reached at step 21000 in epoch 54, switching to `lr: 0.0100` with about 118s remaining. Post-drop accuracy reached 91.57% at epoch 54 and 92.71% at epoch 55.
- Post-drop accuracy peaked at 93.92% in epoch 83, then stayed below the current 94.11% baseline through the end of the fixed 300s training budget.

Key Metrics:
- `best_test_acc`: 93.92%
- `final_test_acc`: 92.84%
- `final_test_loss`: 0.2880
- `training_seconds`: 300.0
- `total_seconds`: 392.6
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.8
- `num_epochs`: 94
- `num_steps`: 36,288
- `num_params`: 822,846
- Verdict: no-improvement. This is below both the 94.11% baseline and the 94.21% improvement threshold.

## Verification Results

### Conditions Checked
- Code-scope check passed: the only tracked code change was `train.py`.
- Syntax check passed: `python3 -m py_compile train.py` exited 0.
- Style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- Startup markers confirmed `Device: cuda`, `Block topology: pre-activation BasicBlock`, and unchanged CutMix `alpha=1.0`, `prob=0.5`, `label smoothing=0.05`.
- Run completed cleanly and produced numeric final summary metrics in `run.log`.
- No traceback, CUDA, import, shape, NaN, non-finite, or timeout error signature was observed.
- Improvement threshold check failed: `best_test_acc=93.92%` is below the required `94.21%`.

### Informational Metrics
- Peak VRAM increased to 660.8 MB from the anchor's roughly 577 MB range, consistent with the small final-BN/topology change and not a resource concern.
- The topology completed 36,288 optimizer steps in the fixed training budget, which is lower than the anchor-region post-activation runs and suggests pre-activation added modest runtime overhead.

## Errors & Dead Ends
- No infrastructure errors or crashes.
- Scientific dead end: the pre-activation BasicBlock topology underperformed the post-activation CutMix anchor, reaching 93.92% rather than the required 94.21%.

## Human Notes

> Autopilot execution; no human intervention during implementation.
