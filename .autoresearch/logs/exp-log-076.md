# EXP-076: Xavier Classifier Init With Zero Bias

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-076.md
- **Plan**: plans/plan-076.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-076
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed; no-improvement (`best_test_acc=93.73%`, below 94.21% threshold)

## Implementation Notes

### Summary

EXP-076 implements the approved classifier-initialization calibration. The branch `autoresearch/exp-076` was created from `autoresearch/dev`, and `train.py` is the only tracked file changed. `_weights_init` now has explicit Conv2d and Linear branches: Conv2d keeps the current default Kaiming normal initialization, while the final Linear classifier uses Xavier uniform weights and a zero bias. A startup marker was added to make the classifier initialization visible in `run.log`.

### Surprises & Discoveries

No implementation surprises. The final classifier is the only `nn.Linear` module in the current model, so the planned Linear branch cleanly targets the classifier head without needing module-name special casing.

### Decisions

Kept Conv2d initialization exactly on the anchor behavior rather than reusing EXP-072 fan-out initialization, because EXP-075 showed the Conv2d fan-out family does not compose reliably with the CutMix anchor. Used `m.bias is not None` before zeroing the bias so the initialization helper remains safe if a future Linear layer is created without bias.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local foreground session 60960 on GPU0
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 22:21 UTC
- **Ended**: 2026-06-09 22:28 UTC

Description:
- This run tests whether a classifier-specific Xavier/zero-bias initialization improves the current CutMix anchor. It preserves Conv2d initialization, CutMix alpha/probability, label smoothing, architecture, optimizer, schedule, transforms, batch size, compile/channels-last, seed, and evaluation harness. The active baseline is 94.11%, so the required improvement threshold is 94.21%.

Observations:
- Preflight scope check passed: `git diff --name-only` listed only `train.py`.
- Preflight syntax check passed: `python3 -m py_compile train.py` exited 0.
- Preflight style check passed: `uv run ruff check train.py` reported `All checks passed!`.
- GPU availability check found both GPU0 and GPU1 idle; Run 1 selected GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Startup confirmed the expected classifier-init variant: `Device: cuda`, unchanged `ResNet-20 | params: 822,790`, `Classifier init: xavier_uniform weight, zero bias`, and `CutMix alpha: 1.0, prob: 0.5, label smoothing: 0.05` (source: run.log L1-L6).
- Early output reached epoch 11 with no error signatures; `test_acc` improved from 39.97% to a current best of 79.48% (source: run.log L8-L28).
- First LR drop was reached at step 21000 in epoch 54, with `lr: 0.0100`; post-drop evaluation immediately improved to 91.53% at epoch 54, then climbed to 93.36% by epoch 67 with no error signatures (source: run.log L113-L140).
- Final summary reported `best_test_acc=93.73%`, `final_test_acc=93.25%`, and `final_test_loss=0.2786` after 101 epochs / 39,345 steps. The result is below both the active baseline 94.11% and the improvement threshold 94.21%, so EXP-076 is classified as no-improvement (source: run.log L210-L219).

Key Metrics:
- `best_test_acc`: 93.73%
- `final_test_acc`: 93.25%
- `final_test_loss`: 0.2786
- `training_seconds`: 300.0
- `total_seconds`: 394.6
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,345
- `num_params`: 822,790

## Verification Results

### Conditions Checked
- Scope: `git diff --name-only` listed only `train.py` — passed.
- Syntax: `python3 -m py_compile train.py` exited 0 — passed.
- Style: `uv run ruff check train.py` reported `All checks passed!` — passed.
- Implementation/log marker: `git diff train.py` shows Conv2d remains Kaiming normal while Linear uses Xavier uniform plus zero bias; `grep "Classifier init:\|CutMix alpha:" run.log` confirmed the classifier marker and unchanged CutMix anchor — passed.
- Run completion metric: `grep "^best_test_acc:\|^peak_vram_mb:" run.log` returned numeric `best_test_acc: 93.73%` and `peak_vram_mb: 660.4` — passed.
- Hard constraints: diff is limited to classifier initialization and startup marker in `train.py`; startup log confirms unchanged params, CutMix settings, and fixed time-budget run — passed.
- Improvement threshold: baseline is 94.11%, required threshold is 94.21%, observed `best_test_acc=93.73%` — failed improvement criterion; classify no-improvement.

### Informational Metrics
- `best_test_acc`: 93.73%
- `final_test_acc`: 93.25%
- `final_test_loss`: 0.2786
- `training_seconds`: 300.0
- `total_seconds`: 394.6
- `startup_seconds`: 1.9
- `peak_vram_mb`: 660.4
- `num_epochs`: 101
- `num_steps`: 39,345
- `num_params`: 822,790

## Errors & Dead Ends

## Human Notes

> Autopilot execution; no human intervention during implementation.
