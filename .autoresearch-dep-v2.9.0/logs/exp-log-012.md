# EXP-012: Conv1x1-based SE Blocks (channels_last-safe)

## Execution

Overall Status & Info:
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-012
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented Conv1x1-based SE blocks per plan-012.md. Added `SEBlock` class before `BasicBlock` in train.py using `nn.Conv2d(kernel_size=1)` for both FC layers (C→C//16 and C//16→C), with `bias=False`. The SE module uses `x.mean(dim=(2, 3), keepdim=True)` to preserve the (B, C, 1, 1) tensor shape, avoiding any reshape that could trigger channels_last format conversion. Integrated into `BasicBlock.__init__` as `self.se = SEBlock(out_channels)` and into `BasicBlock.forward` as `out = self.se(out)` after BN2 and before residual addition. All 9 BasicBlocks (3 per layer × 3 layers) now have SE attention. No other files or hyperparameters changed.

### Surprises & Discoveries

- System python lacks torchvision; import verification required `uv run python -c "import train"` instead of bare `python`. Not a code issue, just an environment detail.

### Decisions

- No deviations from plan. Implementation matched plan-012.md exactly.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local background process
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-cifar10/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-27
- **Ended**: 2026-05-27

Description:
- Running `uv run train.py > run.log 2>&1` on local H20 GPU with 300s training budget. This is the Conv1x1-based SE block experiment — the key hypothesis is that replacing nn.Linear with Conv2d(1x1) in the SE path will eliminate the channels_last format conversion overhead discovered in EXP-011 (~9ms/step), preserving ~95-98 epochs in the budget. Expected best_test_acc: ~95.6-95.9%, exceeding the 95.49% threshold.

Observations:
- Per-step time is ~18-19ms from step 50 onwards, same as EXP-011's nn.Linear SE (source: run.log L5 steps 50-150). Conv2d(1x1) did NOT eliminate the overhead — the hypothesis about channels_last format conversion being the bottleneck was wrong. The overhead appears intrinsic to the SE computation itself (global avg pool + two small convs + sigmoid + multiply).
- Param count: 4,318,282 (source: run.log L2)
- Early convergence normal: 79.65% by epoch 13 (source: run.log eval lines)

Key Metrics:
- best_test_acc: 95.23%
- final_test_acc: 95.23%
- final_test_loss: 0.1429
- training_seconds: 300.0
- total_seconds: 401.8
- startup_seconds: 1.3
- peak_vram_mb: 1034.4
- num_epochs: 83
- num_steps: 16,002
- num_params: 4,318,282
- per_step_dt_ms: 18-19 (at step 50+)

## Verification Results

### Conditions Checked

1. **best_test_acc > 95.49%**: **FAILED** — actual value 95.23% (source: `run.log` summary block)
2. **Summary block present**: **PASSED** — `grep -c "^best_test_acc:" run.log` returned 1
3. **Validation ran at most once per epoch**: **PASSED** — eval_count=83, epoch_count=83 (83 ≤ 83)

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> (autopilot session — no human notes)
