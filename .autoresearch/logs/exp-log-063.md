# EXP-063: Final-Stage-Only SE Gate

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/maximize-cifar10-best-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-063.md
- **Plan**: plans/plan-063.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-063
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented the planned final-stage-only SE variant in `train.py`. The patch adds `SE_STAGE = "layer3"` and `SE_REDUCTION = 16`, defines a channels-last friendly `SEBlock` with adaptive average pooling and `1x1` convolutions, threads `use_se` through `BasicBlock` and `_make_layer`, and enables SE only for `layer3`. The anchor model depth, stage widths, optimizer, schedule, data augmentation, label smoothing, compile/channels-last path, validation cadence, and timing loop are unchanged.

### Surprises & Discoveries

No code-structure surprises. `ResNet.__init__` already constructs the three stages independently, so the stage-limited gate can be implemented without subclassing blocks or adding conditional logic in `forward` beyond the per-block `nn.Identity` / `SEBlock` module.

### Decisions

- Used `Conv2d` layers for the SE MLP rather than reshaping to `Linear`, preserving 4D tensors and channels-last friendliness.
- Kept `max(channels // SE_REDUCTION, 4)` for the SE bottleneck, matching the all-block SE implementation style while avoiding a degenerate hidden width.
- Printed `SE stage: layer3 only, reduction: 16` at startup to distinguish this run from EXP-058 all-block SE during verification.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: local session 10826; shell PID 3598445; uv PID 3598446; main python PID 3598449
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.5-gpt-5-5/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-09 19:14:34 UTC
- **Ended**: 2026-06-09 19:21:36 UTC

Description:
- Local foreground run of EXP-063 on one selected GPU with output captured to `run.log`. This tests whether final-stage-only SE can improve semantic channel calibration while preserving ResNet-20 depth and most of the anchor's step budget. Expected behavior is startup reporting `ResNet-20`, `SE stage: layer3 only, reduction: 16`, unchanged batch geometry, the first LR drop at step 21000, and final `best_test_acc` classified against the 94.07% threshold.

Observations:
- Preflight passed: tracked diff is limited to `train.py`, `python3 -m py_compile train.py` exited 0, and `uv run ruff check train.py` reported `All checks passed!`.
- Baseline for classification: `93.97%`; improvement threshold: `94.07%`.
- 2026-06-09 19:14 UTC: GPU0 selected after `nvidia-smi` showed GPU0 at `0MiB` and `0%` utilization. GPU1 had unrelated work from another checkout and was not touched.
- 2026-06-09 19:14 UTC: Foreground run launched on GPU0. Process table showed shell PID 3598445, uv PID 3598446, and main Python PID 3598449; `/proc/3598449/cwd` verified this project root.
- Startup confirmed CUDA, `ResNet-20 | params: 827,851`, `SE stage: layer3 only, reduction: 16`, 300s budget, and `Batches per epoch: 390`.
- 2026-06-09 19:15 UTC: Early training healthy through epoch 8 with best test accuracy 80.54%, mostly 7-10ms batch timings, no traceback/OOM/runtime-error patterns, and GPU0 active.
- 2026-06-09 19:16 UTC: Mid-run pre-drop progress healthy through epoch 21. Best reached 85.45%; timing stayed mostly 7-10ms and no error patterns were present.
- 2026-06-09 19:18 UTC: Pre-drop training reached epoch 44 with best test accuracy 88.53%. LR remained 0.1000, with about 138s training budget remaining, so the step-21000 first LR drop was still reachable.
- 2026-06-09 19:19 UTC: First LR drop confirmed at `step 21000 ep 54` with `lr: 0.0100`. Pre-drop best was 88.53%; post-drop refinement reached 93.18% by epoch 56 with about 100s training budget remaining.
- 2026-06-09 19:20 UTC: Post-drop refinement plateaued around 93.26% by epoch 70, below the 93.97% baseline and 94.07% improvement threshold, but the run continued for final summary metrics.
- 2026-06-09 19:21 UTC: Run exited cleanly with final summary metrics. Best accuracy remained 93.26%, final accuracy was 92.94%, and the model completed 89 epochs / 34,502 steps in the 300s training budget.

Key Metrics:
- `best_test_acc`: 93.26%
- `final_test_acc`: 92.94%
- `final_test_loss`: 0.2623
- `training_seconds`: 300.0
- `total_seconds`: 393.8
- `startup_seconds`: 2.3
- `peak_vram_mb`: 661.0
- `num_epochs`: 89
- `num_steps`: 34,502
- `num_params`: 827,851
- Verdict for execution: valid no-improvement because 93.26% is below both the 93.97% baseline and the 94.07% improvement threshold.

## Verification Results

### Conditions Checked
- Baseline check: `exp-index.sh baseline` reported `baseline=93.97`, `baseline_commit=755be2c`, `total_experiments=64`, `improvements=9`; pass.
- Scoped diff check: `git diff --name-only` listed only `train.py`; pass.
- Compile check: `python3 -m py_compile train.py` exited 0; pass.
- Style check: `uv run ruff check train.py` reported `All checks passed!`; pass.
- Execution summary check: `run.log` contains numeric final metrics including `best_test_acc: 93.26%`; pass.
- Model-depth check: `run.log` reports `ResNet-20 | params: 827,851`; pass.
- SE-scope check: `run.log` reports `SE stage: layer3 only, reduction: 16`; pass.
- Batch-geometry check: `run.log` reports `Batches per epoch: 390`; pass.
- LR-drop check: `run.log` contains `step 21000 ep 54 ... lr: 0.0100`; pass.
- Error scan: `rg -n "Traceback|CUDA out of memory|RuntimeError|\bnan\b|\binf\b" run.log` returned no matches; pass.
- Classification check: valid run, but `93.26% < 94.07%`; classified as no-improvement.

### Informational Metrics
- Parameter count increased from the 822,790-param anchor to 827,851, less than EXP-058 all-block SE's 830,143 parameters.
- The stage-3-only SE model completed 34,502 steps and reached the first LR drop, but plateaued at 93.26%, below EXP-058's 93.71% and the current anchor.

## Errors & Dead Ends
- None.

## Human Notes

> Autopilot mode; no human approval or intervention requested during execution.
