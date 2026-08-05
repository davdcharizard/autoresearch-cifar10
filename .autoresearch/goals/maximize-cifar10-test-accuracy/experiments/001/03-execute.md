# EXP-001: ResNet-9 (DavidNet) + time-based one-cycle on CIFAR-10

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-001
- **Commit**: 26fdc83 (on autoresearch/maximize-cifar10-test-accuracy-001; merged to integration branch)
- **PR**: (skipped — no git remote configured; commits kept local per TASK.md "Git Rules")
- **Outcome**: completed

## Implementation Notes

### Summary
Rewrote `train.py` per plan `02-plan.md` (Milestone 1). Replaced the CIFAR ResNet-20 (`BasicBlock`/`ResNet`) with the DavidNet/ResNet-9 architecture (`conv_bn`, `Residual`, `ResNet9` with `scale_out=0.125`); confirmed 6,573,120 params via smoke test. Added a pure-torch `Cutout(8)` transform appended after `Normalize`. Swapped hyperparameters to BATCH_SIZE=512, PEAK_LR=0.4, MOMENTUM=0.9, WEIGHT_DECAY=5e-4, label_smoothing=0.2, PCT_START=0.15. Replaced `MultiStepLR` with a **time-based triangular one-cycle**: each step computes `progress = total_training_time / TIME_BUDGET_S` and sets LR (linear ramp 0→0.4 over first 15% of budget, then linear decay 0.4→0). Optimizer is SGD+Nesterov; loss is `CrossEntropyLoss(label_smoothing=0.2)` (mean reduction). Added bf16 autocast + `channels_last` (model + inputs) + `cudnn.benchmark=True` (no GradScaler). Train DataLoader gains `persistent_workers=True, prefetch_factor=4`. Preserved verbatim: the `while total_training_time < TIME_BUDGET_S` loop, per-step `t0`/`synchronize()`/`dt` budget meter, the single `evaluator.evaluate` per epoch, `best_acc` tracking, `torch.manual_seed(42)`, and all summary prints. Added a per-epoch `wall: {elapsed}s` print for live 10-min-cap monitoring. Passed `py_compile` and a one-batch bf16/channels_last fwd+bwd+step smoke test (Milestone 1 complete).

### Surprises & Discoveries
- Baseline `MAX_STEPS=64000` would not bind here anyway, but to make the time budget unambiguously the terminator I set `MAX_STEPS=1_000_000` (a high guard). No effect on accounting — the `while`/`break` on `total_training_time >= TIME_BUDGET_S` still ends the run at 300s training time.
- Smoke-test loss ≈ 2.47 at init is consistent with label-smoothing 0.2 over 10 classes (uniform CE ≈ ln 10 ≈ 2.30 plus the smoothing floor), a good sanity signal that the scale_out + label-smoothing wiring is correct.

### Decisions
- **Time-based one-cycle instead of step-count OneCycleLR** (from brainstorm review concern #1): keying LR on elapsed training time needs no step calibration (which would have done off-budget training updates), cannot overrun, and guarantees the anneal completes by 300s regardless of throughput. At most a single-step LR overshoot at each boundary — negligible.
- **No TTA** (idea-02 scope; keeps eval a single forward) to bound per-epoch eval wall-clock against the 10-min cap.
- **Weight decay applied to all params** (faithful to the DavidNet recipe) rather than the no-decay-on-BN/bias variant (that was idea-01's trick), to keep this experiment a clean test of the chosen recipe.

## Experimental Adjustments

<!-- appended as runs proceed -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local; PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28 (exit code 0)

Description:
- Running the full DavidNet/ResNet-9 + time-based one-cycle recipe on GPU 1 under the fixed 300s training budget via `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'`. Expect ~120–360 epochs to fit, a smoothly annealing LR, and `best_test_acc` in the ~93.5–94.3% range (vs 91.57% baseline). Watching epoch-1 smoothed loss for divergence and the per-epoch `wall:` print vs the 600s cap.

Observations:
- Training healthy from the start — no divergence. Throughput ~32,000 img/s (≈2× baseline's ~16k) at dt≈16ms/step, 97 steps/epoch (source: run.log step lines, e.g. "step 00050 ... img/s: 32,377").
- Smooth one-cycle anneal; accuracy climbed steadily then jumped in the final ~20% as LR→0: ep10 84.19%, ep102 best 89.91% @55% progress, ep187 95.05%, final ep192 (source: run.log "eval ep" lines).
- 192 epochs / 18,529 steps fit in the 300s training budget; total wall 447.4s (well under the 600s cap); startup 1.2s (env pre-built) (source: run.log summary).

Key Metrics:
- best_test_acc: 95.22% @ ep191 (source: run.log summary "best_test_acc: 95.22%")
- final_test_acc: 95.20% @ ep192 (source: run.log "final_test_acc:")
- final_test_loss: 0.3444 (source: run.log "final_test_loss:")
- training_seconds: 300.0 | total_seconds: 447.4 | num_epochs: 192 | num_steps: 18529 (source: run.log summary)
- peak_vram_mb: 1592.0 | num_params: 6,573,120 (source: run.log summary)

## Verification Results

### Conditions Checked

1. **Runs clean within budget** — PASS. Process exited 0 (not 124); `best_test_acc` summary present; total_seconds=447.4 < 600 (source: run.log summary; background task exit 0).
2. **Used full training budget + prepare.py unchanged** — PASS. training_seconds=300.0 (≥295); `git diff --quiet -- prepare.py` clean; `TIME_BUDGET_S=300` intact in prepare.py.
3. **Improves over baseline by ≥ +0.1pp** — PASS. best_test_acc=95.22 ≥ 91.67 (baseline 91.57 + 0.1); Python float compare returned PASS. Delta **+3.65pp**.
4. **Genuine, in-scope, no reward-hack** — PASS. `git diff --name-only` vs integration branch shows only `train.py`; `torch.manual_seed(42)` present once (no seed search); exactly one `evaluator.evaluate` call site; no `train=False` and no `evaluator.loader` access in train.py (only `datasets.CIFAR10(..., train=True, ...)` at L137).

**All necessary conditions passed → verdict: improvement.**

### Informational Metrics
- peak_vram_mb: 1592.0 MB (source: run.log summary) — well within the 98 GB soft constraint (~1.6 GB used).
- training_seconds: 300.0 / num_epochs: 192 / num_steps: 18529 (source: run.log summary) — confirms full budget used; ~62.5 steps/s throughput.
- num_params: 6,573,120 (~6.5M, vs baseline 269,722) (source: run.log summary) — ~24× larger model, still trivial for the H20.
- total_seconds: 447.4 (source: run.log summary) — wall clock vs the 600s/10-min cap (comfortable margin).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
