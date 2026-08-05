# EXP-015: Mild policy-based augmentation (RandAugment), replacement vs add

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-015
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the EXP-015 plan in train.py only (45+/10−). Added `import os` and `from torchvision.transforms import RandAugment`; an env-toggle config block (`AUG_MODE ∈ {baseline, randaug_replace, randaug_add}`, `RANDAUG_N=1`, `RANDAUG_M=6`); a module-level `build_train_tf(aug_mode, n, m)` helper that inserts `RandAugment(n,m)` between the PIL geometric augs and `ToTensor`, keeps `Cutout(12)`, and drops `RandomErasing` only in `randaug_replace`; replaced the static `train_tf` in main with a call to it; and added `aug_mode`/`randaug_n`/`randaug_m` summary prints. Construction smoke (M1) passed for all 3 modes: baseline is regression-equal to the EXP-008 pipeline (`RandomCrop,Flip,ToTensor,Normalize,Cutout,RandomErasing`), replace drops RE, add keeps RE. `git diff --name-only` = train.py only.

### Surprises & Discoveries
- **RandAugment is CPU-bound on this harness (the load-bearing finding).** Pre-smoke DataLoader throughput (M2, inline probe, 8 workers): baseline loader **37,765 img/s** but `randaug_add` loader only **20,586 img/s** — BELOW the ~26k img/s steady GPU compute rate. So with RandAugment the CPU augmentation, not the GPU, is the bottleneck. NUM_WORKERS=8 is imported from frozen prepare.py and cannot be widened.
- **Why this does NOT break the epoch-matched comparison**: the budget accumulates per-step COMPUTE time (`dt` from `t0` to `cuda.synchronize()`), and the DataLoader wait happens at the `for ... in train_loader` line BEFORE `t0` — so loader slowness is excluded from `total_training_time`. Compute budget (300s) and therefore `num_epochs` (~150) stay protected; only WALL `total_seconds` inflates (est. ~+70–80s → ~520s, under the 600s cap). All cells thus get ~matched epochs → the same-session comparison is fair; the only cost is wall time, which doesn't affect the metric. WATCH `total_seconds` < 600 as the abort guard (review #4).

### Decisions
- **Proceed despite the M2 probe missing its ≳27k pass criterion.** The criterion was a throughput-free guarantee; the budget mechanism (compute-timed) means the epoch count is protected anyway, so the experiment stays valid — only wall inflates, and the estimate (~520s) sits under the 600s cap with margin. Recorded the CPU-bound regime explicitly; if any cell's wall approaches ~590s or hits timeout (exit 124), abort that cell.
- **Run cells strictly SEQUENTIALLY on GPU 1** (never concurrent) so each gets full GPU + full CPU workers — concurrent cells would contend and break the epoch-matched design.

## Experimental Adjustments

- **RandAugment N=1, M=6 (mild)**: chosen ≪ the CIFAR default (N=2, M=14) because the ~150-epoch budget is far shorter than policy-aug's canonical 200–2000ep recipes (under-fit risk). (ref: knowledge/references/policy-augmentation.md; idea-review #2)

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background — sequential c0 → cA → cB)
- **Log file(s)**: experiments/015/run_c0.log, run_cA.log, run_cB.log; GPU snapshots gpu_c0/cA/cB.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Same-session 3-cell run on GPU 1. c0 `AUG_MODE=baseline` (noise control), cA `AUG_MODE=randaug_replace` (RandAugment replaces RandomErasing), cB `AUG_MODE=randaug_add` (RandAugment added on top of Cutout12+RandomErasing). Each `CUDA_VISIBLE_DEVICES=1 timeout 600 uv run train.py`, nvidia-smi logged before each. Expect ~150 epochs/cell (compute-protected); policy cells' wall ~520s. Testing whether mild policy aug lifts best_test_acc to ≥96.48.

Observations:
- **All 3 cells epoch-matched at num_epochs=149** — the CPU-bound RandAugment did NOT cut epochs (compute-budget protection held exactly as predicted). (source: run_c0/cA/cB.log `num_epochs:`)
- **Wall inflation as predicted, all under the 600s cap**: c0 450.6s, cA 463.2s, cB 513.9s — cB (heaviest: RandAugment+Cutout+RandomErasing) +63s over baseline. (source: `total_seconds:`)
- **No GPU-1 contention**: foreign PID 1723342 stayed dormant (3843 MiB, 0–3% util) across all 3 cells; c0 ran GPU-bound at ~26.5k img/s. (source: gpu_c0/cA/cB.log; run_c0.log L8)
- **Not under-fit**: ep25 policy cells 92.04 (cA) / 92.19 (cB) vs c0 92.27 — within ~0.2pp; all cells fully annealed (best≈final, no still-climbing). The mild aug is fully absorbed → the null is genuine, not a strength artifact. (source: run_*.log `eval ep  25` + last evals)
- Integrity: `git diff --name-only`=train.py only; prepare.py byte-unchanged; seed fixed 42; 1 eval/epoch.

Key Metrics:
- c0 (baseline)        best_test_acc: **96.36%** @149ep, final 96.26, ep25 92.27, wall 450.6s (source: run_c0.log)
- cA (randaug_replace) best_test_acc: **96.34%** @149ep, final 96.31, ep25 92.04, wall 463.2s (source: run_cA.log)
- cB (randaug_add)     best_test_acc: **96.36%** @149ep, final 96.34, ep25 92.19, wall 513.9s (source: run_cB.log)
- Best policy cell = 96.36 (cB) = same-session c0 (96.36); both below the 96.48 goal bar. num_params 7,784,627 (unchanged), peak_vram 1635 MB.

## Verification Results

### Conditions Checked
- **(a) Run completes without crash, within budget, valid best_test_acc, wall < 600s**: PASS. All 3 cells exited 0, training_seconds=300.0, valid best_test_acc printed, max wall 513.9s < 600s. (source: run_*.log)
- **(b) Best policy cell best_test_acc ≥ 96.48 (baseline 96.38 + 0.1pp)**: **NOT MET**. Best policy cell cB = 96.36% (and = same-session c0 96.36; cA 96.34). 96.36 < 96.48, and 0.00pp over the same-session control. Multiple-comparison caveat moot (neither cell even reaches c0). → routes to no-improvement verdict in analyze.
- **(c) Integrity (only train.py changed, prepare.py unchanged, ≤1 eval/epoch, seed fixed)**: PASS. git diff=train.py only; prepare.py byte-unchanged; 1 evaluate()/epoch; manual_seed(42) fixed.
- Same-session validity: PASS — all cells 149 epochs, no contention, all wall < 600s.

Note: Condition (b) not met = a valid below-bar result (no-improvement), NOT a crash/invalid. Execution Outcome=completed (the run produced valid metrics within all hard constraints); the no-improvement verdict is rendered in the analyze phase (matches EXP-009..014 convention).

### Informational Metrics
- peak_vram_mb: 1635.4 MB (all cells, source: run_*.log)
- num_epochs: 149 (all cells) | training_seconds: 300.0 (all) | total_seconds: 450.6/463.2/513.9 (c0/cA/cB)
- num_params: 7,784,627 (unchanged — aug-only change) (source: run_*.log `num_params:`)
- ep25 test_acc: 92.27 / 92.04 / 92.19 (c0/cA/cB) — under-fit diagnostic, all healthy.

## Errors & Dead Ends
(none — clean run, no retries)

## Human Notes
>
