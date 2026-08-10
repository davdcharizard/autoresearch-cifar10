# Plan EXP-013: Batch-256 Linear Scaling
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement the isolated batch-scaling candidate
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-013` from accepted integration commit `7c1e7d8` and modify only `train.py`.
- [x] Change exactly four training constants: batch 128 to 256 and LR values `0.1/0.01/1e-4` to `0.2/0.02/2e-4`; add only the reviewed fixed-count evaluation control needed to keep 19 test-set looks.
- [x] Pass compilation, Ruff, formatting, pre-commit, exact-diff, model identity, target-format, LR-boundary, and evaluator-cadence checks. Loader-size checks were not reached after the earlier paired timing veto.

### Milestone 2: Pass paired throughput and production-path gates
- [x] Run five fresh-process alternating H20 control/candidate pairs with 100 warmups and 500 timed hard/soft training steps; measured 18.91% higher image throughput, below the required 20%, so the candidate was vetoed.
- [ ] Audit at least 1,000 strong batches plus weak-loader reconstruction for shapes, 45-55% mixing, row-sum semantics, 195 batches/epoch, exactly 49,920 images/epoch, and eight clean worker exits.
- [ ] Run 1,000 integrated real-loader candidate steps and one unchanged fixed evaluator pass; require loader headroom, <=5% uncounted pipeline gap, finite LR-0.2 training, and projected total runtime below 540 seconds.

### Milestone 3: Execute and verify one fixed-seed run
- [ ] Confirm one idle 97,871 MiB H20 and no stale completed log, then run exactly once under a 600-second timeout with all stdout/stderr redirected to `run.log`.
- [ ] Verify the numeric summary, 300-second budget, wall limit, batch-consistent exposure, parameter count, one augmentation switch, worker shutdown, CutMix provenance, and unique evaluation epochs.
- [ ] Compare `best_test_acc` with 94.15%; improvement requires at least 94.25%. Use switch/weak-tail diagnostics only for interpretation, never tuning or retry.

## Code Changes
- **`train.py` training intervention**: set `BATCH_SIZE = 256`, `LR = 0.2`, `ANNEAL_START_LR = 0.02`, and `MIN_LR = 2e-4`.
- **`train.py` measurement control**: replace the open-ended `dense_tail_due` per-epoch condition with a fixed 19-checkpoint elapsed-progress schedule: the accepted `(0.2, 0.4, 0.6, 0.7)` checkpoints plus 15 evenly spaced points from 0.8 through 1.0 inclusive. Keep the single existing evaluator call and `while` checkpoint advancement. This matches EXP-010's 19 observations and prevents batch-256's shorter epochs from gaining 3-4 extra chances to maximize test accuracy.
- The intervention jointly defines standard 2x linear batch scaling. Doubling each LR level approximately preserves gradient displacement and coupled-decay shrinkage per complete 49,920-image pass when updates per pass fall from 390 to 195.
- Model, optimizer type/momentum/decay, augmentation, CutMix, elapsed-time phase boundary, timer, workers, `Eval.evaluate()`, seed, summary, and logging remain unchanged. Only invocation cadence is controlled for equal observation count. The result is the net batch/LR method; it cannot isolate image exposure from gradient-noise, BN-statistic, momentum-horizon, or CutMix-pairing changes.

## Configuration Changes
- `BATCH_SIZE`: 128 -> 256 (measured throughput knee; +28.44% preliminary image exposure).
- `LR`: 0.1 -> 0.2 (linear scaling during the 80% plateau).
- `ANNEAL_START_LR`: 0.01 -> 0.02 (preserve the 10x boundary drop and scaled weak-tail curve).
- `MIN_LR`: `1e-4` -> `2e-4` (scale the complete curve consistently; no post-result endpoint tuning).
- Evaluation opportunities: batch-dependent dense per-epoch tail -> exactly 19 elapsed-progress checkpoints, matching EXP-010's four early plus 15 tail observations. Threshold spacing is pre-registered and accuracy-independent.
- Expected production range: 16.4k-17.3k updates, 4.20M-4.42M image slots, 84-89 dataset passes, 13.1k-13.8k strong updates, 3.3k-3.5k weak updates, and about 1.12 GB peak allocation.
- Deliberately excluded: warmup, momentum adjustment, batch-dependent decay changes, batch 512, gradient accumulation, AMP, compilation, fused SGD, channels-last, augmentation changes, or batch fallback.

## Adversarial Review Response
- Mandatory Claude plan review completed successfully and is preserved in `02-plan-review.md`; no fallback reviewer was used.
- Accepted concerns 1-2: prevent max-over-more-evaluations bias by fixing the candidate to exactly 19 elapsed-progress evaluations, equal to EXP-010. Image exposure remains a mechanism diagnostic and is never presented as proof that an accuracy gain came from exposure.
- Accepted concern 3: clarify that `prepare.py` already fixes evaluation batch size 256 independently of training `BATCH_SIZE`; measure the unchanged evaluator as-is.
- Accepted concern 4: fresh five-pair results must be written as a new trial table in `03-execute.md`; preliminary serial numbers cannot satisfy the gate.
- Accepted concern 5: extend the integrated real-loader test to 1,000 steps across more than five candidate epochs, retain the 1,000-batch loader audit, and monitor actual full-run wall time. With fixed evaluation count, sustained loader delay affects wall time but cannot create extra metric observations.

## Execution Environment
- Method: local `timeout 600s uv run train.py > run.log 2>&1` after all correctness, throughput, loader, lifecycle, and wall gates pass.
- Resources: exactly one idle NVIDIA H20 with approximately 97,871 MiB; existing eight forkserver DataLoader workers; frozen installed dependencies.
- Estimated runtime: 330-370 seconds total; exactly 300 counted training seconds; exactly 19 evaluation passes matching EXP-010.
- Log output: full stdout/stderr only in project-root `run.log`; monitoring uses bounded targeted patterns and process status, never `tee` or full-output streaming.
- Tool skill: none; local execution.

## Abort Criteria
- Do not launch if the tracked diff is not exactly the four reviewed training constants plus fixed-count evaluation control, loaders do not produce 195 batches / 49,920 image slots, target semantics fail, LR boundaries are wrong, model parameter/state shapes change, or evaluator count is not exactly 19.
- Do not launch if either paired timing CV is >=3%, batch-256 image throughput is <1.20x control, ratio-projected image exposure is <4.131M slots, candidate p95 image throughput is <1.15x control, or any loss/gradient/parameter is non-finite at LR 0.2.
- Do not launch if strong or weak loader delivery is <1.25x candidate GPU consumption, p95 iterator wait exceeds 20% of candidate step time, integrated wall/counted ratio exceeds 1.05, worker shutdown leaks, or conservative total projection is >=540 seconds.
- During execution stop on wrong/busy GPU, traceback, OOM, non-finite loss, malformed target, failed worker shutdown, no progress for 90 seconds while alive, or any process exceeding 600 seconds.
- Do not stop for a low but finite checkpoint. A switch checkpoint below 87.08% diagnoses failure of the batch-256/LR-0.2 operating point rather than disproving image exposure; the fixed tail must finish.
- Exactly one mechanically valid seed-42 run. No warmup, LR/momentum/decay adjustment, batch fallback/escalation, or seed reroll after any timing or accuracy observation.

## Verification Protocol

### Verification Procedure
1. Query the moving baseline with `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv`; require `baseline=94.15` and `baseline_commit=7c1e7d8`, hence a 94.25 success threshold.
2. Require `git diff --name-only 7c1e7d8` to print only `train.py`; a zero-context diff may contain only `BATCH_SIZE`, `LR`, `ANNEAL_START_LR`, `MIN_LR`, the 19-value progress schedule, and removal of `dense_tail_due`. Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `uv run pre-commit run --all-files`, and `git diff --check`.
3. In a disposable process construct accepted/candidate models from seed 42 and require exactly 1,073,962 parameters, identical state dicts and CPU/CUDA RNG states, one unchanged SGD group with momentum 0.9/decay `1e-4`, and finite `[256,10]` logits/loss/gradients for separate hard `[256]` and probability `[256,10]` targets at LR 0.2.
4. Audit production loaders. Require both loaders to have length 195, produce `[256,3,32,32]` batches, consume exactly 49,920 slots per full epoch, and retain `drop_last=True`. Across at least 1,000 strong batches require mixed fraction 45-55%, finite probability targets with row sums one, and both target ranks; reconstruct the weak loader, require integer `[256]` targets only, exactly eight stopped strong workers, and no live leak.
5. On one idle H20 run five alternating fresh-process control/candidate timing pairs. Each trial recreates model/optimizer state, warms 100 steps, then times 500 synchronized steps including pinned nonblocking H2D, zero-grad, forward, hard/probability CE, backward, SGD, and synchronize. Require trial-mean CV <3%, candidate/control mean-step ratio <=1.6667, image throughput >=1.20x, ratio-calibrated candidate steps >=16,139 and image slots >=4.131M, candidate p95 image throughput >=1.15x control, finite values, and peak allocation below 1,500 MB.
6. In a fresh process run 1,000 integrated production strong-loader/model steps after warmup while separately accumulating iterator wait, full wall, and existing synchronized counted step time. Require strong delivery >=1.25x GPU consumption, p95 iterator wait <=20% of mean step, wall/count <=1.05, 45-55% mixing, finite values, and healthy workers across more than five candidate epochs. Repeat loader-only delivery checks for weak batches, then require exact eight-worker shutdown and no leak.
7. Measure one `Eval.evaluate()` pass exactly as fixed by `prepare.py` (its batch 256 is independent of training `BATCH_SIZE`). Require `300s + measured pipeline/startup/rebuild gap + 19 * eval_seconds <540s`. In a synthetic progress trace using the paired candidate step distribution and 195 steps/epoch, require the 19 thresholds to yield 19 unique evaluation epochs; otherwise the candidate is a no-go. Write a new five-pair trial table and all loader/evaluator projections into `03-execute.md`, distinct from `00-batch-timing.md`.
8. Confirm exactly one idle H20 using `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader`, require only preserved untracked `data/`, and ensure no completed `run.log` variant exists. Execute `timeout 600s uv run train.py > run.log 2>&1`; require exit 0 (124 is timeout; any other nonzero is crash).
9. Parse all ten summary fields and targeted trajectory lines. Require about 300.0 counted seconds, total <600, 1,073,962 parameters, one `randaugment+cutmix->base` switch near 80%, eight workers stopped, strong mixed fraction 45-55%, hard weak targets, and exactly 19 unique evaluation epochs.
10. Parse `num_steps`, compute `image_slots = num_steps * 256`, and require at least 4.131M slots for mechanism integrity; this confirms only that the throughput mechanism executed, not that it caused any accuracy delta. Parse `best_test_acc`; >=94.25 is improvement only with exactly 19 observations, while a complete lower result is valid no-improvement and cannot be rerun. Compare the switch checkpoint with 89.73% and the 87.08 underfit marker, first weak with 93.16%, final NLL with 0.1934, and final slope/best gap as non-veto diagnostics. A sub-87.08 switch attributes a null to the declared batch/LR operating point, not to lack of exposure headroom.

### Informational Metrics (Optional)
- Final summary metrics: targeted final lines in `run.log` for final accuracy/loss, training/startup/total seconds, peak VRAM, epochs, steps, and parameters.
- Batch/exposure: `num_steps * 256`, dataset passes using 49,920 slots/complete pass, realized CutMix fraction, and strong/weak update split.
- Trajectory: 80% switch accuracy, first weak checkpoint, best epoch, terminal slope, best/final gap, and final NLL.
- Feasibility: paired step/image throughput distributions, real-loader wait/headroom, wall/count ratio, evaluator seconds, projected/actual total, and memory.
