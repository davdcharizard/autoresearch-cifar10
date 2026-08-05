# EXP-005: Second ReZero-gated residual block in layer3 (capacity-via-depth probe)

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-005
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: (pending)

## Implementation Notes

### Summary
Implemented the EXP-005 capacity-via-depth probe on the EXP-004 base (96.00%). One-line edit to `train.py:151`: appended `GatedResidual(512)` to `self.layer3` (`...Residual(512), GatedResidual(512)`), reusing the existing `GatedResidual` ReZero class verbatim. 10→12 learnable convs, all other code byte-identical (PEAK_LR=0.4). Milestone 1 passed: `py_compile` clean; diff is exactly the one-token layer3 append; in-process smoke confirmed (a) identity at init (α=0, `allclose(blk(h),h)`), (b) gate gets gradient (α.grad=0.042), **(b2) the rigorous 2-step trainability check — branch conv grad = 0.000 at α=0 on step 1, α then moves to 5.26e-3, and branch grad = 4.02 at step 2 once α≠0, proving the branch convs train once the gate opens**, (c) out [2,10] + pool input 512×4×4, (d) learnable 12,503,810 / total 12,505,268 (exact), (e) whiten frozen.

### Surprises & Discoveries
- The 2-step smoke (added per the plan-review) cleanly demonstrated the ReZero gradient mechanism: at α=0 the branch convs receive exactly zero gradient (so α.grad≠0 alone would be a necessary-but-not-sufficient liveness check); only after α moves off zero (step 1 → 5.26e-3) do the branch convs get a nonzero gradient (4.02). This confirms the block is genuinely trainable, not merely that the gate scalar moves.

### Decisions
- **Placement at layer3 (4×4), not a 2nd layer2 (8×8) block** — per the brainstorm/idea-review: layer3@4×4 is FLOP-equal to EXP-004's proven block but has the smallest activation footprint → least throughput hit → most annealing budget, which is the decisive factor at the hard 96.10 bar. Caveat acknowledged (4×4 coarse capacity is a less-proven location than EXP-004's 8×8); the fallback if it fails is a 2nd layer2 block.
- **PEAK_LR held at 0.4** — ReZero's gradual ramp removes the LR-retune rationale → clean single-variable capacity test.
- **Init not bit-equivalent to EXP-004** (plan-review #1): the new block's 2 convs consume extra kaiming RNG before `fc`, so `fc.weight` differs — a noise-level perturbation, not a confound (verdict is the metric).

## Experimental Adjustments

<!-- none -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (local, background) — PID recorded at launch
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-28
- **Ended**: 2026-06-28 (exit 0)

Description:
- Official EXP-005 run: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. Appends a ReZero-gated `Residual(512)` block to layer3 (4×4) on the EXP-004 recipe (otherwise byte-identical), to test whether a second capacity block lifts `best_test_acc` from 96.00% within the 300s budget. Expected ~96.05–96.15% (central ~96.08%), HARD bar ≥96.10%.

Observations:
- **NO-IMPROVEMENT: best_test_acc 95.90%** (baseline 96.00%, **−0.10pp**), peaked at ep128, well below the 96.10 bar AND below the 96.00 baseline. Clean run, no divergence. (source: run.log summary)
- **The 4×4/512 block ran SLOWER, not cheaper — the idea's central throughput premise was falsified.** img/s ~23.4k steady (vs EXP-004's ~26.1k, ~10% slower), giving **131 epochs (vs EXP-004's 142, 11 fewer)**. cuDNN evidently picks a less-efficient kernel for small-spatial(4×4)/large-channel(512) convs — exactly the plan-review #6 risk. So instead of *more* annealing budget, the block bought *less*. (source: run.log step lines + summary)
- **The 12-conv net never led EXP-004 — capacity was not the mechanism.** Trajectory: ep1 61.63% (vs 58.70, noise), **ep10 81.16% (vs 85.19 — 4pp BEHIND)**, ep25 92.73% (≈92.63), ep50 94.11% (≈94.00), ep100 95.32%, peak **95.90% @ep128 (vs EXP-004's 96.00%)**. The deeper net converged *slower early* (the extra depth/α-ramp slowed it) and never caught up — it did not exhibit the mid-training capacity lead EXP-004's layer2 block showed (ep25 92.63 vs 88.84). So 4×4 coarse capacity added no usable representational gain here. (source: run.log eval trace vs experiments/004/04-analysis.md)
- peak VRAM 1710 MB; whitening 0.08s off-budget; wall 432.0s < 600s. (source: run.log summary)

Key Metrics:
- best_test_acc: 95.90% @ ep128 (source: run.log "best_test_acc:    95.90%")
- final_test_acc: 95.77% @ ep131 | final_test_loss: 0.3246 (source: run.log summary)
- training_seconds: 300.0 | total_seconds: 432.0 | whitening_seconds: 0.08 (source: run.log summary)
- num_epochs: 131 | num_steps: 12645 | peak_vram_mb: 1710.0 | num_params: 12,505,268 (source: run.log summary)
- Trajectory: ep1 61.63 / ep10 81.16 / ep25 92.73 / ep50 94.11 / ep100 95.32 / peak 95.90 (source: run.log eval trace)

## Verification Results

### Conditions Checked

1. **Clean run within wall guard** — PASS. Run process exited 0 (not 124); `grep -c "^best_test_acc:"` == 1; `total_seconds 432.0` < 600. (source: run.log; RUN_EXIT=0)
2. **Full training budget + scope intact** — PASS. `training_seconds 300.0` ≥ 295; `prepare.py` byte-unchanged (vs worktree and dev); `git diff --name-only <dev>` lists only `train.py`; diff-content is the single one-token `layer3` append (`, GatedResidual(512)`), nothing else (verified at Milestone 1). (source: git)
3. **Improvement ≥ +0.1pp** — **FAIL → no-improvement**. `best_test_acc 95.90%` < bar 96.10% (and < 96.00 baseline, **−0.10pp**). Verification stops here per protocol (first failed necessary condition). The result is a *valid* run (genuineness checks not the blocker — the metric is real, from `Eval.evaluate`, max per-epoch best 95.90 == summary), it simply did not clear the bar. (source: run.log)

**Necessary condition 3 failed → verdict: no-improvement** (valid run, metric below bar; not a crash/invalid — only train.py changed, seed/eval untouched).

### Informational Metrics
- peak_vram_mb: 1710.0 (source: run.log) — ≈ EXP-004's 1635 (the deeper net's params grew but 4×4 activations are tiny).
- num_epochs: 131 (source: run.log) — 11 fewer than EXP-004's 142; the 4×4 block was SLOWER (~23.4k vs 26.1k img/s), falsifying the cheaper-throughput premise.
- num_params: 12,505,268 total / 12,503,810 learnable (source: run.log + smoke) — +4,720,641 over EXP-004 (two `conv_bn(512,512)` + α).
- Mid-trajectory vs EXP-004: ep10 −4.03pp, ep25 +0.10pp, ep50 +0.11pp, peak −0.10pp — never a sustained lead; the block did not deliver a capacity advantage. (source: run.log vs experiments/004)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
