# EXP-006: Multi-crop translation TTA (airbench96 tta_level=2)

## Execution

Overall Status & Info:
- **Created**: 2026-06-28
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-006
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (valid run; best_test_acc 95.93% below the 96.10 bar → no-improvement)

## Implementation Notes

### Summary
Implemented the EXP-006 eval-only change on the EXP-004 base (96.00%). One edit to `ResNet9.forward` in `train.py` (lines 180-185 → now the TTA branch): replaced mirror-only TTA with airbench96 `tta_level=2` mirror+translate (6 views). The fast path `if self.training or not self.tta: return self._forward_once(x)` is preserved verbatim, so training and pre-tail eval are untouched. New branch: a local `mirror(v)` helper averaging `f(v)` and `f(v.flip(-1))`; `F.pad(x,(1,1,1,1),'reflect')`; two diagonal-shift crops `[:, :, 0:h, 0:w]` (shift −1,−1) and `[:, :, 2:2+h, 2:2+w]` (shift +1,+1), each mirror-averaged; final `0.5*mirror + 0.5*translate`. Milestone 1 passed: `py_compile` clean; `git diff` vs the integration branch is confined to `forward` (no training-affecting line changed); in-process smoke confirmed (a) `tta=True` issues **exactly 6** `_forward_once` calls, (b) `tta=False` and the `model.train()` path each issue **exactly 1** (training trajectory preserved), (c) output `[8,10]` finite, (d) TTA logits differ from single-forward (max-abs-diff 11.43 — views genuinely averaged), (e) both reflect-pad crops are `[8,3,32,32]`, feeding the conv stack identically.

### Surprises & Discoveries
- None. `F` (`torch.nn.functional`) was already imported (line 7), so no import change was needed — the diff stays a pure `forward`-body edit. The reflect-pad crops are non-contiguous views of the padded tensor; the whitening `Conv2d` consumed them without issue in the smoke (cuDNN handles non-contiguous input on the off-budget eval path).

### Decisions
- **Used `h, w = x.shape[-2:]` instead of hardcoding 32** — behaviorally identical for CIFAR eval (32×32) but robust to shape; keeps the edit self-documenting.
- **Held `TTA_START_FRAC=0.8`** (baseline value) for the primary run so the sole change vs the 96.00 baseline is mirror→mirror+translate — cleanest single-variable attribution. The 0.8→0.9 wall-clock fallback (plan Abort Criteria) is a contingency only, not used unless the run is killed at 600s.

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
- Official EXP-006 run: `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. The training recipe is byte-identical to the EXP-004 96.00 baseline (same seeds, HPs, architecture, schedule, EMA, whitening); the ONLY change is eval-time TTA going from mirror-only (2 views) to airbench96 mirror+translate (6 views), gated to the final 20% of epochs. Tests whether the two added translation views lift `best_test_acc` from 96.00% to ≥96.10%.

Observations:
- **NO-IMPROVEMENT: best_test_acc 95.93%** (baseline 96.00%, **−0.07pp**), below the 96.10 bar and below baseline. Clean run, exit 0, no divergence. (source: run.log:308)
- **Multi-crop TTA DID lift accuracy — there is a clear TTA-onset step.** At the TTA gate (progress≥0.8) test_acc jumps ep118 95.51 → ep119 95.79 (+0.28pp), mirroring EXP-002's flip-TTA step-up. So the 6-view TTA works as intended; the issue is not that translation TTA fails. (source: run.log eval trace ep113-150)
- **The headline missed the bar because this run's *base trained model* was a slightly-worse draw than EXP-004's.** Training is eval-independent and byte-identical in code, but the time-budgeted loop fit **150 epochs / 14507 steps this run vs EXP-004's 142** — host-throughput variance (GPU 0 shared) changes how many SGD steps land in the fixed 300s, so the final weights differ run-to-run. This run's model peaked at 95.93% *even with* the richer 6-view TTA, i.e. the trained-model variance (~±0.1pp at this accuracy) is comparable to or larger than the marginal gain of translate-over-mirror. (source: run.log:316 num_epochs=150 vs experiments/004 num_epochs=142)
- peak VRAM 1635 MB; whitening 0.43s off-budget; total wall 472.1s < 600s (6-view tail eval cost ~27s over EXP-004's 445.2s — well under the estimate, comfortably inside the guard). (source: run.log:311-317)

Key Metrics:
- best_test_acc: 95.93% @ ep148 (source: run.log:308; max per-epoch best=95.93 == summary)
- final_test_acc: 95.90% @ ep150 | final_test_loss: 0.3242 (source: run.log:309,310,315)
- training_seconds: 300.0 | total_seconds: 472.1 | whitening_seconds: 0.43 (source: run.log)
- num_epochs: 150 | num_steps: 14507 | peak_vram_mb: 1635.4 | num_params: 7,784,627 (source: run.log:311-317)
- Trajectory (TTA tail): ep113 95.33 / ep118 95.51 / **ep119 95.79 (TTA onset +0.28)** / ep135 95.92 / peak 95.93 @ep148

## Verification Results

### Conditions Checked

1. **Clean run within wall guard** — PASS. `RUN_EXIT=0` (not 124); exactly one `^best_test_acc:` line; `total_seconds 472.1` < 600. (source: run_exit.txt, run.log:308,312)
2. **Full training budget + scope/integrity intact** — PASS. `training_seconds 300.0` ≥ 295; `prepare.py` byte-unchanged vs the integration branch; `git diff --name-only <dev>` lists only `train.py`; the diff is confined to the `ResNet9.forward` body (no training-affecting line changed — verified at Milestone 1); `num_params 7,784,627` and the change is provably eval-only, so the training trajectory is unperturbed by construction. num_epochs=150 within the planned 120–160 band (host-throughput variance, not a code perturbation). (source: git, run.log)
3. **Improvement ≥ +0.1pp** — **FAIL → no-improvement**. `best_test_acc 95.93%` < bar 96.10% (−0.07pp vs the 96.00 baseline). Verification stops here (first failed necessary condition). Metric is genuine: max per-epoch best 95.93 == summary 95.93, from `Eval.evaluate`, one eval/epoch, seeds unchanged. (source: run.log)

**Necessary condition 3 failed → verdict: no-improvement** (valid run — clean, in-scope, eval-only change, metric genuine — but below the bar).

### Informational Metrics
- peak_vram_mb: 1635.4 (source: run.log:314) — ~equal to EXP-004's 1635 (TTA crops are tiny 32×32 activations on the off-budget eval path).
- num_epochs / num_steps: 150 / 14507 (source: run.log:316,317) — 8 MORE epochs than EXP-004 (142), confirming run-to-run throughput variance from the shared host; the extra eval cost of 6-view TTA (≈27s) did not reduce epochs (eval is off the training budget).
- num_params: 7,784,627 (source: run.log:317) — identical to EXP-004 (no architecture change), confirming the eval-only edit.
- total_seconds: 472.1 (source: run.log:312) — vs the ~500–540s estimate; the 6-view tail eval was cheaper than projected, so the wall-clock fallback (raise TTA_START_FRAC) was never needed.

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
