# EXP-022: Budget-sized pre-activation Wide ResNet backbone

## Execution

Overall Status & Info:
- **Created**: 2026-06-30
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-test-accuracy-022
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (no-improvement — both WRN cells tie the DavidNet control; 96.48 bar not met)

## Implementation Notes

### Summary
Implemented the EXP-022 plan in `train.py` (sole editable file). Added (1) `import os` + env knobs (`MODEL`, `WRN_DEPTH`, `WRN_WIDTH`, `WRN_PEAK_LR`, `USE_COMPILE`, `COMPILE_MODE`, `SMOKE_SECONDS`), all defaulting to the EXP-008 DavidNet baseline; (2) `PreActBlock` + `WideResNet` modules (pre-activation basic block BN-ReLU-Conv×2 with shared-pre-activation 1×1 projection shortcut, 3 stages of N=(d−4)/6 blocks at 32/16/8, BN-ReLU-GAP-Linear head, no scale_out, `.tta` flip interface matching ResNet9); (3) a `main()` model branch (WRN: no whitening, `peak_lr=WRN_PEAK_LR`; DavidNet: unchanged with whitening + scale_out, `peak_lr=PEAK_LR`); (4) the banked off-budget torch.compile warmup (separate `train_fwd` handle, local-RNG dummies under bf16 autocast, BN snapshot/restore, no optimizer.step, `cuda.synchronize()` before `t_start_training`); (5) a fixed DataLoader `generator(1234)` so the data/aug stream is identical across cells; (6) the `SMOKE_SECONDS` throughput-sizing path (EMA-inclusive per-step cost, skip eval, print img/s + projected_epochs + final train loss); (7) a per-epoch first-step dt recompile monitor; (8) summary prints for model/compile/peak_lr/warmup. Maps to plan Milestone 1.

### Surprises & Discoveries
- WRN param counts landed exactly on the reference table (WRN-16-4 2,748,890; WRN-22-4 4,298,970; WRN-16-8 10,961,370) — wiring confirmed correct on first try.
- The pre-activation block's projection shortcut must be fed the SHARED post-BN-ReLU activation (Zagoruyko's reference), not raw `x`; implemented accordingly.

### Decisions
- **Start WRN without the whitening front-end** (reviewer endorsed): keeps a clean standard WRN and avoids the stem-input-shape confound; whitening is a DavidNet-specific rider for a follow-up if WRN shows promise.
- **DataLoader generator added unconditionally** (applies to both cells equally) so the comparison is clean; this makes c0 a same-session control rather than a byte-identical reproduction of the 96.38 baseline run — acceptable, since the verdict is relative to the same-session control + the absolute 96.48 bar.
- **WRN peak LR chosen by train-loss stability in the smoke**, never by test accuracy (avoids a test-set tuning hole); pre-registered set {0.4, 0.2, 0.1}.

## Experimental Adjustments

<!-- appended over runs -->

## Run Log

### Run 1 — correctness smoke (Milestone 1)
Metadata:
- **Job ID**: local (PYTHONPATH=. uv run python /tmp/exp022_smoke.py)
- **Log file(s)**: stdout (not persisted)
- **Status**: completed
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Observations:
- All 16 correctness checks PASS: WRN-{16-4,22-4,16-8} forward shape/finite (train+eval); param counts within 0.0–0.4% of reference; TTA == manual flip-average; all trainable params get finite grad (no dead block); torch.compile params alias model params; BN buffers restored bit-equal after warmup; params unchanged after warmup (no step); local-RNG warmup leaves global RNG untouched; DavidNet invariant intact (whiten + scale_out=0.125, 2-step loop finite); EMA wraps uncompiled module.

Key Metrics:
- correctness: 16/16 PASS (source: /tmp/exp022_smoke.py stdout)
- WRN params: 16-4=2,748,890 | 22-4=4,298,970 | 16-8=10,961,370 (source: smoke stdout)

### Run 2 — throughput sizing smoke (Milestone 2)
Metadata:
- **Job ID**: local (SMOKE_SECONDS=25 USE_COMPILE=1, per WRN size)
- **Log file(s)**: /tmp/exp022_smoke_{16-4,22-4,16-8}.log
- **Status**: pending
- **Started**: 2026-06-30
- **Ended**: pending

Description:
- Measure compiled steady throughput (EMA-inclusive) for WRN-16-4 / 22-4 / 16-8 to pick the largest size annealing at projected_epochs ≥ 135 (margin over the 130-ep/12610-step gate). Then a train-loss-stability LR check at the chosen size over {0.4, 0.2, 0.1}.

Observations:
- Sizing smoke (SMOKE_SECONDS=25, compiled, EMA-inclusive): WRN-16-4 → 31,528 img/s / proj 190.4 ep; WRN-22-4 → 21,297 img/s / proj 128.7 ep; WRN-16-8 → 11,324 img/s / proj 68.4 ep (UNDER-ANNEAL, disqualified) (source: /tmp/exp022_smoke_*.log).
- Peak-LR stability smoke (SMOKE_SECONDS=60, reaches progress 0.2 PAST the 0.15 LR peak): WRN-16-4 @0.4 loss 1.13 flat/declining at lr 0.38, proj 190.4 ep; WRN-22-4 @0.4 loss 1.14 flat at lr 0.38, proj 130.2 ep — BOTH stable at peak_lr 0.4, no divergence (source: /tmp/exp022_lrsmoke_*.log). 0.4 = the linear-scaled WRN LR (0.1×512/128), independently justified.

Key Metrics:
- WRN-16-4: 31,521 img/s, proj 190.4 ep, stable@0.4 (source: /tmp/exp022_lrsmoke_16-4.log)
- WRN-22-4: 21,548 img/s, proj 130.2 ep, stable@0.4 (source: /tmp/exp022_lrsmoke_22-4.log)
- WRN-16-8: 11,324 img/s, proj 68.4 ep — disqualified under-anneal (source: /tmp/exp022_smoke_16-8.log)

### Run 3 — official same-session triple: c0 / cA(WRN-22-4) / cB(WRN-16-4)
Metadata:
- **Job ID**: local (/tmp/exp022_orchestrate.sh, background)
- **Log file(s)**: /tmp/exp022_c0.log, /tmp/exp022_cA.log, /tmp/exp022_cB.log
- **Status**: completed (clean single-attempt; GPU1 clear, no contention retry)
- **Started**: 2026-06-30
- **Ended**: 2026-06-30

Description:
- Same-session comparison of the compiled DavidNet control (c0) against two compiled budget WRNs: cA=WRN-22-4 (4.3M, ~130 ep — most capacity that anneals) and cB=WRN-16-4 (2.75M, ~190 ep — clean-anneal fallback). All peak_lr 0.4, identical recipe (EMA/TTA/aug/one-cycle) and identical data stream (DataLoader gen 1234). Tests whether a different backbone family clears 96.48; cB disambiguates anneal-vs-ceiling if cA ties.

Observations:
- Clean single attempt (orchestrator attempt 1: GPU1 clear foreign=0MiB util=0%, no retry). All three cells trained the full 300.0s and finished < 600s wall (c0 486.6s / cA 459.9s / cB 483.8s) (source: /tmp/exp022_orchestrate.out, /tmp/exp022_{c0,cA,cB}.log).
- **Both WRN cells TIE the DavidNet control**: cA(WRN-22-4) 96.31 − c0 96.32 = −0.01pp; cB(WRN-16-4) 96.34 − c0 96.32 = +0.02pp. Neither clears the 96.48 bar.
- **cA fully annealed (NOT under-anneal)**: num_steps 12880 ≥ the 12610 (130-ep) gate; recompile monitor steady at 24–26ms first_step_dt from ep2→133 (no mid-run recompile); early trajectory healthy (ep1 33.8% → ep5 69.2%, no divergence at peak_lr 0.4). So cA's tie is a genuine ceiling datapoint. (source: /tmp/exp022_cA.log)
- **cB annealed luxuriously** (196 ep / 18988 steps, best 96.34 @final) and also ties — confirming the WRN tie is not anneal-limited across two sizes (2.75M comfortable-anneal AND 4.30M near-gate-anneal both land ~96.3). (source: /tmp/exp022_cB.log)
- Compile recipe worked cleanly on the new backbone: warmup 12–14s off-budget; c0 compiled control hit 172 ep (matching EXP-014/021's ~173-ep band → recipe re-validated a 3rd time).

Key Metrics:
- c0 (DavidNet+compile): best_test_acc 96.32% @172ep/16668steps, 7,784,627 params, 486.6s wall (source: /tmp/exp022_c0.log)
- cA (WRN-22-4+compile): best_test_acc 96.31% @133ep/12880steps, 4,298,970 params, 459.9s wall, peak_lr 0.4 (source: /tmp/exp022_cA.log)
- cB (WRN-16-4+compile): best_test_acc 96.34% @196ep/18988steps, 2,748,890 params, 483.8s wall, peak_lr 0.4 (source: /tmp/exp022_cB.log)
- cA−c0 = −0.01pp; cB−c0 = +0.02pp (both within ~0.1pp noise floor)

## Verification Results

### Conditions Checked
- **NC1 — completes within budget, valid best_test_acc, wall < 600s**: PASS. All cells training_seconds=300.0, total_seconds 459.9–486.6s < 600s, valid best_test_acc printed (source: /tmp/exp022_*.log).
- **NC3 — anneal gate (experiment-specific, num_steps ≥ 12610)**: PASS for the primary WRN cell — cA 12880 ≥ 12610 (cB 18988). The WRN tie is therefore a genuine ceiling result, not an under-anneal artifact.
- **NC2 — improves over baseline ≥ +0.1pp (best_test_acc ≥ 96.48 AND beat same-session c0 by > noise)**: **FAIL**. cA 96.31 < 96.48 (and −0.01pp vs c0); cB 96.34 < 96.48 (+0.02pp vs c0). Neither WRN cell clears the absolute bar or beats the control beyond noise → no-improvement. Confirmation pair (Milestone 4) NOT triggered (requires ≥96.48 AND >0.15pp lead).
- **NC4 — genuine method change**: PASS. `git diff --quiet -- prepare.py` clean; only train.py modified (`git status`: ` M train.py`); seed fixed 42; WRN LR chosen by train-loss stability (smoke), not test acc; eval runs once per epoch (outside the batch loop).
- **Verdict**: no-improvement (NC2 fails on a valid, fully-annealed result).

### Informational Metrics
- Not collected (NC2 failed — informational metrics are gathered only when all necessary conditions pass). Key sizes/epochs already captured inline above.

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
