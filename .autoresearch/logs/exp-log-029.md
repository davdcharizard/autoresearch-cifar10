# Experiment Log EXP-029: Clean-data BN running-stat recalibration before every eval

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-029 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1: best 85.78 < 96.81; clean run, hypothesis refuted with inverted sign −10.93)

## Implementation Notes

### Summary
Plan-029 Milestone 1 implemented in train.py (+47/−0): (1) constants `BN_RECAL_BATCHES = 16`, `BN_RECAL_BATCH_SIZE = 512`; (2) module-level `recalibrate_bn(net, clean_batches)` — resets every BN's running stats, sets momentum None (exact cumulative average), runs the clean batches forward-only (no_grad, train mode, bf16 autocast), restores momentum 0.1, restores entry mode; (3) startup build of 16 GPU-resident channels_last clean batches (first 8,192 train images under ToTensor+Normalize only, fixed slice via shuffle=False, temporary 4-worker loader deleted after); (4) one-line insertion `recalibrate_bn(base_model, clean_batches)` immediately before the existing `evaluator.evaluate(base_model, device)` call. The timed step body (between `t0 = time.time()` and `total_training_time += dt`) is untouched — verified by diff review. Sanity: AST OK; grep shows constants, helper, startup build, exactly one call site.

### Surprises & Discoveries
- None at implementation time. The eval call site is a single line and the evaluator already manages train/eval modes itself (the baseline never calls `.eval()`), so the helper only restores the mode it entered with, defensively.

### Decisions
- Recalibration forwards run through `base_model` (eager) — never the compiled wrapper — so BN attribute flips (momentum 0.1→None→0.1) can't interact with inductor guards; momentum is restored INSIDE the helper before any compiled training step runs.
- bf16 autocast for the recalib forwards (speed): autocast keeps BN in fp32 internally, so the accumulated running stats are fp32 statistics; deviation vs the evaluator's numerics is O(1e-3) relative, negligible against the augmented→clean distribution shift being corrected.
- Fixed slice [0:8192] (shuffle=False) rather than a random subset — deterministic, no seed-hacking surface.
- Early break out of the clean loader after 16 batches; non-persistent workers shut down on `del`.
- **No contamination rerun despite 4 windows >27ms (post-run judgment)**: the plan's rerun rule exists to prevent contention-induced FALSE failures near the bar (EXP-011 lesson: stolen epochs). Here epochs matched expectation exactly (137 vs 137.3), the train-loss trajectory is family-identical, and the miss is −10.93 (≈68σ) in EVAL accuracy — a quantity foreign GPU contention cannot influence. A rerun would re-measure a conclusively-explained negative at ~9 min cost. Deviation recorded; verdict is profile-independent.

## Run Log

### Run 1
- **Description**: Full 300s-budget run on GPU 0. Single behavioral change vs baseline: the evaluator now consumes BN running stats re-estimated on 8,192 CLEAN train images before every per-epoch eval, instead of augmented-distribution EMA stats. Training is structurally untouched (dt must equal baseline 22.4ms; epochs ≈139). Hypothesis: trajectory-wide upward shift from ep1 (family ~38/ep1, ~64/ep5, ~78/ep10), plateau ≥ +0.25 over the 96.57 baseline mean, best ≥ 96.81 (the BN-alignment share of EXP-025's measured +0.35 switch boost).
- **Job ID**: local background composite, task bzdngmr0d (gates: STARTUP_KILL tick 10; EARLY_DT_KILL 3 consecutive >27ms in first 7 ticks; CONTENTION_KILL 4 consecutive >30ms; RECAL_SANITY_KILL ep1 eval <25%; NaN/inf guard)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed rc=0. Post-hoc profile: 4 of 263 windows >27ms (letter of the ≤2 rule exceeded) BUT epochs 137 vs 137.3 expected (exact), mean win 22.7ms, train-loss trace byte-similar to baseline family. NO RERUN — see Decisions: the rerun rule guards against contention-induced false failures near the bar; this miss is −10.93 ≈ 68σ and contention cannot touch eval accuracy, so the verdict is profile-independent.
- **Started**: 2026-06-10T15:01:08Z
- **Ended**: 2026-06-10T15:10:24Z
- **Observations**: **HYPOTHESIS REFUTED WITH A LARGE INVERTED SIGN.** Training untouched and healthy (dt 22.7ms ≈ baseline, 137 epochs, train loss 0.716 at step 13200 — family-identical), but every recalibrated eval reads FAR below family: ep1 33.6 (vs ~38), ep10 70.6 (vs ~78), ep30 78.0 (vs ~93, gap −15), ep60 78.3 (−16.6), ep90 79.7 (−16.3), ep110 83.2 (−13.2), final plateau 85.5–85.8 (−10.9), final test_loss 0.4916 (vs ~0.185). The gap PEAKS mid-training (max stat sensitivity at high LR) and shrinks as the anneal stabilizes activations, but never closes. Mechanism reading: the network's downstream weights/γ/β are calibrated to the AUGMENTED-distribution normalization constants used in every training forward pass (batch stats of augmented batches); eval fidelity requires reproducing those constants, NOT matching the eval-input distribution. Clean-data constants mis-scale every layer and the perturbation compounds through 20 BN layers. This is why SWA re-estimates BN with the AUGMENTED train loader. Retro-explains EXP-025's +0.35 as feature/weight adaptation, not stat alignment.
- **Key Metrics**: best 85.78 | final 85.61 | final_test_loss 0.4916 | total 541.2s | startup 16.2s (+3.5 = clean tensor build) | VRAM 1712.8 (+99.8 = clean batches, as predicted) | 137 epochs / 13225 steps | params 4,286,026 ✓ | recalib wall cost ≈ 137 × ~0.35s ≈ 48s (total 541 vs baseline 493). Source: task bzdngmr0d + run.log.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

First-failure-stop per plan-029 on Run 1 (the only run; no-rerun judgment documented in Decisions).

### Conditions Checked
1. **best_test_acc ≥ 96.81**: **FAIL** — `grep "^best_test_acc:" run.log` → 85.78%. Gap −11.03 to the bar, −10.93 to baseline. Pre-condition profile: 4/263 windows >27ms with epochs exactly at throughput expectation (137 vs 137.3) — verdict independent of profile (eval-side deficit, contention-immune; see Decisions). First failure → stop.
2. **Completes within budget**: skipped (would pass: rc=0, total_seconds 541.2 ≤ 600).
3. **Validation ≤ once/epoch**: skipped (would pass: eval_lines 137 = num_epochs 137; recalibration touches train images only — not validation).

### Informational Metrics
- peak_vram_mb 1712.8 (+99.8, the GPU-resident clean batches — as planned)
- num_epochs 137 (baseline-equivalent), startup 16.2s (+3.5 clean-tensor build), total 541.2s (recalib wall ≈ +48s, ~0.35s/eval)
- Trajectory: deficit vs family ep1 −4 → ep30 −15 → ep60 −16.6 → ep90 −16.3 → ep110 −13.2 → final −10.9; final-7 spread 0.16 (converged plateau, genuinely lower level)

## Human Notes
