# Experiment Log EXP-025: Final-phase clean-data alignment (aug-off tail at progress ≥ 0.85)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-025 (cut from autoresearch/dev @ 1990397)
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: failed (verification condition 1 not met — converged no-improvement, 95.84 vs 96.81 bar)

## Implementation Notes

### Summary
Three-part edit to `train.py` per plan-025 Milestone 1: (1) new constant `ALIGN_FRAC = 0.85`; (2) a second dataset/loader pair (`clean_set`/`clean_loader`) over the SAME train split with EXACTLY the evaluator's transform — ToTensor + Normalize((0.4914,0.4822,0.4465),(1,1,1)), verified against prepare.py L13–20 — and DataLoader kwargs identical to `train_loader` (batch 512, shuffle, 8 workers, pin_memory, drop_last, persistent_workers); (3) an epoch-level source switch in the while loop: `loader = clean_loader if total_training_time >= ALIGN_FRAC * TIME_BUDGET_S else train_loader`, with the for loop iterating `loader`. The timed step body, schedule, optimizer, compile path, and eval are byte-identical. Diff: 1 file, +29/−1.

### Surprises & Discoveries
- None at implementation time. The mean/std tuple is already in scope at the insertion point (defined above train_tf), so the clean transform reuses it directly.

### Decisions
- Switch keyed to `total_training_time` (the timed budget) rather than epoch count, consistent with the time-keyed schedule philosophy — the alignment phase completes under any throughput.
- clean_loader workers spawn lazily at its first iteration (~85% progress); the one-time spin-up stall lands OUTSIDE dt per the t0-after-yield accounting (infra-errors EXP-013) but inside the 600s wall — headroom ~70–110s makes this safe.
- Expected switch artifact pre-classified for verification: at most ONE >30ms profile window near step ~11400 (the clean loader's first fetch) is the switch, not contention.

## Run Log

### Run 1
- **Description**: Full 300s-budget run on GPU 0. First out-of-recipe intervention after recipe-space closure (EXP-024): the final ~15% of the budget trains on clean test-distribution images so BN running stats and weights align to what the frozen evaluator measures (FixRes mechanism, arXiv 1906.06423; BN-recalibration). Expected: signatures baseline-identical (dt ~22.4ms, ~139 epochs, VRAM ~1613MB); sharp train-loss DROP at the switch (clean data is easier — expected, not divergence); test-acc dip ≤1pp for 1–2 epochs post-switch, then plateau ABOVE the baseline family. Success bar best_test_acc ≥ 96.81 with final-7-evals median ≥ 96.6.
- **Job ID**: local background composite (pre-check + launch + inline watchdog)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed, rc=0, clean signatures
- **Started**: 2026-06-10T13:04:18Z
- **Ended**: 2026-06-10T13:12:49Z
- **Observations**: Pristine run — 0/267 windows >30ms, mean 22.4ms, expected 139.1 vs 139 actual; no switch artifact even appeared in the profile (clean-loader spin-up fully absorbed outside dt; num_steps 13406 ≈ baseline). Switch fired as designed at ~step 11500 (~85%, ep ~118): train loss dropped 0.799→0.572 in one window and FROZE at 0.5033 — the model fit the clean train set almost instantly. Post-switch test trajectory: brief climb 95.45→95.84 (ep118–123), then HARD saturation at 95.73–95.84 for the final 16 epochs while test_loss ROSE 0.2096→0.2185 — classic tail overfitting. No dip-and-recover; the predicted alignment gain never materialized. Comparison: baseline's augmented cosine tail delivers ~+1.3pp over the same epochs (family ~95.3 at ep118 → 96.6+ at ep139); the clean tail delivered +0.6 then forfeited the rest. best 95.84 (−0.87 vs baseline) — worst miss since EXP-018.
- **Key Metrics**: best_test_acc 95.84 | final 95.76 | final_test_loss 0.2185 (vs baseline ~0.187 — overfit signature) | training_seconds 300.0 | total 487.8s | startup 13.2s | VRAM 1613.0MB | 139 epochs | 13406 steps | eval_lines 139 = num_epochs. Source: run.log summary + task bobi9piiy output.

## Experimental Adjustments

## Errors & Dead Ends

## Verification Results

### Conditions Checked

**Pre-condition — clean post-hoc contention profile**: PASSED. `windows>30ms: 0 of 267 | mean win 22.4 ms | expected epochs 139.1` vs 139 actual (within ±3). Not even the documented single switch-artifact window appeared — the clean-loader spin-up was fully absorbed outside the profiled steps. Run is analyzable, not contaminated.

**Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1pp)**: **FAILED**. `grep "^best_test_acc:" run.log` → **95.84%** (−0.87 vs baseline, −0.97 vs bar). Worst miss since EXP-018. Mechanism visible in the trajectory: after the switch (~ep118) the model rose 95.45→95.84 within ~5 epochs, then hard-flatlined at 95.73–95.84 for the final ~16 epochs with test_loss RISING 0.2096→0.2185 and train loss frozen at 0.5033 — the clean tail overfit the un-augmented train set and capped the plateau ~0.9pp below where the baseline's augmented tail climbs (to ~96.7) over the same span. First-failure-stop: conditions 2–3 not evaluated.

**Condition 2 — completes within budget (rc=0, total ≤600s)**: skipped per first-failure-stop; would have passed (TRAIN_EXIT rc=0, total_seconds 487.8).

**Condition 3 — validation at most once per epoch**: skipped per first-failure-stop; would have passed (eval_lines 139 = num_epochs 139).

### Informational Metrics

## Human Notes
