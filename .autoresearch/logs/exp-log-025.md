# EXP-025: Large-batch throughput exploitation (batch 256 + linear LR scaling)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md
- **Plan**: plans/plan-025.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-025
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (run clean; verification Cond 1 failed → no-improvement; launch-bound premise FALSIFIED — compute-bound at batch 256; verdict rendered in analyze)

## Implementation Notes

### Summary
Implemented Milestone 1 — three hyperparameter edits in `train.py`: `BATCH_SIZE` 128→256 (L22), `PEAK_LR` 0.2→0.4 (L23, linear LR scaling rule for 2× batch), `WARMUP_FRAC` 0.05→0.08 (L24, longer warmup for the higher peak). Added two core-torch `DataLoader` kwargs (`persistent_workers=True`, `prefetch_factor=4`) as a wall-clock safeguard only — the accuracy mechanism is gated on per-step compute `dt` (L242), independent of dataloader speed. `git diff --name-only` = `train.py` only; AST parse clean. No architecture change → num_params expected unchanged (4,299,866). The time-fraction cosine schedule (`lr_at_fraction`) consumes PEAK_LR directly so it auto-anneals over the 300s budget regardless of batch — no schedule-shape edit needed.

### Surprises & Discoveries
The key enabling fact (found in planning, not new here): the 300s budget gates on `total_training_time` = Σ(per-step compute `dt`), with the timer starting AFTER the dataloader yields (L218). So #optimizer-steps ≈ 300/mean(dt) and effective epochs = steps·batch/50000. If `dt` stays ~flat at batch 256 (launch-bound regime), this ~doubles effective epochs for free — the entire premise of the experiment. The run's mean `dt` and `num_epochs` are therefore the decisive diagnostic, not just `best_test_acc`.

### Decisions
Chose batch 256 (not 384/512) as the first step — linear LR scaling is best-validated at 2×, and a moderate batch keeps the large-batch generalization gap small while still testing the throughput hypothesis. Kept `num_workers` at the prepare.py value (8) rather than bumping it; more workers wouldn't affect accuracy (compute-gated budget) and risks CPU oversubscription. persistent_workers/prefetch_factor are the minimal safe wall-clock guards.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background bash task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Runs the EXP-012 recipe at batch 256 with PEAK_LR 0.4 (linear-scaled) and warmup 0.08, testing whether the launch-bound k=4 net keeps per-step `dt` ~flat so the compute-gated 300s budget fits markedly more effective epochs than the 91 at batch 128. Expected: if dt stays ~8-9ms and epochs rise (>91) AND the strong-TA recipe is still epoch-hungry, best_test_acc clears 96.32; if dt ~doubles (compute-bound) or the recipe is epoch-saturated / hit by the large-batch generalization gap, a graceful no-improvement that still resolves the launch-bound assumption. KEY confound to watch: mean dt and num_epochs (the launch-bound diagnostic), and total wall-clock < 600s.

Observations:
- Clean run: params 4,299,866 (unchanged), 195 batches/epoch (=50000/256, correct), startup 1.9s, no Traceback, no NaN, total_seconds 377.5 < 600 (source: run.log summary).
- **LAUNCH-BOUND PREMISE FALSIFIED — the net is COMPUTE-BOUND at batch 256.** Per-step dt rose from baseline ~8ms (batch 128) to **~15ms steady-state, ~24-28ms during early/warmup steps** (source: run.log early samples `dt: 26ms… 28ms… 24ms`, late `dt: 15ms`). Mean dt ≈ 300s/13964 steps ≈ 21.5ms ≈ 2.7× baseline.
- **Consequence: optimizer UPDATES collapsed 61%** (baseline ~35,490 steps → 13,964) and epochs dropped 91→72. Because dt more than doubled, batch-256 processed FEWER images AND far fewer gradient updates within the 300s compute budget. Combined with the 2× peak LR (0.4), the cosine schedule converged to a much worse optimum (source: run.log summary + step samples).
- best_test_acc **93.84%** vs baseline 96.22 = **−2.38pp** (largest regression to date), final_test_loss 0.2583 ROSE sharply from 0.195 (under-resolved optimization). img/s late ~17,500 (only +12% over baseline ~15,600) but early ~9,700 — small partial launch-overhead amortization, nowhere near the 2× needed to keep epochs flat (source: run.log).

Key Metrics:
- best_test_acc: 93.84% (source: run.log `best_test_acc:` line)
- num_epochs: **72** (vs baseline 91); num_steps: **13,964** (vs ~35,490 baseline — KEY: −61% optimizer updates)
- dt: ~15ms steady / ~24-28ms warmup (vs ~8ms baseline — COMPUTE-BOUND diagnostic); final_test_loss: 0.2583 (vs 0.195, ROSE); final_test_acc: 93.74%; total_seconds: 377.5; peak_vram_mb: 870.2; num_params: 4,299,866 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAILED**. best_test_acc = 93.84% < 96.32 (−2.38pp vs baseline 96.22). (source: run.log `best_test_acc: 93.84%`)
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failed. (Would pass: total_seconds=377.5 < 600, Traceback=0, summary block present.)
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 1 failed. (Would pass: git diff = train.py only, num_params=4,299,866 unchanged, eval-count=72 == num_epochs=72, only core-torch DataLoader kwargs added / no new deps, seed 42 intact.)

Verdict basis: first necessary condition failed → no-improvement. The result is a CLEAN measurement (not invalid/crash) with a clear mechanism: the launch-bound premise was FALSIFIED — at batch 256 the net is compute-bound (dt ~8→15-26ms), so updates collapsed 61% (35.5k→14k) and accuracy regressed −2.38pp.

### Informational Metrics

- Not collected per protocol (only when all conditions pass). For the record: num_epochs=72 (vs 91), num_steps=13,964 (vs ~35,490, −61% updates), dt ~15ms steady (vs ~8ms — compute-bound), final_test_loss=0.2583 (vs 0.195, ROSE), peak_vram_mb=870.2, img/s ~17,500 late (vs ~15,600).

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)
