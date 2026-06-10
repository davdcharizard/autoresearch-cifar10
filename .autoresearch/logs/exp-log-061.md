# EXP-061: Clean-data BN recalibration before eval in the final epochs

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-061.md
- **Plan**: plans/plan-061.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-061
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented Milestone 1 on top of the EXP-054 baseline: (1) added module-level `recalibrate_bn(model, loader, device, n_batches)` (snapshot augmented BN stats → reset + momentum=None → forward-only over clean batches on the EAGER model → return snapshot) and `restore_bn(bn, backup)`; (2) added constants `BN_RECAL_FRAC=0.025`, `BN_RECAL_BATCHES=16`; (3) built `clean_loader` (CIFAR10 train, transform = ToTensor+Normalize ONLY — matches the frozen eval transform in prepare.py, no crop/flip/AugMix, 2 workers); (4) wrapped the existing per-epoch `evaluate()` so that in the final ~2.5% of the time budget it does recalibrate→eval→restore. All else byte-identical to EXP-054. Smoke: AST OK; scope train.py only; unit test confirms recalibrate_bn changes running_mean, restore_bn restores stats+momentum exactly, 22 BN layers handled, num_params 4,299,866 unchanged.

### Surprises & Discoveries
- None. `nn.BatchNorm2d.reset_running_stats()` + `momentum=None` gives exact cumulative population stats; restoration via buffer `.copy_()` is exact. The model is left in eval mode by recalibrate_bn, which `evaluate()` re-asserts anyway.

### Decisions
- **Recalibration runs on the EAGER `model`, never `compiled_model`**: avoids any CUDA-graph capture interaction (infra-errors EXP-042). The compiled training forward is byte-identical to baseline, so dt should stay 8ms.
- **Restore augmented stats after each tail eval**: keeps the next epoch's training unperturbed (the recalibration is a transient eval-only override).
- **Tiny recalibration (last ~2 epochs, 16 batches ≈ 0.35s wall)**: protects the wall-tight (~593s) AugMix recipe. best_test_acc is a max over epochs, so replacing only the flat tail's evals is no-downside (the natural augmented-BN peak at ~ep87-89 is preserved).
- **GPU 1 chosen**: idle (0 MiB/0%); GPU 0 runs an unrelated v2.9.5 job.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: (pending — background bash, GPU 1)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Clean-data BN recalibration: in the final ~2 epochs, recompute BN running stats on un-augmented training images (matching the clean eval distribution) before the single per-epoch eval, then restore augmented stats. Tests whether correcting the Cutout/AugMix train→eval BN-statistics mismatch lifts best_test_acc. Bar = 96.55. Launched on idle GPU 1. Expected: ~91 epochs, dt 8ms (compiled forward unchanged), wall ~593s.

Observations:
- Clean startup (source: run.log head): params 4,299,866, batch/epoch 390, **dt steady 8ms** (compiled training forward byte-identical → CUDA-graph intact, no EXP-042-style break — confirms the eager-only recalibration design). Loss descending, 0 NaN/error. Early epochs identical to EXP-054 (recalibration fires only in the final ~2 epochs). GPU 1 solo (GPU 0 runs unrelated v2.9.5 job).

Key Metrics:
- **best_test_acc: 96.28%** (baseline 96.45, bar 96.55 → **−0.17pp, no-improvement**); set by an augmented-BN epoch BEFORE the recalibration tail. (source: run.log)
- **Clean-BN tail epochs CRATERED**: ep89 94.71 / ep90 94.91 / ep91 94.83 / ep92 94.65 (loss ~0.225), vs the augmented-BN peak 96.28 → clean-BN recalibration is **~1.6pp WORSE** than augmented BN. (source: run.log `eval ep 89-92`)
- final_test_acc 94.65 (clean-BN ep92), final_test_loss 0.2272. (source: run.log)
- **total_seconds: 604.6 > 600** — recalibration overhead tipped the already-wall-tight AugMix recipe over the 600s budget (EXP-054 was 593s, flagged high-variance). (source: run.log)
- num_epochs 92, num_steps 35535, num_params 4,299,866 ✓, peak_vram 453.8. eval-line count 92 == num_epochs (≤1 eval/epoch ✓). (source: run.log)
- dt steady 8ms throughout (compiled forward untouched, no cudagraph break). The augmented-BN peak 96.28 is itself −0.17 vs EXP-054's 96.45 (run-to-run AugMix variance / extra epoch / mild clean-loader CPU contention).

## Verification Results

### Conditions Checked

- **Necessary condition 1 — best_test_acc >= 96.55**: actual **96.28** → **FAIL** (−0.27 vs bar). Verdict = no-improvement. Stop at first failed necessary condition. (source: run.log `best_test_acc: 96.28%`)
- (For completeness — also failed:) Condition 2 — clean completion within budget: **total_seconds 604.6 > 600 → FAIL (wall-budget breach)**; num_params 4,299,866 ✓, summary printed ✓, 0 NaN/error ✓. The breach is the recalibration overhead on top of the wall-tight AugMix recipe (EXP-054 593s, replication-may-exceed per goal-learnings). Condition 3 — scope: `git diff --name-only` = train.py only ✓; prepare.py/eval untouched ✓; eval-line count 92 == num_epochs (≤1 eval/epoch ✓); no new deps ✓; seed 42 ✓.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> {none — autopilot}
