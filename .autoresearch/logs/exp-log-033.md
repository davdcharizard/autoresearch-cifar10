# EXP-033: Augmentation cooldown — disable TrivialAugment + Cutout for the final ~15%

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-033
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-033: four localized edits to `train.py`, architecture untouched. (1) Added `COOLDOWN_FRAC = 0.15` to the hyperparameter block. (2) Added a second CPU transform `train_tf_clean` (RandomCrop + Flip + ToTensor + Normalize — the full pipeline minus `TrivialAugmentWide()`); `train_set` still defaults to the full `train_tf`. (3) Added an `aug_cooled` flag (init False) and, at each epoch boundary, once `total_training_time/TIME_BUDGET_S >= 1 - COOLDOWN_FRAC`, mutate `train_set.transform = train_tf_clean`, set `aug_cooled=True`, and print a `>>> aug cooldown ON ...` marker. (4) Gated the in-loop `cutout_batch` call behind `if not aug_cooled`. Smoke test passed: clean pipeline lacks TA while full retains it, params 4,299,866 (unchanged), COOLDOWN_FRAC=0.15, AST clean, diff = train.py only.

### Surprises & Discoveries
None — the transform-switch and Cutout-gate are clean localized edits. The CIFAR10 dataset exposes a plain `.transform` attribute that reassigns without error.

### Decisions
- TA and Cutout are switched off TOGETHER at the same epoch boundary via the single `aug_cooled` flag (rather than gating each on a per-step time fraction), keeping the cooldown a clean, single, observable transition aligned to an epoch boundary — necessary anyway because the CPU transform swap only takes effect on the next epoch's freshly-forked workers.
- Kept RandomCrop(pad-4) + HorizontalFlip in the cooldown pipeline (moderate cooldown) — these are mild, label-preserving, near-universally-beneficial geometric augs; only the strong distribution-shifting augs (TA, Cutout) are removed, isolating the train-test-distribution-gap variable.
- Kept `torch.compile(reduce-overhead)` unchanged; the model graph is identical (only the input data distribution changes), so no recompile is triggered by the cooldown.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Full 300s-compute-budget training of the k=4 WideResNet with the augmentation cooldown: full aug (RandomCrop+Flip+TA+Cutout) for the first 85% of the budget, then TA+Cutout disabled (RandomCrop+Flip kept) for the final 15% — the low-LR cosine tail. Hypothesis: aligning the weights and BN running stats with the clean test distribution during the tail lifts best_test_acc above the 96.32 bar at an unchanged ~91 epochs / dt~8ms / 4,299,866 params. KEY CHECKS: the cooldown marker fires once at frac≈0.85; throughput-neutral (epochs ~91); best_test_acc vs the 96.32 bar.

Observations:
- **Cooldown mechanism fired correctly**: `>>> aug cooldown ON at ep 77 frac 0.85` — exactly one marker, at the planned fraction (source: run.log; `grep "aug cooldown"`).
- **Convergence tracked baseline normally through the full-aug phase** (ep10 78.5%, ep40 91.0%, ep70 94.9%, ep76 95.43% — the last full-aug epoch). dt held at 8ms throughout, including post-cooldown (disabling the CPU TA op did not change the GPU-bound step time). No NaN.
- **The clean-data cooldown lifted the tail then plateaued/slightly declined**: ep76(full-aug)=95.43 → ep77=95.78 → ep78=95.96 → ep82=96.01 → ep84=96.06 → **ep86=96.10 (best)** → ep87=96.04 → ep88=96.00 → ep90=96.01. The steep climb over the first ~9 cooldown epochs is the expected clean-distribution-alignment effect (combined with the cosine-anneal tail); the plateau-and-slight-decline after ep86 suggests the 15% window is somewhat too LONG — the clean tail begins to saturate / mildly overfit once aug has been off for ~9 epochs. (source: run.log eval lines ep76-90)
- **final_test_loss 0.2000 ≈ baseline 0.195** (marginally worse), i.e. the cooldown did not improve loss — consistent with the net being at its capacity ceiling.

Key Metrics:
- best_test_acc: **96.10%** @ ep86 (baseline 96.22, bar 96.32 → **−0.12pp vs baseline, −0.22pp vs bar**)
- final_test_acc: 96.01%; final_test_loss: 0.2000 (baseline 0.195)
- num_epochs: 90 (baseline ~91 — throughput-neutral ✓); num_steps: 34,919; mean dt ~8ms
- num_params: 4,299,866 (architecture untouched ✓); peak_vram_mb: 453.8
- total_seconds: 411.0 (<600 ✓); training_seconds: 300.0
- cooldown fired: ep77 @ frac 0.85 ✓ (source: run.log summary block + marker line)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAIL.** best_test_acc = 96.10% < 96.32 (and below the 96.22 baseline by 0.12pp). (source: `grep "^best_test_acc:" run.log` → 96.10%)
- **Cond 2 — clean completion within budget**: **PASS.** Summary block printed, `grep -c Traceback run.log` == 0, total_seconds 411.0 < 600. (Not gating once Cond 1 failed; recorded for completeness.)
- **Cond 3 — no constraint violations**: **PASS.** `git diff --name-only` lists only `train.py`; num_params == 4,299,866 (architecture untouched); eval-count == num_epochs (90 == 90, ≤ once/epoch); core torch only (no new deps); seed 42 unchanged.

**Cooldown + throughput attribution (per plan)**: cooldown fired once at ep77/frac 0.85 (✓ correct), num_epochs 90 ≈ baseline ~91, dt ~8ms → this is a CLEAN, throughput-neutral, fair test of the augmentation-cooldown idea. The result (96.10) is therefore a trustworthy reading of the 15%-cooldown variant, not a compute confound. The steep late climb + post-ep86 plateau is the key qualitative signal (window likely too long).

### Informational Metrics

- peak_vram_mb: 453.8 (≈ baseline, as predicted — no new tensors; slightly lower, Cutout off in the tail)
- num_epochs / num_steps: 90 / 34,919 (throughput-neutral)
- final_test_loss: 0.2000 (≈ baseline 0.195; cooldown did not lower loss)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
