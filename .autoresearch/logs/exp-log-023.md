# EXP-023: Lower label smoothing (LABEL_SMOOTHING 0.1 → 0.05)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-023.md
- **Plan**: plans/plan-023.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-023
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (run clean; verification Cond 1 failed → no-improvement verdict, rendered in analyze)

## Implementation Notes

### Summary
Single-constant change on the clean EXP-012 baseline `train.py`: line 27 `LABEL_SMOOTHING = 0.1` → `0.05` (Milestone 1). Everything else identical (k=4 4.3M params, PEAK_LR 0.2 cosine-to-0, batch 128, Nesterov, WD 1e-4, TrivialAugment + Cutout(16), torch.compile, bf16, channels_last, seed 42). ruff + AST clean; `git diff --name-only` = train.py only.

### Surprises & Discoveries
None. `LABEL_SMOOTHING` flows only into `F.cross_entropy(..., label_smoothing=LABEL_SMOOTHING)` — zero compute/param change.

### Decisions
Chose the single interior step 0.05 (not 0.0) per the project's interior-step discipline; 0.0 reserved as a follow-up if 0.05 helps. Motivated by the convergence-bound insight (EXP-005/011/018/022): reduce a regularizer rather than add one.

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
- Runs the EXP-012 recipe with label smoothing halved (0.1→0.05) to test the project's strongest insight — the recipe is convergence-bound, so REDUCING a regularizer is the indicated direction. LS is the one recipe regularizer never swept. Expected: a small top-1 gain (sharper targets within the budget) clearing 96.32, or graceful no-improvement if LS effects are within noise. NOTE: test loss is NOT comparable across LS values (LS adds a fixed CE offset) — judge on best_test_acc only.

Observations:
- Clean run: params 4,299,866 (unchanged), clean compile, no traceback, no NaN; throughput-neutral (91 epochs, same as the EXP-012 baseline) — LS is scalar-only (source: run.log L2, summary).
- best_test_acc 96.03% vs baseline 96.22 = **−0.19pp**, below the 96.32 bar. final_test_loss 0.1564 < 0.195 is the EXPECTED LS-offset artifact (lower LS → lower CE), NOT a quality signal (flagged in plan). Reducing LS slightly HURT top-1 → 0.1 was near-optimal/load-bearing; the "reduce a regularizer" direction does not help via LS (source: run.log summary + eval lines).

Key Metrics:
- best_test_acc: 96.03% (source: run.log `best_test_acc:` line)
- final_test_loss: 0.1564 (LS-offset artifact — not comparable to baseline 0.195)
- final_test_acc: 95.91%; num_epochs: 91; num_steps: 35316; total_seconds: 403.3; peak_vram_mb: 453.8; num_params: 4,299,866 (source: run.log summary)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAILED**. best_test_acc = 96.03% < 96.32 (−0.19pp vs baseline 96.22). (source: run.log `best_test_acc: 96.03%`)
- **Cond 2 — clean completion within budget**: skipped — aborted after Cond 1 failed. (Would pass: total_seconds=403.3 < 600, Traceback=0, metrics present.)
- **Cond 3 — no constraint violations**: skipped — aborted after Cond 1 failed. (Would pass: git diff = train.py only, num_params=4,299,866, eval-count=91 == num_epochs=91, no new deps, seed 42 intact.)

Verdict basis: first necessary condition failed → no-improvement; remaining conditions not evaluated.

### Informational Metrics

- Not collected (only when all conditions pass). For the record: peak_vram_mb=453.8, num_epochs=91 (= baseline, throughput-neutral), num_steps=35316, final_test_loss=0.1564 (LS-offset artifact, not comparable to baseline 0.195).

## Errors & Dead Ends

<!-- none -->

## Human Notes

> (none — autopilot)
