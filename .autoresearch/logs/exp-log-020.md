# EXP-020: SWA with a lower constant-LR floor (SWA_LR 0.05 → 0.02)

## Execution

Overall Status & Info:
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-020
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Re-applied the EXP-019 SWA implementation (discarded after its no-improvement verdict, so the tree was back at
the EXP-012 baseline) with the SINGLE change `SWA_LR = 0.02` (was 0.05). Six edits, all train.py: imported
`AveragedModel`; added `SWA_START_FRAC=0.75`, `SWA_LR=0.02`, `BN_RECOMPUTE_BATCHES=50`; rewrote `lr_at_fraction`
(warmup → cosine PEAK_LR→0.02 over [5%,75%] → constant 0.02 tail); added `recompute_bn()`; constructed
`swa_model = AveragedModel(model)`; branched the per-epoch eval (tail → update_parameters + recompute_bn + eval
SWA model; main phase → eval raw model). Maps to Milestone 1 (ruff clean, parses, scope=train.py only, SWA_LR=0.02).

### Surprises & Discoveries
- None — mechanical single-constant change to already-validated EXP-019 code.

### Decisions
- **Single-variable change (SWA_LR only; SWA_START_FRAC=0.75 unchanged)**: isolates the floor-LR effect for clean
  attribution vs EXP-019's 95.97, and keeps the ~24-snapshot tail whose SWA-eval curve was still rising at ep 91.

## Experimental Adjustments

<!-- none yet -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — local background run)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-08
- **Ended**: 2026-06-08

Description:
- Re-running EXP-019's proper-SWA setup with the constant-LR floor lowered 0.05→0.02 (all else identical).
  Expect ~91 training epochs (throughput-neutral, params unchanged 4,299,866), constant tail LR reading 0.0200,
  ~24 SWA snapshots. Hypothesis: higher-top-1 snapshots (LR nearer the converged region) while still moving
  enough to average → best_test_acc above EXP-019's 95.97 and plausibly past the 96.32 bar. Risk: if 0.02 is too
  low (too little movement) the average approaches a single constant-0.02 endpoint lacking cosine-to-0 sharpening
  → ~96.0–96.1 (no-improvement). Diagnostic: compare final_test_loss to EXP-019's 0.1788 and baseline 0.195.

Observations:
- Clean startup: `params: 4,299,866` (UNCHANGED), clean compile, no traceback, no NaN.
- LR schedule correct: decayed 0.20 → 0.02 then held EXACTLY `lr: 0.0200` constant through the tail (ep 68–91).
- Tail fired at ep 68 (~75%): 67 `[raw]` + 24 `[swa]` evals = 91 = num_epochs → one evaluate()/epoch (constraint satisfied).
- **Lowering the floor 0.05→0.02 lifted SWA from EXP-019's 95.97 → 96.13 (+0.16pp), confirming the EXP-019 diagnosis** that the 0.05 floor was too high (capped snapshot top-1). final_test_loss 0.1806 (≈ EXP-019's 0.1788, far below baseline 0.195 — still a flat/low-loss minimum).
- SWA-eval curve STILL RISING at the budget end (ep 89→90→91: 96.06→96.08→96.13) — the average had not fully converged, same as EXP-019.
- **Decisive trend**: the SWA floor sweep is monotone toward the baseline-from-below (0.05→95.97, 0.02→96.13). As SWA_LR→0 the constant tail degenerates into cosine-anneal-to-~0 (= the 96.22 baseline), so SWA ASYMPTOTES toward 96.22 from below and cannot exceed it here.
- Throughput-neutral: 91 epochs (= EXP-012/EXP-019), total_seconds 422.0 < 600.

Key Metrics:
- best_test_acc: **96.13%** @ ep 91 (source: run.log summary) — vs baseline 96.22 (**−0.09pp**); below the 96.32 bar. (vs EXP-019 95.97: +0.16pp)
- final_test_acc: 96.13% | final_test_loss: 0.1806 @ ep 91
- num_epochs: 91 | num_params: 4,299,866 | peak_vram_mb: 469.3 | total_seconds: 422.0

## Verification Results

### Conditions Checked
- **Cond 1 — primary metric clears bar**: **FAIL**. best_test_acc = 96.13% < 96.32 bar. Δ = **−0.09pp** vs
  baseline 96.22 (but +0.16pp vs EXP-019's 95.97). → verdict no-improvement. (Decisive; evaluated first.)
- **Cond 2 — clean completion within budget**: PASS (for completeness). Summary present; total_seconds 422.0 < 600;
  Traceback count 0 (source: run.log).
- **Cond 3 — no constraint violations**: PASS (for completeness). git diff = train.py only; num_params 4,299,866
  unchanged; eval-count 91 == num_epochs 91 (67 raw + 24 swa, one evaluate()/epoch); no new deps (swa_utils is
  core torch); seed 42 intact.

### Informational Metrics

## Errors & Dead Ends

## Human Notes

> (none)
