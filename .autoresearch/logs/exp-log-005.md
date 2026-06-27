# EXP-005: Width 5x on the doubly-regularized recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md
- **Plan**: plans/plan-005.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-005
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 2 — metric below baseline + 0.1pp)

## Implementation Notes

### Summary
Single constant change per plan: `WIDTH_MULT = 4` → `5` in train.py (stage widths 80/160/320), inline comment updated. The full doubly-regularized recipe (time-keyed one-cycle peak 0.4, TrivialAugmentWide, RandomErasing, batch 512, selective WD, label smoothing) stays byte-identical to baseline 1174e0d. Ruff clean; only train.py modified; GPU 0 idle at launch.

### Surprises & Discoveries
- None.

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task bs1na573g (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed (metric below bar)
- **Started**: 2026-06-10 04:58
- **Ended**: 2026-06-10 05:04

Description:
- 5x-wide ResNet-20 (~6.7M params) under the TA+RandomErasing recipe — probes the width-epoch optimum from the safe side after augmentation returns collapsed at 4x. Expect ~75–85 epochs, total wall clock ≤ ~416s (fewer evals than EXP-004), best_test_acc ≥ 96.4% (hypothesis), pass bar ≥ 96.33. Undertraining signature to watch: depressed absolute accuracy with final=best.

Observations:
- Params 6,693,850 — matches the ~6.7M quadratic-scaling prediction (source: run.log L2)
- Epoch-1 eval: test_acc 35.79% (source: run.log first `eval ep` line) — above the 15% abort bar, between EXP-004's 34.26% (same recipe, 4x) and EXP-003's 39.23%, consistent with a wider net under the same augmentation

Key Metrics:
- best_test_acc: 95.12% | final_test_acc: 95.12% | final_test_loss: 0.2216 (source: run.log summary block)
- training_seconds: 300.1 | total_seconds: 362.8 (source: run.log summary block)
- peak_vram_mb: 2019.8 | num_epochs: 52 | num_steps: 5,043 | num_params: 6,693,850 (source: run.log summary block)
- Effective throughput: 512 x 5043 / 300 ≈ 8,600 img/s — vs ~18,700 at 4x: a 2.19x slowdown for only 1.56x FLOPs. Stage widths 80/160/320 are not multiples of 32/64; tensor-core/cuDNN kernel efficiency likely collapsed vs the aligned 64/128/256 (source: run.log summary + step lines)
- final = best — the EXP-002 undertraining signature at 52 epochs (predicted 75–85)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 362.8 ≤ 600; exit 0 (source: run.log summary; task bs1na573g)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.33)** — FAIL
   - best_test_acc = 95.12% vs baseline 96.23% → −1.11 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — skipped — aborted after prior failure (informally: 52 eval lines = 52 epochs, compliant)

### Informational Metrics

- (not collected — necessary condition failed; values noted in Key Metrics for the scaling record)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
