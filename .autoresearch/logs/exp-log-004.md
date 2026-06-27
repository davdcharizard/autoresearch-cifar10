# EXP-004: TrivialAugmentWide on top of the regularized recipe

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md
- **Plan**: plans/plan-004.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-004
- **Commit**: 1174e0d
- **PR**: N/A — no git remote configured (intentional per TASK.md)
- **Outcome**: completed

## Implementation Notes

### Summary
One transform inserted into `train_tf` per plan: `TrivialAugmentWide()` (library defaults — tuning-free by design) between RandomHorizontalFlip and ToTensor, since TA operates on PIL images. RandomErasing stays last after Normalize, matching the TA paper's TA-then-cutout ordering. Architecture (4x width) and recipe otherwise untouched. `TrivialAugmentWide` was verified during planning to instantiate and run on a 32x32 PIL image in torchvision 0.24.1. Ruff clean; only train.py modified.

### Surprises & Discoveries
- None.

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task b0xks34zf (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 04:47
- **Ended**: 2026-06-10 04:54

Description:
- 4x-wide ResNet-20 with TrivialAugmentWide added before ToTensor, on top of the EXP-003 RandomErasing recipe — tests whether policy augmentation converts remaining capacity into test accuracy at the 114-epoch schedule. Expect throughput/epochs essentially unchanged (~114 epochs, ~400s total), best_test_acc ≥ 96.25% (hypothesis), pass bar ≥ 96.16. Main risk: over-regularization at the short schedule.

Observations:
- Epoch-1 eval: test_acc 34.26% (source: run.log first `eval ep` line) — above the 15% abort bar; ~5pp below EXP-003's epoch-1 (39.23%), the expected signature of TA slowing early fitting. Watch whether the one-cycle anneal recovers the gap by end of schedule.

Key Metrics:
- best_test_acc: 96.23% | final_test_acc: 96.21% | final_test_loss: 0.1947 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 416.5 (source: run.log summary block)
- peak_vram_mb: 1620.7 | num_epochs: 114 | num_steps: 11,023 | num_params: 4,286,026 (source: run.log summary block)
- Mid-run trajectory: ep40 83.78 / ep80 91.60 / ep85 92.72 — ran below EXP-003 mid-schedule (augmentation pressure) and converged past it in the anneal (source: run.log eval lines)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 416.5 ≤ 600; exit 0 (source: run.log summary; task b0xks34zf)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.16)** — PASS
   - best_test_acc = 96.23% vs baseline 96.06% → +0.17 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — PASS
   - eval lines = 114 = num_epochs (source: run.log)

### Informational Metrics

- peak_vram_mb: 1620.7 (identical to EXP-003 — no GPU memory change)
- num_epochs: 114 (identical to EXP-003 — TA's CPU cost did not become the bound; img/s ~19k as before)
- num_params: 4,286,026 (unchanged — pure augmentation experiment)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
