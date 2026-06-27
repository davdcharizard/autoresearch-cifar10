# EXP-002: Widen to 8x (stage widths 128/256/512)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md
- **Plan**: plans/plan-002.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-002
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (verification condition 2 failed — valid run, metric below bar)

## Implementation Notes

### Summary
Single-constant change per plan: `WIDTH_MULT = 4` → `8` (stage widths 128/256/512, ~17M params expected). Recipe and model class untouched. Ruff clean; only train.py modified; GPU 0 free.

### Surprises & Discoveries
- None — one-line diff.

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task bt05etg6q (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 04:14
- **Ended**: 2026-06-10 04:21

Description:
- 8x-wide ResNet-20 (~17M params) under the unchanged recipe. Continuation of the width gradient (+2.07pp at 4x). Expect ~5–7k img/s → ~30–45 epochs, total ~340–370s, best_test_acc ≥ 95.6% (hypothesis), pass bar ≥ 95.33 (baseline 95.23 + 0.1). Undertraining at low epoch count is the known risk.

Observations:
- Params line confirms width change: "ResNet-20 (8x wide) | params: 17,124,490" (source: run.log L~2)
- Epoch-1 eval: test_acc 22.63%, test_loss 1.9994 — above the 15% abort threshold; lower than EXP-001's epoch-1 (39.20%) as expected for a heavier model earlier in its warmup (source: run.log first `eval ep` line)

Key Metrics:
- best_test_acc: 94.41% | final_test_acc: 94.41% | final_test_loss: 0.2550 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 366.2 (source: run.log summary block)
- peak_vram_mb: 3248.5 | num_epochs: 40 | num_steps: 3,807 | num_params: 17,124,490 (source: run.log summary block)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 366.2 ≤ 600; exit 0 (source: run.log summary; task bt05etg6q)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 95.33)** — FAIL
   - best_test_acc = 94.41% vs baseline 95.23% → −0.82 pp (source: run.log summary; exp-index.sh baseline)
   - Context: 40 epochs / 3,807 steps — the 8x net undertrained badly within the fixed budget (EXP-001 had 114 epochs / 10,965 steps).
3. **Validation at most once per epoch** — skipped (aborted after prior failure; eval_lines 40 = num_epochs 40 for the record)

### Informational Metrics

(not collected — necessary condition failed)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
