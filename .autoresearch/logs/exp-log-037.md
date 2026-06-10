# Experiment Log EXP-037

## Execution
- **Created**: 2026-06-09
- **Brainstorm**: brainstorm/brainstorm-037.md
- **Plan**: plans/plan-037.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-037
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Single-argument edit to `train.py` L158: `transforms.RandomCrop(32, padding=4)` →
`transforms.RandomCrop(32, padding=4, padding_mode="reflect")`. This switches the
train-only crop border from zero-fill (`constant`) to edge-reflection. Milestone 1
(code change + AST + diff scope) complete: `ast.parse` OK, `git diff --name-only`
returns only `train.py`, diff is exactly the one-line argument addition. No other
hyperparameters touched.

### Surprises & Discoveries
None. The change is mechanically trivial; `padding_mode="reflect"` is a stock
torchvision `RandomCrop` argument and reflect padding of 4px on a 32px image is well
within the mode's size constraint (pad < dimension).

### Decisions
No deviations from plan-037. Kept the change to the single planned argument; all other
recipe settings (LR schedule, optimizer, Cutout, TrivialAugment, batch, seed,
torch.compile) unchanged to isolate the augmentation-quality lever.

## Run Log

### Run 1
- **Description**: Full 300s-budget training run of the reflect-padding variant on a
  single H20. Tests whether reflecting (vs zero-filling) the 4-px RandomCrop border —
  removing the black-wedge artifact from translated training crops — improves
  best_test_acc against the bar 96.32 (baseline 96.22). Expected: throughput-neutral
  (~91 epochs, dt ~8ms), most-likely within-noise (~96.1–96.3).
- **Job ID / PID**: (local background)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Observations**: Healthy run throughout — params 4,299,866 confirmed at startup; dt held
  at 8ms for 677/712 sampled steps (34×9ms, 1×11ms) → throughput-neutral, identical to
  baseline. Test acc climbed on the baseline trajectory (ep50 91.59%, ep90 95.96%). No
  errors/NaN. run.log L-tail: final summary block present.
- **Key Metrics**: best_test_acc 96.04% | final_test_acc 96.04% | final_test_loss 0.1960
  | num_epochs 92 | num_steps 35604 | training_seconds 300.0 | total_seconds 421.1
  | startup 3.4s | peak_vram_mb 453.8 | params 4,299,866.

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Primary metric clears the bar** (`best_test_acc >= 96.32`, baseline 96.22 + 0.1):
   **FAILED** — best_test_acc = 96.04% (−0.18pp vs baseline 96.22; −0.28 vs bar).
   Source: run.log `best_test_acc: 96.04%`.
2. **Clean completion within budget**: PASSED — summary block printed, total_seconds
   421.1 (< 600), training_seconds 300.0, exit 0. (Recorded for completeness; verdict
   already determined by condition 1.)
3. **No hard-constraint violations**: PASSED — `git diff --name-only` = train.py only;
   num_params 4,299,866 (unchanged); eval-line count 92 == num_epochs 92 (≤1 eval/epoch);
   no new deps; seed 42 unchanged; prepare.py/eval untouched.

Outcome: run completed cleanly and produced a valid result, but the primary necessary
condition (clear the bar) failed → **no-improvement**. The crop-padding-mode sub-lever
(constant vs reflect) is closed.

### Informational Metrics
- peak_vram_mb: 453.8 (≈ baseline; CPU-side aug change, no VRAM impact)
- num_epochs / num_steps: 92 / 35604 — throughput-neutral vs baseline ~91 ep (confirms
  reflect-padding adds no per-step cost; dt held at 8ms).
- final_test_loss: 0.1960 ≈ baseline 0.195 — converged, NOT an underfit. The reflect
  border simply did not move top-1 (or moved it slightly negative within noise).

## Human Notes
(none — autopilot)
