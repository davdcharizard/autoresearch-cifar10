# EXP-003: RandomErasing on the 4x-wide net

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md
- **Plan**: plans/plan-003.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-003
- **Commit**: 3a62d44
- **PR**: N/A — no git remote configured (intentional per TASK.md)
- **Outcome**: completed

## Implementation Notes

### Summary
One transform appended to `train_tf` after Normalize per plan: `RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random")` — the Random Erasing paper's CIFAR config. Architecture (4x width) and recipe untouched. Verified the `value="random"` string API instantiates cleanly in this torchvision version before launch. Ruff clean; only train.py modified.

### Surprises & Discoveries
- None.

### Decisions
- None beyond the plan.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background bash task b7k1nfss8 (local)
- **Log file(s)**: run.log (project root)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 04:29
- **Ended**: 2026-06-10 04:36

Description:
- 4x-wide ResNet-20 with RandomErasing added to the train transform — tests whether occlusion regularization converts the observed train/test gap (final_test_loss 0.2447 at EXP-001) into test accuracy. Expect throughput/epochs essentially unchanged (~114 epochs, ~395s total), best_test_acc ≥ 95.5% (hypothesis), pass bar ≥ 95.33.

Observations:
- Params unchanged: 4,286,026 (source: run.log L~2)
- Epoch-1 eval: test_acc 39.23% — essentially identical to EXP-001's epoch-1 (39.20%), confirming RandomErasing adds no early-training cost and the pipeline throughput is unaffected (source: run.log first `eval ep` line)

Key Metrics:
- best_test_acc: 96.06% | final_test_acc: 96.06% | final_test_loss: 0.2084 (source: run.log summary block)
- training_seconds: 300.0 | total_seconds: 399.5 (source: run.log summary block)
- peak_vram_mb: 1620.7 | num_epochs: 114 | num_steps: 10,996 | num_params: 4,286,026 (source: run.log summary block)

## Verification Results

### Conditions Checked

1. **Run completes without crashing within the time budget (≤ 10 min total)** — PASS
   - `best_test_acc:` present; total_seconds = 399.5 ≤ 600; exit 0 (source: run.log summary; task b7k1nfss8)
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 95.33)** — PASS
   - best_test_acc = 96.06% vs baseline 95.23% → +0.83 pp (source: run.log summary; exp-index.sh baseline)
3. **Validation executed at most once per epoch** — PASS
   - eval lines = 114 = num_epochs (source: run.log)

### Informational Metrics

- peak_vram_mb: 1620.7 (identical to EXP-001 — transform adds no GPU memory)
- num_epochs: 114 (identical to EXP-001 — throughput unaffected, as predicted)
- num_params: 4,286,026 (unchanged — pure regularization experiment)

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}
