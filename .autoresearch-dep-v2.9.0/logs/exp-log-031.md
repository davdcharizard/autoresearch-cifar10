# Experiment Log: EXP-031

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-031.md
- **Plan**: plans/plan-031.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-031
- **Commit**: 00abbc2
- **PR**: (failed — token permissions; user can create manually from autoresearch/exp-031 → main)
- **Outcome**: completed

## Implementation Notes

### Summary
Two changes: `nesterov=True` in SGD and `padding_mode='reflect'` in RandomCrop. Both are parameter-only changes.

### Surprises & Discoveries
None.

### Decisions
None.

## Experimental Adjustments

(none)

## Run Log

### Run 1
- **Description**: Full training combining Nesterov momentum with reflect-padded RandomCrop. Stacking two individually-tested near-miss changes on orthogonal axes (optimizer + data quality). Expected ~96 epochs at 16ms/step, zero throughput cost.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: Training completed with 99 epochs at 16ms/step — zero throughput cost from both changes. Nesterov + reflect padding combined to produce +0.10pp, meeting the threshold after 7 consecutive failures with single-variable changes. Best accuracy 96.56% achieved during training (not at final epoch — final was 96.50%).
- **Key Metrics**: best_test_acc=96.56%, final_test_acc=96.50%, num_epochs=99, training_seconds=300.0, num_steps=19123

## Verification Results

### Conditions Checked

1. **best_test_acc >= 96.56% (baseline + 0.1pp)**: PASSED. Actual: 96.56% (+0.10pp above baseline 96.46%). Source: run.log `best_test_acc: 96.56%`
2. **Clean completion**: PASSED. Summary block printed with all fields.
3. **Max 1 eval per epoch**: PASSED. 99 evals for 99 epochs.

### Informational Metrics

## Errors & Dead Ends

(none)

## Human Notes

(autopilot — no human interaction)
