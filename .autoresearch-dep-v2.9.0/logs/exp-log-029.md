# Experiment Log: EXP-029

## Execution

- **Created**: 2026-05-28
- **Brainstorm**: brainstorm/brainstorm-029.md
- **Plan**: plans/plan-029.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-029
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Replaced zero-padding shortcuts with learned 1x1 conv projections at stage transitions. Modified `BasicBlock.__init__` to create `self.shortcut` as either `nn.Identity()` (when dimensions match) or `nn.Sequential(Conv2d(1x1, stride), BN)` (when dimensions mismatch). Modified `forward` to use `self.shortcut(x)` unconditionally, eliminating the stride subsampling + F.pad logic. The `_weights_init` method already applies Kaiming init to all Conv2d modules, so the new shortcut convs are initialized correctly.

### Surprises & Discoveries
None — straightforward implementation.

### Decisions
Used nn.Identity() for dimension-matching shortcuts rather than a conditional check in forward(). This is cleaner and has zero overhead since Identity is a no-op.

## Experimental Adjustments

(none)

## Run Log

### Run 1
- **Description**: Full training with learned 1x1 conv shortcut projections replacing zero-padding at stage transitions. Testing whether full gradient flow through shortcuts improves accuracy. Expected ~96 epochs with <3% throughput cost from the tiny 1x1 convs. This is the first experiment targeting shortcut connection quality.
- **Job ID**: local
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-05-28
- **Ended**: 2026-05-28
- **Observations**: Learned shortcuts added zero throughput cost (16ms/step, 98 epochs). Model trained stably but achieved 96.43%, slightly below baseline 96.46%. The zero-padding shortcut may serve as implicit regularization — the forced-zero channels in the shortcut reduce the information flow, acting like a structural dropout. Removing this regularization by providing full learned projections slightly hurt final accuracy in this well-regularized model.
- **Key Metrics**: best_test_acc=96.43%, final_test_acc=96.41%, num_epochs=98, training_seconds=300.0, peak_vram_mb=889.0, num_params=4,327,754

## Verification Results

### Conditions Checked

1. **best_test_acc > 96.56%**: FAILED. Actual: 96.43% (-0.03pp below baseline). Source: run.log `best_test_acc: 96.43%`
2. **Clean completion**: PASSED.
3. **Max 1 eval per epoch**: PASSED. 98 evals for 98 epochs.

### Informational Metrics

## Errors & Dead Ends

(none)

## Human Notes

(autopilot — no human interaction)
