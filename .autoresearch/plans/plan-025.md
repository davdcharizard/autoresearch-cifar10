# Plan EXP-025: Zero-init residual (BN2 gamma=0)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-025.md

## Milestones

### Milestone 1: Code change
- [ ] Add zero-init for BN2 gamma in ResNet.__init__ after `self.apply(self._weights_init)`

### Milestone 2: Training completes
- [ ] ~54 epochs, 300s budget

### Milestone 3: Verification
- [ ] best_test_acc >= 96.49%

## Code Changes
- **train.py**: In `ResNet.__init__`, after `self.apply(self._weights_init)`, add:
  ```python
  for m in self.modules():
      if isinstance(m, BasicBlock):
          nn.init.zeros_(m.bn2.weight)
  ```

## Configuration Changes
- None.

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Estimated runtime: ~5-6 minutes
- Log output: `run.log`

## Abort Criteria
- Loss divergence, crash

## Verification Protocol

### Verification Procedure
1. `grep "^best_test_acc:" run.log` — must be >= 96.49%
2. `grep "^training_seconds:" run.log` — must be <= 300

### Informational Metrics (Optional)
- All standard metrics via `grep "^{metric}:" run.log`
