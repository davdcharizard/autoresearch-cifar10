# Plan EXP-016: Higher BN Momentum (0.5)
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-016.md

## Milestones

### Milestone 1: Code change implemented
- [x] Add 3-line loop after model creation in `train.py` to set all BatchNorm2d layers' momentum from default 0.1 to 0.5
- [x] Verify no syntax errors by inspecting the modified region

### Milestone 2: Experiment completed
- [x] Run training with `uv run train.py 2>&1 | tee run-016.log`
- [x] Confirm training completes with full summary block in log output
- [x] Confirm epoch count is ~98 (zero throughput cost expected)

### Milestone 3: Verification passed
- [ ] Verify best_test_acc > 95.67% (baseline 95.57% + 0.1pp) — **FAILED** (95.59%)
- [x] Verify full summary block present (4 key metrics)
- [x] Verify eval count ≤ epoch count (validation at most once per epoch)

## Code Changes
- **train.py** (after line 149, model creation): Add a 3-line loop to override BN momentum on all BatchNorm2d layers from the PyTorch default 0.1 to 0.5. This is inserted between the `model = ResNet(...)` call and the `num_params = sum(...)` line. The loop iterates `model.modules()`, checks `isinstance(m, nn.BatchNorm2d)`, and sets `m.momentum = 0.5`. No other files or hyperparameters change.

## Configuration Changes
- BatchNorm momentum: 0.1 (PyTorch default) → 0.5 (rationale: cifar10-airbench speedrun recipe uses 0.6 for short-budget training; 0.5 is conservative for our setup with heavier regularization and different LR schedule)

## Execution Environment
- Method: local command — `uv run train.py 2>&1 | tee run-016.log`
- Resources: single H20 GPU, ~865 MB peak VRAM (unchanged from baseline)
- Estimated runtime: ~320s (300s training budget + ~20s startup/eval overhead)
- Log output: `run-016.log` in project root via `tee`
- Tool skill: N/A (local execution)

## Abort Criteria
- NaN or inf in loss values (training divergence)
- No output produced after 60s from launch (hang or crash)
- OOM or CUDA errors in log output
- Throughput drops >10% vs baseline ~16,300 img/s (would indicate unexpected overhead — not expected for this change)

## Verification Protocol

### Verification Procedure

Baseline: 95.57% (EXP-015, commit 626e9d1). Direction: higher is better. Threshold: baseline + 0.1pp = 95.67%.

**Condition 1 — best_test_acc > 95.67%**:
```bash
grep "^best_test_acc:" run-016.log
```
Extract the numeric value. PASS if > 95.67%, FAIL otherwise. Timeout: 5s.

**Condition 2 — Full summary block present (count >= 4)**:
```bash
grep -c "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^peak_vram_mb:" run-016.log
```
PASS if count >= 4, FAIL otherwise. Timeout: 5s.

**Condition 3 — Validation runs at most once per epoch**:
```bash
eval_count=$(grep -c "eval ep" run-016.log)
num_epochs=$(grep "^num_epochs:" run-016.log | awk '{print $2}')
```
PASS if eval_count <= num_epochs, FAIL otherwise. Timeout: 5s.

### Informational Metrics (Optional)
- training_seconds: `grep "^training_seconds:" run-016.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run-016.log`
- final_test_acc: `grep "^final_test_acc:" run-016.log`
- final_test_loss: `grep "^final_test_loss:" run-016.log`
- num_epochs: `grep "^num_epochs:" run-016.log`
- num_steps: `grep "^num_steps:" run-016.log`
- num_params: `grep "^num_params:" run-016.log`
