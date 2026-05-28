# Plan EXP-028: Deeper Architecture NUM_BLOCKS=4 (ResNet-26)
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md

## Milestones

### Milestone 1: Increase depth and adjust schedule
- [ ] Change `NUM_BLOCKS = 3` to `NUM_BLOCKS = 4` in train.py
- [ ] Change `ESTIMATED_EPOCHS = 100` to `ESTIMATED_EPOCHS = 80` to better match expected epoch count (~75) for cosine schedule
- [ ] Verify no other code changes needed (BasicBlock and _make_layer handle arbitrary block counts)

### Milestone 2: Run experiment and capture output
- [ ] Run `uv run python train.py > run.log 2>&1` and confirm training starts
- [ ] Confirm model reports ResNet-26 and ~5.7M params
- [ ] Confirm training completes within 300s budget (expect ~72-80 epochs)

### Milestone 3: Verify results
- [ ] Extract best_test_acc from run.log
- [ ] Check best_test_acc > 96.56% (baseline 96.46% + 0.1pp threshold)

## Code Changes
- **train.py** (line 18): Change `NUM_BLOCKS = 3` to `NUM_BLOCKS = 4` — creates a ResNet-26 (6×4+2=26 layers) with 4 blocks per stage instead of 3
- **train.py** (line 178): Change `ESTIMATED_EPOCHS = 100` to `ESTIMATED_EPOCHS = 80` — adjusts cosine schedule to match the expected ~75-80 epoch count, ensuring LR decays fully to near-zero by the final epoch

## Configuration Changes
- NUM_BLOCKS: 3 → 4 — adds 3 BasicBlocks (one per stage), increasing parameters from ~4.3M to ~5.7M (+33%)
- ESTIMATED_EPOCHS: 100 → 80 — recalibrates cosine decay for the shorter epoch budget

## Execution Environment
- Method: local command `uv run python train.py > run.log 2>&1`
- Resources: single H20 GPU
- Estimated runtime: ~310-320s total
- Log output: stdout+stderr captured to `run.log` in project root
- Tool skill: none

## Abort Criteria
- No output in run.log after 60s
- Loss goes to NaN/inf
- Epoch count drops below 60 — would indicate excessive throughput cost
- Per-step time > 25ms — would mean >56% throughput regression, putting us in the SE-block failure regime
- OOM — the ~5.7M param model at batch 256 should fit in H20's memory, but check

## Verification Protocol

### Verification Procedure

**Condition 1: best_test_acc > 96.56%** (baseline 96.46% + 0.1pp threshold)
- Command: `grep "^best_test_acc:" run.log | awk '{print $2}' | tr -d '%'`
- Pass: extracted value > 96.56
- Fail: value <= 96.56 or missing
- Timeout: 10s

**Condition 2: Clean completion**
- Command: `grep "^best_test_acc:" run.log`
- Pass: line exists with a numeric value
- Fail: missing or malformed
- Timeout: 10s

**Condition 3: Max 1 eval per epoch**
- Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log | awk '{print $2}'`
- Pass: eval count <= epoch count
- Fail: eval count > epoch count
- Timeout: 10s

### Informational Metrics (Optional)
- final_test_acc: `grep "^final_test_acc:" run.log | awk '{print $2}'`
- num_epochs: `grep "^num_epochs:" run.log | awk '{print $2}'`
- training_seconds: `grep "^training_seconds:" run.log | awk '{print $2}'`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log | awk '{print $2}'`
- num_params: `grep "^num_params:" run.log | awk '{print $2}'`
