# Plan EXP-000: Training Recipe Modernization
- **Created**: 2026-05-27
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-000.md

## Milestones

### Milestone 1: Code changes implemented
- [x] Replace `MultiStepLR` with `CosineAnnealingLR` in `train.py`
- [x] Add Cutout augmentation (16x16 patch) to the training transform pipeline
- [x] Add label smoothing (eps=0.1) to the cross-entropy loss
- [x] Enable Nesterov momentum in the SGD optimizer
- [x] Remove `MAX_STEPS` cap (let the time budget alone control training duration)
- [x] Verify code passes `ruff` linting

### Milestone 2: Experiment runs successfully
- [x] Run `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- [x] Confirm the script prints the full summary block (no crash)
- [x] Confirm training completes within the time budget

### Milestone 3: Results verified
- [x] Extract `best_test_acc` from `run.log` — 88.79%
- [ ] ~~Verify best_test_acc >= 91.82%~~ — FAILED (88.79% < 91.82%)
- [ ] ~~Collect informational metrics~~ — skipped

## Code Changes

- **`train.py`**: Four specific modifications, all within the existing file structure:

  1. **Cosine annealing LR schedule** (replaces lines 145-146):
     Replace `MultiStepLR(optimizer, milestones=[32000, 48000], gamma=0.1)` with `CosineAnnealingLR`. Since the schedule should span the full training, and we don't know the exact number of steps in advance (time-budgeted), use a large `T_max` value (e.g., 100000) so the cosine curve decays smoothly over the entire run. Alternatively, use epoch-based stepping: move `scheduler.step()` to after each epoch and set `T_max` to a reasonable epoch count (e.g., 200). The epoch-based approach is cleaner since the training loop already has an epoch structure.

     **Decision**: Use epoch-based cosine annealing with `T_max=200` (conservative upper bound — baseline completes ~97 epochs, so cosine will decay to ~cos(97π/200) ≈ 0.02 of initial LR by end of training). Move `scheduler.step()` from inside the batch loop to after each epoch's eval.

  2. **Cutout augmentation** (added to training transforms, lines 117-123):
     Implement Cutout as a custom transform class within `train.py`. After `transforms.Normalize(mean, std)`, add `Cutout(n_holes=1, length=16)`. The implementation randomly masks a 16x16 square region with zeros. This is a standard technique that does not require external libraries.

  3. **Label smoothing** (modify loss computation, line 173):
     Replace `F.cross_entropy(outputs, targets)` with `F.cross_entropy(outputs, targets, label_smoothing=0.1)`. PyTorch's `F.cross_entropy` natively supports the `label_smoothing` parameter since PyTorch 1.10.

  4. **Nesterov momentum** (modify optimizer, line 143):
     Add `nesterov=True` to the SGD constructor. This is a one-parameter change.

  5. **Remove MAX_STEPS cap**:
     The `MAX_STEPS = 64000` constant and its checks in the training loop (`step < MAX_STEPS`, `step >= MAX_STEPS`) artificially limit training. Since the time budget in `prepare.py` already controls duration, remove `MAX_STEPS` from the while condition and the break condition to let the time budget be the sole stopping criterion. This allows the model to train for as many steps as possible within 300s.

## Configuration Changes

- **LR schedule**: `MultiStepLR(milestones=[32000, 48000], gamma=0.1)` → `CosineAnnealingLR(T_max=200)` (epoch-based stepping). Rationale: cosine annealing naturally adapts to the actual training duration and is proven more effective than step decay on CIFAR-10.
- **Augmentation**: add `Cutout(n_holes=1, length=16)` after normalization. Rationale: 16x16 is the standard Cutout size for CIFAR-10 (32x32 images), proven in the fast CIFAR-10 paper and WRN recipes.
- **Loss**: `label_smoothing=0.0` → `label_smoothing=0.1`. Rationale: standard value from Inception-v3 paper, proven effective with Cutout on CIFAR-10.
- **Optimizer**: `nesterov=False` → `nesterov=True`. Rationale: Nesterov momentum provides better convergence with negligible cost, used in all modern CIFAR-10 recipes.
- **MAX_STEPS**: `64000` → removed. Rationale: let time budget alone control training; the step cap is an artifact of the original ResNet paper's fixed-iteration design.

## Execution Environment

- **Method**: Local command `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- **Resources**: Single NVIDIA H20 GPU (98GB), minimal VRAM (~330MB for ResNet-20)
- **Estimated runtime**: ~6 minutes total (300s training + ~60s startup/eval)
- **Log output**: All stdout/stderr redirected to `run.log` in project root. Do not use `tee` or let output flood context.

## Abort Criteria

- No output in `run.log` after 2 minutes → likely crash during setup/data download
- `run.log` shows Python traceback or error → crash, stop and investigate
- Total wall-clock exceeds 10 minutes → timeout, kill the process
- Loss values show NaN or Inf → divergence, stop and investigate

## Verification Protocol

### Verification Procedure

After the experiment completes, verify the three necessary conditions from the goal file in order. Stop on first failure.

**Condition 1: best_test_acc improves over baseline by at least 0.1 percentage points**
```bash
grep "^best_test_acc:" run.log
```
Extract the numeric value. It must be >= 91.82% (baseline 91.72% + 0.1% delta). If `grep` returns empty, the run crashed (fail).

**Condition 2: Script completes without crash (prints full summary block)**
```bash
grep "^---$" run.log
```
If the `---` separator line is present, the summary block was printed and the script completed normally. Also verify the other summary fields exist:
```bash
grep "^best_test_acc:\|^final_test_acc:\|^training_seconds:\|^peak_vram_mb:" run.log
```
All four must be present.

**Condition 3: Validation runs at most once per epoch**
The current code runs `evaluator.evaluate()` once per epoch at the end of the epoch loop (line 205). Our changes do not add any additional eval calls, so this condition is satisfied by code inspection. Verify by counting eval lines:
```bash
grep -c "eval ep" run.log
```
This count should equal the `num_epochs` value in the summary. If it exceeds `num_epochs`, the condition is violated.

### Informational Metrics (Optional)

Collected only when all necessary conditions pass:
- training_seconds: `grep "^training_seconds:" run.log`
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- final_test_acc: `grep "^final_test_acc:" run.log`
- final_test_loss: `grep "^final_test_loss:" run.log`
- num_epochs: `grep "^num_epochs:" run.log`
- num_steps: `grep "^num_steps:" run.log`
- num_params: `grep "^num_params:" run.log`
