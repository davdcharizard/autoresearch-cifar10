# Plan EXP-022: Reflect Padding + Cutout Replacing RandomErasing
- **Created**: 2026-05-28
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-022.md

## Milestones

### Milestone 1: Code changes implemented
- [ ] Change `RandomCrop(32, padding=4)` to `RandomCrop(32, padding=4, padding_mode='reflect')` in train.py
- [ ] Replace `RandomErasing(p=0.25, scale=(0.02, 0.2))` with a custom Cutout transform (12×12 fixed square, zero-fill, p=0.5)
- [ ] Verify syntax and imports are correct with `uv run python -c "import train"`

### Milestone 2: Experiment submitted and confirmed running
- [ ] Run `uv run python train.py > run.log 2>&1` in background
- [ ] Confirm log file is being written to within 30s of launch

### Milestone 3: Experiment completed and results captured
- [ ] Training completes within 300s budget
- [ ] `best_test_acc:` line present in run.log
- [ ] Full 10-field summary block printed

## Code Changes

- **train.py line 144**: Change `transforms.RandomCrop(32, padding=4)` → `transforms.RandomCrop(32, padding=4, padding_mode='reflect')`. Reflect padding preserves edge statistics better than zero padding — no artificial black borders at crop boundaries. Validated by airbench96 recipe.

- **train.py lines 150-151**: Remove `transforms.RandomErasing(p=0.25, scale=(0.02, 0.2))` and replace with a custom `Cutout` transform class. The class applies a fixed 12×12 zero-filled square mask at a random position on the normalized tensor with probability 0.5. This matches airbench96's Cutout specification.

- **train.py (new class, before `main()`)**: Add a minimal `Cutout` transform class:
  ```python
  class Cutout:
      def __init__(self, size=12, p=0.5):
          self.size = size
          self.p = p

      def __call__(self, img):
          if torch.rand(1).item() > self.p:
              return img
          c, h, w = img.shape
          y = torch.randint(0, h, (1,)).item()
          x = torch.randint(0, w, (1,)).item()
          y1 = max(0, y - self.size // 2)
          y2 = min(h, y + self.size // 2)
          x1 = max(0, x - self.size // 2)
          x2 = min(w, x + self.size // 2)
          img[:, y1:y2, x1:x2] = 0
          return img
  ```
  This is placed after `transforms.Normalize` (operates on tensors). The mask can extend beyond image boundaries (matching DeVries & Taylor 2017 behavior — edges get smaller effective cutout regions). Using `torch.rand`/`torch.randint` for consistency with PyTorch transforms.

  Why not approximate with `RandomErasing(value=0, ratio=(1,1))`? RandomErasing samples scale uniformly, so the actual cutout size varies — it cannot produce a fixed 12×12 square. A custom class is the cleanest faithful implementation.

## Configuration Changes

- `padding_mode`: `'constant'` (default zero) → `'reflect'` (no new hyperparameters)
- Cutout size: 12px (from airbench96; smaller than Cutout paper's 16px recommendation for WRN-28-10 — appropriate for our smaller model)
- Cutout probability: 0.5 (airbench96 applies Cutout unconditionally; using p=0.5 as a moderate setting that avoids over-regularization given our existing TrivialAugmentWide)
- RandomErasing removed entirely (swapped, not stacked)

## Execution Environment

- Method: Local command `uv run python train.py > run.log 2>&1`
- Resources: Single H20 GPU (same as baseline)
- Estimated runtime: ~7-8 minutes total (300s training + ~110s TTA evaluation + startup)
- Log output: stdout+stderr captured to `run.log` in project root via redirection
- Tool skill: None (local execution)

## Abort Criteria

- No output in run.log after 60s → kill and investigate
- Loss NaN/inf in first 500 steps → kill (augmentation change may cause instability)
- Per-step time > 20ms sustained → kill (unexpected throughput regression — both changes should be zero-cost)
- CUDA OOM → kill (should not occur since model/batch unchanged)

## Verification Protocol

### Verification Procedure

Baseline: 96.46% (EXP-020, queried from experiment index).
Threshold: best_test_acc > 96.56% (baseline + 0.1pp).

**Condition 1: Primary metric exceeds threshold**
```bash
grep 'best_test_acc:' run.log | awk '{print $2}' | tr -d '%'
```
Pass if value > 96.56. Fail otherwise.

**Condition 2: Training script completes and prints full summary block**
```bash
grep -c 'best_test_acc:\|final_test_acc:\|final_test_loss:\|training_seconds:\|total_seconds:\|startup_seconds:\|peak_vram_mb:\|num_epochs:\|num_steps:\|num_params:' run.log
```
Pass if count = 10 (all 10 summary fields present). Fail otherwise.

**Condition 3: Validation runs at most once per epoch**
```bash
EVALS=$(grep -c 'eval ep' run.log)
EPOCHS=$(grep 'num_epochs:' run.log | awk '{print $2}')
```
Pass if EVALS ≤ EPOCHS. Fail otherwise.

### Informational Metrics (Optional)
- `num_epochs`: `grep 'num_epochs:' run.log | awk '{print $2}'` — confirm ~99 epochs (zero throughput cost)
- `peak_vram_mb`: `grep 'peak_vram_mb:' run.log | awk '{print $2}'` — confirm unchanged from baseline
- `final_test_acc`: `grep 'final_test_acc:' run.log | awk '{print $2}'` — last-epoch accuracy
