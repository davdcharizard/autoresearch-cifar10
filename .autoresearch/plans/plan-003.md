# Plan EXP-003: RandomErasing on the 4x-wide net
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-003.md

## Milestones

### Milestone 1: Experiment branch + transform change implemented
- [x] Create experiment branch `autoresearch/exp-003` from `autoresearch/dev`
- [x] Append `transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random")` as the last entry of `train_tf` in train.py (after Normalize — RandomErasing operates on tensors)
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params ~4.29M (unchanged); epoch-1 eval completed with finite loss and acc ≥ 15%

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-003.md

## Code Changes

- **train.py** (only file modified — hard constraint): single transform addition to `train_tf`:
  `transforms.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random")` appended after `transforms.Normalize(...)`. Architecture (WIDTH_MULT=4) and the full recipe stay byte-identical. The EVAL transform in prepare.py is frozen and untouched — RandomErasing applies to training images only, so eval semantics are unchanged by construction.

  Why this tests the hypothesis: the only change is occlusion regularization on training inputs; any accuracy delta vs 95.23 is attributable to it.

  Risk/edge cases: config follows the Random Erasing paper's CIFAR best (p=0.5, area 2–40%, random fill). `value="random"` requires torchvision's string API — verify spelling; it erases AFTER normalization, which is the paper-standard placement. Per-image cost is a tensor op on CPU workers — negligible at 8 workers.

## Configuration Changes
- Train transform: + RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value="random") (arXiv 1708.04896 CIFAR config)
- No other changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~1.6GB (same as EXP-001)
- Estimated runtime: ~6.5 min total (same profile as EXP-001: ~114 epochs, ~395s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- No output in run.log within 120s of launch → kill and diagnose
- Empty `grep "^best_test_acc:" run.log` after exit → crash; read `tail -n 50 run.log`

## Verification Protocol

### Verification Procedure
Run from project root after process exit. Baseline via `exp-index.sh baseline` = **95.23** (commit bd0976e) at planning time → pass threshold **≥ 95.33** (+0.1 pp).

1. **Run completes without crashing within budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` present AND `total_seconds:` ≤ 600. Timeout: kill if alive >10 min → fail.
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 95.33)**
   - Command: `grep "^best_test_acc:" run.log` → parse; compare against fresh `exp-index.sh baseline` at verification time.
   - Pass: value ≥ 95.33. Evaluation stops at first failure.
3. **Validation at most once per epoch**
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`
   - Pass: eval-line count ≤ num_epochs.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1.6GB)
- num_epochs: `grep "^num_epochs:" run.log` (expect ~110–115 — confirms throughput unchanged; a large drop would mean the transform is unexpectedly expensive)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026 — unchanged)
