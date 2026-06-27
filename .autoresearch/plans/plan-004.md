# Plan EXP-004: TrivialAugmentWide on top of the regularized recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-004.md

## Milestones

### Milestone 1: Experiment branch + transform change implemented
- [x] Create experiment branch `autoresearch/exp-004` from `autoresearch/dev`
- [x] Insert `transforms.TrivialAugmentWide()` into `train_tf` in train.py between RandomHorizontalFlip and ToTensor (TA is a PIL-stage op; RandomErasing stays last, after Normalize — matching the TA paper's TA-then-cutout ordering)
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params 4,286,026 (unchanged); epoch-1 eval completed with finite loss and acc ≥ 15%

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-004.md

## Code Changes

- **train.py** (only file modified — hard constraint): single transform insertion into `train_tf`:
  `transforms.TrivialAugmentWide()` placed after `transforms.RandomHorizontalFlip()` and before `transforms.ToTensor()`. Architecture (WIDTH_MULT=4) and recipe otherwise byte-identical to baseline 3a62d44, including the EXP-003 RandomErasing which remains last. The eval transform in frozen prepare.py is untouched — TA applies to training images only.

  Why this tests the hypothesis: the only change is policy augmentation on training inputs; any accuracy delta vs 96.06 is attributable to it. Placement is dictated by API (TA operates on PIL images, pre-ToTensor) and matches the TA paper's protocol of applying occlusion cutout after TA.

  Risk/edge cases: defaults (`num_magnitude_bins=31`, NEAREST interpolation, fill=None) are the paper configuration — verified to instantiate and run on a 32x32 PIL image in this torchvision (0.24.1) during planning. Over-regularization at 114 one-cycle epochs is the research risk (paper uses 200 epochs); fails cleanly as no-improvement. Per-image CPU cost is one PIL op across 8 workers — at the GPU-bound 4x width, throughput should be unchanged; a large epoch-count drop in the summary would falsify that.

## Configuration Changes
- Train transform: + TrivialAugmentWide() with library defaults (arXiv 2103.10158 — tuning-free by design; do NOT tune bins/interpolation)
- No other changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~1.6GB (same as EXP-003)
- Estimated runtime: ~6.5 min total (same profile as EXP-003: ~114 epochs, ~400s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- No output in run.log within 120s of launch → kill and diagnose
- Empty `grep "^best_test_acc:" run.log` after exit → crash; read `tail -n 50 run.log`
- img/s in step lines < ~12k sustained (vs ~18.7k expected) → CPU augmentation is throughput-binding; let the run finish (still valid) but flag in exp-log

## Verification Protocol

### Verification Procedure
Run from project root after process exit. Baseline via `exp-index.sh baseline` = **96.06** (commit 3a62d44) at planning time → pass threshold **≥ 96.16** (+0.1 pp).

1. **Run completes without crashing within budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` present AND `total_seconds:` ≤ 600. Timeout: kill if alive >10 min → fail.
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.16)**
   - Command: `grep "^best_test_acc:" run.log` → parse; compare against fresh `exp-index.sh baseline` at verification time.
   - Pass: value ≥ 96.16. Evaluation stops at first failure.
3. **Validation at most once per epoch**
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`
   - Pass: eval-line count ≤ num_epochs.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~1.6GB)
- num_epochs: `grep "^num_epochs:" run.log` (expect ~110–115 — confirms CPU augmentation did not become the bound; a large drop means TA is throughput-binding on the host)
- num_params: `grep "^num_params:" run.log` (expect 4,286,026 — unchanged)
