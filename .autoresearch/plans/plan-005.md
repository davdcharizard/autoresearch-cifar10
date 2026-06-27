# Plan EXP-005: Width 5x on the doubly-regularized recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-005.md

## Milestones

### Milestone 1: Experiment branch + width change implemented
- [x] Create experiment branch `autoresearch/exp-005` from `autoresearch/dev`
- [x] Change `WIDTH_MULT = 4` to `WIDTH_MULT = 5` in train.py (stage widths become 80/160/320); update the inline comment accordingly
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params ~6.7M (quadratic-in-width scaling from 4,286,026 at 4x); epoch-1 eval completed with finite loss and acc ≥ 15%

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-005.md

## Code Changes

- **train.py** (only file modified — hard constraint): single constant change, `WIDTH_MULT = 4` → `WIDTH_MULT = 5` (with comment updated from "(16,32,64) -> (64,128,256)" to "-> (80,160,320)"). Recipe — time-keyed one-cycle (PEAK_LR 0.4), TrivialAugmentWide, RandomErasing, batch 512, selective WD, label smoothing — stays byte-identical to baseline 1174e0d.

  Why this tests the hypothesis: the only change is model capacity; any accuracy delta vs 96.23 is attributable to the width-epoch tradeoff under the regularized recipe.

  Risk/edge cases: epoch count predicted at 75–85 from in-project scaling (time/epoch sublinear in FLOPs: 2.63s at 4x, 7.5s at 8x). Undertraining signature to watch in analysis: depressed absolute accuracy with final=best. PEAK_LR stays 0.4 — EXP-002 ran 8x at the same peak without instability, so LR is not a confound to introduce. VRAM ~2.5GB (1.6GB at 4x, quadratic-ish scaling), far under any limit. Wall clock should DROP vs EXP-004's 416.5s (fewer epochs → fewer ~1s evals).

## Configuration Changes
- WIDTH_MULT: 4 → 5 (~4.29M → ~6.7M params; stage widths 80/160/320)
- Rationale: largest width step that keeps predicted epochs (~80) clearly above the measured starvation floor (8x failed at 40 epochs, EXP-002), while capacity is the demonstrated bottleneck after augmentation returns collapsed (goal-learnings § Patterns)
- No other changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~2.5GB expected
- Estimated runtime: ~6.5 min total (~80 epochs; eval count drops vs EXP-004, total ≤ ~410s)
- Log output: stdout/stderr → `run.log`; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- No output in run.log within 120s of launch → kill and diagnose
- Empty `grep "^best_test_acc:" run.log` after exit → crash; read `tail -n 50 run.log`
- Note: mid-schedule eval accuracy below EXP-004's trajectory is EXPECTED (wider net, fewer epochs, augmented data — see knowledge/papers/trivialaugment.md) and is NOT an abort signal

## Verification Protocol

### Verification Procedure
Run from project root after process exit. Baseline via `exp-index.sh baseline` = **96.23** (commit 1174e0d) at planning time → pass threshold **≥ 96.33** (+0.1 pp).

1. **Run completes without crashing within budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` present AND `total_seconds:` ≤ 600. Timeout: kill if alive >10 min → fail.
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 96.33)**
   - Command: `grep "^best_test_acc:" run.log` → parse; compare against fresh `exp-index.sh baseline` at verification time.
   - Pass: value ≥ 96.33. Evaluation stops at first failure.
3. **Validation at most once per epoch**
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`
   - Pass: eval-line count ≤ num_epochs.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~2.5GB)
- num_epochs: `grep "^num_epochs:" run.log` (expect 75–85 — the key scaling datapoint; record for the width-epoch curve regardless of verdict)
- num_params: `grep "^num_params:" run.log` (expect ~6.7M)
