# Plan EXP-002: Widen to 8x (stage widths 128/256/512, ~17M params)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-002.md

## Milestones

### Milestone 1: Experiment branch + width change implemented
- [x] Create experiment branch `autoresearch/exp-002` from `autoresearch/dev`
- [x] Set `WIDTH_MULT = 4` → `8` in train.py (only change)
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [ ] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [ ] Early signal: params line shows ~17M; epoch-1 eval completed with finite loss and acc ≥ 15%

### Milestone 3: Run completed and metrics extracted
- [ ] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [ ] `total_seconds:` ≤ 600
- [ ] Full summary block recorded into logs/exp-log-002.md

## Code Changes

- **train.py** (only file modified — hard constraint): single-constant change `WIDTH_MULT = 4` → `WIDTH_MULT = 8`. Stage widths become (128, 256, 512), ~17M params. Everything else — model class, time-keyed one-cycle (peak 0.4, 15% warmup), bf16/TF32/channels_last, batch 512 nesterov, selective WD 5e-4, label smoothing 0.1, eval once/epoch, seed 42 — stays byte-identical.

  Why this tests the hypothesis: pure capacity continuation; any delta vs 95.23 is attributable to width. WRN evidence says LR transfers across widths (paper uses one base LR for all widths), so no LR change is warranted — keeping it fixed preserves the single-variable property.

  Risk/edge cases: ~30–45 epochs expected (undertraining watch); per-step dt ~75–110ms at batch 512; VRAM ~4–8GB (fine).

## Configuration Changes
- WIDTH_MULT: 4 → 8 (WRN width study: gains persist to 8–12x at 16-layer depth; project-measured gradient +2.07pp at 4x)
- No other changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM ~4–8GB expected
- Estimated runtime: ~6 min total — 300s training + ~30–45 per-epoch evals (~30–40s) + startup; wall clock comfortable per goal-learnings § Protocol Findings (heavier model = fewer evals)
- Log output: stdout/stderr → `run.log` in project root; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% → kill, diagnose
- No output in run.log within 150s of launch (first epoch is ~10s of training but model init + cudnn.benchmark autotune is heavier at 8x) → kill and diagnose
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
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~4–8GB)
- num_epochs: `grep "^num_epochs:" run.log` (expect ~30–45; undertraining signal if accuracy fails with very low epochs)
- num_params: `grep "^num_params:" run.log` (expect ≈ 17M — confirms the width change)
