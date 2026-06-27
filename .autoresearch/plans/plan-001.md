# Plan EXP-001: Widen ResNet-20 4x (WRN-style) on the validated recipe
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-001.md

## Milestones

### Milestone 1: Experiment branch + width change implemented
- [x] Create experiment branch `autoresearch/exp-001` from `autoresearch/dev`
- [x] Implement the width-multiplier change in train.py (see Code Changes)
- [x] Static check passes: `uv run ruff check train.py`
- [x] Scope check: `git status --porcelain` shows only `train.py` modified

### Milestone 2: Experiment running on GPU 0
- [x] GPU 0 free per `nvidia-smi` (wait if busy — hard constraint)
- [x] Launch: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` in background
- [x] Early signal: params line shows ~4.3M; epoch-1 eval completed with finite loss and acc ≥ 15%

### Milestone 3: Run completed and metrics extracted
- [x] Process exited; `grep "^best_test_acc:" run.log` non-empty
- [x] `total_seconds:` ≤ 600
- [x] Full summary block recorded into logs/exp-log-001.md

## Code Changes

- **train.py** (only file modified — hard constraint). Single-variable capacity change, recipe untouched:
  1. Add hyperparameter `WIDTH_MULT = 4` next to `NUM_BLOCKS`.
  2. `ResNet.__init__`: accept `width_mult` (or read the module constant) and scale stage widths: conv1/bn1 16→`16*w`, layer1 `16*w`, layer2 `32*w` (stride 2), layer3 `64*w` (stride 2), fc in-features `64*w`. The existing zero-pad shortcut in `BasicBlock` already handles arbitrary widths (pads `out_channels - in_channels`); no change needed there.
  3. Update the params print line to include the width multiplier (cosmetic).
  4. Everything else — time-keyed one-cycle (peak 0.4, 15% warmup), bf16 autocast, TF32, channels_last, batch 512 nesterov, selective WD 5e-4, label smoothing 0.1, persistent workers, eval once/epoch, seed 42, summary block, per-step synchronize — stays byte-identical (goal-learnings § Patterns: recipe is validated and composable).

  Why this tests the hypothesis: capacity is the only variable changed; any accuracy delta vs the 93.16 baseline is attributable to width. Expected num_params ≈ 4.3M (16x the 270k baseline).

  Risk/edge cases: peak LR 0.4 may be slightly hot for the wide net (warmup mitigates); fewer epochs (~60–120) could undertrain — accepted risk per brainstorm; per-step time rises to ~25–50ms at batch 512, fine.

## Configuration Changes
- WIDTH_MULT: (new) 4 — stage widths (16,32,64) → (64,128,256) (WRN evidence: consistent gains at 1–12x width for 16-layer nets; arXiv 1605.07146)
- No other hyperparameter changes (single-variable experiment)

## Execution Environment
- Method: local command, background: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Resources: single NVIDIA H20 (GPU 0 only — hard constraint; wait if busy). VRAM expected ~2–6GB (soft constraint, fine)
- Estimated runtime: ~6–7.5 min total — 300s training + startup + per-epoch evals; throughput drops ~16x in FLOPs/image → expect ~10–20k img/s → ~60–120 epochs → eval overhead ~50–100s (relieves the 600s-cap risk from goal-learnings § Protocol Findings, High Importance)
- Log output: stdout/stderr → `run.log` in project root; metrics via grep; delete run.log after the experiment concludes
- Tool skill: none (local execution)

## Abort Criteria
- Wall clock exceeds 10 minutes total → kill, treat as failure (hard constraint)
- Training loss NaN/inf, or epoch-1 eval accuracy < 15% (divergence at peak-LR scale on wider net) → kill, diagnose
- No output in run.log within 120s of launch → kill and diagnose (dataloader/driver hang)
- Empty `grep "^best_test_acc:" run.log` after exit → crash; read `tail -n 50 run.log`

## Verification Protocol

### Verification Procedure
Run from project root after process exit. Baseline via `exp-index.sh baseline` on `.autoresearch/experiment-indices/maximize-cifar10-test-accuracy.tsv` = **93.16** (commit be45820) at planning time → pass threshold **≥ 93.26** (+0.1 pp).

1. **Run completes without crashing within budget (≤ 10 min total)**
   - Command: `grep "^best_test_acc:\|^total_seconds:" run.log`
   - Pass: `best_test_acc:` present AND `total_seconds:` ≤ 600. Timeout: kill if alive >10 min after launch → fail.
2. **best_test_acc ≥ baseline + 0.1 pp (≥ 93.26)**
   - Command: `grep "^best_test_acc:" run.log` → parse percentage; compare against fresh `exp-index.sh baseline` value at verification time.
   - Pass: value ≥ 93.26. Evaluation stops at first failure.
3. **Validation at most once per epoch**
   - Command: `grep -c "eval ep" run.log` vs `grep "^num_epochs:" run.log`
   - Pass: eval-line count ≤ num_epochs (loop structure unchanged; checked anyway).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` (expect ~2–6GB; soft constraint)
- num_epochs: `grep "^num_epochs:" run.log` (expect ~60–120; also confirms eval-overhead relief)
- num_params: `grep "^num_params:" run.log` (expect ≈ 4.3M — confirms the width change took effect)
